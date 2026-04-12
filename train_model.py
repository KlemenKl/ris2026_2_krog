import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from preprocessing import BacteriaDataset_2D
from model_architecture_2d import ResNet2D
from model_validation import plot_confusion_matrix, history_loss_acc

# Reproducibilnost
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, train_labels, epochs=120, lr=0.0005, fold_idx=0):
    """
    Enofazno treniranje z cosine annealing schedulerom.
    Brez dvofaznega pristopa ker model ni pretrained (random backbone).
    Brez utezi razredov ker prevec agresivne za ta dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'train_loss': [], 'val_acc': []}
    patience = 20
    best_val_acc = 0.0
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validacija
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        val_acc = 100 * correct / total

        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}% - LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_fold_{fold_idx}.pth')
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping v epohi {epoch+1}")
            break

    # Nalozimo najboljsi model
    model.load_state_dict(torch.load(f'best_model_fold_{fold_idx}.pth', weights_only=True))
    return model, history


# ==========================================
# Priprava podatkov
# ==========================================
class_map = {
    'Ecoli': 0, 'Efae': 1, 'Kaer': 2, 'Kpne': 3,
    'Paer': 4, 'Saur': 5, 'Sepi': 6, 'Spyo': 7
}

data_folder = 'data/train/'
file_paths = []
labels = []

print("Nalaganje podatkov...")
for class_name, class_idx in class_map.items():
    search_path = os.path.join(data_folder, class_name, "*.npy")
    files = glob.glob(search_path)
    file_paths.extend(files)
    labels.extend([class_idx] * len(files))
    print(f"  {class_name}: {len(files)} slik")

# 96x96 namesto 64x64 - vec prostorskih informacij
TARGET_SHAPE = (184, 96, 96)

K_FOLDS = 5
file_paths = np.array(file_paths)
labels = np.array(labels)

print(f"\nSkupno {len(file_paths)} slik v {len(class_map)} razredih")
print(f"Ciljna oblika: {TARGET_SHAPE}")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
print(f"\nStarting {K_FOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
    print(f"\n{'='*50}")
    print(f"--- Treniranje Fold {fold+1}/{K_FOLDS} ---")
    print(f"{'='*50}")

    train_sub_files = file_paths[train_idx]
    train_sub_labels = labels[train_idx]
    val_sub_files = file_paths[val_idx]
    val_sub_labels = labels[val_idx]

    train_dataset = BacteriaDataset_2D(train_sub_files, train_sub_labels,
                                        target_shape=TARGET_SHAPE, use_augmentation=True)
    val_dataset = BacteriaDataset_2D(val_sub_files, val_sub_labels,
                                      target_shape=TARGET_SHAPE, use_augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet2D(num_classes=8)

    trained_model, history = train_model(
        model, train_loader, val_loader, train_sub_labels,
        epochs=120, lr=0.0005, fold_idx=fold
    )

    best_acc = max(history['val_acc'])
    fold_results.append(best_acc)
    print(f"\nFold {fold+1} koncan. Najboljsa natancnost: {best_acc:.2f}%")


# ==========================================
# Ensemble evaluacija
# ==========================================
print(f"\n{'='*50}")
print("Zaganjam Ensemble evaluacijo...")
print(f"{'='*50}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensemble_predict(models, data_loader, device):
    all_preds = []
    all_labels = []
    for inputs, lbls in data_loader:
        inputs = inputs.to(device)
        ensemble_logits = torch.zeros((inputs.size(0), 8)).to(device)
        for m in models:
            m.eval()
            with torch.no_grad():
                outputs = m(inputs)
                ensemble_logits += torch.softmax(outputs, dim=1)
        _, predicted = torch.max(ensemble_logits, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(lbls.numpy())
    return np.array(all_preds), np.array(all_labels)


ensemble_models = []
for i in range(K_FOLDS):
    m = ResNet2D(num_classes=8)
    m.load_state_dict(torch.load(f'best_model_fold_{i}.pth', weights_only=True))
    m.to(device)
    ensemble_models.append(m)

y_pred, y_true = ensemble_predict(ensemble_models, val_loader, device)
class_names = ['Ecoli', 'Efae', 'Kaer', 'Kpne', 'Paer', 'Saur', 'Sepi', 'Spyo']

print("\nEnsemble klasifikacijsko porocilo (zadnji fold):")
print(classification_report(y_true, y_pred, target_names=class_names))

print(f"\nSkupna povprecna natancnost: {np.mean(fold_results):.2f}% (+/- {np.std(fold_results):.2f}%)")
print("Model training: DONE")

plot_confusion_matrix(trained_model, val_loader, class_names)
history_loss_acc(history)

torch.save(trained_model.state_dict(), 'bacteria_resnet2d_10.pth')
print("Done!")
