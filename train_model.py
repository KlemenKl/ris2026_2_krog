import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
import math
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

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


def train_model_two_phase(model, train_loader, val_loader, train_labels, fold_idx=0):
    """
    Dvofazno treniranje: najprej zamrznemo backbone in treniramo samo FC glavo,
    nato odmrznemo vse in fine-tunamo z nizkim LR. Ta strategija se je izkazala
    ze v prvem krogu tekmovanja.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Blage utezi razredov z sqrt inverzne frekvence (ne preagresivno)
    label_counts = Counter(train_labels.tolist() if hasattr(train_labels, 'tolist') else list(train_labels))
    total_samples = sum(label_counts.values())
    num_classes = 8
    raw_weights = [total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
    # Uporabimo sqrt da omilimo ekstremne utezi (namesto 13x za Spyo dobimo ~3.6x)
    class_weights = torch.tensor([math.sqrt(w) for w in raw_weights], dtype=torch.float32).to(device)
    print(f"  Utezi razredov (sqrt): {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    history = {'train_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    # ==========================================
    # FAZA 1: Zamrznemo backbone, treniramo samo FC glavo
    # ==========================================
    print("  Faza 1: Treniranje FC glave (backbone zamrznjen)...")
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    optimizer_phase1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=0.05
    )
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=30)

    best_val_acc, history = _run_phase(
        model, train_loader, val_loader, criterion,
        optimizer_phase1, scheduler_phase1, device,
        num_epochs=30, fold_idx=fold_idx,
        history=history, best_val_acc=best_val_acc
    )

    # Nalozimo najboljsi model iz faze 1
    model.load_state_dict(torch.load(f'best_model_fold_{fold_idx}.pth', weights_only=True))

    # ==========================================
    # FAZA 2: Odmrznemo vse parametre, fine-tuning z nizkim LR
    # ==========================================
    print("  Faza 2: Fine-tuning celotnega modela...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_phase2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=70)

    best_val_acc, history = _run_phase(
        model, train_loader, val_loader, criterion,
        optimizer_phase2, scheduler_phase2, device,
        num_epochs=70, fold_idx=fold_idx,
        history=history, best_val_acc=best_val_acc
    )

    # Nalozimo najboljsi model skupaj
    model.load_state_dict(torch.load(f'best_model_fold_{fold_idx}.pth', weights_only=True))
    return model, history


def _run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
               device, num_epochs, fold_idx, history, best_val_acc):
    """Izvede eno fazo treniranja z early stoppingom."""
    patience = 15
    counter = 0

    for epoch in range(num_epochs):
        # Treniranje
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

        print(f"    Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_fold_{fold_idx}.pth')
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"    Early stopping v epohi {epoch+1}")
            break

    return best_val_acc, history


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

# SPREMEMBA: 96x96 namesto 64x64 - model je bil zasnovan za to velikost
TARGET_SHAPE = (184, 96, 96)

# Konfiguracija
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

    # Ustvarjanje dataloaderjev
    train_dataset = BacteriaDataset_2D(train_sub_files, train_sub_labels,
                                        target_shape=TARGET_SHAPE, use_augmentation=True)
    val_dataset = BacteriaDataset_2D(val_sub_files, val_sub_labels,
                                      target_shape=TARGET_SHAPE, use_augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Ponovna inicializacija modela
    model = ResNet2D(num_classes=8)

    # Dvofazno treniranje
    trained_model, history = train_model_two_phase(
        model, train_loader, val_loader, train_sub_labels, fold_idx=fold
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


# Nalozimo vse fold modele
ensemble_models = []
for i in range(K_FOLDS):
    m = ResNet2D(num_classes=8)
    m.load_state_dict(torch.load(f'best_model_fold_{i}.pth', weights_only=True))
    m.to(device)
    ensemble_models.append(m)

# Ensemble napovedi na zadnjem val_loaderju
y_pred, y_true = ensemble_predict(ensemble_models, val_loader, device)
class_names = ['Ecoli', 'Efae', 'Kaer', 'Kpne', 'Paer', 'Saur', 'Sepi', 'Spyo']

print("\nEnsemble klasifikacijsko porocilo (zadnji fold):")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==========================================
# Skupna evaluacija
# ==========================================
print(f"\nSkupna povprecna natancnost: {np.mean(fold_results):.2f}% (+/- {np.std(fold_results):.2f}%)")
print("Model training: DONE")

plot_confusion_matrix(trained_model, val_loader, class_names)
history_loss_acc(history)

torch.save(trained_model.state_dict(), 'bacteria_resnet2d_10.pth')
print("Done!")
