import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


from preprocessing import BacteriaDataset_2D
from model_architecture_2d import ResNet2D
from model_validation import plot_confusion_matrix, history_loss_acc

def train_model(model, train_loader, val_loader, train_labels, epochs=50, lr=0.001, fold_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Uteži razredov za uravnoteženje neuravnoteženih podatkov
    # Sepi=315, Spyo=6 -> brez uteži model predvideva skoraj samo Sepi
    num_classes = 8
    from collections import Counter
    label_counts = Counter(train_labels.tolist() if hasattr(train_labels, 'tolist') else list(train_labels))
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * label_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print(f"  Uteži razredov: {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    history = {'train_loss': [], 'val_acc': []}
    patience = 10
    best_val_loss = float('inf')
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

        # Validacija
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        val_acc = 100 * correct / total
        scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_fold_{fold_idx}.pth')
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping v epohi {epoch}")
            break

    return model, history

class_map = {
    'Ecoli': 0, 'Efae': 1, 'Kaer': 2, 'Kpne': 3,
    'Paer': 4, 'Saur': 5, 'Sepi': 6, 'Spyo': 7
}

data_folder = 'data/train/'
file_paths = []
labels = []

print("Before splitted data: DONE")
for class_name, class_idx in class_map.items():
    search_path = os.path.join(data_folder, class_name, "*.npy")
    files = glob.glob(search_path)
    file_paths.extend(files)
    labels.extend([class_idx] * len(files))

print("Splitted data: DONE")

TARGET_SHAPE = (184, 64, 64) 

# Konfiguracija
K_FOLDS = 5
file_paths = np.array(file_paths) # Pretvori v numpy za lažje indeksiranje
labels = np.array(labels)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []
print(f"Starting {K_FOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
    print(f"\n--- Treniranje Fold {fold+1}/{K_FOLDS} ---")
    
    # Razdelitev podatkov za ta fold
    train_sub_files = file_paths[train_idx]
    train_sub_labels = labels[train_idx]
    val_sub_files = file_paths[val_idx]
    val_sub_labels = labels[val_idx]
    
    # Ustvarjanje dataloaderjev
    train_dataset = BacteriaDataset_2D(train_sub_files, train_sub_labels, target_shape=TARGET_SHAPE, use_augmentation=True)
    val_dataset = BacteriaDataset_2D(val_sub_files, val_sub_labels, target_shape=TARGET_SHAPE, use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Ponovna inicializacija modela (da začne iz nič!)
    model = ResNet2D(num_classes=8)
    
    print("Preprocessing: DONE")
    # Treniranje
    trained_model, history = train_model(model, train_loader, val_loader, train_sub_labels, epochs=100, lr=0.0005, fold_idx=fold)

    # Shrani najboljšo natančnost tega folda
    best_acc = max(history['val_acc'])
    fold_results.append(best_acc)
    print(f"Fold {fold+1} končan. Najboljša natančnost: {best_acc:.2f}%")


def ensemble_predict(models, data_loader, device):
    all_preds = []
    all_labels = []
    
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        # Zbirnik za verjetnosti vseh modelov za ta batch
        ensemble_logits = torch.zeros((inputs.size(0), 8)).to(device)
        
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                # Seštevamo logite (ali softmax verjetnosti)
                ensemble_logits += torch.softmax(outputs, dim=1)
        
        # Končna odločitev: razred z najvišjo povprečno verjetnostjo
        _, predicted = torch.max(ensemble_logits, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    return np.array(all_preds), np.array(all_labels)


##################
# Ensebml
##################

print("\nZaganjam Ensemble evaluacijo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensemble_models = []

# Naložimo vseh 5 najboljših modelov
for i in range(K_FOLDS):
    m = ResNet2D(num_classes=8)
    m.load_state_dict(torch.load(f'best_model_fold_{i}.pth'))
    m.to(device)
    ensemble_models.append(m)

# Uporabimo zadnji val_loader
y_pred, y_true = ensemble_predict(ensemble_models, val_loader, device)
class_names = ['Ecoli', 'Efae', 'Kaer', 'Kpne', 'Paer', 'Saur', 'Sepi', 'Spyo']

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, target_names=class_names))

##################
# Skupna evaluacija
##################
print(f"\nSkupna povprečna natančnost: {np.mean(fold_results):.2f}% (+/- {np.std(fold_results):.2f}%)")
print("Model training: DONE")

plot_confusion_matrix(trained_model, val_loader, class_names)

history_loss_acc(history)

# Shranjevanje modela
torch.save(trained_model.state_dict(), 'bacteria_resnet2d_10.pth')
print("Done!")