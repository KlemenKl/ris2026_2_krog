from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def plot_confusion_matrix(model, val_loader, class_names):
    model.eval()
    all_preds = []
    all_true = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())

    # Izračun matrike
    cm = confusion_matrix(all_true, all_preds)
    
    # Vizualizacija
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Klasifikacija bakterij')
    plt.ylabel('Dejansko')
    plt.xlabel('Napovedano')
    plt.savefig('confusion_matrix_2d.png')

    # Izpis natančnega poročila (Precision, Recall, F1-score)
    print(classification_report(all_true, all_preds, target_names=class_names))

def history_loss_acc(history):
    ##############################
    # Vizualizacija izgube in natančnosti
    ##############################

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Izguba (Loss)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Natančnost (Accuracy)')
    plt.legend()
    plt.savefig('training_history_2d.png')

