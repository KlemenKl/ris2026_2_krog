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

def ensemble_inference(models, data_loader, device):
    all_preds = []
    all_file_names = []
    
    for inputs, file_names in data_loader:
        inputs = inputs.to(device)
        ensemble_logits = torch.zeros((inputs.size(0), 8)).to(device)
        
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                ensemble_logits += torch.softmax(outputs, dim=1)
        
        _, predicted = torch.max(ensemble_logits, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_file_names.extend(file_names)
        
    return all_preds, all_file_names



TARGET_SHAPE = (184, 64, 64)
K_FOLDS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######
# GENERATE PREDICTIONS
#######

# 1. Testne datoteke
test_folder = 'data/test/' 
test_files = glob.glob(os.path.join(test_folder, "*.npy"))

# 2. BacteriaDataset in DataLoader
test_dataset = BacteriaDataset_2D(test_files, labels=None, target_shape=TARGET_SHAPE, use_augmentation=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

ensemble_models = []

# Naložimo vseh 5 najboljših modelov
for i in range(K_FOLDS):
    m = ResNet2D(num_classes=8)
    m.load_state_dict(torch.load(f'best_model_fold_{i}.pth'))
    m.to(device)
    ensemble_models.append(m)

predictions, filenames = ensemble_inference(ensemble_models, test_loader, device)

class_names = ['Ecoli', 'Efae', 'Kaer', 'Kpne', 'Paer', 'Saur', 'Sepi', 'Spyo']

print("Končne napovedi za neznane vzorce:")
for fname, p_idx in zip(filenames, predictions):
    print(f"Datoteka: {fname} -> Napovedana bakterija: {class_names[p_idx]}")

import pandas as pd
results_df = pd.DataFrame({'IME_SLIKE': filenames, 'OZNAKA': [class_names[p] for p in predictions]})
results_df.to_csv('siR.csv', index=False)