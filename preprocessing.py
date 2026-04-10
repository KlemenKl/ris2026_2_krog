import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def preprocess_hsi(data, target_shape=(200, 64, 64)):
    """
    Normalizacija, maskiranje in zero-padding za hiperspektralno kocko.
    target_shape: (lambda, H, W)
    """
    # 1. Clipping in Normalizacija
    data = np.clip(data, 0, 1)
    
    # 2. Maskiranje okolice (kjer je bilo -1, nastavimo na 0 po normalizaciji)
    # Ker smo naredili clip(0,1), so vrednosti -1 postale 0. 
    
    # 3. Padding na fiksno velikost
    d, h, w = data.shape # Privzamemo (lambda, y, x)
    td, th, tw = target_shape
    
    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    
    # Dopolnimo z ničlami (vse strani enakomerno ali samo na koncu)
    padded_data = np.pad(data, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # Če je slika večja od target_shape, jo obrežemo
    return padded_data[:td, :th, :tw]

class BacteriaDataset(Dataset):
    def __init__(self, file_paths, labels, target_shape=(200, 64, 64)):
        self.file_paths = file_paths
        self.labels = labels
        self.target_shape = target_shape

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw_data = np.load(self.file_paths[idx])
        # Preuredimo iz (x, y, lambda) v (lambda, y, x) če je potrebno
        if raw_data.shape[-1] == self.target_shape[0]:
            raw_data = np.transpose(raw_data, (2, 1, 0))
            
        processed_data = preprocess_hsi(raw_data, self.target_shape)
        
        # PyTorch 3D CNN pričakuje: (Channel, Depth, Height, Width)
        tensor_data = torch.from_numpy(processed_data).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx]).long()
        
        return tensor_data, label