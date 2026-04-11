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

import random

def augment_hsi(data):
    """
    data: numpy array oblika (C, D, H, W) -> (1, lambda, H, W)
    """
    # 1. Horizontalno in vertikalno zrcaljenje
    if random.random() > 0.5:
        data = np.flip(data, axis=2) # Flip Height
    if random.random() > 0.5:
        data = np.flip(data, axis=3) # Flip Width

    # 2. Rotacije za 90, 180 ali 270 stopinj
    k = random.randint(0, 3)
    data = np.rot90(data, k, axes=(2, 3))

    # 3. Dodajanje spektralnega šuma (zelo pomembno za HSI!)
    # Doda majhne variacije v intenziteto odboja (0.1% šuma)
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.001, data.shape)
        data = data + noise
        data = np.clip(data, 0, 1) # Ohranimo range [0, 1]

    return data.copy()

class BacteriaDataset(Dataset):
    def __init__(self, file_paths, labels, target_shape=(200, 96, 96), use_augmentation=False):
        self.file_paths = file_paths
        self.labels = labels
        self.target_shape = target_shape
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.file_paths)

    def augment_hsi(self, data):
        """
        Izvaja prostorsko in spektralno augmentacijo na 4D polju (C, D, H, W).
        """
        # 1. Horizontalno in vertikalno zrcaljenje
        if random.random() > 0.5:
            data = np.flip(data, axis=2)  # Os H
        if random.random() > 0.5:
            data = np.flip(data, axis=3)  # Os W

        # 2. Rotacije za 90, 180 ali 270 stopinj
        k = random.randint(0, 3)
        data = np.rot90(data, k, axes=(2, 3))

        # 3. Dodajanje majhnega spektralnega šuma (Gaussov šum)
        # Pomaga pri robusnosti na napake senzorja
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.001, data.shape)
            data = data + noise
            data = np.clip(data, 0, 1)

        return data.copy()

    def __getitem__(self, idx):
        # Nalaganje podatkov
        raw_data = np.load(self.file_paths[idx])
        
        # Preureditev v (lambda, y, x)
        if raw_data.shape[-1] == self.target_shape[0]:
            raw_data = np.transpose(raw_data, (2, 1, 0))
            
        # Tvoja obstoječa funkcija za normalizacijo in padding
        processed_data = preprocess_hsi(raw_data, self.target_shape)
        
        # Priprava za PyTorch (Channel, Depth, Height, Width)
        # Dodamo Channel dimenzijo (1, 200, 64, 64)
        tensor_data = processed_data[np.newaxis, ...]
        
        # Izvedba augmentacije, če je vklopljena
        if self.use_augmentation:
            tensor_data = self.augment_hsi(tensor_data)
            
        # Pretvorba v tensorje
        x = torch.from_numpy(tensor_data).float()
        y = torch.tensor(self.labels[idx]).long()
        
        return x, y