import torch
import torch.nn as nn
import numpy as np
import os
import glob
import random

from preprocessing import BacteriaDataset_2D, preprocess_hsi
from model_architecture_2d import ResNet2D

# Reproducibilnost
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TARGET_SHAPE = (184, 96, 96)
K_FOLDS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Ecoli', 'Efae', 'Kaer', 'Kpne', 'Paer', 'Saur', 'Sepi', 'Spyo']


def tta_predict_single(models, raw_data, device):
    """
    Test Time Augmentation za eno sliko.
    Ustvari 6 deterministicnih pogledov (original + flips + rotacije),
    povpreci softmax verjetnosti cez vse modele in vse poglede.
    """
    # Preprocess
    if raw_data.shape[-1] == TARGET_SHAPE[0]:
        raw_data = np.transpose(raw_data, (2, 0, 1))
    processed = preprocess_hsi(raw_data, TARGET_SHAPE)

    # 6 deterministicnih pogledov
    views = []
    data = processed.copy()

    # 1. Original
    views.append(data.copy())
    # 2. Horizontal flip
    views.append(np.flip(data, axis=1).copy())
    # 3. Vertical flip
    views.append(np.flip(data, axis=2).copy())
    # 4. Rotacija 90
    views.append(np.rot90(data, 1, axes=(1, 2)).copy())
    # 5. Rotacija 180
    views.append(np.rot90(data, 2, axes=(1, 2)).copy())
    # 6. Rotacija 270
    views.append(np.rot90(data, 3, axes=(1, 2)).copy())

    avg_probs = None
    for view in views:
        inp = torch.from_numpy(view).float().unsqueeze(0).to(device)
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(inp)
                probs = torch.softmax(outputs, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = avg_probs + probs

    avg_probs /= (len(views) * len(models))
    pred_idx = avg_probs.argmax(dim=1).item()
    confidence = avg_probs.max().item()
    return pred_idx, confidence


# ==========================================
# Nalozimo ensemble modele
# ==========================================
print("Nalaganje modelov...")
ensemble_models = []
for i in range(K_FOLDS):
    m = ResNet2D(num_classes=8)
    m.load_state_dict(torch.load(f'best_model_fold_{i}.pth', weights_only=True))
    m.to(device)
    m.eval()
    ensemble_models.append(m)
    print(f"  Model fold {i} nalozen.")

# ==========================================
# Poisci testne datoteke
# ==========================================
# Preverimo vec moznih lokacij
test_folders = ['test_set/', 'data/test/']
test_files = []
for folder in test_folders:
    found = glob.glob(os.path.join(folder, "*.npy"))
    # Izlocimo lam.npy (valovne dolzine, ne testna slika)
    found = [f for f in found if 'lam' not in os.path.basename(f).lower()]
    if found:
        test_files = found
        print(f"Najdenih {len(test_files)} testnih datotek v {folder}")
        break

if not test_files:
    print("NAPAKA: Ni najdenih testnih datotek!")
    print("Preverite, da so testne datoteke v test_set/ ali data/test/")
    exit(1)

# ==========================================
# Napovedi s TTA
# ==========================================
print(f"\nGeneriranje napovedi za {len(test_files)} testnih slik s TTA...")
results = []

for i, fpath in enumerate(test_files):
    raw_data = np.load(fpath)
    fname = os.path.basename(fpath)

    pred_idx, confidence = tta_predict_single(ensemble_models, raw_data, device)
    pred_class = class_names[pred_idx]
    results.append((fname, pred_class))

    print(f"  [{i+1}/{len(test_files)}] {fname} -> {pred_class} (confidence: {confidence:.4f})")

# ==========================================
# Zapis rezultatov v siR.csv
# ==========================================
output_file = 'siR.csv'
with open(output_file, 'w') as f:
    f.write('IME_SLIKE,OZNAKA\n')
    for fname, pred_class in results:
        f.write(f'{fname},{pred_class}\n')

print(f"\nNapovedi zapisane v {output_file}")
print(f"Skupno {len(results)} napovedi.")
print("Done!")
