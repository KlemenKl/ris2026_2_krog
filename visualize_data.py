import numpy as np
import matplotlib.pyplot as plt
import glob
import os

input_dir = 'ucni_set'  # Data folder
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dimensions(data_folder):
    widths = []
    heights = []
    
    # Poišči vse .npy datoteke (tudi v podmapah)
    file_paths = glob.glob(os.path.join(data_folder, "**/*.npy"), recursive=True)

    for f in file_paths:
        try:
            print(f"Loading {f}...")
            # Naložimo samo obliko (shape), ne celih podatkov (hitreje)
            data = np.load(f, mmap_mode='r')
            # Navodila pravijo (x, y, lambda)
            widths.append(data.shape[0])
            heights.append(data.shape[1])
        except Exception as e:
            print(f"Napaka pri datoteki {f}: {e}")

    # Izris škatle z brki
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[widths, heights], palette="Set2")
    plt.xticks([0, 1], ['Širina (X)', 'Višina (Y)'])
    plt.ylabel('Število pikslov')
    plt.title('Distribucija velikosti bakterijskih kolonij')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Izpis statistike
    stats = {
        "Max width": np.max(widths),
        "Max height": np.max(heights),
        "95. width": np.percentile(widths, 95),
        "95. height": np.percentile(heights, 95),
        "Mediana width": np.median(widths)
    }
    
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")

    plt.savefig('dimensions_analysis.png')
    print("Stats are", stats)
    return stats

# Uporaba:
stats = analyze_dimensions(input_dir)

# 1. Setup paths
# input_dir = 'ucni_set'  # Change this to your folder
# output_dir = 'visualizations'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 2. Get list of all .npy files
# files = glob.glob(os.path.join(input_dir, "*.npy"))

# print(f"Found {len(files)} files. Starting processing...")

# for file_path in files:
#     file_name = os.path.basename(file_path)
#     print(f"Processing: {file_name}")
    
# try:
# Load data
# data = np.load("lam.npy")
# print(len(data))

# # Create a figure with two views
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # View 1: The "Long" slice (Horizontal)
# mid_z = data.shape[0] // 2
# ax1.imshow(data[mid_z, :, :], cmap='magma', aspect='auto')
# ax1.set_title(f"Side View (Slice {mid_z})")

# # View 2: The "Circular" cross-section (Vertical)
# mid_x = data.shape[2] // 2
# ax2.imshow(data[:, :, mid_x], cmap='magma', aspect='equal')
# ax2.set_title(f"Cross-section (Slice {mid_x})")

# plt.suptitle(f"File: {file_name}\nShape: {data.shape}")

# # Save instead of showing
# save_path = os.path.join(output_dir, file_name.replace('.npy', '.png'))
# plt.savefig(save_path)
# plt.close(fig) # Close figure to free up RAM
        
    #except Exception as e:
    #    print(f"Could not process {file_name}: {e}")

print("Done! Check the 'visualizations' folder.")