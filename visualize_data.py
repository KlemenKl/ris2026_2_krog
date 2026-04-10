import numpy as np
import matplotlib.pyplot as plt
import glob
import os

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
data = np.load("lam.npy")
print(len(data))

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