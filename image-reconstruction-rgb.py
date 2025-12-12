import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from scipy.ndimage import median_filter
import os
from PIL import Image

# load image

image_filename = 'uploaded_image.jpg'
csv_filename = 'field_test_image_color.csv'
normalize_pixels = True
missing_fraction = 0.6

# ----- Colab-safe / local load -----
try:
    from google.colab import files
    if not os.path.exists(image_filename):
        print("Please upload an image:")
        uploaded = files.upload()
        image_filename = next(iter(uploaded.keys()))
except ImportError:
    if not os.path.exists(image_filename):
        raise FileNotFoundError(f"{image_filename} not found locally.")

img = Image.open(image_filename).convert("RGB")
img_array = np.array(img, dtype=np.float32)  # shape (H, W, 3)

# normalize
if normalize_pixels:
    img_array /= 255.0

H, W, C = img_array.shape
assert C == 3, "Color image must have 3 channels."

# create missing pixels per channel

img_missing = img_array.copy()
num_pixels = img_missing.size
num_missing = int(num_pixels * missing_fraction)

flat = img_missing.reshape(-1, 3)
missing_idx = np.random.choice(flat.shape[0], num_missing // 3, replace=False)
flat[missing_idx] = np.nan  # mark missing for all 3 channels

img_missing = flat.reshape(img_array.shape)

# save CSV: flattened as H,W,3
np.savetxt(csv_filename, img_missing.reshape(-1, 3), delimiter=",", fmt="%.4f")
print(f"Saved color image with missing pixels as: {csv_filename}")

# reload from csv

loaded = np.genfromtxt(csv_filename, delimiter=',')
img = loaded.reshape(H, W, 3)

print("Loaded CSV shape:", img.shape)

# show corrupted image
plt.figure(figsize=(6,6))
plt.imshow(np.nan_to_num(img, nan=1.0))
plt.axis('off')
plt.title("Corrupted Image (missing → white)")
plt.show()

# 2d cosine basis

def basis_chip(u, v, N=8):
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.cos((np.pi/N)*(X+0.5)*u) * np.cos((np.pi/N)*(Y+0.5)*v)

N = 8
basis_matrix = np.zeros((64, 64))

col = 0
for u in range(N):
    for v in range(N):
        basis_matrix[:, col] = basis_chip(u, v).flatten()
        col += 1

# lasso on single 8x8 channel

def reconstruct_block_cv(chip, basis_matrix, lambda_grid, kf):

    y = chip.flatten()
    mask = ~np.isnan(y)

    if mask.sum() == 64:   # no missing pixels
        return chip.copy()

    y_obs = y[mask]
    A_obs = basis_matrix[mask, :]

    # cross-validate
    mse_list = []
    for lam in lambda_grid:
        fold_err = []
        for train, test in kf.split(A_obs):
            model = Lasso(alpha=lam, fit_intercept=False, max_iter=20000)
            model.fit(A_obs[train], y_obs[train])
            pred = model.predict(A_obs[test])
            fold_err.append(np.mean((pred - y_obs[test])**2))
        mse_list.append(np.mean(fold_err))

    best_lam = lambda_grid[np.argmin(mse_list)]

    # refit full
    final_model = Lasso(alpha=best_lam, fit_intercept=False, max_iter=20000)
    final_model.fit(A_obs, y_obs)

    out = basis_matrix @ final_model.coef_
    return out.reshape(8, 8)

# reconstruct a single rgb block

def reconstruct_color_block(block, basis_matrix, lambda_grid, kf):
    rec = np.zeros((8,8,3), dtype=np.float32)
    for ch in range(3):
        rec[:,:,ch] = reconstruct_block_cv(block[:,:,ch], basis_matrix, lambda_grid, kf)
    return rec

# full color reconstruction

def reconstruct_full_color(img, basis_matrix, lambda_grid=None):
    if lambda_grid is None:
        lambda_grid = np.logspace(-3, 7, 11)

    H, W, _ = img.shape
    rec = np.zeros_like(img)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    print("Starting full RGB reconstruction...")
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            block = img[r:r+8, c:c+8, :]
            if block.shape != (8,8,3):
                continue
            rec_block = reconstruct_color_block(block, basis_matrix, lambda_grid, kf)
            rec[r:r+8, c:c+8, :] = rec_block
        print(f"Completed row block {r//8 + 1} of {H//8}")

    print("Done reconstructing full RGB image.")
    return rec

# run reconstruction
lambda_grid = np.logspace(-3, 7, 11)

reconstructed = reconstruct_full_color(img, basis_matrix, lambda_grid)

# median filter applied per channel
reconstructed_med = median_filter(reconstructed, size=(3,3,1))

# plot final results

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(np.nan_to_num(img, nan=1.0))
plt.title("Corrupted RGB Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(np.clip(reconstructed,0,1))
plt.title("Reconstructed (No Median Filter)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(np.clip(reconstructed_med,0,1))
plt.title("Reconstructed (Median Filtered)")
plt.axis('off')

plt.show()