import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from scipy.ndimage import median_filter
import os
from PIL import Image

image_filename = 'uploaded_image.jpg'
csv_filename = 'field_test_image.csv'
normalize_pixels = True
missing_fraction = 0.6

try:
    from google.colab import files
    if not os.path.exists(image_filename):
        print("Please upload an image:")
        uploaded = files.upload()  # Colab file picker
        image_filename = next(iter(uploaded.keys()))
except ImportError:
    if not os.path.exists(image_filename):
        raise FileNotFoundError(f"{image_filename} not found locally.")

img = Image.open(image_filename)
img_gray = img.convert("L")  # grayscale

# convert to float to allow NaN values
img_array = np.array(img_gray, dtype=np.float32)

# normalize pixel values to [0,1] if desired
if normalize_pixels:
    img_array /= 255.0

# ONLY RUN THIS IF YOUR IMAGE ISNT ALREADY CORRUPTED AND YOU WANT TO ARTIFICIALLY CORRUPT YOUR IMAGE
num_pixels = img_array.size
num_missing = int(num_pixels * missing_fraction)

# choose random indices to remove
flat_indices = np.random.choice(num_pixels, num_missing, replace=False)
img_flat = img_array.flatten()
img_flat[flat_indices] = np.nan  # assign NaN
img_array_missing = img_flat.reshape(img_array.shape)

np.savetxt(csv_filename, img_array_missing, delimiter=",", fmt="%.4f")
print(f"Saved grayscale image with missing pixels as CSV: {csv_filename}")

# replace with your actual filename if different
image_path = 'field_test_image.csv'

# load the image
img = np.genfromtxt(image_path, delimiter=',')
print("Image shape:", img.shape)

# RGB image to mark missing pixels
rgb_img = np.dstack([img, img, img])

# norm pixel values to [0,1] if not already
rgb_img = (rgb_img - np.nanmin(rgb_img)) / (np.nanmax(rgb_img) - np.nanmin(rgb_img))

# missing pixels
missing_mask = np.isnan(img)

# replace missing pixel locations with red
rgb_img[missing_mask] = [1, 0, 0]  # RGB red

# display the image
plt.figure(figsize=(6,6))
plt.imshow(rgb_img, cmap='gray')
plt.axis('off')
plt.title("Corrupted Image (Missing Pixels in Red)")
plt.show()

# convert to 0-indexed
row, col = 8 - 1, 8 - 1

# extract the 8x8 chip
chip = img[row:row+8, col:col+8]

print("Extracted 8x8 chip shape:", chip.shape)

# RGB version of the chip
rgb_chip = np.dstack([chip, chip, chip])

# norm valid pixels to [0,1]
valid_pixels = ~np.isnan(chip)
if np.any(valid_pixels):
    rgb_chip[valid_pixels] = (rgb_chip[valid_pixels] - np.nanmin(chip)) / (np.nanmax(chip) - np.nanmin(chip))

# color missing pixels red
rgb_chip[np.isnan(chip)] = [1, 0, 0]

plt.figure(figsize=(4,4))
plt.imshow(rgb_chip)
plt.axis('off')
plt.title("8×8 Chip (Missing Pixels in Red)")
plt.show()

def basis_chip(u, v, N=8):
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # apply the 2D cosine basis formula
    chip = np.cos((np.pi / N) * (X + 0.5) * u) * np.cos((np.pi / N) * (Y + 0.5) * v)
    return chip

# generate basis chips
chip_34 = basis_chip(3, 4)
chip_52 = basis_chip(5, 2)

# display (u,v) = (3,4)
plt.figure(figsize=(5, 5))
plt.imshow(chip_34, cmap='gray', interpolation='nearest')
plt.title("Basis Chip (u, v) = (3, 4)")
plt.axis('off')
plt.show()

# display (u,v) = (5,2)
plt.figure(figsize=(5, 5))
plt.imshow(chip_52, cmap='gray', interpolation='nearest')
plt.title("Basis Chip (u, v) = (5, 2)")
plt.axis('off')
plt.show()

N = 8
basis_matrix = np.zeros((N*N, N*N))

# fill columns with flattened basis chips
col = 0
chip_temp = chip
for u in range(N):
    for v in range(N):
        chip_temp = basis_chip(u, v, N)
        basis_matrix[:, col] = chip_temp.flatten()
        col += 1

print("Basis matrix shape:", basis_matrix.shape)

plt.figure(figsize=(8, 8))
plt.imshow(basis_matrix, cmap='gray', interpolation='nearest')
plt.title("8×8 Basis Vector Matrix (64×64)")
plt.xlabel("Basis Vector Index (u,v combinations)")
plt.ylabel("Pixel Index (Flattened 8×8 Chip)")
plt.show()

y = chip.flatten()

mask = ~np.isnan(y)

print(mask.sum(), "valid pixels")

y_obs = y[mask]
A_obs = basis_matrix[mask, :]

print("Observed y shape:", y_obs.shape)
print("Observed A shape:", A_obs.shape)

# regularization parameter
alpha = 0.05

# fit w lasso
lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
lasso.fit(A_obs, y_obs)

# coefficients
weights = lasso.coef_

print("Nonzero coefficients:", np.sum(weights != 0))

# sort by absolute value
top_indices = np.argsort(np.abs(weights))[::-1][:10]

# get (u, v) pairs for each column
uv_pairs = [(i // 8, i % 8) for i in range(64)]

print("Top 10 basis components:")
for idx in top_indices:
    u, v = uv_pairs[idx]
    print(f"  (u, v) = ({u}, {v}), weight = {weights[idx]:.4f}")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, idx in enumerate(top_indices):
    u, v = uv_pairs[idx]
    chip_uv = basis_chip(u, v)
    axes[i].imshow(chip_uv, cmap='gray', interpolation='nearest')
    axes[i].set_title(f"(u,v)=({u},{v})\nWeight={weights[idx]:.3f}")
    axes[i].axis('off')

plt.suptitle("Top 10 LASSO Basis Chips (Largest Weights)")
plt.tight_layout()
plt.show()

# reconstruct the chip
y_hat = basis_matrix @ weights
reconstructed_chip = y_hat.reshape(8, 8)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(chip, cmap='gray')
plt.title("Original (Corrupted)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_chip, cmap='gray')
plt.title("Reconstructed (LASSO)")
plt.axis('off')

plt.show()

y_obs = y[mask]
A_obs = basis_matrix[mask, :]

# 10 folds over the observed pixels
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# 11 lambda values, log-spaced over 10 decades (10^-3 to 10^7)
lambdas = np.logspace(-3, 7, 11)

# extract first fold
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(A_obs)):
    if fold_idx == 0:
        print("Using fold:", fold_idx)
        A_train, y_train = A_obs[train_idx], y_obs[train_idx]
        A_test, y_test = A_obs[test_idx], y_obs[test_idx]
        break

print("Train size:", A_train.shape, " Test size:", A_test.shape)

N = 8
uv_pairs = [(i // 8, i % 8) for i in range(64)]

plt.figure(figsize=(15, 5 * len(lambdas)))

for i, lam in enumerate(lambdas):

    # fit lasso with lambda
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=20000)
    model.fit(A_train, y_train)

    weights = model.coef_

    # full reconstruction
    y_hat = basis_matrix @ weights
    reconstructed_chip = y_hat.reshape(N, N)

    # plot
    plt.subplot(len(lambdas), 3, 3 * i + 1)
    plt.imshow(chip, cmap='gray', vmin=np.nanmin(chip), vmax=np.nanmax(chip))
    plt.title(f"Original Chip\n(Missing in Red)")
    plt.axis('off')

    plt.subplot(len(lambdas), 3, 3 * i + 2)
    plt.imshow(reconstructed_chip, cmap='gray')
    plt.title(f"LASSO Reconstructed\nλ = {lam:.1e}")
    plt.axis('off')

    plt.subplot(len(lambdas), 3, 3 * i + 3)
    x = np.arange(len(weights))
    markerline, stemlines, baseline = plt.stem(x, weights)
    plt.setp(markerline, 'marker', '.')   # cleaner markers
    plt.title(f"Weights (Stem Plot)\nλ = {lam:.1e}")
    plt.xlabel("Basis Index")
    plt.ylabel("Weight")

plt.tight_layout()
plt.show()

# compute mse
mse_values = []

for lam in lambdas:
    model = Lasso(alpha=lam, fit_intercept=False, max_iter=20000)
    model.fit(A_train, y_train)

    # Predict only the held-out test pixels
    y_test_hat = model.predict(A_test)

    # Compute MSE for this λ
    mse = np.mean((y_test_hat - y_test)**2)
    mse_values.append(mse)

# plot
plt.figure(figsize=(8, 5))
plt.plot(lambdas, mse_values, marker='o')
plt.xscale('log')
plt.xlabel("Regularization Parameter λ (log scale)")
plt.ylabel("Test MSE (First CV Fold)")
plt.title("Cross-Validation: Test MSE vs λ")
plt.grid(True)
plt.show()

all_mse = []

kf = KFold(n_splits=10, shuffle=True, random_state=0)

for train_idx, test_idx in kf.split(A_obs):
    # Training and test sets using the indices
    A_train, y_train = A_obs[train_idx], y_obs[train_idx]
    A_test, y_test = A_obs[test_idx], y_obs[test_idx]

    fold_mse = []

    for lam in lambdas:
        model = Lasso(alpha=lam, fit_intercept=False, max_iter=20000)
        model.fit(A_train, y_train)

        y_hat = model.predict(A_test)
        mse = np.mean((y_hat - y_test) ** 2)
        fold_mse.append(mse)

    all_mse.append(fold_mse)

all_mse = np.array(all_mse)
avg_mse = np.mean(all_mse, axis=0)

# plot
plt.figure(figsize=(10, 6))

for fold_idx in range(10):
    plt.plot(lambdas, all_mse[fold_idx, :], color='gray', linestyle='--', alpha=0.6, label=f'Individual Fold' if fold_idx == 0 else "")

plt.plot(lambdas, avg_mse, color='red', linewidth=3, marker='o', markersize=8, label='Average MSE')

plt.xscale('log')
plt.xlabel('Regularization Parameter λ (log scale)')
plt.ylabel('MSE on Held-Out Pixels')
plt.title('10-Fold Cross-Validation: MSE vs λ')
plt.grid(True)
plt.legend()
plt.show()

# find min mse lambda
best_idx = np.argmin(avg_mse)
best_lambda = lambdas[best_idx]
print(f"Best λ (minimizing average CV MSE): {best_lambda:.4e}")

# fit lasso
model = Lasso(alpha=best_lambda, fit_intercept=False, max_iter=20000)
model.fit(A_obs, y_obs)

# reconstruct
y_hat_full = model.predict(basis_matrix)
reconstructed_chip = y_hat_full.reshape(8, 8)

# display original and reconstructed chip
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(chip, cmap='gray', vmin=np.nanmin(chip), vmax=np.nanmax(chip))
plt.title("Original (Corrupted) 8×8 Chip")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_chip, cmap='gray', vmin=np.nanmin(chip), vmax=np.nanmax(chip))
plt.title(f"Reconstructed 8×8 Chip\n(λ = {best_lambda:.1e})")
plt.axis('off')

plt.show()

# stem plot
weights = model.coef_
plt.figure(figsize=(8, 5))
x = np.arange(len(weights))
markerline, stemlines, baseline = plt.stem(x, weights)
plt.setp(markerline, 'marker', '.')  # smaller markers
plt.xlabel("Basis Index")
plt.ylabel("Weight")
plt.title("LASSO Model Weights (Best λ)")
plt.show()

def reconstruct_block_cv(chip, basis_matrix, lambda_grid, kf):
    """
    performs cross validated lasso reconstruction for one 8x8 chip
    returns the reconstructed 8x8 chip
    """

    y = chip.flatten()
    mask = ~np.isnan(y)

    # if block has no missing pixels, return as-is
    if mask.sum() == 64:
        return chip.copy()

    y_obs = y[mask]
    A_obs = basis_matrix[mask, :]

    # cross validation for each lambda
    mse_list = []
    for lam in lambda_grid:
        fold_errors = []
        for train_idx, test_idx in kf.split(A_obs):
            A_train, y_train = A_obs[train_idx], y_obs[train_idx]
            A_test, y_test = A_obs[test_idx], y_obs[test_idx]

            model = Lasso(alpha=lam, fit_intercept=False, max_iter=20000)
            model.fit(A_train, y_train)
            pred = model.predict(A_test)
            fold_errors.append(np.mean((pred - y_test) ** 2))

        mse_list.append(np.mean(fold_errors))

    best_lam = lambda_grid[np.argmin(mse_list)]

    # refit on all observed data
    model = Lasso(alpha=best_lam, fit_intercept=False, max_iter=20000)
    model.fit(A_obs, y_obs)

    # reconstruct full 8x8 chip
    y_hat_full = basis_matrix @ model.coef_
    return y_hat_full.reshape(8, 8)

def reconstruct_full_image(img, basis_matrix, lambda_grid=None):
    if lambda_grid is None:
        lambda_grid = np.logspace(-3, 7, 11)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    H, W = img.shape
    rec = np.zeros_like(img)

    print("starting full reconstruction...")
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            chip = img[r:r+8, c:c+8]

            if chip.shape != (8, 8):
                continue

            rec_chip = reconstruct_block_cv(chip, basis_matrix, lambda_grid, kf)
            rec[r:r+8, c:c+8] = rec_chip

        print(f"finished row block {r//8 + 1} of {H//8}")

    print("done reconstructing full image")
    return rec

lambda_grid = np.logspace(-3, 7, 11)

# reconstructed without median filtering
reconstructed_no_median = reconstruct_full_image(img, basis_matrix, lambda_grid)

# apply median filtering (3x3 window)
reconstructed_median = median_filter(reconstructed_no_median, size=3)

# display side-by-side
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Corrupted Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_no_median, cmap='gray')
plt.title("Reconstructed (No Median Filter)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_median, cmap='gray')
plt.title("Reconstructed (With Median Filter)")
plt.axis('off')

plt.show()

# reconstruct full image and track per-block best lambda
H, W = img.shape
blocks_H = H // 8
blocks_W = W // 8

lambda_grid = np.logspace(-3, 7, 11)
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# initialize arrays
reconstructed_full = np.zeros_like(img)
best_lambda_map = np.zeros((blocks_H, blocks_W))

print("reconstructing image and recording best lambda per block...")

for i, r in enumerate(range(0, H, 8)):
    for j, c in enumerate(range(0, W, 8)):
        chip = img[r:r+8, c:c+8]
        if chip.shape != (8, 8):
            continue

        y = chip.flatten()
        mask = ~np.isnan(y)
        if mask.sum() == 64:
            reconstructed_full[r:r+8, c:c+8] = chip
            best_lambda_map[i, j] = np.log10(lambda_grid[0])  # arbitrary since no missing
            continue

        y_obs = y[mask]
        A_obs = basis_matrix[mask, :]

        # CV to choose lambda
        mse_list = []
        for lam in lambda_grid:
            fold_errors = []
            for train_idx, test_idx in kf.split(A_obs):
                A_train, y_train = A_obs[train_idx], y_obs[train_idx]
                A_test, y_test = A_obs[test_idx], y_obs[test_idx]

                model = Lasso(alpha=lam, fit_intercept=False, max_iter=20000)
                model.fit(A_train, y_train)
                pred = model.predict(A_test)
                fold_errors.append(np.mean((pred - y_test) ** 2))
            mse_list.append(np.mean(fold_errors))

        best_lam = lambda_grid[np.argmin(mse_list)]
        best_lambda_map[i, j] = np.log10(best_lam)

        # refit and reconstruct
        final_model = Lasso(alpha=best_lam, fit_intercept=False, max_iter=20000)
        final_model.fit(A_obs, y_obs)
        y_hat = basis_matrix @ final_model.coef_
        reconstructed_full[r:r+8, c:c+8] = y_hat.reshape(8, 8)

    print(f"finished row block {i+1} of {blocks_H}")

print("done reconstructing image and collecting best lambda values")

# === visualize log10 lambda map ===
plt.figure(figsize=(8, 6))
plt.imshow(best_lambda_map, cmap='viridis', interpolation='nearest')
plt.colorbar(label='log10(lambda)')
plt.title('Variation of Regularization Parameter Across Image (log10 scale)')
plt.xlabel('Block Column Index')
plt.ylabel('Block Row Index')
plt.show()

