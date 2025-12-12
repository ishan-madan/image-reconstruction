# $\textsf{\color{skyblue} Image Reconstruction with LASSO Regression}$

## $\textsf{\color{lightgreen} Description}$

This project implements a full grayscale image–reconstruction pipeline using **LASSO sparse regression**, **2D cosine basis functions**, and **cross-validated regularization tuning**.  
The system takes a user-provided image, converts it to grayscale, artificially removes a configurable percentage of pixels, and reconstructs the missing values using block-wise (8×8) LASSO on a learned cosine basis.  
It also provides visualization tools—including basis chips, weight distributions, block-wise λ-maps, and reconstruction comparisons—to help users understand the structure of the recovered image.

The result is a comprehensive framework that demonstrates sparse signal recovery, compressed sensing intuition, and pixel-level reconstruction through modern machine-learning techniques.

## $\textsf{\color{lightgreen} Features}$

### 1. **Local or Colab-Compatible Image Upload**
- Works seamlessly in both Google Colab and local environments.
- Accepts an input image and converts it to grayscale automatically.

### 2. **Pixel Corruption for Testing**
- Randomly removes a configurable fraction of pixels (default: 60%).
- Missing pixels are encoded as `NaN` and saved to a CSV file for downstream processing.
- The corrupted image is displayed with missing pixels highlighted in **red**.

### 3. **8×8 Block Extraction & Analysis**
- The image is divided into non-overlapping 8×8 blocks ("chips").
- For each block, the script:
  - Displays the corrupted version,
  - Generates cosine basis functions,
  - Builds a full 64×64 basis matrix for sparse reconstruction.

### 4. **Cosine Basis Generation**
- Implements 2D cosine basis chips parameterized by `(u, v)` frequencies.
- Displays sample basis chips and the full 64-vector basis matrix.

### 5. **LASSO Sparse Reconstruction**
- Uses LASSO (`sklearn.linear_model.Lasso`) to reconstruct missing pixels.
- Computes:
  - Top contributing basis components,
  - Reconstructed chips,
  - Coefficient stem plots.

### 6. **Cross-Validation for Optimal λ**
- Performs **10-fold cross-validation** on only the *observed* pixels.
- Evaluates reconstruction error across λ values ranging from `10^-3` to `10^7`.
- Visualizes:
  - Test MSE vs λ,
  - Fold-by-fold error curves,
  - Average MSE curve.

### 7. **Full-Image Reconstruction**
- Reconstructs every 8×8 block with its own optimal λ.
- Reassembles all blocks into the final reconstructed image.
- Optionally applies a **median filter** to smooth block artifacts.

### 8. **Best-Lambda Heatmap**
- Generates a heatmap showing the spatial variation of `log10(λ)` across the image.
- Useful for understanding where the reconstruction is more/less constrained.


## $\textsf{\color{lightgreen} Pipeline Overview}$

### **1. Image Processing**
- Load image

### **2. Convert to Grayscale or RGB**
- Depending on selected script.
- Optional normalization

### **3. Pixel Removal**
- Randomly sets a chosen fraction of pixels to `NaN`
- Saves corrupted result to CSV  
  (`field_test_image.csv`)

### **4. Visualization**
- Corrupted image with missing pixels in red
- Extracted 8×8 example chip

### **5. Basis Construction**
- Builds 64 basis chips using:
  \[
  \cos\left(\frac{\pi}{N}(x+0.5)u\right)\cos\left(\frac{\pi}{N}(y+0.5)v\right)
  \]
- Creates a 64×64 basis matrix

### **6. LASSO Reconstruction**
- Solves:
  \[
  \min_w \|A_\text{obs} w - y_\text{obs}\|^2 + \lambda \|w\|_1
  \]
- Reconstructs the chip using sparse coefficients  

### **7. Cross-Validation**
- 10-fold CV on observed pixels  
- Finds the λ with minimum MSE

### **8. Full-Image Recovery**
- Applies chip-wise reconstruction to entire image
- Uses block-wise best λ
- Optionally applies median filtering

### **9. Diagnostics**
- Weight stem plots  
- CV curves  
- λ heatmap  


## $\textsf{\color{lightgreen} Parameters You Can Modify}$

| Parameter | Meaning | Default |
|----------|---------|---------|
| `missing_fraction` | % of pixels randomly removed | `0.6` |
| `normalize_pixels` | Scale image to [0,1] | `True` |
| `lambda_grid` | CV λ values | `10^-3 → 10^7` |
| `max_iter` | LASSO solver iterations | `20000` |
| `chip size` | Reconstruction block size | `8×8` |


## $\textsf{\color{lightgreen} Libraries Used}$

- NumPy  
- SciPy (median filter)  
- scikit-learn (LASSO, KFold)  
- Matplotlib  
- PIL / Pillow  
- OS utilities  


## $\textsf{\color{skyblue} Project Walkthrough}$

A full visual walkthrough—including:
- experimental corrupted images  
- basis matrices  
- top weighted basis chips  
- LASSO reconstructions  
- CV curves  
- full image reconstruction (with & without median filter)  
- λ parameter heatmap  

…is produced automatically when running the script.

This project is ideal for exploring:
- compressed sensing  
- sparse coding  
- linear inverse problems  
- per-block cross-validated reconstruction  
- basis analysis techniques  
