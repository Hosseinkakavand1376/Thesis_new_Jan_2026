# CCARS Pipeline - Complete Documentation

## Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [Background Concepts](#background-concepts)
3. [The CCARS Algorithm](#the-ccars-algorithm)
4. [How to Run the Pipeline](#how-to-run-the-pipeline)
5. [Module-by-Module Explanation](#module-by-module-explanation)
6. [Mathematical Formulas](#mathematical-formulas)
7. [Output Files](#output-files)

---

## What is This Project?

This project classifies **hyperspectral images** (HSI) using machine learning. A hyperspectral image is like a regular photo, but instead of 3 color channels (RGB), it has **200+ spectral bands** representing light at different wavelengths.

### The Problem

With 200+ features (wavelengths), we have:

- **Too many features** → slow training, overfitting risk
- **Redundant information** → many wavelengths carry similar data

### The Solution: CCARS

**CCARS (Competitive Calibration Adaptive Reweighted Sampling)** selects the **most important wavelengths** (e.g., 50 out of 200), reducing dimensionality while maintaining classification accuracy.

### Datasets Used

| Dataset          | Size           | Bands | Classes | Description                                |
| ---------------- | -------------- | ----- | ------- | ------------------------------------------ |
| **Salinas**      | 512×217 pixels | 204   | 16      | Agricultural crops (lettuce, grapes, etc.) |
| **Indian Pines** | 145×145 pixels | 200   | 16      | Crops and vegetation types                 |

---

## Background Concepts

### 1. Hyperspectral Imaging (HSI)

A hyperspectral image is a 3D data cube:

```
        ┌─────────────────┐
       /                 /│
      /    BANDS        / │
     /   (wavelengths) /  │
    ┌─────────────────┐   │
    │                 │   │
    │     HEIGHT      │   │
    │                 │  /
    │                 │ /
    └─────────────────┘
           WIDTH
```

**Shape:** (Height × Width × Bands) = (512 × 217 × 204) for Salinas

Each pixel has a "spectrum" - a vector of 204 values representing reflectance at each wavelength.

### 2. Classification Task

Given a pixel's spectrum (204 values), predict which of 16 crop types it belongs to.

### 3. Wavelength Selection (Feature Selection)

Instead of using all 204 wavelengths:

- Select the **most informative** ones (e.g., 50)
- Train classifier on reduced data
- Faster training, often better accuracy

### 4. Train/Test Splitting

**Problem with random split:** Neighboring pixels are similar. If neighbors end up in train AND test, we get artificially high accuracy (data leakage).

**Solution:** **Checkerboard spatial split** - divide image into blocks like a chess board:

```
┌───┬───┬───┬───┐
│ T │ t │ T │ t │   T = Train (white squares)
├───┼───┼───┼───┤   t = Test (black squares)
│ t │ T │ t │ T │
├───┼───┼───┼───┤
│ T │ t │ T │ t │
└───┴───┴───┴───┘
```

This ensures spatial independence between train and test sets.

---

## The CCARS Algorithm

### Overview

CCARS has two key innovations:

1. **CARS (Competitive Adaptive Reweighted Sampling):** Iteratively eliminates less important wavelengths
2. **Calibration Split:** Uses separate data for wavelength selection vs. final evaluation

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Data                                                   │
│   - Load hyperspectral cube (H × W × Bands)                         │
│   - Flatten to 2D: (n_samples × Bands)                              │
│   - Apply checkerboard split → Train/Test sets                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Preprocessing (Log10 + SNV)                                 │
│   - Log10 transform: x' = log₁₀(x)                                  │
│   - SNV normalize: x'' = (x' - mean) / std                          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Calibration Split (Nicola's Innovation)                     │
│   - Split TRAIN data 50/50:                                         │
│     • Calibration Set: For wavelength selection ONLY                │
│     • Final Set: For final model evaluation ONLY                    │
│   - This prevents information leakage!                              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: CARS Wavelength Selection (on Calibration Set)              │
│   - Run 500 Monte Carlo runs                                        │
│   - Each run: 100 iterations of variable elimination                │
│   - Count how often each wavelength is selected                     │
│   - Top 50 most frequent wavelengths = SELECTED                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Train Classifiers (on Final Set)                            │
│   - Use ONLY selected wavelengths                                   │
│   - Train: SVM-RBF, Random Forest, PLS-DA                           │
│   - Evaluate on held-out test set                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Validation                                                  │
│   - Permutation tests: Is accuracy statistically significant?       │
│   - Learning curves: Is the model overfitting?                      │
└─────────────────────────────────────────────────────────────────────┘
```

### CARS Algorithm Details

Inside each Monte Carlo run (500 total):

```
Iteration 1: Start with ALL 204 wavelengths
     ↓
For i = 1 to 100:
     1. Sample 80% of training data randomly
     2. Fit PLS-DA model on current wavelengths
     3. Get importance score for each wavelength (|coefficient|)
     4. Calculate how many to keep: n = 204 × ratio(i)
     5. Keep top n wavelengths by importance
     ↓
Iteration 100: Final selected wavelengths (~1-5 remaining)
     ↓
Record which wavelengths survived each run
```

After 500 runs, count frequency of each wavelength. Select Top-N most frequent.

---

## Mathematical Formulas

### 1. Log10 Transformation

Converts reflectance values to log scale:

```
x' = log₁₀(x + ε)
```

Where ε = 10⁻¹⁰ (small value to avoid log(0))

**Why:** Log scale reduces dynamic range and makes data more normally distributed.

### 2. SNV (Standard Normal Variate) Normalization

Normalizes each spectrum (row) independently:

```
x_normalized = (x - μ) / σ

Where:
  μ = mean of spectrum (mean across all wavelengths for one pixel)
  σ = standard deviation of spectrum
```

**Why:** Removes baseline shifts and scatter effects. Each spectrum becomes zero-mean, unit-variance.

### 3. CARS Exponential Decay Ratio

Determines how many variables to keep at iteration i:

```
ratio(i) = a × exp(-k × i)

Where:
  N = total iterations (100)
  P = initial number of variables (204)
  a = (P/2)^(1/(N-1))
  k = ln(P/2) / (N-1)
```

This creates smooth exponential decay from 204 → ~1 variable over 100 iterations.

**Example for Salinas (P=204, N=100):**

- Iteration 1: Keep ~195 wavelengths
- Iteration 50: Keep ~14 wavelengths
- Iteration 100: Keep ~1 wavelength

### 4. Adaptive Reweighted Sampling (ARS)

Probabilistic selection based on importance scores:

```
probability(i) = |coefficient(i)| / Σ|coefficients|
```

Higher importance → higher chance of selection. This adds stochasticity across runs.

### 5. PLS-DA (Partial Least Squares Discriminant Analysis)

A supervised dimensionality reduction + regression method:

```
1. One-hot encode labels: y → Y (n_samples × n_classes)
2. PLS regression: X → latent variables T
3. Regression: T → Y_pred
4. Classification: argmax(Y_pred)
```

**Wavelength Importance:** The PLS coefficients indicate which wavelengths contribute most:

```
importance(j) = mean(|coefficient[j, c]|) for all classes c
```

### 6. Permutation Test

Tests null hypothesis: "Model has no predictive power"

```
1. Train model, get actual accuracy: A_actual
2. For i = 1 to n_permutations:
     - Shuffle labels randomly
     - Train model, get shuffled accuracy: A_shuffled(i)
3. p-value = (count(A_shuffled >= A_actual) + 1) / (n_permutations + 1)
```

If p-value < 0.05, accuracy is statistically significant.

### 7. Learning Curve Analysis

Detects overfitting by plotting train vs validation accuracy:

```
For training_size in [10%, 20%, ..., 100%]:
    - Train model on subset
    - Compute train accuracy and validation accuracy

Overfitting gap = train_accuracy - validation_accuracy

If gap > 10%: OVERFITTING
If gap 5-10%: MILD OVERFITTING
If gap < 5%: OK
```

---

## How to Run the Pipeline

### Basic Usage

```bash
# Navigate to the folder
cd CCARS_Tuned_Hyperparameters

# Run on Salinas dataset
python ccars_tuned_pipeline.py --dataset salinas

# Run on Indian Pines
python ccars_tuned_pipeline.py --dataset indian_pines
```

### Full Command with All Options

```bash
python ccars_tuned_pipeline.py \
    --dataset salinas \
    --components 2 3 4 \
    --wavelengths 10 20 30 50 \
    --cars_runs 500 \
    --cars_iterations 100 \
    --preprocessing log10_snv \
    --adaptive_permutations \
    --compute_learning_curves \
    --output my_results
```

### CLI Arguments Explained

| Argument                    | Default            | Description                                |
| --------------------------- | ------------------ | ------------------------------------------ |
| `--dataset`                 | (required)         | `salinas` or `indian_pines`                |
| `--components`              | `[2, 3, 4]`        | PLS components to test                     |
| `--wavelengths`             | `[10, 20, 30, 50]` | Number of wavelengths to select            |
| `--cars_runs`               | `500`              | Monte Carlo runs                           |
| `--cars_iterations`         | `100`              | Iterations per run                         |
| `--preprocessing`           | `log10_snv`        | Preprocessing method                       |
| `--adaptive_permutations`   | `False`            | Use classifier-specific permutation counts |
| `--compute_learning_curves` | `False`            | Generate learning curve plots              |
| `--output`                  | auto-generated     | Output directory                           |

---

## Module-by-Module Explanation

### 1. `ccars_tuned_pipeline.py` - Main Entry Point

**What it does:** Coordinates the entire analysis pipeline.

**Key Functions:**

| Function                           | What it does                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| `load_cube_and_gt()`               | Opens .mat files containing the hyperspectral image and ground truth labels        |
| `optimized_checkerboard_split()`   | Divides image into train/test using chess-board pattern to prevent spatial leakage |
| `TunedClassifierFramework`         | Contains all classifiers with optimized hyperparameters                            |
| `run_ccars_wavelength_selection()` | Runs the CARS algorithm to find best wavelengths                                   |
| `run_tuned_ccars_pipeline()`       | Master function that calls everything in order                                     |

---

### 2. `multiclass_cars.py` - The CARS Algorithm

**What it does:** Selects the most important wavelengths using Monte Carlo simulation.

**Key Functions:**

| Function                      | What it does                                                          |
| ----------------------------- | --------------------------------------------------------------------- |
| `prepare_calibration_split()` | Splits data 50/50 so wavelength selection doesn't see final test data |
| `run_cars()`                  | Runs 500 Monte Carlo simulations                                      |
| `_compute_ratio()`            | Calculates exponential decay for variable reduction                   |
| `_select_variables()`         | Probabilistically selects wavelengths based on importance             |
| `get_selected_wavelengths()`  | Counts frequencies and returns top N wavelengths                      |

**The Core Loop:**

```python
for run in range(500):           # 500 Monte Carlo runs
    selected = all_wavelengths   # Start with all 204

    for iteration in range(100): # 100 elimination rounds
        # 1. Sample 80% of data
        sample = random_sample(data, 0.8)

        # 2. Fit PLS-DA on current wavelengths
        model.fit(sample[:, selected])

        # 3. Get importance scores
        importance = abs(model.coefficients)

        # 4. How many to keep?
        n_keep = len(selected) * decay_ratio(iteration)

        # 5. Keep top n_keep by importance
        selected = top_n(selected, importance, n_keep)

    # Record which wavelengths survived
    record_survivors(selected)

# Count across all runs
frequencies = count_occurrences(all_survivors)
return top_N_most_frequent(frequencies, N=50)
```

---

### 3. `multiclass_plsda.py` - PLS-DA Classifier

**What it does:** A classifier that also provides wavelength importance scores.

**How PLS-DA Works:**

```
Step 1: One-Hot Encode Labels
  y = [0, 2, 1, 0, 2]  →  Y = [[1,0,0], [0,0,1], [0,1,0], [1,0,0], [0,0,1]]

Step 2: PLS Regression
  Find latent variables T that maximize correlation between X and Y
  T = X × Weights

Step 3: Predict
  Y_pred = T × Loadings
  class = argmax(Y_pred)  # Pick highest value
```

**Wavelength Importance:**
The PLS model has coefficients for each wavelength and each class. We aggregate:

```
importance[wavelength] = mean(|coefficient[wavelength, class]|) for all classes
```

---

### 4. `hsi_preprocessing.py` - Data Preprocessing

**What it does:** Transforms raw spectral data to normalized format.

**Preprocessing Pipeline:**

```
Raw Spectrum: [0.15, 0.23, 0.18, 0.31, ...]  (reflectance values)
        ↓
Step 1: Log10 Transform
    [log₁₀(0.15), log₁₀(0.23), ...] = [-0.82, -0.64, -0.74, -0.51, ...]
        ↓
Step 2: SNV Normalization
    mean = -0.68, std = 0.12
    [(-0.82+0.68)/0.12, (-0.64+0.68)/0.12, ...] = [-1.17, 0.33, -0.50, 1.42, ...]
        ↓
Final: Zero-mean, unit-variance spectrum
```

---

### 5. `permutation_test.py` - Statistical Validation

**What it does:** Tests if classifier performance is better than random guessing.

**How it Works:**

```
1. Train normally: accuracy = 85%

2. Shuffle labels randomly and repeat 100 times:
   - Shuffle 1: accuracy = 6%
   - Shuffle 2: accuracy = 7%
   - ...
   - Shuffle 100: accuracy = 5%

3. Count: How many shuffled accuracies ≥ 85%?
   Answer: 0 out of 100

4. p-value = (0 + 1) / (100 + 1) = 0.0099 < 0.05

5. Conclusion: Our 85% accuracy is statistically significant!
```

---

### 6. `learning_curve.py` - Overfitting Detection

**What it does:** Checks if model memorizes training data instead of learning patterns.

**How it Works:**

```
Train with 10% data:
  Training accuracy: 95%
  Validation accuracy: 60%
  Gap: 35% → OVERFITTING!

Train with 50% data:
  Training accuracy: 92%
  Validation accuracy: 82%
  Gap: 10% → Mild overfitting

Train with 100% data:
  Training accuracy: 90%
  Validation accuracy: 88%
  Gap: 2% → OK!
```

**Interpretation:**

- Large gap = Model memorizes training data
- Small gap = Model generalizes well
- If validation doesn't improve with more data = Need more features or different model

---

### 7. `roc_optimizer.py` - ROC Analysis

**What it does:** Finds optimal decision thresholds for each class.

**The Problem:**
By default, we predict the class with highest probability. But if classes are imbalanced, this doesn't work well.

**The Solution:**
Use ROC curves to find optimal threshold per class:

```
For each class:
  1. Compute ROC curve (FPR vs TPR at different thresholds)
  2. Find optimal point (closest to top-left corner)
  3. Use that threshold for this class

Prediction: For each class, if probability > threshold → predict positive
```

---

### 8. `multiclass_classifiers.py` - Classifier Collection

**What it does:** Provides multiple classifiers for comparison.

**Classifiers Available:**

| Classifier        | Description                   | When to Use                             |
| ----------------- | ----------------------------- | --------------------------------------- |
| **PLS-DA**        | Linear, uses latent variables | Best for CARS importance                |
| **SVM-Linear**    | Linear decision boundary      | When classes are linearly separable     |
| **SVM-RBF**       | Non-linear (Gaussian kernel)  | Complex boundaries, often best accuracy |
| **Random Forest** | Ensemble of decision trees    | Robust, handles noise well              |
| **k-NN**          | Distance-based                | Simple baseline                         |

---

### 9. Other Supporting Modules

| Module                       | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `hsi_data_loader.py`         | Load .mat files, convert 3D→2D                |
| `hsi_config.py`              | Dataset URLs, class names, download utilities |
| `hsi_evaluation.py`          | Compute metrics, plot confusion matrices      |
| `compute_learning_curves.py` | Standalone script for learning curves         |

---

## Tuned Hyperparameters

These were found using grid search with cross-validation:

| Classifier    | Parameter    | Tuned Value | What it Controls                                |
| ------------- | ------------ | ----------- | ----------------------------------------------- |
| SVM-RBF       | C            | 100         | Regularization strength (higher = more complex) |
| SVM-RBF       | gamma        | 0.01        | Kernel width (higher = more local)              |
| Random Forest | n_estimators | 500         | Number of trees                                 |
| Random Forest | max_depth    | None        | Tree depth (None = unlimited)                   |
| SVM-Linear    | C            | 1           | Regularization strength                         |

---

## Output Files

After running the pipeline:

```
salinas_full/
├── component_2/                    # Results for PLS component = 2
│   ├── cars_results/
│   │   ├── coefficients_all.csv   # All PLS coefficients per run/iteration
│   │   ├── statistics_all.csv     # Accuracy per run/iteration
│   │   └── wavelengths.txt        # All wavelength values (nm)
│   │
│   ├── comprehensive_results.csv  # Final accuracy table
│   │
│   ├── permutation_tests/
│   │   └── perm_*.png             # Histograms of permutation distributions
│   │
│   └── learning_curves/
│       ├── lc_*.png               # Learning curve plots
│       └── lc_*.csv               # Learning curve data
│
├── component_3/
├── component_4/
└── ...
```

### Key Output Files Explained

**`comprehensive_results.csv`:**

```csv
dataset,method,classifier,n_wavelengths,accuracy,f1_macro,kappa,p_value
salinas,CCARS_50,SVM-RBF,50,0.926,0.963,0.918,0.024
salinas,CCARS_30,SVM-RBF,30,0.908,0.949,0.897,0.024
...
```

**`coefficients_all.csv`:**

```csv
Run,Iteration,Wavelength,Coefficient
0,1,414.29,0.028
0,1,423.98,0.027
...
```

Contains the PLS coefficient for each wavelength at each iteration of each run.

---

## Glossary

| Term           | Definition                                            |
| -------------- | ----------------------------------------------------- |
| **HSI**        | Hyperspectral Image - image with 200+ spectral bands  |
| **Band**       | Single wavelength channel in HSI                      |
| **Wavelength** | Specific light frequency (measured in nm)             |
| **Spectrum**   | All band values for one pixel                         |
| **CARS**       | Competitive Adaptive Reweighted Sampling              |
| **CCARS**      | CARS with Calibration split                           |
| **PLS-DA**     | Partial Least Squares Discriminant Analysis           |
| **SNV**        | Standard Normal Variate normalization                 |
| **OA**         | Overall Accuracy                                      |
| **Kappa**      | Cohen's Kappa (agreement score accounting for chance) |
| **p-value**    | Probability of seeing result by random chance         |

---

## References

1. Dilillo, N. et al. - CCARS methodology for hyperspectral calibration
2. Li, H. et al. (2009) - Key wavelengths screening using competitive adaptive reweighted sampling
3. Barker, M. & Rayens, W. (2003) - Partial Least Squares for Discrimination
