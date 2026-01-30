#!/usr/bin/env python
"""
ccars_tuned_hyperparameters.py

CCARS Evaluation with Tuned Hyperparameters.
Uses the optimized classifier hyperparameters from checkerboard grid search.

Tuned Hyperparameters:
- SVM-RBF: C=100, gamma=0.01 (was C=10, gamma='scale')
- Random Forest: n_estimators=500, max_depth=None (was n_estimators=200, max_depth=20)

Usage:
    python ccars_tuned_hyperparameters.py --dataset salinas
    python ccars_tuned_hyperparameters.py --dataset indian_pines --pls_components 2 3 4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import argparse
import warnings
import sys
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import time

THESIS_DIR = Path(r"D:\AdvanceMachineLearning\Advance project\AML_3DOS\Thesis")
DATASET_BASE = THESIS_DIR / "CCARS_Thesis" / "dataset"
OUTPUT_DIR = THESIS_DIR / "ccars_tuned_results"
RANDOM_SEED = 42

# Add CCARS module path
if str(THESIS_DIR / "CCARS_Thesis") not in sys.path:
    sys.path.insert(0, str(THESIS_DIR / "CCARS_Thesis"))

# ===========================================================================
# TUNED HYPERPARAMETERS (from checkerboard grid search)
# ===========================================================================
TUNED_CLASSIFIERS = {
    "indian_pines": {
        "SVM-RBF": {"C": 100, "gamma": 0.01},  # Tuned (was C=10, gamma='scale')
        "RF": {"n_estimators": 500, "max_depth": None}  # Tuned (was n_estimators=200, max_depth=20)
    },
    "salinas": {
        "SVM-RBF": {"C": 100, "gamma": 0.01},
        "RF": {"n_estimators": 500, "max_depth": None}
    }
}

# CCARS/Before_ccars configurations
CARS_N_RUNS = 120
CARS_MC_SAMPLES = 0.80
K_VALUES = [10, 20, 30, 50]
PLS_COMPONENTS_GRID = [2, 3, 4]
BLOCK_SIZES = {"indian_pines": 8, "salinas": 16}

DATASETS = {
    "indian_pines": {
        "cube_path": DATASET_BASE / "indian_pines_corrected.mat",
        "gt_path": DATASET_BASE / "indian_pines_gt.mat",
        "cube_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
    },
    "salinas": {
        "cube_path": DATASET_BASE / "salinas_corrected.mat",
        "gt_path": DATASET_BASE / "salinas_gt.mat",
        "cube_key": "salinas_corrected",
        "gt_key": "salinas_gt",
    }
}


# ===========================================================================
# PLS-DA CLASSIFIER
# ===========================================================================
class PLSDA:
    """PLS-DA classifier."""
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pls = None
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        Y = np.zeros((len(y), n_classes))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0
        n_comp = min(self.n_components, X.shape[1], n_classes)
        self.pls = PLSRegression(n_components=n_comp, scale=False)
        self.pls.fit(X, Y)
        return self
    
    def predict(self, X):
        Y_pred = self.pls.predict(X)
        return self.classes_[np.argmax(Y_pred, axis=1)]


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_cube_and_gt(dataset_name):
    config = DATASETS[dataset_name]
    cube_mat = loadmat(str(config["cube_path"]))
    gt_mat = loadmat(str(config["gt_path"]))
    
    cube_key = config["cube_key"]
    for key in [cube_key, cube_key.lower(), cube_key.replace("_", "")]:
        if key in cube_mat:
            cube = cube_mat[key].astype(np.float32)
            break
    else:
        arrays = {k: v for k, v in cube_mat.items() if isinstance(v, np.ndarray) and v.ndim == 3}
        cube = arrays[max(arrays, key=lambda k: arrays[k].size)].astype(np.float32)
    
    gt_key = config["gt_key"]
    for key in [gt_key, gt_key.lower(), gt_key.replace("_", "")]:
        if key in gt_mat:
            gt = gt_mat[key].astype(np.int32)
            break
    else:
        arrays = {k: v for k, v in gt_mat.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        gt = arrays[max(arrays, key=lambda k: arrays[k].size)].astype(np.int32)
    
    return cube, gt


def optimized_checkerboard_split(cube, gt, block_size, max_retries=100):
    """Optimized checkerboard split with class balance checking."""
    H, W, B = cube.shape
    all_classes = set(np.unique(gt)) - {0}
    
    best_split = None
    best_score = -1
    
    for offset in range(max_retries):
        offset_r = offset % block_size
        offset_c = (offset // block_size) % block_size
        
        rows = np.arange(H)
        cols = np.arange(W)
        
        block_rows = (rows + offset_r) // block_size
        block_cols = (cols + offset_c) // block_size
        
        row_matrix = block_rows[:, np.newaxis]
        col_matrix = block_cols[np.newaxis, :]
        
        checkerboard = (row_matrix + col_matrix) % 2 == 0
        
        train_mask = checkerboard & (gt > 0)
        test_mask = ~checkerboard & (gt > 0)
        
        train_classes = set(np.unique(gt[train_mask])) - {0}
        test_classes = set(np.unique(gt[test_mask])) - {0}
        
        common = train_classes & test_classes
        score = len(common)
        
        if score > best_score:
            best_score = score
            best_split = (train_mask, test_mask)
        
        if train_classes == all_classes and test_classes == all_classes:
            break
    
    train_mask, test_mask = best_split
    
    X_train = cube[train_mask]
    y_train = gt[train_mask]
    X_test = cube[test_mask]
    y_test = gt[test_mask]
    
    return X_train, X_test, y_train, y_test


# ===========================================================================
# CCARS WITH CALIBRATION (Nicola's approach)
# ===========================================================================
import math
import random

def ccars_pls_selection(X_train, X_calibration, y_train, y_calibration, K, 
                        n_components=4, n_runs=120, mc_samples=0.80):
    """
    CCARS-PLS with Calibration Split (Nicola's methodology).
    
    Key difference from CARS: uses calibration set for cross-validation
    during variable selection, preventing overfitting to training data.
    """
    n_samples, n_features = X_train.shape
    P = n_features
    classes = np.unique(y_train)
    
    # One-hot encode for train and calibration
    Y_train = np.zeros((len(y_train), len(classes)), dtype=float)
    Y_calib = np.zeros((len(y_calibration), len(classes)), dtype=float)
    for i, c in enumerate(classes):
        Y_train[y_train == c, i] = 1.0
        Y_calib[y_calibration == c, i] = 1.0
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calibration)
    
    band_frequency = np.zeros(n_features)
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    K_mc = int(mc_samples * n_samples)
    N = n_runs
    
    for run in range(n_runs):
        vars_selected = list(range(n_features))
        shuffle_idx = rng.permutation(n_samples)
        X_run = X_train_scaled[shuffle_idx]
        Y_run = Y_train[shuffle_idx]
        
        for iteration in range(1, N + 1):
            if len(vars_selected) <= K:
                break
            
            mc_indices = rng.choice(n_samples, size=min(K_mc, n_samples), replace=False)
            n_comp = min(n_components, len(vars_selected), len(classes))
            n_comp = max(1, n_comp)
            
            # Train on train subset
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_run[mc_indices][:, vars_selected], Y_run[mc_indices])
            
            # CCARS difference: Validate on calibration set
            calib_pred = pls.predict(X_calib_scaled[:, vars_selected])
            calib_error = np.mean((calib_pred - Y_calib) ** 2)
            
            # Get coefficients
            coef_matrix = np.asarray(pls.coef_)
            if coef_matrix.ndim == 1:
                abs_coefs = np.abs(coef_matrix)
            elif coef_matrix.shape[0] == len(vars_selected):
                abs_coefs = np.abs(coef_matrix).mean(axis=1)
            else:
                abs_coefs = np.abs(coef_matrix).mean(axis=0)
            
            if len(abs_coefs) != len(vars_selected):
                abs_coefs = np.ones(len(vars_selected))
            
            # EDF ratio
            a = (P / 2) ** (1 / (N - 1)) if N > 1 else 1.0
            k = math.log(P / 2) / (N - 1) if N > 1 else 0.0
            r = a * math.exp(-k * iteration)
            
            n_to_keep = max(K, int(round(r * P, 0)))
            n_to_keep = min(n_to_keep, len(vars_selected))
            
            if n_to_keep >= len(vars_selected):
                continue
            
            # ARS weighted selection
            total = np.sum(abs_coefs)
            weights = abs_coefs / total if total > 0 else np.ones(len(vars_selected)) / len(vars_selected)
            
            sampled = set()
            attempts = 0
            while len(sampled) < n_to_keep and attempts < n_to_keep * 10:
                chosen = random.choices(range(len(vars_selected)), weights=weights.tolist(), k=1)[0]
                sampled.add(chosen)
                attempts += 1
            
            if len(sampled) < n_to_keep:
                sorted_by_weight = np.argsort(-abs_coefs)
                for idx in sorted_by_weight:
                    if len(sampled) >= n_to_keep:
                        break
                    sampled.add(idx)
            
            keep_indices = sorted(list(sampled))
            vars_selected = [vars_selected[i] for i in keep_indices]
        
        for v in vars_selected:
            band_frequency[v] += 1
    
    ranking = np.argsort(-band_frequency)
    return ranking[:K]


# ===========================================================================
# CLASSIFIER TRAINING WITH TUNED HYPERPARAMETERS
# ===========================================================================
def train_and_evaluate_tuned(X_train, y_train, X_test, y_test, band_indices, 
                              dataset_name, classifier="SVM-RBF"):
    """Train and evaluate with TUNED hyperparameters."""
    
    X_tr = X_train[:, band_indices] if band_indices is not None else X_train
    X_te = X_test[:, band_indices] if band_indices is not None else X_test
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    params = TUNED_CLASSIFIERS.get(dataset_name, TUNED_CLASSIFIERS["indian_pines"])
    
    start_time = time.time()
    
    if classifier == "SVM-RBF":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED, **params["SVM-RBF"])
    elif classifier == "RF":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED, **params["RF"])
    elif classifier == "PLS-DA":
        clf = PLSDA(n_components=4)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_tr_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred = clf.predict(X_te_scaled)
    
    return {
        "OA": accuracy_score(y_test, y_pred),
        "macroF1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "train_time": train_time
    }


# ===========================================================================
# WAVELENGTH OPTIMIZATION ANALYSIS (from main_hsi_cars_comprehensive.py)
# ===========================================================================
def find_best_wavelength_count(results_dict):
    """
    Find wavelength count with highest accuracy.
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
    
    Returns:
        best_count: Wavelength count with highest accuracy
        best_accuracy: Best accuracy achieved
    """
    if not results_dict:
        return None, None
    
    best_count = max(results_dict.keys(), key=lambda k: results_dict[k])
    best_accuracy = results_dict[best_count]
    
    return best_count, best_accuracy


def find_minimum_acceptable(results_dict, threshold=0.95):
    """
    Find minimum wavelength count achieving threshold of best accuracy.
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
        threshold: Fraction of best accuracy (default 0.95 = 95%)
    
    Returns:
        min_count: Minimum wavelength count meeting threshold
        min_accuracy: Accuracy at that count
    """
    if not results_dict:
        return None, None
    
    best_count, best_accuracy = find_best_wavelength_count(results_dict)
    target = threshold * best_accuracy
    
    # Sort by wavelength count
    sorted_counts = sorted(results_dict.keys())
    
    for count in sorted_counts:
        if results_dict[count] >= target:
            return count, results_dict[count]
    
    # If none meet threshold, return the best
    return best_count, best_accuracy


def detect_knee_point(results_dict):
    """
    Detect knee point (elbow) in accuracy vs wavelength count curve.
    Uses derivative-based approach to find point of diminishing returns.
    
    Args:
        results_dict: Dict mapping wavelength_count -> accuracy
    
    Returns:
        knee_count: Wavelength count at knee point
        knee_accuracy: Accuracy at knee point
    """
    if len(results_dict) < 3:
        return find_best_wavelength_count(results_dict)
    
    # Sort by wavelength count
    counts = sorted(results_dict.keys())
    accuracies = [results_dict[c] for c in counts]
    
    # Calculate first derivative (rate of change)
    derivatives = []
    for i in range(1, len(counts)):
        delta_acc = accuracies[i] - accuracies[i-1]
        delta_count = counts[i] - counts[i-1]
        derivatives.append(delta_acc / delta_count if delta_count > 0 else 0)
    
    # Find where derivative drops significantly (knee point)
    if len(derivatives) < 2:
        return counts[0], accuracies[0]
    
    # Look for point where derivative drops below mean
    mean_deriv = np.mean(derivatives)
    
    for i, deriv in enumerate(derivatives):
        if deriv < mean_deriv * 0.5:  # Derivative drops to < 50% of mean
            knee_idx = i
            return counts[knee_idx], accuracies[knee_idx]
    
    # If no clear knee, return smallest count above 90% of best
    best_acc = max(accuracies)
    for i, acc in enumerate(accuracies):
        if acc >= 0.90 * best_acc:
            return counts[i], acc
    
    return counts[0], accuracies[0]


def analyze_wavelength_optimization(results_df, dataset_name, classifier="SVM-RBF"):
    """
    Analyze wavelength optimization from results DataFrame.
    
    Args:
        results_df: DataFrame with columns [K, OA, classifier, method]
        dataset_name: Dataset name
        classifier: Classifier to analyze
    
    Returns:
        Dict with optimization results
    """
    # Filter for CCARS results with this classifier
    mask = (results_df['dataset'] == dataset_name) & \
           (results_df['classifier'] == classifier) & \
           (results_df['method'].str.startswith('CCARS'))
    
    subset = results_df[mask]
    
    if subset.empty:
        return None
    
    # Build results dict: K -> best OA across PLS components
    results_dict = {}
    for K in subset['K'].unique():
        K_results = subset[subset['K'] == K]
        results_dict[K] = K_results['OA'].max()
    
    # Find optimal points
    best_count, best_acc = find_best_wavelength_count(results_dict)
    min_count, min_acc = find_minimum_acceptable(results_dict, threshold=0.95)
    knee_count, knee_acc = detect_knee_point(results_dict)
    
    return {
        'best': {'K': best_count, 'OA': best_acc},
        'minimum_95': {'K': min_count, 'OA': min_acc},
        'knee': {'K': knee_count, 'OA': knee_acc},
        'all_results': results_dict
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='CCARS with Tuned Hyperparameters')
    parser.add_argument('--dataset', type=str, default='both', choices=['indian_pines', 'salinas', 'both'])
    parser.add_argument('--K', type=int, nargs='+', default=K_VALUES, help='Band counts')
    parser.add_argument('--pls_components', type=int, nargs='+', default=PLS_COMPONENTS_GRID)
    parser.add_argument('--classifiers', type=str, nargs='+', default=['SVM-RBF', 'RF', 'PLS-DA'])
    parser.add_argument('--calibration_fraction', type=float, default=0.5)
    
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = ['indian_pines', 'salinas'] if args.dataset == 'both' else [args.dataset]
    
    all_results = []
    
    for ds_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# CCARS EVALUATION: {ds_name.upper()}")
        print(f"# Using TUNED Hyperparameters")
        print(f"{'#'*70}")
        
        ds_output = OUTPUT_DIR / ds_name
        ds_output.mkdir(parents=True, exist_ok=True)
        
        cube, gt = load_cube_and_gt(ds_name)
        H, W, B = cube.shape
        print(f"  Cube: {H}×{W}×{B} bands")
        
        block_size = BLOCK_SIZES[ds_name]
        X_train_full, X_test, y_train_full, y_test = optimized_checkerboard_split(cube, gt, block_size)
        print(f"  Train: {len(y_train_full)}, Test: {len(y_test)}")
        
        # Split training into train + calibration for CCARS
        n_train = len(y_train_full)
        n_calib = int(n_train * args.calibration_fraction)
        
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(n_train)
        calib_indices = indices[:n_calib]
        train_indices = indices[n_calib:]
        
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
        X_calib = X_train_full[calib_indices]
        y_calib = y_train_full[calib_indices]
        
        print(f"  CCARS split: Train={len(y_train)}, Calibration={len(y_calib)}")
        
        # Full bands baseline
        print(f"\n  [ALL_BANDS] {B} bands...")
        for clf in args.classifiers:
            result = train_and_evaluate_tuned(X_train_full, y_train_full, X_test, y_test, None, ds_name, clf)
            all_results.append({
                "dataset": ds_name, "method": "ALL_BANDS", "K": B, 
                "classifier": clf, "pls_comp": None, **result
            })
            print(f"    {clf}: OA={result['OA']:.4f}")
        
        # CCARS for each PLS component and K value
        for n_comp in args.pls_components:
            print(f"\n  [CCARS PLS-{n_comp}]")
            
            for K in args.K:
                print(f"    K={K}...", end=" ", flush=True)
                try:
                    selected = ccars_pls_selection(
                        X_train, X_calib, y_train, y_calib, K,
                        n_components=n_comp, n_runs=CARS_N_RUNS, mc_samples=CARS_MC_SAMPLES
                    )
                    
                    method_oas = []
                    for clf in args.classifiers:
                        result = train_and_evaluate_tuned(X_train_full, y_train_full, X_test, y_test, selected, ds_name, clf)
                        method_oas.append(result['OA'])
                        all_results.append({
                            "dataset": ds_name, "method": f"CCARS_PLS{n_comp}", "K": K, 
                            "classifier": clf, "pls_comp": n_comp, **result
                        })
                    print(f"✓ (best OA={max(method_oas):.4f})")
                except Exception as e:
                    print(f"✗ {e}")
        
        # Save dataset-specific results
        ds_df = pd.DataFrame([r for r in all_results if r['dataset'] == ds_name])
        ds_df = ds_df.sort_values("OA", ascending=False)
        ds_df.to_csv(ds_output / f"{ds_name}_ccars_tuned_results.csv", index=False)
        
        # Wavelength Optimization Analysis
        print(f"\n  [WAVELENGTH OPTIMIZATION ANALYSIS]")
        for clf in args.classifiers:
            opt_results = analyze_wavelength_optimization(ds_df, ds_name, clf)
            if opt_results:
                print(f"    {clf}:")
                print(f"      Best:     K={opt_results['best']['K']}, OA={opt_results['best']['OA']:.4f}")
                print(f"      Min(95%): K={opt_results['minimum_95']['K']}, OA={opt_results['minimum_95']['OA']:.4f}")
                print(f"      Knee:     K={opt_results['knee']['K']}, OA={opt_results['knee']['OA']:.4f}")
    
    # Save all results
    df = pd.DataFrame(all_results)
    df = df.sort_values("OA", ascending=False)
    
    out_path = OUTPUT_DIR / "ccars_tuned_all_results.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {out_path}")
    print(f"{'='*70}")
    print("\nTop 15 Results:")
    print(df.head(15).to_string(index=False))
    
    # Summary: Comparison table
    print(f"\n{'='*70}")
    print("SUMMARY: BEST CCARS RESULTS BY DATASET & CLASSIFIER")
    print("="*70)
    
    for ds_name in datasets:
        full_bands_oa = df[(df['dataset'] == ds_name) & (df['method'] == 'ALL_BANDS')].groupby('classifier')['OA'].first()
        best_ccars = df[(df['dataset'] == ds_name) & (df['method'].str.startswith('CCARS'))].groupby('classifier').first()
        
        print(f"\n{ds_name.upper()}:")
        print(f"{'Classifier':<12} {'Full Bands':<12} {'Best CCARS':<12} {'K':<8} {'Reduction':<10}")
        print("-" * 56)
        
        for clf in args.classifiers:
            if clf in full_bands_oa.index and clf in best_ccars.index:
                full_oa = full_bands_oa[clf]
                ccars_row = best_ccars.loc[clf]
                ccars_oa = ccars_row['OA']
                ccars_k = ccars_row['K']
                n_bands = df[df['dataset'] == ds_name]['K'].max()
                reduction = (1 - ccars_k / n_bands) * 100
                print(f"{clf:<12} {full_oa:<12.4f} {ccars_oa:<12.4f} {ccars_k:<8.0f} {reduction:<10.1f}%")
    
    # Compare with old hyperparameters
    print(f"\n{'='*70}")
    print("TUNED vs OLD HYPERPARAMETERS")
    print("="*70)
    print("SVM-RBF: C=100, gamma=0.01 (was C=10, gamma='scale')")
    print("RF: n_estimators=500, max_depth=None (was n_estimators=200, max_depth=20)")
    
    return df


if __name__ == "__main__":
    main()
