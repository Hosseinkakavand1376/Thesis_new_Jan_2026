#!/usr/bin/env python
"""
checkerboard_comprehensive.py

Comprehensive Feature Selection & Classification Pipeline with Checkerboard Split.
Includes: Fisher Score, MRMR, RFE, CARS-PLS, BOSS
Uses optimized hyperparameters from tuning.

Usage:
    python checkerboard_comprehensive.py --dataset both
    python checkerboard_comprehensive.py --dataset indian_pines --K 20 30
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

THESIS_DIR = Path(r"D:\AdvanceMachineLearning\Advance project\AML_3DOS\Thesis")
DATASET_BASE = THESIS_DIR / "CCARS_Thesis" / "dataset"
OUTPUT_DIR = THESIS_DIR / "checkerboard_comprehensive_results"
RANDOM_SEED = 42

BLOCK_SIZES = {"indian_pines": 8, "salinas": 16}
K_VALUES = [10, 20, 30, 50]

# Best hyperparameters from tuning
BEST_PARAMS = {
    "indian_pines": {
        "SVM-RBF": {"C": 100, "gamma": 0.01},
        "Random Forest": {"n_estimators": 500, "max_depth": None},
        "SVM-Linear": {"C": 1}
    },
    "salinas": {
        "SVM-RBF": {"C": 100, "gamma": 0.01},
        "Random Forest": {"n_estimators": 500, "max_depth": None},
        "SVM-Linear": {"C": 1}
    }
}

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


# ============================================================================
# DATA LOADING
# ============================================================================
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
    """Optimized checkerboard split with class balance."""
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


# ============================================================================
# FEATURE SELECTION METHODS
# ============================================================================
def fisher_score_selection(X_train, y_train, K):
    """Fisher Score / ANOVA F-value based selection."""
    F_scores, _ = f_classif(X_train, y_train)
    F_scores = np.nan_to_num(F_scores, nan=0.0)
    ranking = np.argsort(-F_scores)
    return ranking[:K]


def mrmr_selection(X_train, y_train, K, lambda_weight=1.0):
    """Minimum Redundancy Maximum Relevance (mRMR) selection."""
    n_features = X_train.shape[1]
    
    relevance = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)
    relevance = np.nan_to_num(relevance, nan=0.0)
    
    corr_matrix = np.corrcoef(X_train.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    selected = []
    remaining = list(range(n_features))
    
    first_idx = np.argmax(relevance)
    selected.append(first_idx)
    remaining.remove(first_idx)
    
    while len(selected) < K and remaining:
        best_score = -np.inf
        best_feature = None
        
        for f in remaining:
            rel = relevance[f]
            if selected:
                red = np.mean(np.abs(corr_matrix[f, selected]))
            else:
                red = 0.0
            score = rel - lambda_weight * red
            
            if score > best_score:
                best_score = score
                best_feature = f
        
        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
    
    return np.array(selected)


def rfe_selection(X_train, y_train, K, C=1.0, max_iter=10000):
    """Recursive Feature Elimination with LinearSVC."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    estimator = LinearSVC(C=C, max_iter=max_iter, dual=True, random_state=RANDOM_SEED)
    rfe = RFE(estimator, n_features_to_select=K, step=0.1)
    rfe.fit(X_scaled, y_train)
    
    return np.where(rfe.support_)[0]


def cars_pls_selection(X_train, y_train, K, n_components=4, n_runs=50, n_iterations=50, mc_samples=0.8):
    """CARS-PLS feature selection (simplified for speed)."""
    import random
    import math
    
    n_samples, n_features = X_train.shape
    P = n_features
    classes = np.unique(y_train)
    
    Y = np.zeros((n_samples, len(classes)), dtype=float)
    for i, c in enumerate(classes):
        Y[y_train == c, i] = 1.0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    band_frequency = np.zeros(n_features)
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    K_mc = int(mc_samples * n_samples)
    
    for run in range(n_runs):
        vars_selected = list(range(n_features))
        shuffle_idx = rng.permutation(n_samples)
        X_run = X_scaled[shuffle_idx]
        Y_run = Y[shuffle_idx]
        
        for iteration in range(1, n_iterations + 1):
            if len(vars_selected) <= K:
                break
            
            mc_indices = rng.choice(n_samples, size=K_mc, replace=False)
            n_comp = min(n_components, len(vars_selected), len(classes))
            n_comp = max(1, n_comp)
            
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_run[mc_indices][:, vars_selected], Y_run[mc_indices])
            
            coef_matrix = np.asarray(pls.coef_)
            if coef_matrix.ndim == 1:
                abs_coefs = np.abs(coef_matrix)
            elif coef_matrix.shape[0] == len(vars_selected):
                abs_coefs = np.abs(coef_matrix).mean(axis=1)
            else:
                abs_coefs = np.abs(coef_matrix).mean(axis=0)
            
            if len(abs_coefs) != len(vars_selected):
                abs_coefs = np.ones(len(vars_selected))
            
            N = n_iterations
            a = (P / 2) ** (1 / (N - 1)) if N > 1 else 1.0
            k = math.log(P / 2) / (N - 1) if N > 1 else 0.0
            r = a * math.exp(-k * iteration)
            
            n_to_keep = max(K, int(round(r * P, 0)))
            n_to_keep = min(n_to_keep, len(vars_selected))
            
            if n_to_keep >= len(vars_selected):
                continue
            
            keep_indices = np.argsort(-abs_coefs)[:n_to_keep].tolist()
            vars_selected = [vars_selected[i] for i in keep_indices]
        
        for v in vars_selected:
            band_frequency[v] += 1
    
    ranking = np.argsort(-band_frequency)
    return ranking[:K]


def boss_selection(X_train, y_train, K, n_submodels=30, max_iters=10, n_components=3):
    """BOSS feature selection (simplified for speed)."""
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    
    Y = np.zeros((n_samples, len(classes)), dtype=float)
    for i, c in enumerate(classes):
        Y[y_train == c, i] = 1.0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    band_frequency = np.zeros(n_features)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    
    for sub in range(n_submodels):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_scaled[boot_idx]
        Y_boot = Y[boot_idx]
        
        remaining = np.arange(n_features)
        
        for iteration in range(max_iters):
            if len(remaining) <= K:
                break
            
            n_comp = min(n_components, len(remaining), len(classes) - 1)
            n_comp = max(1, n_comp)
            
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_boot[:, remaining], Y_boot)
            
            coef_matrix = np.asarray(pls.coef_)
            if coef_matrix.ndim == 1:
                coefs = np.abs(coef_matrix)
            elif coef_matrix.shape[0] == len(remaining):
                coefs = np.abs(coef_matrix).mean(axis=1)
            else:
                coefs = np.abs(coef_matrix).mean(axis=0)
            
            if len(coefs) != len(remaining):
                coefs = np.ones(len(remaining))
            
            threshold = np.mean(coefs)
            keep_mask = coefs >= threshold
            
            if keep_mask.sum() >= K:
                remaining = remaining[keep_mask]
            else:
                keep_idx = np.argsort(-coefs)[:max(K, int(len(remaining) * 0.8))]
                remaining = remaining[keep_idx]
        
        band_frequency[remaining] += 1
    
    ranking = np.argsort(-band_frequency)
    return ranking[:K]


# ============================================================================
# CLASSIFICATION
# ============================================================================
def train_and_evaluate(X_train, y_train, X_test, y_test, band_indices, dataset_name, classifier="SVM-RBF"):
    """Train and evaluate with best hyperparameters."""
    
    X_tr = X_train[:, band_indices] if band_indices is not None else X_train
    X_te = X_test[:, band_indices] if band_indices is not None else X_test
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    params = BEST_PARAMS.get(dataset_name, BEST_PARAMS["indian_pines"])
    
    if classifier == "SVM-RBF":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED, **params["SVM-RBF"])
    elif classifier == "Random Forest":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED, **params["Random Forest"])
    elif classifier == "SVM-Linear":
        clf = SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED, **params["SVM-Linear"])
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_tr_scaled, y_train)
    y_pred = clf.predict(X_te_scaled)
    
    return {
        "OA": accuracy_score(y_test, y_pred),
        "macroF1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred)
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Comprehensive Checkerboard Evaluation')
    parser.add_argument('--dataset', type=str, default='both', choices=['indian_pines', 'salinas', 'both'])
    parser.add_argument('--K', type=int, nargs='+', default=K_VALUES, help='Band counts to test')
    parser.add_argument('--classifiers', type=str, nargs='+', default=['SVM-RBF', 'Random Forest'])
    
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = ['indian_pines', 'salinas'] if args.dataset == 'both' else [args.dataset]
    
    methods = {
        "Fisher": fisher_score_selection,
        "MRMR": mrmr_selection,
        "RFE": rfe_selection,
        "CARS": cars_pls_selection,
        "BOSS": boss_selection,
    }
    
    all_results = []
    
    for ds_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name.upper()}")
        print(f"{'#'*70}")
        
        cube, gt = load_cube_and_gt(ds_name)
        H, W, B = cube.shape
        print(f"  Cube: {H}×{W}×{B} bands")
        
        block_size = BLOCK_SIZES[ds_name]
        X_train, X_test, y_train, y_test = optimized_checkerboard_split(cube, gt, block_size)
        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
        
        # Full bands baseline
        print(f"\n  [ALL_BANDS] Evaluating {B} bands...")
        for clf in args.classifiers:
            result = train_and_evaluate(X_train, y_train, X_test, y_test, None, ds_name, clf)
            all_results.append({
                "dataset": ds_name, "method": "ALL_BANDS", "K": B, "classifier": clf, **result
            })
            print(f"    {clf}: OA={result['OA']:.4f}")
        
        # Feature selection methods
        for K in args.K:
            print(f"\n  [K={K}]")
            
            for method_name, method_func in methods.items():
                print(f"    {method_name}...", end=" ")
                try:
                    selected = method_func(X_train, y_train, K)
                    
                    for clf in args.classifiers:
                        result = train_and_evaluate(X_train, y_train, X_test, y_test, selected, ds_name, clf)
                        all_results.append({
                            "dataset": ds_name, "method": method_name, "K": K, "classifier": clf, **result
                        })
                    
                    print(f"✓ (best OA={max(r['OA'] for r in all_results if r['method']==method_name and r['K']==K and r['dataset']==ds_name):.4f})")
                except Exception as e:
                    print(f"✗ {e}")
    
    # Save results
    df = pd.DataFrame(all_results)
    df = df.sort_values("OA", ascending=False)
    
    out_path = OUTPUT_DIR / "comprehensive_results.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {out_path}")
    print(f"{'='*70}")
    print("\nTop 15 Results:")
    print(df.head(15).to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
