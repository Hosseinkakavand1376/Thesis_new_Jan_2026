#!/usr/bin/env python
"""
checkerboard_before_ccars_config.py

Feature Selection with Checkerboard Split + Before_ccars.py Configurations.
Uses method parameters from Before_ccars.py + tuned classifier hyperparameters.

Before_ccars.py Configurations:
- CARS: n_runs=120, PLS components=[2,3,4], mc_samples=0.80
- BOSS: n_submodels=120, max_iters=18, sample_ratio=0.632
- Fisher: pruning enabled, r_threshold=0.95, min_gap=1
- MRMR: lambda=1.0
- RFE: C=1.0, max_iter=10000

Tuned Hyperparameters:
- SVM-RBF: C=100, gamma=0.01
- Random Forest: n_estimators=500, max_depth=None
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import argparse
import json
import math
import random
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
OUTPUT_DIR = THESIS_DIR / "checkerboard_before_ccars_results"
RANDOM_SEED = 42

# ===========================================================================
# BEFORE_CCARS.PY CONFIGURATIONS
# ===========================================================================

# Block sizes for checkerboard
BLOCK_SIZES = {"indian_pines": 8, "salinas": 16}

# K values to evaluate
K_GRID = [5, 10, 15, 20, 30, 50]

# Fisher settings
FISHER_ENABLE_PRUNING = True
FISHER_R_THRESHOLD = 0.95
FISHER_MIN_INDEX_GAP = 1

# MRMR settings
MRMR_LAMBDA = 1.0

# RFE settings
RFE_C_LINEAR = 1.0
RFE_MAX_ITER = 10000

# CARS settings (from Before_ccars.py)
CARS_N_RUNS = 120
CARS_EDF_END_RATIO = 2.0
CARS_MC_SAMPLES = 0.80
PLS_COMPONENTS_GRID = [2, 3, 4]  # Run CARS with each

# BOSS settings (from Before_ccars.py)
BOSS_K_SUBMODELS = 120
BOSS_MAX_ITERS = 18
BOSS_SAMPLE_RATIO = 0.632

# Tuned classifier hyperparameters
BEST_PARAMS = {
    "indian_pines": {"SVM-RBF": {"C": 100, "gamma": 0.01}, "RF": {"n_estimators": 500, "max_depth": None}},
    "salinas": {"SVM-RBF": {"C": 100, "gamma": 0.01}, "RF": {"n_estimators": 500, "max_depth": None}}
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
# FEATURE SELECTION METHODS (with Before_ccars.py configurations)
# ===========================================================================
def corr_matrix(X):
    """Compute correlation matrix."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    Xn = Xc / std
    C = (Xn.T @ Xn) / max(Xn.shape[0] - 1, 1)
    return np.clip(C, -1.0, 1.0)


def fisher_score_selection(X_train, y_train, K):
    """Fisher Score with optional correlation pruning (Before_ccars config)."""
    F_scores, _ = f_classif(X_train, y_train)
    F_scores = np.nan_to_num(F_scores, nan=0.0)
    rank_idx = np.argsort(-F_scores)
    
    if not FISHER_ENABLE_PRUNING:
        return rank_idx[:K]
    
    # Pruning with correlation thresholding
    C = corr_matrix(X_train)
    selected = []
    
    for b in rank_idx:
        if len(selected) >= K:
            break
        # Check index gap
        if any(abs(b - k) <= FISHER_MIN_INDEX_GAP for k in selected):
            continue
        # Check correlation
        if selected and any(abs(C[b, k]) >= FISHER_R_THRESHOLD for k in selected):
            continue
        selected.append(b)
    
    # Fill if needed
    if len(selected) < K:
        for b in rank_idx:
            if len(selected) >= K:
                break
            if b not in selected:
                selected.append(b)
    
    return np.array(selected[:K])


def mrmr_selection(X_train, y_train, K):
    """MRMR with Before_ccars config (lambda=1.0)."""
    n_features = X_train.shape[1]
    
    relevance = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)
    relevance = np.nan_to_num(relevance, nan=0.0)
    
    corr = np.corrcoef(X_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    
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
                red = np.mean(np.abs(corr[f, selected]))
            else:
                red = 0.0
            score = rel - MRMR_LAMBDA * red
            
            if score > best_score:
                best_score = score
                best_feature = f
        
        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
    
    return np.array(selected)


def rfe_selection(X_train, y_train, K):
    """RFE with Before_ccars config (C=1.0)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    estimator = LinearSVC(C=RFE_C_LINEAR, max_iter=RFE_MAX_ITER, dual=True, random_state=RANDOM_SEED)
    rfe = RFE(estimator, n_features_to_select=K, step=0.1)
    rfe.fit(X_scaled, y_train)
    
    return np.where(rfe.support_)[0]


def cars_pls_selection(X_train, y_train, K, n_components=4):
    """CARS-PLS with Before_ccars config (n_runs=120, mc_samples=0.80)."""
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
    
    K_mc = int(CARS_MC_SAMPLES * n_samples)
    N = CARS_N_RUNS
    
    for run in range(CARS_N_RUNS):
        vars_selected = list(range(n_features))
        shuffle_idx = rng.permutation(n_samples)
        X_run = X_scaled[shuffle_idx]
        Y_run = Y[shuffle_idx]
        
        for iteration in range(1, N + 1):
            if len(vars_selected) <= K:
                break
            
            mc_indices = rng.choice(n_samples, size=min(K_mc, n_samples), replace=False)
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


def boss_selection(X_train, y_train, K, n_components=3):
    """BOSS with Before_ccars config (n_submodels=120, max_iters=18)."""
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    
    Y = np.zeros((n_samples, len(classes)), dtype=float)
    for i, c in enumerate(classes):
        Y[y_train == c, i] = 1.0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    band_frequency = np.zeros(n_features)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    
    boot_size = int(BOSS_SAMPLE_RATIO * n_samples)
    
    for sub in range(BOSS_K_SUBMODELS):
        boot_idx = rng.choice(n_samples, size=boot_size, replace=True)
        X_boot = X_scaled[boot_idx]
        Y_boot = Y[boot_idx]
        
        remaining = np.arange(n_features)
        
        for iteration in range(BOSS_MAX_ITERS):
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


# ===========================================================================
# CLASSIFICATION
# ===========================================================================
def train_and_evaluate(X_train, y_train, X_test, y_test, band_indices, dataset_name, classifier="SVM-RBF"):
    """Train and evaluate with tuned hyperparameters."""
    
    X_tr = X_train[:, band_indices] if band_indices is not None else X_train
    X_te = X_test[:, band_indices] if band_indices is not None else X_test
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    params = BEST_PARAMS.get(dataset_name, BEST_PARAMS["indian_pines"])
    
    if classifier == "SVM-RBF":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED, **params["SVM-RBF"])
    elif classifier == "RF":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED, **params["RF"])
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_tr_scaled, y_train)
    y_pred = clf.predict(X_te_scaled)
    
    return {
        "OA": accuracy_score(y_test, y_pred),
        "macroF1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred)
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='Before_ccars Config + Checkerboard Evaluation')
    parser.add_argument('--dataset', type=str, default='both', choices=['indian_pines', 'salinas', 'both'])
    parser.add_argument('--K', type=int, nargs='+', default=K_GRID, help='Band counts')
    parser.add_argument('--classifiers', type=str, nargs='+', default=['SVM-RBF', 'RF'])
    
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = ['indian_pines', 'salinas'] if args.dataset == 'both' else [args.dataset]
    
    all_results = []
    
    for ds_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds_name.upper()}")
        print(f"{'#'*70}")
        
        cube, gt = load_cube_and_gt(ds_name)
        H, W, B = cube.shape
        print(f"  Cube: {H}×{W}×{B} bands")
        
        # Create dataset-specific output directory
        ds_output_dir = OUTPUT_DIR / ds_name
        ds_output_dir.mkdir(parents=True, exist_ok=True)
        ds_results = []
        
        block_size = BLOCK_SIZES[ds_name]
        X_train, X_test, y_train, y_test = optimized_checkerboard_split(cube, gt, block_size)
        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
        
        # Full bands baseline
        print(f"\n  [ALL_BANDS] {B} bands...")
        for clf in args.classifiers:
            result = train_and_evaluate(X_train, y_train, X_test, y_test, None, ds_name, clf)
            res_entry = {"dataset": ds_name, "method": "ALL_BANDS", "K": B, "classifier": clf, "pls_comp": None, **result}
            all_results.append(res_entry)
            ds_results.append(res_entry)
            print(f"    {clf}: OA={result['OA']:.4f}")
        
        # Feature selection methods
        for K in args.K:
            print(f"\n  [K={K}]")
            
            # Fisher
            print(f"    Fisher...", end=" ", flush=True)
            try:
                selected = fisher_score_selection(X_train, y_train, K)
                method_oas = []
                for clf in args.classifiers:
                    result = train_and_evaluate(X_train, y_train, X_test, y_test, selected, ds_name, clf)
                    method_oas.append(result['OA'])
                    res_entry = {"dataset": ds_name, "method": "Fisher", "K": K, "classifier": clf, "pls_comp": None, **result}
                    all_results.append(res_entry)
                    ds_results.append(res_entry)
                print(f"✓ (best OA={max(method_oas):.4f})")
            except Exception as e:
                print(f"✗ {e}")
            
            # MRMR
            print(f"    MRMR...", end=" ", flush=True)
            try:
                selected = mrmr_selection(X_train, y_train, K)
                method_oas = []
                for clf in args.classifiers:
                    result = train_and_evaluate(X_train, y_train, X_test, y_test, selected, ds_name, clf)
                    method_oas.append(result['OA'])
                    res_entry = {"dataset": ds_name, "method": "MRMR", "K": K, "classifier": clf, "pls_comp": None, **result}
                    all_results.append(res_entry)
                    ds_results.append(res_entry)
                print(f"✓ (best OA={max(method_oas):.4f})")
            except Exception as e:
                print(f"✗ {e}")
            
            # RFE
            print(f"    RFE...", end=" ", flush=True)
            try:
                selected = rfe_selection(X_train, y_train, K)
                method_oas = []
                for clf in args.classifiers:
                    result = train_and_evaluate(X_train, y_train, X_test, y_test, selected, ds_name, clf)
                    method_oas.append(result['OA'])
                    res_entry = {"dataset": ds_name, "method": "RFE", "K": K, "classifier": clf, "pls_comp": None, **result}
                    all_results.append(res_entry)
                    ds_results.append(res_entry)
                print(f"✓ (best OA={max(method_oas):.4f})")
            except Exception as e:
                print(f"✗ {e}")
            
            # CARS with different PLS components
            for n_comp in PLS_COMPONENTS_GRID:
                print(f"    CARS_PLS{n_comp}...", end=" ", flush=True)
                try:
                    selected = cars_pls_selection(X_train, y_train, K, n_components=n_comp)
                    method_oas = []
                    for clf in args.classifiers:
                        result = train_and_evaluate(X_train, y_train, X_test, y_test, selected, ds_name, clf)
                        method_oas.append(result['OA'])
                        res_entry = {"dataset": ds_name, "method": f"CARS_PLS{n_comp}", "K": K, "classifier": clf, "pls_comp": n_comp, **result}
                        all_results.append(res_entry)
                        ds_results.append(res_entry)
                    print(f"✓ (best OA={max(method_oas):.4f})")
                except Exception as e:
                    print(f"✗ {e}")
            
            # BOSS
            print(f"    BOSS...", end=" ", flush=True)
            try:
                selected = boss_selection(X_train, y_train, K)
                method_oas = []
                for clf in args.classifiers:
                    result = train_and_evaluate(X_train, y_train, X_test, y_test, selected, ds_name, clf)
                    method_oas.append(result['OA'])
                    res_entry = {"dataset": ds_name, "method": "BOSS", "K": K, "classifier": clf, "pls_comp": None, **result}
                    all_results.append(res_entry)
                    ds_results.append(res_entry)
                print(f"✓ (best OA={max(method_oas):.4f})")
            except Exception as e:
                print(f"✗ {e}")

        # Save dataset-specific results
        ds_df = pd.DataFrame(ds_results)
        ds_df = ds_df.sort_values("OA", ascending=False)
        ds_out_path = ds_output_dir / f"{ds_name}_results.csv"
        ds_df.to_csv(ds_out_path, index=False)
        print(f"\n  ✓ Saved {ds_name} results to: {ds_out_path}")
    
    # Save results
    df = pd.DataFrame(all_results)
    df = df.sort_values("OA", ascending=False)
    
    out_path = OUTPUT_DIR / "comprehensive_results.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {out_path}")
    print(f"{'='*70}")
    print("\nTop 20 Results:")
    print(df.head(20).to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
