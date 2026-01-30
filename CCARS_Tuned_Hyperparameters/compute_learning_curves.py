#!/usr/bin/env python
"""
compute_learning_curves.py

Standalone script to compute learning curves for existing CCARS results.
Uses saved wavelength selections - no need to re-run CARS.

Usage:
    python compute_learning_curves.py --dataset salinas --component 2 --wavelengths 50
    python compute_learning_curves.py --dataset indian_pines --component 3 --wavelengths 10 20 30 50
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_BASE = SCRIPT_DIR / "dataset"

DATASETS = {
    "indian_pines": {
        "cube_path": DATASET_BASE / "indian_pines_corrected.mat",
        "gt_path": DATASET_BASE / "indian_pines_gt.mat",
        "cube_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "block_size": 8,
        "wavelengths_csv": SCRIPT_DIR / "indianpines_wavelengths_200.csv"
    },
    "salinas": {
        "cube_path": DATASET_BASE / "salinas_corrected.mat",
        "gt_path": DATASET_BASE / "salinas_gt.mat",
        "cube_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "block_size": 16,
        "wavelengths_csv": SCRIPT_DIR / "wavelengths_salinas_corrected_204.csv"
    }
}

TUNED_PARAMS = {
    "SVM-RBF": {"C": 100, "gamma": 0.01},
    "Random Forest": {"n_estimators": 500, "max_depth": None}
}

RANDOM_SEED = 42


def load_cube_and_gt(dataset_name):
    """Load hyperspectral cube and ground truth."""
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


def checkerboard_split(cube, gt, block_size):
    """Checkerboard spatial split."""
    H, W, B = cube.shape
    all_classes = set(np.unique(gt)) - {0}
    
    best_split = None
    best_score = -1
    
    for offset in range(min(100, block_size * block_size)):
        offset_r = offset % block_size
        offset_c = (offset // block_size) % block_size
        
        block_rows = (np.arange(H) + offset_r) // block_size
        block_cols = (np.arange(W) + offset_c) // block_size
        
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
    return (cube[train_mask], cube[test_mask], gt[train_mask], gt[test_mask])


def apply_log10_snv(X):
    """Apply Log10 + SNV preprocessing (Nicola's method)."""
    X_log = np.log10(np.clip(X, 1e-10, None))
    mean = X_log.mean(axis=1, keepdims=True)
    std = X_log.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (X_log - mean) / std


def load_selected_wavelengths(results_dir, n_wavelengths, all_wavelengths_nm):
    """Load selected wavelengths from CARS results and convert to indices."""
    cars_results = results_dir / "cars_results"
    
    # Try to load from coefficients_all.csv
    coef_file = cars_results / "coefficients_all.csv"
    if coef_file.exists():
        df = pd.read_csv(coef_file)
        # Count frequency of each wavelength across all runs (last iteration per run)
        # Get last iteration for each run
        last_iters = df.groupby('Run')['Iteration'].max().reset_index()
        last_iters_df = df.merge(last_iters, on=['Run', 'Iteration'])
        
        # Count how often each wavelength appears in final selections
        wl_counts = last_iters_df['Wavelength'].value_counts()
        top_wl_nm = wl_counts.head(n_wavelengths).index.values
        
        # Convert nm to indices
        indices = []
        for wl_nm in top_wl_nm:
            # Find closest wavelength in the array
            idx = np.argmin(np.abs(all_wavelengths_nm - wl_nm))
            if idx not in indices:
                indices.append(idx)
        
        if len(indices) >= n_wavelengths:
            return np.array(indices[:n_wavelengths])
        else:
            # Fill with top indices if not enough unique
            return np.array(indices[:len(indices)])
    
    # Fallback: try wavelengths.txt
    wl_file = cars_results / "wavelengths.txt"
    if wl_file.exists():
        wl_nm = np.loadtxt(wl_file)
        indices = []
        for w in wl_nm[:n_wavelengths]:
            idx = np.argmin(np.abs(all_wavelengths_nm - w))
            if idx not in indices:
                indices.append(idx)
        return np.array(indices[:n_wavelengths])
    
    return None


def compute_and_plot_learning_curve(clf, clf_name, X_train, y_train, output_dir, 
                                     dataset_name, n_wavelengths, component):
    """Compute and plot learning curve."""
    print(f"    Computing learning curve for {clf_name}...")
    
    # Compute learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        clf, X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    
    # Calculate means and stds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Check for overfitting
    gap = train_mean[-1] - val_mean[-1]
    overfitting = "Yes" if gap > 0.1 else "Mild" if gap > 0.05 else "No"
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label=f'Training score')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='orange', label=f'Validation score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve - {dataset_name} - {clf_name} - {n_wavelengths} wavelengths (comp {component})')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.annotate(f'Gap: {gap:.3f}\nOverfitting: {overfitting}', 
                 xy=(0.02, 0.02), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save
    plot_path = output_dir / f"lc_{clf_name.replace(' ', '_')}_{n_wavelengths}wl_comp{component}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'train_size': train_sizes_abs,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std
    })
    csv_path = output_dir / f"lc_{clf_name.replace(' ', '_')}_{n_wavelengths}wl_comp{component}.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"      ✓ Saved: {plot_path.name}")
    
    return {
        'classifier': clf_name,
        'n_wavelengths': n_wavelengths,
        'component': component,
        'train_final': train_mean[-1],
        'val_final': val_mean[-1],
        'gap': gap,
        'overfitting': overfitting
    }


def main():
    parser = argparse.ArgumentParser(description='Compute learning curves for existing CCARS results')
    parser.add_argument('--dataset', type=str, required=True, choices=['salinas', 'indian_pines'])
    parser.add_argument('--component', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--wavelengths', type=int, nargs='+', default=[10, 20, 30, 50])
    parser.add_argument('--classifiers', type=str, nargs='+', default=['SVM-RBF', 'Random Forest'])
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"LEARNING CURVE COMPUTATION - {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Components: {args.component}")
    print(f"Wavelengths: {args.wavelengths}")
    print(f"Classifiers: {args.classifiers}")
    
    # Load data
    print("\nLoading data...")
    cube, gt = load_cube_and_gt(args.dataset)
    block_size = DATASETS[args.dataset]["block_size"]
    X_train_full, X_test_full, y_train, y_test = checkerboard_split(cube, gt, block_size)
    
    # Preprocess
    print("Applying Log10+SNV preprocessing...")
    X_train_full = apply_log10_snv(X_train_full)
    X_test_full = apply_log10_snv(X_test_full)
    
    print(f"Train: {X_train_full.shape}, Test: {X_test_full.shape}")
    
    # Load wavelength values from CSV
    wl_csv = DATASETS[args.dataset].get("wavelengths_csv")
    if wl_csv and wl_csv.exists():
        wl_df = pd.read_csv(wl_csv)
        all_wavelengths_nm = wl_df.iloc[:, 0].values
        print(f"Loaded {len(all_wavelengths_nm)} wavelengths from {wl_csv.name}")
    else:
        all_wavelengths_nm = np.arange(X_train_full.shape[1]).astype(float)
        print(f"Using sequential indices as wavelengths")
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = SCRIPT_DIR / f"{args.dataset}_learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for comp in args.component:
        print(f"\n{'='*50}")
        print(f"Component {comp}")
        print(f"{'='*50}")
        
        # Find results directory
        results_dir = SCRIPT_DIR / f"{args.dataset}_full" / f"component_{comp}"
        
        if not results_dir.exists():
            print(f"  ⚠️ Results directory not found: {results_dir}")
            continue
        
        for n_wl in args.wavelengths:
            print(f"\n  {n_wl} wavelengths:")
            
            # Load selected wavelengths
            selected_wl = load_selected_wavelengths(results_dir, n_wl, all_wavelengths_nm)
            
            if selected_wl is None:
                print(f"    ⚠️ Could not load selected wavelengths")
                # Fallback: use first n_wl bands
                selected_wl = np.arange(n_wl)
            
            # Filter by valid indices
            max_idx = X_train_full.shape[1] - 1
            selected_wl = selected_wl[selected_wl <= max_idx]
            
            if len(selected_wl) == 0:
                print(f"    ⚠️ No valid wavelengths")
                continue
            
            # Select features
            X_train_sel = X_train_full[:, selected_wl]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            
            for clf_name in args.classifiers:
                if clf_name == 'SVM-RBF':
                    clf = SVC(kernel='rbf', **TUNED_PARAMS['SVM-RBF'], 
                             class_weight='balanced', random_state=RANDOM_SEED)
                elif clf_name == 'Random Forest':
                    clf = RandomForestClassifier(**TUNED_PARAMS['Random Forest'],
                                                 class_weight='balanced_subsample',
                                                 n_jobs=-1, random_state=RANDOM_SEED)
                else:
                    continue
                
                result = compute_and_plot_learning_curve(
                    clf, clf_name, X_train_scaled, y_train,
                    output_dir, args.dataset, n_wl, comp
                )
                all_results.append(result)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_dir / "learning_curves_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved: {summary_path}")
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"✓ LEARNING CURVES COMPLETE")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
