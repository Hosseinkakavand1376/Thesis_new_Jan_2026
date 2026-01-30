"""
HSI Preprocessing Pipeline
Supports multiple preprocessing methods: SNV-only (Nicola's exact), Log10+SNV (HSI-adapted), or none
"""

import numpy as np
import pandas as pd
from typing import Union


def apply_log10_transform(X_df, epsilon=1e-10):
    """
    Apply log10 transformation to reflectance values
    
    Args:
        X_df: DataFrame with reflectance values
        epsilon: Small value to add before log to avoid log(0)
    
    Returns:
        DataFrame with log-transformed values
    """
    print("Applying Log10 transformation...")
    
    # Add epsilon to avoid log(0), then apply log10
    X_log = X_df.copy()
    
    # Check for negative or zero values
    min_val = X_df.min().min()
    if min_val <= 0:
        print(f"  Warning: Found values <= 0 (min: {min_val:.6f}). Adding offset.")
        offset = abs(min_val) + epsilon
        X_log = np.log10(X_df + offset)
    else:
        X_log = np.log10(X_df)
    
    print(f"  ✓ Log10 applied. Range: [{X_log.min().min():.4f}, {X_log.max().max():.4f}]")
    
    return X_log


def apply_snv_normalization(X_df):
    """
    Apply Standard Normal Variate (SNV) normalization
    
    SNV: (x - mean(x)) / std(x) per spectrum
    This removes scatter effects and baseline shifts
    
    For HSI data, SNV normalizes each spectrum (row) by computing
    mean and std across wavelengths (columns).
    
    Args:
        X_df: DataFrame where rows are spectra, columns are wavelengths
    
    Returns:
        DataFrame with SNV-normalized values
    """
    print(f"Applying SNV normalization (per spectrum, across wavelengths)...")
    
    # Apply SNV row-wise (across wavelengths for each spectrum)
    # axis=1 means compute mean/std across columns (wavelengths)
    epsilon = 1e-10
    
    means = X_df.mean(axis=1)
    stds = X_df.std(axis=1)
    
    # Replace zero/nan stds with epsilon to avoid division by zero
    stds = stds.where(stds > epsilon, epsilon)
    
    # Normalize: (value - row_mean) / row_std
    X_snv = X_df.sub(means, axis=0).div(stds, axis=0)
    
    # Clean up any NaN or Inf values that might have appeared
    # Replace NaN with 0 (mean-centered)
    X_snv = X_snv.fillna(0)
    
    # Replace Inf values with large finite values
    X_snv = X_snv.replace([np.inf, -np.inf], [10, -10])
    
    # Verify normalization
    sample_means = X_snv.mean(axis=1)
    sample_stds = X_snv.std(axis=1)
    
    # Check for remaining invalid values
    has_nan = X_snv.isna().any().any()
    has_inf = np.isinf(X_snv.values).any()
    
    if has_nan or has_inf:
        print(f"  Warning: Still has NaN: {has_nan}, Inf: {has_inf}")
    
    print(f"  ✓ SNV applied.")
    print(f"    Mean of spectrum means: {sample_means.mean():.6f} (should be ~0)")
    print(f"    Mean of spectrum stds: {sample_stds.mean():.6f} (should be ~1)")
    print(f"    Overall range: [{X_snv.min().min():.4f}, {X_snv.max().max():.4f}]")
    
    return X_snv


def preprocess_hsi_data(X_df, method='snv_only', epsilon=1e-10):
    """
    Complete preprocessing pipeline with multiple method options
    
    IMPORTANT: Nicola Dilillo uses ONLY SNV (confirmed from code review)
    
    Args:
        X_df: Input DataFrame with reflectance values
        method: Preprocessing method:
            - 'snv_only': SNV normalization only (Nicola's exact approach) [DEFAULT]
            - 'log10_snv': Log10 transformation + SNV (HSI-adapted approach)
            - 'none': No preprocessing
        epsilon: Small value for log transformation
    
    Returns:
        Preprocessed DataFrame
    """
    print("=" * 70)
    print("HSI Preprocessing Pipeline")
    print("=" * 70)
    print(f"Input shape: {X_df.shape}")
    print(f"Method: {method}")
    
    if method == 'snv_only':
        print("Pipeline: SNV only (Nicola's exact approach)")
        print()
        X_processed = apply_snv_normalization(X_df)
    
    elif method == 'log10_snv':
        print("Pipeline: Log10 → SNV (HSI-adapted)")
        print()
        X_processed = apply_log10_transform(X_df, epsilon=epsilon)
        X_processed = apply_snv_normalization(X_processed)
    
    elif method == 'none':
        print("Pipeline: No preprocessing")
        print()
        X_processed = X_df.copy()
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}. "
                        f"Choose 'snv_only', 'log10_snv', or 'none'")
    
    print()
    print("=" * 70)
    print("✅ Preprocessing complete!")
    print("=" * 70)
    
    return X_processed


def preprocess_train_test(X_train_df, X_test_df, method='snv_only'):
    """
    Preprocess train and test sets independently
    
    Important: Each set is preprocessed independently to avoid data leakage.
    SNV normalization is computed per spectrum, so no leakage occurs.
    
    Args:
        X_train_df: Training DataFrame
        X_test_df: Test DataFrame
        method: Preprocessing method ('snv_only', 'log10_snv', or 'none')
    
    Returns:
        Tuple of (X_train_preprocessed, X_test_preprocessed)
    """
    print("\n" + "=" * 70)
    print("Preprocessing Train and Test Sets")
    print("=" * 70)
    
    print("\n📊 Training set:")
    X_train_processed = preprocess_hsi_data(X_train_df, method=method)
    
    print("\n📊 Test set:")
    X_test_processed = preprocess_hsi_data(X_test_df, method=method)
    
    return X_train_processed, X_test_processed


def validate_preprocessing(X_original, X_processed, n_samples=5):
    """
    Validate preprocessing by checking a few sample spectra
    
    Args:
        X_original: Original DataFrame
        X_processed: Preprocessed DataFrame
        n_samples: Number of samples to check
    """
    print("\n" + "=" * 70)
    print("Preprocessing Validation")
    print("=" * 70)
    
    # Check for NaN or Inf values
    has_nan = X_processed.isna().any().any()
    has_inf = np.isinf(X_processed.values).any()
    
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("⚠️  Warning: Invalid values detected!")
        if has_nan:
            nan_count = X_processed.isna().sum().sum()
            print(f"  NaN count: {nan_count}")
        if has_inf:
            inf_count = np.isinf(X_processed.values).sum()
            print(f"  Inf count: {inf_count}")
    else:
        print("✓ No invalid values detected")
    
    # Sample statistics
    print(f"\nSample statistics ({n_samples} random spectra):")
    sample_indices = np.random.choice(len(X_processed), min(n_samples, len(X_processed)), replace=False)
    
    for idx in sample_indices:
        spectrum_orig = X_original.iloc[idx].values
        spectrum_proc = X_processed.iloc[idx].values
        
        print(f"\n  Spectrum {idx}:")
        print(f"    Original: mean={spectrum_orig.mean():.4f}, std={spectrum_orig.std():.4f}")
        print(f"    Processed: mean={spectrum_proc.mean():.4f}, std={spectrum_proc.std():.4f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    """Test preprocessing pipeline"""
    print("\n" + "=" * 70)
    print("Testing HSI Preprocessing Pipeline")
    print("=" * 70)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_bands = 50
    n_classes = 5
    
    # Simulate reflectance values (0.01 to 1.0)
    X_synthetic = np.random.rand(n_samples, n_bands) * 0.99 + 0.01
    y_synthetic = np.random.randint(0, n_classes, n_samples)
    sample_ids = np.arange(n_samples)
    
    # Create DataFrame
    index_tuples = list(zip(y_synthetic, sample_ids))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Class', 'Sample_ID'])
    wavelengths = np.linspace(400, 900, n_bands)
    
    X_df_synthetic = pd.DataFrame(X_synthetic, index=multi_index, columns=wavelengths)
    
    print(f"\nSynthetic data shape: {X_df_synthetic.shape}")
    print(f"Value range: [{X_df_synthetic.min().min():.4f}, {X_df_synthetic.max().max():.4f}]")
    
    # Test all preprocessing methods
    print("\n" + "=" * 70)
    print("Testing All Preprocessing Methods")
    print("=" *70)
    
    methods = ['snv_only', 'log10_snv', 'none']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method}")
        print(f"{'='*70}")
        X_processed = preprocess_hsi_data(X_df_synthetic, method=method)
        validate_preprocessing(X_df_synthetic, X_processed, n_samples=2)
    
    print("\n" + "=" * 70)
    print("✅ Preprocessing test complete!")
    print("=" * 70)
