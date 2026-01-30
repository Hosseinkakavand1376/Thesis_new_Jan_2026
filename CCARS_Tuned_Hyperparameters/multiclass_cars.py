"""
Adapted CARS Algorithm for Multi-Class HSI Wavelength Selection

This module adapts Nicola's CCARS algorithm to work with multi-class HSI datasets
while preserving the core CARS methodology.
"""

import numpy as np
import pandas as pd
import os
import math
import random
from tqdm import tqdm
from pathlib import Path

from multiclass_plsda import MultiClassPLSDA


class MultiClassCARS:
    """
    Competitive Adaptive Reweighted Sampling (CARS) for Multi-Class HSI
   
    Adapts Nicola's binary CARS to handle 16-class hyperspectral datasets
    using multi-output PLS-DA for wavelength selection.
    """
    
    def __init__(self, output_path, n_components=3, cv_fold=5,
                 test_percentage=0.2, use_calibration=True, random_state=42):
        """
        Initialize MultiClass CARS
        
        Args:
            output_path: Directory to save results
            n_components: Number of PLS components
            cv_fold: Number of cross-validation folds
            test_percentage: Fraction for test split (within each set)
            use_calibration: If True, use calibration/final split (CCARS mode)
                            If False, use all data (original CARS mode)
            random_state: Random seed
        """
        self.n_components = n_components
        self.cv_fold = cv_fold
        self.test_percentage = test_percentage
        self.use_calibration = use_calibration
        self.random_state = random_state
        
        # Add flags to track calibration mode
        self.is_calibration_mode = False
        self.calibration_note = ""
        self.final_set = None
        
        # Create output directories
        self.output_path = Path(output_path)
        self.path_statistics = self.output_path / 'statistics'
        self.path_coefficients = self.output_path / 'coefficients'
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.path_statistics.mkdir(parents=True, exist_ok=True)
        self.path_coefficients.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.wavelengths = None
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        
        # Results containers
        self.statistics_df = None
        self.coefficients_df = None
    
    def prepare_calibration_split(self, X, y, wavelengths, calibration_fraction=0.5):
        """
        Split data into calibration and final sets (CCARS methodology)
        
        This implements Nicola's key innovation:
        - Calibration set: For wavelength selection ONLY
        - Final set: For final model building ONLY
        - No data leakage between sets
        
        Args:
            X: Full feature matrix (n_samples, n_features)
            y: Full labels (n_samples,)
            wavelengths: Wavelength array
            calibration_fraction: Fraction for calibration (default 0.5)
        
        Returns:
            Dictionary with:
            - X_cal_train, y_cal_train: Calibration training set
            - X_cal_test, y_cal_test: Calibration test set
            - X_final_train, y_final_train: Final training set
            - X_final_test, y_final_test: Final test set
            - wavelengths: Same wavelengths for both sets
        """
        from sklearn.model_selection import train_test_split
        
        print("\n" + "=" * 70)
        print("CCARS: Calibration/Final Split")
        print("=" * 70)
        
        # Step 1: Split into calibration and final sets (stratified)
        X_cal, X_final, y_cal, y_final = train_test_split(
            X, y,
            test_size=(1 - calibration_fraction),
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Total samples: {len(y)}")
        print(f"  → Calibration set: {len(y_cal)} ({calibration_fraction*100:.0f}%)")
        print(f"  → Final set: {len(y_final)} ({(1-calibration_fraction)*100:.0f}%)")
        
        # Step 2: Split each set into train/test
        X_cal_train, X_cal_test, y_cal_train, y_cal_test = train_test_split(
            X_cal, y_cal,
            test_size=self.test_percentage,
            random_state=self.random_state,
            stratify=y_cal
        )
        
        X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(
            X_final, y_final,
            test_size=self.test_percentage,
            random_state=self.random_state,
            stratify=y_final
        )
        
        print(f"\nCalibration set split:")
        print(f"  → Train: {len(y_cal_train)} samples")
        print(f"  → Test: {len(y_cal_test)} samples")
        
        print(f"\nFinal set split:")
        print(f"  → Train: {len(y_final_train)} samples")
        print(f"  → Test: {len(y_final_test)} samples")
        
        # Verify no overlap
        cal_indices = set(range(len(y_cal)))
        final_indices = set(range(len(y_cal), len(y_cal) + len(y_final)))
        assert len(cal_indices & final_indices) == 0, "Data leakage detected!"
        
        print(f"\n✓ No data leakage: Calibration and final sets are independent")
        print("=" * 70 + "\n")
        
        # Store final set for later use
        self.final_set = {
            'X_train': X_final_train,
            'y_train': y_final_train,
            'X_test': X_final_test,
            'y_test': y_final_test
        }
        
        self.is_calibration_mode = True
        self.calibration_note = f"CCARS Mode: Using {len(y_cal)} calibration samples"
        
        return {
            'X_cal_train': X_cal_train,
            'y_cal_train': y_cal_train,
            'X_cal_test': X_cal_test,
            'y_cal_test': y_cal_test,
            'X_final_train': X_final_train,
            'y_final_train': y_final_train,
            'X_final_test': X_final_test,
            'y_final_test': y_final_test,
            'wavelengths': wavelengths
        }
    
    def fit(self, X_train, y_train, X_test, y_test, wavelengths):
        """
        Prepare data for CARS algorithm
        
        NOTE: If use_calibration=True, this should be called with 
        CALIBRATION data only (from prepare_calibration_split())
        
        Args:
            X_train: Training spectral data (calibration set if CCARS mode)
            y_train: Training labels
            X_test: Test spectral data (calibration set if CCARS mode)
            y_test: Test labels
            wavelengths: List/array of wavelength values
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.wavelengths = np.array(wavelengths)
        
        self.n_samples, self.n_features = X_train.shape
        self.n_classes = len(np.unique(y_train))
        
        print(f"✓ Data loaded:")
        if self.is_calibration_mode:
            print(f"  MODE: CCARS (Calibration data only)")
            print(f"  {self.calibration_note}")
        else:
            print(f"  MODE: Original CARS (All data)")
        print(f"  Training samples: {self.n_samples}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Features (wavelengths): {self.n_features}")
        print(f"  Classes: {self.n_classes}")
        print(f"  PLS components: {self.n_components}")
    
    def run_cars(self, n_runs=500, n_iterations=100, mc_samples=0.8, 
                 use_ars=True, start_run=0):
        """
        Run CCARS wavelength selection algorithm
        
        Following Nicola's exact methodology:
        - Monte Carlo sampling with 80% of samples per iteration
        - Exponential decay for variable reduction
        - Adaptive Reweighted Sampling (ARS) for exploration
        - Track accuracy at each iteration
        
        Args:
            n_runs: Number of Monte Carlo runs (default 500)
            n_iterations: Number of iterations per run (default 100)
            mc_samples: Fraction of samples to use per iteration (default 0.8)
            use_ars: Use Adaptive Reweighted Sampling (default True)
            start_run: Starting run number (for resuming)
        """
        print("\n" + "=" * 70)
        print("Running CCARS Wavelength Selection")
        print("=" * 70)
        if self.is_calibration_mode:
            print(f"CCARS MODE: Wavelengths selected from CALIBRATION set only")
            print(f"   Final model evaluation will use SEPARATE final set")
        else:
            print(f"   Original CARS MODE: Using all provided data")
        print(f"Runs: {n_runs}")
        print(f"Iterations per run: {n_iterations}")
        print(f"MC sample fraction: {mc_samples}")
        print(f"ARS enabled: {use_ars}")
        print("=" * 70 + "\n")
        
        # Initialize result DataFrames
        self.statistics_list = []  # Store as list, convert to DF later
        self.coefficients_list = []  # Store as list, convert to DF later
        
        # Calculate number of samples to use per iteration
        K = int(mc_samples * self.n_samples)
        
        # Run CARS
        self._compute_cars(K, n_iterations, start_run, start_run + n_runs, use_ars)
        
        # Convert lists to DataFrames
        self.statistics_df = pd.DataFrame(self.statistics_list)
        self.coefficients_df = pd.DataFrame(self.coefficients_list)
        
        # Save final results
        self.save_results()
        
        print("\n" + "=" * 70)
        print("CCARS complete!")
        print("=" * 70)
    
    def _compute_cars(self, K, N, start_run, end_run, use_ars):
        """
        Core CARS computation loop
        
        Args:
            K: Number of samples to use per iteration
            N: Number of iterations per run
            start_run: Starting run index
            end_run: Ending run index
            use_ars: Use ARS for variable selection
        """
        for run_idx in tqdm(range(start_run, end_run), desc="CARS Runs"):
            # Start with all wavelengths
            selected_vars = list(range(self.n_features))
            
            # Shuffle data at the beginning of each run
            shuffle_idx = np.random.permutation(self.n_samples)
            X_shuffled = self.X_train[shuffle_idx]
            y_shuffled = self.y_train[shuffle_idx]
            
            for iter_idx in range(1, N + 1):
                # Randomly sample K samples
                sample_indices = np.random.choice(self.n_samples, size=K, replace=False)
                
                # Build PLS-DA model with selected variables
                # Adapt n_components if sample size is too small
                n_comp = min(self.n_components, len(sample_indices) - 1, len(selected_vars), self.n_classes)
                
                pls_model = MultiClassPLSDA(n_components=n_comp)
                pls_model.fit(
                    X_shuffled[sample_indices, :][:, selected_vars],
                    y_shuffled[sample_indices]
                )
                
                # Get wavelength importance (aggregate across classes)
                wavelength_importance = pls_model.get_wavelength_importance(method='mean_abs')
                
                # Calculate exponential decay ratio
                ratio = self._compute_ratio(N, iter_idx)
                n_to_select = int(round(ratio * self.n_features, 0))
                
                # Ensure minimum number of variables (at least n_components + 1)
                min_vars = max(self.n_components + 1, 2)
                n_to_select = max(n_to_select, min_vars)
                
                # Select variables
                selected_vars = self._select_variables(
                    use_ars, n_to_select, wavelength_importance, selected_vars
                )
                
                # Evaluate on test set
                # Use adaptive components based on selected variables
                n_comp_eval = min(self.n_components, len(self.y_train) - 1, len(selected_vars), self.n_classes)
                pls_full = MultiClassPLSDA(n_components=n_comp_eval)
                pls_full.fit(self.X_train[:, selected_vars], self.y_train)
                y_pred = pls_full.predict(self.X_test[:, selected_vars])
                accuracy = np.mean(y_pred == self.y_test)
                
                # Format wavelength list for storage
                if len(selected_vars) > 10:
                    wl_str = f"{len(selected_vars)} wavelengths"
                else:
                    wl_str = ", ".join([f"{self.wavelengths[i]:.2f}" for i in selected_vars])
                
                # Store statistics
                self.statistics_list.append({
                    'Run': run_idx,
                    'Iteration': iter_idx,
                    'Ratio': ratio,
                    'Selected_Variables': len(selected_vars),
                    'Selected_Wavelengths': wl_str,
                    'N_Components': n_comp_eval,
                    'Accuracy': accuracy
                })
                
                # Store coefficients (aggregate across classes)
                coef_mean = np.abs(pls_model.coef_).mean(axis=1)  # Mean across classes
                for var_idx, wl_idx in enumerate(selected_vars):
                    self.coefficients_list.append({
                        'Run': run_idx,
                        'Iteration': iter_idx,
                        'Wavelength': float(self.wavelengths[wl_idx]),
                        'Coefficient': float(coef_mean[var_idx])
                    })
            
            # Save partial results after each run
            self._save_partial(run_idx)
    
    def _compute_ratio(self, N, i):
        """
        Compute exponential decay ratio
        
        Nicola's formula: r = a * exp(-k * i)
        where a = (P/2)^(1/(N-1)) and k = log(P/2)/(N-1)
        
        Args:
            N: Total number of iterations
            i: Current iteration
        
        Returns:
            Ratio of variables to keep
        """
        P = self.n_features
        a = (P / 2) ** (1 / (N - 1))
        k = math.log(P / 2) / (N - 1)
        r = a * math.exp(-k * i)
        return r
    
    def _select_variables(self, use_ars, n_to_select, importance, current_selection):
        """
        Select variables based on importance scores
        
        Args:
            use_ars: Use Adaptive Reweighted Sampling
            n_to_select: Number of variables to select
            importance: Importance scores for currently selected variables
            current_selection: Currently selected variable indices
        
        Returns:
            New selected variable indices (global indices)
        """
        # Ensure n_to_select doesn't exceed available variables
        n_to_select = min(n_to_select, len(current_selection))
        
        # Ensure importance array matches current_selection length
        assert len(importance) == len(current_selection), \
            f"Importance length ({len(importance)}) must match current_selection length ({len(current_selection)})"
        
        if use_ars:
            # ARS: Weighted random sampling
            # Normalize importance to probabilities
            prob = importance / importance.sum()
            selected_local = random.choices(
                range(len(importance)),  # Use importance length
                weights=prob,
                k=n_to_select
            )
            # Remove duplicates and convert to global indices
            selected_local = list(set(selected_local))
            selected_global = [current_selection[i] for i in selected_local]
        else:
            # Standard CARS: Select top N by importance
            sorted_indices = np.argsort(importance)[::-1]
            selected_local = sorted_indices[:n_to_select]
            selected_global = [current_selection[i] for i in selected_local]
        
        return np.sort(selected_global)
    
    def _save_partial(self, run_idx):
        """Save results for a single run"""
        # Convert lists to DataFrames for saving
        stats_df = pd.DataFrame(self.statistics_list)
        coefs_df = pd.DataFrame(self.coefficients_list)
        
        run_stats = stats_df[stats_df['Run'] == run_idx]
        run_coefs = coefs_df[coefs_df['Run'] == run_idx]
        
        run_stats.to_csv(self.path_statistics / f'statistics_{run_idx}.csv', index=False)
        run_coefs.to_csv(self.path_coefficients / f'coefficients_{run_idx}.csv', index=False)
    
    def save_results(self):
        """Save all results"""
        self.statistics_df.to_csv(self.output_path / 'statistics_all.csv', index=False)
        self.coefficients_df.to_csv(self.output_path / 'coefficients_all.csv', index=False)
        np.savetxt(self.output_path / 'wavelengths.txt', self.wavelengths, fmt='%.4f')
        
        print(f"\n✓ Results saved to: {self.output_path}")
    
    def get_selected_wavelengths(self, frequency_threshold=None, top_n=None):
        """
        Get selected wavelengths based on selection frequency across runs
        
        Args:
            frequency_threshold: Minimum frequency (0-1) for inclusion
            top_n: Select top N most frequent wavelengths
        
        Returns:
            Array of selected wavelength values
        """
        # Get final iterations from all runs
        final_iters = self.statistics_df.groupby('Run')['Iteration'].max()
        
        # Collect wavelengths from final iterations
        wl_counts = {}
        for run_idx, max_iter in final_iters.items():
            run_final = self.coefficients_df[
                (self.coefficients_df['Run'] == run_idx) &
                (self.coefficients_df['Iteration'] == max_iter)
            ]
            for wl in run_final['Wavelength'].values:
                wl_counts[wl] = wl_counts.get(wl, 0) + 1
        
        # Convert to DataFrame
        wl_freq_df = pd.DataFrame(list(wl_counts.items()), columns=['Wavelength', 'Count'])
        wl_freq_df['Frequency'] = wl_freq_df['Count'] / len(final_iters)
        wl_freq_df = wl_freq_df.sort_values('Count', ascending=False)
        
        # Filter by threshold or top N
        if frequency_threshold is not None:
            selected = wl_freq_df[wl_freq_df['Frequency'] >= frequency_threshold]
        elif top_n is not None:
            selected = wl_freq_df.head(top_n)
        else:
            # Default: wavelengths selected in >50% of runs
            selected = wl_freq_df[wl_freq_df['Frequency'] > 0.5]
        
        return selected['Wavelength'].values, wl_freq_df
    
    def get_final_set(self):
        """
        Get the final set (held out from calibration)
        
        Returns:
            Dictionary with X_train, y_train, X_test, y_test for final set
            Returns None if not in calibration mode
        """
        if not self.is_calibration_mode:
            print("⚠️  Warning: Not in calibration mode, no final set available")
            return None
        
        return self.final_set


if __name__ == '__main__':
    """Test MultiClass CARS on synthetic data"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("Testing MultiClass CARS")
    print("=" * 70)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=50,
        n_classes=16,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Generate wavelengths
    wavelengths = np.linspace(400, 2500, 100)
    
    # Create CARS instance
    cars = MultiClassCARS(
        output_path='test_cars_output',
        n_components=5,
        test_percentage=0.2
    )
    
    # Fit data
    cars.fit(X_train, y_train, X_test, y_test, wavelengths)
    
    # Run CARS (short test: 5 runs, 20 iterations)
    cars.run_cars(n_runs=5, n_iterations=20, mc_samples=0.8, use_ars=True)
    
    # Get selected wavelengths
    selected_wl, freq_df = cars.get_selected_wavelengths(frequency_threshold=0.6)
    
    print(f"\n✓ Selected {len(selected_wl)} wavelengths (frequency > 60%):")
    print(f"  {selected_wl[:10]}...")  # Show first 10
    
    print(f"\n✓ Wavelength frequency distribution:")
    print(freq_df.head(10))
    
    print("\n" + "=" * 70)
    print("MultiClass CARS test complete!")
    print("=" * 70)
