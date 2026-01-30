"""
Learning Curve Analysis for Model Validation

Implements learning curve methodology to detect overfitting and
determine optimal training set size by plotting model performance
vs. training set size.

Based on: Nicola Dilillo's CCARS methodology
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve as sklearn_learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class LearningCurve:
    """
    Learning curve analysis for classifier validation
    
    Generates curves showing train/validation scores vs training set size
    to detect overfitting and determine optimal dataset size.
    """
    
    def __init__(self, cv_splits=5, random_state=42):
        """
        Initialize learning curve analyzer
        
        Args:
            cv_splits: Number of cross-validation splits
            random_state: Random seed for reproducibility
        """
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.results = {}
    
    def compute_learning_curve(self, model, X, y, 
                               train_sizes=None,
                               scoring='accuracy'):
        """
        Compute learning curve for a model
        
        Args:
            model: Classifier with .fit() and .score() methods
            X: Feature matrix
            y: Labels
            train_sizes: Array of training sizes to test (default: np.linspace(0.1, 1.0, 10))
            scoring: Metric to use ('accuracy', 'f1_weighted', etc.)
        
        Returns:
            Dictionary with train_sizes, train_scores, validation_scores
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        print(f"\nComputing learning curve...")
        print(f"  Train sizes: {[f'{s:.1%}' for s in train_sizes]}")
        print(f"  CV splits: {self.cv_splits}")
        print(f"  Scoring: {scoring}")
        
        # Use ShuffleSplit for cross-validation
        cv = ShuffleSplit(
            n_splits=self.cv_splits,
            test_size=0.2,
            random_state=self.random_state
        )
        
        # Compute learning curve
        train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        # Store results
        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'train_scores_raw': train_scores,
            'val_scores_raw': val_scores
        }
        
        # Calculate gap (overfitting indicator)
        results['gap_mean'] = results['train_scores_mean'] - results['val_scores_mean']
        results['gap_std'] = np.sqrt(results['train_scores_std']**2 + results['val_scores_std']**2)
        
        return results
    
    def analyze_overfitting(self, results, threshold=0.05):
        """
        Analyze learning curve results for overfitting
        
        Args:
            results: Results from compute_learning_curve()
            threshold: Gap threshold for overfitting detection (default 0.05 = 5%)
        
        Returns:
            Dictionary with overfitting analysis
        """
        gap = results['gap_mean']
        
        # Find maximum gap
        max_gap_idx = np.argmax(gap)
        max_gap = gap[max_gap_idx]
        max_gap_size = results['train_sizes'][max_gap_idx]
        
        # Find minimum gap (best generalization)
        min_gap_idx = np.argmin(gap)
        min_gap = gap[min_gap_idx]
        min_gap_size = results['train_sizes'][min_gap_idx]
        
        # Detect overfitting at final size
        final_gap = gap[-1]
        final_size = results['train_sizes'][-1]
        
        is_overfitting = final_gap > threshold
        
        analysis = {
            'max_gap': max_gap,
            'max_gap_size': max_gap_size,
            'min_gap': min_gap,
            'min_gap_size': min_gap_size,
            'final_gap': final_gap,
            'final_size': final_size,
            'is_overfitting': is_overfitting,
            'overfitting_severity': 'High' if final_gap > 0.1 else 'Moderate' if final_gap > 0.05 else 'Low'
        }
        
        return analysis
    
    def plot_learning_curve(self, results, title, output_path=None):
        """
        Plot learning curve with confidence intervals
        
        Args:
            results: Results from compute_learning_curve()
            title: Plot title
            output_path: Path to save plot (optional)
        """
        train_sizes = results['train_sizes']
        train_mean = results['train_scores_mean']
        train_std = results['train_scores_std']
        val_mean = results['val_scores_mean']
        val_std = results['val_scores_std']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Learning curves
        ax1.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=2, 
                label='Training score', markersize=6)
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='#2E86AB')
        
        ax1.plot(train_sizes, val_mean, 'o-', color='#A23B72', linewidth=2,
                label='Validation score', markersize=6)
        ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='#A23B72')
        
        ax1.set_xlabel('Training Set Size', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title(f'Learning Curve - {title}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Generalization gap
        gap = results['gap_mean']
        gap_std = results['gap_std']
        
        ax2.plot(train_sizes, gap, 'o-', color='#F18F01', linewidth=2, markersize=6)
        ax2.fill_between(train_sizes, gap - gap_std, gap + gap_std,
                        alpha=0.2, color='#F18F01')
        ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.7,
                   label='5% threshold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax2.set_xlabel('Training Set Size', fontsize=12)
        ax2.set_ylabel('Generalization Gap', fontsize=12)
        ax2.set_title(f'Overfitting Analysis - {title}', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {output_path}")
        
        plt.close()
    
    def save_results(self, results, analysis, output_path):
        """
        Save learning curve results to CSV
        
        Args:
            results: Results from compute_learning_curve()
            analysis: Analysis from analyze_overfitting()
            output_path: Path to save CSV
        """
        # Create DataFrame
        df = pd.DataFrame({
            'train_size': results['train_sizes'],
            'train_score_mean': results['train_scores_mean'],
            'train_score_std': results['train_scores_std'],
            'val_score_mean': results['val_scores_mean'],
            'val_score_std': results['val_scores_std'],
            'gap_mean': results['gap_mean'],
            'gap_std': results['gap_std']
        })
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
        
        # Also save analysis summary
        analysis_path = output_path.parent / f"{output_path.stem}_analysis.txt"
        with open(analysis_path, 'w') as f:
            f.write("LEARNING CURVE ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Maximum gap: {analysis['max_gap']:.4f} at {analysis['max_gap_size']} samples\n")
            f.write(f"Minimum gap: {analysis['min_gap']:.4f} at {analysis['min_gap_size']} samples\n")
            f.write(f"Final gap: {analysis['final_gap']:.4f} at {analysis['final_size']} samples\n\n")
            f.write(f"Overfitting detected: {analysis['is_overfitting']}\n")
            f.write(f"Overfitting severity: {analysis['overfitting_severity']}\n")
        
        print(f"✓ Analysis saved to: {analysis_path}")


# Example usage
if __name__ == '__main__':
    """Test learning curve module"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "=" * 70)
    print("Testing Learning Curve Module")
    print("=" * 70)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=50, n_informative=30,
        n_classes=5, random_state=42
    )
    
    # Create model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Create learning curve analyzer
    lc = LearningCurve(cv_splits=5, random_state=42)
    
    # Compute learning curve
    results = lc.compute_learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Analyze overfitting
    analysis = lc.analyze_overfitting(results, threshold=0.05)
    
    print("\n" + "=" * 70)
    print("Analysis Results:")
    print("=" * 70)
    print(f"Maximum gap: {analysis['max_gap']:.4f} at {analysis['max_gap_size']} samples")
    print(f"Minimum gap: {analysis['min_gap']:.4f} at {analysis['min_gap_size']} samples")
    print(f"Final gap: {analysis['final_gap']:.4f} at {analysis['final_size']} samples")
    print(f"Overfitting: {analysis['is_overfitting']} ({analysis['overfitting_severity']} severity)")
    print("=" * 70)
    
    # Plot
    lc.plot_learning_curve(results, 'Random Forest Test', 'learning_curve_example.png')
    
    # Save results
    from pathlib import Path
    lc.save_results(results, analysis, Path('learning_curve_results.csv'))
    
    print("\n" + "=" * 70)
    print("✅ Learning curve module validation complete!")
    print("=" * 70)
