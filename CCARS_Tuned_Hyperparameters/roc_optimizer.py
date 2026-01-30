"""
ROC Threshold Optimization for Multi-Class Classification

Implements one-vs-rest ROC curve analysis to find optimal decision
thresholds for each class, improving classification accuracy especially
for imbalanced classes.

Based on: Nicola Dilillo's CCARS methodology (Section 4.2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ROCOptimizer:
    """
    ROC curve analysis and threshold optimization for multi-class classification
    
    Uses one-vs-rest approach to find optimal classification thresholds
    for each class independently, improving performance over default 0.5 threshold.
    """
    
    def __init__(self, class_names=None, random_state=42):
        """
        Initialize ROC optimizer
        
        Args:
            class_names: List of class names (optional, for plotting)
            random_state: Random seed for reproducibility
        """
        self.class_names = class_names
        self.random_state = random_state
        
        # Will be populated by compute_roc_curves
        self.roc_data = {}  # {class_idx: {'fpr', 'tpr', 'thresholds', 'auc'}}
        self.optimal_thresholds = None
        self.n_classes = None
    
    def compute_roc_curves(self, y_true, y_proba):
        """
        Compute ROC curves for each class using one-vs-rest approach
        
        Args:
            y_true: True labels (1D array, shape: n_samples)
            y_proba: Predicted probabilities (2D array, shape: n_samples x n_classes)
        
        Returns:
            Dictionary with ROC data for each class
        """
        self.n_classes = y_proba.shape[1]
        
        # Binarize labels for one-vs-rest
        y_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        print(f"\nComputing ROC curves for {self.n_classes} classes...")
        
        for class_idx in range(self.n_classes):
            # Extract binary labels and probabilities for this class
            y_class = y_bin[:, class_idx]
            y_score =y_proba[:, class_idx]
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_class, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Store results
            self.roc_data[class_idx] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
            
            class_name = self.class_names[class_idx] if self.class_names else f"Class {class_idx}"
            print(f"  {class_name}: AUC = {roc_auc:.4f}")
        
        return self.roc_data
    
    def find_optimal_thresholds(self, method='closest'):
        """
        Find optimal threshold for each class
        
        Args:
            method: Optimization method
                - 'closest': Closest point to (0, 1) on ROC curve (default)
                - 'youden': Maximize Youden's J statistic (TPR - FPR)
                - 'balanced': Balance sensitivity and specificity
        
        Returns:
            Array of optimal thresholds for each class
        """
        if not self.roc_data:
            raise ValueError("Must call compute_roc_curves() first")
        
        print(f"\nFinding optimal thresholds using '{method}' method...")
        
        self.optimal_thresholds = np.zeros(self.n_classes)
        
        for class_idx in range(self.n_classes):
            fpr = self.roc_data[class_idx]['fpr']
            tpr = self.roc_data[class_idx]['tpr']
            thresholds = self.roc_data[class_idx]['thresholds']
            
            if method == 'closest':
                # Find point closest to (0, 1)
                distances = np.sqrt((1 - tpr)**2 + fpr**2)
                optimal_idx = np.argmin(distances)
            
            elif method == 'youden':
                # Maximize Youden's J = TPR - FPR
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
            
            elif method == 'balanced':
                # Balance sensitivity (TPR) and specificity (1-FPR)
                balance = (tpr + (1 - fpr)) / 2
                optimal_idx = np.argmax(balance)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Store optimal threshold
            self.optimal_thresholds[class_idx] = thresholds[optimal_idx]
            
            class_name = self.class_names[class_idx] if self.class_names else f"Class {class_idx}"
            print(f"  {class_name}: {self.optimal_thresholds[class_idx]:.4f}")
        
        print(f"\nMean threshold: {self.optimal_thresholds.mean():.4f}")
        print(f"Std threshold: {self.optimal_thresholds.std():.4f}")
        
        return self.optimal_thresholds
    
    def apply_thresholds(self, y_proba):
        """
        Apply class-specific optimal thresholds to probability predictions
        
        Args:
            y_proba: Predicted probabilities (shape: n_samples x n_classes)
        
        Returns:
            Predicted class labels using optimal thresholds
        """
        if self.optimal_thresholds is None:
            raise ValueError("Must call find_optimal_thresholds() first")
        
        n_samples = y_proba.shape[0]
        
        # Apply threshold to each class
        above_threshold = y_proba >= self.optimal_thresholds[np.newaxis, :]
        
        # For each sample, choose class with highest probability among those above threshold
        # If no class above threshold, choose highest probability
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            above = np.where(above_threshold[i])[0]
            
            if len(above) > 0:
                # Choose class with highest probability among those above threshold
                predictions[i] = above[np.argmax(y_proba[i, above])]
            else:
                # No class above threshold, choose highest probability
                predictions[i] = np.argmax(y_proba[i])
        
        return predictions
    
    def plot_roc_curves(self, output_path=None, figsize=(15, 10)):
        """
        Plot ROC curves for all classes with optimal thresholds marked
        
        Args:
            output_path: Path to save plot (optional)
            figsize: Figure size (width, height)
        """
        if not self.roc_data:
            raise ValueError("Must call compute_roc_curves() first")
        
        # Calculate grid size
        n_cols = min(4, self.n_classes)
        n_rows = int(np.ceil(self.n_classes / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        for class_idx in range(self.n_classes):
            ax = axes[class_idx]
            
            fpr = self.roc_data[class_idx]['fpr']
            tpr = self.roc_data[class_idx]['tpr']
            thresholds = self.roc_data[class_idx]['thresholds']
            roc_auc = self.roc_data[class_idx]['auc']
            
            # Plot ROC curve
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            
            # Plot diagonal (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
            
            # Mark optimal threshold if available
            if self.optimal_thresholds is not None:
                optimal_threshold = self.optimal_thresholds[class_idx]
                # Find index of optimal threshold
                optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
                ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                       label=f'Optimal (t={optimal_threshold:.3f})')
            
            # Formatting
            class_name = self.class_names[class_idx] if self.class_names else f"Class {class_idx}"
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(class_name, fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
        
        # Hide unused subplots
        for idx in range(self.n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('ROC Curves - One-vs-Rest Classification', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ ROC curves plot saved to: {output_path}")
        
        plt.close()
    
    def save_results(self, output_dir):
        """
        Save ROC analysis results to CSV files
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if self.optimal_thresholds is not None:
            # Save optimal thresholds
            threshold_df = pd.DataFrame({
                'class': range(self.n_classes),
                'class_name': [self.class_names[i] if self.class_names else f"Class {i}" 
                              for i in range(self.n_classes)],
                'optimal_threshold': self.optimal_thresholds,
                'default_threshold': [0.5] * self.n_classes,
                'difference': self.optimal_thresholds - 0.5
            })
            threshold_path = output_dir / 'optimal_thresholds.csv'
            threshold_df.to_csv(threshold_path, index=False)
            print(f"✓ Thresholds saved to: {threshold_path}")
        
        # Save AUC scores
        auc_df = pd.DataFrame({
            'class': range(self.n_classes),
            'class_name': [self.class_names[i] if self.class_names else f"Class {i}" 
                          for i in range(self.n_classes)],
            'auc': [self.roc_data[i]['auc'] for i in range(self.n_classes)]
        })
        auc_path = output_dir / 'roc_auc_scores.csv'
        auc_df.to_csv(auc_path, index=False)
        print(f"✓ AUC scores saved to: {auc_path}")


# Example usage
if __name__ == '__main__':
    """Test ROC optimizer module"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    print("\n" + "=" * 70)
    print("Testing ROC Optimizer Module")
    print("=" * 70)
    
    # Generate synthetic imbalanced data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=5, n_clusters_per_class=1,
        weights=[0.4, 0.25, 0.2, 0.1, 0.05],  # Imbalanced
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get probabilities
    y_proba = clf.predict_proba(X_test)
    
    # Test ROC optimizer
    roc_opt = ROCOptimizer(class_names=[f'Class_{i}' for i in range(5)])
    
    # Compute ROC curves
    roc_opt.compute_roc_curves(y_test, y_proba)
    
    # Find optimal thresholds
    optimal_thresholds = roc_opt.find_optimal_thresholds(method='closest')
    
    # Compare predictions
    y_pred_default = clf.predict(X_test)
    y_pred_optimized = roc_opt.apply_thresholds(y_proba)
    
    acc_default = accuracy_score(y_test, y_pred_default)
    acc_optimized = accuracy_score(y_test, y_pred_optimized)
    
    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print(f"Default threshold (0.5):  Accuracy = {acc_default:.4f}")
    print(f"Optimized thresholds:     Accuracy = {acc_optimized:.4f}")
    print(f"Improvement:              {(acc_optimized - acc_default)*100:+.2f}%")
    print("=" * 70)
    
    # Plot ROC curves
    roc_opt.plot_roc_curves('roc_curves_test.png')
    
    # Save results
    roc_opt.save_results(Path('roc_test_results'))
    
    print("\n" + "=" * 70)
    print("✅ ROC optimizer module validation complete!")
    print("=" * 70)
