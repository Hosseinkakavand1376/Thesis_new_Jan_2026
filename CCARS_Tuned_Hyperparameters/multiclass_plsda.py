"""
Multi-Class PLS-DA Classifier
Extension of Nicola's binary PLS-DA to handle multi-class (16 classes)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict
from roc_optimizer import ROCOptimizer


class MultiClassPLSDA(BaseEstimator, ClassifierMixin):
    """
    Multi-class PLS-DA using multi-output PLS regression
    
    This extends Nicola's binary PLS-DA to support 16 classes by:
    1. One-hot encoding the target  (n_classes binary outputs)
    2. Training a single multi-output PLS regression model
    3. Predicting class as argmax of continuous outputs
    
    This maintains the regression framework while handling multiple classes.
    """
    
    def __init__(self, n_components=3, optimize_roc=False):
        """
        Initialize Multi-Class PLS-DA
        
        Args:
            n_components: Number of PLS components (latent variables)
            optimize_roc: If True, find optimal classification thresholds via ROC analysis
        """
        self.n_components = n_components
        self.optimize_roc = optimize_roc
        self.pls = PLSRegression(n_components=self.n_components)
        self.classes_ = None
        self.n_classes_ = None
        self.roc_optimizer = None
    
    def fit(self, X, y):
        """
        Fit PLS-DA model
        
        Args:
            X: (n_samples, n_features) - Spectral data
            y: (n_samples,) - Class labels (integer, 0-indexed)
        
        Returns:
            self
        """
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # One-hot encode targets
        y_binary = label_binarize(y, classes=self.classes_)
        
        # Handle binary case (label_binarize returns 1D for 2 classes)
        if self.n_classes_ == 2:
            y_binary = np.hstack([1 - y_binary, y_binary])
        
        # Fit PLS regression
        self.pls.fit(X, y_binary)
        
        # Optimize ROC thresholds if requested
        if self.optimize_roc:
            # Get cross-validated probabilities to avoid overfitting thresholds
            y_pred_cv = cross_val_predict(self.pls, X, y_binary, cv=5, method='predict')
            # Convert to probabilities via softmax
            y_pred_exp = np.exp(y_pred_cv - np.max(y_pred_cv, axis=1, keepdims=True))
            y_proba_cv = y_pred_exp / np.sum(y_pred_exp, axis=1, keepdims=True)
            
            # Compute ROC curves and find optimal thresholds
            class_names = [f"Class {c}" for c in self.classes_]
            self.roc_optimizer = ROCOptimizer(class_names=class_names)
            self.roc_optimizer.compute_roc_curves(y, y_proba_cv)
            self.roc_optimizer.find_optimal_thresholds(method='closest')
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            y_pred: (n_samples,) - Predicted class labels
        """
        if self.optimize_roc and self.roc_optimizer is not None:
            # Use ROC-optimized thresholds
            y_proba = self.predict_proba(X)
            y_pred_indices = self.roc_optimizer.apply_thresholds(y_proba)
            y_pred = self.classes_[y_pred_indices]
        else:
            # Default: use argmax of continuous predictions
            y_pred_continuous = self.pls.predict(X)
            y_pred_indices = np.argmax(y_pred_continuous, axis=1)
            y_pred = self.classes_[y_pred_indices]
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities (pseudo-probabilities via softmax)
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            y_proba: (n_samples, n_classes) - Probability-like scores
        """
        # Get continuous predictions
        y_pred_continuous = self.pls.predict(X)
        
        # Convert to probabilities using softmax
        # Subtract max for numerical stability
        y_pred_exp = np.exp(y_pred_continuous - np.max(y_pred_continuous, axis=1, keepdims=True))
        y_proba = y_pred_exp / np.sum(y_pred_exp, axis=1, keepdims=True)
        
        return y_proba
    
    def decision_function(self, X):
        """
        Get raw decision scores (continuous PLS outputs)
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            scores: (n_samples, n_classes) - Raw PLS predictions
        """
        return self.pls.predict(X)
    
    # @property
    # def coef_(self):
    #     """
    #     Get PLS regression coefficients
    #
    #     Returns:
    #         coef: (n_features, n_classes) - Regression coefficients
    #     """
    #     return self.pls.coef_
    #
    @property
    def x_weights_(self):
        """Get X weights"""
        return self.pls.x_weights_
    
    @property
    def y_loadings_(self):
        """Get Y loadings"""
        return self.pls.y_loadings_
    
    @property
    def coef_(self):
        """
        Get PLS coefficients in shape (n_features, n_classes)
        
        sklearn PLSRegression stores coef_ as (n_targets, n_features)
        We transpose to (n_features, n_targets) for consistency
        """
        return self.pls.coef_.T  # Transpose to (n_features, n_classes)
    
    def get_wavelength_importance(self, method='mean_abs'):
        """
        Compute wavelength importance for variable selection
        
        For multi-class, we have (n_features, n_classes) coefficients.
        We need to aggregate to get a single importance score per wavelength.
        
        Args:
            method: 'mean_abs' (mean absolute), 'max_abs' (max absolute), or 'l2_norm'
        
        Returns:
            importance: (n_features,) - Wavelength importance scores
        """
        coef = self.coef_  # Shape: (n_features, n_classes)
        
        if method == 'mean_abs':
            # Average absolute coefficient across classes
            importance = np.abs(coef).mean(axis=1)
        elif method == 'max_abs':
            # Maximum absolute coefficient across classes
            importance = np.abs(coef).max(axis=1)
        elif method == 'l2_norm':
            # L2 norm across classes
            importance = np.linalg.norm(coef, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'mean_abs', 'max_abs', or 'l2_norm'")
        
        return importance


def test_multiclass_plsda():
    """Test Multi-Class PLS-DA on synthetic data"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 70)
    print("Testing Multi-Class PLS-DA")
    print("=" * 70)
    
    # Generate synthetic multi-class data
    n_samples = 500
    n_features = 50
    n_classes = 16
    n_informative = 30
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nData:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    
    # Test different numbers of components
    for n_comp in [3, 5, 10]:
        print(f"\n{'-' * 70}")
        print(f"Testing with {n_comp} components")
        print('-' * 70)
        
        # Train model
        model = MultiClassPLSDA(n_components=n_comp)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\nAccuracy:")
        print(f"  Train: {train_acc:.4f}")
        print(f"  Test:  {test_acc:.4f}")
        
        # Check coefficient shape
        print(f"\nModel attributes:")
        print(f"  Coefficients shape: {model.coef_.shape}")
        print(f"  Expected: ({n_features}, {n_classes})")
        
        # Test wavelength importance
        importance = model.get_wavelength_importance(method='mean_abs')
        print(f"  Wavelength importance shape: {importance.shape}")
        print(f"  Top 5 important features: {np.argsort(importance)[::-1][:5]}")
        
        # Test probability prediction
        y_proba = model.predict_proba(X_test[:5])
        print(f"\nProbability prediction (first 5 samples):")
        print(f"  Shape: {y_proba.shape}")
        print(f"  Sum of probabilities (should be 1.0): {y_proba.sum(axis=1)}")
    
    print("\n" + "=" * 70)
    print("✅ Multi-Class PLS-DA test complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_multiclass_plsda()
