# CCARS Tuned Hyperparameters

Self-contained folder with all scripts and modules for CCARS with **tuned hyperparameters**.

## Tuned Hyperparameters

| Classifier    | Old                            | Tuned                                |
| ------------- | ------------------------------ | ------------------------------------ |
| SVM-RBF       | C=10, gamma='scale'            | **C=100, gamma=0.01**                |
| Random Forest | n_estimators=200, max_depth=20 | **n_estimators=500, max_depth=None** |

## Files

### Main Pipeline

- **`ccars_tuned_pipeline.py`** - Complete CCARS with tuned params + checkerboard split

### Core Modules (Nicola's methodology)

- `multiclass_cars.py` - CCARS algorithm
- `multiclass_plsda.py` - PLS-DA classifier
- `multiclass_classifiers.py` - Multi-classifier framework
- `hsi_preprocessing.py` - SNV normalization
- `hsi_data_loader.py`, `hsi_config.py`, `hsi_evaluation.py`
- `permutation_test.py`, `learning_curve.py`

### Data

- `dataset/` - Indian Pines and Salinas .mat files

## Usage

```bash
cd CCARS_Tuned_Hyperparameters

# Full run with all components (2, 3, 4)
python ccars_tuned_pipeline.py --dataset salinas --adaptive_permutations --output salinas_tuned

# Skip permutations for faster testing
python ccars_tuned_pipeline.py --dataset indian_pines --skip_permutation
```

## Permutation Counts (--adaptive_permutations)

- PLS-DA: 100 | SVM: 40 | RF: 80
