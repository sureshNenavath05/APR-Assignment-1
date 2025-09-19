# Anime Score Classification using SVM

**Author:** Nenavath Suresh  
**Roll No:** 2201AI25

## Objective
Classify anime shows into score categories (`Very Good`, `Good`, `Average`, `Low`) based on features such as episodes, duration, genres, producers, studios, licensors, source material, and rating.

## Dataset
- Source file: `anime.csv`
- Key preprocessing:
  - Convert `Score` to numeric and bucket into classes:
    - `Very Good` (score >= 8.0)
    - `Good` (7.0 <= score < 8.0)
    - `Average` (6.0 <= score < 7.0)
    - `Low` (score < 6.0)
  - Numeric features: `Episodes`, `Duration` (converted to minutes)
  - Multi-label features parsed into lists: `Genres`, `Producers`, `Licensors`, `Studios`
  - Categorical features: `Source`, `Rating`
  - Missing numeric values filled with median; missing categorical filled as `Unknown`.

## Feature Engineering
- MultiLabelBinarizer for genres (kept genres with >= 30 occurrences)
- Top-k binary features for producers, studios, licensors
- Final feature matrix includes numeric features, one-hot encoded categorical features, and binary multi-label features.

## Model
- Support Vector Machine (SVC) with RBF kernel
- Hyperparameters:
  - `kernel='rbf'`
  - `C=10.0`
  - `gamma='scale'`
  - `class_weight='balanced'`
  - `random_state=42`
- Pipeline: preprocessing (StandardScaler + OneHotEncoder) → SVM classifier

## Training & Evaluation
- Train/test split: `test_size=0.2`, `random_state=42`, `stratify=y`
- Reported accuracy: **~63.5%**
- Observations:
  - Best performance for `Average` and `Low` classes.
  - `Very Good` class has few samples, leading to lower recall (~0.40).
  - Multi-label features (genres, producers, studios) contributed significantly.

## How to run
1. Place `anime.csv` in the project directory.
2. Install dependencies (example):
   ```
   pip install -r requirements.txt
   ```
3. Run the training pipeline script (example):
   ```
   python train_svm.py
   ```

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn

## Results (summary)
- Accuracy: ~63.5%
- Confusion matrix and classification report generated on test set.

## Future Work
- Hyperparameter tuning (GridSearch / RandomizedSearch)
- Ensemble methods
- Dimensionality reduction (PCA / feature selection)
- Use embeddings for multi-label textual features

## Files
- `train_svm.py` — training script
- `preprocess.py` — preprocessing and feature engineering
- `anime.csv` — dataset (not included)
- `svm_report.pdf` — detailed report. fileciteturn0file0

