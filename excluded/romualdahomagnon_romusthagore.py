# Core Libraries
import numpy as np
import pandas as pd
import warnings

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score, precision_recall_curve, confusion_matrix
from lightgbm import LGBMClassifier

# Configurations
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score, roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from tqdm import tqdm
import joblib



import kagglehub

# Download latest version
path = kagglehub.dataset_download("jiaoyouzhang/stock-pledge-defaults-prediction")

print("Path to dataset files:", path)


# -------------------------
# Configuration
# -------------------------
target = "IsDefault"
n_splits = 7  # Number of cross-validation splits
random_state = 42

# -------------------------
# Load datasets
# -------------------------
train = pd.read_csv("/kaggle/input/stock-pledge-defaults-prediction/train.csv", index_col="Stock code")
test = pd.read_csv("/kaggle/input/stock-pledge-defaults-prediction/test.csv", index_col="Stock code")

# Initialize submission DataFrame
sub_fl = pd.DataFrame(index=test.index, columns=[target], dtype=np.int8).fillna(0)

# Separate features and target
Xtrain = train.drop(columns=[target])
ytrain = train[target]
Xtest = test.copy()

# Standardize column names
for df in [Xtrain, Xtest]:
    df.columns = [f"col{i}" for i in range(len(df.columns))]

# Handle categorical columns
cat_cols = Xtrain.select_dtypes(include=["object", "string", "category"]).columns
Xtrain[cat_cols] = Xtrain[cat_cols].fillna("missing").astype("category")
Xtest[cat_cols] = Xtest[cat_cols].fillna("missing").astype("category")

# Reset indices
Xtrain.index = range(len(Xtrain))
ytrain.index = range(len(Xtrain))
Xtest.index = range(len(Xtest))



print("\n================== OFFLINE MODEL TRAINING ==================\n")

# Initialize arrays for predictions
oof_preds = np.zeros(len(Xtrain))
test_preds = np.zeros(len(Xtest))

# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Model training loop
for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(Xtrain, ytrain), total=n_splits, desc="Folds")):
    print(f"\n============= FOLD {fold + 1} =============")
    
    # Split the data
    X_tr, X_val = Xtrain.iloc[train_idx], Xtrain.iloc[val_idx]
    y_tr, y_val = ytrain.iloc[train_idx], ytrain.iloc[val_idx]
    
    # Train model with early stopping
    model = LGBMClassifier(n_estimators=1000, random_state=random_state, verbose=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(50)]
    )
    
    # Predictions
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(Xtest)[:, 1]
    
    # Store predictions
    oof_preds[val_idx] = val_pred
    test_preds += test_pred / n_splits
    
    # Metrics
    logloss = log_loss(y_val, val_pred)
    f1_raw = f1_score(y_val, np.round(val_pred))
    auc = roc_auc_score(y_val, val_pred)
    print(f"Log-loss: {logloss:.8f} | F1: {f1_raw:.8f} | AUC: {auc:.8f}")

# -------------------------
# ðŸ”¥ Optimal Threshold Search
# -------------------------
thresholds = np.arange(0.05, 0.99, 0.01)
scores = {t: f1_score(ytrain, (oof_preds >= t).astype(int)) for t in thresholds}
best_threshold = max(scores, key=scores.get)
best_f1 = scores[best_threshold]

print(f"\nâœ… Best Threshold = {best_threshold:.4f} with F1 Score = {best_f1:.8f}")

# -------------------------
# Final Evaluation
# -------------------------
final_f1 = f1_score(ytrain, (oof_preds >= best_threshold).astype(int))
print(f"\nðŸš€ Final F1 Score with Best Threshold = {final_f1:.8f}")

# -------------------------
# Save Final Submission
# -------------------------
sub_fl[target] = (test_preds >= best_threshold).astype(int)

# Save the submission file
sub_fl.to_csv('final_submission.csv')
print("\nâœ… Submission file saved as 'final_submission.csv'")

# -------------------------
# Save Model for Later Use
# -------------------------
joblib.dump(model, 'best_model.pkl')
print("\nâœ… Best model saved as 'best_model.pkl'")



# Finalize predictions and save submission file
sub_fl[target] = (test_preds >= best_threshold).astype(np.uint8)
sub_fl.to_csv("submission.csv", index=True)

# Confirm that the file is saved
import os
print("\nâœ… Files in current directory:")
print(os.listdir())

# Display the first few rows of the submission file
print("\nâœ… Preview of submission.csv:")
print(sub_fl.head())





