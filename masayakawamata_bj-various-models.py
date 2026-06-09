import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv")
test = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv")
print("Train Shape:", train.shape)
print("Test Shape :", test.shape)
train.head(3)


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


TARGET = 'ev'
X = train.drop([TARGET, "id"], axis=1).copy()
y = train[TARGET].copy()
X_test = test.drop(columns='id').copy()


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Define number of folds and initialize KFold
FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Dictionary mapping model names to their respective regressor instances
models = {
    "XGBRegressor": XGBRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        random_state=42,
        verbosity=0
    ),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTreesRegressor": ExtraTreesRegressor(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        random_state=42
    ),
    "LGBMRegressor": LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        colsample_bytree=0.5,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "KNeighborsRegressor": KNeighborsRegressor(
        n_neighbors=5,
        n_jobs=-1
    ),
    "SVR": SVR(
        C=1.0,
        epsilon=0.1,
        kernel='rbf'
    )
}

# Dictionary to store results for each model
results = {}

# Loop through each model and perform cross-validation
for model_name, model in models.items():
    print(f"\nTraining with {model_name} ...")
    oof_pred = np.zeros(len(X))
    fold_mse = []
    test_preds = np.zeros((len(X_test), FOLDS))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        print(f"  Fold {fold} ...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # For XGBoost and LightGBM, use early stopping
        if model_name in ["XGBRegressor"]:
            model.fit(
                X_train, y_train,
                early_stopping_rounds=100,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif model_name in ["LGBMRegressor"]:
            model.fit(
                X_train, y_train,
                callbacks=[early_stopping(100)],
                eval_set=[(X_val, y_val)],
            )
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        mse_fold = mean_squared_error(y_val, y_pred)
        fold_mse.append(mse_fold)
        oof_pred[val_idx] = y_pred
        print(f"    Fold {fold} MSE: {mse_fold:.8f}")
        
        test_preds[:, fold - 1] = model.predict(X_test)
    
    overall_mse = mean_squared_error(y, oof_pred)
    final_test_pred = test_preds.mean(axis=1)
    
    print(f"\n{model_name} Overall OOF MSE: {overall_mse:.8f}")
    print(f"{model_name} Final test predictions (first 10 samples):")
    print(final_test_pred[:10])
    print("-" * 50)
    
    results[model_name] = {
        "fold_mse": fold_mse,
        "overall_mse": overall_mse,
        "test_pred": final_test_pred
    }


# Convert the results dictionary to a list of records
records = []
for model_name, result in results.items():
    record = {
        "Model": model_name,
        "Overall_MSE": result["overall_mse"],
        # Convert the list of fold MSEs to a comma-separated string
        "Fold_MSE": ", ".join([f"{mse:.8f}" for mse in result["fold_mse"]])
    }
    records.append(record)

# Create a DataFrame from the records
results_df = pd.DataFrame(records)

# Save the DataFrame to a CSV file
results_df.to_csv("model_results.csv", index=False)
print("Results saved to model_results.csv")

