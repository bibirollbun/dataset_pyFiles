import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


def data_overview(df, name="DataFrame", show_visuals=False):
    print(f"ğŸ§¾ OVERVIEW: {name}")
    print("=" * 70)
    
    # Shape
    print(f"ğŸ“� Shape: {df.shape}")
    print("=" * 70)
    
    # Data Types and Info
    print("ğŸ“‹ Data Types and Memory Usage:")
    print(df.info())
    print("=" * 70)

    # Missing Values
    nulls = df.isnull().sum()
    null_percent = (nulls / len(df)) * 100
    null_df = pd.DataFrame({"Missing Values": nulls, "% Missing": null_percent})
    print("â�“ Missing Values:")
    print(null_df[null_df["Missing Values"] > 0].sort_values(by="% Missing", ascending=False))
    print("=" * 70)

    # Unique Value Counts
    print("ğŸ”¢ Unique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    print("=" * 70)

    # Descriptive Stats
    print("ğŸ“Š Statistical Summary (Numerical):")
    display(df.describe().T)
    print("=" * 70)

    # Sample Records
    print("ğŸ”� Sample Rows:")
    display(df.head(5))
    print("=" * 70)

    # Cardinality Check (useful for modeling)
    print("ğŸ§  Column Cardinality:")
    cardinality = df.apply(lambda x: x.nunique()).sort_values()
    display(cardinality)

    # Visuals
    if show_visuals:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols].hist(bins=20, figsize=(15, 10))
        plt.suptitle(f"ğŸ“ˆ Histograms of Numeric Features - {name}", fontsize=16)
        plt.tight_layout()
        plt.show()


data_overview(train, name="Train Data", show_visuals=True)


def preprocess_train_and_test(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()

    # Boolean conversion
    bool_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in bool_cols:
        for df in [train, test]:
            df[col] = df[col].fillna("No").replace({"Yes": 1, "No": 0}).astype(int)

    # Impute numeric columns with train median
    for col in train.columns:
        if train[col].dtype in ['float64', 'int64']:
            median = train[col].median()
            train[col].fillna(median, inplace=True)
            test[col].fillna(median, inplace=True)

    # Drop ID column if it exists
    if 'id' in train.columns:
        train.drop(columns='id', inplace=True)
    if 'id' in test.columns:
        test.drop(columns='id', inplace=True)

    # Drop target if exists (youâ€™ll split it later)
    if 'Personality' in train.columns:
        train.drop(columns='Personality', inplace=True)

    # Scale
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

    # Feature engineering
    for df in [train_scaled, test_scaled]:
        df['Social_ratio'] = df['Social_event_attendance'] / (df['Friends_circle_size'] + 1)
        df['Outside_diff'] = df['Going_outside'] - df['Drained_after_socializing']

    return train_scaled, test_scaled


X_train_df, X_test_df = preprocess_train_and_test(train, test)
y = train['Personality'].map({'Extrovert': 0, 'Introvert': 1})


def evaluate_model_cv(model, X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, preds)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        accuracies.append(acc)

    print(f"\nâœ… Average Accuracy: {np.mean(accuracies):.4f}")
    return model


xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
cat = CatBoostClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, verbose=0, random_state=42)

print("ğŸ”· XGBoost:")
xgb_model = evaluate_model_cv(xgb, X_train_df, y)

print("\nğŸŸ¢ LightGBM:")
lgb_model = evaluate_model_cv(lgb, X_train_df, y)

print("\nğŸŸ¡ CatBoost:")
cat_model = evaluate_model_cv(cat, X_train_df, y)


final_preds = cat_model.predict(X_test_df)

# Create submission
submission = pd.DataFrame({
    'id': sample_submission['id'],  # Original test IDs
    'Personality': ['Extrovert' if p == 0 else 'Introvert' for p in final_preds]
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")




