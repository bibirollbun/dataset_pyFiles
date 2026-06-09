import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load data
def load_data(train_path, test_path=None):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path else None
    return train, test

# Define RMSLE metric
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

# Build preprocessing pipeline
numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
categorical_features = ['Sex']

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define base models
def get_models():
    return {
        'ridge': Ridge(alpha=1.0, random_state=42),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'et': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
        'lgb': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

# Training & blending pipeline
def run_cv_and_blend(train_df, test_df=None, n_splits=5):
    features = categorical_features + numeric_features
    target = 'Calories'

    #train_df = train_df.sample(10000, random_state=43)

    X = train_df[features]
    y = train_df[target]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_models()

    # DataFrames to store predictions
    oof_preds = pd.DataFrame(index=train_df.index)
    test_preds = pd.DataFrame(index=test_df.index) if test_df is not None else None

    # Loop through each model
    for name, model in models.items():
        print(f"Training model: {name}")
        oof = np.zeros(len(train_df))
        test_fold_pred = np.zeros(len(test_df)) if test_df is not None else None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            pipe.fit(X_train, y_train)

            val_pred = pipe.predict(X_val)
            oof[val_idx] = val_pred

            fold_rmsle = rmsle(y_val, val_pred)
            print(f"  Fold {fold+1}/{n_splits} RMSLE: {fold_rmsle:.5f}")

            if test_df is not None:
                test_pred = pipe.predict(test_df[features])
                test_fold_pred += test_pred

        # Store out-of-fold predictions
        oof_preds[name] = oof
        overall_rmsle = rmsle(y, oof)
        print(f"  {name} OOF RMSLE: {overall_rmsle:.5f}\n")

        # Store averaged test predictions
        if test_df is not None:
            test_preds[name] = test_fold_pred / n_splits

    # Blend by simple average
    blended_oof = oof_preds.mean(axis=1)
    print(f"Blended OOF RMSLE: {rmsle(y, blended_oof):.5f}")

    if test_df is not None:
        blended_test = test_preds.mean(axis=1)
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Calories': blended_test
        })
        submission ["Calories"] = submission["Calories"].clip(0)
        submission.to_csv('submission.csv', index=False)
        print("Submission file 'submission.csv' created.")

    return oof_preds, test_preds if test_df is not None else None

# Example usage
if __name__ == '__main__':
    train_data, test_data = load_data(
        '/kaggle/input/playground-series-s5e5/train.csv',
        '/kaggle/input/playground-series-s5e5/test.csv'
    )  # adjust paths
    run_cv_and_blend(train_data, test_data)





