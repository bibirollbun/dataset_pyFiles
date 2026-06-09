import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')


# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# === Encode 'Sex' ===
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])
train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')


# === Feature Engineering ===
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    for col in numerical_features:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(df[col])


# === Cross Terms ===
def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df

cross_features = numerical_features + ['BMI']
train = add_cross_terms(train, cross_features)
test = add_cross_terms(test, cross_features)


# === Prepare Data ===
X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])
X_test = test.drop(columns=['id'])


# === Model Setup ===
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

models = {
    'CatBoost': CatBoostRegressor(verbose=0, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
    'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9,
                            n_estimators=2000, learning_rate=0.02, gamma=0.01,
                            max_delta_step=2, early_stopping_rounds=100,
                            eval_metric='rmse', enable_categorical=True, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10,
                              colsample_bytree=0.7, subsample=0.9, random_state=42, verbose=-1)
}

results = {name: {'pred': np.zeros(len(test)), 'rmsle': []} for name in models}


# === Training Loop ===
for name, model in models.items():
    print(f"\nðŸš€ Training {name}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_val, y_val))
        else:
            model.fit(x_train, y_train)

        y_pred_val = model.predict(x_val)
        y_pred_test = model.predict(X_test)
        results[name]['pred'] += y_pred_test / FOLDS
        score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)))
        results[name]['rmsle'].append(score)
        print(f"Fold {fold + 1} RMSLE: {score:.5f}")


# === Blend Predictions ===
blend_preds = (
    0.4 * np.expm1(results['XGBoost']['pred']) +
    0.3 * np.expm1(results['CatBoost']['pred']) +
    0.3 * np.expm1(results['LightGBM']['pred'])
)

submission['Calories'] = np.clip(blend_preds, 1, 314)
submission.to_csv("/kaggle/working/submission_blend_bmi_squared.csv", index=False)



# === Print Summary ===
for name in models:
    scores = results[name]['rmsle']
    print(f"{name} Mean RMSLE: {np.mean(scores):.5f} Â± {np.std(scores):.5f}")

print("\nâœ… submission_blend_bmi_squared.csv is saved.")

