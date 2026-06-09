import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/mlx-2-0-regression/train.csv")
test = pd.read_csv("/kaggle/input/mlx-2-0-regression/test.csv")
sample = pd.read_csv("/kaggle/input/mlx-2-0-regression/sample_submission.csv")

print(df.shape)
print(test.shape)
df.head()


df.describe()


df.info()


duplicates = df[df.drop(columns=['id']).duplicated()]
duplicates


df = df.drop_duplicates(subset=df.columns.difference(['id']))
df


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Data Heatmap")
plt.show()


numerical_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
categorical_cols = list(df.select_dtypes(include=["object"]).columns)
numerical_cols.remove("target")
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

test[numerical_cols] = test[numerical_cols].fillna(df[numerical_cols].mean())
test[categorical_cols] = test[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


# imputer = SimpleImputer(strategy='mean')  # or 'median'

# X = df.drop('target', axis=1)
# y = df['target']
# X_test = test.copy()

# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)



plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Data Heatmap")
plt.show()


categorical_cols


df[categorical_cols]


# Drop ID columns that are not predictive
useless_cols = ['track_identifier', 'creator_collective', 'composition_label_0', 'composition_label_1', 'composition_label_2']
df.drop(useless_cols, axis=1, inplace=True)
test.drop(useless_cols, axis=1, inplace=True)

# Convert date
df['publication_timestamp'] = pd.to_datetime(df['publication_timestamp'])
test['publication_timestamp'] = pd.to_datetime(test['publication_timestamp'])

# Add year, month, day, etc.
for dataset in [df, test]:
    dataset['year'] = dataset['publication_timestamp'].dt.year
    dataset['month'] = dataset['publication_timestamp'].dt.month
    dataset['day'] = dataset['publication_timestamp'].dt.day
    dataset['dayofweek'] = dataset['publication_timestamp'].dt.dayofweek
    dataset.drop(['publication_timestamp'], axis=1, inplace=True)

# # Encode categorical columns
# cat_cols = ['weekday_of_release', 'season_of_release', 'lunar_phase']
# le = LabelEncoder()
# for col in cat_cols:
#     df[col] = le.fit_transform(df[col].astype(str))
#     test[col] = le.transform(test[col].astype(str))



from sklearn.preprocessing import LabelEncoder

# Categorical columns to encode
cat_cols = ['season_of_release', 'lunar_phase', 'weekday_of_release']

# Initialize a dictionary to store encoders for each column
encoders = {}

# Apply Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    
    # Fit on combined data to cover all categories
    combined = pd.concat([df[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    
    # Transform train and test
    df[col] = le.transform(df[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    
    # Save the encoder if you need it later (e.g., for inverse transform)
    encoders[col] = le


df['weekday_of_release'].unique()


# Create a function to apply to both train and test
def feature_engineering(data):
    # Averages across 3 versions
    data['duration_avg'] = data[['duration_ms_0', 'duration_ms_1', 'duration_ms_2']].mean(axis=1)
    data['intensity_avg'] = data[['intensity_index_0', 'intensity_index_1', 'intensity_index_2']].mean(axis=1)
    data['organic_texture_avg'] = data[['organic_texture_0', 'organic_texture_1', 'organic_texture_2']].mean(axis=1)
    data['vocal_presence_avg'] = data[['vocal_presence_0', 'vocal_presence_1', 'vocal_presence_2']].mean(axis=1)
    data['groove_efficiency_avg'] = data[['groove_efficiency_0', 'groove_efficiency_1', 'groove_efficiency_2']].mean(axis=1)
    data['emotional_charge_avg'] = data[['emotional_charge_0', 'emotional_charge_1', 'emotional_charge_2']].mean(axis=1)
    data['emotional_resonance_avg'] = data[['emotional_resonance_0', 'emotional_resonance_1', 'emotional_resonance_2']].mean(axis=1)
    data['performance_authenticity_avg'] = data[['performance_authenticity_0', 'performance_authenticity_1', 'performance_authenticity_2']].mean(axis=1)
    data['instrumental_density_avg'] = data[['instrumental_density_0', 'instrumental_density_1', 'instrumental_density_2']].mean(axis=1)
    data['beat_frequency_avg'] = data[['beat_frequency_0', 'beat_frequency_1', 'beat_frequency_2']].mean(axis=1)
    data['rhythmic_cohesion_avg'] = data[['rhythmic_cohesion_0', 'rhythmic_cohesion_1', 'rhythmic_cohesion_2']].mean(axis=1)
    data['organic_immersion_avg'] = data[['organic_immersion_0', 'organic_immersion_1', 'organic_immersion_2']].mean(axis=1)

    # Variance-based features
    data['duration_var'] = data[['duration_ms_0', 'duration_ms_1', 'duration_ms_2']].var(axis=1)
    data['intensity_var'] = data[['intensity_index_0', 'intensity_index_1', 'intensity_index_2']].var(axis=1)

    # Interaction features (example)
    data['duration_x_intensity'] = data['duration_avg'] * data['intensity_avg']
    data['texture_x_groove'] = data['organic_texture_avg'] * data['groove_efficiency_avg']
    
    return data

# Apply to both datasets
df = feature_engineering(df)
test = feature_engineering(test)



df = df.fillna(-1)


test = test.fillna(-1)


plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix")
plt.show()



features = df.drop(['target'], axis=1)

mi_scores = mutual_info_regression(features, df['target'])
mi_scores = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
print(mi_scores)

plt.figure(figsize=(10, 6))
mi_scores.plot(kind='bar')
plt.title("Mutual Information Scores")
plt.show()



# ---------------------- Plot Functions ----------------------
def plot_prediction_error(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title(f"{title} - Prediction Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residuals")
    plt.title(f"{title} - Residual Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
        plt.title(f"{title} - Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.show()


from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error, mean_squared_log_error
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

# Feature selection
high_mi_features = mi_scores[mi_scores > 1.0].index.tolist()
X = df[high_mi_features]
y = df['target']
X_test_final = test[high_mi_features]

def evaluate_model(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)  # To avoid issues with log metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return rmse, mae, r2, mape, rmsle

def get_model_predictions(X, y, df_test, model_func, needs_scaling=False):
    name = next((k for k, v in models.items() if v[0] == model_func), "UnknownModel")
    test_preds = np.zeros(len(df_test))
    val_preds = np.zeros(len(X))
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_val, y_val = X.iloc[valid_ind], y.iloc[valid_ind]

        if needs_scaling:
            model = make_pipeline(StandardScaler(), model_func())
        else:
            model = model_func()

        # Fit model with early stopping if supported
        if model_func == lgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(-1)]
            )
        elif model_func == xgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0
            )
        elif model_func == catboost_model:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val)
            )
        else:
            model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        y_pred_val = np.round(y_pred_val)

        rmse, mae, r2, mape, rmsle = evaluate_model(y_val, y_pred_val)
        print("-" * 60)
        print(f"{model_func.__name__} | Fold {fold + 1}")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        print(f"MAPE: {mape:.4f} | RMSLE: {rmsle:.4f}")
        print("-" * 60)

        val_preds[valid_ind] = y_pred_val
        test_preds += model.predict(df_test) / cv.n_splits

        gc.collect()

    test_preds = np.round(test_preds)
    plot_prediction_error(y, val_preds, name)
    plot_residuals(y, val_preds, name)
    plot_feature_importance(model, X.columns, name)
    
    return val_preds, test_preds



# === Model Factory Functions ===
def rf_model():
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

def et_model():
    return ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)

def lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        
    )

def xgb_model():
    return xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42
    )

def catboost_model():
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        # task_type='GPU',
        eval_metric='RMSE',
        verbose=0
    )

def ridge_model():
    return Ridge(alpha=1.0)

def lasso_model():
    return Lasso(alpha=0.1)

def svr_model():
    return SVR(C=1.0, epsilon=0.1)

# === Model Dictionary ===
models = {
    "RandomForest": (rf_model, False),
    "ExtraTrees": (et_model, False),
    "LightGBM": (lgb_model, False),
    "XGBoost": (xgb_model, False),
    "CatBoost": (catboost_model, False),
    "Ridge": (ridge_model, True),
    "Lasso": (lasso_model, True),
    "SVR": (svr_model, True)
}


print("1. XGBRegressor")
xgb_val_preds, xgb_test_preds = get_model_predictions(X, y, X_test_final, xgb_model)


print("2. LGBMRegressor")
lgb_val_preds, lgb_test_preds = get_model_predictions(X, y, X_test_final, lgb_model)


print("3. CatBoostRegressor")
cat_val_preds, cat_test_preds = get_model_predictions(X, y, X_test_final, catboost_model)



print("4. RandomForestRegressor")
rf_val_preds, rf_test_preds = get_model_predictions(X, y, X_test_final, rf_model)



print("5. ExtraTreesRegressor")
et_val_preds, et_test_preds = get_model_predictions(X, y, X_test_final, et_model)


print("6. Ridge Regression (scaled)")
ridge_val_preds, ridge_test_preds = get_model_predictions(X, y, X_test_final, ridge_model, needs_scaling=True)


print("7. Lasso Regression (scaled)")
lasso_val_preds, lasso_test_preds = get_model_predictions(X, y, X_test_final, lasso_model, needs_scaling=True)


print("8. Support Vector Regressor (SVR, scaled)")
svr_val_preds, svr_test_preds = get_model_predictions(X, y, X_test_final, svr_model, needs_scaling=True)


val_preds_df = pd.DataFrame({
    'xgb': xgb_val_preds,
    'lgb': lgb_val_preds,
    'catb': cat_val_preds,
    'rf': rf_val_preds,
    'et': et_val_preds,
    'ridge': ridge_val_preds,
    'lasso': lasso_val_preds,
    'svr': svr_val_preds,
})

test_preds_df = pd.DataFrame({
    'xgb': xgb_test_preds,
    'lgb': lgb_test_preds,
    'catb': cat_test_preds,
    'rf': rf_test_preds,
    'et': et_test_preds,
    'ridge': ridge_test_preds,
    'lasso': lasso_test_preds,
    'svr': svr_test_preds,
})


from lightgbm import LGBMRegressor

meta_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

meta_model.fit(val_preds_df, y)

meta_train_preds = meta_model.predict(val_preds_df)
meta_test_preds = meta_model.predict(test_preds_df)

residuals = y - meta_train_preds

rmse, mae, r2, mape, rmsle = evaluate_model(y, meta_train_preds)

print("Stacking Meta Model Evaluation:")
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
print(f"MAPE: {mape:.4f} | RMSLE: {rmsle:.4f}")

# Plot for stacking model
plot_prediction_error(y, meta_train_preds, "Stacking")
plot_residuals(y, meta_train_preds, "Stacking")

# Feature importance for stacking model
plt.figure(figsize=(8, 6))
sns.barplot(x=meta_model.feature_importances_, y=val_preds_df.columns)
plt.title('Stacking Model Feature Importance')
plt.xlabel('Importance')
plt.savefig('stacking_feature_importance.png')
plt.show()


# # Choose the best model (e.g., LGBM)
# final_model = RandomForestRegressor(n_estimators=100, random_state=42)
# final_model.fit(X_train_mi, y_train)



# predictions = final_model.predict(X_test_mi)

# submission = pd.read_csv("/kaggle/input/mlx-2-0-regression/sample_submission.csv")
# submission['target'] = predictions
# submission.to_csv("RF_MIScores.csv", index=False)


sample['target'] = xgb_test_preds
sample.to_csv('xgb_submission.csv', index=False)


sample['target'] = lgb_test_preds
sample.to_csv('lgb_submission.csv', index=False)


sample['target'] = cat_test_preds
sample.to_csv('cat_submission.csv', index=False)


sample['target'] = rf_test_preds
sample.to_csv('rf_submission.csv', index=False)


sample['target'] = et_test_preds
sample.to_csv('et_submission.csv', index=False)


sample['target'] = lasso_test_preds
sample.to_csv('lasso_submission.csv', index=False)


sample['target'] = svr_test_preds
sample.to_csv('svr_submission.csv', index=False)


sample['target'] = meta_test_preds
sample.to_csv('meta_submission.csv', index=False)


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("kaggle_api")

import json
import os

# Save the API key
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": "uom220315r", "key": api_key}, f)

os.chmod("/root/.kaggle/kaggle.json", 600)


# !kaggle competitions submit -c mlx-2-0-regression -f xgb_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f lgb_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f cat_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f rf_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f et_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f lasso_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f svr_submission.csv -m "Message"


# !kaggle competitions submit -c mlx-2-0-regression -f meta_submission.csv -m "Message"

