# Datatool kit
import numpy as np
import pandas as pd
from scipy import stats

# visualization toolkit
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# modeling
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

# HyperParameter tuning
from sklearn.model_selection import RandomizedSearchCV
import optuna

# metric
from sklearn.metrics import mean_squared_error, r2_score

# sup-press warnings
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

print("imported all Successfully")


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test =  pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("setup completed")


train.info()


train.describe()


for col in train.select_dtypes("object").columns:
    print(col , "\n", train[col].unique())


target_col = 'accident_risk'
# numerical and categorical columns
num_cols = train.select_dtypes(include=['number']).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

# Remove ID and target columns if present
for col in ['id', target_col]:
    if col in num_cols:
        num_cols.remove(col)

print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")



fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes = axes.flatten()

for i, col in enumerate(num_cols[:4]):  # limit to first 4 columns
    sns.histplot(train[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes = axes.flatten()

for i, col in enumerate(cat_cols[:4]):  # limit to first 4 columns
    result = train[col].value_counts().reset_index()
    sns.barplot(x=result.iloc[:, 0], y=result.iloc[:, 1], ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


numerical_cols_corr = num_cols.copy()
numerical_cols_corr.append('accident_risk')

print(num_cols)

if len(numerical_cols_corr) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = train[numerical_cols_corr].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=1)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()




for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train, x=train[col], y='accident_risk')
    plt.title(f'Accident Risk by {col}')
plt.show()


label_encoders = {}
categorical_cols_encoded = []

for col in cat_cols:
    if col in train.columns and col in test.columns:
        le = LabelEncoder()
        
        # Fit on combined unique values from train and test
        combined_values = pd.concat([train[col], test[col]]).unique()
        le.fit(combined_values)
        
        # Transform both datasets
        train[col + '_encoded'] = le.transform(train[col])
        test[col + '_encoded'] = le.transform(test[col])
        categorical_cols_encoded.append(col+'_encoded')
        label_encoders[col] = le


def create_advanced_features(train, test):
    """
    Generate advanced interaction and transformation features 
    to enhance model performance.
    """
    import pandas as pd, numpy as np

    train, test = train.copy(), test.copy()
    new_feats = []

    print("Creating advanced features...")

    # --- Helper to add features safely ---
    def add_feature(name, func):
        train[name] = func(train)
        test[name] = func(test)
        new_feats.append(name)

    # --- 1. Key interactions ---
    add_feature('lighting_speed_interaction', lambda df: df['lighting_encoded'] * df['speed_limit'])
    add_feature('weather_curvature_interaction', lambda df: df['weather_encoded'] * df['curvature'])
    add_feature('lighting_weather_interaction', lambda df: df['lighting_encoded'] * df['weather_encoded'])
    add_feature('speed_curvature_interaction', lambda df: df['speed_limit'] * df['curvature'])

    # --- 2. Binned features ---
    speed_bins = [0, 30, 50, 70, 90, np.inf]
    curvature_bins = [-np.inf, 0.1, 0.3, 0.5, np.inf]

    add_feature('speed_limit_binned', lambda df: pd.cut(df['speed_limit'], bins=speed_bins, labels=False))
    add_feature('curvature_binned', lambda df: pd.cut(df['curvature'], bins=curvature_bins, labels=False))

    # --- 3. Log transforms ---
    add_feature('log_accidents', lambda df: np.log1p(df['num_reported_accidents']))
    add_feature('log_curvature', lambda df: np.log1p(df['curvature']))

    # --- 4. Complex interactions ---
    add_feature('risk_score', lambda df: 0.5 * df['lighting_encoded'] + 
                                         0.3 * (df['speed_limit'] / 100) + 
                                         0.2 * df['weather_encoded'])
    add_feature('top3_interaction', lambda df: df['lighting_encoded'] * df['speed_limit'] * df['curvature'])

    # --- 5. Ratios ---
    add_feature('accidents_per_lane', lambda df: df['num_reported_accidents'] / (df['num_lanes'] + 1e-8))
    add_feature('speed_curvature_ratio', lambda df: df['speed_limit'] / (df['curvature'] + 1e-8))

    print(f" Created {len(new_feats)} new features:")
    print(", ".join(new_feats))

    return train, test, new_feats


train, test, new_feats = create_advanced_features(train, test)


train.info()



X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']
test_ids = test['id'].copy()
X_test = test


from IPython.display import display as d
print("=====Train_Data=====")
d(X.head(3))
print('\n')
print("=====Target=====")
d(y.head(3))
print('\n')
print("=====Test_ID=====")
d(test_ids)
print('\n')
print("=====Test_data=====")
d(X_test.head(3))


# splitting into train, test, val

X_train, x_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)
print("done")


X_train_enc = X_train.copy()

cat_cols = X_train_enc.select_dtypes(include=['object', 'bool']).columns

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    le = LabelEncoder()
    X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str))


kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "xgboost": xgb.XGBRegressor(random_state=42, verbosity=0),
    "catboost": cb.CatBoostRegressor(random_state=42, verbose=0),
    "lgb": lgb.LGBMRegressor(random_state=42, verbosity=0)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_enc, y_train, 
                           cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)  # Convert to RMSE
    print(f"{name}: RMSE {rmse_scores.mean():.4f} Â± {rmse_scores.std():.4f}")


# def objective_xgb(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
#         'random_state': 42
#     }

#     model = xgb.XGBRegressor(**params, eval_metric='rmse')

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     rmse_scores = []
    
#     for train_idx, val_idx in kf.split(X_train_enc):
#         #  Correct Pandas row selection
#         X_t, X_v = X_train_enc.iloc[train_idx], X_train_enc.iloc[val_idx]
#         y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
#         model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
#         preds = model.predict(X_v)
#         rmse = mean_squared_error(y_v, preds, squared=False)
#         rmse_scores.append(rmse)
    
#     return np.mean(rmse_scores)


# print("Best CatBoost Params:", study_cat.best_params)
# print("Lowest RMSE:", study_cat.best_value)


xgb_params = {
    'n_estimators': 996,
    'max_depth': 9,
    'learning_rate': 0.04269262619258986,
    'subsample': 0.5738482313824345,
    'colsample_bytree': 0.7006170358861727,
    'gamma': 0.004459148897939975,
    'reg_alpha': 8.558050657535242,
    'reg_lambda': 9.983448401108683,
    'tree_method': 'gpu_hist',      # GPU
    'predictor': 'gpu_predictor',
    'random_state': 42
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train_enc, y_train)


# catboost model
cat_model = cb.CatBoostRegressor(
    task_type='GPU',
    verbose=0,
    random_seed=42
)
cat_model.fit(X_train_enc, y_train)


X_test_enc = X_test.copy()

cat_cols = X_test_enc.select_dtypes(include=['object', 'bool']).columns

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    le = LabelEncoder()
    X_test_enc[col] = le.fit_transform(X_test_enc[col].astype(str))




X_test_enc = X_test_enc.drop("id", axis=1)


X_test_enc.head(5)


X_train_enc.head(3)


xgb_preds = xgb_model.predict(X_test_enc)
cat_preds = cat_model.predict(X_test_enc)


# Weighted blend
blend_weight = 0.5
final_preds = blend_weight * xgb_preds + (1 - blend_weight) * cat_preds

# Clip to valid range [0, 1]
final_preds = np.clip(final_preds, 0, 1)


submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': np.clip(cat_preds, 0, 1)
})
submission.to_csv('submission.csv', index=False)

print("submission made")

