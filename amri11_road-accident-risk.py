import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (KFold, RandomizedSearchCV, 
StratifiedKFold, GridSearchCV, train_test_split)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import (mean_squared_error, make_scorer,
mean_absolute_error, r2_score, mean_absolute_percentage_error)
import math
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


%%time
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


test.head()


train.info()


test.info()


train.duplicated().sum()


test.duplicated().sum()


train.describe()


train.describe(include = 'object')


train_numeric = train.select_dtypes(include=['number'])

# Calculate correlation
corrtrain_matrix = train_numeric.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corrtrain_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Correlation")
plt.show()


# --- Target Distribution: Accident Risk ---
plt.figure(figsize=(8,5))
sns.histplot(train["accident_risk"], bins=40, kde=True, color="skyblue")
plt.title("Distribution of Accident Risk (Target)", fontsize=14)
plt.xlabel("Accident Risk")
plt.ylabel("Count")
plt.show()


exclude_cols = ["accident_risk", "id"]

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]   # handle single row
    sns.histplot(train[col], bins=40, kde=True, color="orange", ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

for j in range(len(num_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# --- Numerical Features vs Target (Scatter Plot) ---
exclude_cols = ["accident_risk", "id"]
target_col = "accident_risk"

num_cols = train.select_dtypes(include=["int64", "float64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]   # handle single row
    
    sns.scatterplot(x=train[col], y=train[target_col], alpha=0.5, color="orange", ax=ax)
    ax.set_title(f"{col} vs {target_col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)

for j in range(len(num_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


# --- Categorical Features vs Numeric Target ---
exclude_cols = ["id"]   # contoh exclude
target_col = "accident_risk"  # target numerik

cat_cols = train.select_dtypes(include=["object"]).columns
cat_cols = [c for c in cat_cols if c not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

for i, col in enumerate(cat_cols):
    r, c = divmod(i, n_cols)
    ax = axes[r, c] if n_rows > 1 else axes[c]
    
    sns.boxplot(x=train[col], y=train[target_col], ax=ax)
    ax.set_title(f"{target_col} by {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel(target_col)
    ax.tick_params(axis="x", rotation=45)

# hapus subplot kosong
for j in range(len(cat_cols), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r, c] if n_rows > 1 else axes[c])

plt.tight_layout()
plt.show()


train.head()


def feature_engineering(df: pd.DataFrame, scaler=None, fit_scaler=True) -> pd.DataFrame:
    # --- Mapping categorical ke numeric ---
    road_type_risk_mapping = {'highway': 3, 'rural': 2, 'urban': 1}
    lighting_mapping = {'daylight': 0, 'dim': 1, 'night': 2}
    weather_risk_mapping = {'clear': 0, 'rainy': 1, 'foggy': 2}
    
    df['road_type_encoded'] = df['road_type'].map(road_type_risk_mapping)
    df['lighting_encoded'] = df['lighting'].map(lighting_mapping)
    df['weather_encoded'] = df['weather'].map(weather_risk_mapping)

    # --- Cyclical time encoding ---
    def encode_time_cyclical(time_str):
        if isinstance(time_str, str) and ':' in time_str:
            hour = int(time_str.split(':')[0])
        else:
            time_mapping = {'morning': 8, 'afternoon': 14, 'evening': 18, 
                            'morning_rush': 8, 'evening_rush': 18}
            hour = time_mapping.get(time_str, 12)
        return np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)

    df['time_sin'], df['time_cos'] = zip(*df['time_of_day'].apply(encode_time_cyclical))

    # --- Risk features ---
    df['risk_combination'] = (df['weather_encoded'] + 
                              df['lighting_encoded'] + (df['curvature'] > 0.5).astype(int))
    df['speed_curve_risk'] = df['speed_limit'] * df['curvature']
    df['visibility_risk'] = df['weather_encoded'] * df['lighting_encoded']
    df['rush_hour_risk'] = (
        ((df['time_of_day'] == 'morning_rush') | (df['time_of_day'] == 'evening_rush')) &
        (df['weather_encoded'] > 0)).astype(int)
    df['night_driving_risk'] = ((df['lighting_encoded'] >= 1) & (df['weather_encoded'] > 0)).astype(int)

    # --- Transformasi numerik (pakai nilai asli, amanin negatif) ---
    df['curvature_log'] = np.log1p(df['curvature'].clip(lower=0))
    df['curvature_sqrt'] = np.sqrt(df['curvature'].clip(lower=0))

    # --- Standardisasi numerical features ---
    numerical_features = ['curvature', 'num_lanes', 'speed_limit']
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
    else:
        df[numerical_features] = scaler.transform(df[numerical_features])

    # --- Groupby statistics ---
    road_type_accidents = (
        df.groupby('road_type')['num_reported_accidents']
          .agg(['mean', 'std'])
          .rename(columns={'mean': 'road_type_accident_mean',
                           'std': 'road_type_accident_std'}))
    df = df.merge(road_type_accidents, on='road_type', how='left')

    # --- Composite features ---
    df['accident_rate_per_lane'] = df['num_reported_accidents'] / df['num_lanes']
    df['accident_intensity'] = df['num_reported_accidents'] * df['curvature']
    df['holiday_school_risk'] = (df['holiday'] | df['school_season']).astype(int)

    # --- Final risk score ---
    df['composite_risk_score'] = (
        df['road_type_encoded'] * 0.2 +
        df['weather_encoded'] * 0.25 +
        df['lighting_encoded'] * 0.2 +
        df['curvature'] * 0.15 +
        (df['speed_limit'] / 100) * 0.1 +
        df['risk_combination'] * 0.1)

    return df, scaler



train_finish, scaler = feature_engineering(train, fit_scaler=True)
print(train_finish.head())

test_finish, _ = feature_engineering(test, scaler=scaler, fit_scaler=False)
print(test_finish.head())


# Features akhir yang akan digunakan
final_features = [
    # Numerical
    'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
    
    # Encoded categorical
    'road_type_encoded', 'lighting_encoded', 'weather_encoded',
    
    # Cyclical time
    'time_sin', 'time_cos',
    
    # Interaction features
    'risk_combination', 'speed_curve_risk', 'visibility_risk',
    'rush_hour_risk', 'night_driving_risk',
    
    # Historical patterns
    'road_type_accident_mean', 'road_type_accident_std',
    'accident_rate_per_lane', 'accident_intensity',
    
    # Temporal
    'holiday_school_risk',
    
    # Composite
    'composite_risk_score',
    
    # Boolean (tetap sebagai 0/1)
    'road_signs_present', 'public_road', 'holiday', 'school_season']


test_feat = test_finish[final_features]


def build_and_evaluate_regression(df, final_features, test_size=0.2, random_state=101):
    # --- Data ---
    X = df[final_features]
    y = df["accident_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # --- Models ---
    models = {
        "RandomForest": RandomForestRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(random_state=random_state, eval_metric="rmse")
    }

    # --- Hyperparameter grids ---
    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 300],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False]
        },
        "XGBoost": {
            "n_estimators": [100, 300],
            "max_depth": [5, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1, 2]
        }
    }

    # --- Cross-validation setup ---
    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    # --- Results storage ---
    results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")

        grid = GridSearchCV(
            model, param_grids[name],
            cv=kf, scoring="neg_root_mean_squared_error",
            n_jobs=-1, verbose=1)
        
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # --- Evaluation metrics ---
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Best Params for {name}:", grid.best_params_)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RÂ²: {r2:.4f}")

        # --- Plot actual vs predicted ---
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([0,1],[0,1], color="red", linestyle="--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted ({name})")
        plt.show()

        # --- Residual plot ---
        residuals = y_test - y_pred
        plt.figure(figsize=(6,5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title(f"Residual Distribution ({name})")
        plt.show()

        results[name] = {
            "model": best_model,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_params": grid.best_params_}

        # --- Feature importance ---
        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance})
            feat_imp = feat_imp.sort_values("Importance", ascending=False)

            plt.figure(figsize=(8,5))
            sns.barplot(x="Importance", y="Feature", data=feat_imp.head(10))
            plt.title(f"Top 10 Feature Importance ({name})")
            plt.show()

        # --- SHAP dengan error handling ---
        try:
            print(f"\nCalculating SHAP values for {name}...")
            
            # Untuk RandomForest, gunakan TreeExplainer khusus
            if name == "RandomForest":
                explainer = shap.TreeExplainer(best_model)
            else:
                # Untuk XGBoost, gunakan sample data yang lebih kecil
                X_train_sample = X_train[:1000]  # sample untuk menghindari memory issues
                explainer = shap.TreeExplainer(best_model, X_train_sample)
            
            # Calculate SHAP values dengan sample yang lebih kecil
            X_test_sample = X_test[:500]
            shap_values = explainer.shap_values(X_test_sample)
            
            print(f"SHAP Summary Plot ({name}):")
            shap.summary_plot(shap_values, X_test_sample, feature_names=X.columns.tolist())
            
        except Exception as e:
            print(f"SHAP calculation failed for {name}: {str(e)}")
            print("Continuing without SHAP analysis...")
    return results


# Run evaluation
results = build_and_evaluate_regression(train_finish, final_features)

# Comparasion and Chose Best Model with otomaticaly
def select_best_model(results, metric='rmse', lower_is_better=True):
    """
    Choose the best model based on specific metrics
    """
    best_model_name = None
    best_score = float('inf') if lower_is_better else float('-inf')
    
    for model_name, metrics in results.items():
        score = metrics[metric]
        
        if (lower_is_better and score < best_score) or (not lower_is_better and score > best_score):
            best_score = score
            best_model_name = model_name
    
    return best_model_name, best_score

# Select the best model based on RMSE (lower is better)
best_model_name, best_rmse = select_best_model(results, metric='rmse', lower_is_better=True)
print(f"ğŸ�¯ BEST MODEL: {best_model_name} dengan RMSE: {best_rmse:.4f}")

# Take the best model
best_model = results[best_model_name]["model"]
print(f"âœ… Use model: {best_model_name}")

# Show detailed comparison of all models
print("\nğŸ“Š COMPARISON ALL MODELS:")
print("=" * 50)
for model_name, metrics in results.items():
    is_best = "ğŸŒŸ" if model_name == best_model_name else "  "
    print(f"{is_best} {model_name:15} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | RÂ²: {metrics['r2']:.4f}")


df_sub = sample.drop('accident_risk', axis=1)
df_sub.head


df_sub['accident_risk'] = best_model.predict(test_feat)
df_sub.to_csv('submission.csv', index=False)
df_sub.value_counts()
df_sub.head()

