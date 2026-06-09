import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from itertools import combinations
from scipy.stats import skew
import shap

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

import warnings
warnings.filterwarnings('ignore')


USE_GPU = True

TRAIN_FILE_PATH = '/kaggle/input/playground-series-s5e6/train.csv'
ORIGINAL_FILE_PATH = '/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv'
TEST_FILE_PATH = '/kaggle/input/playground-series-s5e6/test.csv'

target_column = 'Fertilizer Name'
ID_COLUMN_TEST = 'Id'

numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_cols = ['Soil Type', 'Crop Type']

FE_CORRELATION_THRESHOLD = 0.95
FE_SKEWNESS_THRESHOLD = 0.5
FE_EPSILON = 1e-6

N_TOP_SHAP_FEATURES = 20

OPTUNA_N_TRIALS = 30

cv_folds = 5


def load_and_clean_data(train_file_path, test_file_path, original_file_path, test_id_col_name='Id'):
    print(f"Loading training data from: {train_file_path}")
    df_train = pd.read_csv(train_file_path)
    df_train.columns = df_train.columns.str.strip()
    if 'Humidity ' in df_train.columns: df_train.rename(columns={'Humidity ':'Humidity'}, inplace=True)
    if 'Id' in df_train.columns: df_train = df_train.drop(columns=['Id'])
    elif 'id' in df_train.columns: df_train = df_train.drop(columns=['id'])

    df_original = pd.read_csv(original_file_path)
    df_train = pd.concat((df_train, df_original))
    print(f"Training data loaded. Shape: {df_train.shape}")

    print(f"Loading test data from: {test_file_path}")
    df_test_raw = pd.read_csv(test_file_path)
    df_test_raw.columns = df_test_raw.columns.str.strip()
    if 'Humidity ' in df_test_raw.columns: df_test_raw.rename(columns={'Humidity ':'Humidity'}, inplace=True)
    
    test_ids = None
    if test_id_col_name in df_test_raw.columns:
        test_ids = df_test_raw[test_id_col_name].copy()
        df_test = df_test_raw.drop(columns=[test_id_col_name])
    elif test_id_col_name.lower() in df_test_raw.columns:
        test_id_col_name_actual = test_id_col_name.lower()
        test_ids = df_test_raw[test_id_col_name_actual].copy()
        df_test = df_test_raw.drop(columns=[test_id_col_name_actual])
    else:
        print(f"CRITICAL WARNING: Test ID column '{test_id_col_name}' not found. Generating mock IDs.")
        df_test = df_test_raw.copy()
        test_ids = pd.Series(range(len(df_test)), name=test_id_col_name)


    print(f"Test data loaded. Shape (features): {df_test.shape}, Test IDs: {len(test_ids) if test_ids is not None else 'N/A'}")
    return df_train, df_test, test_ids

df_train, df_test, test_ids_for_submission = load_and_clean_data(
    TRAIN_FILE_PATH, TEST_FILE_PATH, ORIGINAL_FILE_PATH, test_id_col_name=ID_COLUMN_TEST
)


df_train = pd.read_csv(TRAIN_FILE_PATH)
df_train


df_train = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
df_train


df_train.sample(5)


df_test.sample(5)


print("Training set shape:", df_train.shape)
print("Test set shape:", df_test.shape)


print("Missing values in training set:")
print(df_train.isna().sum())


print("Missing values in test set:")
print(df_test.isna().sum())


print("Statistical summary of training data:")
display(df_train.describe().T)


display(df_test.describe().T)


plt.figure(figsize=(12, 7))
sns.countplot(y=df_train[target_column], order = df_train[target_column].value_counts().index, palette='viridis')
plt.title('Distribution of Fertilizer Name', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Fertilizer Name', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


df_train[numerical_cols].hist(figsize=(14, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(y=df_train['Soil Type'], order = df_train['Soil Type'].value_counts().index, palette='Spectral')
plt.title('Distribution of Soil Type', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Soil Type', fontsize=12)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
sns.countplot(y=df_train['Crop Type'], order = df_train['Crop Type'].value_counts().index, palette='coolwarm')
plt.title('Distribution of Crop Types', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Crop Type', fontsize=12)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
correlation_matrix = df_train[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Numeric Features', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 10)) # Increased height to accommodate all fertilizer names
sns.boxplot(x=df_train['Nitrogen'], y=df_train[target_column], palette='Paired', order=sorted(df_train[target_column].unique()))
plt.title('Boxplot of Nitrogen by Fertilizer Name', fontsize=15)
plt.xlabel('Nitrogen Content', fontsize=12)
plt.ylabel('Fertilizer Name', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


soil_fertilizer_crosstab = pd.crosstab(df_train['Soil Type'], df_train[target_column])
plt.figure(figsize=(18, 8))
sns.heatmap(soil_fertilizer_crosstab, annot=True, cmap="BuPu", fmt='d', linewidths=.5)
plt.title('Heatmap of Soil Type vs Fertilizer Name', fontsize=15)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Soil Type', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
sns.boxplot(x=df_train['Potassium'], y=df_train[target_column], palette='Set2', order=sorted(df_train[target_column].unique()))
plt.title('Boxplot of Potassium by Fertilizer Name', fontsize=15)
plt.xlabel('Potassium Content', fontsize=12)
plt.ylabel('Fertilizer Name', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
sns.boxplot(x=df_train['Phosphorous'], y=df_train[target_column], palette='pastel', order=sorted(df_train[target_column].unique()))
plt.title('Boxplot of Phosphorous by Fertilizer Name', fontsize=15)
plt.xlabel('Phosphorous Content', fontsize=12)
plt.ylabel('Fertilizer Name', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()


mean_nutrients = df_train.groupby(target_column)[['Nitrogen', 'Potassium', 'Phosphorous']].mean().reset_index()
mean_nutrients_melted = mean_nutrients.melt(id_vars=target_column, var_name='Nutrient', value_name='Mean Value')

plt.figure(figsize=(18, 10))
sns.barplot(x=target_column, y='Mean Value', hue='Nutrient', data=mean_nutrients_melted, palette={'Nitrogen':'blue', 'Potassium':'green', 'Phosphorous':'red'})
plt.title('Mean Nutrient Values (N, P, K) per Fertilizer Name', fontsize=15)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Mean Nutrient Content', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Nutrient', fontsize=10)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


sample_df = df_train.sample(n=min(1000, len(df_train)), random_state=42) if len(df_train) > 1000 else df_train

plt.figure(figsize=(14, 9))
sns.scatterplot(x='Temparature', y='Humidity', hue=target_column, data=sample_df, palette='tab20', s=70, alpha=0.8)
plt.title('Scatter Plot: Temperature vs. Humidity colored by Fertilizer Name (Sampled)', fontsize=15)
plt.xlabel('Temperature (Â°C)', fontsize=12)
plt.ylabel('Humidity (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.barplot(x=target_column, y='Moisture', data=df_train, estimator=np.mean, errorbar=None, palette='crest', order=sorted(df_train[target_column].unique())) # ci=None to remove error bars for mean
plt.title('Average Soil Moisture per Fertilizer Name', fontsize=15)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Average Moisture', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 9))
sns.boxplot(x=df_train[target_column], y=df_train['Temparature'], palette='magma', order=sorted(df_train[target_column].unique()))
plt.title('Boxplot of Temperature by Fertilizer Name', fontsize=15)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Temperature (Â°C)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 9))
sns.boxplot(x=df_train[target_column], y=df_train['Humidity'], palette='viridis_r', order=sorted(df_train[target_column].unique()))
plt.title('Boxplot of Humidity by Fertilizer Name', fontsize=15)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Humidity (%)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


crop_fertilizer_crosstab = pd.crosstab(df_train['Crop Type'], df_train[target_column])

plt.figure(figsize=(20, 10))
sns.heatmap(crop_fertilizer_crosstab, annot=True, cmap="YlGnBu", fmt='d', linewidths=.5)
plt.title('Heatmap of Crop Type vs Fertilizer Name', fontsize=15)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Crop Type', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# For a cleaner pair plot, let's select a subset of features,
pairplot_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

sample_df_pairplot = df_train.sample(n=min(500, len(df_train)), random_state=42) if len(df_train) > 500 else df_train

plt.figure(figsize=(12,12))
pp = sns.pairplot(sample_df_pairplot[pairplot_features])
pp.fig.suptitle("Pair Plot of Numerical Features (Sampled Data)", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


sample_df_pairplot_soil = df_train.sample(n=min(500, len(df_train)), random_state=42) if len(df_train) > 500 else df_train
pp_soil = sns.pairplot(sample_df_pairplot_soil, vars=['Temparature', 'Humidity', 'Moisture', 'Nitrogen'], hue='Soil Type', palette='tab10')
pp_soil.fig.suptitle("Pair Plot of Selected Features by Soil Type (Sampled Data)", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


def map_at_3(y_true, y_pred_proba, k=3):
    map_score = 0.0
    y_true_arr = np.array(y_true)
    for i in range(len(y_true_arr)):
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]
        if y_true_arr[i] in top_k_preds:
            rank = np.where(top_k_preds == y_true_arr[i])[0][0] + 1
            map_score += 1.0 / rank
    return map_score / len(y_true_arr) if len(y_true_arr) > 0 else 0


def feature_engineering_transform(df_train_raw, df_test_raw, numeric_cols, categorical_cols, target_col):
    print("\nStarting Feature Engineering...")
    train_fe = df_train_raw.copy()
    test_fe = df_test_raw.copy()

    for df, name in [(train_fe, "train"), (test_fe, "test")]:
        if df is None: continue
        for col_list, default_val, unknown_val in [(numeric_cols, 0, None), (categorical_cols, None, 'Unknown')]:
            for col in col_list:
                if col not in df.columns:
                    fill_val = default_val if default_val is not None else unknown_val
                    df[col] = fill_val
    
    for col in numeric_cols:
        if col in train_fe.columns: train_fe[f'{col}_Squared'] = train_fe[col] ** 2
        if col in test_fe.columns: test_fe[f'{col}_Squared'] = test_fe[col] ** 2
    for col1, col2 in combinations(numeric_cols, 2):
        if col1 in train_fe.columns and col2 in train_fe.columns:
            train_fe[f'{col1}_{col2}_Interaction'] = train_fe[col1] * train_fe[col2]
        if col1 in test_fe.columns and col2 in test_fe.columns:
            test_fe[f'{col1}_{col2}_Interaction'] = test_fe[col1] * test_fe[col2]
    if all(n in train_fe.columns for n in ['Nitrogen', 'Potassium', 'Phosphorous']):
        train_fe['NPK_Interaction'] = train_fe['Nitrogen'] * train_fe['Potassium'] * train_fe['Phosphorous']
    if all(n in test_fe.columns for n in ['Nitrogen', 'Potassium', 'Phosphorous']):
        test_fe['NPK_Interaction'] = test_fe['Nitrogen'] * test_fe['Potassium'] * test_fe['Phosphorous']

    for col1, col2 in combinations(numeric_cols, 2):
        if col1 in train_fe.columns and col2 in train_fe.columns:
            train_fe[f'{col1}_{col2}_Ratio'] = train_fe[col1] / (train_fe[col2] + FE_EPSILON)
        if col1 in test_fe.columns and col2 in test_fe.columns:
            test_fe[f'{col1}_{col2}_Ratio'] = test_fe[col1] / (test_fe[col2] + FE_EPSILON)

    for cat_col in categorical_cols:
        if cat_col not in train_fe.columns: continue
        for num_col in numeric_cols:
            if num_col not in train_fe.columns: continue
            feature_name = f'{num_col}_by_{cat_col}_mean'
            group_means = train_fe.groupby(cat_col)[num_col].mean()
            train_fe[feature_name] = train_fe[cat_col].map(group_means)
            global_mean_train = train_fe[num_col].mean()
            train_fe[feature_name].fillna(global_mean_train, inplace=True)
            if cat_col in test_fe.columns:
                test_fe[feature_name] = test_fe[cat_col].map(group_means)
                test_fe[feature_name].fillna(global_mean_train, inplace=True)
            else:
                test_fe[feature_name] = global_mean_train
    
    if all(c in train_fe.columns for c in ['Soil Type', 'Crop Type']):
        train_fe['Soil_Crop_Interaction'] = train_fe['Soil Type'].astype(str) + "_" + train_fe['Crop Type'].astype(str)
    if all(c in test_fe.columns for c in ['Soil Type', 'Crop Type']):
        test_fe['Soil_Crop_Interaction'] = test_fe['Soil Type'].astype(str) + "_" + test_fe['Crop Type'].astype(str)
    
    for col in numeric_cols:
        if col in train_fe.columns and pd.api.types.is_numeric_dtype(train_fe[col]):
            safe_train_col = train_fe[col].clip(lower=0)
            if not safe_train_col.isnull().all():
                skewness = skew(safe_train_col.dropna())
                if abs(skewness) > FE_SKEWNESS_THRESHOLD:
                    train_fe[f'Log_{col}'] = np.log1p(safe_train_col)
                    if col in test_fe.columns and pd.api.types.is_numeric_dtype(test_fe[col]):
                         safe_test_col = test_fe[col].clip(lower=0)
                         test_fe[f'Log_{col}'] = np.log1p(safe_test_col)

    y_train_labels = train_fe[target_col]
    X_train_fe = train_fe.drop(columns=[target_col])
    X_test_fe = test_fe.copy()

    train_cols = X_train_fe.columns.tolist()
    X_test_fe_aligned = pd.DataFrame(columns=train_cols)
    for col in train_cols:
        if col in X_test_fe.columns:
            X_test_fe_aligned[col] = X_test_fe[col]
        else:
            if pd.api.types.is_numeric_dtype(X_train_fe[col]): X_test_fe_aligned[col] = 0
            else: X_test_fe_aligned[col] = 'Unknown_FE'
    X_test_fe = X_test_fe_aligned[train_cols]
            
    print(f"Feature Engineering complete. Train features: {X_train_fe.shape}, Test features: {X_test_fe.shape}")
    return X_train_fe, y_train_labels, X_test_fe

X_train_full_fe, y_train_original_labels, X_test_full_fe = feature_engineering_transform(
     df_train, df_test, numerical_cols, categorical_cols, target_column
)


target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train_original_labels)
print(f" Target variable '{target_column}' encoded. Classes: {len(target_encoder.classes_)}")


def get_processed_data_and_top_features(X_train, X_test, y_train, base_cat_cols, n_top_features):
    print("\n Starting Preprocessing and SHAP Feature Selection...")
    
    # Define feature types from the feature-engineered data
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Create preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='passthrough'
    )

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    processed_feature_names = preprocessor.get_feature_names_out()
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)

    # Transform the test 
    X_test_processed = preprocessor.transform(X_test)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)

    print(f" Training a base XGBoost model for SHAP on {X_train_processed_df.shape[1]} features...")
    shap_params = {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'random_state': 42}
    if USE_GPU: shap_params.update({'device': 'cuda', 'tree_method': 'hist'})
    
    shap_base_model = xgb.XGBClassifier(**shap_params, n_estimators=100)
    shap_base_model.fit(X_train_processed_df, y_train)

    print("ğŸŒ³ Calculating SHAP values...")
    explainer = shap.TreeExplainer(shap_base_model)
    shap_values = explainer.shap_values(X_train_processed_df.sample(min(1000, len(X_train_processed_df)), random_state=42), check_additivity=False)
    
    mean_abs_shap = np.mean([np.abs(s_val) for s_val in shap_values], axis=0).mean(axis=0)
    
    shap_summary_df = pd.DataFrame({'feature': processed_feature_names, 'mean_abs_shap_value': mean_abs_shap})
    shap_summary_df = shap_summary_df.sort_values(by='mean_abs_shap_value', ascending=False)
    
    top_features = shap_summary_df['feature'].head(n_top_features).tolist()
    print(f"\n Top {n_top_features} features selected by SHAP.")
    
    # Return the processed dataframes pruned to the top features
    X_train_selected = X_train_processed_df[top_features]
    X_test_selected = X_test_processed_df[top_features]
    
    print(f" Data pruned to top {n_top_features} features. Train shape: {X_train_selected.shape}, Test shape: {X_test_selected.shape}")
    return X_train_selected, X_test_selected

X_train_selected, X_test_selected = get_processed_data_and_top_features(
    X_train_full_fe, X_test_full_fe, y_train_encoded, categorical_cols, N_TOP_SHAP_FEATURES
)


def get_model_device_params(model_name, use_gpu_flag):
    if not use_gpu_flag: return {'device': 'cpu', 'tree_method': 'hist'} if model_name == "XGBoost" else {}
    if model_name == "XGBoost": return {'device': 'cuda', 'tree_method': 'hist'}
    elif model_name == "LightGBM": return {'device_type': 'gpu'}
    elif model_name == "CatBoost": return {'task_type': 'GPU', 'devices': '0'}
    return {}


def _objective_optuna_with_cv(trial, X, y, model_name, model_class, common_params, specific_param_func):
    gpu_params = get_model_device_params(model_name, USE_GPU)
    model_specific_params = specific_param_func(trial)
    if model_name == "CatBoost":
        model_specific_params['verbose'] = 0
        model_specific_params['allow_writing_files'] = False
    
    model = model_class(**common_params, **gpu_params, **model_specific_params)
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba_fold = model.predict_proba(X_val_fold)
        fold_map3_score = map_at_3(y_val_fold, y_pred_proba_fold)
        cv_scores.append(fold_map3_score)
        
    return np.mean(cv_scores)


def xgb_params_optuna(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'gamma': trial.suggest_float('gamma', 0.0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
    }

def lgb_params_optuna(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 1600),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 15, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 18),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10),
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }

def cb_params_optuna(trial):
    return {
        'iterations': trial.suggest_int('iterations', 800, 1600),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.15),
        'depth': trial.suggest_int('depth', 3, 15),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.3, 30.0),
        'border_count': trial.suggest_int('border_count', 30, 255),
    }


optuna_studies = {}
models_to_tune = {
    # "LightGBM": (lgb.LGBMClassifier, {'objective': 'multiclass', 'metric': 'multi_logloss', 'random_state': 42, 'n_jobs': -1, 'verbose': -1}, lgb_params_optuna),
    "XGBoost": (xgb.XGBClassifier, {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'random_state': 42, 'n_jobs': -1}, xgb_params_optuna),
    # "CatBoost": (cb.CatBoostClassifier, {'loss_function': 'MultiClass', 'eval_metric': 'MultiClass', 'random_seed': 42, 'thread_count': -1}, cb_params_optuna)
}

print("\n\n--- ğŸš€ Starting Optuna Hyperparameter Tuning ---")
for model_name, (model_class, common_params, specific_param_func) in models_to_tune.items():
    print(f"\n--- Tuning {model_name} ---")
    study = optuna.create_study(direction='maximize', study_name=f"{model_name}_study")
    study.optimize(
        lambda trial: _objective_optuna_with_cv(
            trial, X_train_selected, y_train_encoded, model_name, model_class, common_params, specific_param_func
        ),
        n_trials=OPTUNA_N_TRIALS,
    )
    
    print(f" Best CV MAP@3 for {model_name}: {study.best_value:.4f}")
    print(f" Best params: {study.best_params}")
    optuna_studies[model_name] = study


skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
num_classes = len(target_encoder.classes_)

all_oof_preds = {}
all_test_preds = {}

for model_name, (model_class, common_params, _) in models_to_tune.items():
    print(f"\n--- Processing Model: {model_name} ---")
    if model_name not in optuna_studies:
        print(f" Optuna study for {model_name} not found. Skipping.")
        continue

    best_hyperparams = optuna_studies[model_name].best_params
    gpu_final_params = get_model_device_params(model_name, USE_GPU)
    
    oof_preds = np.zeros((len(X_train_selected), num_classes))
    test_preds_sum = np.zeros((len(X_test_selected), num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train_encoded)):
        print(f"  --- Fold {fold + 1}/{cv_folds} ---")
        X_train_fold, X_val_fold = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
        y_train_fold = y_train_encoded[train_idx]
        
        fold_model = model_class(**common_params, **gpu_final_params, **best_hyperparams)
        if model_name == "CatBoost":
            fold_model.set_params(verbose=0, allow_writing_files=False)

        fold_model.fit(X_train_fold, y_train_fold)
        oof_preds[val_idx] = fold_model.predict_proba(X_val_fold)
        test_preds_sum += fold_model.predict_proba(X_test_selected)

    
    overall_oof_map3 = map_at_3(y_train_encoded, oof_preds)
    print(f" {model_name} - Overall OOF MAP@3 Score: {overall_oof_map3:.5f}")

    avg_test_preds = test_preds_sum / cv_folds
    
    all_oof_preds[model_name] = oof_preds
    all_test_preds[model_name] = avg_test_preds

    indices = np.argsort(avg_test_preds, axis=1)[:, -3:][:, ::-1]
    names = np.array([target_encoder.inverse_transform(preds) for preds in indices])
    submission_preds = [" ".join(names) for names in names]
    
    submission_df = pd.DataFrame({ID_COLUMN_TEST: test_ids_for_submission, target_column: submission_preds})
    file_name = f"submission_{model_name}.csv"
    submission_df.to_csv(file_name, index=False)
    print(f" Submission file saved: {file_name}")


# final_test_preds = (0.4 * all_test_preds['LightGBM']) + (0.2 * all_test_preds['XGBoost']) + (0.4 * all_test_preds['CatBoost'])
# # model_weights = [1/len(all_test_preds)] * len(all_test_preds) # Equal weights
# # ensembled_test_preds = np.zeros_like(list(all_test_preds.values())[0])

# # for i, model_name in enumerate(all_test_preds.keys()):
# #     print(f"Averaging {model_name} with weight {model_weights[i]:.2f}")
# #     ensembled_test_preds += model_weights[i] * all_test_preds[model_name]

# indices_ensemble = np.argsort(final_test_preds, axis=1)[:, -3:][:, ::-1]
# top_3_names_ensemble = np.array([target_encoder.inverse_transform(preds) for preds in indices_ensemble])
# submission_preds_ensemble = [" ".join(names) for names in top_3_names_ensemble]

# submission_df_ensemble = pd.DataFrame({ID_COLUMN_TEST: test_ids_for_submission, target_column: submission_preds_ensemble})
# file_name_ensemble = "submission_Ensemble_Average.csv"
# submission_df_ensemble.to_csv(file_name_ensemble, index=False)

# print(f"\n Ensemble submission file saved: {file_name_ensemble}")
# print("Final Submission DataFrame head:")
# print(submission_df_ensemble.head())




