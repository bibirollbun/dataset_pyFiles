# Check GPU availability
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))


# !pip install scikeras --quiet
# !pip install keras-tuner --quiet

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
# from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from xgboost import XGBClassifier

tf.random.set_seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


# Load and optimize data
def reduce_memory_usage(df):
    """Downcasts numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

train = reduce_memory_usage(train)
test = reduce_memory_usage(test)

# Split training data into features and target
X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']

# Save test ids and features
test_ids = test['id']
X_test = test.drop(['id'], axis=1)


# Extended Feature Engineering Functions
def advanced_features(df, is_train=False, target_series=None):
    """
    Create advanced features:
      - Build a date from 'day' (assume day=1 corresponds to 2024-01-01)
      - Extract date-based features: month, day_of_year, week_of_year, quarter, day_of_week, is_weekend
      - Create periodic features (sine & cosine transforms)
      - Compute interaction/ratio features (e.g., temp_range, humidity_cloud_ratio)
      - For training data, compute lag features (gap_before_rain, gap_after_rain)
    """
    df = df.copy()
    base_date = pd.to_datetime('2024-01-01')
    df['date'] = base_date + pd.to_timedelta(df['day'] - 1, unit='D')
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    # df['day_of_week'] = df['date'].dt.weekday
    # df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    if 'maxtemp' in df.columns and 'mintemp' in df.columns:
        df['temp_range'] = df['maxtemp'] - df['mintemp']
    if 'temparature' in df.columns and 'dewpoint' in df.columns:
        df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    if 'humidity' in df.columns and 'cloud' in df.columns:
        df['humidity_cloud_ratio'] = df['humidity'] / (df['cloud'] + 1e-3)
    if 'sunshine' in df.columns and 'cloud' in df.columns:
        df['sunshine_cloud_ratio'] = df['sunshine'] / (df['cloud'] + 1e-3)
    if 'pressure' in df.columns and 'winddirection' in df.columns:
        df['pressure_wind_interaction'] = df['pressure'] * df['winddirection']
    if 'temparature' in df.columns and 'pressure' in df.columns:
        df['temp_pressure_ratio'] = df['temparature'] / (df['pressure'] + 1e-3)
    if 'windspeed' in df.columns and 'pressure' in df.columns:
        df['wind_pressure_ratio'] = df['windspeed'] / (df['pressure'] + 1e-3)
    # if is_train:
    #     if target_series is not None:
    #         df['rainfall'] = target_series.values
    #     df = df.sort_values('date').reset_index(drop=True)
    #     df['rain_prev_day'] = df['rainfall'].shift(1).fillna(0)
    #     df['rain_next_day'] = df['rainfall'].shift(-1).fillna(0)
    #     df['gap_before_rain'] = df.groupby((df['rain_prev_day'] != df['rainfall']).cumsum()).cumcount()
    #     df['gap_after_rain'] = df[::-1].groupby((df['rain_next_day'] != df['rainfall']).cumsum()).cumcount()
    #     df.drop(['rain_prev_day', 'rain_next_day'], axis=1, inplace=True)
    # else:
    #     df['gap_before_rain'] = 0
    #     df['gap_after_rain'] = 0
    df.drop(['date'], axis=1, inplace=True, errors='ignore')
    return df

def additional_poly_features(df, cols_to_expand):
    """
    Generate polynomial features (degree 2) for selected columns.
    Missing values are imputed (median) before transformation.
    """
    df_imputed = df[cols_to_expand].fillna(df[cols_to_expand].median())
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df_imputed)
    poly_feature_names = poly.get_feature_names_out(cols_to_expand)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    return poly_df

def rolling_features(df, window_sizes=[7, 14]):
    """
    Generate rolling window statistics (mean and std) for all numeric columns.
    Returns a DataFrame with new rolling features as 1D arrays.
    """
    df_roll = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        series_col = df[col]
        for window in window_sizes:
            roll_mean = series_col.rolling(window=window, min_periods=1).mean().values
            roll_std = series_col.rolling(window=window, min_periods=1).std().fillna(0).values
            if roll_mean.ndim > 1:
                roll_mean = roll_mean[:, 0]
            if roll_std.ndim > 1:
                roll_std = roll_std[:, 0]
            df_roll[f'{col}_roll_mean_{window}'] = roll_mean
            df_roll[f'{col}_roll_std_{window}'] = roll_std
    return df_roll


def extended_features_train(df, target_series):
    # Compute advanced features using training data (with target)
    df_adv = advanced_features(df, is_train=True, target_series=target_series)
    # Extract target and drop it from features
    y_out = df_adv.pop('rainfall')
    # Remove any columns that were derived solely from the target (if any)
    cols_to_drop = [col for col in df_adv.columns if col.startswith('rainfall')]
    df_adv = df_adv.drop(columns=cols_to_drop, errors='ignore')
    
    # Generate polynomial features for selected candidate columns
    candidate_cols = ['maxtemp', 'mintemp', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'pressure', 'winddirection', 'windspeed']
    cols_for_poly = [col for col in candidate_cols if col in df_adv.columns]
    if cols_for_poly:
        poly_df = additional_poly_features(df_adv, cols_for_poly)
        df_adv = pd.concat([df_adv, poly_df], axis=1)
    
    # Generate rolling window features
    roll_df = rolling_features(df_adv, window_sizes=[7, 14])
    df_adv = pd.concat([df_adv, roll_df], axis=1)
    
    return df_adv, y_out

def extended_features_test(df):
    # For test data, compute advanced features without lag features
    df_ext = advanced_features(df, is_train=False)
    candidate_cols = ['maxtemp', 'mintemp', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'pressure', 'winddirection', 'windspeed']
    cols_for_poly = [col for col in candidate_cols if col in df_ext.columns]
    if cols_for_poly:
        poly_df = additional_poly_features(df_ext, cols_for_poly)
        df_ext = pd.concat([df_ext, poly_df], axis=1)
    roll_df = rolling_features(df_ext, window_sizes=[7, 14])
    df_ext = pd.concat([df_ext, roll_df], axis=1)
    return df_ext

# Compute extended features for training and test data.
train_ext, y_ext = extended_features_train(pd.concat([X, y], axis=1), target_series=y)
X_ext_full = train_ext.copy()
X_test_ext_full = extended_features_test(X_test)

# Align training and test columns (fill missing with 0)
all_cols = X_ext_full.columns.union(X_test_ext_full.columns)
X_ext_full = X_ext_full.reindex(columns=all_cols, fill_value=0)
X_test_ext_full = X_test_ext_full.reindex(columns=all_cols, fill_value=0)

print("Extended Train Shape:", X_ext_full.shape)
print("Extended Test Shape:", X_test_ext_full.shape)


def treat_outliers_iqr(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].squeeze()  # Ensure it's a 1D array/Series
        Q1 = np.nanquantile(series, 0.25)
        Q3 = np.nanquantile(series, 0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Convert to NumPy array, clip, and assign back
        df[col] = np.clip(series.values, lower, upper)
    return df

# Now apply the function on the extended feature sets:
X_ext_iqr = treat_outliers_iqr(X_ext_full)
X_test_ext_iqr = treat_outliers_iqr(X_test_ext_full)


from sklearn.feature_selection import SelectFromModel

def train_and_submit(X_train, y_train, X_pred, model, param_dist, model_name):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42), threshold='median')),
        ('clf', model)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Increase iterations for more thorough tuning.
    search = RandomizedSearchCV(pipeline, param_dist, n_iter=30, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_score = search.best_score_
    print(f"Best CV ROC AUC for {model_name}: {best_score:.4f}")
    print("Best Params:", search.best_params_)
    
    # Extract selected features from the selector step
    selector = search.best_estimator_.named_steps['selector']
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask]
    print(f"Selected features for {model_name}:")
    print(selected_features.tolist())
    
    # Reindex test set to match training set columns
    training_cols = X_train.columns
    X_pred_aligned = X_pred.reindex(columns=training_cols, fill_value=0)
    
    try:
        preds = search.predict_proba(X_pred_aligned.values)[:, 1]
    except Exception as e:
        print(f"Error in predict_proba for {model_name}: {e}")
        preds = np.full(len(X_pred_aligned), 0.5)
    
    submission = pd.DataFrame({'id': test_ids, 'rainfall': preds})
    submission.to_csv(f"{model_name}_submission.csv", index=False)
    return best_score


# Define parameter grids for each model:
param_grids = {
    'LogisticRegression': {
        'clf__C': np.logspace(-3, 2, 10),
        'clf__solver': ['lbfgs', 'liblinear']
    },
    'DecisionTree': {
        'clf__max_depth': [3, 5, 7, None],
        'clf__min_samples_split': [2, 5, 10]
    },
    'ExtraTrees': {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [5, 7, None]
    },
    'RandomForest': {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [5, 7, None]
    },
    'XGBoost': {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [3, 5, 7]
    },
    'CatBoost': {
        'clf__iterations': [100, 200],
        'clf__depth': [4, 6]
    },
    'LGBM': {
        'clf__num_leaves': [31, 50],
        'clf__n_estimators': [50, 100, 150]
    },
    'KNN': {
        'clf__n_neighbors': [3, 5, 7],
        'clf__weights': ['uniform', 'distance']
    },
    'SVC': {
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf']
    }
}

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'LGBM': LGBMClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(probability=True, random_state=42)
}

model_results = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name} on Extended IQR-treated & Standard Scaled features:")
    auc = train_and_submit(X_ext_iqr, y_ext, X_test_ext_iqr, model, param_grids[model_name], model_name)
    model_results[model_name] = auc


# # Neural Network Model
# # Build the NN model-building function.
# def build_nn_model(units_1=128, units_2=64, dropout_rate=0.3, learning_rate=0.001, optimizer_choice='adam'):
#     model = Sequential()
#     model.add(Dense(units_1, activation='relu', input_shape=(X_ext_iqr.shape[1],), kernel_regularizer=l2(1e-4)))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(units_2, activation='relu', kernel_regularizer=l2(1e-4)))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(1, activation='sigmoid'))
    
#     if optimizer_choice == 'adam':
#         opt = Adam(learning_rate=learning_rate)
#     elif optimizer_choice == 'rmsprop':
#         opt = RMSprop(learning_rate=learning_rate)
#     else:
#         opt = Adam(learning_rate=learning_rate)
    
#     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
#     return model

# # Use the official TensorFlow KerasClassifier wrapper.
# nn_wrapper = KerasClassifier(build_fn=build_nn_model, epochs=20, batch_size=32, verbose=0)

# # Expanded hyperparameter grid for the NN.
# nn_param_grid = {
#     'clf__model__units_1': [64, 128, 256],
#     'clf__model__units_2': [32, 64, 128],
#     'clf__model__dropout_rate': [0.2, 0.3, 0.4],
#     'clf__model__learning_rate': [1e-3, 1e-4, 5e-4],
#     'clf__model__optimizer_choice': ['adam', 'rmsprop'],
#     'clf__epochs': [20, 30, 50],
#     'clf__batch_size': [16, 32, 64]
# }

# nn_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler()),
#     ('clf', nn_wrapper)
# ])

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# nn_search = RandomizedSearchCV(nn_pipeline, nn_param_grid, n_iter=50, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1)
# nn_search.fit(X_ext_iqr, y_ext)

# best_nn_score = nn_search.best_score_
# logger.info(f"Best NN CV ROC AUC: {best_nn_score:.4f}")
# logger.info(f"Best NN Params: {nn_search.best_params_}")

# # Align test set.
# X_test_nn = X_test_ext_iqr.reindex(columns=X_ext_iqr.columns, fill_value=0)
# try:
#     nn_preds = nn_search.predict_proba(X_test_nn.values)[:, 1]
# except Exception as e:
#     logger.error(f"NN predict_proba error: {e}")
#     nn_preds = np.full(len(X_test_nn), 0.5)
    
# nn_submission = pd.DataFrame({'id': test_ids, 'rainfall': nn_preds})
# nn_submission_filename = "NN_submission.csv"
# nn_submission.to_csv(nn_submission_filename, index=False)
# logger.info(f"NN submission saved as: {nn_submission_filename}")
# model_results['NN'] = best_nn_score


# Performance Comparison
results_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'AUC']).sort_values(by='AUC', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='AUC', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison (Extended Features, IQR-treated & Standard Scaled)')
plt.xlabel('ROC AUC')
plt.xlim(0.75, 1.0)
plt.show()


# Final Submission Selection
import shutil

submission_files = {
    'LogisticRegression': "LogisticRegression_submission.csv",
    'DecisionTree': "DecisionTree_submission.csv",
    'ExtraTrees': "ExtraTrees_submission.csv",
    'RandomForest': "RandomForest_submission.csv",
    'XGBoost': "XGBoost_submission.csv",
    'CatBoost': "CatBoost_submission.csv",
    'LGBM': "LGBM_submission.csv",
    'KNN': "KNN_submission.csv",
    'SVC': "SVC_submission.csv",
    'NN': "NN_submission.csv"
}

best_model_name = max(model_results, key=model_results.get)
best_submission_file = submission_files.get(best_model_name)
if best_submission_file is None or not os.path.exists(best_submission_file):
    raise ValueError("Best model submission file not found.")
else:
    shutil.copy(best_submission_file, "submission.csv")
    logger.info(f"Best scoring model: {best_model_name}")
    logger.info(f"Copied {best_submission_file} as 'submission.csv'")

logger.info("\nAll individual model submissions:")
for model, filename in submission_files.items():
    logger.info(f"- {filename}")


results_df




