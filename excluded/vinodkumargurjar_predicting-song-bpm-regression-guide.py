# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df_train=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


df_train.head(5)


df_test.head(5)


df_train.drop(columns=["id"], axis=1, inplace=True)
df_test.drop(columns=["id"], axis=1, inplace=True)


target_variable="BeatsPerMinute"


import seaborn as sns
import matplotlib.pyplot as plt
def eda_pipeline(df_train, df_test):
    
    # Display first few rows
    print("\n--- First few rows of train data ---")
    display(df_train.head())
    
    print("\n--- First few rows of test data ---")
    display(df_test.head())
    
    # Dataset info
    print("\n--- Train Data Info ---")
    print(df_train.info())
    
    print("\n--- Test Data Info ---")
    print(df_test.info())
    
    # Missing values
    print("\n--- Missing Values in Train Data ---")
    print(df_train.isnull().sum())
    
    print("\n--- Missing Values in Test Data ---")
    print(df_test.isnull().sum())
    
    print("\n--- Percentage of Missing Values in Train Data ---")
    print((df_train.isnull().sum() / len(df_train)) * 100)
    
    print("\n--- Percentage of Missing Values in Test Data ---")
    print((df_test.isnull().sum() / len(df_test)) * 100)
    
    # Summary statistics
    print("\n--- Train Data Summary Statistics ---")
    print(df_train.describe())
    
    print("\n--- Test Data Summary Statistics ---")
    print(df_test.describe())
    
    # Identify categorical columns
    train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
    test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']
    
    print("\n--- Categorical Columns in Train Data ---")
    print(train_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Train) ---")
    print(df_train[train_cat_columns].nunique())
    
    print("\n--- Categorical Columns in Test Data ---")
    print(test_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Test) ---")
    print(df_test[test_cat_columns].nunique())
    
    # Identify numerical columns
    train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    print("\n--- Numerical Columns in Train Data ---")
    print(train_num_columns)
    
    print("\n--- Numerical Columns in Test Data ---")
    print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # Correlation matrix (excluding non-numeric columns)
    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    plt.show()
       
    # Correlation with Target Variable
    print("\n--- Correlation with Target Variable ---")
    target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    print(target_corr)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title(f'Feature Correlation with {target_variable}')
    plt.show()   
    
    # Distribution plots for numerical features
    print("\n--- Distribution of Numerical Features ---")
    df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    plt.show()
    
    # Box plots for outlier detection
    print("\n--- Box Plots for Outlier Detection ---")
    for col in train_num_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_train[col])
        plt.title(f'Box plot of {col}')
        plt.show()
    
    # Value counts for categorical features
    print("\n--- Value Counts for Categorical Columns ---")
    for col in train_cat_columns:
        print(f"\nValue counts for {col}:")
        print(df_train[col].value_counts())


eda_pipeline(df_train, df_test)


print("\n--- Distribution of Target Variable ---\n")
plt.figure(figsize=(8,5))
sns.histplot(df_train[target_variable], bins=50, kde=True)
plt.xlabel("Beats Per Minute (BPM)")
plt.ylabel("Frequency")
plt.title("Distribution of Target Variable (Regression)")
plt.show()


def data_preprocessing_pipeline_ohe(df_train, df_test, target_column='y'):
    """
    Preprocess data using One-Hot Encoding.
    Ensures same columns in train and test by combining before encoding.
    Excludes target column from test set during encoding.
    """

    # Drop target from test set if accidentally included
    if target_column in df_test.columns:
        df_test = df_test.drop(columns=[target_column])

   
    cat_cols = df_train.drop(columns=[target_column]).select_dtypes(include=['object', 'category']).columns.tolist()

    # Combine train and test for consistent encoding
    df_train['__is_train__'] = 1
    df_test['__is_train__'] = 0
    combined = pd.concat([df_train, df_test], axis=0)

    # Apply One-Hot Encoding
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    # Split back
    df_train_encoded = combined[combined['__is_train__'] == 1].drop(columns=['__is_train__'])
    df_test_encoded = combined[combined['__is_train__'] == 0].drop(columns=['__is_train__', target_column], errors='ignore')

    return df_train_encoded, df_test_encoded


# df_train_processed, df_test_processed =data_preprocessing_pipeline_ohe(df_train, df_test, target_column=target_variable)


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test):
   
    label_encoders = {}

    # Encode categorical features in training set
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le

    # Encode categorical features in test set using train encoders
    for column in df_test.columns:
        if column in label_encoders:
            le = label_encoders[column]
            df_test[column] = df_test[column].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        elif df_test[column].dtype == 'object':
            df_test[column] = -1  # default encoding for unknown categorical column

    return df_train, df_test, label_encoders


# df_train,df_test,label_encoders=data_preprocessing_pipeline(df_train, df_test)


from sklearn.preprocessing import StandardScaler

def standardize_data(df_train, df_test):

    # Separate target column from train data
    target_values = df_train[target_variable]
    df_train = df_train.drop(columns=[target_variable])
    
    # Ensure both datasets have the same feature columns
    common_columns = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit on train data and transform both train and test data
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=common_columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=common_columns)
    
    # Reattach the target column to the scaled train data
    df_train_scaled[target_variable] = target_values.reset_index(drop=True)
    
    return df_train_scaled, df_test_scaled,scaler


df_train_scalled, df_test_scalled,std_scaler = standardize_data(df_train, df_test)


df_train_scalled.head(2)


df_test_scalled.head(2)


df_train.head(2)


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]

X_scalled = df_train_scalled.drop(columns=[target_variable])
y_scalled = df_train_scalled[target_variable]


X.head(2)


X_scalled.head(2)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, 
                                                    random_state=42)

X_train_scalled, X_test_scalled, y_train_scalled, y_test_scalled = train_test_split(
    X_scalled, y_scalled, test_size=0.2,stratify=y_scalled,random_state=42)


from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# =========================
# Models
# =========================
catboost_model_non_scaled = CatBoostRegressor(verbose=0, random_state=42)
xgb_model_non_scaled = XGBRegressor(random_state=42, n_jobs=-1)

catboost_model_scaled = CatBoostRegressor(verbose=0, random_state=42)
xgb_model_scaled = XGBRegressor(random_state=42, n_jobs=-1)

# =========================
# Non-scaled Data
# =========================
catboost_model_non_scaled.fit(X_train, y_train)
preds_catboost_non_scaled = catboost_model_non_scaled.predict(X_test)
rmse_catboost_non_scaled = np.sqrt(mean_squared_error(y_test, preds_catboost_non_scaled))

xgb_model_non_scaled.fit(X_train, y_train)
preds_xgb_non_scaled = xgb_model_non_scaled.predict(X_test)
rmse_xgb_non_scaled = np.sqrt(mean_squared_error(y_test, preds_xgb_non_scaled))

# =========================
# Scaled Data
# =========================
catboost_model_scaled.fit(X_train_scalled, y_train_scalled)
preds_catboost_scaled = catboost_model_scaled.predict(X_test_scalled)
rmse_catboost_scaled = np.sqrt(mean_squared_error(y_test_scalled, preds_catboost_scaled))

xgb_model_scaled.fit(X_train_scalled, y_train_scalled)
preds_xgb_scaled = xgb_model_scaled.predict(X_test_scalled)
rmse_xgb_scaled = np.sqrt(mean_squared_error(y_test_scalled, preds_xgb_scaled))

# =========================
# Results
# =========================
print("\n--- RMSE Results ---")
print(f"CatBoost (Non-Scaled): {rmse_catboost_non_scaled:.4f}")
print(f"XGBoost  (Non-Scaled): {rmse_xgb_non_scaled:.4f}")
print(f"CatBoost (Scaled):     {rmse_catboost_scaled:.4f}")
print(f"XGBoost  (Scaled):     {rmse_xgb_scaled:.4f}")


# import optuna
# from sklearn.metrics import mean_squared_error
# from catboost import CatBoostRegressor
# import numpy as np

# # =========================
# # Objective function for Optuna
# # =========================
# def objective(trial):
#     # Suggest hyperparameters
#     params = {
#         "iterations": trial.suggest_int("iterations", 500, 2000),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
#         "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "random_state": 42,
#         "verbose": 0
#     }

#     # Initialize model
#     model = CatBoostRegressor(**params)

#     # Train
#     model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=0)

#     # Predict
#     preds = model.predict(X_test)

#     # RMSE
#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     return rmse

# # =========================
# # Run Optuna Study
# # =========================
# study = optuna.create_study(direction="minimize")  # minimize RMSE
# study.optimize(objective, n_trials=30)  # try 30 trials (increase for better tuning)

# # =========================
# # Best Parameters & RMSE
# # =========================
# print("Best RMSE:", study.best_value)
# print("Best hyperparameters:", study.best_params)
# # # =========================
# # # Train final model with best params
# # # =========================
# # best_catboost = CatBoostRegressor(**study.best_params, random_state=42, verbose=0)
# # best_catboost.fit(X_train, y_train)

# # # Final prediction
# # final_preds = best_catboost.predict(X_test)
# # final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
# # print("Final CatBoost RMSE:", final_rmse)



best_params= {'iterations': 1014, 'depth': 8, 'learning_rate': 0.029973231246867238,
              'l2_leaf_reg': 5.3670034228396375, 'random_strength': 0.03348421868450528,
              'bagging_temperature': 2.9871910174115057, 'border_count': 35}


# =========================
# Train final model with best params
# =========================
tuned_catboost = CatBoostRegressor(**best_params, random_state=42, verbose=0)
tuned_catboost.fit(X_train, y_train)

# Final prediction
tuned_catboost_preds = tuned_catboost.predict(X_test)
tuned_catboost_rmse = np.sqrt(mean_squared_error(y_test, tuned_catboost_preds))
print("Tuned CatBoost RMSE:", tuned_catboost_rmse)


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# =========================
# Initialize LGBM Model
# =========================
lgbm_model = LGBMRegressor(
    n_estimators=1000,      # number of boosting rounds
    learning_rate=0.05,    # smaller -> slower but more accurate
    random_state=42,
    n_jobs=-1
)

# =========================
# Train on Non-Scaled Data
# =========================
lgbm_model.fit(X_train, y_train)

# =========================
# Predict
# =========================
y_pred_lgbm = lgbm_model.predict(X_test)

# =========================
# Evaluate with RMSE
# =========================
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f"LightGBM (Non-Scaled) RMSE: {rmse_lgbm:.4f}")



from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# =========================
# Initialize Ridge Model
# =========================
ridge_model = Ridge(alpha=1, random_state=42)

# =========================
# Train on Non-Scaled Data
# =========================
ridge_model.fit(X_train, y_train)

# =========================
# Predict
# =========================
y_pred_ridge = ridge_model.predict(X_test)

# =========================
# Evaluate with RMSE
# =========================
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"Ridge Regression (Non-Scaled) RMSE: {rmse_ridge:.4f}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# =========================
# Train Linear Regression
# =========================
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)

# =========================
# Predict
# =========================
y_pred_linreg = linreg_model.predict(X_test)

# =========================
# Evaluate RMSE
# =========================
rmse_linreg = np.sqrt(mean_squared_error(y_test, y_pred_linreg))
print(f"Linear Regression (Non-Scaled) RMSE: {rmse_linreg:.4f}")


from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# =========================
# Define ElasticNetCV with scaling
# =========================
elasticnet_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("elasticnetcv", ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],  # test different L1/L2 balances
        alphas=np.logspace(-3, 3, 20),           # penalty strength values
        cv=5,
        max_iter=10000,
        random_state=42
    ))
])

# =========================
# Train ElasticNetCV
# =========================
elasticnet_pipeline.fit(X_train, y_train)

# =========================
# Predict
# =========================
y_pred_enet = elasticnet_pipeline.predict(X_test)

# =========================
# Evaluate RMSE
# =========================
rmse_enet = np.sqrt(mean_squared_error(y_test, y_pred_enet))
print(f"ElasticNetCV (Scaled) RMSE: {rmse_enet:.4f}")

# Best alpha and l1_ratio
best_alpha = elasticnet_pipeline.named_steps["elasticnetcv"].alpha_
best_l1_ratio = elasticnet_pipeline.named_steps["elasticnetcv"].l1_ratio_
print(f"Best alpha: {best_alpha}")
print(f"Best l1_ratio: {best_l1_ratio}")


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import numpy as np

# =========================
# Base learners (no DNN)
# =========================
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression

# Define base models
base_learners = [
    ('catboost', CatBoostRegressor(verbose=0, random_state=42)),
    ('xgboost', XGBRegressor(random_state=42, n_jobs=-1)),
    ('lightgbm', LGBMRegressor(random_state=42)),
    ('ridge', Ridge(alpha=1000.0, random_state=42)),
    ('elastic', ElasticNet(random_state=42)),
    ('linear', LinearRegression())
]

# Meta-model (stacking blender)
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])

# =========================
# Stacking Regressor
# =========================
stack_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_model,
    n_jobs=-1,
    passthrough=False  # if True â†’ raw features also go to meta-model
)

# Fit
stack_model.fit(X_train, y_train)

# Predict
stack_preds = stack_model.predict(X_test)

# RMSE
rmse_stack = np.sqrt(mean_squared_error(y_test, stack_preds))
print(f"Stacking Regressor RMSE: {rmse_stack:.4f}")


# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers, callbacks
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # =========================
# # Build Advanced DNN Model
# # =========================
# def build_best_dnn(input_dim):
#     model = models.Sequential([
#         layers.Dense(256, activation='relu', input_shape=(input_dim,),
#                      kernel_regularizer=regularizers.l2(1e-4)),
#         layers.BatchNormalization(),
#         layers.Dropout(0.3),

#         layers.Dense(128, activation='relu',
#                      kernel_regularizer=regularizers.l2(1e-4)),
#         layers.BatchNormalization(),
#         layers.Dropout(0.3),

#         layers.Dense(64, activation='relu',
#                      kernel_regularizer=regularizers.l2(1e-4)),
#         layers.BatchNormalization(),
#         layers.Dropout(0.2),

#         layers.Dense(32, activation='relu'),
#         layers.Dense(1)  # Regression output
#     ])

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#                   loss='mse',
#                   metrics=[tf.keras.metrics.RootMeanSquaredError()])
#     return model

# # =========================
# # Callbacks
# # =========================
# early_stop = callbacks.EarlyStopping(
#     monitor='val_loss', patience=10, restore_best_weights=True)

# checkpoint = callbacks.ModelCheckpoint(
#     "best_dnn_model.h5", monitor='val_loss', save_best_only=True)

# # =========================
# # Train Model
# # =========================
# dnn_model = build_best_dnn(X_train_scalled.shape[1])

# history = dnn_model.fit(
#     X_train_scalled, y_train_scalled,
#     validation_data=(X_test_scalled, y_test_scalled),
#     epochs=100,  # can increase with GPU
#     batch_size=256,
#     callbacks=[early_stop, checkpoint],
#     verbose=1
# )

# # =========================
# # Evaluate
# # =========================
# y_pred_dnn = dnn_model.predict(X_test_scalled)
# rmse_dnn = np.sqrt(mean_squared_error(y_test_scalled, y_pred_dnn))
# print(f"\n DNN Model RMSE: {rmse_dnn:.4f}")


print("\n---All  RMSE Results ---")
print(f"CatBoost (Non-Scaled): {rmse_catboost_non_scaled:.4f}")
print(f"XGBoost  (Non-Scaled): {rmse_xgb_non_scaled:.4f}")
print(f"CatBoost (Scaled):     {rmse_catboost_scaled:.4f}")
print(f"XGBoost  (Scaled):     {rmse_xgb_scaled:.4f}")
print(f"CatBoost Tuned RMSE:  {tuned_catboost_rmse:.4f}")
print(f"LightGBM (Non-Scaled) RMSE: {rmse_lgbm:.4f}")
print(f"Ridge Regression (Non-Scaled) RMSE: {rmse_ridge:.4f}")
print(f"Linear Regression (Non-Scaled) RMSE: {rmse_linreg:.4f}")
print(f"ElasticNetCV (Scaled) RMSE: {rmse_enet:.4f}")
print(f"Stacking Regressor RMSE: {rmse_stack:.4f}")


final_prediction = stack_model.predict(df_test)


final_prediction


sample_submission.head(5)


sample_submission["BeatsPerMinute"]=final_prediction
sample_submission.to_csv('submission.csv', index=False)
print('Submission file saved.')


sample_submission.head(5)




