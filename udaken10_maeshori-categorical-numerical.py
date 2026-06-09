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


import os

# 0 = all logs, 1 = filter info, 2 = filter warnings, 3 = filter errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import math
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# --- LOAD EXTERNAL DATA ---
df_orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

df_orig.head()


print("Training Data Head:")
df_train.head()


print("\nTraining Data Info:")
df_train.info()


print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())


print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())


# Descriptive statistics for numerical columns
df_train.describe()


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='diagnosed_diabetes', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Diagnosed Diabetes')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# A more compact view of categorical features vs the target
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9, 0.66, 0.33])
target = 'diagnosed_diabetes'

for i, col in enumerate(categorical_features):
    grouped = df_train.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]
print(numerical_features)


for col in df_train[numerical_features]:
    print(col, len(df_train[col].unique()))


# alcohol_consumption_per_week 9, family_history_diabetes 2, hypertension_history 2, cardiovascular_history 2 はカテゴリカル変数として扱うのが妥当

col_numeric_categorical_list = ['alcohol_consumption_per_week', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
df_train[col_numeric_categorical_list] = df_train[col_numeric_categorical_list].astype('category')
df_test[col_numeric_categorical_list] = df_test[col_numeric_categorical_list].astype('category')
df_orig[col_numeric_categorical_list] = df_orig[col_numeric_categorical_list].astype('category')
df_train.info()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)



numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]
print(numerical_features)


# (Histogram/KDE) and the outlier check (Boxplot)
# Filter numerical features
numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]

# --- Configuration for the Grid ---
features_per_row = 3
total_features = len(numerical_features)
n_rows = math.ceil(total_features / features_per_row)

# We need 2 columns per feature (1 for Hist, 1 for Box), so n_cols = features_per_row * 2
fig, axes = plt.subplots(n_rows, features_per_row * 2, figsize=(20, 4 * n_rows))
axes = axes.flatten() # Flatten to make indexing easier

for i, col in enumerate(numerical_features):
    # Calculate the exact spots for this feature in the flattened grid
    # Each feature takes up 2 spots: index 2*i and 2*i + 1
    hist_idx = i * 2
    box_idx = i * 2 + 1
    
    # --- Plot 1: Distribution (Histogram + KDE) ---
    sns.histplot(df_train[col], kde=True, ax=axes[hist_idx], color='skyblue')
    axes[hist_idx].set_title(f"{col} Dist", fontsize=10)
    axes[hist_idx].set_xlabel('')
    axes[hist_idx].set_ylabel('') # Save space
    
    # --- Plot 2: Boxplot (Outliers) ---
    sns.boxplot(x=df_train[col], ax=axes[box_idx], color='lightcoral')
    axes[box_idx].set_title(f"{col} Box", fontsize=10)
    axes[box_idx].set_xlabel('')
    
# Hide any unused subplots (if features aren't a perfect multiple of 3)
for j in range(len(numerical_features) * 2, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# Violin Plot
n_cols = 3
n_rows = math.ceil(len(numerical_features) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.violinplot(x=df_train[col], ax=axes[i], color='mediumpurple')
    axes[i].set_title(col)

# Hide empty plots
for j in range(len(numerical_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt

# def identify_optimal_features(X, y):
#     print(f"Starting Feature Selection with {X.shape[1]} features...")
    
#     # We use LightGBM because it's 10x faster than XGBoost for this loop
#     params = {
#         'n_estimators': 1000,
#         'learning_rate': 0.05,
#         'num_leaves': 31,
#         'metric': 'auc',
#         'verbosity': -1,
#         'n_jobs': -1
#     }
    
#     # Current list of features to check
#     current_features = list(X.columns)
#     history = {'n_features': [], 'score': []}
#     best_score = 0
#     best_features = []
    
#     # LOOP: Remove 5 least important features every round
#     step_size = 5 
    
#     while len(current_features) > 35:
#         # 1. Fast Validation (3-Fold is enough for selection)
#         kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#         fold_scores = []
#         feature_importances = np.zeros(len(current_features))
        
#         for train_idx, val_idx in kf.split(X[current_features], y):
#             X_tr, X_val = X.iloc[train_idx][current_features], X.iloc[val_idx][current_features]
#             y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
#             model = lgb.LGBMClassifier(**params)
#             model.fit(X_tr, y_tr, callbacks=[lgb.early_stopping(50, verbose=False)], eval_set=[(X_val, y_val)])
            
#             # Record Score
#             preds = model.predict_proba(X_val)[:, 1]
#             fold_scores.append(roc_auc_score(y_val, preds))
            
#             # Record Importance
#             feature_importances += model.feature_importances_
            
#         avg_score = np.mean(fold_scores)
#         print(f"Features: {len(current_features)} | CV AUC: {avg_score:.6f}")
        
#         # Save history
#         history['n_features'].append(len(current_features))
#         history['score'].append(avg_score)
        
#         # Keep track of best
#         if avg_score > best_score:
#             best_score = avg_score
#             best_features = list(current_features)
        
#         # 2. Identify Weakest Features
#         # Sort features by importance
#         imp_df = pd.DataFrame({'feature': current_features, 'imp': feature_importances})
#         imp_df = imp_df.sort_values(by='imp', ascending=True)
        
#         # 3. Drop the bottom 'step_size' features
#         worst_features = imp_df.head(step_size)['feature'].values
#         current_features = [f for f in current_features if f not in worst_features]

#     # --- PLOT THE CURVE ---
#     plt.figure(figsize=(10, 5))
#     plt.plot(history['n_features'], history['score'], marker='o')
#     plt.gca().invert_xaxis() # Invert x-axis to show "removing features" from left to right
#     plt.title(f'Feature Selection Curve (Best: {best_score:.5f} with {len(best_features)} feats)')
#     plt.xlabel('Number of Features')
#     plt.ylabel('CV AUC Score')
#     plt.grid(True)
#     plt.show()
    
#     return best_features

# # --- EXECUTE ---
# # Run this on your df_train (make sure categorical columns are encoded properly)
# # For the selection loop, it's safest to just use Label Encoding for everything
# X_select = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
# y_select = df_train['diagnosed_diabetes']

# # Encode categoricals just for the selector
# for col in X_select.select_dtypes(include='object').columns:
#     X_select[col] = X_select[col].astype('category').cat.codes

# # GET THE WINNERS
# final_feature_list = identify_optimal_features(X_select, y_select)
# print("Best Features:", final_feature_list)


# # Update your training data to keep ONLY the winners
# X_train = df_train[final_feature_list]
# X_test  = df_test[final_feature_list]

# print(f"Train shape: {X_train.shape}")
# print(f"Test shape: {X_test.shape}")


df_train


X_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
X_test = df_test.drop(['id'], axis=1)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


numerical_features = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'diagnosed_diabetes']]

numerical_features



# # Separate target
# y_train = df_bntrain['diagnosed_diabetes'].values
# X_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
# X_test = df_test.drop(['id'], axis=1)

# print(f"Train shape: {X_train.shape}")
# print(f"Test shape: {X_test.shape}")


# Identify categorical columns
categorical_cols = categorical_features
# numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

numerical_cols = numerical_features



print(f"\nCategorical columns: {categorical_cols}")
print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"\nNumerical columns: {numerical_cols}")

target = df_train['diagnosed_diabetes']
X_raw = X_train.drop(['diagnosed_diabetes', 'id'], axis=1, errors='ignore')
X_test_raw = X_test.drop(['id'], axis=1, errors='ignore')

# --- A) Create Label Encoded Version (For XGBoost & LightGBM) ---
print("Creating Label Encoded dataset for XGB/LGBM...")
X_le = X_raw.copy()
X_test_le = X_test_raw.copy()

for col in X_le[categorical_cols].columns:
    le = LabelEncoder()
        # Handle NaNs before encoding to prevent crash
        # X_le[col] = X_le[col].fillna("MISSING").astype(str)
        # X_test_le[col] = X_test_le[col].fillna("MISSING").astype(str)
        
        # Fit on combined to cover all categories
    full_data = pd.concat([X_le[col], X_test_le[col]])
    le.fit(full_data)
    X_le[col] = le.transform(X_le[col])
    X_test_le[col] = le.transform(X_test_le[col])

# --- B) Create Native Categorical Version (For CatBoost) ---
print("Creating Native Categorical dataset for CatBoost...")
X_cat = X_raw.copy()
X_test_cat = X_test_raw.copy()

# CatBoost likes NaNs in categories to be filled with a string
"""
for col in categorical_cols:
    if col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)
        X_test_cat[col] = X_test_cat[col].fillna("Missing").astype(str)

"""


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# print(f"\n{'='*20} PREPARING DATA FOR NEURAL NET (ONE-HOT) {'='*20}")

# # 1. Identify Columns
# # We want to One-Hot Encode categoricals, and Scale numericals
# cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or X_train[c].dtype.name == 'category']
# num_cols = [c for c in X_train.columns if c not in cat_cols and c not in ['id', 'diagnosed_diabetes']]

# print(f"Categoricals: {len(cat_cols)}")
# print(f"Numericals:   {len(num_cols)}")

# # 2. Define the Preprocessing Pipeline
# # This automatically handles the "Dummy Variables" creation
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), num_cols),
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
#     ])

# # 3. Create X_nn
# # We fit on train, transform on both
# X_nn_raw = X_train.copy()
# X_test_nn_raw = X_test.copy()

# # Fit-Transform
# X_nn_processed = preprocessor.fit_transform(X_nn_raw)
# X_test_nn_processed = preprocessor.transform(X_test_nn_raw)

# # Target
# y_nn = df_train['diagnosed_diabetes']

# print(f"New NN Feature Count: {X_nn_processed.shape[1]} (was {X_nn_raw.shape[1]})")


## parameter hypertuned with optuna but just for baseline |LightGBM|XGBoost|CatBoost
## I should do it again 

xgb_params ={
    'n_estimators': 2000, # We control this via early stopping
    'early_stopping_rounds': 50,
    'booster': 'gbtree',
    'tree_method': 'hist',     # Fast training
    'eval_metric': 'logloss',
    'learning_rate': 0.010586281318793418, 
    'max_depth': 5, 
    'subsample': 0.9419910623833896, 
    'colsample_bytree': 0.5244058847875112, 
    'min_child_weight': 7, 
    'reg_alpha': 0.00015151084454479046, 
    'reg_lambda': 2.161158791085214e-08, 
    'gamma': 2.240078485583776e-07}


lgb_params = {
    'n_estimators': 2000,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.04151567000333162, 
    'num_leaves': 93, 'max_depth': 3, 
    'min_child_samples': 97, 
    'subsample': 0.8336810469662667, 
    'colsample_bytree': 0.5021699121748862, 
    'reg_alpha': 0.015640727219830758, 
    'reg_lambda': 1.374990603296636e-06
}



cat_params = {'iterations': 2000,
    'eval_metric': 'AUC',
    'verbose': 0,
    'task_type': 'CPU', # Change to 'GPU' if available
    'cat_features': [c for c in categorical_cols if c in X_cat.columns],
 'learning_rate': 0.08141363864155182, 
 'depth': 4, 
 'l2_leaf_reg': 2.721242066354407, 
 'random_strength': 0.3197413721687479, 
 'subsample': 0.8585190651619243}


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF (Out of Fold) Predictions for CV scoring
oof_xgb = np.zeros(len(X_le))
oof_lgb = np.zeros(len(X_le))
oof_cat = np.zeros(len(X_cat))

# Test Predictions
pred_xgb = np.zeros(len(X_test_le))
pred_lgb = np.zeros(len(X_test_le))
pred_cat = np.zeros(len(X_test_cat))

print(f"\n{'='*20} Starting Cross-Validation {'='*20}")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_le, target)):
    
    # --- XGBoost & LightGBM (Use Label Encoded Data) ---
    X_tr_le, X_val_le = X_le.iloc[train_idx], X_le.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le, y_val)], verbose=500)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_le)[:, 1]
    pred_xgb += model_xgb.predict_proba(X_test_le)[:, 1] / 5
    
    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val_le)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test_le)[:, 1] / 5
    
    # --- CatBoost (Use Native Data) ---
    X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]
    pred_cat += model_cat.predict_proba(X_test_cat)[:, 1] / 5
    
    print(f"Fold {fold+1} done.")





# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# from sklearn.preprocessing import StandardScaler

# print(f"\n{'='*20} TRAINING NEURAL NETWORK (on X_nn_processed) {'='*20}")


# # --- 2. DEFINE THE MODEL ---
# def make_model(input_shape):
#     inputs = layers.Input(shape=input_shape)
    
#     x = layers.Dense(256, activation='swish')(inputs) # Increased size due to One-Hot
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.3)(x)
    
#     x = layers.Dense(128, activation='swish')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.2)(x)
    
#     x = layers.Dense(64, activation='swish')(x)
#     x = layers.BatchNormalization()(x)
    
#     outputs = layers.Dense(1, activation='sigmoid')(x)
    
#     model = models.Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
#     return model

# # --- 3. TRAIN LOOP ---
# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# oof_nn = np.zeros(len(X_nn_processed))
# pred_nn = np.zeros(len(X_test_nn_processed))

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_nn_processed, y_nn)):
#     X_tr, X_val = X_nn_processed[train_idx], X_nn_processed[val_idx]
#     y_tr, y_val = y_nn.iloc[train_idx], y_nn.iloc[val_idx]
    
#     model = make_model((X_nn_processed.shape[1],))
    
#     es = callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
#     lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, verbose=0, mode='max')
    
#     model.fit(X_tr, y_tr, 
#               validation_data=(X_val, y_val), 
#               epochs=60, 
#               batch_size=1024, 
#               callbacks=[es, lr], 
#               verbose=0)
    
#     # Predict
#     val_preds = model.predict(X_val, batch_size=1024, verbose=0).flatten()
#     oof_nn[val_idx] = val_preds
    
#     test_preds = model.predict(X_test_nn_processed, batch_size=1024, verbose=0).flatten()
#     pred_nn += test_preds / n_splits
    
#     print(f"NN Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")

# print(f"Overall NN AUC: {roc_auc_score(y_nn, oof_nn):.5f}")


# corr_df = pd.DataFrame({
#     'XGB': oof_xgb, 'LGB': oof_lgb, 
#     'CAT': oof_cat, 'NN': oof_nn
# }).corr()
# print(corr_df)


# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# from sklearn.preprocessing import StandardScaler

# print(f"\n{'='*20} TRAINING NEURAL NETWORK 2 (on (X_nn_processed) {'='*20}")


# # --- 2. DEFINE THE MODEL ---
# def make_model_v2(input_shape):
#     inputs = layers.Input(shape=input_shape)
    
#     # Wider network
#     x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('swish')(x)
#     x = layers.Dropout(0.4)(x)
    
#     x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('swish')(x)
#     x = layers.Dropout(0.3)(x)
    
#     x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('swish')(x)
#     x = layers.Dropout(0.2)(x)
    
#     outputs = layers.Dense(1, activation='sigmoid')(x)
    
#     model = models.Model(inputs=inputs, outputs=outputs)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#         loss='binary_crossentropy', 
#         metrics=['AUC']
#     )
#     return model

# # --- 3. TRAIN LOOP ---
# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# oof_nn = np.zeros(len(X_nn_processed))
# pred_nn = np.zeros(len(X_test_nn_processed))

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_nn_processed, y_nn)):
#     X_tr, X_val = X_nn_processed[train_idx], X_nn_processed[val_idx]
#     y_tr, y_val = y_nn.iloc[train_idx], y_nn.iloc[val_idx]
    
#     model = make_model_v2((X_nn_processed.shape[1],))
    
#     # Training settings
#     es = callbacks.EarlyStopping(monitor='val_auc', patience=25, mode='max', restore_best_weights=True)
#     lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, verbose=0, min_lr=1e-6,mode='max')


#     model.fit(X_tr, y_tr, 
#               validation_data=(X_val, y_val), 
#               epochs=150,  # More room to learn
#               batch_size=256,  # Smaller batches
#               callbacks=[es, lr], 
#               verbose=0)
    
#     # Predict
#     val_preds = model.predict(X_val, batch_size=1024, verbose=0).flatten()
#     oof_nn[val_idx] = val_preds
    
#     test_preds = model.predict(X_test_nn_processed, batch_size=1024, verbose=0).flatten()
#     pred_nn += test_preds / n_splits
    
#     print(f"NN Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")

# print(f"Overall NN AUC: {roc_auc_score(y_nn, oof_nn):.5f}")


# corr_df = pd.DataFrame({
#     'XGB': oof_xgb, 'LGB': oof_lgb, 
#     'CAT': oof_cat, 'NN': oof_nn
# }).corr()
# print(corr_df)


# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# from sklearn.preprocessing import StandardScaler

# print(f"\n{'='*20} TRAINING NEURAL NETWORK 3 (on (X_nn_processed) {'='*20}")


# # --- 2. DEFINE THE MODEL ---
# def make_model_v3(input_shape):
#     inputs = layers.Input(shape=input_shape)
    
#     # Keep original width, less aggressive regularization
#     x = layers.Dense(256, activation='swish')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.25)(x)  # Reduced
    
#     x = layers.Dense(128, activation='swish')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.15)(x)  # Reduced
    
#     x = layers.Dense(64, activation='swish')(x)
#     x = layers.BatchNormalization()(x)
    
#     outputs = layers.Dense(1, activation='sigmoid')(x)
    
#     model = models.Model(inputs=inputs, outputs=outputs)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),  # Slightly lower than default
#         loss='binary_crossentropy', 
#         metrics=['AUC']
#     )
#     return model

# # --- 3. TRAIN LOOP ---
# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# oof_nn = np.zeros(len(X_nn_processed))
# pred_nn = np.zeros(len(X_test_nn_processed))

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_nn_processed, y_nn)):
#     X_tr, X_val = X_nn_processed[train_idx], X_nn_processed[val_idx]
#     y_tr, y_val = y_nn.iloc[train_idx], y_nn.iloc[val_idx]
    
#     model = make_model_v3((X_nn_processed.shape[1],))
    
#     # Training settings
#     es = callbacks.EarlyStopping(monitor='val_auc', patience=20, mode='max', restore_best_weights=True)
#     lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=7, verbose=0, mode='max')

#     model.fit(X_tr, y_tr, 
#               validation_data=(X_val, y_val), 
#               epochs=100,
#               batch_size=512,  # Middle ground
#               callbacks=[es, lr], 
#               verbose=0)
    
#     # Predict
#     val_preds = model.predict(X_val, batch_size=1024, verbose=0).flatten()
#     oof_nn[val_idx] = val_preds
    
#     test_preds = model.predict(X_test_nn_processed, batch_size=1024, verbose=0).flatten()
#     pred_nn += test_preds / n_splits
    
#     print(f"NN Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")

# print(f"Overall NN AUC: {roc_auc_score(y_nn, oof_nn):.5f}")


# corr_df = pd.DataFrame({
#     'XGB': oof_xgb, 'LGB': oof_lgb, 
#     'CAT': oof_cat, 'NN': oof_nn
# }).corr()
# print(corr_df)



# from scipy.optimize import minimize

# # --- 4. OPTIMIZED WEIGHTED BLEND (4 Models) ---
# print(f"\n{'='*20} FINDING OPTIMAL BLEND WEIGHTS {'='*20}")

# # Define the function we want to minimize (Negative AUC)
# def minimize_auc(weights):
#     # Normalize weights so they sum to 1.0
#     w = np.abs(weights)
#     w = w / w.sum()
    
#     # Create the blended OOF prediction (NOW WITH NN)
#     final_oof = (w[0] * oof_xgb + 
#                  w[1] * oof_lgb + 
#                  w[2] * oof_cat + 
#                  w[3] * oof_nn)  # <--- Added Neural Net
    
#     # Return negative AUC
#     return -roc_auc_score(target, final_oof)

# # Starting guess: Equal weights [0.25, 0.25, 0.25, 0.25]
# initial_weights = [0.25, 0.25, 0.25, 0.25]
# bounds = [(0, 1)] * 4  # 4 bounds for 4 models

# # Run the mathematical optimization
# result = minimize(minimize_auc, initial_weights, bounds=bounds, method='SLSQP')

# # Extract the best weights
# opt_weights = np.abs(result.x) / np.abs(result.x).sum()
# best_auc_optimized = -result.fun

# print(f"Optimal Weights Found:")
# print(f"  XGBoost:    {opt_weights[0]:.4f}")
# print(f"  LightGBM:   {opt_weights[1]:.4f}")
# print(f"  CatBoost:   {opt_weights[2]:.4f}")
# print(f"  Neural Net: {opt_weights[3]:.4f}")
# print(f"Optimized Blend AUC: {best_auc_optimized:.6f}")

# # Calculate the final Optimized Test Predictions
# pred_optimized = (opt_weights[0] * pred_xgb + 
#                   opt_weights[1] * pred_lgb + 
#                   opt_weights[2] * pred_cat + 
#                   opt_weights[3] * pred_nn)

# # --- 5. COMPARISON & SELECTION ---

# # 1. Single Model Scores
# auc_xgb = roc_auc_score(target, oof_xgb)
# auc_lgb = roc_auc_score(target, oof_lgb)
# auc_cat = roc_auc_score(target, oof_cat)
# auc_nn  = roc_auc_score(target, oof_nn)  # <--- Added NN

# # 2. Probability Blend (Simple Average of 4)
# oof_blend_prob = (oof_xgb + oof_lgb + oof_cat + oof_nn) / 4
# auc_blend_prob = roc_auc_score(target, oof_blend_prob)

# # 3. Rank Blend (Average of Ranks of 4)
# oof_blend_rank = (pd.Series(oof_xgb).rank(pct=True) + 
#                   pd.Series(oof_lgb).rank(pct=True) + 
#                   pd.Series(oof_cat).rank(pct=True) + 
#                   pd.Series(oof_nn).rank(pct=True)) / 4
# auc_blend_rank = roc_auc_score(target, oof_blend_rank)

# # Store everything in the results dictionary
# results = {
#     'XGBoost': {'score': auc_xgb, 'preds': pred_xgb},
#     'LightGBM': {'score': auc_lgb, 'preds': pred_lgb},
#     'CatBoost': {'score': auc_cat, 'preds': pred_cat},
#     'NeuralNet': {'score': auc_nn, 'preds': pred_nn},
    
#     'Prob_Blend': {
#         'score': auc_blend_prob, 
#         'preds': (pred_xgb + pred_lgb + pred_cat + pred_nn) / 4
#     },
    
#     'Rank_Blend': {
#         'score': auc_blend_rank, 
#         'preds': (pd.Series(pred_xgb).rank(pct=True) + 
#                   pd.Series(pred_lgb).rank(pct=True) + 
#                   pd.Series(pred_cat).rank(pct=True) + 
#                   pd.Series(pred_nn).rank(pct=True)) / 4
#     },
    
#     'Optimized_Blend': {'score': best_auc_optimized, 'preds': pred_optimized}
# }

# # Print Summary
# print(f"\n{'='*10} FINAL SCOREBOARD {'='*10}")
# print(f"XGBoost:         {auc_xgb:.6f}")
# print(f"LightGBM:        {auc_lgb:.6f}")
# print(f"CatBoost:        {auc_cat:.6f}")
# print(f"Neural Net:      {auc_nn:.6f}")
# print(f"Simple Average:  {auc_blend_prob:.6f}")
# print(f"Rank Blend:      {auc_blend_rank:.6f}")
# print(f"Optimized Blend: {best_auc_optimized:.6f}  <-- EXPECTED WINNER")
# print(f"{'='*40}")



# from scipy.optimize import minimize

# # --- 4. OPTIMIZED WEIGHTED BLEND (3 Models) ---
# print(f"\n{'='*20} FINDING OPTIMAL BLEND WEIGHTS {'='*20}")

# # Define the function we want to minimize (Negative AUC)
# def minimize_auc(weights):
#     # Normalize weights so they sum to 1.0
#     w = np.abs(weights)
#     w = w / w.sum()
    
#     # Create the blended OOF prediction (NOW WITH NN)
#     final_oof = (w[0] * oof_xgb + 
#                  w[1] * oof_lgb + 
#                  w[2] * oof_cat ) 
#                  #w[3] * oof_nn)  # <--- Added Neural Net
    
#     # Return negative AUC
#     return -roc_auc_score(target, final_oof)

# # Starting guess: Equal weights [0.25, 0.25, 0.25, 0.25]
# initial_weights = [0.33, 0.33, 0.33]
# bounds = [(0, 1)] * 3  # 3 bounds for 3 models

# # Run the mathematical optimization
# result = minimize(minimize_auc, initial_weights, bounds=bounds, method='SLSQP')

# # Extract the best weights
# opt_weights = np.abs(result.x) / np.abs(result.x).sum()
# best_auc_optimized = -result.fun

# print(f"Optimal Weights Found:")
# print(f"  XGBoost:    {opt_weights[0]:.4f}")
# print(f"  LightGBM:   {opt_weights[1]:.4f}")
# print(f"  CatBoost:   {opt_weights[2]:.4f}")
# #print(f"  Neural Net: {opt_weights[3]:.4f}")
# print(f"Optimized Blend AUC: {best_auc_optimized:.6f}")

# # Calculate the final Optimized Test Predictions
# pred_optimized = (opt_weights[0] * pred_xgb + 
#                   opt_weights[1] * pred_lgb + 
#                   opt_weights[2] * pred_cat ) 
#                   #opt_weights[3] * pred_nn)

# # --- 5. COMPARISON & SELECTION ---

# # 1. Single Model Scores
# auc_xgb = roc_auc_score(target, oof_xgb)
# auc_lgb = roc_auc_score(target, oof_lgb)
# auc_cat = roc_auc_score(target, oof_cat)
# #auc_nn  = roc_auc_score(target, oof_nn)  # <--- Added NN

# # 2. Probability Blend (Simple Average of 3)
# oof_blend_prob = (oof_xgb + oof_lgb + oof_cat) / 3
# auc_blend_prob = roc_auc_score(target, oof_blend_prob)

# # 3. Rank Blend (Average of Ranks of 3)
# oof_blend_rank = (pd.Series(oof_xgb).rank(pct=True) + 
#                   pd.Series(oof_lgb).rank(pct=True) + 
#                   pd.Series(oof_cat).rank(pct=True)) / 3 
# auc_blend_rank = roc_auc_score(target, oof_blend_rank)

# # Store everything in the results dictionary
# results = {
#     'XGBoost': {'score': auc_xgb, 'preds': pred_xgb},
#     'LightGBM': {'score': auc_lgb, 'preds': pred_lgb},
#     'CatBoost': {'score': auc_cat, 'preds': pred_cat},
    
#     'Prob_Blend': {
#         'score': auc_blend_prob, 
#         'preds': (pred_xgb + pred_lgb + pred_cat) / 3
#     },
    
#     'Rank_Blend': {
#         'score': auc_blend_rank, 
#         'preds': (pd.Series(pred_xgb).rank(pct=True) + 
#                   pd.Series(pred_lgb).rank(pct=True) + 
#                   pd.Series(pred_cat).rank(pct=True)) / 3
#     },
    
#     'Optimized_Blend': {'score': best_auc_optimized, 'preds': pred_optimized}
# }

# # Print Summary
# print(f"\n{'='*10} FINAL SCOREBOARD {'='*10}")
# print(f"XGBoost:         {auc_xgb:.6f}")
# print(f"LightGBM:        {auc_lgb:.6f}")
# print(f"CatBoost:        {auc_cat:.6f}")
# print(f"Simple Average:  {auc_blend_prob:.6f}")
# print(f"Rank Blend:      {auc_blend_rank:.6f}")
# print(f"Optimized Blend: {best_auc_optimized:.6f}  <-- EXPECTED WINNER")
# print(f"{'='*40}")


from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# --- BLENDING STRATEGY 1: SCIPY OPTIMIZER (Gradient Descent) ---
print(f"\n{'='*20} STRATEGY 1: SCIPY MINIMIZE {'='*20}")

def minimize_auc(weights):
    # Normalize weights so they sum to 1.0
    w = np.abs(weights)
    w = w / w.sum()
    
    # Create the blended OOF prediction (3 Models)
    final_oof = (w[0] * oof_xgb + 
                 w[1] * oof_lgb + 
                 w[2] * oof_cat)
    
    return -roc_auc_score(target, final_oof)

# Starting guess: Equal weights
initial_weights = [0.33, 0.33, 0.33]
bounds = [(0, 1)] * 3

# Run optimization
result = minimize(minimize_auc, initial_weights, bounds=bounds, method='SLSQP')

# Extract results
weights_scipy = np.abs(result.x) / np.abs(result.x).sum()
auc_scipy = -result.fun
pred_scipy = (weights_scipy[0] * pred_xgb + 
              weights_scipy[1] * pred_lgb + 
              weights_scipy[2] * pred_cat)

print(f"Scipy Weights: XGB: {weights_scipy[0]:.4f}, LGB: {weights_scipy[1]:.4f}, CAT: {weights_scipy[2]:.4f}")
print(f"Scipy Blend AUC: {auc_scipy:.6f}")


# --- 4. BLENDING STRATEGY 2: RANDOM SEARCH (Brute Force) ---
print(f"\n{'='*20} STRATEGY 2: RANDOM SEARCH {'='*20}")

# Lists for iteration
oof_list = [oof_xgb, oof_lgb, oof_cat]
pred_list = [pred_xgb, pred_lgb, pred_cat]

best_auc_random = 0
best_weights_random = [0.33, 0.33, 0.33]

# Try 5000 random combinations
np.random.seed(42) # Reproducibility
for i in range(5000):
    # Generate 3 random weights that sum to 1
    weights = np.random.dirichlet(np.ones(3), size=1)[0]
    
    # Fast weighted average
    current_oof = (weights[0] * oof_xgb + 
                   weights[1] * oof_lgb + 
                   weights[2] * oof_cat)
    
    current_auc = roc_auc_score(target, current_oof)
    
    if current_auc > best_auc_random:
        best_auc_random = current_auc
        best_weights_random = weights

pred_random = (best_weights_random[0] * pred_xgb + 
               best_weights_random[1] * pred_lgb + 
               best_weights_random[2] * pred_cat)

print(f"Random Weights: XGB: {best_weights_random[0]:.4f}, LGB: {best_weights_random[1]:.4f}, CAT: {best_weights_random[2]:.4f}")
print(f"Random Blend AUC: {best_auc_random:.6f}")


# --- 5. FINAL COMPARISON & SELECTION ---

# Single Model Scores
auc_xgb = roc_auc_score(target, oof_xgb)
auc_lgb = roc_auc_score(target, oof_lgb)
auc_cat = roc_auc_score(target, oof_cat)

# Simple Average
oof_simple = (oof_xgb + oof_lgb + oof_cat) / 3
auc_simple = roc_auc_score(target, oof_simple)

# Rank Blend
oof_rank = (pd.Series(oof_xgb).rank(pct=True) + 
            pd.Series(oof_lgb).rank(pct=True) + 
            pd.Series(oof_cat).rank(pct=True)) / 3
auc_rank = roc_auc_score(target, oof_rank)

# Store results
results = {
    'XGBoost': {'score': auc_xgb, 'preds': pred_xgb},
    'LightGBM': {'score': auc_lgb, 'preds': pred_lgb},
    'CatBoost': {'score': auc_cat, 'preds': pred_cat},
    'Simple_Avg': {'score': auc_simple, 'preds': (pred_xgb + pred_lgb + pred_cat) / 3},
    'Rank_Blend': {'score': auc_rank, 
                   'preds': (pd.Series(pred_xgb).rank(pct=True) + 
                             pd.Series(pred_lgb).rank(pct=True) + 
                             pd.Series(pred_cat).rank(pct=True)) / 3},
    'Scipy_Blend': {'score': auc_scipy, 'preds': pred_scipy},
    'Random_Blend': {'score': best_auc_random, 'preds': pred_random}
}

# Print Summary
print(f"\n{'='*10} FINAL SCOREBOARD {'='*10}")
print(f"XGBoost:       {auc_xgb:.6f}")
print(f"LightGBM:      {auc_lgb:.6f}")
print(f"CatBoost:      {auc_cat:.6f}")
print(f"Simple Avg:    {auc_simple:.6f}")
print(f"Rank Blend:    {auc_rank:.6f}")
print(f"Scipy Blend:   {auc_scipy:.6f}")
print(f"Random Blend:  {best_auc_random:.6f}")
print(f"{'='*40}")


# # 1. Prepare Data
# model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'Simple Avg', 'Rank Blend', 'Optimized Blend']
# model_scores = [auc_xgb, auc_lgb, auc_cat, auc_blend_prob, auc_blend_rank, best_auc_optimized]

# # Create DataFrame
# df_results = pd.DataFrame({'Model': model_names, 'AUC': model_scores})
# df_results = df_results.sort_values(by='AUC', ascending=True) # Sort for chart

# # 2. Setup Plot
# plt.figure(figsize=(12, 6))
# sns.set_style("whitegrid")

# # Create Custom Colors (Green for Winner, Grey for others)
# colors = ['#bdc3c7' if x < df_results['AUC'].max() else '#2ecc71' for x in df_results['AUC']]

# # 3. Draw Bar Chart
# ax = sns.barplot(x='AUC', y='Model', data=df_results, palette=colors)

# # 4. Add Labels
# for i, v in enumerate(df_results['AUC']):
#     # Place text slightly to the right of the bar
#     ax.text(v, i, f'  {v:.5f}', va='center', fontweight='bold', color='#2c3e50')

# # 5. Fine Tuning (Zoom in to see differences)
# min_score = min(model_scores)
# max_score = max(model_scores)
# margin = (max_score - min_score) * 0.2
# plt.xlim(min_score - margin, max_score + margin)

# plt.title('Leaderboard: Model Comparison (3-Model Ensemble)', fontsize=16, fontweight='bold', pad=20)
# plt.xlabel('ROC AUC Score', fontsize=12)
# plt.ylabel('')

# # Show
# plt.tight_layout()
# plt.show()


# --- VISUALIZATION ---

# Prepare Data
model_names = ['XGB', 'LGBM', 'CatBoost', 'Simple', 'Rank', 'Scipy', 'Random']
model_scores = [auc_xgb, auc_lgb, auc_cat, auc_simple, auc_rank, auc_scipy, best_auc_random]

# Create DataFrame
df_results = pd.DataFrame({'Model': model_names, 'AUC': model_scores})
df_results = df_results.sort_values(by='AUC', ascending=True)

# Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Highlight winner
colors = ['#bdc3c7' if x < df_results['AUC'].max() else '#2ecc71' for x in df_results['AUC']]

ax = sns.barplot(x='AUC', y='Model', data=df_results, palette=colors)

for i, v in enumerate(df_results['AUC']):
    ax.text(v, i, f' {v:.6f}', va='center', fontweight='bold', color='#2c3e50')

# Zoom x-axis
min_score = min(model_scores)
max_score = max(model_scores)
margin = (max_score - min_score) * 0.2
plt.xlim(min_score - margin, max_score + margin)

plt.title('Leaderboard: 3 models , Scipy , Random Search Blend', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('ROC AUC Score', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.show()



# # 1. Prepare Data
# model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'NeuralNet', 'Simple Avg', 'Rank Blend', 'Optimized Blend']
# model_scores = [auc_xgb, auc_lgb, auc_cat, auc_nn, auc_blend_prob, auc_blend_rank, best_auc_optimized]

# # Create DataFrame
# df_results = pd.DataFrame({'Model': model_names, 'AUC': model_scores})
# df_results = df_results.sort_values(by='AUC', ascending=True) # Sort for chart

# # 2. Setup Plot
# plt.figure(figsize=(12, 6))
# sns.set_style("whitegrid")

# # Create Custom Colors (Green for Winner, Grey for others)
# colors = ['#bdc3c7' if x < df_results['AUC'].max() else '#2ecc71' for x in df_results['AUC']]

# # 3. Draw Bar Chart
# ax = sns.barplot(x='AUC', y='Model', data=df_results, palette=colors)

# # 4. Add Labels
# for i, v in enumerate(df_results['AUC']):
#     # Place text slightly to the right of the bar
#     ax.text(v, i, f'  {v:.5f}', va='center', fontweight='bold', color='#2c3e50')

# # 5. Fine Tuning (Zoom in to see differences)
# min_score = min(model_scores)
# max_score = max(model_scores)
# margin = (max_score - min_score) * 0.2
# plt.xlim(min_score - margin, max_score + margin)

# plt.title('Leaderboard: Model Comparison (4-Model Ensemble)', fontsize=16, fontweight='bold', pad=20)
# plt.xlabel('ROC AUC Score', fontsize=12)
# plt.ylabel('')

# # Show
# plt.tight_layout()
# plt.show()


# --- SUBMISSION ---
# Automatically pick the best one
best_method = max(results, key=lambda x: results[x]['score'])
best_score = results[best_method]['score']
final_predictions = results[best_method]['preds']

print(f"\n✅ Best Strategy: {best_method} with CV: {best_score:.6f}")
print("Generating submission file...")

submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': np.clip(final_predictions, 0.001, 0.999) # Clip is good practice
})

submission.to_csv('submission.csv', index=False)
print("Saved to 'submission.csv'")



# corr_df = pd.DataFrame({
#     'XGB': oof_xgb, 'LGB': oof_lgb, 
#     'CAT': oof_cat, 'NN': oof_nn
# }).corr()
# print(corr_df)




