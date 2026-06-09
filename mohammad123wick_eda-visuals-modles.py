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


# Import everything here to avoid repetation and confusion
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier # For imputation and adversarial validation
from sklearn.model_selection import KFold 
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping as lgb_early_stopping
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


import warnings

warnings.filterwarnings('ignore') 


# Load the data 
personality = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
personality.head()


# Info 
personality.info()


# Shape 
personality.shape


# null amount
personality.isnull().sum()


# Check the min,max,mean,count,top,freq,std,unique
personality.describe(include='all')


# Check the imbalance of the target 
summary = pd.DataFrame({'count': personality['Personality'].value_counts(), 'percentage': personality['Personality'].value_counts(normalize=True)})
summary


# Target Distribution Visual (Class Imbalance)
sns.countplot(data=personality, x='Personality', palette='pastel')
plt.title('Target Distribution (Personality)')
plt.show()


# Missing Value Heatmap
plt.figure(figsize=(18,6))
sns.heatmap(personality.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# Target Correlation with Features

df = personality.copy()
df.drop('id', axis=1, inplace=True)
binary_cols = ['Stage_fear','Drained_after_socializing']
for cols in binary_cols:
    df[cols] = df[cols].map({"Yes": 1, "No": 0})

df['target']= df['Personality'].map({'Extrovert':1, "Introvert": 0})

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only = True), annot=True, cmap="coolwarm")
plt.title("Correlation with Target")
plt.show()


# Number of features per row
num_cols = df.select_dtypes(include='number').columns.drop(['target'])
cols_per_row = 3
num_features = len(num_cols)
num_rows = int(np.ceil(num_features / cols_per_row))

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row*4, num_rows*3))
axes = axes.flatten()  

for i, col in enumerate(num_cols):
    ax = axes[i]
    
    sns.kdeplot(
        data=df[df['Personality'] == 'Extrovert'],
        x=col, fill=True, color='red', label='Extrovert', alpha=0.4, linewidth=2, ax=ax
    )
    sns.kdeplot(
        data=df[df['Personality'] == 'Introvert'],
        x=col, fill=True, color='blue', label='Introvert', alpha=0.4, linewidth=2, ax=ax
    )
    
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

# Remove empty subplots if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


target_variable = 'Personality'

numerical_features = personality.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = personality.select_dtypes(include=['object']).columns.tolist()


if 'id' in numerical_features:
    numerical_features.remove('id')

if target_variable in numerical_features:
    numerical_features.remove(target_variable)
if target_variable in categorical_features:
    categorical_features.remove(target_variable)


sns.set_style("whitegrid")

# Create density plots for numerical features
print("Generating density plots for numerical features...")
num_numerical_plots = len(numerical_features)
# Determine grid size for numerical plots
num_cols_numerical = 2
num_rows_numerical = (num_numerical_plots + num_cols_numerical - 1) // num_cols_numerical

plt.figure(figsize=(num_cols_numerical * 6, num_rows_numerical * 5))
plt.suptitle('Density Distributions of Numerical Features by Personality', y=1.02, fontsize=16)

for i, col in enumerate(numerical_features):
    plt.subplot(num_rows_numerical, num_cols_numerical, i + 1)
    sns.kdeplot(data=personality, x=col, hue=target_variable, fill=True, common_norm=False, palette='viridis', alpha=0.6)
    plt.title(f'Distribution of {col} by {target_variable}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
plt.show()

# Create count plots for categorical features
print("\nGenerating count plots for categorical features...")
num_categorical_plots = len(categorical_features)
# Determine grid size for categorical plots
num_cols_categorical = 2
num_rows_categorical = (num_categorical_plots + num_cols_categorical - 1) // num_cols_categorical

plt.figure(figsize=(num_cols_categorical * 7, num_rows_categorical * 5))
plt.suptitle('Count Distributions of Categorical Features by Personality', y=1.02, fontsize=16)

for i, col in enumerate(categorical_features):
    plt.subplot(num_rows_categorical, num_cols_categorical, i + 1)
    sns.countplot(data=personality, x=col, hue=target_variable, palette='viridis')
    plt.title(f'Count of {col} by {target_variable}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.tight_layout()
plt.show()

print("\nPlots generated successfully.")


# Boxplots / Stripplots (Feature vs Target)
num_cols = df.select_dtypes(include='number').columns.drop(['target', 'Stage_fear', 'Drained_after_socializing'])

for col in num_cols:
    sns.boxplot(data=df, x='Personality', y=col, palette='coolwarm')
    plt.title(f'{col} by Personality')
    plt.show()
    filename = f'boxplot_{col}.png' 
    plt.savefig(filename, dpi=300, bbox_inches='tight') # Save with high resolution and tight bounding box
    plt.close()


# Barplot for Binary Categorical Features
cat_cols = ['Stage_fear', 'Drained_after_socializing']

for col in cat_cols:
    sns.countplot(data=personality, x=col, hue='Personality', palette='pastel')
    plt.title(f'{col} by Personality')
    plt.show()


# Drop Unnecessary Columns
personality.drop('id', axis=1, inplace=True)


# Split target and features
X = personality.drop('Personality', axis=1)
y = personality['Personality']

# Train-test split
X_train, X_test, y_train_dummy, y_test_dummy = train_test_split(X, y, test_size=0.3, random_state=42)

# Identify columns with missing values
features_with_missing = X_train.columns[X_train.isnull().any()].tolist()

# Print counts
print("Missing values in X_train:\n", X_train[features_with_missing].isnull().sum())

# Prepare dictionary to store fitted imputers
fitted_imputers = {}



# We will use LGBMRegressor for numerical columns to impute the missing values
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in features_with_missing:
    if col in numerical_cols:
        # Prepare training data for imputer model
        notnull_mask = X_train[col].notnull()
        X_imputer_train = X_train.loc[notnull_mask].drop(columns=[col])
        y_imputer_train = X_train.loc[notnull_mask, col]
        
        # Remove non-numeric features before fitting
        X_imputer_train = X_imputer_train.select_dtypes(include=['int64', 'float64'])

        # Fit model
        model = LGBMRegressor(random_state=42)
        model.fit(X_imputer_train, y_imputer_train)

        # Predict on missing values
        null_mask = X_train[col].isnull()
        X_imputer_predict = X_train.loc[null_mask].drop(columns=[col])
        X_imputer_predict = X_imputer_predict.select_dtypes(include=['int64', 'float64'])

        preds = model.predict(X_imputer_predict)
        X_train.loc[null_mask, col] = preds

        # Save model
        fitted_imputers[col] = model



# We will use LGBMClassifier for categorical columns to impute the missing values
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in features_with_missing:
    if col in categorical_cols:
        notnull_mask = X_train[col].notnull()
        X_imputer_train = X_train.loc[notnull_mask].drop(columns=[col])
        y_imputer_train = X_train.loc[notnull_mask, col]

        # Remove non-numeric features from inputs
        X_imputer_train = X_imputer_train.select_dtypes(include=['int64', 'float64'])

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_imputer_train)

        model = LGBMClassifier(random_state=42)
        model.fit(X_imputer_train, y_encoded)

        null_mask = X_train[col].isnull()
        X_imputer_predict = X_train.loc[null_mask].drop(columns=[col])
        X_imputer_predict = X_imputer_predict.select_dtypes(include=['int64', 'float64'])

        preds = model.predict(X_imputer_predict)
        decoded_preds = le.inverse_transform(preds)
        X_train.loc[null_mask, col] = decoded_preds

        fitted_imputers[col] = (model, le)



print("Remaining NaNs in X_train:\n", X_train.isnull().sum())
print("\nRemaining NaNs in X_test:\n", X_test.isnull().sum())


# Fallback imputation (only if any values still missing)
for col in X_test.columns:
    if X_test[col].isnull().sum() > 0:
        if X_test[col].dtype == 'object':
            X_test[col].fillna(X_test[col].mode()[0], inplace=True)
        else:
            X_test[col].fillna(X_test[col].median(), inplace=True)



print("Remaining NaNs in X_train:\n", X_train.isnull().sum())
print("\nRemaining NaNs in X_test:\n", X_test.isnull().sum())


# Binary encode directly in the same column
personality['Personality'] = personality['Personality'].map({'Extrovert': 0, 'Introvert': 1})



categorical_cols = X_train.select_dtypes(include='object').columns

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le


# Ensure target variable is encoded (if not already done consistently)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_dummy)
y_test_encoded = le.transform(y_test_dummy)

# Get class weights for imbalance handling
# From your EDA: Extrovert (majority): 13699, Introvert (minority): 4825
neg_count = (y_train_dummy == 'Extrovert').sum()
pos_count = (y_train_dummy == 'Introvert').sum()
scale_pos_weight_value = neg_count / pos_count
print(f"Scale Pos Weight: {scale_pos_weight_value:.3f}")


# Define Hyperparameters for Base Models 

xgb_params = {
    'grow_policy': 'lossguide',
    'n_estimators': 503,
    'learning_rate': 0.017171776859474693,
    'gamma': 0.45588646381058495,
    'subsample': 0.6676074256540468,
    'colsample_bytree': 0.6252370621933254,
    'max_depth': 9,
    'min_child_weight': 7,
    'reg_lambda': 2.3374020331916116e-07, 
    'reg_alpha': 6.393724062860496e-06,   
    'random_state': 42,
    'booster': 'gbtree',
    'device': 'cpu', # Use 'cuda' or 'gpu' if you have a compatible NVIDIA GPU setup, else 'cpu' for general compatibility
    'verbosity': 0, 
    'tree_method': 'hist', 
    'eval_metric': 'auc',
    'objective': 'binary:logistic', 
    'scale_pos_weight': scale_pos_weight_value # Handle imbalance
}

lgbm_params = {
    'n_estimators': 310,
    'learning_rate': 0.010127141474319246,
    'max_depth': 14,
    'min_child_samples': 20,
    'subsample': 0.7677914267798617,
    'colsample_bytree': 0.5273274632656546,
    'num_leaves': 135,
    'random_state': 42,
    'verbose': -1, 
    'metric': 'auc', 
    'objective': 'binary',
    'scale_pos_weight': scale_pos_weight_value # Handle imbalance
    # Alternatively, you can use 'is_unbalance': True with 'objective': 'binary'
}

cat_params = {
    'iterations': 550, # Number of trees
    'learning_rate': 0.010072755990590003,
    'depth': 6,
    'l2_leaf_reg': 0.058001080943184846, 
    'random_state': 42,
    'verbose': False, 
    'eval_metric': 'AUC', 
    'objective': 'Logloss', 
    'auto_class_weights': 'Balanced', # Handle imbalance
   
}




# Define a list of models to train and evaluate
models = [
    ('XGBoost', XGBClassifier(**xgb_params)),
    ('LightGBM', LGBMClassifier(**lgbm_params)),
    ('CatBoost', CatBoostClassifier(**cat_params))
]

#  Prepare for Cross-Validation
N_SPLITS = 5 
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


for name, model in models:
    print(f"\n--- Evaluating {name} ---")

    fold_roc_auc_scores = []
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        print(f"  Fold {fold + 1}/{N_SPLITS}")
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_encoded[train_idx], y_train_encoded[val_idx]

        if name == 'CatBoost':
            cat_features_indices = [X_train.columns.get_loc(col) for col in X_train.select_dtypes(include='object').columns]
            model.fit(
                X_train_fold, y_train_fold,
                cat_features=cat_features_indices,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=50,
                verbose=False
            )
        elif name == 'LightGBM':
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[
                    lgb_early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0)  # No verbose
                ]
            )
        else:  # XGBoost
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=50,
                verbose=False
            )

        val_preds_proba = model.predict_proba(X_val_fold)[:, 1]
        val_preds_class = (val_preds_proba > 0.5).astype(int)

        fold_roc_auc = roc_auc_score(y_val_fold, val_preds_proba)
        fold_f1 = f1_score(y_val_fold, val_preds_class)

        fold_roc_auc_scores.append(fold_roc_auc)
        fold_f1_scores.append(fold_f1)

        print(f"    ROC AUC: {fold_roc_auc:.4f}, F1-Score: {fold_f1:.4f}")

    print(f"  Average ROC AUC: {np.mean(fold_roc_auc_scores):.4f} (+/- {np.std(fold_roc_auc_scores):.4f})")
    print(f"  Average F1-Score: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")

    print(f"  Training {name} on full X_train for final evaluation...")

    if name == 'CatBoost':
        model.fit(X_train, y_train_encoded, cat_features=cat_features_indices, verbose=False)
    elif name == 'LightGBM':
        model.fit(
            X_train, y_train_encoded,
            eval_metric='auc',  # Optional but good practice
            callbacks=[lgb.log_evaluation(0)]

          )
    else:
        model.fit(X_train, y_train_encoded, verbose=False)

    test_preds_proba = model.predict_proba(X_test)[:, 1]
    test_preds_class = (test_preds_proba > 0.5).astype(int)

    final_roc_auc = roc_auc_score(y_test_encoded, test_preds_proba)
    final_f1 = f1_score(y_test_encoded, test_preds_class)

    print(f"  {name} Final Test Set ROC AUC: {final_roc_auc:.4f}")
    print(f"  {name} Final Test Set F1-Score: {final_f1:.4f}")
    print("  Confusion Matrix on Test Set:\n", confusion_matrix(y_test_encoded, test_preds_class))
    print("  Classification Report on Test Set:\n", classification_report(y_test_encoded, test_preds_class, target_names=le.classes_))


# Load the raw test data and sample submission file
raw_test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Store test IDs for submission before dropping 'id'
test_ids = raw_test_df['id']

# Create a working copy of the test DataFrame
test_df_processed = raw_test_df.copy()
test_df_processed.drop('id', axis=1, inplace=True) # Drop 'id' from test_df

# Preprocessing the Test Set

# Apply fitted imputers from training phase to test_df_processed
for col in test_df_processed.columns:
    if test_df_processed[col].isnull().sum() > 0: 
        if col in fitted_imputers:
            imputer_info = fitted_imputers[col]
            null_mask = test_df_processed[col].isnull()
            X_imputer_predict_test = test_df_processed.loc[null_mask].drop(columns=[col])

            if isinstance(imputer_info, tuple):
                model, le_imputer = imputer_info
                X_imputer_predict_test_numeric = X_imputer_predict_test.select_dtypes(include=np.number)
                # Align columns, assuming feature names are consistent from model.feature_name_
                X_imputer_predict_test_numeric = X_imputer_predict_test_numeric[model.feature_name_] 
                preds = model.predict(X_imputer_predict_test_numeric)
                decoded_preds = le_imputer.inverse_transform(preds)
                test_df_processed.loc[null_mask, col] = decoded_preds
            else: 
                model = imputer_info
                X_imputer_predict_test_numeric = X_imputer_predict_test.select_dtypes(include=np.number)
                X_imputer_predict_test_numeric = X_imputer_predict_test_numeric[model.feature_name_]
                preds = model.predict(X_imputer_predict_test_numeric)
                test_df_processed.loc[null_mask, col] = preds
        else:
            
            if test_df_processed[col].dtype == 'object':
                test_df_processed[col].fillna(test_df_processed[col].mode()[0], inplace=True)
            else:
                test_df_processed[col].fillna(test_df_processed[col].median(), inplace=True)

# Apply fitted LabelEncoders to categorical columns in test_df_processed
categorical_cols_test = test_df_processed.select_dtypes(include='object').columns
for col in categorical_cols_test:
    if col in encoders:
        le_fitted = encoders[col]
        test_df_processed[col] = le_fitted.transform(test_df_processed[col].astype(str))
    else:
        # This case implies a categorical column appeared in test_df but not in X_train with NaNs, 
        # or was not recognized as categorical during X_train processing.
        # For robustness, fit a new encoder if absolutely necessary, but ideally ensure consistency.
        print(f"Warning: No fitted encoder for test column '{col}'. Fitting a new one (ensure consistency!).")
        new_le = LabelEncoder()
        test_df_processed[col] = new_le.fit_transform(test_df_processed[col].astype(str))
        encoders[col] = new_le # Store it for future reference if needed

# Ensure column order matches X_train if any reordering happened during preprocessing
test_df_processed = test_df_processed[X_train.columns]

print("Test set preprocessing complete. No remaining NaNs:", test_df_processed.isnull().sum().sum() == 0)


#  Individual Model Predictions for Submission ---

print("\n--- Generating Individual Model Predictions for Submission ---")


fitted_xgb_model = XGBClassifier(**xgb_params)
fitted_lgbm_model = LGBMClassifier(**lgbm_params)
fitted_cat_model = CatBoostClassifier(**cat_params)

# Quick fit for demonstration (assuming full X_train training took place conceptually)
fitted_xgb_model.fit(X_train, y_train_encoded, verbose=False)
fitted_lgbm_model.fit(X_train, y_train_encoded)
cat_features_indices = [X_train.columns.get_loc(col) for col in X_train.select_dtypes(include=['object']).columns]
fitted_cat_model.fit(X_train, y_train_encoded, cat_features=cat_features_indices, verbose=False)


individual_fitted_models = [
    ('XGBoost', fitted_xgb_model),
    ('LightGBM', fitted_lgbm_model),
    ('CatBoost', fitted_cat_model)
]

for name, model in individual_fitted_models:
    print(f"Predicting with {name}...")
    
    # Predict probabilities for the positive class (Introvert)
    test_predictions_proba = model.predict_proba(test_df_processed)[:, 1]

    # Convert probabilities to class labels (0 or 1) using a 0.5 threshold
    test_predictions_class_encoded = (test_predictions_proba > 0.5).astype(int)
    
    # Convert encoded labels back to original string labels ('Extrovert'/'Introvert')
    test_predictions_original_labels = le.inverse_transform(test_predictions_class_encoded)

    # Create submission DataFrame
    submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions_original_labels})

    # Save submission file
    submission_filename = f'submission_individual_{name.lower().replace(" ", "_")}.csv'
    submission_df.to_csv(submission_filename, index=False)
    print(f"  Submission file '{submission_filename}' created.")



print("\n--- Generating Stacking Model Predictions ---")

categorical_column_names = X_train.select_dtypes(include=['object']).columns.tolist()


original_cat_cols_for_catboost = ['Stage_fear', 'Drained_after_socializing'] 
cat_features_indices_for_catboost = [X_train.columns.get_loc(col) for col in original_cat_cols_for_catboost]


estimators = [
    ('xgb', XGBClassifier(**xgb_params)),
    ('lgbm', LGBMClassifier(**lgbm_params)), 
    ('cat', CatBoostClassifier(**cat_params, cat_features=cat_features_indices_for_catboost)) 
]

# Initialize StackingClassifier
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1,
    verbose=0 
)

print("\n--- Training StackingClassifier ---")

# Fit the StackingClassifier on the full X_train data
# Now, StackingClassifier.fit() will NOT need 'cat_features' as it's set in the CatBoost constructor.
stacked_model.fit(X_train, y_train_encoded) # Removed cat_features=... from here

# --- Generate Final Stacking Predictions ---
print("\n--- Generating final stacked predictions on preprocessed test_df ---")
final_stacked_preds_proba = stacked_model.predict_proba(test_df_processed)[:, 1]
final_stacked_preds_class_encoded = (final_stacked_preds_proba > 0.5).astype(int)
final_stacked_preds_original_labels = le.inverse_transform(final_stacked_preds_class_encoded)

# Create Stacking Submission DataFrame
stacked_submission_df = pd.DataFrame({'id': test_ids, 'Personality': final_stacked_preds_original_labels})

# Save Stacking Submission file
stacked_submission_filename = 'submission_stacked_model.csv'
stacked_submission_df.to_csv(stacked_submission_filename, index=False)
print(f"Stacked submission file '{stacked_submission_filename}' created successfully.")

print("\nAll tasks complete!")


# Prepare base features
X_train_fe = X_train.copy()
X_test_fe = test_df_processed.copy()

# 1. Scale numerical features for PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)  # ← FIXED

# 2. PCA with mle
pca = PCA(n_components='mle', svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)  # ← FIXED

# 3. Convert to DataFrames (auto-detect how many components PCA kept)
pca_cols = [f'pca_{i}' for i in range(X_train_pca.shape[1])]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train_fe.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test_fe.index)

# 4. Add PCA features to original
X_train_fe = pd.concat([X_train_fe.reset_index(drop=True), X_train_pca_df.reset_index(drop=True)], axis=1)
X_test_fe = pd.concat([X_test_fe.reset_index(drop=True), X_test_pca_df.reset_index(drop=True)], axis=1)




xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    verbose=0,
    random_state=42
)

xgb_model.fit(X_train_fe, y_train_encoded)
lgb_model.fit(X_train_fe, y_train_encoded)
cat_model.fit(X_train_fe, y_train_encoded)


from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgb_model),
        ('cat', cat_model)
    ],
    voting='soft'
)

voting_clf.fit(X_train_fe, y_train_encoded)





# --- Generate Submission File for VotingClassifier ---
print("\n--- Generating submission file for VotingClassifier ---")

# Predict probabilities for the positive class (Introvert) using the fitted voting_clf
voting_test_predictions_proba = voting_clf.predict_proba(X_test_fe)[:, 1]

# Convert probabilities to class labels (0 or 1) using a 0.5 threshold
voting_test_predictions_class_encoded = (voting_test_predictions_proba > 0.5).astype(int)

# Convert encoded labels back to original string labels ('Extrovert'/'Introvert')
# 'le' is the LabelEncoder fitted on y_train_dummy.
voting_test_predictions_original_labels = le.inverse_transform(voting_test_predictions_class_encoded)

# Create submission DataFrame using the stored test_ids
submission_df_voting = pd.DataFrame({'id': test_ids, 'Personality': voting_test_predictions_original_labels})

# Define the submission filename
submission_filename_voting = 'submission_voting_classifier_pca.csv'

# Save submission file
submission_df_voting.to_csv(submission_filename_voting, index=False)

print(f"Submission file '{submission_filename_voting}' created successfully.")

print("\nSubmission task complete!")

