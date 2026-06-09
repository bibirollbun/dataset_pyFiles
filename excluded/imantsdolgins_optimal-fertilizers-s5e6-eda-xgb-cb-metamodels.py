import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')

# --- Style and Color Palette ---
sns.set_theme(style="whitegrid", palette="tab10")


# === Load Data ===
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# === Explore Train Data ===
print("\n â˜‘ï¸� Data Info:")
train_df.info()
print("\n â˜‘ï¸� Numerical Features Summary:")
display(train_df.describe())
print("\n â˜‘ï¸� First 5 Rows and Last 5 Rows of the Dataset:")
display(train_df)


# --- Create Feature Lists ---
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type']
target_variable = 'Fertilizer Name'

print("âœ… Setup complete. Feature lists are ready.")


# --- Target Variable Distribution ---
plt.figure(figsize=(12, 6))
sns.countplot(y=train_df[target_variable], order=train_df[target_variable].value_counts().index)
plt.title('Distribution of Fertilizer Name (Target Variable)', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Fertilizer Name', fontsize=12)
plt.tight_layout()
plt.show()

# --- Numerical Feature Distributions ---
print("\nâœ”ï¸� Plotting distributions for numerical features...")
fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, 20))
for i, feature in enumerate(numerical_features):
    sns.histplot(data=train_df, x=feature, ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {feature}', fontsize=14)
plt.tight_layout()
plt.show()

# --- Categorical Feature Distributions ---
print("\nâœ”ï¸� Plotting distributions for categorical features...")
fig, axes = plt.subplots(1, len(categorical_features), figsize=(15, 6))
for i, feature in enumerate(categorical_features):
    sns.countplot(x=train_df[feature], ax=axes[i], order = train_df[feature].value_counts().index)
    axes[i].set_title(f'Distribution of {feature}', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45) # Rotate labels for better readability
plt.tight_layout()
plt.show()


# --- Generate a statistical report for numerical features ---
print("="*40)
print("   Numerical Data Distribution Report")
print("="*40)

for feature in numerical_features:
    # Calculate skewness and kurtosis
    skew = train_df[feature].skew()
    kurt = train_df[feature].kurt()
    
    print(f"\nFeature: {feature}")
    print("-" * (len(feature) + 10))
    print(f"  Skewness: {skew:.4f}")
    print(f"  Kurtosis: {kurt:.4f}")

    if abs(skew) < 0.5:
        print("  Interpretation: The distribution is approximately symmetric.")
    elif skew > 0.5:
        print("  Interpretation: The distribution is moderately skewed to the right.")
    else:
        print("  Interpretation: The distribution is moderately skewed to the left.")
        
print("\n" + "="*40)
print("           End of Report")
print("="*40)


# --- Plotting Numerical Features vs. Target Variable ---
print("âœ”ï¸� Plotting relationships between numerical features and the target variable...")
for feature in numerical_features:
    plt.figure(figsize=(12, 8))
    sns.boxplot(y=train_df[target_variable], x=train_df[feature])
    plt.title(f'{feature} by Fertilizer Name', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Fertilizer Name', fontsize=12)
    plt.tight_layout()
    plt.show()


# --- Plotting Categorical Features vs. Target Variable ---
print("\nâœ”ï¸�  Plotting relationships between categorical features and the target variable...")
for feature in categorical_features:
    crosstab = pd.crosstab(train_df[feature], train_df[target_variable])
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    crosstab_normalized.plot(kind='bar', stacked=True, figsize=(14, 8),
                             colormap='tab20')
    
    plt.title(f'Proportion of Fertilizer Name by {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


print("ğŸ“Š Plotting relationships between categorical features and target variable using heatmaps...")

for feature in categorical_features:
    
    # Create the cross-tabulation of counts
    crosstab_counts = pd.crosstab(train_df[feature], train_df[target_variable])
    
    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(crosstab_counts, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
    
    plt.title(f'Counts of Fertilizer Name by {feature}', fontsize=16)
    plt.xlabel('Fertilizer Name', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --- 2. Heatmap of Proportions (Normalized) ---
    crosstab_normalized = crosstab_counts.div(crosstab_counts.sum(axis=1), axis=0)
    
    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(crosstab_normalized, annot=True, fmt=".1%", cmap="Blues", linewidths=.5)
    
    plt.title(f'Proportion of Fertilizer Name by {feature}', fontsize=16)
    plt.xlabel('Fertilizer Name', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --- Correlation Heatmap for Numerical Features ---
print("\nâœ”ï¸� Calculating and plotting the correlation matrix for numerical features...")
corr_matrix = train_df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()


# --- Report for Numerical Features vs. Target ---
print("="*60)
print("  â˜‘ï¸� Report 1: Summary of Numerical Features by Fertilizer Name")
print("="*60)

for feature in numerical_features:
    print(f"\n\n âœ”ï¸� --- Analysis for: {feature} ---")
    summary_table = train_df.groupby(target_variable)[feature].describe()
    summary_table['median'] = train_df.groupby(target_variable)[feature].median()
    summary_table = summary_table[['mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    print(summary_table)


# --- Report for Categorical Features vs. Target ---
print("\n\n" + "="*70)
print("  â˜‘ï¸� Report 2: Breakdown of Categorical Features by Fertilizer Name")
print("="*70)

for feature in categorical_features:
    print(f"\n\nâœ”ï¸� --- Analysis for: {feature} vs. {target_variable} ---")
    crosstab_counts = pd.crosstab(train_df[feature], train_df[target_variable])
    print("\nCounts Table:")
    print(crosstab_counts)
    crosstab_normalized = pd.crosstab(train_df[feature], train_df[target_variable], normalize='index')
    print(f"\nProportions Table (Percentage of Fertilizers per {feature}):")
    print(crosstab_normalized.mul(100).round(2).astype(str) + '%')


# --- Report for Numerical Feature Correlations ---
print("\n\n" + "="*65)
print("        â˜‘ï¸� Report 3: Correlation Matrix for Numerical Features")
print("="*65)

# Calculate the correlation matrix
corr_matrix = train_df[numerical_features].corr()

print("\nâœ”ï¸� Correlation Coefficients:")
print(corr_matrix.round(4))
print("\n\n" + "="*60)
print("                     ğŸ”š End of Bivariate Report")
print("="*60)


print("\nğŸš€ Applying feature engineering...")

# --- Interaction Feature ---
train_df['Soil_Crop_Interaction'] = train_df['Soil Type'] + '_' + train_df['Crop Type']
test_df['Soil_Crop_Interaction'] = test_df['Soil Type'] + '_' + test_df['Crop Type']

# --- Nutrient Ratios and Totals ---
for df in [train_df, test_df]:
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
    df['N_K_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1)
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1)
    df['Total_Nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']

# --- Binning Numerical Feature ---
bins = pd.qcut(train_df['Moisture'], q=4, retbins=True, duplicates='drop')[1]
num_bins = len(bins) - 1
labels = ['Low', 'Medium', 'High', 'Very High'][:num_bins]
for df in [train_df, test_df]:
    df['Moisture_Category'] = pd.cut(df['Moisture'], bins=bins, labels=labels, include_lowest=True)


print("ğŸš€ Applying one-hot encoding...")
categorical_cols = ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction', 'Moisture_Category']

n_train = len(train_df)
combined_df = pd.concat([train_df.drop('Fertilizer Name', axis=1), test_df], ignore_index=True)
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)
bool_cols = combined_df_encoded.select_dtypes(include=['bool']).columns
combined_df_encoded[bool_cols] = combined_df_encoded[bool_cols].astype(int)
train_df_encoded = combined_df_encoded.iloc[:n_train].copy()
test_df_encoded = combined_df_encoded.iloc[n_train:].copy()
train_df_encoded['Fertilizer Name'] = train_df['Fertilizer Name']


# --- 1. Prepare Data ---
X = train_df_encoded.drop(columns=['id', 'Fertilizer Name'])
y = train_df_encoded['Fertilizer Name']

# --- 2. Encode the Target Variable ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 3. Train the Random Forest "Probe" Model ---
print("âš™ï¸� Training the scikit-learn Random Forest model on the CPU...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y_encoded)
print("âœ… Training complete.")

# --- 4. Extract and Visualize Feature Importances ---
print("ğŸ�¨ Extracting and plotting feature importances...")

# Create a DataFrame of feature names and their importance scores
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
importances_sorted = importances.sort_values(by='Importance', ascending=False)

# --- Plotting and Printing ---
top_n = 50
plt.figure(figsize=(12, 15))
sns.barplot(x='Importance', y='Feature', data=importances_sorted.head(top_n))
plt.title(f'Top {top_n} Most Important Features (Scikit-learn RF)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(True)
plt.show()

# print(f"\nğŸ”� Top 50 most important features:")
# print(importances_sorted.head(50))


# --- 1. Final Data Preparation (X, y, X_test) ---
print("ğŸš€ Preparing final X, y, and X_test variables...")
X = train_df_encoded.drop(columns=['id', 'Fertilizer Name'])
y = train_df_encoded['Fertilizer Name']
X_test = test_df_encoded.drop(columns=['id'])
X_test = X_test[X.columns]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# --- 2. Competition Metric: Mean Average Precision @ k (MAP@k) ---
def mapk(y_true, y_pred_proba, k=3):
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    avg_precisions = []
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_labels = top_k_preds[i]
        if true_label in pred_labels:
            rank = np.where(pred_labels == true_label)[0][0] + 1
            precision_at_k = 1 / rank
            avg_precisions.append(precision_at_k)
        else:
            avg_precisions.append(0.0)
    return np.mean(avg_precisions)

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print("\nâœ… Setup complete. All data is ready.")


print("âš™ï¸� --- [Model 1/3] XGBoost Training & Evaluation (GPU Enabled) ---")

# --- Initialization ---
oof_preds_xgb = np.zeros((len(X), num_classes))
test_preds_xgb = np.zeros((len(X_test), num_classes)) # <<< ADDED for test set prediction
feature_importances_xgb = pd.DataFrame(index=X.columns)

# --- Cross-Validation Loop ---
for fold, (train_index, val_index) in enumerate(kf.split(X, y_encoded)):
    print(f"\nğŸš€ --- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # --- Model Training with Standard Hyperparameters ---
    model_xgb = xgb.XGBClassifier(
        tree_method='gpu_hist',
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model_xgb.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=100 # Show progress every 100 rounds
                 )

    # --- Predictions and Fold Evaluation ---
    val_preds_proba = model_xgb.predict_proba(X_val)
    fold_map3 = mapk(y_val, val_preds_proba, k=3)
    print(f"  âœ”ï¸� Fold {fold+1} MAP@3: {fold_map3:.6f}")

    # --- Store Predictions and Importances ---
    oof_preds_xgb[val_index] = val_preds_proba
    test_preds_xgb += model_xgb.predict_proba(X_test) # <<< ADDED for test set prediction
    
    f_scores = model_xgb.get_booster().get_score(importance_type='gain')
    fold_importances = pd.Series([f_scores.get(f, 0.) for f in X.columns], index=X.columns)
    feature_importances_xgb[f'fold_{fold+1}'] = fold_importances

# --- Average Test Predictions After Loop ---
test_preds_xgb /= N_SPLITS

# --- Overall Evaluation ---
map3_xgb = mapk(y_encoded, oof_preds_xgb, k=3)
print(f"\n\nâœ… XGBoost Overall OOF MAP@3 Score: {map3_xgb:.6f}")
print("-------------------------------------------------")

# --- Feature Importance Visualization and Printout ---
feature_importances_xgb['mean'] = feature_importances_xgb.mean(axis=1)
importances_sorted_xgb = feature_importances_xgb.sort_values(by='mean', ascending=False)

# Plotting
plt.figure(figsize=(12, 15))
sns.barplot(x='mean', y=importances_sorted_xgb.index[:50], data=importances_sorted_xgb.head(50))
plt.title('Top 50 Feature Importances (XGBoost by "Gain")', fontsize=16)
plt.xlabel('Average Importance Score (Gain)')
plt.ylabel('Feature')
plt.show()

# # Printing
# print("\n--- ğŸ”� Top 50 XGBoost Features (by Gain) ---")
# print(importances_sorted_xgb[['mean']].head(50))


print("âš™ï¸� --- [Model 2/3] LightGBM Training & Evaluation (GPU Enabled) ---")

# --- Initialization ---
oof_preds_lgbm = np.zeros((len(X), num_classes))
test_preds_lgbm = np.zeros((len(X_test), num_classes)) # <<< ADDED for test set prediction
feature_importances_lgbm = pd.DataFrame(index=X.columns)

# --- Cross-Validation Loop ---
for fold, (train_index, val_index) in enumerate(kf.split(X, y_encoded)):
    print(f"\nğŸš€ --- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # --- Model Training with Standard Hyperparameters ---
    model_lgbm = lgb.LGBMClassifier(
        device='gpu',
        objective='multiclass',
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=40,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model_lgbm.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   eval_metric='multi_logloss',
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)] # Show progress
                  )

    # --- Predictions and Fold Evaluation ---
    val_preds_proba = model_lgbm.predict_proba(X_val)
    fold_map3 = mapk(y_val, val_preds_proba, k=3)
    print(f"  âœ”ï¸� Fold {fold+1} MAP@3: {fold_map3:.6f}")

    # --- Store Predictions and Importances ---
    oof_preds_lgbm[val_index] = val_preds_proba
    test_preds_lgbm += model_lgbm.predict_proba(X_test) # <<< ADDED for test set prediction
    feature_importances_lgbm[f'fold_{fold+1}'] = model_lgbm.feature_importances_

# --- Average Test Predictions After Loop ---
test_preds_lgbm /= N_SPLITS

# --- Overall Evaluation ---
map3_lgbm = mapk(y_encoded, oof_preds_lgbm, k=3)
print(f"\n\nâœ… LightGBM Overall OOF MAP@3 Score: {map3_lgbm:.6f}")
print("--------------------------------------------------")

# --- Feature Importance Visualization and Printout ---
feature_importances_lgbm['mean'] = feature_importances_lgbm.mean(axis=1)
importances_sorted_lgbm = feature_importances_lgbm.sort_values(by='mean', ascending=False)

# Plotting
plt.figure(figsize=(12, 15))
sns.barplot(x='mean', y=importances_sorted_lgbm.index[:50], data=importances_sorted_lgbm.head(50))
plt.title('Top 50 Feature Importances (LightGBM)', fontsize=16)
plt.xlabel('Average Importance Score')
plt.ylabel('Feature')
plt.show()

# # Printing
# print("\n--- ğŸ”� Top 50 LightGBM Features ---")
# print(importances_sorted_lgbm[['mean']].head(50))


print("--- [Model 3/3] Complete CatBoost Workflow ---")

# --- Initialization ---
oof_preds_cat = np.zeros((len(X), num_classes))
test_preds_cat = np.zeros((len(X_test), num_classes)) # <<< THIS LINE CREATES THE VARIABLE
feature_importances_cat = pd.DataFrame(index=X.columns)

# --- Cross-Validation Loop ---
for fold, (train_index, val_index) in enumerate(kf.split(X, y_encoded)):
    print(f"\nğŸš€ --- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    model_cat = cat.CatBoostClassifier(
        task_type='GPU',
        loss_function='MultiClass',
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_state=42
    )
    
    model_cat.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=100
                 )

    # --- Predictions and Fold Evaluation ---
    val_preds_proba = model_cat.predict_proba(X_val)
    fold_map3 = mapk(y_val, val_preds_proba, k=3)
    print(f"  âœ”ï¸� Fold {fold+1} MAP@3: {fold_map3:.6f}")

    # --- Store Predictions and Importances ---
    oof_preds_cat[val_index] = val_preds_proba
    test_preds_cat += model_cat.predict_proba(X_test) # <<< THIS LINE FILLS THE VARIABLE
    feature_importances_cat[f'fold_{fold+1}'] = model_cat.get_feature_importance()

# --- Average Test Predictions After Loop ---
test_preds_cat /= N_SPLITS

# --- Overall Evaluation ---
map3_cat = mapk(y_encoded, oof_preds_cat, k=3)
print(f"\nâœ… CatBoost Overall OOF MAP@3 Score: {map3_cat:.6f}")
print("------------------------------------------------")

# --- Feature Importance Visualization and Printout ---
feature_importances_cat['mean'] = feature_importances_cat.mean(axis=1)
importances_sorted_cat = feature_importances_cat.sort_values(by='mean', ascending=False)

plt.figure(figsize=(12, 15))
sns.barplot(x='mean', y=importances_sorted_cat.index[:50], data=importances_sorted_cat.head(50))
plt.title('Top 50 Feature Importances (CatBoost)', fontsize=16)
plt.xlabel('Average Importance Score')
plt.ylabel('Feature')
plt.show()

# print("\n--- ğŸ”� Top 50 CatBoost Features ---")
# print(importances_sorted_cat[['mean']].head(50))


print("ğŸ”„ Generating XGBoost submission file...")

# Get top 3 predicted class indices
top3_preds_indices_xgb = np.argsort(test_preds_xgb, axis=1)[:, ::-1][:, :3]

# Convert indices back to fertilizer names
top3_preds_names_xgb = label_encoder.inverse_transform(top3_preds_indices_xgb.flatten()).reshape(-1, 3)

# Join names with a space for the submission format
submission_preds_xgb = [' '.join(preds) for preds in top3_preds_names_xgb]

# Create the submission DataFrame
submission_xgb = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': submission_preds_xgb})

# Save the file
submission_xgb.to_csv('submission_xgb.csv', index=False)
print("âœ… submission_xgb.csv created successfully!")


print("ğŸ”„ Generating LightGBM submission file...")

# Get top 3 predicted class indices
top3_preds_indices_lgbm = np.argsort(test_preds_lgbm, axis=1)[:, ::-1][:, :3]

# Convert indices back to fertilizer names
top3_preds_names_lgbm = label_encoder.inverse_transform(top3_preds_indices_lgbm.flatten()).reshape(-1, 3)

# Join names with a space for the submission format
submission_preds_lgbm = [' '.join(preds) for preds in top3_preds_names_lgbm]

# Create the submission DataFrame
submission_lgbm = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': submission_preds_lgbm})

# Save the file
submission_lgbm.to_csv('submission_lgbm.csv', index=False)
print("âœ… submission_lgbm.csv created successfully!")


print("ğŸ”„ Generating CatBoost submission file...")

# Get top 3 predicted class indices
top3_preds_indices_cat = np.argsort(test_preds_cat, axis=1)[:, ::-1][:, :3]

# Convert indices back to fertilizer names
top3_preds_names_cat = label_encoder.inverse_transform(top3_preds_indices_cat.flatten()).reshape(-1, 3)

# Join names with a space for the submission format
submission_preds_cat = [' '.join(preds) for preds in top3_preds_names_cat]

# Create the submission DataFrame
submission_cat = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': submission_preds_cat})

# Save the file
submission_cat.to_csv('submission_cat.csv', index=False)
print("âœ… submission_cat.csv created successfully!")


print("--- Creating Final Ensemble Submission File ---")

try:
    # --- 1. Simple Averaging Ensemble ---
    print("âš–ï¸�  Averaging test set predictions from XGBoost, LightGBM, and CatBoost...")
    ensemble_preds = (test_preds_xgb + test_preds_lgbm + test_preds_cat) / 3

    # --- 2. Format Predictions for Submission ---
    print("ğŸ”„  Formatting predictions into submission format...")
    
    # Get top 3 predicted class indices for each test sample
    top3_preds_indices = np.argsort(ensemble_preds, axis=1)[:, ::-1][:, :3]
    
    # Convert the indices back to their original string names
    top3_preds_names = label_encoder.inverse_transform(top3_preds_indices.flatten()).reshape(-1, 3)
    
    # Join the three fertilizer names with a space
    submission_preds = [' '.join(preds) for preds in top3_preds_names]

    # --- 3. Create and Save the Submission File ---
    submission_df = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': submission_preds})
    
    # Save the final file
    submission_df.to_csv('submission_ensemble.csv', index=False)
    
    print("\nâœ… submission_ensemble.csv created successfully!")

except NameError as e:
    print(f"\nâ�Œ Error: A required prediction variable is not in memory.")
    print("Please ensure you have successfully run the training cells for all three models (XGBoost, LightGBM, and CatBoost) first.")
    print(f"Details: {e}")


print("âœ… --- Evaluating Model Performance using Out-of-Fold (OOF) Predictions ---")

try:
    # --- Individual Model Scores ---
    print("\n--- Individual Model OOF Scores ---")
    
    # XGBoost Score
    map3_oof_xgb = mapk(y_encoded, oof_preds_xgb, k=3)
    print(f"âœ… XGBoost OOF MAP@3 Score: {map3_oof_xgb:.6f}")
    
    # LightGBM Score
    map3_oof_lgbm = mapk(y_encoded, oof_preds_lgbm, k=3)
    print(f"âœ… LightGBM OOF MAP@3 Score: {map3_oof_lgbm:.6f}")
    
    # CatBoost Score
    map3_oof_cat = mapk(y_encoded, oof_preds_cat, k=3)
    print(f"âœ… CatBoost OOF MAP@3 Score: {map3_oof_cat:.6f}")
    
    
    # --- Ensemble Score ---
    print("\n--- Ensemble OOF Score ---")
    
    # Create the ensemble from OOF predictions by averaging
    oof_ensemble_preds = (oof_preds_xgb + oof_preds_lgbm + oof_preds_cat) / 3
    
    # Calculate the ensemble's score
    map3_oof_ensemble = mapk(y_encoded, oof_ensemble_preds, k=3)
    print(f"ğŸ�† Ensemble OOF MAP@3 Score: {map3_oof_ensemble:.6f}")
    print("-----------------------------------")
    
except NameError as e:
    print(f"\nâ�Œ Error: OOF prediction variables not found.")
    print("Please re-run the model training cells to generate the OOF predictions first.")
    print(f"Details: {e}")




