# ğŸ“¦ Core libraries
import pandas as pd
import numpy as np
import warnings

# ğŸ“Š Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# âš™ï¸� Scikit-learn: model selection, preprocessing, metrics
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ğŸ“ˆ Gradient-boosting libraries
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb

# ğŸ§ª SHAP for model interpretation
import shap

# ğŸ”§ Settings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# ğŸ“Œ Jupyter magic
%matplotlib inline


# ğŸ“¥ Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")

# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

# original['dataset'] = 'train'

# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# ğŸ§¾ Display dataset shape
print("Dataset shape:", df.shape)

# ğŸ‘�ï¸� Preview the data
df.head()


# original


train


test


df.shape


# ğŸ“‹ Check column types and non-null counts
df.info()


# âœ… Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


# ğŸ”� Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


# ğŸ“Š Descriptive statistics for numerical columns
df[numerical_cols].describe()


# ğŸ”¢ Unique value counts for categorical columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())


# ğŸ�¯ Target Variable Distribution
# We begin by analyzing the distribution of our target variable, Personality, to see if the dataset is balanced between Extrovert and Introvert.

# ===== Target Variable Distribution =====

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Personality', palette='pastel', edgecolor='black')
plt.title('Distribution of Personality Types', fontsize=14)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Display normalized value counts (as proportions)
print("\nğŸ“Š Personality Value Counts (Proportions):")
print(df['Personality'].value_counts(normalize=True).round(3))


# ğŸ“ˆ Distribution of Numerical Features
# Next, we explore the distribution of the numerical features using histograms. This helps us understand the spread and skewness of the data.

# ===== Visualize Distribution of Numerical Features =====

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print descriptive statistics
    print(f'\nğŸ“Š Descriptive Stats for {col}:\n')
    print(df[col].describe(), '\n' + '-'*40)


# ğŸ“¦ Outlier Detection via Boxplots
# We use boxplots to identify potential outliers in numerical features.

plt.figure(figsize=(14, 6))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df, y=col, color='#FFA726')
    plt.title(f"Boxplot: {col}")
plt.tight_layout()
plt.show()


# ğŸ“Š Distribution of Categorical Features
# We use countplots to examine the balance of categorical features: Stage_fear and Drained_after_socializing.

# ===== Visualize Distribution of Categorical Features =====

cat_cols = ['Stage_fear', 'Drained_after_socializing']

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index,
        palette='Set2',
        edgecolor='black'
    )
    plt.title(f'{col} Distribution', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f'\nğŸ“Š Proportion of Each Category in "{col}":\n')
    print(df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)


# ===== Categorical Feature Distributions by Personality =====

for col, palette in zip(['Stage_fear', 'Drained_after_socializing'], ['Set1', 'Set1']):
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=col,
        hue='Personality',
        palette=palette,
        edgecolor='black'
    )
    plt.title(f'Distribution of {col} by Personality', fontsize=14)
    plt.xlabel(f'{col} (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Personality', labels=['Introvert (0)', 'Extrovert (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# Correlation Between Numerical Features
# A heatmap is plotted to assess correlations between numerical features, which may influence feature selection or interaction terms later.

plt.figure(figsize=(6, 4))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()


# ğŸ§  Feature vs Target Relationship (Numerical Features by Personality)

plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(
        data=df,
        x='Personality',
        y=col,
        palette='Set2',
        linewidth=1.2,
        fliersize=4
    )
    plt.title(f'{col} by Personality', fontsize=14, fontweight='semibold', color='#2E4057')
    plt.xlabel('Personality', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


#################


# # Handling Missing Values

# # For numerical columns: Impute missing values with the median (robust to outliers).
# # For categorical columns: Impute missing values with the mode (most frequent value).

# # Impute numerical columns
# for col in num_cols:
#     df[col].fillna(df[col].mean(), inplace=True)

# # Impute categorical columns
# for col in cat_cols:
#     # df[col].fillna(df[col].mode()[0], inplace=True)
#     df[col].fillna('Missing', inplace=True)

# # Confirm no missing values remain
# print(df.isnull().sum())


# Encoding Categorical Variables

# Convert Stage_fear and Drained_after_socializing from Yes/No to 1/0.
# Convert target variable Personality from Extrovert/Introvert to 1/0.

mapping_yes_no = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(mapping_yes_no)
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(mapping_yes_no)

# mapping_personality = {'Extrovert': 1, 'Introvert': 0}
# df['Personality'] = df['Personality'].map(mapping_personality)

le = LabelEncoder()
df["Personality"] = le.fit_transform(df["Personality"])


# Separate train and test datasets
train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns =['dataset'], errors='ignore')


# Drop unnecessary columns from both datasets
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['Personality'], errors='ignore')


# Feature and Target Separation

X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

# y = y.astype(int)


X


cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=4,
    random_seed=42,
    verbose=False
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scoring with X and y directly
cv_accuracy_scores = cross_val_score(cat_model, X, y, cv=cv, scoring='accuracy')
cv_f1_scores = cross_val_score(cat_model, X, y, cv=cv, scoring='f1_weighted')

print("CatBoost Model Cross-Validation Results")
print("ğŸ”� CV Accuracy Scores:", cv_accuracy_scores)
print("âœ… Mean CV Accuracy:", cv_accuracy_scores.mean())
print("ğŸ“‰ Std CV Accuracy:", cv_accuracy_scores.std())
print("ğŸ”� CV Weighted F1 Scores:", cv_f1_scores)
print("âœ… Mean CV Weighted F1 Score:", cv_f1_scores.mean())
print("ğŸ“‰ Std CV Weighted F1 Score:", cv_f1_scores.std())

# Now train on the full dataset and predict on the same (or you can predict on a separate test set if available)
cat_model.fit(X, y)

y_pred = cat_model.predict(X)

test_accuracy = accuracy_score(y, y_pred)
test_f1 = f1_score(y, y_pred, average='weighted')

print("\nğŸ§ª Full Data Accuracy:", test_accuracy)
print("ğŸ§ª Full Data Weighted F1 Score:", test_f1)
print("\nğŸ“‹ Classification Report:\n", classification_report(y, y_pred, digits=4))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.classes_)
disp.plot(cmap='Blues')
plt.title("CatBoost Confusion Matrix - Full Data")
plt.show()

if hasattr(cat_model, 'get_feature_importance'):
    importances = cat_model.get_feature_importance()
    feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
    
    feature_importance_df = (
        pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        .sort_values(by='Importance', ascending=False)
    )

    print("\nğŸ“Š CatBoost Feature Importances:")
    print(feature_importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('CatBoost Top Feature Importances')
    plt.tight_layout()
    plt.show()
else:
    print("âš ï¸� CatBoost model does not support direct feature importance.")





# # 1) Define your model exactly once:
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     tree_method='gpu_hist',
#     predictor='gpu_predictor',
#     enable_categorical=False,
#     random_state=42,
#     n_estimators=1000,
#     learning_rate=0.006358,
#     max_depth=8,
#     subsample=0.8854,
#     colsample_bytree=0.6,
#     reg_lambda=0.8295,
#     reg_alpha=5.5149,
#     gamma=0.0395,
#     min_child_weight=2,
#     use_label_encoder=False
# )

# # 2) Set up CV
# cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

# # 3) Crossâ€‘val scoring
# cv_accuracy_scores = cross_val_score(
#     xgb_model, X, y, cv=cv, scoring='accuracy'
# )
# cv_f1_scores = cross_val_score(
#     xgb_model, X, y, cv=cv, scoring='f1_weighted'
# )

# print("XGBoost Model Crossâ€‘Validation Results")
# print("ğŸ”� CV Accuracy Scores:", cv_accuracy_scores)
# print("âœ… Mean CV Accuracy:", cv_accuracy_scores.mean())
# print("ğŸ“‰ Std CV Accuracy:", cv_accuracy_scores.std())
# print("ğŸ”� CV Weighted F1 Scores:", cv_f1_scores)
# print("âœ… Mean CV Weighted F1 Score:", cv_f1_scores.mean())
# print("ğŸ“‰ Std CV Weighted F1 Score:", cv_f1_scores.std())

# # 4) Train on full data
# xgb_model.fit(X, y)

# # 5) Predict on the same data (just for demonstration; normally you'd have holdâ€‘out)
# y_pred = xgb_model.predict(X)

# # 6) Report metrics
# test_accuracy = accuracy_score(y, y_pred)
# test_f1 = f1_score(y, y_pred, average='weighted')
# print("\nğŸ§ª Full Data Accuracy:", test_accuracy)
# print("ğŸ§ª Full Data Weighted F1 Score:", test_f1)
# print("\nğŸ“‹ Classification Report:\n",
#       classification_report(y, y_pred, digits=4))

# # 7) Plot confusion matrix
# cm = confusion_matrix(y, y_pred)
# disp = ConfusionMatrixDisplay(cm, display_labels=xgb_model.classes_)
# disp.plot(cmap='Blues')
# plt.title("XGBoost Confusion Matrix - Full Data")
# plt.show()

# # 8) Feature importances
# importances = xgb_model.feature_importances_
# feature_names = (X.columns 
#                  if hasattr(X, 'columns') 
#                  else [f"Feature {i}" for i in range(X.shape[1])])
# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# print("\nğŸ“Š XGBoost Feature Importances:\n", feature_importance_df)

# plt.figure(figsize=(10, 6))
# sns.barplot(
#     x='Importance', 
#     y='Feature', 
#     data=feature_importance_df, 
#     palette='viridis'
# )
# plt.title('XGBoost Top Feature Importances')
# plt.tight_layout()
# plt.show()


# import xgboost as xgb
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
# from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
#                              confusion_matrix, ConfusionMatrixDisplay)

# # 1) Define your optimized model (tuned or from Optuna, e.g.)
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     tree_method='gpu_hist',
#     predictor='gpu_predictor',
#     enable_categorical=False,
#     use_label_encoder=False,
#     random_state=42,
#     n_estimators=2000,              # Increased to allow early stopping
#     learning_rate=0.0055,           # Slightly lower for better convergence
#     max_depth=9,
#     subsample=0.9,
#     colsample_bytree=0.65,
#     reg_lambda=1.0,
#     reg_alpha=4.5,
#     gamma=0.02,
#     min_child_weight=1.5,
#     verbosity=0
# )

# # 2) Cross-validation setup
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # 3) Cross-validation with multiple metrics
# cv_results = cross_validate(
#     xgb_model, X, y, 
#     cv=cv,
#     scoring=['accuracy', 'f1_weighted'],
#     return_train_score=False,
#     n_jobs=-1
# )

# print("ğŸ”� Cross-Validation Results")
# print(f"âœ… Mean Accuracy: {cv_results['test_accuracy'].mean():.4f} Â± {cv_results['test_accuracy'].std():.4f}")
# print(f"âœ… Mean Weighted F1: {cv_results['test_f1_weighted'].mean():.4f} Â± {cv_results['test_f1_weighted'].std():.4f}")

# # 4) Train on full data with early stopping
# eval_set = [(X, y)]
# xgb_model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50, verbose=False)

# # 5) Predict (on same data, just for demonstration)
# y_pred = xgb_model.predict(X)

# # 6) Metrics on full data
# print("\nğŸ§ª Full Data Performance")
# print(f"âœ… Accuracy: {accuracy_score(y, y_pred):.4f}")
# print(f"âœ… Weighted F1 Score: {f1_score(y, y_pred, average='weighted'):.4f}")
# print("\nğŸ“‹ Classification Report:\n", classification_report(y, y_pred, digits=4))

# # 7) Confusion Matrix
# cm = confusion_matrix(y, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_model.classes_)
# disp.plot(cmap='Blues')
# plt.title("XGBoost Confusion Matrix - Full Data")
# plt.tight_layout()
# plt.show()

# # 8) Feature Importances
# importances = xgb_model.feature_importances_
# feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# print("\nğŸ“Š Top Feature Importances:\n", importance_df.head(10))

# # Plot feature importances
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     x='Importance', 
#     y='Feature', 
#     data=importance_df.head(20), 
#     palette='viridis'
# )
# plt.title('Top 20 Feature Importances - XGBoost')
# plt.tight_layout()
# plt.show()


# # 0. Load your trained models (if not already in memory)
# # cat_model = CatBoostClassifier().load_model('cat_model.cbm')
# # xgb_model = xgb.XGBClassifier().load_model('xgb_model.json')

# # 1. Prepare test features
# test_features = test_df.drop(columns=['id'], errors='ignore')

# # 2. Get raw predictions from each model
# #    For hard voting we need class labels; for soft voting we need probabilities.
# cat_preds = cat_model.predict(test_features)
# xgb_preds = xgb_model.predict(test_features)

# cat_proba = cat_model.predict_proba(test_features)
# xgb_proba = xgb_model.predict_proba(test_features)

# # 3a. HARD VOTING (majority vote)
# hard_preds = np.where(cat_preds + xgb_preds >= 1, 1, 0)
# # â€” if both say 1 â†’ 2 â‰¥ 1 â†’ predict 1
# # â€” if one says 1 â†’ 1 â‰¥ 1 â†’ predict 1
# # â€” if none â†’ 0 â†’ predict 0

# # 3b. SOFT VOTING (average probabilities)
# avg_proba = (cat_proba + xgb_proba) / 2
# # assuming binary classification with proba[:, 1] = P(class=1)
# soft_preds = (avg_proba[:, 1] >= 0.5).astype(int)
# # you could also choose a different threshold, e.g. 0.4 or 0.6, if youâ€™ve tuned it on a validation set

# # Pick whichever you prefer:
# ensemble_preds = soft_preds   # or hard_preds

# # 4. Map numeric labels back to strings
# mapping = {1: 'Extrovert', 0: 'Introvert'}
# vec_map = np.vectorize(mapping.get)
# ensemble_labels = vec_map(ensemble_preds)

# # 5. Build submission DataFrame
# submission = pd.DataFrame({
#     'id': test_df.get('id', np.arange(len(test_df))),  # if no id, just use row index
#     'Fertilizer Name': ensemble_labels
# })

# # 6. Save to CSV
# submission.to_csv('submission.csv', index=False)


# Prepare test features by dropping the 'id' column if it exists
test_features = test_df.drop(columns=['id'], errors='ignore')

predictions = cat_model.predict(test_features)
# predictions = xgb_model.predict(test_features)

predictions = le.inverse_transform(predictions)

# # mapping dict
# mapping = {1: 'Extrovert', 0: 'Introvert'}

# # Method 1: using numpy.vectorize
# vec_map = np.vectorize(mapping.get)
# predictions = vec_map(predictions.astype(int))

# Build submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],  # if 'id' exists in test_df
    'Personality': predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)


submission

