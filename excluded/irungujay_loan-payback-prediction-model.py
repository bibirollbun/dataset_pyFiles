import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
sub = pd.read_csv('/content/sample_submission.csv')


print('\nTrain shape:', train.shape)
print('Test shape:', test.shape)
print('\nTrain columns:', train.columns.tolist())

train.head()


train.columns


#Basic Info
train.info()
train.describe(include='all')


# Missing Values
missing = train.isna().sum().to_frame('n_missing')
missing['pct'] = missing['n_missing'] / len(train) * 100
missing.sort_values('pct', ascending=False)


#Duplicates
train.duplicated().sum()


from collections import Counter
print("Class distribution before resampling:", Counter(train['loan_paid_back']))

train['loan_paid_back'].value_counts(normalize=True)


#Visualize class balance:
import seaborn as sns, matplotlib.pyplot as plt
sns.countplot(data=train, x='loan_paid_back')
plt.title('Target Balance')
plt.show()


from sklearn.utils import class_weight

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train['loan_paid_back']),
    y=train['loan_paid_back']
)

# Convert class weights to a dictionary
class_weight_dict = dict(zip(np.unique(train['loan_paid_back']), class_weights))

print("Class Weights:")
print(class_weight_dict)


num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop('loan_paid_back').drop('id')
train[num_cols].hist(bins=50, figsize=(12,8))
plt.tight_layout()
plt.show()


cat_cols = [col for col in train.select_dtypes('object').columns if col != 'id']
for col in cat_cols:
    plt.figure(figsize=(8,3))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index[:10])
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()


def feature_summary(df):
    summary = []
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            top_value = df[col].mode()[0] if not df[col].mode().empty else None
            top_freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else None
        else:
            unique_count = df[col].nunique()
            top_value = None
            top_freq = None
        summary.append([col, df[col].count(), unique_count, top_value, top_freq, missing_pct])

    return pd.DataFrame(summary, columns=['Feature', 'Count', 'Unique', 'Top', 'Freq', 'Missing%'])

feature_summary_df = feature_summary(train)
display(feature_summary_df)


# Create bins
df = train.copy()
df['credit_bin'] = pd.qcut(df['credit_score'], 8, duplicates='drop')
df['dti_bin'] = pd.qcut(df['debt_to_income_ratio'], 8, duplicates='drop')

pivot = df.pivot_table(
    index='credit_bin', columns='dti_bin', values='loan_paid_back', aggfunc='mean'
)

plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Payback Probability Heatmap: Credit Score vs DTI')
plt.ylabel('Credit Score Bin')
plt.xlabel('Debt-to-Income Ratio Bin')
plt.show()



sns.violinplot(data=train, x='loan_paid_back', y='credit_score', palette='Set2', scale='width')
plt.title('Distribution of Credit Score by Loan Payback Outcome')
plt.xlabel('Loan Paid Back (0=No, 1=Yes)')
plt.ylabel('Credit Score')
plt.show()


num_cols = train.select_dtypes(['float64','int64']).columns
corr = train[num_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='vlag', fmt='.2f', center=0)
plt.title('Correlation Heatmap: Numeric Features')
plt.show()


purpose_rate = (
    train.groupby('loan_purpose')['loan_paid_back']
    .agg(['mean','count'])
    .sort_values('mean', ascending=False)
)

plt.figure(figsize=(10,5))
sns.barplot(
    x=purpose_rate.index, y=purpose_rate['mean'],
    palette='viridis'
)
plt.title('Payback Rate by Loan Purpose')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Payback Probability')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Drop rows with missing values in the target variable
train_cleaned = train.dropna(subset=['loan_paid_back']).copy()

X = train_cleaned.drop(columns=['loan_paid_back','id'])
y = train_cleaned['loan_paid_back']

# Identify categorical columns
categorical_cols = X.select_dtypes(include='object').columns

# Create a column transformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns
)

# Create a pipeline with preprocessing and the RandomForestClassifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42))])

# Fit the pipeline
pipeline.fit(X, y)

# Get feature importances from the trained classifier in the pipeline
# The feature names after one-hot encoding can be obtained from the preprocessor
onehot_features = pipeline.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(categorical_cols)
numeric_features = [col for col in X.columns if col not in categorical_cols]
feature_names = list(onehot_features) + numeric_features

importances = pd.Series(pipeline.named_steps['classifier'].feature_importances_, index=feature_names)
top_imp = importances.nlargest(15)

plt.figure(figsize=(8,6))
top_imp.plot(kind='barh', color='teal')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.show()


train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1)
train['income_log'] = np.log1p(train['annual_income'])
train['debt_ratio'] = train['debt_to_income_ratio']


test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1)
test['income_log'] = np.log1p(test['annual_income'])
test['debt_ratio'] = test['debt_to_income_ratio']


train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
test = pd.get_dummies(test, columns=cat_cols, drop_first=True)


train, test = train.align(test, join='left', axis=1, fill_value=0)


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features (X) and target (y) from the training data
X = train.drop(columns=['loan_paid_back', 'id'])
y = train['loan_paid_back']

# Separate features for test data, dropping 'id' and the 'loan_paid_back' column
X_test = test.drop(columns=['id', 'loan_paid_back'], errors='ignore')


# Identify numerical columns from the current DataFrames (after one-hot encoding)
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
print("Numerical columns (for scaling):", numerical_cols)

# Create preprocessing pipeline for numerical features (scaling)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ],
    remainder='passthrough' # Keep already encoded categorical features and engineered features
)

# Apply the preprocessor to the training data
X_processed = preprocessor.fit_transform(X)

# Apply the same preprocessor to the test data
X_test_processed = preprocessor.transform(X_test)

print("\nShape of preprocessed training data:", X_processed.shape)
print("Shape of preprocessed test data:", X_test_processed.shape)


from sklearn.preprocessing import StandardScaler

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include='object').columns

# Create a column transformer for scaling numerical and one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (like engineered features that might be added later)
)

# The preprocessor is now defined and ready to be used in a pipeline
print("Preprocessor defined with StandardScaler for numerical features and OneHotEncoder for categorical features.")


# Identify numerical columns, including the engineered features
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'loan_to_income', 'income_log', 'debt_ratio']

# Identify categorical columns
categorical_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'id']

# Create a column transformer for scaling numerical and one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (like the one-hot encoded features from previous steps)
)

# The preprocessor is now defined and ready to be used in a pipeline
print("Preprocessor defined with StandardScaler for numerical features and OneHotEncoder for categorical features.")


numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop('loan_paid_back').drop('id')
skewness = train[numerical_cols].skew().sort_values(ascending=False)
print("Skewness of numerical features:")
display(skewness)


skewed_cols = ['loan_to_income', 'annual_income', 'debt_to_income_ratio', 'debt_ratio']

for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col]) # Apply the same transformation to the test set

print("Skewness of numerical features after transformation (train dataset):")
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop('loan_paid_back').drop('id')
skewness_after_transform = train[numerical_cols].skew().sort_values(ascending=False)
display(skewness_after_transform)


from sklearn.model_selection import train_test_split

# Drop rows with missing values in the target variable from the train DataFrame
train_cleaned = train.dropna(subset=['loan_paid_back']).copy()

X = train_cleaned.drop(columns=['loan_paid_back', 'id'])
y = train_cleaned['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Define the models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# Create pipelines for each model including the preprocessor
pipelines = {
    name: Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    for name, model in models.items()
}

# X and y are the original features and target before preprocessing
X = train.drop(columns=['loan_paid_back', 'id'])
y = train['loan_paid_back']

# Split the original data (before manual preprocessing) for consistent evaluation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


for name, pipeline in pipelines.items():
    print(f"Training {name}...")
    try:
        # Fit the pipeline on the training split
        pipeline.fit(X_train, y_train)

        # Make predictions on the validation split
        preds = pipeline.predict_proba(X_valid)[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_valid, preds)
        print(f'{name}: AUC = {auc:.4f}')
    except Exception as e:
        print(f"Error training or predicting with {name}: {e}")


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_valid, preds)
plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.show()


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

scoring_metric = 'roc_auc'

for name, pipeline in pipelines.items():
    print(f"Performing cross-validation for {name}...")
    try:
        # Use cross_val_score on the original data (X, y) as the pipeline includes preprocessing
        scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring_metric, n_jobs=-1)
        print(f'{name}: Average AUC = {scores.mean():.4f} (+/- {scores.std():.4f})')
    except Exception as e:
        print(f"Error during cross-validation for {name}: {e}")


from sklearn.model_selection import KFold

cv_results = {
    'LogisticRegression': 0.9110,
    'RandomForest': 0.9105,
    'GradientBoosting': 0.9153,
    'XGBoost': 0.9201,
    'LightGBM': 0.9195
}
cv_ranking = pd.Series(cv_results).sort_values(ascending=False)

top_2_model_names = cv_ranking.head(2).index.tolist()
top_2_models = {name: pipelines[name] for name in top_2_model_names}

X = train.drop(columns=['loan_paid_back', 'id'])
y = train['loan_paid_back']

oof_preds_top2 = pd.DataFrame(index=X.index)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in top_2_models.items():
    oof_preds_top2[name] = np.zeros(X.shape[0])
    print(f"Generating out-of-fold predictions for {name}...")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline.fit(X_train_fold, y_train_fold)

        oof_preds_top2.loc[valid_idx, name] = pipeline.predict_proba(X_valid_fold)[:, 1]

print("\nHead of Out-of-Fold Predictions for Top 2 Models:")
display(oof_preds_top2.head())

corr_matrix_top2 = oof_preds_top2.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_top2, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix of Top 2 Base Model Out-of-Fold Predictions')
plt.show()


from sklearn.model_selection import KFold

# Select the best performing models based on cross-validation results
base_models = {
    'XGBoost': pipelines['XGBoost'],
    'LightGBM': pipelines['LightGBM']
}

# Assuming X and y are the original features and target before preprocessing
X = train.drop(columns=['loan_paid_back', 'id'])
y = train['loan_paid_back']


# Generate out-of-fold predictions
oof_preds = pd.DataFrame(index=X.index)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in base_models.items():
    oof_preds[name] = np.zeros(X.shape[0])
    print(f"Generating out-of-fold predictions for {name}...")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        # Fit the pipeline on the training fold
        pipeline.fit(X_train_fold, y_train_fold)

        # Predict on the validation fold
        oof_preds.loc[valid_idx, name] = pipeline.predict_proba(X_valid_fold)[:, 1]

# Calculate and visualize the correlation matrix of OOF predictions
corr_matrix = oof_preds.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix of Base Model Out-of-Fold Predictions')
plt.show()

# Display model ranking based on cross-validation AUCs
print("\nModel Performance Ranking (based on Cross-Validation AUC):")

cv_results = {
    'LogisticRegression': 0.9110,
    'RandomForest': 0.9105,
    'GradientBoosting': 0.9153,
    'XGBoost': 0.9201,
    'LightGBM': 0.9197
}
cv_ranking = pd.Series(cv_results).sort_values(ascending=False)
print(cv_ranking)


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression # Import LogisticRegression

kf = KFold(n_splits=5, shuffle=True, random_state=42)
stacked_ensemble_auc_scores = []

print("Performing cross-validation for Stacked Ensemble...")

# Initialize the meta-model
meta_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# Iterate through each fold
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"Processing Fold {fold + 1}/{kf.get_n_splits()}...")

    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    oof_meta_features_train = pd.DataFrame(index=X_train_fold.index)
    meta_features_valid = pd.DataFrame(index=X_valid_fold.index)

    for name, pipeline in top_2_models.items():
        cloned_pipeline = clone(pipeline)

        cloned_pipeline.fit(X_train_fold, y_train_fold)

        oof_meta_features_train[name] = cloned_pipeline.predict_proba(X_train_fold)[:, 1]

        meta_features_valid[name] = cloned_pipeline.predict_proba(X_valid_fold)[:, 1]


    cloned_meta_model = clone(meta_model)
    cloned_meta_model.fit(oof_meta_features_train, y_train_fold)

    stacked_preds_valid = cloned_meta_model.predict_proba(meta_features_valid)[:, 1]

    fold_auc = roc_auc_score(y_valid_fold, stacked_preds_valid)
    stacked_ensemble_auc_scores.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

average_auc = np.mean(stacked_ensemble_auc_scores)
std_auc = np.std(stacked_ensemble_auc_scores)
print(f"\nStacked Ensemble Average AUC: {average_auc:.4f} (+/- {std_auc:.4f})")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Initialize the meta-model (Logistic Regression)
# Using class_weight='balanced' due to the imbalanced target variable
meta_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# Train the meta-model on the out-of-fold predictions
meta_model.fit(oof_preds_top2, y)

# Evaluate the meta-model on the out-of-fold predictions
meta_preds = meta_model.predict_proba(oof_preds_top2)[:, 1]
meta_auc = roc_auc_score(y, meta_preds)

print(f"Meta-model (Logistic Regression) AUC on OOF predictions: {meta_auc:.4f}")


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predictions from the trained meta-model on the out-of-fold predictions
meta_preds = meta_model.predict_proba(oof_preds_top2)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y, meta_preds)

# Calculate AUC
meta_auc = roc_auc_score(y, meta_preds)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Meta-model AUC = {meta_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random') # Plot random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stacked Ensemble Meta-model')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.metrics import roc_auc_score

# Create a DataFrame to store model results using the results from the previous training step
# Assuming the models dictionary and X_valid, y_valid are still available from the previous cell

results = {'Model': [], 'AUC': []}
for name, model in models.items():
    try:
        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        results['Model'].append(name)
        results['AUC'].append(auc)
    except Exception as e:
        print(f"Error predicting with {name}: {e}")
        # Handle models that failed to predict if necessary, perhaps skip them or assign a low AUC

results_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
print(results_df)

best_model_name = results_df.iloc[0, 0]
print(f"\n Best Performing Model: {best_model_name}")

# Fit best model on all data
best_model = models[best_model_name]

# Drop rows with missing values in the target variable from the train DataFrame
train_cleaned = train.dropna(subset=['loan_paid_back']).copy()

X = train_cleaned.drop(columns=['loan_paid_back', 'id'])
y = train_cleaned['loan_paid_back']

best_model.fit(X, y)


# Assuming top_2_models (dictionary of trained base model pipelines from OOF generation)
# and X, y (original features and target) are available

# Retrain each of the top 2 base models on the full training data
for name, pipeline in top_2_models.items():
    print(f"Retraining {name} on the full training data...")
    pipeline.fit(X, y)
    print(f"{name} retraining complete.")

# The retrained pipelines are now stored back in the top_2_models dictionary,
# ready to be used for generating predictions on the test set.


# Generate predictions on the test set using each retrained base model
test_meta_features = pd.DataFrame(index=X_test.index)

for name, pipeline in top_2_models.items():
    print(f"Generating test predictions for {name}...")
    # Predict probabilities on the test data
    test_meta_features[name] = pipeline.predict_proba(X_test)[:, 1]
    print(f"Test predictions generated for {name}.")

# The test_meta_features DataFrame now contains the predictions from the base models on the test set.
print("\nHead of Test Predictions from Base Models:")
display(test_meta_features.head())


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

meta_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# Train the meta-model on the out-of-fold predictions
print("Training the meta-model on out-of-fold predictions...")
meta_model.fit(oof_preds_top2, y)

# Evaluate the meta-model on the out-of-fold predictions
meta_preds = meta_model.predict_proba(oof_preds_top2)[:, 1]
meta_auc = roc_auc_score(y, meta_preds)

print(f"Meta-model (Logistic Regression) AUC on OOF predictions: {meta_auc:.4f}")

print("\nGenerating final predictions using the trained meta-model...")
final_test_predictions = meta_model.predict_proba(test_meta_features)[:, 1]

# The final_test_predictions array contains the stacked ensemble's predictions for the test set.
print("Final stacked ensemble predictions generated for the test set.")
print("\nFirst 5 final predictions:", final_test_predictions[:5])

# Prepare the submission file
submission = pd.DataFrame({
    'id': test['id'], # Assuming 'test' DataFrame with 'id' is still available
    'loan_paid_back': final_test_predictions
})


# Prepare the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': final_test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
display(submission.head())

