import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


train.head()


train.info()


# Statistical summary of quantitative variables
train.describe()


# Statistical summary of categorical variables
print("--- Categorical Features Summary ---")
display(train.describe(include='object'))

# Checking for data integrity
null_count = train.isnull().sum().sum()
duplicate_count = train.duplicated().sum()

print(f"Total Missing Values: {null_count}")
print(f"Total Duplicate Rows: {duplicate_count}")


# Visualizing numerical distributions
quant_vars = train.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])
fig, axs = plt.subplots(3, 2, figsize=(16, 16))

for i, var in enumerate(quant_vars):
    row, col = i // 2, i % 2
    axs[row, col].hist(train[var], bins=30, color='skyblue', edgecolor='black')
    axs[row, col].set_title(f'Distribution of {var}', fontsize=14)
    axs[row, col].set_xlabel(var)
    axs[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Visualizing categorical frequencies
quali_vars = train.select_dtypes(include=['object'])
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for i, var in enumerate(quali_vars):
    row, col = i // 2, i % 2
    counts = train[var].value_counts()
    axes[row, col].bar(counts.index, counts.values, color='salmon')
    axes[row, col].set_title(f'Frequency of {var}', fontsize=14)
    axes[row, col].set_xticklabels(counts.index, rotation=45)
    axes[row, col].set_ylabel('Count')

plt.tight_layout()
plt.show()


# Comprehensive view of numerical interactions
sns.pairplot(quant_vars, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.show()


# Compute the correlation matrix
corr = quant_vars.corr()

# Generate a mask for the upper triangle (to avoid redundant information)
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with professional formatting
sns.heatmap(
    corr, 
    mask=mask, 
    cmap=sns.diverging_palette(230, 20, as_cmap=True), 
    vmax=1, vmin=-1, center=0,
    square=True, linewidths=.5, annot=True, fmt=".2f",
    cbar_kws={"shrink": .5}
)

plt.title("Triangular Correlation Matrix", fontsize=15)
plt.show()


# Analyzing the relationship between categorical features and the target
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
quali_vars = train.select_dtypes(include=['object'])

for i, var in enumerate(quali_vars):
    row, col = i // 2, i % 2
    
    # Group by feature and target to visualize repayment rates
    counts = train.groupby([var, 'loan_paid_back']).size().unstack(fill_value=0)
    
    # Side-by-side bar plot
    counts.plot(kind='bar', ax=axes[row, col], color=['#e74c3c', '#2ecc71']) # Red for 0, Green for 1
    
    axes[row, col].set_title(f'{var} vs Loan Paid Back', fontsize=12)
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Count')
    axes[row, col].legend(title='Paid Back', labels=['No (0)', 'Yes (1)'])

plt.tight_layout()
plt.show()


# --- 5.1.1 Data Preparation ---
# We create copies of the original dataframes to preserve the raw data
df_train = train.copy()
df_test = test.copy()

# --- 5.1.2 Ordinal Mapping ---
# Manual mapping to ensure the logical hierarchy of education is preserved
edu_mapping = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3, 'Other': 4}

# Applying the mapping to both sets
df_train['education_level_ord'] = df_train['education_level'].map(edu_mapping)
df_test['education_level_ord'] = df_test['education_level'].map(edu_mapping)

# --- 5.1.3 Label Encoding for Subgrades ---
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# fit_transform on train, transform on test to prevent data leakage
df_train['grade_subgrade_le'] = le.fit_transform(df_train['grade_subgrade'])
df_test['grade_subgrade_le'] = le.transform(df_test['grade_subgrade'])

# Dropping original columns after transformation to avoid redundancy
df_train = df_train.drop(columns=['education_level', 'grade_subgrade'])
df_test = df_test.drop(columns=['education_level', 'grade_subgrade'])

print("Ordinal encoding successful. New shape:", df_train.shape)


df_train.head()


# --- 5.1.2 One-Hot Encoding (Nominal Variables) ---
from sklearn.preprocessing import OneHotEncoder

# Initializing encoder: dropping first column to avoid multicollinearity (Dummy Variable Trap)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
categorical_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']

# Fit on train and transform both to prevent data leakage
encoded_train = encoder.fit_transform(df_train[categorical_cols])
encoded_test = encoder.transform(df_test[categorical_cols])

# Reconstruction of DataFrames
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

# Integrating encoded features and cleaning up
df_train = pd.concat([df_train.drop(columns=categorical_cols), encoded_train_df], axis=1)
df_test = pd.concat([df_test.drop(columns=categorical_cols), encoded_test_df], axis=1)


# --- 5.2.1 Feature Standardization ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

num_vars = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

# Scaling numerical features
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_test[num_vars] = scaler.transform(df_test[num_vars])

print("Preprocessing complete. Data is now normalized and encoded.")


df_train.head()


from sklearn.model_selection import train_test_split

# Separating features (X) from the target (y)
X = df_train.drop(['id', 'loan_paid_back'], axis=1)  
y = df_train['loan_paid_back']

# Performing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_test.shape[0]} samples")


import xgboost as xgb

# Initializing the XGBoost Classifier
# We use tree_method='hist' for faster training on large datasets
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,          # Number of boosting rounds
    learning_rate=0.05,        # Step size shrinkage to prevent overfitting
    max_depth=6,               # Maximum depth of a tree
    subsample=0.8,             # Fraction of samples used per tree
    colsample_bytree=0.8,      # Fraction of features used per tree
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'          # Monitoring AUC during training
)

# Training the model
xgb_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Predicting probabilities for the positive class
y_prob = xgb_model.predict_proba(X_test)[:, 1] 
y_pred = xgb_model.predict(X_test)

# Metrics calculation
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'XGBoost ROC-AUC Score: {roc_auc:.4f}')
print('\nConfusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', class_report)


# --- 7.1. Combining X and y for the full dataset ---
X_full = df_train.drop(['id', 'loan_paid_back'], axis=1)
y_full = df_train['loan_paid_back']

# --- 7.2. Retraining the best model (XGBoost) ---
print("Retraining XGBoost on the full dataset...")
xgb_final = xgb.XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42,
    eval_metric='auc'
)

xgb_final.fit(X_full, y_full)

# --- 7.3. Final Prediction on Test Set ---
# Prepare the test features
X_test_final = df_test.drop(columns=['id'])

# Predict probabilities
final_probs = xgb_final.predict_proba(X_test_final)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': final_probs
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Final submission file created successfully!")

