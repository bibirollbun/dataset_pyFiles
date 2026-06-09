pip install -q missingno


# Import necessary libraries for data manipulation and visualization
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Modeling libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,roc_auc_score
# Display plots inline for Jupyter
%matplotlib inline
sns.set_style("whitegrid")

# Load transaction and identity datasets (paths may need adjustment to your local Kaggle setup)
df_trans = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
df_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

# Merge datasets on TransactionID (left join to retain all transactions)
df = df_trans.merge(df_id, how='left', on='TransactionID')

# Examine the merged dataset
print("Merged Data Shape:", df.shape)
print("\nColumn Data Types:\n", df.dtypes[:10])
print("\nExample Rows:\n", df.head().T)



# 1) Define only the raw dataset columns you want to impute
numeric = [
    'TransactionAmt', 'card1', 'card2', 'card3',
    'addr1', 'addr2', 'dist1', 'dist2'
]
categorical = [
    'ProductCD', 'card4', 'card5', 'card6',
    'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
]

# 2) Impute numeric columns (median) and clip outliers
for col in numeric:
    med = df[col].median()
    df[col].fillna(med, inplace=True)
    lo, hi = df[col].quantile([0.001, 0.999])
    df[col] = df[col].clip(lo, hi)

# 3) Impute categorical columns
for col in categorical:
    if df[col].dtype == object or df[col].dtype.name == 'category':
        # Use mode for nearâ€�complete categoricals, 'Missing' for sparse if >50% null
        null_frac = df[col].isna().mean()
        if null_frac <= 0.5:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            
        else:
            df[col].fillna('Missing', inplace=True)
    else:
        # For any numeric-like categorical e.g. DeviceType encoded as ints
        med = df[col].median()
        df[col].fillna(med, inplace=True)


# Quick summary of numeric features (TransactionAmt and basic transaction fields)
print(df[['TransactionAmt']].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

# Check total missing values per column (sorted descending)
missing_counts = df.isnull().sum().sort_values(ascending=False)
print("\nTop 10 columns by missing values:\n", missing_counts.head(10))

# Visualize missing data pattern for a subset of columns
plt.figure(figsize=(6,4))
msno.matrix(df.sample(5000), color=(0.3,0.6,0.9))
plt.title("Missing Value Matrix (sample of 5k rows)")
plt.show()



plt.figure(figsize=(12,5))

# Original scale
plt.subplot(1,2,1)
sns.histplot(df['TransactionAmt'], bins=50, kde=False, color='skyblue')
plt.title('Transaction Amount Distribution (raw)')
plt.xlabel('TransactionAmt')

# Log scale (log1p to handle zero values safely)
plt.subplot(1,2,2)
sns.histplot(np.log1p(df['TransactionAmt']), bins=50, kde=False, color='orange')
plt.title('Transaction Amount Distribution (log scale)')
plt.xlabel('log(1 + TransactionAmt)')

plt.tight_layout()
plt.show()



categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'addr1']
for col in categorical_cols:
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette='Set2')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=30)
    plt.ylabel("Count")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()



fraud_rate = df['isFraud'].mean()
print(f"Proportion of Fraudulent Transactions: {fraud_rate*100:.2f}%")
sns.countplot(x='isFraud', data=df, palette=['#87cefa','#ff7f0e'])
plt.title('Target Distribution (0 = Non-Fraud, 1 = Fraud)')
plt.show()



plt.figure(figsize=(6,4))
sns.boxplot(x='isFraud', y='TransactionAmt', data=df, showfliers=False, palette=['#87cefa','#ff7f0e'])
plt.yscale('log')
plt.title('Transaction Amount by Fraud Status (log scale)')
plt.xlabel('isFraud')
plt.ylabel('TransactionAmt (log scale)')
plt.show()



for col in ['ProductCD', 'card4', 'card6']:
    rate = df.groupby(col)['isFraud'].mean().sort_values(ascending=False)
    plt.figure(figsize=(5,3))
    sns.barplot(x=rate.index, y=rate.values, palette='Set1')
    plt.title(f'Fraud Rate by {col}')
    plt.ylabel('Fraud Rate')
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.ylim(0, rate.max()*1.2)
    plt.tight_layout()
    plt.show()



# Convert TransactionDT to days and hours
df['Days'] = (df['TransactionDT'] // (3600*24)).astype(int)
df['Hour'] = ((df['TransactionDT'] % (3600*24)) // 3600).astype(int)



# Fraud rate by day
day_rate = df.groupby('Days')['isFraud'].mean()
plt.figure(figsize=(6,3))
day_rate.plot(kind='line', color='teal')
plt.title('Daily Fraud Rate Over Time')
plt.ylabel('Fraud Rate')
plt.xlabel('Days Since Reference')
plt.show()

# Fraud rate by hour of day (aggregated across all days)
hour_rate = df.groupby('Hour')['isFraud'].mean()
plt.figure(figsize=(6,3))
sns.lineplot(x=hour_rate.index, y=hour_rate.values, marker='o', color='purple')
plt.title('Fraud Rate by Hour of Day')
plt.ylabel('Fraud Rate')
plt.xlabel('Hour (0-23)')
plt.xticks(range(0,24,3))
plt.show()



numeric_cols = ['TransactionAmt', 'Days', 'Hour']
# If identity numeric columns exist (e.g., 'C1','C2',...),
numeric_cols += [col for col in df.columns if col.startswith('C')][:5]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(5,4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu', center=0)
plt.title("Correlation Matrix of Selected Numeric Features")
plt.show()



target_corr = df[numeric_cols].join(df['isFraud']).corr()['isFraud'].drop('isFraud')
print("Correlation with Fraud (numeric features):\n", target_corr)



# Example: Average transaction amount by ProductCD and card4
group_stats = df.groupby(['ProductCD', 
                          'card4'])['TransactionAmt'].mean().unstack()
print("Mean Transaction Amount by ProductCD and card4:\n", group_stats)



for m in ['M1','M2','M3','M4','M5','M6','M7','M8','M9']:
    if m in df.columns:
        rate = df.groupby(m)['isFraud'].mean()
        print(f"Fraud rate when {m}='T': {rate.get('T', np.nan):.4f}, when 'F': {rate.get('F', np.nan):.4f}")



# Log transform TransactionAmt
df['LogTransactionAmt'] = np.log1p(df['TransactionAmt'])

# Deviation from global mean (and scaled by std)
amt_mean = df['TransactionAmt'].mean()
amt_std = df['TransactionAmt'].std()
df['Amt_minus_mean'] = df['TransactionAmt'] - amt_mean
df['Amt_minus_std'] = (df['Amt_minus_mean']) / amt_std

# Ratio to card1/card4 group mean/std
for col in ['card1', 'card4']:
    if col in df.columns:
        mean_col = df.groupby(col)['TransactionAmt'].transform('mean')
        std_col = df.groupby(col)['TransactionAmt'].transform('std')
        df[f'Amt_to_mean_{col}'] = df['TransactionAmt'] / mean_col
        df[f'Amt_to_std_{col}'] = np.select(
            [std_col.isna(), std_col == 0],
            [0, 0],  # Assign 0 for both NaN and zero std
            df['TransactionAmt'] / std_col
        )


# If actual reference start-day is known to be some weekday, one can derive day-of-week.
# As an approximation (if transaction spans continuous days):
df['Weekday'] = df['Days'] % 7  # rough day of week indicator

# Nighttime indicator (e.g., hours 0-6 as night)
df['isNight'] = df['Hour'].apply(lambda x: 1 if x < 6 else 0)


# Identify V feature columns (assuming they start with 'V' and are numeric)
v_cols = [col for col in df.columns if col.startswith('V')]

# Fill missing values in V features (with min-1 so they do not dominate after scaling)
df_v = df[v_cols].copy()
for col in v_cols:
    df_v[col].fillna(df_v[col].min() - 1, inplace=True)

# Scale V features to [0,1] range
scaler = MinMaxScaler()
df_v_scaled = scaler.fit_transform(df_v)

# Apply PCA to the scaled V features, keep first 10 components
pca = PCA(n_components=10, random_state=42)
V_pca = pca.fit_transform(df_v_scaled)

# Add PCA component features to the main DataFrame
for i in range(V_pca.shape[1]):
    df[f'V_PCA_{i+1}'] = V_pca[:, i]

# Drop original V features to reduce dimensionality (optional, depending on modeling)
df.drop(columns=v_cols, inplace=True)


# Choose a subset of features for modeling
features = [
    'LogTransactionAmt', 'Amt_to_mean_card1', 'Amt_to_std_card1',
    'Amt_to_mean_card4', 'Amt_to_std_card4', 'Days', 'Hour', 'isNight'
]
# Add PCA features
features += [f'V_PCA_{i+1}' for i in range(10)]

# Include a couple categorical columns (one-hot encode later)
categorical = ['ProductCD', 'card4', 'card6']

# Prepare feature matrix with one-hot encoding for categorical variables
df_model = df[features + categorical + ['isFraud']].copy()
df_model = pd.get_dummies(df_model, columns=categorical, drop_first=True)

X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

# Split into training and validation sets (stratified to maintain class balance)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Training set shape:", X_train.shape, "Validation set shape:", X_val.shape)


# Baseline model: logistic regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
y_prob_lr = lr.predict_proba(X_val)[:,1]

# Evaluation metrics
print("Logistic Regression Evaluation:")
print(classification_report(y_val, y_pred_lr, digits=4))
roc_auc = roc_auc_score(y_val, y_prob_lr)
print(f"ROC AUC: {roc_auc:.4f}")



# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12],
    'class_weight': ['balanced']  # to handle class imbalance
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Set up Grid Search with 3-fold cross-validation
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

# Fit the model on training data
grid.fit(X_train, y_train)

# Extract the best model from grid search
best_rf = grid.best_estimator_
print("Best Parameters Found:", grid.best_params_)

# Evaluate on the validation set
y_pred_rf = best_rf.predict(X_val)
y_prob_rf = best_rf.predict_proba(X_val)[:, 1]

# Calculate metrics
roc_auc_rf = roc_auc_score(y_val, y_prob_rf)  # Added missing variable

print("\nTuned Random Forest Evaluation:")
print(confusion_matrix(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf, digits=4))
print(f"ROC AUC: {roc_auc_rf:.4f}") 



# Confusion matrix for the tuned Random Forest
cm = confusion_matrix(y_val, y_pred_rf)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Non-Fraud','Fraud'], yticklabels=['Non-Fraud','Fraud'])

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_prob_rf)

plt.figure(figsize=(6,4))
plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC = {roc_auc:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {roc_auc_rf:.3f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

