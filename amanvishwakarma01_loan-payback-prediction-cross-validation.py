# ğŸ“˜ Goal: Predict the probability that a borrower will pay back their loan.
# ğŸ�† Evaluation Metric: AUC-ROC


# ğŸ“¦ Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# ğŸ“‚ Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')  
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print("âœ… Data Loaded Successfully!")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


train.head()


# ğŸ”� Exploratory Data Analysis (EDA)
# ğŸ“Š Basic Info

print("Train Data Info:")
print(train.info())


print("\nTest Data Info:")
print(test.info())


# ğŸ�·ï¸� Target Variable Distribution
if 'loan_paid_back' in train.columns:
    plt.figure(figsize=(5,5))
    ax = sns.countplot(x='loan_paid_back', data=train, palette='Set2')
    plt.title("Target Variable Distribution (loan_paid_back)")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x()+0.3, p.get_height()+50))
    plt.show()
    print(train['loan_paid_back'].value_counts(normalize=True))


# ğŸ”¢ Separate Numeric & Categorical Columns
num_cols = train.select_dtypes(include=np.number).columns.tolist()
cat_cols = train.select_dtypes(exclude=np.number).columns.tolist()
if 'loan_paid_back' in num_cols:
    num_cols.remove('loan_paid_back')

print(f"ğŸ“˜ Numerical Columns: {len(num_cols)}")
print(f"ğŸ�·ï¸� Categorical Columns: {len(cat_cols)}")


# ğŸ“ˆ Numerical Feature Distributions
train[num_cols].hist(figsize=(15, 10), bins=20, color="#3498db", edgecolor="white")
plt.suptitle("ğŸ“Š Numerical Feature Distributions", fontsize=14)
plt.show()


# ğŸ§­ Correlation Matrix
plt.figure(figsize=(10,6))
corr = train[num_cols + ['loan_paid_back']].corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()


# ğŸ§© Categorical Feature Analysis
for col in cat_cols[:5]:  # limit to first 5 for readability
    plt.figure(figsize=(6,4))
    sns.countplot(data=train, x=col, hue='loan_paid_back', palette='Set2')
    plt.title(f"{col} vs loan_paid_back")
    plt.xticks(rotation=45)
    plt.show()


# âš™ï¸� Data Preprocessing & Feature Engineering
# ğŸ§© Handling Missing Values

# Fill numerical columns with median, categorical with mode
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna(train[col].mode()[0])
        if col in test.columns:
            test[col] = test[col].fillna(train[col].mode()[0])
    else:
        train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col] = test[col].fillna(train[col].median())

print("âœ… Missing values handled successfully!")


# ğŸ”  Encoding Categorical Features
cat_cols = train.select_dtypes(exclude=np.number).columns.tolist()
encoder = LabelEncoder()

for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    encoder.fit(combined)
    train[col] = encoder.transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))

print("âœ… Label Encoding Completed!")


# ğŸ”� Feature & Target Split
X = train.drop(columns=['loan_paid_back', 'id'], errors='ignore')
y = train['loan_paid_back']
X_test = test.drop(columns=['id'], errors='ignore')

print(f"âœ… Final Train Shape: {X.shape}")
print(f"âœ… Final Test Shape: {X_test.shape}")


# ğŸ“� Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("âœ… Scaling Complete!")


# ğŸ§© Train-Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")


# âš™ï¸� Model Setup
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


# ğŸ§ª Stratified K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    X_tr, X_val = X_scaled[train_idx], X_scaled[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)
    print(f"ğŸ“‚ Fold {fold+1} | AUC = {auc:.4f}")

print("\nâœ… Cross-Validation Complete!")
print(f"Average AUC: {np.mean(auc_scores):.4f} Â± {np.std(auc_scores):.4f}")


# ğŸ“ˆ Feature Importance Plot
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette='viridis')
plt.title("Top 15 Important Features")
plt.show()


# ğŸ§® Predict on Validation Set (for sanity check)
val_preds = model.predict_proba(X_valid)[:, 1]
val_auc = roc_auc_score(y_valid, val_preds)
print(f"Validation AUC (holdout set): {val_auc:.4f}")


final_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_scaled, y)
print("âœ… Final model trained on full training data!")


# ğŸ“Š Predict Probabilities on Test Set
test_preds = final_model.predict_proba(X_test_scaled)[:, 1]


# ğŸ§¾ Create Submission File
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_preds
})

# Match sample submission order (optional safety check)
if 'id' in sample.columns:
    submission = submission.set_index('id').reindex(sample['id']).reset_index()

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file created successfully!")
submission.head()

