# ğŸ“¦ Basic Libraries
import pandas as pd
import numpy as np

# ğŸ“Š Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# ğŸ§¹ Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# âš™ï¸� Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ğŸ“ˆ Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)



# ğŸ“� Load the application_train dataset
df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
# ğŸ–¹ Preview
df.head()

# ğŸ“Š Basic Overview
print("Dataset shape:", df.shape)
print("\nTarget variable distribution:")
print(df['TARGET'].value_counts())




# ğŸ“‰ Class Distribution
sns.countplot(data=df, x='TARGET')
plt.title('Distribution of Loan Status')
plt.xticks([0, 1], ['Repaid (0)', 'Defaulted (1)'])
plt.ylabel('Count')
plt.show()

# Percentage distribution
target_dist = df['TARGET'].value_counts(normalize=True) * 100
print(target_dist)



# ğŸ”� Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_percent = (missing / len(df)) * 100
top_missing = missing_percent.head(20)

# Generate a colormap with gradually changing colors (viridis)
cmap = cm.get_cmap('viridis', len(top_missing))
colors = [cmap(i) for i in range(len(top_missing))]

# Plot
plt.figure(figsize=(10, 8))
top_missing.plot(kind='barh', color=colors)
plt.title("Top 20 Columns with Missing Values (%)", fontsize=14)
plt.xlabel("Percentage", fontsize=12)
plt.gca().invert_yaxis()  # Highest on top
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Get % of missing values for all columns
missing_percent = (df.isnull().sum() / len(df)) * 100

# Get top 60 columns with highest missing %
top_60_missing = missing_percent[missing_percent > 0].sort_values(ascending=False).head(60)

# Display
print("ğŸ”� Top 60 Columns with Highest % of Missing Values:\n")
print(top_60_missing.round(2))



# Compare income by target class
plt.figure(figsize=(8, 5))
sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df)
plt.title("Income Distribution by Loan Status")
plt.xticks([0, 1], ['Repaid (0)', 'Defaulted (1)'])
plt.show()



# Convert DAYS_BIRTH to age in years
df['AGE'] = (-df['DAYS_BIRTH']) / 365

plt.figure(figsize=(10, 5))
sns.kdeplot(data=df[df['TARGET'] == 0], x='AGE', label='Repaid (0)')
sns.kdeplot(data=df[df['TARGET'] == 1], x='AGE', label='Defaulted (1)')
plt.title("Age Distribution by Loan Status")
plt.xlabel("Age (years)")
plt.legend()
plt.show()



# Keep only numeric columns
numeric_df = df.select_dtypes(include='number')

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Top 15 features most correlated with TARGET
top_corr = corr_matrix['TARGET'].abs().sort_values(ascending=False).head(20)

# Plot heatmap of those features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(
    numeric_df[top_corr.index].corr(),
    annot=True,
    fmt='.2f',              # ğŸ‘ˆ This line ensures two decimal places
    cmap='coolwarm'
)
plt.title("Top 15 Correlated Numerical Features with TARGET")
plt.show()



# Set Pandas to display all rows (no truncation)
pd.set_option('display.max_rows', None)

# Show full describe table (transposed)
df.describe().T



df.dtypes.value_counts()


# Drop columns with > 60% missing values
threshold = 60
missing_percent = (df.isnull().sum() / len(df)) * 100
high_missing_cols = missing_percent[missing_percent > threshold].index
df.drop(columns=high_missing_cols, inplace=True)



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
import pandas as pd

def smart_imputer(df):
    df_copy = df.copy()

    # Separate numeric and categorical columns
    num_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_copy.select_dtypes(include='object').columns

    print(f"ğŸ§  Numeric Columns with Missing Values: {df_copy[num_cols].isnull().sum().gt(0).sum()}")
    print(f"ğŸ“¦ Categorical Columns with Missing Values: {df_copy[cat_cols].isnull().sum().gt(0).sum()}")

    # Impute numeric columns using IterativeImputer with BayesianRidge (fast and smart)
    numeric_df = df_copy[num_cols]
    imp_num = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=10)
    df_copy[num_cols] = imp_num.fit_transform(numeric_df)

    # Impute categorical columns using most frequent strategy
    categorical_df = df_copy[cat_cols]
    imp_cat = SimpleImputer(strategy='most_frequent')
    df_copy[cat_cols] = imp_cat.fit_transform(categorical_df)

    print("âœ… All missing values filled (numeric: ML-based, categorical: frequent value).")
    return df_copy



df = smart_imputer(df)
print("\nğŸ”� Remaining Missing Values:", df.isnull().sum().sum())



df.head(20)


label_enc = LabelEncoder()

# Apply label encoding to all object or category type columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = label_enc.fit_transform(df[col])



# Total number of duplicate rows
duplicate_rows = df.duplicated()
num_duplicates = duplicate_rows.sum()

print(f"ğŸ”� Total Duplicate Rows: {num_duplicates}")



# Drop duplicate rows if needed
df = df.drop_duplicates()
print(f"ğŸ§¹ Cleaned DataFrame shape: {df.shape}")



X = df.drop('TARGET', axis=1)
y = df['TARGET']
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("Train set:", X_train.shape)
print("Test set :", X_test.shape)


# Store models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # for ROC-AUC
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    results.append((name, acc, auc))
    print(f"{name}: Accuracy = {acc:.4f}, ROC-AUC = {auc:.4f}")



# Create DataFrame of results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC-AUC"])
results_df.sort_values(by="ROC-AUC", ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)
results_df



# Bar plot for ROC-AUC
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="ROC-AUC", palette="Blues_d")
plt.title("Model Comparison (ROC-AUC)")
plt.ylim(0.5, 1.0)
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Bar plot for Accuracy
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="Greens_d")
plt.title("Model Comparison (Accuracy)")
plt.ylim(0.5, 1.0)
plt.ylabel("Accuracy Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# Define business cost values
COST_FALSE_POSITIVE = 10000  # Approved but defaulted
COST_FALSE_NEGATIVE = 500    # Rejected but would have repaid



# Get prediction probabilities
best_model = models["CatBoost"]
y_probs = best_model.predict_proba(X_test)[:, 1]


thresholds = np.arange(0.01, 1.0, 0.01)
costs = []

for thresh in thresholds:
    y_pred_thresh = (y_probs > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    total_cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
    costs.append(total_cost)

# Find optimal threshold
best_thresh_index = np.argmin(costs)
best_thresh = thresholds[best_thresh_index]
min_cost = costs[best_thresh_index]

print(f"Optimal Threshold: {best_thresh:.2f}")
print(f"Minimum Business Cost: ${min_cost:,.0f}")



plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs, marker='o', color='crimson')
plt.title("Business Cost vs Classification Threshold")
plt.xlabel("Threshold")
plt.ylabel("Total Cost")
plt.axvline(best_thresh, color='green', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


