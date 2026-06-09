import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)




warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


#=============================================================
#2.DATA LOADING
#=============================================================
train_csv = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print("="*80)
print("DATA LOADING COMPLETE")
print("="*80)
print(f"Training set shape: {train_csv.shape}")
print(f"Test set shape: {test_csv.shape}")


#==============================================================
#3.DATA OVERVIEW & QUALITY CHECK
#==============================================================

print("\n" + "="*80)
print("DATA OVERVIEW")
print("="*80)

print("\n--- First 5 rows ---")
display(train_csv.head())

print("\n--- Dataset Info ---")
print(train_csv.info())

print("\n--- Statistical Summary ---")
display(train_csv.describe())

print("\n--- Missing Values Analysis ---")
missing_data = pd.DataFrame({
    'Feature': train_csv.columns,
    'Missing_Count': train_csv.isnull().sum(),
    'Missing_Percentage': (train_csv.isnull().sum() / len(train_csv) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data)

print("\n--- Duplicate Records ----")
duplicates = train_csv.duplicated().sum()
print(f"Number of duplicate rows:{duplicates} ({duplicates/len(train_csv)*100:.2f}%)")


#==========================================================
#3.TARGET VARIABLE ANALYSIS
#==========================================================

print("TARGET VARIABLE ANALYSIS")


target_col = 'diagnosed_diabetes'

target_dist = train_csv[target_col].value_counts() # Target distribution
target_pct = train_csv[target_col].value_counts(normalize=True) * 100

print("\n--- TARGET DISTRIBUTION -----")
print(f"Class 0 (No Diabetes): {target_dist[0]} ({target_pct[0]:.2f}%)")
print(f"Class 1 (Diabetes): {target_dist[1]} ({target_pct[1]:.2f}%)")
print(f"Class Imbalance Ratio: {target_dist[0]/target_dist[1]:.2f}:1")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=train_csv, x=target_col, ax=axes[0])
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
for container in axes[0].containers:
    axes[0].bar_label(container)

# Pie chart
axes[1].pie(target_dist, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', 
            startangle=90, colors=['#66b3ff', '#ff9999'])
axes[1].set_title('Target Variable Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()



#=====================================================
#4.NUMERICAL ANALYSIS
#====================================================

print('NUMERICAL ANALYSIS')

numerical_cols = train_csv.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

id_col = [col for col in numerical_cols if 'id' in col.lower()]
for col in id_col:
    numerical_cols.remove(col)


print(f"\nNumerical features: {numerical_cols}")



n_cols = 3
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.histplot(train_csv[col], kde=True, ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
    axes[idx].axvline(train_csv[col].mean(), color='red', linestyle='--', label='Mean')
    axes[idx].axvline(train_csv[col].median(), color='green', linestyle='--', label='Median')
    axes[idx].legend()

for idx in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


#OUTLIER DETECTION
print("Outlier Detection")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(y=train_csv[col], ax=axes[idx], color='lightcoral')
    axes[idx].set_title(f'Box Plot - {col}', fontweight='bold')

for idx in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


# Skewness analysis
print("\n--- Skewness Analysis ---")
skewness = pd.DataFrame({
    'Feature': numerical_cols,
    'Skewness': [train_csv[col].skew() for col in numerical_cols]
}).sort_values('Skewness', ascending=False)
print(skewness)



print("BIVARIATE ANALYSIS - FEATURES VS TARGET")


# Distribution comparison by target class
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.violinplot(data=train_csv, x=target_col, y=col, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{col} by Diabetes Status', fontweight='bold')
    axes[idx].set_xlabel('Diabetes')

for idx in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()



print("CORRELATION ANALYSIS")

# Correlation matrix
correlation_matrix = train_csv[numerical_cols + [target_col]].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlation with target
target_corr = correlation_matrix[target_col].sort_values(ascending=False)
print("\n--- Correlation with Target ---")
print(target_corr)

# Visualize correlation with target
plt.figure(figsize=(10, 6))
target_corr_plot = target_corr.drop(target_col)
colors = ['green' if x > 0 else 'red' for x in target_corr_plot]
plt.barh(range(len(target_corr_plot)), target_corr_plot.values, color=colors)
plt.yticks(range(len(target_corr_plot)), target_corr_plot.index)
plt.xlabel('Correlation Coefficient')
plt.title('Feature Correlation with Target', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()


#================================================
#------CATEGORICAL FEATURE EXPLORATION-----------
#================================================

cat_cols = train_csv.select_dtypes(include=['object','category']).columns.tolist()

cat_cols = list(set(cat_cols))   # remove duplicates
print("Categorical columns:", cat_cols)


for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train_csv, x=col)
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


print('Categorical Feature Importance')

for col in cat_cols:
    print(f"\n--- Chi-square Test for {col} ---")
    contingency_table = pd.crosstab(train_csv[col], train_csv[target_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi2 = {chi2:.4f}")
    print(f"p-value = {p:.4f}")
    if p < 0.05:
        print("➡ Significant relationship with target\n")
    else:
        print("➡ Not significant\n")


# Choosed only numerical features for training
X_train = train_csv[numerical_cols]
X_train.shape


X = X_train
y = train_csv[target_col].copy()

print(f"Shape of Features:{X.shape}")
print(f"Shape of Target:{y.shape}")


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_test.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled,y_train)


y_pred_train = model.predict(X_train_scaled)
y_pred_val = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_test, y_pred_val)
val_precision = precision_score(y_test, y_pred_val)
val_recall = recall_score(y_test, y_pred_val)
val_f1 = f1_score(y_test, y_pred_val)
val_roc_auc = roc_auc_score(y_test, y_pred_proba)


print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1 Score: {val_f1:.4f}")
print(f"ROC AUC: {val_roc_auc:.4f}")


print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_val, 
                          target_names=['No Diabetes', 'Diabetes']))


cm = confusion_matrix(y_test,y_pred_val)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


test = test_csv[numerical_cols].copy()


test_scaled = scaler.transform(test)


test_predictions = model.predict(test_scaled)
test_proba = model.predict_proba(test_scaled)[:,1]


print(test_proba)


submission_df = pd.DataFrame({
    'id': test_csv['id'],
    'diagnosed_diabetes' : test_proba
})


submission_df.head()

