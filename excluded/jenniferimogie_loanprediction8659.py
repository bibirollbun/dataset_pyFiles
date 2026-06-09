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


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#import neccessary libraries 
import seaborn as sns
import matplotlib.pyplot as plt


#file_path_train ="/kaggle/input/playground-series-s5e11/train"
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")


#view dataset
df.head()


#information on dataset
df.info()


loan_distribution = df['loan_paid_back'].value_counts()

# Define labels and colors
labels = ['Paid', 'Unpaid']
colors = ['green', 'red']

# Plot pie chart without labels
plt.figure(figsize=(6,3))
patches, texts, autotexts = plt.pie(loan_distribution, autopct='%1.1f%%', startangle=90, colors=colors)

# Add legend
plt.legend(patches, labels, title="Loan Status", loc="best")
plt.title('Loan Repayment Distribution')
plt.axis('equal')  # Ensures pie is drawn as a circle
plt.tight_layout()
plt.show()



# Separate dataset into demographic attributes for further analysis
demographic_features = df[["id","gender","marital_status", "education_level", "employment_status","loan_paid_back"]]



demographic_features.sample(5)


demographic_features_ = demographic_features.drop('id',axis=1)
for column in demographic_features_.columns:
    print(f"Value counts for '{column}':")
    print(demographic_features_[column].value_counts())
    print("\n" + "-"*40 + "\n")



# Group by education level and calculate repayment rate
repayment_rate = demographic_features.groupby('marital_status')['loan_paid_back'].mean().reset_index()

# Plot
plt.figure(figsize=(6,3))
sns.barplot(data=repayment_rate, x='marital_status', y='loan_paid_back', palette='viridis')
plt.title('Loan Repayment Rate by Education Level')
plt.ylabel('Repayment Rate')
plt.xlabel('Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Group by education level and calculate repayment rate
repayment_rate = demographic_features.groupby('employment_status')['loan_paid_back'].mean().reset_index()

# Plot
plt.figure(figsize=(6,3))
sns.barplot(data=repayment_rate, x='employment_status', y='loan_paid_back', palette='viridis')
plt.title('Loan Repayment Rate by Employment Status')
plt.ylabel('Repayment Rate')
plt.xlabel('Employment Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


## Separate dataset into financial attributes for further analysis
financial_features = df[["id","annual_income","debt_to_income_ratio","credit_score","loan_paid_back"]]
financial_features.sample(5)


financial_features.describe()


#check for skewness in the annual_income


financial_features_ = financial_features.drop('id',axis=1)
for column in financial_features_.columns:
    skew = financial_features_[column].skew()
    if  skew > 1:
        print(f"For '{column}' the skewness is > 1, it’s highly skewed. with skewness at", skew)
        print()
    else:
        print(f"For '{column}', it’s close to 0, the distribution is fairly normal.")
        print()


# Transform columns
df['log_annual_income'] = np.log1p(df['annual_income'])
df['log_debt_to_income_ratio'] = np.log1p(df['debt_to_income_ratio'])


fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Raw income histogram
sns.histplot(df['annual_income'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Raw Annual Income Distribution')
axes[0].set_xlabel('Annual Income')

# Log-transformed income histogram
sns.histplot(df['log_annual_income'], bins=30, kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Log-Transformed Annual Income')
axes[1].set_xlabel('log(1 + Annual Income)')

plt.tight_layout()
plt.show()



#Create a new column "target" for visualization 
df['target'] = df['loan_paid_back'].map({1.0:'Paid', 0.0:'Unpaid'})


df.sample()


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)

# Define custom palette
custom_palette = {"Paid": 'green', "Unpaid": 'red'}

# Employed
sns.scatterplot(
    data = df[df['employment_status'].isin(['Employed',"Self-employed", "Retired"])],
    x='credit_score',
    y='log_debt_to_income_ratio',
    hue='target',
    palette=custom_palette,
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title('Employed')

# Unemployed
sns.scatterplot(
     data = df[df['employment_status'].isin(['Unemployed',"Student"])],
    x='credit_score',
    y='log_debt_to_income_ratio',
    hue='target',
    palette=custom_palette,
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title('Unemployed')

for ax in axes:
    ax.set_xlabel('Credit Score')
    ax.set_ylabel('Debt-to-Income Ratio')

plt.tight_layout()
plt.show()



loan_features =df[["id","loan_amount","interest_rate","loan_purpose","grade_subgrade","loan_paid_back"]]
loan_features.sample(5)


loan_features.describe()


# Define bins
bins = np.arange(0, 50001, 2500)

# Separate paid vs unpaid
paid = df[df['target'] == "Paid"]['loan_amount']
unpaid = df[df['target'] == "Unpaid"]['loan_amount']

# Plot histogram
plt.figure(figsize=(12,6))
plt.hist([paid, unpaid], bins=bins, stacked=False,
         color=['green','red'], label=['Paid','Unpaid'])

# Labels and legend
plt.xlabel("Loan Amount")
plt.ylabel("Count of Loans")
plt.title("Loan Amount Distribution with Default Rate Overlay")
plt.legend()
plt.grid(alpha=0.3)

plt.show()



# Create categorical bins
loan_bins = pd.cut(df['loan_amount'], bins=bins)

# Count totals and unpaid loans per bin
bin_counts = loan_bins.value_counts().sort_index()
unpaid_counts = loan_bins[df['target'] == "Unpaid"].value_counts().sort_index()

# Align indices to avoid mismatches
unpaid_counts = unpaid_counts.reindex(bin_counts.index, fill_value=0)

# Compute default rate (%)
default_rate = (unpaid_counts / bin_counts) * 100

# Bin centers for plotting
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.plot(bin_centers, default_rate, color='black', marker='o', label='Default Rate (%)')

# Labels and legend
plt.xlabel("Loan Amount")
plt.ylabel("Count of Loans")
plt.title("Defaulters with Loan Amount")
plt.legend()
plt.grid(alpha=0.3)

plt.show()



bins = np.arange(3.0,30.0 , 3.0)

# Create categorical bins
rate_bins = pd.cut(df['interest_rate'], bins=bins)

# Count totals and unpaid loans per bin
bin_counts = rate_bins.value_counts().sort_index()
unpaid_counts = rate_bins[df['target'] == "Unpaid"].value_counts().sort_index()

# Align indices to avoid mismatches
unpaid_counts = unpaid_counts.reindex(bin_counts.index, fill_value=0)

# Compute default rate (%)
default_rate = (unpaid_counts / bin_counts) * 100

# Bin centers for plotting
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.plot(bin_centers, default_rate, color='black', marker='o', label='Default Rate (%)')

# Labels and legend
plt.xlabel("Loan Amount")
plt.ylabel("Count of Loans")
plt.title("Interest rate with defaulters")
plt.legend()
plt.grid(alpha=0.3)

plt.show()



unpaid_loans = loan_features[loan_features['loan_purpose'] == "Debt consolidation"]


plt.figure(figsize=(12,6))
sns.countplot(data=unpaid_loans, x='grade_subgrade', order=unpaid_loans['grade_subgrade'].value_counts().index, palette='Reds')

plt.title('Unpaid Loans by Grade/Subgrade')
plt.xlabel('Grade/Subgrade')
plt.ylabel('Number of Unpaid Loans')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Step 1: Create a count table
count_table = loan_features.groupby(['grade_subgrade', 'loan_paid_back']).size().unstack(fill_value=0)

# Step 2: Plot stacked bar chart
count_table.plot(kind='bar', stacked=True, figsize=(6,6), color=['red', 'green'])

plt.title('Loan Repayment by Loan Purpose')
plt.xlabel('Loan Purpose')
plt.ylabel('Number of Loans')
plt.legend(title='Loan Paid Back')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#separate categorical variable 
numerical_variable = df[['loan_amount',"credit_score","interest_rate","log_annual_income","log_debt_to_income_ratio","loan_paid_back"]]
categorical_cariable = df[["id","gender","marital_status","education_level","employment_status","loan_purpose","grade_subgrade"]]


# Assuming df is your DataFrame
# Compute correlation matrix
corr_matrix = numerical_variable.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Plot heatmap
sns.heatmap(
    corr_matrix,
    annot=True,           # show correlation values
    cmap='coolwarm',      # diverging color palette
    fmt=".2f",            # format for annotations
    linewidths=0.5,       # line between cells
    square=True           # square cells
)

# Titles and labels
plt.title("Correlation Heatmap of Numeric Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



!pip install catboost


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

X = df.drop(["loan_paid_back", "target","annual_income","debt_to_income_ratio","id","gender", "marital_status",], axis=1)
y = df["loan_paid_back"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

# Identify categorical columns by index
categorical_features = [X.columns.get_loc(col) for col in [ "education_level", "employment_status","loan_purpose","grade_subgrade"]]


#Build model
model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.024,                # Controls step size; lower for fine-tuning
    depth=6,                            # Tree depth; balances bias-variance
    l2_leaf_reg=5,                      # L2 regularization on leaf values
    random_strength=2.5,               # Adds noise to tree splits for robustness
    bagging_temperature=0.5,            # Controls sampling randomness (0 = deterministic)
    border_count=128,                   # Number of splits for numerical features
    grow_policy='Depthwise',            # Alternatives: 'Depthwise', 'Lossguide', 'SymmetricTree'
    boosting_type='Plain',             # Alternatives: 'Ordered' (for small datasets)
    eval_metric='AUC',
    early_stopping_rounds=500,
    #eval_fraction=0.2,
    verbose=500,
    random_seed=67,                   # Ensures reproducibility
    use_best_model=True,                # Retain best iteration
    # task_type='GPU',
    # Use 'GPU' if available for speed
    od_type='Iter',                     # Overfitting detector type
    # od_wait=50, 
)


# Train with categorical features specified
model.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_test, y_test))


from sklearn.metrics import roc_auc_score

y_pred = model.predict_proba(X_test)[:,1]
print("Test AUC:", roc_auc_score(y_test, y_pred))


from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, classification_report
model.save_model("catboost_model.cbm")

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability for positive class

# Apply custom threshold of 0.90
threshold = 0.80
y_pred_custom = (y_pred_proba >= threshold).astype(int)

# Precision, Recall, F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)

# AUC (Area Under ROC Curve)
auc = roc_auc_score(y_test, y_pred_proba)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
print("AUC:", auc)
# Precision, Recall, F1 with threshold
precisionc = precision_score(y_test, y_pred_custom)
recallc = recall_score(y_test, y_pred_custom)
f1c = f1_score(y_test, y_pred_custom)
accuracyc = accuracy_score(y_test,y_pred_custom)
print()
# AUC (Area Under ROC Curve) with threshold
aucc = roc_auc_score(y_test, y_pred_proba)
print("Precision:", precisionc)
print("Recall:", recallc)
print("F1-score:", f1c)
print("Accuracy:", accuracyc)
print("AUC:", aucc)

# Full classification report
#print("\nClassification Report:\n", classification_report(y_test, y_pred))



from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Assume you already have:
# y_test = true labels
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,5))

# Plot ROC
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.subplot(1,2,2)
plt.plot(recall, precision, color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()



# Get feature importance values
importances = model.get_feature_importance()
feature_names = X.columns  # exclude target column

# Create DataFrame for plotting
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

# Plot horizontal bar chart
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# Get predictions for final submission
X_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
#y_pred = model.predict(X_test)
X_sub.head()


# Use log1p to handle zero safely
X_sub['log_annual_income'] = np.log1p(X_sub['annual_income'])
X_sub['log_debt_to_income_ratio'] = np.log1p(X_sub['debt_to_income_ratio'])
X_subi= X_sub.drop(["gender", "marital_status","id","annual_income","debt_to_income_ratio"], axis=1)


# Identify categorical columns by index
categorical_features = [X_subi.columns.get_loc(col) for col in [ "education_level", "employment_status","loan_purpose","grade_subgrade"]]


y_pred_sub = model.predict(X_subi)
# Get predictions
#y_pred_sub= model.predict(X_sub)
#y_pred_sub_proba = model.predict_proba(X_sub)[:, 1]  # probability for positive class

# Apply custom threshold of 0.90
#threshold = 0.80
#y_pred_sub_custom = (y_pred_sub_proba >= threshold).astype(float)


submission = pd.DataFrame({
    "id": X_sub["id"],        # replace "Id" with the actual column name in your test set
    "loan_paid_back": y_pred_sub      # or the required column name (check competition instructions)
})

# Save to CSV for Kaggle submission
submission.to_csv("submission.csv", index=False)


submission.head()

