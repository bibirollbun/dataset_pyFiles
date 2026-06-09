
# Required Modules and functions 
import pandas as pd 
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import LabelEncoder
print("modules loaded")





import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


sample_submission 


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train


train.info()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()




# Count data types
dtype_counts = train.dtypes.value_counts()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=dtype_counts.index.astype(str), y=dtype_counts.values, palette='Set2')

# Add labels
for i, val in enumerate(dtype_counts.values):
    plt.text(i, val + 0.1, str(val), ha='center', va='bottom', fontsize=12)

plt.title("Data Types in train.csv")
plt.xlabel("Data Type")
plt.ylabel("Number of Columns")
plt.tight_layout()
plt.show()

dtype_counts


non_null_counts = train.notnull().sum().sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=non_null_counts.values, y=non_null_counts.index, palette='Blues_r')
plt.title("Non-Null Counts per Column")
plt.xlabel("Count")
plt.ylabel("Column")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset

sns.set(style="whitegrid")

# Set up the figure
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Target Variable Distribution (Top-Left)
sns.countplot(x='y', data=train, palette='Set2', ax=axs[0, 0])
axs[0, 0].set_title('Target Variable Distribution (y)')
axs[0, 0].set_xlabel('Subscribed (1 = Yes, 0 = No)')
axs[0, 0].set_ylabel('Count')

# Plot 2: Age Distribution (Top-Right)
sns.histplot(train['age'], bins=30, kde=True, color='skyblue', ax=axs[0, 1])
axs[0, 1].set_title('Age Distribution of Clients')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Frequency')

# Plot 3: Subscription Rate by Job Type (Bottom-Left)
sns.barplot(
    x='job', 
    y='y', 
    data=train, 
    estimator=lambda x: sum(x) / len(x), 
    palette='coolwarm', 
    ax=axs[1, 0]
)
axs[1, 0].set_title('Subscription Rate by Job Type')
axs[1, 0].set_xlabel('Job Type')
axs[1, 0].set_ylabel('Subscription Rate')
axs[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Correlation Heatmap (Bottom-Right)
numeric_cols = train.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f", ax=axs[1, 1])
axs[1, 1].set_title('Correlation Heatmap')

# Adjust layout
plt.tight_layout()
plt.show()



train 




col = train.select_dtypes(include='object')
colT = test.select_dtypes(include='object')
le = LabelEncoder()
col.info()

for i in col.columns:
    col[i]= le.fit_transform(col[i])

for i in colT:
    colT[i] = le.fit_transform(colT[i])
    






colname = col.columns

train[colname] = col
test[colT.columns] = colT
train
test


X= train.drop(columns=["y","id"])
y = train["y"]
X_train ,X_test , y_train , y_test = train_test_split(X,y,random_state=42)



xgbmodel = XGBClassifier(n_estimators=2700,
    learning_rate=3e-1,
    max_depth=5,
    subsample=0.93,
    colsample_bytree=0.76,
    objective='binary:logistic', eval_metric='auc', random_state=42)
xgbmodel.fit(X_train, y_train)

for i, j in xgbmodel.get_params().items():
    if j is not None:
        print(i, ":", j)

    


plt.figure(figsize=(10, 6))
sn.heatmap(pd.DataFrame(xgbmodel.feature_importances_, index=X_train.columns))
plt.title("Feature Importances")
plot_importance(xgbmodel)


y_pred = xgbmodel.predict_proba(X_test)
auc = roc_auc_score(y_test, y_pred[:, 1])
print(f"AUC: {auc:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
auc_score = roc_auc_score(y_test, y_pred[:, 1])

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='red')
plt.fill_between(fpr, tpr, color='blue', alpha=0.2) #ill under the curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


test.info()


x_test = test.drop(columns=["id"])
y_pred_proba = xgbmodel.predict_proba(x_test)[:, 1]  # class 1 probability



submission = pd.DataFrame({
    'id': test['id'],
    'y': y_pred_proba
})
submission.to_csv("submission.csv", index=False)
print("Yappie Yappiee model ")

