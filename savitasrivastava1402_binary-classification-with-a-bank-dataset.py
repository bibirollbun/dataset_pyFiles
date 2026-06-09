import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load the datasets
train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


#Display first 5 row
train_dataset.head()


#Check shape
print("Train dataset shape:", train_dataset.shape)


#Get info about datatypes 
train_dataset.info()


#check missing values
train_dataset.isnull().sum()


#check duplicate Row
train_dataset.duplicated().sum()


# Drop unwanted columns
train_dataset.drop(columns=["id"], inplace=True)


# check Target variable distribution
train_dataset['y'].value_counts()
#Target variable percentage:")
print(train_dataset['y'].value_counts(normalize=True) * 100)


# Count values of target variable
target_counts = train_dataset['y'].value_counts()

# Labels
labels = ['No', 'Yes']

colors = ['skyblue', 'lightcoral']

# Plot pie chart
plt.figure(figsize=(6.5, 4.5))
plt.pie(target_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title('Target Variable Distribution (y)', fontsize=14)
plt.axis('equal')  
plt.show()




plt.figure(figsize=(7,5))
sns.histplot(data=train_dataset, x="campaign", bins=30, hue="y", multiple="stack", stat="percent")
plt.title("Campaign Count Distribution by Target")
plt.xlabel("Number of Contacts in Campaign")
plt.ylabel("Percentage")
plt.show()


# Prepare grouped data
marital_target = train_dataset.groupby(['marital', 'y']).size().reset_index(name='count')
marital_pivot = marital_target.pivot(index='marital', columns='y', values='count').fillna(0)

# Plot
marital_pivot.plot(kind='bar', stacked=True, figsize=(8,5),
                   color=['skyblue','lightcoral'])

plt.title("Marital Status Distribution by Target Variable", fontsize=14)
plt.xlabel("Marital Status", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title="Target")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()



# Correlation Heatmap
numeric_dataset = train_dataset.select_dtypes(include='number')
plt.figure(figsize=(8,6))
sns.heatmap(numeric_dataset.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar_kws={'label':'Correlation'})
plt.title("Correlation Heatmap of Numeric Features")
plt.show()



# Separate features and target
X = train_dataset.drop('y', axis=1)
y = train_dataset['y']
X_test = test_dataset.drop('id', axis=1)

# Categorical columns to encode
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 
            'loan', 'contact', 'month', 'poutcome']

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    label_encoders[col] = le

print("✅ Data preprocessing done")
print("Training shape:", X.shape)
print("Test shape:", X_test.shape)



# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Training target distribution:")
print(y_train.value_counts(normalize=True))
print("Validation target distribution:")
print(y_val.value_counts(normalize=True))


# LightGBM model

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42
)

param_dist = {
    'n_estimators': [100, 250, 500, 1000],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [10, 20, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}


# Randomized Search CV

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='roc_auc',
    cv=skf,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit
random_search.fit(X_train, y_train)

# Best model from search

best_model = random_search.best_estimator_
print("Best Params:", random_search.best_params_)
print("Best CV ROC AUC:", random_search.best_score_)


# Evaluate on Validation set

y_val_pred = best_model.predict(X_val)             
y_val_proba = best_model.predict_proba(X_val)[:,1] 

# ROC AUC
roc_auc = roc_auc_score(y_val, y_val_proba)
print("Validation ROC AUC:", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_val, y_val_pred))


test_ids = test_dataset['id']

X_final_test = test_dataset.drop(columns=['id'])


from sklearn.preprocessing import LabelEncoder

for col in X_final_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_final_test[col] = le.fit_transform(X_final_test[col].astype(str))

# Get probability predictions using  best model
y_preds = best_model.predict_proba(X_final_test)[:, 1]

# Build submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'y': y_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")






