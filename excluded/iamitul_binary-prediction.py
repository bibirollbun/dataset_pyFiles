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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



# loading the dataset

df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv",index_col='id')


df.head()


df.info()


df['rainfall'].value_counts()


df.isnull().sum()


df.shape


df.duplicated().sum()


df.describe()


sns.histplot(data=df, x="day", hue="rainfall")


col = ['pressure','maxtemp', 'temparature', 'mintemp', 'dewpoint','humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
fig , axes = plt.subplots(5,2,figsize=(10,10))

for i, (ax, x) in enumerate(zip(axes.flatten(), col)):
    sns.kdeplot(data=df, x=x, hue="rainfall", ax=ax)
    ax.set_title(x)

plt.tight_layout()  # Adjusts layout to prevent overlap
plt.show()


sns.pairplot(data=df, hue="rainfall")


col = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

fig, axes = plt.subplots(2, 5, figsize=(15, 8))  # Creating a 2x5 grid

for ax, y in zip(axes.flatten(), col):  
    sns.scatterplot(data=df, x="sunshine", y=y, hue="rainfall", ax=ax)  
    ax.set_title(f"sunshine vs {y}")  

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



col = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

fig, axes = plt.subplots(2, 5, figsize=(15, 8))  # Creating a 2x5 grid

for ax, x in zip(axes.flatten(), col):  
    sns.boxplot(data=df, x=x, ax=ax,hue='rainfall')  
    ax.set_title(x)  

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


from imblearn.combine import SMOTETomek

X = df.drop('rainfall', axis=1)
y = df['rainfall']

ros = SMOTETomek(random_state=42)

# Apply Random Over_sampling
X_resampled_ros, y_resampled_ros = ros.fit_resample(X, y)
df_random = pd.DataFrame(X_resampled_ros, columns=X.columns)
df_random['rainfall'] = y_resampled_ros


print(df_random.shape)
print(df_random['rainfall'].value_counts())


# train test and split
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(df_random.drop(columns='rainfall'),df_random['rainfall'],random_state=42)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

model = RandomForestClassifier(random_state=42)
model1 = XGBClassifier(random_state=42)

cv_R_R = cross_val_score(model, X_train, y_train, cv=5)
cv_X_R = cross_val_score(model1, X_train, y_train, cv=5)
print("Ramdom forest in random data : ",np.mean(cv_R_R))
print("XGB in random data : ",np.mean(cv_X_R))


#XGBoost classifier

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


# Initialize the XGBoost Classifier
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define the hyperparameters to tune
param_dist = {
    'h_estimators': np.arange(50, 90, 10),  # Reduce boosting rounds to prevent overfitting
    'max_depth': np.arange(3, 10, 2),  # Reduce tree depth to avoid too complex models
    'learning_rate': np.linspace(0.01, 0.1, 5),  # Lower learning rate for stability
    'subsample': np.linspace(0.5, 0.8, 3),  # Reduce subsampling to add randomness
    'colsample_bytree': np.linspace(0.5, 0.8, 3),  # Use fewer features per tree
    'min_child_weight': np.arange(3, 10, 2),  # Increase min_child_weight to prevent small leaves
    'gamma': np.linspace(0.1, 0.5, 3),  # Slightly increase gamma for more conservative splits
    'reg_lambda': np.array([5, 10, 15, 20]),  # Increase L2 regularization
    'reg_alpha': np.array([5, 10, 15, 20])  # Increase L1 regularization

}

# Use RandomizedSearchCV (n_iter controls how many random samples to test)
random_search_xgb = RandomizedSearchCV(
    estimator=xgb, param_distributions=param_dist, 
    n_iter=100,  # Try 100 random combinations instead of 270,000
    cv=5, n_jobs=-1, verbose=2, random_state=42
)

# Fit the model
random_search_xgb.fit(X_train, y_train)

# Best parameters found by RandomizedSearchCV
print(f"Best parameters: {random_search_xgb.best_params_}")

# Evaluate on test set
y_pred_xgb = random_search_xgb.predict(X_test)
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred_xgb)}")



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameters to tune
param_dist_rf = {
    'n_estimators': np.arange(50, 300, 50),  # Number of trees
    'max_depth': np.arange(10, 100, 10),  # Maximum depth of trees
    'min_samples_split': np.arange(2, 21, 2),  # Minimum samples required to split
    'min_samples_leaf': np.arange(1, 11, 2),  # Minimum samples required at a leaf
    'max_features': ['sqrt', 'log2'],  # Feature selection strategy
    'bootstrap': [True, False],  # Bootstrap sampling
    'criterion': ["gini", "entropy"],  # Splitting criteria
}

# Use RandomizedSearchCV to find the best hyperparameters
random_search_rf = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist_rf, 
    n_iter=100,  # Try 100 random combinations
    cv=5, n_jobs=-1, verbose=2, random_state=42
)

# Fit the model
random_search_rf.fit(X_train, y_train)

# Best parameters found by RandomizedSearchCV
print(f"Best parameters: {random_search_rf.best_params_}")

# Evaluate on test set
y_pred_rf = random_search_rf.predict(X_test)
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred_rf)}")



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Best XGBoost Parameters
best_xgb_params = {
    "n_estimators": 70,  # Adjusted number of boosting rounds
    "max_depth": 7,  # Balanced depth to avoid overfitting
    "learning_rate": 0.0775,  # Tuned learning rate
    "subsample": 0.8,  # Percentage of samples used per boosting round
    "colsample_bytree": 0.5,  # Percentage of features used per tree
    "min_child_weight": 3,  # Minimum weight sum for leaves to prevent overfitting
    "gamma": 0.5,  # Minimum loss reduction to split
    "reg_lambda": 5,  # L2 regularization (reduces overfitting)
    "reg_alpha": 5  # L1 regularization (reduces overfitting)
}

# Train the XGBoost Model
xgb = XGBClassifier(**best_xgb_params, random_state=42)
xgb.fit(X_train, y_train)

# Predictions
y_pred_rf = xgb.predict(X_test)
y_prob_rf = xgb.predict_proba(X_test)[:, 1]  # For ROC-AUC


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average="macro")
recall = recall_score(y_test, y_pred_rf, average="macro")
f1 = f1_score(y_test, y_pred_rf, average="macro")
roc_auc = roc_auc_score(y_test, y_prob_rf, multi_class="ovr")

# Print Metrics
print(f"ðŸ”¹ Random Forest Classifier Metrics:")
print(f"âœ… Accuracy: {accuracy:.4f}")
print(f"âœ… Precision: {precision:.4f}")
print(f"âœ… Recall: {recall:.4f}")
print(f"âœ… F1-Score: {f1:.4f}")
print(f"âœ… ROC-AUC Score: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



# using cross validation
from sklearn.model_selection import cross_val_score

# Perform 5-Fold Cross Validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")

# Print Results
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")



train_accuracy = accuracy_score(y_train, rf.predict(X_train))
test_accuracy = accuracy_score(y_test, rf.predict(X_test))

print(f"âœ… Training Accuracy: {train_accuracy:.4f}")
print(f"âœ… Test Accuracy: {test_accuracy:.4f}")



# Load the test dataset
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')


df_test.isnull().sum()



from sklearn.impute import SimpleImputer

# Impute missing values in the test dataset with the median of the column
imputer = SimpleImputer(strategy='median')
df_test_imputed = imputer.fit_transform(df_test)

y_pred_stack = rf.predict(df_test_imputed)

submission = pd.DataFrame({
    'id': df_test.index,  # Correct usage of the index
    'rainfall': y_pred_stack
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file saved as 'submission.csv'")





