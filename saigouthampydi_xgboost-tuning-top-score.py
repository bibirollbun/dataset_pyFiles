# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib.pyplot as plt 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#coz why not?
import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


#lets just drop of the 'id' column
df_train=df_train.drop('id',axis=1)



df_train.head()



df_train.shape


#lucky us, we got not missing values in our training set
df_train.info()


#lets just divide the features into categorical and numerical 
cat_cols=df_train.select_dtypes(include=['object']).columns
num_cols=df_train.select_dtypes(include=['int']).columns

print(f'Number of Categorical Columns in the data is {len(cat_cols)}')
print(f'Number of Numerical Columns in the data is {len(num_cols)}')



for i in num_cols: 
    plt.hist(df_train[i],bins='auto')
    plt.title(f'Histogram of {i}')
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(8, 5)) # Create a new figure for each plot
    sns.countplot(x=col, data=df_train.sort_values(col), palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Countplot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()



correlation_matrix = df_train[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,      
    cmap='coolwarm', 
    fmt=".2f",       
    linewidths=.5,   
    cbar=True        
)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)              
plt.tight_layout()                  
plt.show()








# Separate features and target

X=df_train.drop(['y'],axis=1)
y=df_train.y



# lets get those categorical columns some dummies 
X=pd.concat([X,pd.get_dummies(X[cat_cols],drop_first=True)],axis=1)
X.drop(cat_cols,inplace=True,axis=1)



#performing the same step on the testing data 
df_test=pd.concat([df_test,pd.get_dummies(df_test[cat_cols],drop_first=True)],axis=1)
df_test.drop(cat_cols,axis=1,inplace=True)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

negative_class_count = sum(y_train == 0)
positive_class_count = sum(y_train == 1)


xgb_classifier = xgb.XGBClassifier(objective='binary:logistic',  
                                   n_estimators=1000,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   use_label_encoder=False, 
                                   eval_metric='logloss',random_state=42)

print("Training the XGBoost model...")
xgb_classifier.fit(X_train, y_train)
print("Training complete.")

y_pred = xgb_classifier.predict(X_test)



#Let the model evaluation begin
print("\nModel Evaluation:")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature Importances:")
feature_importances = xgb_classifier.get_booster().get_score(importance_type='weight')
sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importances:
    print(f"  {feature}: {importance}")



import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Instantiate the XGBClassifier
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid,
    scoring='accuracy',  # or 'f1', 'roc_auc', etc. as suitable
    cv=3,                # 3-fold cross-validation
    n_jobs=-1,           # Use all available CPUs
    verbose=2
)

# Fit the grid search
print("Searching for best hyperparameters...")
grid_search.fit(X_train, y_train)
print("Best hyperparameters found:", grid_search.best_params_)

# Use the best estimator from grid search
best_model = grid_search.best_estimator_

# Fit the best model on all training data (optional; GridSearchCV already refits by default)
# best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance (by default, 'weight' importance)
print("\nFeature Importances:")
feature_importances = best_model.get_booster().get_score(importance_type='weight')
sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importances:
    print(f"{feature}: {importance}")



best_model


submission_df = pd.DataFrame({
    'id': df_test.id,
    'probability': best_model.predict_proba(df_test.drop('id',axis=1))[:,1] # Get the probability of the positive class
})


submission_df.head()


submission_df.to_csv('submission.csv',index=False)




