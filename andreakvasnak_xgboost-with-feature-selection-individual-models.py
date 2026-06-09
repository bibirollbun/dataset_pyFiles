# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import warnings 
warnings.filterwarnings('ignore')

# Import Model Packages
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# importing model fit metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score






#Create Dataframe of the Sample Submission
#This is the format that your submission to Kaggle will need to be in for Kaggle to accept it
sample_sub=pd.read_excel(f"/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")

sample_sub.head()


#Create Dataframes of the Test Datasets of Data Dictionary
test_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")


# Join test_categorical and test_quant on participant_id
# Perform the join on the 'ID' feature
test_cat_quant = pd.merge(test_categorical, test_quant, on='participant_id', how='inner')


#Create Dataframes of the Train Datasets of Data Dictionary
train_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
train_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx")
train_solution=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")



# Join train_categorical and train_quant on participant_id
# Perform the join on the 'ID' feature
train_cat_quant = pd.merge(train_categorical, train_quant, on='participant_id', how='inner')


# Before combining the datasets, create a new field in each that identifies the source of the data, 
# this will allow the datasets to be split after data cleaning
test_cat_quant['Dataset'] = 'Test'
train_cat_quant['Dataset'] = 'Train'

# Union the two DataFrames
combined_cat = pd.concat([test_cat_quant, train_cat_quant], ignore_index=True)

combined_cat.head()


# Fill nulls with median
# Remember to check other methods for filling nulls and does not have to be same method for all features
# Create a copy of the dataframe
combined_cat2 = combined_cat.copy()  # Create a copy of the DataFrame
combined_cat2 = combined_cat2.fillna(combined_cat.median(numeric_only=True))

combined_cat2.info()


# Splitting the dataframe based on a condition for a specific feature
feature_name = 'Dataset'  # Example feature to split on
value_to_split = 'Test'  # Value to filter on

# Create two dataframes based on the condition
df_split_test = combined_cat2[combined_cat2[feature_name] == value_to_split]
df_split_train = combined_cat2[combined_cat2[feature_name] != value_to_split]



# Append Solution to df_split_train
train_solution = pd.merge(df_split_train, train_solution, on='participant_id', how='inner')
train_solution.info()


#drop dataset from train_solution
train_solution = train_solution.drop(columns=['Dataset'])


# Append connectnomes to train_solution
train_conn_solution = pd.merge(train_solution,train_connectome, on='participant_id', how='inner')
train_conn_solution.info()


# Append connectnomes to df_split_test
test_conn = pd.merge(df_split_test,test_connectome, on='participant_id', how='inner')
test_conn.info()


# Drop 'participant_id','ADHD_Outcome', 'Sex_F'  from train data
X_train = train_conn_solution.drop(columns=['participant_id', 'ADHD_Outcome', 'Sex_F'])
y_train_sex = train_conn_solution[['Sex_F']]


# Prep test data
X_test = test_conn.drop(columns=['participant_id','Dataset'])


# Identify k best variables
selector = SelectKBest(f_classif, k=20)

# Fit the selector and transform the training data
X_new = selector.fit_transform(X_train, y_train_sex)

# Get a boolean mask of the selected features
mask = selector.get_support()

# Use the mask to extract the feature names from the DataFrame
selected_features = X_train.columns[mask]
print("Selected features:", selected_features)




# Split the train dataset
# This allows us to train on part of the data and the test on another part that has the target value unlike the test dataset supplied in comp
#set seed so results are repeatable
train_X, val_X, train_Y, val_Y= train_test_split(X_train, y_train_sex, random_state=1618)

# Instantiate the XGBoost classifier
model = XGBClassifier(seed=42,
    objective='binary:logistic',
    #gamma=0,
    #use_label_encoder=False, 
    eval_metric='error',
    max_depth=10,
    #n_estimators=6,
    learning_rate=0.05,
    #subsample=0.5,
    #colsample_bytree=0.5
                     )

# Fit the model using only the selected features
model.fit(train_X[selected_features], train_Y)

# Now, to make predictions, use the same features on the validation set
predictions = model.predict(val_X[selected_features])



#Evaluate how well the model fit the actual and predicted values of the validation dataset

# Calculate Accuracy
accuracy = accuracy_score(val_Y, predictions)
print("Accuracy:", accuracy)

# Generate Confusion Matrix
cm = confusion_matrix(val_Y, predictions)
print("Confusion Matrix:\n", cm)

# Print a Detailed Classification Report
report = classification_report(val_Y, predictions)
print("Classification Report:\n", report)

# If your model supports predicting probabilities (which XGBoost does), you can calculate ROC AUC
y_probs = model.predict_proba(val_X[selected_features])[:, 1]  # Probability of the positive class
roc_auc = roc_auc_score(val_Y, y_probs)
print("ROC AUC:", roc_auc)



# Fit seems decent, time to apply to the actual test dataset
predictions_gender = model.predict(X_test[selected_features])
predictions_gender


y_train_adhd = train_conn_solution[['ADHD_Outcome']]

# Identify k best variables
selector2 = SelectKBest(f_classif, k=2000)

# Fit the selector and transform the training data
X_new2 = selector2.fit_transform(X_train, y_train_adhd)

# Get a boolean mask of the selected features
mask2 = selector.get_support()

# Use the mask to extract the feature names from the DataFrame
selected_features2 = X_train.columns[mask2]
print("Selected features:", selected_features2)


# Split the train dataset
# This allows us to train on part of the data and the test on another part that has the target value unlike the test dataset supplied in comp
#set seed so results are repeatable
train_X2, val_X2, train_Y2, val_Y2= train_test_split(X_train, y_train_adhd, random_state=1618)

# Set up the XGBoost Parameters
# note that there are some texted out parameters, try different versions of the model 
# by changing the parameters by untexting or even changing values

model2 = XGBClassifier(seed=42,
    objective='binary:logistic',
    gamma=0,
    #use_label_encoder=False, 
    #eval_metric='error',
    #eval_metric='auc',   
    eval_metric='aucpr'
    #max_depth=10,
    #n_estimators=6,
    #learning_rate=0.05,
    #subsample=0.5,
    #colsample_bytree=0.5
                     )

# Fit the model using only the selected features
model2.fit(train_X[selected_features2], train_Y2)

# Now, to make predictions, use the same features on the validation set
predictions2 = model2.predict(val_X2[selected_features2])



#Evaluate how well the model fit the actual and predicted values of the validation dataset

# Calculate Accuracy
accuracy2 = accuracy_score(val_Y2, predictions2)
print("Accuracy:", accuracy2)

# Generate Confusion Matrix
cm2 = confusion_matrix(val_Y2, predictions2)
print("Confusion Matrix:\n", cm2)

# Print a Detailed Classification Report
report2 = classification_report(val_Y2, predictions2)
print("Classification Report:\n", report)

# If your model supports predicting probabilities (which XGBoost does), you can calculate ROC AUC
y_probs2 = model2.predict_proba(val_X2[selected_features2])[:, 1]  # Probability of the positive class
roc_auc2 = roc_auc_score(val_Y2, y_probs2)
print("ROC AUC:", roc_auc2)



# Fit seems decent, time to apply to the actual test dataset
predictions_adhd = model2.predict(X_test[selected_features2])
predictions_adhd


# Create a DataFrame with participant_id and predictions
test_predictions = test_conn[['participant_id']].copy()
test_predictions['ADHD_Outcome'] = predictions_adhd
test_predictions['Sex_F'] = predictions_gender

# Save results to CSV
test_predictions.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")


