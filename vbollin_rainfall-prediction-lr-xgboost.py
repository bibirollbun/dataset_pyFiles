import os
import torch
import scipy
import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
import xgboost as xgb
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


for dirpath, _,  filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirpath, filename))
print(f'Directory path: {dirpath}')
print(f'Filename: {filenames}')


warnings.simplefilter("ignore")


train_data_link = '/kaggle/input/playground-series-s5e3/train.csv'

test_data = '/kaggle/input/playground-series-s5e3/test.csv'

df = pd.read_csv(train_data_link)

print(df.dtypes)


testdata = pd.read_csv(test_data)

print(testdata.dtypes)

missing_test_data = pd.isna(testdata).sum()

missing_test_data


# since there is only one missing value and that doesn't impact the testing significantly so replaced the missing value with medium
testdata['winddirection'].fillna(testdata['winddirection'].median(), inplace=True)


df.head(10)


missing_values = pd.isnull(df).sum()

# missed_values = pd.isna(df).sum()

missing_values
# missed_values


df.shape



x = df

print(x.dtypes)


new_features = {
    "rain_predictor": x["humidity"] * x["cloud"],
    "temp_range": x["maxtemp"] - x["mintemp"],
    "due_spread": x["temparature"] - x["dewpoint"],
}

# adding new columns for the train data
for feature, values in new_features.items():
    if feature not in x.columns:
        x[feature] = values

# adding new columns for the test data
for f, v in new_features.items():
    if f not in testdata.columns:
        testdata[f] = v


print(df.columns)

# separated target variable 
y = df['rainfall']
# print(y)


# correlation values between the variables
x_corr = x.corr()
print(x_corr)


# plotting the correlation matrix heatmap
plt.figure(figsize=(12, 10))

# creating mask for only to show the lower triangle values
mask = np.triu(np.ones_like(x_corr)) 

sns.heatmap(x_corr, cmap="YlGnBu", annot=True, mask=mask)

plt.title("Correlation HeatMap", weight='bold', fontsize=22)
# Display heatmap
plt.show()


# summary statistics of all variables
x.describe()


df.info()


print(testdata)


# created a list of features with out the features day and rainfall
features_list = ['pressure', 'maxtemp', 'temparature','mintemp','dewpoint', 'humidity','cloud','sunshine','winddirection','windspeed', 'rain_predictor',
                'temp_range', 'due_spread']


# Univariate analysis of the features

for feature in features_list:
    plt.figure(figsize=(10,8))
    sns.histplot(data=x, x=feature)
    plt.title(f'Univariate analysis for {feature}')

# Box plot distribution of the features
for feature in features_list:
    plt.figure(figsize=(10,8))
    sns.boxplot(data=x, x=feature)
    plt.title(f'Boxplot analysis for {feature}')


# cheking the skewness of the data
for feat in features_list:
    skewness = skew(x[feat])
    if skewness == 0:
        print(f"Skewness of {feat} is {skewness} Normal distributed")
    elif skewness > 0:
        print(f"Skewness of {feat} is {skewness} having Right Skewness")
    else:
        print(f"Skewness of {feat} is {skewness} having Left Skewness")


# Checking the importance of each variable with target variable 
for feat in features_list:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=x['rainfall'], y = x[feat])
    plt.title(f'Rainfall Vs {feat} analysis')
    plt.show()


print(features_list)


# Checking the importance of each variable with target variable 
for feat in features_list:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=x['rainfall'], y = x[feat])
    plt.title(f'Rainfall Vs {feat} analysis')
    plt.show()


# Plotted the pairplot between each variable to indetify the rainfall
sns.pairplot(x, hue="rainfall")
plt.show()


for feat in features_list:
    plt.figure(figsize=(8,6))
    sns.violinplot(x=x['rainfall'], y=x[feat], palette = 'coolwarm')
    plt.title(f"Distribution of {feat} by Rainfall")
    plt.show()


fig = px.scatter_3d(x, x="rain_predictor", y="humidity", z="cloud", color=x["rainfall"],
                    title="3D Scatter: rain_predictor vs Humidity vs cloud ",
                    labels={"rainfall": "Rainfall"},
                    opacity=0.2)
fig.show()


X_train = x.drop(columns=['id','rainfall'])
Y_train = x['rainfall']

X_test = testdata.drop(columns=['id'])

scaler = MinMaxScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test) 

model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42, solver="liblinear")

auc_scores = cross_val_score(model, scaled_X_train, Y_train, cv=10, scoring='roc_auc')

print(f"Mean AUC: {auc_scores.mean():.6f}")

model.fit(scaled_X_train, Y_train)

Y_test_pred = model.predict_proba(scaled_X_test)[:, 1]

assert "id" in testdata.columns, "Error: 'id' column missing in test dataset"

# # Create submission DataFrame
# submission = pd.DataFrame({
#     "id": testdata["id"],
#     "rainfall": Y_test_pred
# })


# # Save to CSV
# submission.to_csv("submission_logreg.csv", index=False)

# print("Submission file saved successfully.")


# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 100, 200], 
    'max_iter': [100, 200, 500],         
    'penalty': ['l1', 'l2'], 
}

grid_search = GridSearchCV(model, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)

grid_search.fit(scaled_X_train, Y_train)


print("Best Parameters:", grid_search.best_params_)

print(f"Best Mean AUC: {grid_search.best_score_:.6f}")

best_model = grid_search.best_estimator_

Y_test_pred_grid = best_model.predict_proba(scaled_X_test)[:, 1]

# Ensure 'id' column exists in testdata
assert "id" in testdata.columns, "Error: 'id' column missing in test dataset"

# Create submission DataFrame
submission = pd.DataFrame({
    "id": testdata["id"],
    "rainfall": Y_test_pred_grid
})

# Save to CSV
submission.to_csv("submission_logistic_with_gridsearch.csv", index=False)



xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=1,
    booster='gbtree'
)


xgb_auc_scores = cross_val_score(xgb_model, scaled_X_train, Y_train, cv=10, scoring='roc_auc')

print(f"Mean AUC: {xgb_auc_scores.mean():.6f}")


xgb_model.fit(scaled_X_train, Y_train)


Y_test_pred_xgb = xgb_model.predict_proba(scaled_X_test)[:, 1]





param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4],
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1)


grid_search.fit(scaled_X_train, Y_train)


print("Best Parameters:", grid_search.best_params_)
print(f"Best Mean AUC: {grid_search.best_score_:.6f}")


best_xgb_model = grid_search.best_estimator_
Y_test_pred_xgb_grid = best_xgb_model.predict_proba(scaled_X_test)[:, 1]


assert "id" in testdata.columns, "Error: 'id' column missing in test dataset"


submission = pd.DataFrame({
    "id": testdata["id"],
    "rainfall": Y_test_pred_xgb_grid
})


submission.to_csv("submission_xgboost_with_gridsearch.csv", index=False)

print("Submission file saved successfully.")


import os

# Print the current working directory
print(os.getcwd())


import os

# List all files in the '/kaggle/working' directory
files = os.listdir('/kaggle/working')
print(files)




