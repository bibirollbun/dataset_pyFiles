# install catboost library
!pip install catboost -q


# to handle data
import pandas as pd
import numpy as np

# to visualize data
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# to preprocess data

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



# machine learning tasks
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# metrics
from sklearn.metrics import accuracy_score,precision_score, f1_score ,recall_score ,confusion_matrix, classification_report, mean_absolute_error,mean_squared_error,r2_score

# ignore warnings

import warnings
warnings.filterwarnings('ignore')


# train data

df_train = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/train.csv')
df_train.head()


# test data

df_test = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/test.csv')
df_test.head()


# sample_submission file

df_sample = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/sample_submission.csv')
df_sample.head()


print('In Train dataset')
print(f'Number of rows: {df_train.shape[0]}')
print(f'Number of columns: {df_train.shape[1]}')
print('------------------------')
print('In Test dataset')
print(f'Number of rows: {df_test.shape[0]}')
print(f'Number of columns: {df_test.shape[1]}')


# columns in train

df_train.columns


# columns in test

df_test.columns


# Information about train dataset

df_train.info()


# Information about test dataset

df_test.info()


# Summary of train dataset in transpose

df_train.describe().T


# Unique values in train datset

print('Unique values in Train dataset\n')
print(df_train.nunique())


# Check null values

print(df_train.isnull().sum().sort_values(ascending=False)/len(df_train)*100)


# plot it using seaborn

fig = plt.figure(figsize=(12, 6))
sns.heatmap(df_train.isnull(), cmap='magma', annot=False, fmt='.2f', linewidths=.5)
plt.title('Missing Values Heatmap', fontsize=16)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Rows', fontsize=12)

plt.show()


df_train.info()


# split data into numerical & categorical columns

num_cols = [col for col in df_train.columns if df_train[col].dtype!='O']
cat_cols = [col for col in df_train.columns if col not in num_cols]



# numerical columns

num_cols


# make Boxplot of numeric columns using for loop
plt.figure(figsize=(22, 32))

# Extend the colors list to have at least as many colors as num_cols
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'brown', 'cyan', 'magenta','pink','lightblue']

# Calculate the number of rows needed based on the number of columns
num_rows = (len(num_cols) + 1) // 2  # Divide by 2 and round up

for i, col in enumerate(num_cols):
    plt.subplot(num_rows, 4, i+1)  # Adjusted to dynamic rows, 2 columns
    sns.boxplot(x=df_train[col], color=colors[i % len(colors)]) # Use modulo operator to cycle through colors
    plt.title(col)
plt.show()


df = df_train.copy()


# Explore age column

# histplot of age using seaborn

fig = plt.figure(figsize=(12,6))
sns.histplot(df['Age'], kde=True)
plt.axvline(df['Age'].mean(),color='red')
plt.axvline(df['Age'].median(),color='green')
plt.axvline(df['Age'].mode()[0],color='blue')
plt.title('Age Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['Age'].mean())
print('Median',df['Age'].median())
print('Mode',df['Age'].mode())



# Explore Balance column

# histplot of age using seaborn

fig = plt.figure(figsize=(12,6))
sns.histplot(df['Balance'], kde=True)
plt.axvline(df['Balance'].mean(),color='red')
plt.axvline(df['Balance'].median(),color='green')
plt.axvline(df['Balance'].mode()[0],color='blue')
plt.title('Balance Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['Balance'].mean())
print('Median',df['Balance'].median())
print('Mode',df['Balance'].mode())



# Explore EstimatedSalary column

# histplot of EstimatedSalary using seaborn

fig = plt.figure(figsize=(36,10))
sns.histplot(df['EstimatedSalary'], kde=True)
plt.axvline(df['EstimatedSalary'].mean(),color='red')
plt.axvline(df['EstimatedSalary'].median(),color='green')
plt.axvline(df['EstimatedSalary'].mode()[0],color='blue')
plt.title('EstimatedSalary Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['EstimatedSalary'].mean())
print('Median',df['EstimatedSalary'].median())
print('Mode',df['EstimatedSalary'].mode())



# Explore Credit Score column

# histplot of Credit Score using seaborn

fig = plt.figure(figsize=(12,6))
sns.histplot(df['CreditScore'], kde=True)
plt.axvline(df['CreditScore'].mean(),color='red')
plt.axvline(df['CreditScore'].median(),color='green')
plt.axvline(df['CreditScore'].mode()[0],color='blue')
plt.title('Credit Score Distribution')
plt.show()

# print the values of mean, median & mode
print("-----------------------")
print('Mean',df['CreditScore'].mean())
print('Median',df['CreditScore'].median())
print('Mode',df['CreditScore'].mode())



cat_cols


# countplot of Geography

fig = plt.figure(figsize=(12, 6))
sns.countplot(df, x='Geography', palette='viridis')  # Use a color palette
plt.title('Countplot of Geography', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Geography', fontsize=14)  # Enhance x-axis label
plt.ylabel('Count', fontsize=14)  # Enhance y-axis label
plt.xticks(fontsize=10)  # Enhance x-axis tick labels
plt.yticks(fontsize=10)  # Enhance y-axis tick labels
sns.despine()  # Remove top and right spines for a cleaner look

plt.show()


# countplot of Gender

fig = plt.figure(figsize=(12,6))
sns.countplot(df,x ='Gender',palette='viridis')
plt.title('Countplot of Gender', fontsize=16, fontweight='medium')
plt.ylabel('Count', fontsize=14)  # Enhance y-axis label
plt.xticks(fontsize=10)  # Enhance x-axis tick labels
plt.yticks(fontsize=10)  # Enhance y-axis tick labels
sns.despine()  # Remove top and right spines for a cleaner look
plt.show()


# countplot of Geography based on Gender

fig = plt.figure(figsize=(12, 6))
sns.countplot(df, x='Geography', palette='viridis', hue='Gender')  # Use a color palette
plt.title('Countplot of Geography', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Geography', fontsize=14)  # Enhance x-axis label
plt.ylabel('Count', fontsize=14)  # Enhance y-axis label
plt.xticks(fontsize=10)  # Enhance x-axis tick labels
plt.yticks(fontsize=10)  # Enhance y-axis tick labels
sns.despine()  # Remove top and right spines for a cleaner look


plt.show()


df.describe()


# Scale creditscore, balance, estimatedsalary in train data using standard scalar

df['CreditScore'] = StandardScaler().fit_transform(df[['CreditScore']])
df['Balance'] = StandardScaler().fit_transform(df[['Balance']])
df['EstimatedSalary'] = StandardScaler().fit_transform(df[['EstimatedSalary']])



# encode categorical columns in train data separately using label encoder

df['Geography'] = LabelEncoder().fit_transform(df['Geography'])
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])



df.head()


# Scale creditscore, balance, estimatedsalary in test data using standard scalar

df_test['CreditScore'] = StandardScaler().fit_transform(df_test[['CreditScore']])
df_test['Balance'] = StandardScaler().fit_transform(df_test[['Balance']])
df_test['EstimatedSalary'] = StandardScaler().fit_transform(df_test[['EstimatedSalary']])



# encode categorical columns in test data separately using label encoder

df_test['Geography'] = LabelEncoder().fit_transform(df_test['Geography'])
df_test['Gender'] = LabelEncoder().fit_transform(df_test['Gender'])



df_test.head()


# Check Columns

df.columns


# Define features and target

X = df.drop(['id', 'CustomerId', 'Surname','Exited'], axis=1)
y = df['Exited']

df_test = df_test.drop(['id', 'CustomerId', 'Surname'], axis=1)


# Spilit the data into X train and y train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X.head()


# Create a dictionaries of list of models to evaluate performance
models = {
          'LogisticRegression' : LogisticRegression(random_state=42),
          'SVC' : SVC(random_state=42),
          'DecisionTreeClassifier' :DecisionTreeClassifier(random_state=42),
          'RandomForestClassifier' :RandomForestClassifier(random_state=42),
          'KNeighborsClassifier' : KNeighborsClassifier(),
          'GradientBoostingClassifier' : GradientBoostingClassifier(random_state=42),
          'XGBClassifier' : XGBClassifier(),
          'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
          'GaussianNB': GaussianNB(), # Changed to only the model
          'LGBMClassifier': LGBMClassifier(verbose =-1, random_state=42),
          'CatBoostClassifier': CatBoostClassifier(verbose=0, random_state=42)
          }

# train and predict each model with evaluation metrics as well making a for loop to iterate over the models

model_scores = []
for name, model in models.items():
    # fit each model from models on training data
    model.fit(X_train, y_train)

    # make prediction from each model
    y_pred = model.predict(X_test)
    metric = mean_absolute_error(y_test, y_pred)
    model_scores.append((name, metric))

    # print the performing metric
    print(name, 'MSE: ', mean_squared_error(y_test, y_pred))
    print(name, 'R2: ', r2_score(y_test, y_pred))
    print(name, 'MAE: ', mean_absolute_error(y_test, y_pred))
    print('\n')

# selecting the best model from all above models with evaluation metrics sorting method
refine_models = sorted(model_scores, key=lambda x: x[1], reverse=False)
for model in refine_models:
    print('Mean absolute error for', f"{model[0]} is {model[1]: .2f}")


# Create a dictionaries of list of models to evaluate performance
models = {
          'LogisticRegression' : LogisticRegression(random_state=42),
          'SVC' : SVC(random_state=42),
          'DecisionTreeClassifier' :DecisionTreeClassifier(random_state=42),
          'RandomForestClassifier' :RandomForestClassifier(random_state=42,class_weight='balanced'),
          'KNeighborsClassifier' : KNeighborsClassifier(),
          'GradientBoostingClassifier' : GradientBoostingClassifier(random_state=42),
          'XGBClassifier' : XGBClassifier(),
          'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
          'GaussianNB': GaussianNB(), # Changed to only the model
          'LGBMClassifier': LGBMClassifier(verbose =-1, random_state=42),
          'CatBoostClassifier': CatBoostClassifier(verbose=0, random_state=42)
          }

# train and predict each model with evaluation metrics as well making a for loop to iterate over the models

model_scores = []
for name, model in models.items():
    # fit each model from models on training data
    model.fit(X_train, y_train)

    # make prediction from each model
    y_pred = model.predict(X_test)
    metric = accuracy_score(y_test, y_pred)
    model_scores.append((name, metric))

    # print the performing metric
    print(name,'Accuracy score: ', accuracy_score(y_test, y_pred))
    print(name,'Precision score: ', precision_score(y_test, y_pred, average='micro'))
    print(name,'Recall score: ', recall_score(y_test, y_pred, average='micro'))
    print(name,'F1 score: ', f1_score(y_test, y_pred, average='micro'))
    print('\n')

# selecting the best model from all above models with evaluation metrics sorting method
refine_models = sorted(model_scores, key=lambda x: x[1], reverse=True)
for model in refine_models:
    print('Accuracy Score for', f"{model[0]} is {model[1]: .2f}")


# Create a dictionaries of list of models to evaluate performance

models = {
    'LogisticRegression': (LogisticRegression(random_state=42), {'model__penalty': ['l1', 'l2'],'model__C': [0.001, 0.1, 1],'model__solver': ['liblinear', 'saga']}),
    'SVC': (SVC(random_state=42), {'model__kernel': ['linear'],'model__degree': [2]}),
    'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [None, 5, 10], 'model__splitter': ['best', 'random']}),
    'RandomForestClassifier': (RandomForestClassifier(random_state=42), {'model__n_estimators': [10, 100, 1000], 'model__max_depth': [None, 5, 10]}),
    'KNeighborsClassifier': (KNeighborsClassifier(), {'model__n_neighbors': np.arange(3, 100, 2), 'model__weights': ['uniform', 'distance']}),
    'GaussianNB': (GaussianNB(), {'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}),
    'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), {'model__loss': ['log_loss', 'exponential'], 'model__n_estimators': [10, 100, 1000]}),
    'AdaBoostClassifier': (AdaBoostClassifier(random_state=42), {'model__n_estimators': [10, 100, 1000], 'model__learning_rate': [0.1, 0.01, 0.001]}),
    'LGBMClassifier': (LGBMClassifier(max_depth=10,min_data_in_leaf=20,num_leaves=31,learning_rate=0.01,n_estimators=200,lambda_l1=0.1,lambda_l2=0.1,boosting_type='gbdt'), {}),
    'CatBoostClassifier': (CatBoostClassifier(verbose=0, random_state=42), {'model__iterations': [100, 500, 1000], 'model__learning_rate': [0.01, 0.1, 1.0]}),
    'XGBClassifier': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {}),
}

results = []

# Train and predict each model with evaluation metrics
for name, (model, params) in models.items():
    # Create a pipeline with the model
    pipeline = Pipeline(steps=[('model', model)])

    # Create a grid search CV to tune the hyperparameters
    grid_search = GridSearchCV(pipeline, params, cv=5)

    # Fit the pipeline
    grid_search.fit(X_train, y_train)

    # Make predictions
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # print the performing metric
    print(name, 'MSE: ', mean_squared_error(y_test, y_pred))
    print(name, 'R2: ', r2_score(y_test, y_pred))
    print(name, 'MAE: ', mean_absolute_error(y_test, y_pred))
    print('\n')


    # Store results
    results.append({"Model": name, "MSE": mse, "R2": r2, "MAE": mae})

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Select the best model based on the lowest MSE
best_model = results_df.loc[results_df['MAE'].idxmin()]

print("\nBest Model:")
print(best_model)


# plot the confusion matrix

fig = plt.figure(figsize=(12, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Predicted', fontsize=12)  # Enhance x-axis label
plt.ylabel('True', fontsize=12)  # Enhance y-axis label

plt.show()


# Create a dictionaries of list of models to evaluate performance

models = {
    'LogisticRegression': (LogisticRegression(random_state=42), {'model__penalty': ['l1', 'l2'],'model__C': [0.001, 0.1, 1],'model__solver': ['liblinear', 'saga']}),
    'SVC': (SVC(random_state=42), {'model__C': [0.1, 1, 10],'model__kernel': ['linear','sigmoid'],'model__degree': [2,4], }),
    'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {'model__criterion': ['gini', 'entropy'],'model__max_depth': [None, 5, 10], 'model__splitter': ['best', 'random']}),
    'RandomForestClassifier': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'model__n_estimators': [10, 100, 1000], 'model__max_depth': [None, 5, 10],'model__min_samples_split': [2, 5, 10]}), # Changed here to model__min_samples_split
    'KNeighborsClassifier': (KNeighborsClassifier(), {'model__n_neighbors': np.arange(3, 100, 2), 'model__weights': ['uniform', 'distance']}),
    'GaussianNB': (GaussianNB(), {'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}),
    'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), {'model__loss': ['log_loss', 'exponential'], 'model__n_estimators': [10, 100, 1000],'model__learning_rate': [0.01, 0.1, 0.2],'model__max_depth': [3, 4, 5]}),
    'AdaBoostClassifier': (AdaBoostClassifier(random_state=42), {'model__n_estimators': [10, 100, 1000], 'model__learning_rate': [0.1, 0.01, 0.001],'model__algorithm': ['SAMME', 'SAMME.R']}),
    'LGBMClassifier': (LGBMClassifier(random_state=42), {'model__boosting_type': ['gbdt', 'dart', 'goss'],'model__learning_rate': [0.01, 0.1, 0.2],'model__n_estimators': [100, 200, 300],'model__max_depth': [-1, 5, 10],}),
    'CatBoostClassifier': (CatBoostClassifier(verbose=0, random_state=42), {'model__iterations': [100, 500, 1000], 'model__learning_rate': [0.01, 0.1, 1.0],'model__depth': [4, 6, 8],}),
    #'XGBClassifier': (XGBClassifier(random_state= 42,use_label_encoder=False, eval_metric='logloss'), {'max_depth': [3, 5, 7],'learning_rate': [0.1, 0.01, 0.001],'n_estimators': [100, 500, 1000],'gamma': [0, 0.1, 0.2],'subsample': [0.8, 0.9, 1.0]}),
}

results = []

# Train and predict each model with evaluation metrics
for name, (model, params) in models.items():
    # Create a pipeline with the model
    pipeline = Pipeline(steps=[('model', model)])

    # Create a grid search CV to tune the hyperparameters
    grid_search = GridSearchCV(pipeline, params, cv=5)

    # Fit the pipeline
    grid_search.fit(X_train, y_train)

    # Make predictions
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    Accuracy_Score = accuracy_score(y_test, y_pred)
    Precision_Score = precision_score(y_test, y_pred)
    Recall_Score= recall_score(y_test, y_pred)
    F1_Score = f1_score(y_test, y_pred)


    # print the performing metric
    print(name,'Accuracy score: ', accuracy_score(y_test, y_pred))
    print(name,'Precision score: ', precision_score(y_test, y_pred, average='micro'))
    print(name,'Recall score: ', recall_score(y_test, y_pred, average='micro'))
    print(name,'F1 score: ', f1_score(y_test, y_pred, average='micro'))
    print('\n')

    # Print the performing metrics
    #print(f"{name} - MSE: {mse}, R2: {r2}, MAE: {mae}")

    # Store results
    results.append({
        "Model": name,
        "Accuracy score": Accuracy_Score,  # Use the calculated Accuracy_Score
        "Precision score": Precision_Score,  # Use the calculated Precision_Score
        "Recall score": Recall_Score,  # Use the calculated Recall_Score
        "F1 score": F1_Score  # Use the calculated F1_Score
    })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Select the best model based on the lowest MSE
best_model = results_df.loc[results_df['Accuracy score'].idxmax()]

print("\nBest Model:")
print(best_model)


# Sort the results DataFrame by MSE in ascending order
sorted_results_df = results_df.sort_values(by='Accuracy score')

# Set the aesthetics of the plot
sns.set(style="darkgrid", palette="pastel")

# Create a bar plot
plt.figure(figsize=(12, 8))
bars = plt.bar(sorted_results_df['Model'], sorted_results_df['Accuracy score'], color=sns.color_palette("viridis", len(sorted_results_df)))

# Highlight the best model
best_model_index = sorted_results_df['Accuracy score'].idxmax()

# Adding labels and title with enhanced font styling
plt.xlabel('Models', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score ', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison ', fontsize=16, fontweight='bold')
plt.xticks(rotation=90, fontsize=12, fontweight='medium')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding a shadow effect to the bars
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_alpha(0.9)  # Slight transparency for better visibility

# Add data labels on top of the bars with more styling
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2),
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Show the plot with a tight layout
plt.tight_layout()
plt.show()


# plot the confusion matrix

fig = plt.figure(figsize=(12, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix', fontsize=16, fontweight='medium')  # Enhance title
plt.xlabel('Predicted', fontsize=12)  # Enhance x-axis label
plt.ylabel('True', fontsize=12)  # Enhance y-axis label

plt.show()


# create a submission file

model = grid_search.best_estimator_  # Get the best model from grid search

# Ensure y_pred has the same length as df_sample
y_pred_full = model.predict(df_test) # Predict on the entire test data
y_pred_full = (y_pred_full > 0.5).astype(int)

df_sample['Exited'] = y_pred_full  # Assign the full predictions to the DataFrame
df_sample.to_csv('submission.csv', index=False)

