# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20G to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train =pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
t1 = train.copy()
t2 = test.copy()


# pip install ipdb


# pd.set_option("display.max_columns", None)


train


cat_col = train.select_dtypes(include = 'object').columns.drop('Personality')
num_col = train.select_dtypes(include = 'number').columns


target = 'Personality'


train[cat_col]


train[cat_col].isnull().sum()*100/len(train)


test[cat_col].isnull().sum()*100/len(test)


train[cat_col]= train[cat_col].fillna('Not available')
test[cat_col] = test[cat_col].fillna('Not available')


train.Drained_after_socializing.unique()


train[cat_col].isnull().sum()*100/len(train)


test[cat_col].isnull().sum()*100/len(test)


train[num_col].isnull().sum()*100/len(train)


test[num_col].isnull().sum()*100/len(test)


import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(df, column):
    # import ipdb; ipdb.set_trace()
    # import pdb; pdb.set_trace()
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column].dropna(), kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplot(df, column):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column], color='lightblue')
    plt.title(f'Boxplot (Quartiles) of {column}')
    plt.grid(True)
    plt.show()



# for i in num_col:
#     plot_boxplot(train, i)


train[num_col] = train[num_col].fillna(train[num_col].median())
test[num_col] = test[num_col].fillna(test[num_col].median())



train[num_col].isnull().sum()*100/len(train)


test[num_col].isnull().sum()*100/len(test)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# One-hot encode categorical features
train = pd.get_dummies(train, columns=cat_col)
test = pd.get_dummies(test, columns=cat_col)


train.columns


# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score

# # # 1. Load data
# # train = pd.read_csv('train.csv')
# # test = pd.read_csv('test.csv')

features = [ 'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency', 'Stage_fear_No',
       'Stage_fear_Not available', 'Stage_fear_Yes',
       'Drained_after_socializing_No',
       'Drained_after_socializing_Not available',
       'Drained_after_socializing_Yes']
#
# # # 3. Define features and target
# # features = [col for col in train.columns if col not in ['id', 'target']]
X = train[features]
y = train['Personality']

# X = df[features]
# y = df['Personality']

X_test = test[features]

# # 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 5. Stratified K-Fold Cross Validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6. Logistic Regression with cross-validated regularization
lr1 = LogisticRegressionCV(
    Cs=10,
    cv=cv,
    scoring='accuracy',  # or 'roc_auc' if you prefer
    max_iter=2000,
    solver='lbfgs',
    random_state=42
)

# 7. Cross-validation scores
cv_scores = cross_val_score(lr1, X_scaled, y, cv=cv, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"CV Accuracy scores: {cv_scores}")

# 8. Fit on full training data
lr1.fit(X_scaled, y)

# 9. Predict on test data
test_preds = lr1.predict(X_test_scaled)

# 10. Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'target': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")



len(test_preds)


test['id']


# df.Stage_fear.unique()


# X = train[features]
# y = train['Personality']
# X_test = test[features]


# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# # # Load your data
# # train = pd.read_csv("train.csv")
# # test = pd.read_csv("test.csv")

# # # -------------------------------
# # # âœ… 1. Preprocessing & Feature Setup
# # # -------------------------------
# # # Basic imputation (you can expand this later)
# # imputer = SimpleImputer(strategy='mean')
# # train.fillna(train.median(), inplace=True)
# # test.fillna(train.median(), inplace=True)
# features = ['id', 'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
#        'Friends_circle_size', 'Post_frequency', 'Stage_fear_No',
#        'Stage_fear_Not available', 'Stage_fear_Yes',
#        'Drained_after_socializing_No',
#        'Drained_after_socializing_Not available',
#        'Drained_after_socializing_Yes']
# #


# # # Features & Target
# # features = [col for col in train.columns if col not in ['id', 'target']]
# X = train[features]
# y = train['Personality']
# X_test = test[features]

# # -------------------------------
# # âœ… 2. Pipeline: Scaling + Polynomial Interactions + L1 Logistic Regression
# # -------------------------------

# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
#     ('lr', LogisticRegressionCV(
#         Cs=10,
#         cv=5,
#         penalty='l1',
#         solver='liblinear',  # Required for L1
#         scoring='accuracy',
#         max_iter=3000,
#         random_state=42
#     ))
# ])

# # -------------------------------
# # âœ… 3. Cross-Validated Accuracy
# # -------------------------------
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

# print(f"Mean CV Accuracy with L1 + Interaction Features: {cv_scores.mean():.5f}")
# print("Fold-wise Accuracy:", np.round(cv_scores, 5))

# # # -------------------------------
# # # âœ… 4. Train Final Model on Full Data
# # # -------------------------------
# # pipeline.fit(X, y)

# # # -------------------------------
# # # âœ… 5. Predict on Test Set
# # # -------------------------------
# # test_preds = pipeline.predict(X_test)

# # submission = pd.DataFrame({
# #     'id': test['id'],
# #     'target': test_preds
# # })

# # submission.to_csv("submission_poly_l1.csv", index=False)
# # print("âœ… Submission saved as submission_poly_l1.csv")



# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer



# # -------------------------------
# # # âœ… 1. Load Data
# # # -------------------------------
# train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
# test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# # # -------------------------------
# # # âœ… 2. Identify Features and Target
# # # -------------------------------
# # target_col = 'target'
# # id_col = 'id'
# # num_cols = [col for col in train.columns if col not in [id_col, target_col]]

# # -------------------------------
# # âœ… 3. Impute Missing Values
# # -------------------------------
# # train[num_col] = train[num_col].fillna(train[num_col].median())
# # test[num_col] = test[num_col].fillna(train[num_col].median())
# cat_col = train.select_dtypes(include = 'object').columns.drop('Personality')
# num_col = train.select_dtypes(include = 'number').columns

# train[cat_col]= train[cat_col].fillna('Not available')
# test[cat_col] = test[cat_col].fillna('Not available')

# train[num_col] = train[num_col].fillna(train[num_col].median())
# test[num_col] = test[num_col].fillna(test[num_col].median())

# # One-hot encode categorical features
# train = pd.get_dummies(train, columns=cat_col)
# test = pd.get_dummies(test, columns=cat_col)

# # -------------------------------
# # âœ… 4. Define X and y
# # -------------------------------


# features = ['id', 'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
#        'Friends_circle_size', 'Post_frequency', 'Stage_fear_No',
#        'Stage_fear_Not available', 'Stage_fear_Yes',
#        'Drained_after_socializing_No',
#        'Drained_after_socializing_Not available',
#        'Drained_after_socializing_Yes']
# X = train[features]
# train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})
# y = train['Personality']
# # Convert target to binary


# # X = train[num_col]
# # y = train[target_col]
# X_test = test[features]

# # -------------------------------
# # âœ… 5. Feature Scaling (Optional for XGBoost, but OK)
# # -------------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

# # -------------------------------
# # âœ… 6. Define XGBoost Model
# # -------------------------------
# xgb = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )

# # -------------------------------
# # âœ… 7. Cross-Validated Accuracy
# # -------------------------------
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(xgb, X_scaled, y, cv=cv, scoring='accuracy')

# print("ðŸ“Š Mean CV Accuracy (XGBoost):", cv_scores.mean())
# print("Fold-wise Accuracy:", np.round(cv_scores, 5))

# # -------------------------------
# # âœ… 8. Fit on Full Data and Predict on Test Set
# # -------------------------------
# # xgb.fit(X_scaled, y)
# # test_preds = xgb.predict(X_test_scaled)

# # -------------------------------
# # âœ… 9. Prepare Submission
# # -------------------------------
# # submission = pd.DataFrame({
# #     'id': test[id_col],
# #     'target': test_preds
# # })

# # submission.to_csv("submission_xgboost.csv", index=False)
# # print("âœ… Submission saved as 'submission_xgboost.csv'")



# t1[['Stage_fear','Personality']].value_counts()


# df = t1
# x = 'Stage_fear'
# y = 'Personality'


# df = df.dropna()


# df = train.copy()


# train


def plot_boxplot(df,col):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Personality', y=col, palette='Set2')
    plt.title(col+ ":  Extrovert vs Introvert")
    plt.ylabel(col)
    plt.xlabel('Personality')
    plt.show()



# cat_col = train.select_dtypes(include = 'object').columns.drop('Personality')
# num_col = train.select_dtypes(include = 'number').columns


# for i in num_col[1:]:
#     plot_boxplot(df,i)


def plot_countplot(df,col):
    sns.countplot(data=df, x=col, hue='Personality', palette='Set2')
    plt.title(col+ ' vs Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Personality')
    plt.show()



# for i in cat_col:
#     plot_countplot(df,i)


# df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Missing')
# test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna('Missing')

# df['Stage_fear'] = df['Stage_fear'].fillna('Missing')
# test['Stage_fear'] = test['Stage_fear'].fillna('Missing')


# def randomsampleimputation(df, variable):
#     df[variable]=df[variable]
#     random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
#     random_sample.index=df[df[variable].isnull()].index
#     df.loc[df[variable].isnull(),variable]=random_sample


# for i in cat_col:
#     randomsampleimputation(df,i)





# for i in cat_col:
#     randomsampleimputation(test,i)


# for i in cat_col:
#     plot_countplot(df,i)


# for i in num_col:
#     randomsampleimputation(df,i)


# for i in num_col:
    # randomsampleimputation(test,i)




