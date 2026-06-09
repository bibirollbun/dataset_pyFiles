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


train_df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample_df = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')
train_df.tail()


train_df.drop(columns=['id'], inplace=True)
train_df.head()


train_df.info()


train_df.isna().sum()


test_df.isna().sum()


train_df.describe()


train_df['person_emp_length'] = train_df['person_emp_length'].astype(int)
train_df.head()


train_df.columns


columns = list(train_df.columns)
num_cols = []
cat_cols = []

for col in columns:
    if(train_df[col].dtype == 'object' or train_df[col].dtype == 'int64') and len(train_df[col].unique()) < 12:
        cat_cols.append(col)
    else:
        num_cols.append(col)
        


if 'loan_status' in cat_cols:
    cat_cols.remove('loan_status')


cat_cols


num_cols


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Implement pipelines
num_transformer = Pipeline(steps = [
    ('my_scaler', StandardScaler())
])

cat_transformer =Pipeline(steps=[
    ('this_scaler', OrdinalEncoder())
])

# Define column Transformer
main_pipe = ColumnTransformer(transformers= [
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# implement final pipeline
pipeline = Pipeline(steps=[('preprocessor', main_pipe)])


# split the dataset
y = train_df['loan_status']
X = train_df.drop(columns=['loan_status'])


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)


X_train_transform = pipeline.transform(X_train)
X_val_transform = pipeline.transform(X_val)


%%time
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

params ={
    'max_depth': [4,6,8],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2]
}

model = GridSearchCV(XGBClassifier(), params, cv=5, scoring='roc_auc')

model.fit(X_train_transform, y_train)
bestparams = model.best_params_
bestscore = model.best_score_

print('Best Parameter', bestparams)
print('Best Score', bestscore)


from sklearn.metrics import f1_score
best_model = XGBClassifier(**bestparams)
best_model.fit(X_train_transform, y_train)

y_pred_proba = best_model.predict_proba(X_val_transform)[:,1]

auc_score = roc_auc_score(y_val, y_pred_proba)

final_prediction = best_model.predict(X_val_transform)

# Calculate F1 score for the final XGBoost model
final_f1_xgb = f1_score(y_val, final_prediction)

print("Final AUC score for XGBoost:", auc_score)
print("Final F1 Score for XGBoost:", final_f1_xgb)


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr, label='XGBoost Classifier')
plt.title('ROC Curve for XGBoost Classifier')
plt.legend(loc='lower right')
plt.show()


test_df = pipeline.transform(test_df)

prediction = best_model.predict_proba(test_df)[:,1]
print(prediction)


sample_df.head()


sample_df['loan_status'] = prediction
sample_df.head()


sample_df.to_csv('sample_df.csv', index=False)

