# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
train_data


train_data.drop(columns=['id'], axis=1).describe()


numerical_col = train_data.drop(columns=['id', 'loan_paid_back'], axis=1).select_dtypes(include='number').columns.tolist()

categorical_col = train_data.select_dtypes(include='object').columns.tolist()

for col in numerical_col:
    plt.figure(figsize=(13, 8))
    sns.histplot(train_data[[col]], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.show()

for col in categorical_col:
    plt.figure(figsize=(13,8))
    sns.countplot(data=train_data, x=col, hue='loan_paid_back', palette='Set2')
    plt.title(f"Count of {col}")
    plt.show()



train_data.info()


for col in categorical_col:
    print(f"Col value : {train_data[col].unique()}")


train_data.isnull().sum()


train_data.info()


X_train = train_data.drop(columns=['id', 'loan_paid_back'])
y_train = train_data.loan_paid_back

X_test = test_data.drop(columns=['id'])


numeric_preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_preprocessor = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_col),
    ('numerical', numeric_preprocessor, numerical_col)
])

models = {
    "Logistic Regression": LogisticRegression(n_jobs=-1, max_iter=1000),
    "RF": RandomForestClassifier(n_jobs=-1, max_depth=15, n_estimators=50),
    "Hist GDB": HistGradientBoostingClassifier(),
}

result = []


print('Starting training...')

for name, model in models.items():
    pipe = make_pipeline(preprocessor, model)
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')

    result.append({
        "Model": name,
        "Mean accuracy": scores.mean(),
        "ecart-type": scores.std() 
    })

    print(f"{name} done.")


df_results = pd.DataFrame(result).sort_values(by="Mean accuracy", ascending=False)
df_results


def save_file (predictions):
    """Save submission file."""
    # Save test predictions to file
    output = pd.DataFrame({'id': sample_submission.id,
                       'loan_paid_back': predictions})
    output.to_csv('submission.csv', index=False)
    print ("Submission file is saved")

ids = test_data['id']
X_test = test_data.drop(columns=['id'], axis=1)

final_model = HistGradientBoostingClassifier()
final_pipeline = make_pipeline(preprocessor, final_model)
final_pipeline.fit(X_train, y_train)

print("HistGDB done training.")

final_pred = final_pipeline.predict(X_test)

save_file(final_pred)

