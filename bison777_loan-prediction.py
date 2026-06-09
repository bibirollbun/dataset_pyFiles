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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


print(train_df.shape)
print(test_df.shape)


print(f"\n{train_df.info()}")
print(train_df.isnull().sum())
print(train_df.duplicated())
print(train_df.columns)


train_df.describe().T


train_df.head()


for item in train_df.columns[1:]:
    print(f'{item} : {train_df[item].unique()}')
    print('-'* 50)


cat_col = train_df.select_dtypes(include=['object']).columns
for items in cat_col:
    print(f'{train_df[items].value_counts()}')
    print('-'* 50)


plt.figure(figsize=(8,6))
train_df['loan_status'].value_counts().plot.pie(autopct = '%1.1f%%', explode = [0.15, 0.15], shadow = True).set_title('status counts')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
sns.barplot(data = train_df,  x = 'cb_person_cred_hist_length' ,y = 'cb_person_default_on_file', hue = 'loan_status', palette='viridis')
plt.tight_layout()
plt.show()


# plot for the count for categorical
for cols in cat_col:
    plt.figure(figsize=(8,6))
    sns.countplot(data=train_df, x = cols, hue="loan_status", palette='viridis')
    plt.title(f'Count of the {cols}')
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.show()


train_df.columns


loan_box  = [
    'person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'person_emp_length','cb_person_cred_hist_length'
]
for boxcol in loan_box:
    plt.figure(figsize=(10,6))
    sns.boxplot(x= boxcol , data = train_df, hue = 'loan_status', palette='viridis', showmeans = True)
    plt.title(f'Box plot of :{boxcol}')
    plt.ylabel('counts')
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(12,6))
sns.swarmplot(data = train_df.sample(1000), y = 'loan_amnt' ,x = 'person_emp_length', hue = 'loan_status')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.swarmplot(data = train_df.sample(1000), x = 'loan_grade' ,y = 'person_age', hue = 'loan_status', palette='viridis')
plt.title('Relation of age and grade')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.swarmplot(data = train_df.sample(1000), x = 'person_home_ownership' ,y = 'loan_amnt', hue = 'loan_status', palette='viridis')
plt.title('Relation of amt and ownership')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(data = train_df, y = 'loan_amnt' ,x = 'loan_grade',hue = 'loan_status' )
plt.title('Relation of age and grade')
plt.tight_layout()
plt.show()


train_df[train_df['person_age']>90]


train_df[train_df['person_income']>1e6]


train_df[train_df['person_emp_length']>60]


train_df = train_df[train_df['person_age']<90].reset_index(drop=True)
train_df = train_df[train_df['person_income']<1e6].reset_index(drop=True)
train_df = train_df[train_df['person_emp_length']<60].reset_index(drop=True)


train_df.describe().T


hist_plot  = [
    'person_age', 'person_income', 'loan_amnt', 'loan_int_rate','person_emp_length','cb_person_cred_hist_length'
]
for cols in hist_plot:
    plt.figure(figsize=(8,6))
    sns.histplot(data = train_df, x = cols, hue = 'loan_status', palette='viridis',kde = True)
    plt.title(f'hist plot of {cols}')
    plt.tight_layout()
    plt.show()


on_file ={
    'N' : 0,
    'Y' : 1
    }

train_df['cb_person_default_on_file'] = train_df['cb_person_default_on_file'].map(on_file)
test_df['cb_person_default_on_file'] = test_df['cb_person_default_on_file'].map(on_file)



cat_cols = train_df.select_dtypes(['object']).columns
stand_col = ['person_age', 'person_income','person_emp_length', 'loan_amnt','loan_int_rate', 'loan_percent_income','cb_person_cred_hist_length']


x_train = train_df.drop(columns='loan_status', axis =1)
y_train = train_df['loan_status']
x_test = test_df


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier



def testing_diff_model(x_train, y_train, x_test, model):
    combine = pd.concat([x_train, x_test])
    coltrans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), cat_cols),
        (StandardScaler(), stand_col),
        remainder='passthrough'
    )

    pipe = make_pipeline(coltrans, model)
    cv_scor = cross_val_score(pipe, x_train, y_train, cv =10  )

    print(f'For the model {model.__class__.__name__}')
    print(f'The Cv score is : {cv_scor}')
    print(f'Mean of the CV is : {cv_scor.mean():.4f}')
    print(f'SD of the CV is : {cv_scor.std():.4f}')

    pipe.fit(x_train, y_train)
    ypred = pipe.predict(x_test)

    return ypred


models = [
    # LogisticRegression(max_iter=1000),
    # LinearSVC(),
    XGBClassifier(n_estimators = 350),
    # RandomForestClassifier(n_estimators=150),
    # AdaBoostClassifier(learning_rate=.15,n_estimators=100),
    # KNeighborsClassifier(n_neighbors=5),
    # GradientBoostingClassifier(n_estimators=200, learning_rate=0.15)
]
predictions = []
for model in models :
    predic = testing_diff_model(x_train, y_train, x_test, model)
    predictions.append(predic)
#    testing_diff_model(x_train, y_train, x_test, model)
#    print('-'* 50)


test_id = test_df['id']
output = pd.DataFrame({
    'id' : test_id,
    'loan_status' : predic
})
output.to_csv('To_submmit.csv', index=False)
output.head()

