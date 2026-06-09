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
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, classification_report,confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import uniform, randint
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error


train_data=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


train_data


train_data.info()


train_data.isnull().sum()


new_train=train_data.dropna().reset_index(drop=True)


new_train


new_train.isnull().sum()


new_train.info()


num_column=new_train.select_dtypes('float64')
cat_column=new_train.select_dtypes('object')
cat_column.drop(columns='date',inplace=True)


sns.histplot(data=num_column)


for col in cat_column.columns:
    plt.figure(figsize=(6, 6))
    cat_column[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title(f'Pie Chart of {col}')
    plt.ylabel('')
    plt.show()



for col in cat_column.columns:
    plt.figure(figsize=(8, 6))  # Set figure size
    new_train.groupby(col)['num_sold'].sum().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Bar Chart of {col}')
    plt.ylabel('Total num_sold')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
    plt.show()


chi2_scores = {}
for column in cat_column:
    contingency_table = pd.crosstab(new_train[column], new_train['num_sold']) 
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    chi2_scores[column] = chi2

chi2_df = pd.DataFrame(list(chi2_scores.items()), columns=['Feature', 'Chi-Square']).sort_values(by='Chi-Square', ascending=False)
print(chi2_df)


plt.figure(figsize=(6, 4))
sns.boxplot(data=num_column, color='skyblue')
plt.title('Box Plot of number of item sold')
plt.ylabel('Value')
plt.show()


train_x = new_train.drop(columns=['id', 'num_sold'])
train_y = new_train['num_sold']


train_x


train_y


train_x['date'] = pd.to_datetime(train_x['date'])
train_x['Year'] = train_x['date'].dt.year
train_x['Month'] = train_x['date'].dt.month
train_x['Day'] = train_x['date'].dt.day
train_x['weekday'] = train_x['date'].dt.weekday  # 0 = Monday, 6 = Sunday
train_x['is_weekend'] = train_x['weekday'].apply(lambda x: 1 if x >= 5 else 0)


train_x


train_x.drop(columns=['date'],inplace=True)


train_x


train_x.Year.value_counts()


categorical_features = ['country','store','product']
numeric_features = ['Year','Month','Day','weekday','is_weekend']
numeric_transformer = Pipeline(steps=[ 
    ('scaler', StandardScaler())  
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
]).set_output(transform='pandas')


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


x_train_p=preprocessor.fit_transform(X_train)
x_test_p=preprocessor.transform(X_test)


x_train_p


model1=XGBRegressor(n_estimators=2000,random_state=42)
model1.fit(x_train_p,y_train)


y_pred1 = model1.predict(x_test_p)
mape_score1 = mean_absolute_percentage_error(y_test, y_pred1)
print(mape_score1)


r2_score1 = model1.score(x_test_p, y_test)
print(f'R² Score: {r2_score1:.4f}')


model2=XGBRegressor(n_estimators=2000,random_state=42)

param_grid = {'learning_rate': [.01,.05],'max_depth': [5,8]}

grid_search2 = GridSearchCV(model2, param_grid, cv=3, n_jobs=-1, verbose=0)

grid_search2.fit(x_train_p, y_train)

print("Best Parameters: ", grid_search2.best_params_)

y_pred2 = grid_search2.predict(x_test_p)

mape2 = mean_absolute_percentage_error(y_test, y_pred2)
print(mape2)


r2_score = grid_search2.score(x_test_p, y_test)
print(f'R² Score: {r2_score:.4f}')


from lightgbm import LGBMRegressor
model3 = LGBMRegressor(random_state=42)
model3.fit(x_train_p, y_train)
y_pred3 = model3.predict(x_test_p)
mape3 = mean_absolute_percentage_error(y_test, y_pred3)
print(mape3)
r2_score3 = model3.score(x_test_p, y_test)
print(f'R² Score: {r2_score3:.4f}')


model4=LGBMRegressor(n_estimators=2000,random_state=42)
param_grid = {'learning_rate': [.01,.05],'max_depth': [5,8]}
grid_search4 = GridSearchCV(model4, param_grid, cv=3, n_jobs=-1, verbose=0)
grid_search4.fit(x_train_p, y_train)
print("Best Parameters: ", grid_search4.best_params_)
y_pred4 = grid_search4.predict(x_test_p)
mape4 = mean_absolute_percentage_error(y_test, y_pred4)
print(mape4)


model5=LGBMRegressor(n_estimators=2000,random_state=42,max_depth=8,learning_rate=0.05,
                     reg_alpha=1,reg_lambda=1)
model5.fit(x_train_p, y_train)
y_pred5 = model5.predict(x_test_p)
mape5 = mean_absolute_percentage_error(y_test, y_pred5)
print(mape5)


from sklearn.ensemble import VotingRegressor
vot = VotingRegressor(estimators=[('xgb', grid_search2),('lgbm', model5)])
vot.fit(x_train_p, y_train)
vot_pred = vot.predict(x_test_p)
mape_vot = mean_absolute_percentage_error(y_test, vot_pred)
print(mape_vot)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred2, label="Model 1", marker='o', color='red', alpha=0.7)
plt.scatter(y_test, y_pred5, label="Model 2", marker='s', color='blue', alpha=0.7)
plt.scatter(y_test, vot_pred, label="Model 3", marker='^', color='green', alpha=0.7)
plt.plot(y_test, y_test, linestyle='--', color='black', label="Ideal Fit (y=x)")
plt.xlabel("Actual Y Values")
plt.ylabel("Predicted Y Values")
plt.title("Actual vs Predicted Values for Three Regression Models")
plt.legend()
plt.grid(True)
plt.show()


Test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


Test


Test['date'] = pd.to_datetime(Test['date'])
Test['Year'] = Test['date'].dt.year
Test['Month'] = Test['date'].dt.month
Test['Day'] = Test['date'].dt.day
Test['weekday'] = Test['date'].dt.weekday  # 0 = Monday, 6 = Sunday
Test['is_weekend'] = Test['weekday'].apply(lambda x: 1 if x >= 5 else 0)


Test


Test.drop(columns=['id','date'],inplace=True)


Test


test=preprocessor.transform(Test)


test


pred=vot.predict(test)


pred


submission = pd.DataFrame({"id": range(230130,328680), 
                           "num_sold": pred}) 

submission.to_csv('submission.csv',index=False)




