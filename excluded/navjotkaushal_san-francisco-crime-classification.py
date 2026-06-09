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
warnings.filterwarnings(action = 'ignore')


df = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')


df.sample(3)


df.info()


df['Dates'] = pd.to_datetime(df['Dates'])


df.isnull().sum()


plt.figure(figsize=(10,8))
sns.barplot(x = df['Category'].value_counts().index,
           y = df['Category'].value_counts().values, palette = 'viridis')
plt.title('Distribution of target column', fontsize = 15)
plt.tight_layout()
plt.show()


sns.boxplot(df, palette = 'viridis')
plt.title('Outlier detecting', fontsize = 15)
plt.tight_layout()
plt.show()


def cap_outliers(df):
    num_col = df.select_dtypes(include = 'number')
    for col in num_col:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower,upper = Q1 - 1.5 * IQR , Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

df = cap_outliers(df)


num_col = df.select_dtypes(include = 'number')
plt.figure(figsize=(15,50))
index = 1

for col in num_col:
    plt.subplot(11, 4, index)
    sns.kdeplot(df[f'{col}'])
    index +=1



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, FunctionTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report


X = df.drop(columns = ['Category','Address'], axis = 1)
y = df['Category']

le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


numerical_cols = ['X','Y']

num_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])


categorical_cols = ['Descript','PdDistrict','Resolution','DayOfWeek']

cat_pipeline = Pipeline(steps=[
    ('Onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))
])


preprocessor = ColumnTransformer(transformers = [
    ('num_pipeline', num_pipeline, numerical_cols),
    ('cat_pipeline', cat_pipeline, categorical_cols)
])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


acc_score = accuracy_score(y_test, y_pred)
class_re = classification_report(y_test, y_pred)

print('Accuracy Score : ', 100 * acc_score)
print(class_re)

