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


import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)





df = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
df__test = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")

y_train = df.CORRUCYSTIC_DENSITY
df_train = df.drop(["CORRUCYSTIC_DENSITY","LOCAL_IDENTIFIER"], axis = 1)
df_test = df__test.drop(["LOCAL_IDENTIFIER"], axis = 1)

columns_names = df_train.columns.tolist()
print(len(y_train))


df_train.head()


df_test.head()


y_train.describe()


df_train.info()


object_columns = (df_train.select_dtypes(include = "object").columns).tolist()
numerics_columns = (df_train.select_dtypes(exclude = "object").columns).tolist()
print("Number of Obect columns : ",len(object_columns),"\n")
print("Number of Numerical columns : ",len(numerics_columns))


nan_columnns = [col for col in df_train.columns if df_train[col].isna().any()]
count_nan = [df_train[col].isna().sum() for col in nan_columnns ]
nan_values_count = dict(zip(nan_columnns, count_nan))
print("Count Nan for type Nurical\n")

i = 0
for key,val in nan_values_count.items():
    if key in numerics_columns:
        i+=1
        print(key,"-------", val)
print("Numerical Nan Rows number : ",i)

i=0
print("\nCount Nan for type Object \n")
for key,val in nan_values_count.items():
    if key in object_columns:
        i+=1
        print(key,"-------", val)
print("Categorical Nan Rows number : ",i)


for col in object_columns:
    print("Unique values for rows ",col," : ",df_train[col].unique(),"\n","number of unique values : ",len(df_train[col].unique()),"\n")


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numerical_transformers = SimpleImputer(strategy= "median")
categorical_transformers = Pipeline(steps=[("imputer", SimpleImputer(strategy = "most_frequent"))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformers, numerics_columns),
    ("cat", categorical_transformers, object_columns)
])

train_transform = preprocessor.fit_transform(df_train)
feature_names = preprocessor.get_feature_names_out()
train_transform = pd.DataFrame(train_transform, columns = (numerics_columns + object_columns))

test_transform = preprocessor.fit_transform(df_test)
feature_names = preprocessor.get_feature_names_out()
test_transform = pd.DataFrame(test_transform, columns = (numerics_columns + object_columns))



df_concat = pd.concat([train_transform, y_train], axis = 1)
df_concat.columns = df_concat.columns.astype(str)
df_concat = df_concat.dropna()
y_train = df_concat.CORRUCYSTIC_DENSITY
train_transform = df_concat.drop("CORRUCYSTIC_DENSITY", axis = 1)


train_transform


train_transform[numerics_columns] = train_transform[numerics_columns].astype("float64")
test_transform[numerics_columns] = test_transform[numerics_columns].astype("float64")


train_transform[numerics_columns].describe()


import matplotlib.pyplot as plt
import seaborn as sns

for col in numerics_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_transform[col], kde = True, color='skyblue', edgecolor = "black")
    plt.title(f"Distribution of {col}", fontsize = 12)
    plt.xlabel(col, fontsize = 10)
    plt.ylabel("Count",fontsize = 10)
    plt.grid(axis = "y", linestyle = "--", alpha = 0.5)
    plt.tight_layout()
    plt.show()

    print(train_transform[col].describe())


plt.figure(figsize=(20,30))
for i, col in enumerate(numerics_columns):
    plt.subplot(9,5,i+1)
    sns.boxplot(data = train_transform, y = col, color = '#FFA728')
    plt.title(f"Boxplot: {col}")
    plt.grid(True)
plt.tight_layout()
plt.show()


for col in object_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data = train_transform, x = col, order = train_transform[col].value_counts().index, palette = "Set2", edgecolor = "black")
    plt.title(f"Distribution of {col}", fontsize = 12)
    plt.xlabel(col, fontsize = 10)
    plt.ylabel("Count",fontsize = 10)
    plt.grid(axis = "y", linestyle = "--", alpha = 0.5)
    plt.tight_layout()
    plt.show()
    print(df[col].value_counts(normalize=True)*100)


from sklearn.preprocessing import OneHotEncoder

train_transform.drop(object_columns[1],axis = 1, inplace = True)
test_transform.drop(object_columns[1],axis = 1, inplace = True)

object_columns.pop(1)

hotencoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_hot_transform = pd.DataFrame(hotencoder.fit_transform(train_transform[object_columns]))
test_hot_transform = pd.DataFrame(hotencoder.fit_transform(test_transform[object_columns]))

train_hot_transform.index = train_transform.index
test_hot_transform.index = test_transform.index

train_transform.drop(object_columns,axis = 1, inplace = True)
test_transform.drop(object_columns,axis = 1, inplace = True)

train_final = pd.concat([train_transform,train_hot_transform], axis = 1)
test_final = pd.concat([test_transform,test_hot_transform], axis = 1)

train_final.columns = train_final.columns.astype(str)
test_final.columns = test_final.columns.astype(str)


train_final.describe()


am = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Ä Ö Ü 1 2 3 4 5 6 7 8 9 0 ß"
print()

# change featurs name
import random
new_featurs_name = []
number = 0
names = "q,"

nam = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Ä Ö Ü 1 2 3 4 5 6 7 8 9 0 ß"

for name in range(len(train_final.columns)):
    if len(new_featurs_name) < len(nam.split()):
        new_featurs_name.append(nam.split()[name])
    else:
        if len(new_featurs_name) <= len(train_final.columns):
            names = nam.split()[random.randint(0, len(nam.split())-1)] + nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]
            if names in new_featurs_name:
                name = name -1
            else :
                new_featurs_name.append(names)
train_final.columns = new_featurs_name
test_final.columns = new_featurs_name


import math

for val in train_final.columns:
    if np.max(train_final[val]) > 11 or np.min(train_final[val]) < -11:
        for i in range(len(train_final[val])-1):
            if train_final[val].iloc[i] < 0:
                train_final[val].iloc[i] = train_final[val].iloc[i] * -1
                train_final[val].iloc[i] = math.sqrt(train_final[val].iloc[i])
                train_final[val].iloc[i] = train_final[val].iloc[i] * -1
            else :
                train_final[val].iloc[i] = math.sqrt(train_final[val].iloc[i])

for val in test_final.columns:
    if np.max(test_final[val]) > 11 or np.min(test_final[val]) < -11:
        for i in range(len(test_final[val]) - 1):
            if test_final[val].iloc[i] < 0:
                test_final[val].iloc[i] = test_final[val].iloc[i] * -1
                test_final[val].iloc[i] = math.sqrt(test_final[val].iloc[i])
                test_final[val].iloc[i] = test_final[val].iloc[i] * -1
            else :
                test_final[val].iloc[i] = math.sqrt(test_final[val].iloc[i])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

x_train, x_test, Y_train, y_test = train_test_split(train_final,y_train, train_size=0.75, random_state=42)

linear_model = LinearRegression()

random_model = RandomForestRegressor(
    n_estimators=200,
    criterion='squared_error',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=1.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
)

gauss_model = GaussianProcessRegressor()

xgb_model = XGBRegressor()

lgb_model = LGBMRegressor(
        random_state=42,
        verbosity=-1,
        n_estimators=40000,
        learning_rate=0.0358306214515723,
        min_child_samples=83,
        subsample=0.8700304020753131,
        colsample_bytree=0.6169349166144594,
        num_leaves=228,
        max_depth=6,
        max_bin=3600,
        reg_alpha=3.700714656885025,
        reg_lambda=4.709578317972932, 
    )

models = [linear_model,random_model,gauss_model,lgb_model]

for model in models:
    model.fit(x_train, Y_train)
    print(f"{model} : ")

    train_pred = model.predict(x_train)
    print(f"{model}", mean_absolute_error(Y_train, train_pred))
    test_pred = model.predict(x_test)
    print(f"{model}", mean_absolute_error(y_test, test_pred))


pred = models[3].predict(test_final)

print(pred)


submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df__test['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': pred
})

submission.to_csv('submission.csv', index=False)
print("Submission saved!")




