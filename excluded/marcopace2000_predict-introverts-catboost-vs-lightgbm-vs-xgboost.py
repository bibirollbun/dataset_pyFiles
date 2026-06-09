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


import seaborn as sns
import matplotlib.pylab as plt

plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv").drop(columns="id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.info(), df_test.info()


df_train.head()


num_cols = df_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [ col for col in df_train.columns if col not in num_cols ]

for col in df_train.columns:
    if col in num_cols:
        df_train.loc[:,col] = df_train[col].fillna(df_train[col].mean())
        df_test.loc[:,col] = df_test[col].fillna(df_train[col].mean())
    else:
        mode_value = df_train[col].mode()[0] if not df_train[col].mode().empty else "Unknown"
        df_train.loc[:,col] = df_train[col].fillna(mode_value)
        if col != "Personality":
            df_test.loc[:,col] = df_test[col].fillna(mode_value)

for col in cat_cols:
    df_train[col] = df_train[col].astype("category")
    if col != "Personality":
        df_test[col] = df_test[col].astype("category")
    
df_train.isnull().sum(), df_test.isnull().sum()


df_train.describe().T


num_cols = df_train.shape[1]
fig, axs = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 4 * num_cols))

for i, column in enumerate(df_train.columns):
    ax = axs[i]
    if pd.api.types.is_numeric_dtype(df_train[column]):
        sns.histplot(df_train[column], bins=30, kde=True, color='skyblue', ax=ax)
    else:
        sns.countplot(x=df_train[column], palette="pastel", ax=ax)
    ax.set_title(f'Distribution of {column}', fontsize=14)

plt.tight_layout()
plt.show()


sns.heatmap(df_train.corr(numeric_only= True), annot = True, cmap="coolwarm")
plt.title("Correlations between numerical features")
plt.show()


from sklearn.model_selection import train_test_split

x = df_train.drop(columns="Personality")
y = df_train["Personality"]

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state = 0)


cat_cols.remove("Personality")


from catboost import CatBoostClassifier

cat_class = CatBoostClassifier(cat_features = cat_cols, verbose = 0)
cat_class.fit(x_train,y_train)

cat_class.score(x_val,y_val)


from lightgbm import LGBMClassifier

lgbm_class = LGBMClassifier(categorical_feature = cat_cols)
lgbm_class.fit(x_train,y_train)

lgbm_class.score(x_val, y_val)


from xgboost import XGBClassifier

y_train_xgb = y_train.map({"Extrovert":0,"Introvert":1})
y_val_xgb = y_val.map({"Extrovert":0,"Introvert":1})

xgb_class = XGBClassifier(enable_categorical=True)
xgb_class.fit(x_train, y_train_xgb)

xgb_class.score(x_val, y_val_xgb)


# from sklearn.model_selection import GridSearchCV
# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)


# estimator = XGBClassifier(enable_categorical=True)
# param_grid = {
#         "n_estimators": [100, 300, 500,1000],
#         "max_depth": [-1, 3, 5, 7, 10],
#         'learning_rate': [0.1,0.3,0.8],
#         'subsample': [0.2,0.3,0.8],
#         'colsample_by_tree': [0.2,0.3,0.8],
#         'max_depth': [-1, 5, 7,10],
#         'reg_lambda': [0.1,0.5,1],
#         'reg_alpha': [1,5,10]
#         }

# grid_search = GridSearchCV(
#     estimator=estimator,
#     param_grid=param_grid,
#     scoring="precision",
#     cv=4,
#     verbose=1
# )

# grid_search.fit(x_train, y_train_xgb)

# print("LGBM")
# print("Best score:", grid_search.best_score_)
# print("Best parameters:", grid_search.best_params_)
# print("Validation set score: ", grid_search.score(x_val,y_val_xgb))

# Fitting 4 folds for each of 3888 candidates, totalling 15552 fits
# LGBM
# Best score: 0.945000731610065
# Best parameters: {'colsample_by_tree': 0.2, 'learning_rate': 0.3, 'max_depth': 10, 'n_estimators': 300, 'reg_alpha': 10, 'reg_lambda': 0.5, 'subsample': 0.8}
# Validation set score:  0.953340402969247



params = {'colsample_by_tree': 0.2,
          'learning_rate': 0.3,
          'max_depth': 10,
          'n_estimators': 300,
          'reg_alpha': 10,
          'reg_lambda': 0.5,
          'subsample': 0.8,
          'enable_categorical':True
         }

xgb_class = XGBClassifier(**params)
xgb_class.fit(x_train, y_train_xgb)

print(xgb_class.score(x_val,y_val_xgb))

prediction = xgb_class.predict(df_test.drop(columns="id"))


arr = np.array(["Extrovert","Introvert"])

cat_prediction = arr[prediction]

out_df = pd.DataFrame(cat_prediction, columns=["Personality"], index = df_test["id"])


out_df["Personality"].value_counts()


out_df.to_csv("submission.csv")

