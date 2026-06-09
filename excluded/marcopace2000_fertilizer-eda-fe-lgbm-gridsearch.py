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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").drop(columns="id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train.info()


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


sns.pairplot(df_train.sample(1000))
plt.show()


sns.heatmap(df_train.corr(numeric_only= True))
plt.show()


df_train.columns


x = pd.get_dummies(df_train.drop(columns="Fertilizer Name"),drop_first = True)
y = df_train["Fertilizer Name"]

x_test = pd.get_dummies(df_test.drop(columns="id"),drop_first = True)


x.columns


num_features = x.select_dtypes(include=['number']).columns.tolist()

def fe(df):
    for i, col1 in enumerate(num_features):
        for j in range(i, len(num_features)):
            col2 = num_features[j]
            df[col1 + "*" + col2] = df[col1] * df[col2]
    return df


x = fe(x)
x_test = fe(x_test)
x.columns


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2)


from lightgbm import LGBMClassifier

model_lgbm = LGBMClassifier(verbose=0)
model_lgbm.fit(x_train,y_train).score(x_val,y_val)


from catboost import CatBoostClassifier

model_cat = CatBoostClassifier(iterations = 200, verbose=50)
model_cat.fit(x_train,y_train).score(x_val,y_val)


from sklearn.ensemble import AdaBoostClassifier

model_ada = AdaBoostClassifier()
model_ada.fit(x_train,y_train).score(x_val,y_val)


# from sklearn.model_selection import GridSearchCV

# estimator = LGBMClassifier(verbose=0)
# param_grid = {"n_estimators": [500, 1000,4000],
#               "max_depth": [-1],
#               "min_data_in_leaf" : [100,1000],
#               "num_iterations" : [100,500,1000],
#               "learning_rate":[0.05]
#              }


# grid_search = GridSearchCV(
#     estimator=estimator,
#     param_grid=param_grid,
#     scoring= 'accuracy',
#     cv=5,
#     verbose=1
# )

# # Fit the model on the training data
# grid_search.fit(x_train, y_train)
# # Show results
# print("LGBM")
# print("Best score:", grid_search.best_score_)
# print("Best parameters:", grid_search.best_params_)


probs = LGBMClassifier(n_estimators=1000, max_depth=-1, objective="multiclass" , learning_rate=0.05, num_iterations=1000,min_data_in_leaf = 1000,random_state=0).fit(x_train,y_train).predict_proba(x_test)  # shape: (n_samples, n_classes)

# Get indices of top 3 probabilities for each row
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # sort and reverse for descending

# Get class labels for those indices
class_labels = model_cat.classes_  # available after training

top3_classes = class_labels[top3_indices]
top3_concatenated = np.array([' '.join(row) for row in top3_classes]) 


out_df = pd.DataFrame(top3_concatenated, columns=['Fertilizer Name'],index=df_test.id)
out_df.to_csv("submission.csv")
out_df.head()

