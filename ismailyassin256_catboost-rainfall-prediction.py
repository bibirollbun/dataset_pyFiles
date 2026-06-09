# IMporting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the files
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# The shape of the data
df_train.shape


# The number of null values in the data
df_train.isnull().sum()


# The infor about the data
df_train.info()


# The description about the data
df_train.describe()


# View of the first few rows of the dataset
df_train.head()


# Checking for distribution of the data
df_train['rainfall'].value_counts()


# Plotting heat map
plt.figure(figsize=(10,5))
plt.title('Heat map')
sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm')
plt.show()


# Defining a wrangle function for cleanining the data
def wrangle(filepath):
  # Importing the data through the function
  df = pd.read_csv(filepath)

  # Dropping less contributung columns
  df = df.drop(columns='id')
  return df


# Passing the data through the function
df_train_cleaned = wrangle('/kaggle/input/playground-series-s5e3/train.csv')


df_train_cleaned.head()


from sklearn.model_selection import train_test_split
# features
X = df_train_cleaned.drop(columns=['rainfall'])

# target
y = df_train_cleaned['rainfall']


# feature selection
from mlxtend.feature_selection import SequentialFeatureSelector
from catboost import CatBoostClassifier
model_feature_selection=CatBoostClassifier()

forward_feature_selection = SequentialFeatureSelector( 
    model_feature_selection,
    k_features=(1,11),
    forward=True,
    floating=False,
    verbose=2,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,).fit(X, y)


# Best features
forward_feature_selection.k_feature_names_


# Getting columns from X_train
FEATURES = ['day', 'maxtemp', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine']
print (FEATURES)


X.shape


# Importing multple classifiers
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# dictionary of classifiers
cls = {
    'lightgbm': LGBMClassifier(),
    'catboost': CatBoostClassifier(),
    'xgboost': XGBClassifier(),
    'randomforest': RandomForestClassifier(),
    'decisiontree': DecisionTreeClassifier()
}


%%time
from sklearn.model_selection import StratifiedKFold, KFold

FOLDS = 5



skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(df_train_cleaned))
pred_cat = np.zeros(len(df_test))

for i, (train_index, test_index) in enumerate(skf.split(X,y)):

   print('#'*30)
   print(f'#### {i + 1} fold')
   print('#'*30)

   X_train = df_train_cleaned.loc[train_index, FEATURES]
   y_train = df_train_cleaned.loc[train_index, "rainfall"]
   X_val = df_train_cleaned.loc[test_index, FEATURES]
   y_val = df_train_cleaned.loc[test_index, "rainfall"]
   X_test = df_test[FEATURES]

   model = CatBoostClassifier(iterations=1000,learning_rate=0.03,depth=6, l2_leaf_reg=6,random_seed=42)
                             
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100, early_stopping_rounds=2,)

   # infer oof
   oof_cat[test_index] = model.predict_proba(X_val)[:, 1]

   # infer pred_cat
   pred_cat += model.predict_proba(X_test)[:, 1]

# Return averages of pred_cat
pred_cat /= FOLDS







from sklearn.metrics import roc_auc_score
true_values = df_train_cleaned.rainfall.values
s = roc_auc_score(true_values, oof_cat)
print(f"Catboost has AUC score of {s:.3f}")


# make prediction
test = df_test[FEATURES]
pred = model.predict_proba(test)[:,1]

submission = pd.DataFrame({'id': df_test.id, 'rainfall': pred})
submission.to_csv('submission.csv', index=False)








