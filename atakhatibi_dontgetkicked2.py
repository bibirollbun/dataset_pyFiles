import pandas as pd
import numpy as np


train_FS = pd.read_csv('/kaggle/input/dontgetkicked/Carvana_train_FS.csv')
train_FS = train_FS.rename(columns={'Unnamed: 0': 'Id'})
train_FS.set_index('Id', inplace=True)


y_train = train_FS.IsBadBuy
x_train = train_FS.drop('IsBadBuy', axis=1)
x_train.info()


# !pip install --upgrade scikit-learn


from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor

# configure to select all features
selector = RFECV(estimator=DecisionTreeRegressor(random_state=29), step=1, min_features_to_select=10, cv=5, n_jobs=-1)


# learn relationship from training data
selector.fit(x_train, y_train)

selector.get_support()

print(f"Optimal number of features: {selector.n_features_}")

print("="*50)

wrapper_fs = selector.get_feature_names_out()
print("Wrapper Optimal Feature List:")
print(wrapper_fs)

X_train_wrapper_fs = x_train[wrapper_fs]


X_train_wrapper_fs.to_csv('/kaggle/working/X_train_wrapper_fs.csv')

