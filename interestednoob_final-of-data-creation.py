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


#getting data analysis and ML packages
import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time

from sklearn.decomposition import PCA
import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


!pip install ipympl
# Restart the kernel after installing for the changes to take effect.
# After restarting, you can run the cell again.


df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')


df_train


df_train['year']   = df_train.index.year
df_train['month']  = df_train.index.month
df_train['day']    = df_train.index.day
df_train['hour']   = df_train.index.hour
df_train['minute'] = df_train.index.minute
df_train['weekday'] = df_train.index.weekday
df_train['date'] = df_train.index.date


df_train['second_cont'] = (df_train.index - df_train.index[0]).total_seconds()
df_train['minute_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 60
df_train['hour_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 3600



## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_train.index = range(0,len(df_train.index))


df_train['label'].mean()


df_train['lag_1'] = df_train['label'].shift(periods=1)
df_train['lag_2'] = df_train['label'].shift(periods=2)
df_train['lag_3'] = df_train['label'].shift(periods=3)
df_train['lag_4'] = df_train['label'].shift(periods=4)
df_train['lag_5'] = df_train['label'].shift(periods=5)
df_train.loc[:,'lag_1':'lag_5'] = df_train.loc[:,'lag_1':'lag_5'].replace({np.nan: df_train['label'].mean()})


## Dropping perfectly one-half of all perfectly correlated pairs of RVs
df_train = df_train.drop(['X104',
                            'X62',
                            'X116',
                            'X74',
                            'X122',
                            'X80',
                            'X128',
                            'X86',
                            'X140',
                            'X98',
                            'X351',
                            'X309',
                            'X357',
                            'X315',
                            'X363',
                            'X321',
                            'X369',
                            'X327',
                            'date'], axis=1)


df_train


## Dropping infs
infcol = df_train.columns[np.isinf(df_train).any(axis=0)].tolist()
print(infcol)

df_train = df_train.drop(columns=infcol)


from sklearn.cross_decomposition import PLSRegression, PLSSVD

X = df_train.loc[:,'X1':'X780']
y = df_train.loc[:,'label']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)


r2_score(y_test, pls.predict(X_test))


r2_score(df_train.loc[1250:2250,'label'], pls.predict(df_train.loc[1250:2250,'X1':'X780']))


# Variance explained in y per component
explained_var_y = np.var(pls.y_scores_, axis=0, ddof=0)
total_var_y = np.var(y_train, ddof=0)
explained_ratio_y = explained_var_y / total_var_y
cumulative_var = np.cumsum(explained_ratio_y)

# Plot scree (elbow) plot
plt.figure(figsize=(30, 30))
plt.plot(range(1, 276), cumulative_var, marker='v')
#plt.axhline(0.7, color='r', linestyle='--', label='70% threshold')
plt.xlabel('Number of PLS Components')
plt.ylabel('Cumulative Variance Explained in y')
plt.title('PLS Scree Plot (Elbow Method)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/PLS-scree.png", dpi=300, bbox_inches='tight')
plt.show()


pls = PLSRegression(n_components=762)
pls.fit(X_train, y_train)
explained_var_y = np.var(pls.y_scores_, axis=0, ddof=0)
total_var_y = np.var(y_train, ddof=0)
explained_ratio_y = explained_var_y / total_var_y
print(explained_ratio_y.sum())


# Evaluate the model performance
import shap

# Assuming 'pls_model' is your trained PLS regression model
# Assuming 'X_train' is your training set features
# Assuming 'X_test' is your test set features

# Create a SHAP explainer for the PLS regression model using KernelExplainer
explainer = shap.KernelExplainer(pls.predict, X_train)

# Calculate SHAP values for the entire test set
shap_values = explainer.shap_values(X_test)

# Summary plot for all instances
shap.summary_plot(shap_values, X_test)


## Inter-factor correl
temp = pandas.DataFrame(pls.transform(X_train)).iloc[:,0:201].corr()
seaborn.heatmap(temp, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Inter-latent variable Correlation Heatmap 1m')
plt.savefig("/kaggle/working/inter-factor-pls.png")
plt.show()


## Factor-target correl
seaborn.heatmap(pandas.DataFrame(temp.corrwith(y_train)), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Latent variables-target Correlation Heatmap 1m')
plt.savefig("/kaggle/working/target-factor-pls.png")
plt.show()


pandas.DataFrame(temp.corrwith(y_train)).sort_values(0,ascending=False)


import joblib
joblib.dump(pls, "/kaggle/working/pls.pkl")

