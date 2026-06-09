import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_theme(style="whitegrid")

from sklearn.dummy import DummyRegressor 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

import warnings 
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
train.head()


train = train.drop('id', axis=1)
# test = test.drop('id', axis=1)


cat_c = ['Episode_Title', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment','Podcast_Name','Genre']

def change_col_cat(df) :

    for col in cat_c :
        df[col] = df[col].astype('category')
    return df

train = change_col_cat(train)
test = change_col_cat(test)


train.describe()


train.info()


train.isnull().sum()


fig, axs = plt.subplots(nrows=3, ncols=2)
fig.suptitle("Individual histograms")
fig.set_size_inches(28, 12)

num_features = list(train.describe().columns) 

for i in range(len(num_features)):
    fig.add_subplot(3,2,i+1)
    fig.tight_layout()
    plt.hist(train[num_features[i]], bins = 50)
    plt.title(num_features[i])
    plt.xlabel('Range')
    plt.ylabel('Frequency')


train.plot.box(vert=False);


train.head()


X = train.drop('Listening_Time_minutes', axis=1) 
# train.drop('id', axis=1)
y = train['Listening_Time_minutes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from lightgbm import LGBMRegressor

model = LGBMRegressor()
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score  # for classification
from sklearn.metrics import mean_squared_error  # for regression

y_pred = model.predict(X_test)

# # Regression example
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


X = train.iloc[:, :-1]
y = train.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
shuffle_split_cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


def dummy_regressor_baseline(strategy: str = "median", constant_val: float = None, quantile_val: float = None) -> pd.Series :
    baseline_model_median = DummyRegressor(
        strategy=strategy, constant=constant_val, quantile=quantile_val
    )

    baseline_median_cv_results = cross_validate(
        baseline_model_median, X_train, y_train, cv=shuffle_split_cv, 
        scoring="neg_root_mean_squared_error", n_jobs=2
    )
    
    return pd.Series(-baseline_median_cv_results["test_score"], name="Dummy regressor error")


baseline_median_cv_results_error = dummy_regressor_baseline(strategy = 'median')
baseline_mean_cv_results_error = dummy_regressor_baseline(strategy = 'mean')
baseline_constant_cv_results_error = dummy_regressor_baseline(strategy = 'constant', constant_val=2)
baseline_quantile_cv_results_error = dummy_regressor_baseline(strategy = 'quantile', quantile_val=0.55)

dummy_error_df = pd.concat([
    baseline_median_cv_results_error, baseline_mean_cv_results_error,
    baseline_constant_cv_results_error, baseline_quantile_cv_results_error
    ], axis=1
)
            
dummy_error_df.columns = ['Median cv', 'Mean cv', 'Constant cv', 'Quantile cv']
dummy_error_df


def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound) | df_clean.Episode_Length_minutes.isnull()]
    
    return df_clean


clean_train = remove_outliers_iqr(train, columns=['Episode_Length_minutes', 'Number_of_Ads'])
clean_train.head()


clean_train.plot.box(vert=False);


X = clean_train.drop('Listening_Time_minutes', axis=1) 
# train.drop('id', axis=1)
y = clean_train['Listening_Time_minutes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LGBMRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# # Regression example
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


test_sub = test.drop('id', axis=1)
y_pred = model.predict(test_sub)
submission = pd.DataFrame({
     "id":test['id'],"Listening_Time_minutes": y_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created ✅")

