import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train_df=pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sample_df=pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')

df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


train_df.columns = train_df.columns.str.lower().str.replace(' ', '_')
test_df.columns = test_df.columns.str.lower().str.replace(' ', '_')
sample_df.columns = sample_df.columns.str.lower().str.replace(' ', '_')

df.columns = df.columns.str.lower().str.replace(' ', '_')


train_df.shape


train_df.isnull().sum()


test_df.shape


df.head()


df['policy_start_date'] = pd.to_datetime(df['policy_start_date'])


df['year'] =df['policy_start_date'].dt.year
df['quarter'] = df['policy_start_date'].dt.quarter
df['month'] = df['policy_start_date'].dt.month
df['day'] = df['policy_start_date'].dt.day
df['day_of_week'] = df['policy_start_date'].dt.day_name()
df['week_of_year'] = df['policy_start_date'].dt.isocalendar().week

df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0)
df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)
df['group']=(df['year']-2010)*48+df['month']*4+df['day']//7


df['quarter'] = df['quarter'].astype('str')
df['month'] = df['month'].astype('str')
df['day_of_week'] = df['day_of_week'].astype('str')
df['week_of_year'] = df['week_of_year'].astype('str')




df = df.drop('policy_start_date', axis=1)


df.info()


def check_df(dataframe):
    print("------------- Shape -------------")
    print(dataframe.shape)
    print("------------- Types -------------")
    print(dataframe.dtypes)
    print("------------- Head -------------")
    print(dataframe.head(3))
    print("------------- Tail -------------")
    print(dataframe.tail(3))
    print("------------- NA -------------")
    print(dataframe.isnull().sum())
    
    
check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    num_cols = [col for col in num_cols if col not in ['premium_amount', 'id']]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')


    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df['premium_amount'] = np.log1p(df['premium_amount'])


def cat_summary(dataframe,col_name):
  print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                      'Ratio':100*dataframe[col_name].value_counts()/len(dataframe)}))
  print("----------------------------------")


for col in cat_cols:
  cat_summary(df,col)


def cat_summary(dataframe,col_name,plot=False):
  print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                      'Ratio':100*dataframe[col_name].value_counts()/len(dataframe)}))
  print("----------------------------------")

  if plot:
    sns.countplot(data=dataframe,x=dataframe[col_name])
    plt.show(block=True)


for col in cat_cols:
  cat_summary(df,col,plot=True)


for col in cat_cols:
    if df[col].dtypes =="bool":
        df[col] = df[col].astype(int)
        cat_summary(df,col,plot=True)
    else:
        cat_summary(df,col,plot=True)



def num_summary(dataframe,numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("----------------------------------")


num_summary(df,"premium_amount")

for col in num_cols:
    num_summary(df,col)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
      sns.histplot(data=dataframe, x=numerical_col)
      plt.show(block=True)

for col in num_cols:
  num_summary(df,col,plot=True)


def target_summary_with_cat(dataframe,target,categorical_col):
  print(pd.DataFrame({'Target_Mean':dataframe.groupby(categorical_col,observed=True)[target].mean()}), end="\n\n\n")
  print("----------------------------------")
  

for col in cat_cols:
  target_summary_with_cat(df,'premium_amount',col)


def target_summary_with_num(dataframe,target,numerical_col):
  print(dataframe.groupby(target).agg({numerical_col:'mean'}), end="\n\n\n")
  print("----------------------------------")

for col in num_cols:
  target_summary_with_num(df,'premium_amount',col)


#fig = px.histogram(df, x='premium_amount', nbins=100, title='Premium Amount Distribution')
#fig.show()


#df['log_premium_amount'] = np.log1p(df['premium_amount'])

#fig = px.histogram(df, x='log_premium_amount', nbins=50, title='Log Transformed Premium Amount Distribution')

#fig.show()



df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


def quick_missing_imp(data, num_method="median", cat_length=20, target="premium_amount"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)


df.isnull().sum()


def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "premium_amount":
      print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "premium_amount":
        replace_with_thresholds(df,col)


for col in num_cols:
    if col != "premium_amount":
      print(col, check_outlier(df, col))


label_encoders = {col: LabelEncoder() for col in cat_cols}


for col in cat_cols:
    le = label_encoders[col]
    le.fit(df[col])
    df[col] = le.transform(df[col])


df.head()


df['week_of_year'] = df['week_of_year'].astype(int)


from lightgbm import LGBMClassifier


train_df.shape, test_df.shape


train_df = df[df['premium_amount'].notna()].copy()
test_df = df[df['premium_amount'].isna()].copy()


train_df.shape, test_df.shape


test_df.drop('premium_amount', axis=1, inplace=True)


train_df.shape, test_df.shape


X = train_df.drop(['premium_amount'], axis=1)
y = train_df['premium_amount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.metrics import mean_absolute_percentage_error



def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)



lgbm_params = {
    'num_leaves': 71,
    'learning_rate': 0.05412467152424433,
    'n_estimators': 595,
    'max_depth': 12,
    'min_data_in_leaf': 97,
    'bagging_fraction': 0.5200288825838669,
    'feature_fraction': 0.9881738491942492,
    'n_jobs': -1,
    'verbose': -1
}


lgbm_model = LGBMRegressor(**lgbm_params)
lgbm_model.fit(X_train, y_train)


y_preds = lgbm_model.predict(X_test)


lgbm_mape = mean_absolute_percentage_error(y_test, y_preds)
print(f"LightGBM MAPE: {lgbm_mape:.4f}")


test_preds = lgbm_model.predict(test_df)



submission = pd.DataFrame({
    'id': test_df['id'], 
    'premium_amount': test_preds 
})


submission_filename = "submission.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission file saved as {submission_filename}")

