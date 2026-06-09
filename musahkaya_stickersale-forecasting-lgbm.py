import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fast")
plt.style.use('fivethirtyeight')
import warnings
import holidays
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import requests


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


def data_info(data):
    cache = []
    columns = data.columns
    display(data.head(3))
    print("Shape of data is:")
    print(data.shape)
    
    for i in data.columns:
        sample_size = data[i].shape[0]
        non_null_content = data[i].notnull().sum()
        dtypes = data[i].dtype
        unique = data[i].nunique()
        
        nan_values = data[i].isnull().sum()
        duplicated = data.duplicated().sum()
        
        cache.append([i,sample_size, non_null_content, dtypes, unique, nan_values, duplicated])

    cache = pd.DataFrame(cache, columns = ["Column", "Sample_Size", "Non_Null_Content", "D_Type", "Unique", "Nan_Values", "Duplicated"])
    return cache


df_train_info = data_info(df_train)
print("Train dataset info")
display(df_train_info)
print("------------------------------")


df_train.describe()


df_test_info = data_info(df_test)
print("Test dataset info")
display(df_test_info)
print("--------")


df_train = df_train.dropna()
cat_columns = ["country", "store", "product"]


for cat in cat_columns:
    print("Value Count of {categ} column".format(categ=cat))
    display(df_train[cat].value_counts())
    print("---------------------")


df_train["date"] = pd.to_datetime(df_train["date"], format = "%Y-%m-%d")
df_test["date"] = pd.to_datetime(df_test["date"], format = "%Y-%m-%d")


plt.figure(figsize=(12, 6))
ax = sns.lineplot(
    data=df_train,
    x="date",
    y="num_sold",
    errorbar=None,
    linewidth=0.4
)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
plt.title("Number Sold - Years", size=10)
plt.show()


total_sales_by_country = df_train.groupby("country")["num_sold"].sum()
fig, axs = plt.subplots(figsize=(12,10), nrows=2)

ax_0 = sns.barplot(x=total_sales_by_country.index, y=total_sales_by_country.values, ax = axs[0])
ax_0.set(ylabel = "Total Sales", xlabel = "Country")
ax_0.set_title("Total Sales by Country")

for p in ax_0.patches:
    ax_0.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')


ax_1 = sns.lineplot(data=df_train, x="date", y="num_sold", hue = "country" ,errorbar=None, linewidth=0.4, ax = axs[1])
ax_1.set(ylabel = "Number Sold", xlabel = "Year")
ax_1.set_title("Total Sales by Country")
ax_1.tick_params(axis="both")
ax_1.set_title("Number Sold - Years By Country")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


total_sales_by_product = df_train.groupby("product")["num_sold"].sum()

fig, axs = plt.subplots(figsize=(12,10), nrows=2)

ax_0 = sns.barplot(x=total_sales_by_product.index, y=total_sales_by_product.values, ax = axs[0])
ax_0.set(ylabel = "Total Sales", xlabel = "Product")
ax_0.set_title("Total Sales by Product")

for p in ax_0.patches:
    ax_0.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')


ax_1 = sns.lineplot(data=df_train, x="date", y="num_sold", hue = "product" ,errorbar=None, linewidth=0.4, ax = axs[1])
ax_1.set(ylabel = "Number Sold", xlabel = "Year")
ax_1.set_title("Total Sales by Product")
ax_1.tick_params(axis="both")
ax_1.set_title("Number Sold - Years By Product")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


total_sales_by_store = df_train.groupby("store")["num_sold"].sum()

fig, axs = plt.subplots(figsize=(12,10), nrows=2)

ax_0 = sns.barplot(x=total_sales_by_store.index, y=total_sales_by_store.values, ax = axs[0])
ax_0.set(ylabel = "Total Sales", xlabel = "Store")
ax_0.set_title("Total Sales by Store")

for p in ax_0.patches:
    ax_0.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')


ax_1 = sns.lineplot(data=df_train, x="date", y="num_sold", hue = "store" ,errorbar=None, linewidth=0.4, ax = axs[1])
ax_1.set(ylabel = "Number Sold", xlabel = "Year")
ax_1.set_title("Total Sales by Store")
ax_1.tick_params(axis="both")
ax_1.set_title("Number Sold - Years By Store")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def date_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Quarter'] = df['date'].dt.quarter
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week

    df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['Year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['Year'] / 7)

    return df


df_train = date_feature(df_train)
df_test = date_feature(df_test)


df_train["country_code"] = df_train["country"].map({"Canada": "CA", "Finland" : "FI", "Italy" : "IT", "Kenya": "KE", "Norway": "NO", "Singapore": "SG"})
df_test["country_code"] = df_test["country"].map({"Canada": "CA", "Finland" : "FI", "Italy" : "IT", "Kenya": "KE", "Norway": "NO", "Singapore": "SG"})


'''
def get_gdp(row):
    country = row["country_code"]
    year = row["Year"]
    url = "https://api.worldbank.org/v2/country/{Code}/indicator/NY.GDP.PCAP.CD?date={YYYY}&format=json".format(Code = country,
                                                                                                               YYYY = year)
    response = requests.get(url).json()
    return response[1][0]["value"]
'''


def get_gdp(df):
    temp_df = []
    countries = df["country_code"].unique()
    min_year = min(df["Year"])
    max_year = max(df["Year"]) + 1

    for country in countries:
        for year in range(min_year, max_year):
            url = "https://api.worldbank.org/v2/country/{Code}/indicator/NY.GDP.PCAP.CD?date={YYYY}&format=json".format(Code = country,
                                                                                                               YYYY = year)
            response = requests.get(url).json()
            value = response[1][0]['value']
            temp_df.append([value, year, country])
    return pd.DataFrame(temp_df, columns = ["gdp", "Year", "country_code"])


def get_inflation(df):
    temp_df = []
    countries = df["country_code"].unique()
    min_year = min(df["Year"])
    max_year = max(df["Year"]) + 1

    for country in countries:
        for year in range(min_year, max_year):
            url = "https://api.worldbank.org/v2/country/{Code}/indicator/FP.CPI.TOTL.ZG?date={YYYY}&format=json".format(Code = country,
                                                                                                               YYYY = year)
            response = requests.get(url).json()
            value = response[1][0]['value']
            temp_df.append([value, year, country])
    return pd.DataFrame(temp_df, columns = ["inflation", "Year", "country_code"])


train_gdp = get_gdp(df_train)
df_train = df_train.merge(train_gdp, how = "left")
test_gdp = get_gdp(df_test)
df_test = df_test.merge(test_gdp, how = "left")


train_inflation = get_inflation(df_train)
df_train = df_train.merge(train_inflation, how = "left")
test_inflation = get_inflation(df_test)
df_test = df_test.merge(test_inflation, how = "left")


def get_holiday(row):
    country = row['country_code']
    date = row['date']
    try:
        country_holidays = holidays.CountryHoliday(country)
    except KeyError:
        return None
    return country_holidays.get(date)


df_train['holiday'] = df_train.apply(get_holiday, axis=1)
df_test['holiday'] = df_test.apply(get_holiday, axis=1)


df_train["group"] = (df_train['Year']-2010)*48+df_train['Month']*4+df_train['Day']//7
df_test["group"] = (df_test['Year']-2010)*48+df_test['Month']*4+df_test['Day']//7


df_train.drop(columns= {"country_code", "date"}, axis = 1, inplace = True)
df_test.drop(columns= {"country_code", "date"}, axis = 1, inplace = True)


df_train["holiday"].value_counts()


from sklearn.preprocessing import LabelEncoder

cat_cols = list(df_train.select_dtypes(include=['object']).columns)
#label_encoders = {col: LabelEncoder() for col in cat_cols}

for col in cat_cols:
    combined_data = pd.concat([df_train[col], df_test[col]])
    le = LabelEncoder()
    le.fit(combined_data)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])


df_train["num_sold"] = np.log1p(df_train['num_sold'])


df_train.drop(columns = {"inflation"}, inplace= True)
df_test.drop(columns = {"inflation"}, inplace= True)


X = df_train.drop(['num_sold'], axis=1)
y = df_train['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer


def objective(trial):
    # Define hyperparameter space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
    }
    # Initialize model
    lgbm_model = LGBMRegressor(**params, verbose=0)
    
    # Fit the model
    lgbm_model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Predict and calculate MAPE
    y_pred = lgbm_model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return mape


#study = optuna.create_study(direction='minimize')  # Minimize MAPE
#study.optimize(objective, n_trials=100)


#print("Best parameters found:", study.best_params)
#print("Best score (MAPE):", study.best_value)


#paramslgbm = {'n_estimators': 930, 'learning_rate': 0.07359701109356444, 'num_leaves': 89, 'max_depth': 13, 'min_child_samples': 33, 'min_child_weight': 0.6186170197028386, 'subsample': 0.550316134876355, 'colsample_bytree': 0.946663821227242, 'reg_alpha': 0.4260787234500738, 'reg_lambda': 0.0013992554949328472}
paramslgbm = {'n_estimators': 972, 'learning_rate': 0.07886835774940058, 'num_leaves': 70, 'max_depth': 11, 'min_child_samples': 33, 'min_child_weight': 0.0034200823993429507, 'subsample': 0.9066501031637197, 'colsample_bytree': 0.5400387926077018, 'reg_alpha': 0.004372010855425869, 'reg_lambda': 0.0014856980584924064}
#paramslgbm {'n_estimators': 997, 'learning_rate': 0.09295024373715989, 'num_leaves': 74, 'max_depth': 10, 'min_child_samples': 17, 'min_child_weight': 0.20078002345038728, 'subsample': 0.6530069212617008, 'colsample_bytree': 0.5179348545134469, 'reg_alpha': 0.648912272436726, 'reg_lambda': 0.23340719869030896}


lgbm_model = LGBMRegressor(random_state=42, **paramslgbm)
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
lgbm_mape = mape(y_test, lgbm_preds)
print(f"LightGBM MAPE: {lgbm_mape:.4f}")


main_predict = np.expm1(lgbm_model.predict(df_test))
sample_sub["num_sold"] = main_predict
sample_sub.to_csv("submission.csv", index= False)




