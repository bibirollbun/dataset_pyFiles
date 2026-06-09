!pip install -U --q pycountry


# Data
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import matplotlib.gridspec as gridspec

# Statistical
from scipy import stats
import holidays
import pycountry

# Preprocessing and Modeling
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor

# LightGBM
import lightgbm as lgb

# Other
import datetime
import warnings
warnings.filterwarnings('ignore')

# Global Config
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
color_palette = sns.color_palette(['#D62728', '#9467BD', '#8C564B', '#1F77B4', '#FF7F0E', '#2CA02C'])
np.random.seed(42)

# Config variables
SEED = 42
n_splits = 5
n_estimators = 1000
early_stopping_rounds = 100
FE_holidays = True



train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", index_col = "id")

print("shape: train, test\n", train.shape,test.shape)
display(train.tail())


train.dropna(subset=["num_sold"], inplace=True)
X = train.copy()
y = X.pop("num_sold")

full = pd.concat([X,test], axis = 0)


print("Missing:\n Target")
display(train["num_sold"].isnull().sum())

print("y: ")
y.isnull().sum()



train.duplicated().sum(),test.duplicated().sum()


print("train. \n info: \n",train.info())
print("\ndescriptive statistics: \n",train.describe())


print("y missing : ")
print("%: ",y.isnull().mean())
print("sum: ", y.isnull().sum())

def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple. 
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(full)

# Missing Matrix ...
# msno.matrix(full)


print("train.num_sold.median() : ", train.num_sold.median())
print("train.num_sold.mean() : ", train.num_sold.mean())
print("train.num_sold.max() : ", train.num_sold.max())

train.num_sold.hist(bins = 40)
plt.title("Frequency Histogram")
plt.axvline(train.num_sold.median(), label = "median", color = "r")
plt.axvline(train.num_sold.mean(), label = "mean", color = "y")
plt.axvline(train.num_sold.max(), label = "max", color = "g")
plt.legend()


print(" with a log1np transformation ...\n")
# train.num_sold.hist(bins = 40)
plt.hist(np.log1p(train.num_sold), bins = 40, color = "black")
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

def plotting_3_chart(df, feature):
    # Ensure matplotlib style is set correctly
    style.use('fivethirtyeight')
    
    # Drop missing values
    df = df.dropna(subset=[feature])
    
    # Create a figure with constrained layout
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    
    # Create a grid layout
    grid = GridSpec(ncols=3, nrows=3, figure=fig)

    # Histogram plot
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Distribution')
    sns.histplot(df[feature], kde=True, ax=ax1)  # Updated from sns.distplot to sns.histplot

    # QQ plot
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ Plot')
    stats.probplot(df[feature], plot=ax2)

    # Box plot
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(y=df[feature], ax=ax3)  # Adjusted for modern Seaborn usage

    # Display skewness and kurtosis
    print("Skewness: {:.2f}".format(df[feature].skew()))
    print("Kurtosis: {:.2f}".format(df[feature].kurt()))
    
    # Show the plots
    plt.show()


plotting_3_chart(train, 'num_sold')



from scipy.stats import f_oneway, kruskal
import pandas as pd

def cat_kruskal(train_df: pd.DataFrame, target: str, to_drop: str = None) -> pd.DataFrame:
    """
    Perform ANOVA and Kruskal-Wallis tests on categorical features in the DataFrame.
    
    Parameters:
        train_df (pd.DataFrame): The training DataFrame.
        target (str): The target column name.
        to_drop (str): The column to exclude from the analysis, if any.

    Returns:
        pd.DataFrame: A DataFrame with test statistics and p-values.
    """
    test_cat_list = []
    test_cat_cols = []
    
    # Select categorical columns
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
    
    # Drop the specified column if to_drop is not None
    if to_drop is not None:
        cat_cols = cat_cols.drop(to_drop)
    
    for col in cat_cols:
        # Group target values by the categorical feature
        test_group = train_df.groupby(col)[target].apply(list)
        
        # Perform ANOVA and Kruskal-Wallis tests
        f_oneway_result = f_oneway(*test_group)
        kruskal_result = kruskal(*test_group)
    
        # Store results
        test_cat_list.append(
            [f_oneway_result.statistic, f_oneway_result.pvalue, kruskal_result.statistic, kruskal_result.pvalue]
        )
        test_cat_cols.append(col)
    
    # Return results as a DataFrame
    return pd.DataFrame(
        test_cat_list,
        index=test_cat_cols,
        columns=['anova_statistic', 'anova_pvalue', 'kruskal_statistic', 'kruskal_pvalue']
    )

# Example usage
cat_kruskal(train_df = train, target = 'num_sold', to_drop='date')





sold_product = train.groupby(['product'])['num_sold'].sum()
sold_store = train.groupby(['store'])['num_sold'].sum()
sold_country = train.groupby(['country'])['num_sold'].sum()

fig = plt.figure(figsize=(15,10))
fig.set_facecolor('white')

ax1 = fig.add_subplot(1, 2, 1)
ax1.bar(sold_country.keys(), sold_country.values)
ax1.set_title('Sales on Every Country')
ax1.ticklabel_format(style='plain', axis='y')
ax1.set_ylim(0, 3e6)

ax2 = fig.add_subplot(2, 2, 2)
ax2.pie(sold_product.values, labels=sold_product.keys(), autopct="%.1f%%")
ax2.set_title('Total Product Sales', fontsize=16)

ax3 = fig.add_subplot(2, 2, 4)
ax3.pie(sold_store.values, labels=sold_store.keys(), autopct="%.1f%%")
ax3.set_title('Total Store Sales', fontsize=16);



fig = plt.figure(figsize=(20,10))
fig.set_facecolor('white')

ax1 = fig.add_subplot(1, 2, 1)
sns.barplot(data=train, x='country', y='num_sold', hue='product')
ax1.set_title('Product Sales by Product')

ax2 = fig.add_subplot(1, 2, 2)
sns.barplot(data=train, x='store', y='num_sold', hue='product')
ax2.set_title("Product Sales by Store");



import requests

# Function to fetch GDP per capita
def get_gdp_per_capita(country, year):
    alpha3 = {
        'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA',
        'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'
    }
    url = f"https://api.worldbank.org/v2/country/{alpha3[country]}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    response = requests.get(url).json()
    try:
        return response[1][0]['value']
    except (IndexError, TypeError):
        return None

countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
years = range(2010, 2020)
gdp_data = {}

for country in countries:
    for year in years:
        gdp_data[(country, year)] = get_gdp_per_capita(country, year)

# Add GDP feature to train and test DataFrames
def add_gdp_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  # Extract year from the date
    df['gdp'] = df.apply(lambda row: gdp_data.get((row['country'], row['year']), None), axis=1)
    return df

add_gdp_feature(full)
full.tail()



def get_indicator_data(country, year, indicator_code):
    alpha3 = {
        'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA',
        'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'
    }
    url = f"https://api.worldbank.org/v2/country/{alpha3[country]}/indicator/{indicator_code}?date={year}&format=json"
    response = requests.get(url).json()
    try:
        return response[1][0]['value']
    except (IndexError, TypeError):
        return None

# Indicator codes
indicators = {
    'Final Consumption Expenditure': 'NE.CON.TOTL.ZS',
    # 'Household and Business Savings': 'NY.GDP.PCAP.CD', # similar to gdp
    'Exports as Percentage of GDP': 'NE.EXP.GNFS.ZS',
    'Imports as Percentage of GDP': 'NE.IMP.GNFS.ZS',
    #'Poverty Rate': 'SI.POV.Dday', # Incomplete data
    'Unemployment Rate': 'SL.UEM.TOTL.ZS',
    'Population': 'SP.POP.TOTL',
    # 'Gross National Income Per Capita': 'NY.GDP.PCAP.CD' # similar to gdp
}

# List of countries and years
countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'] # this or: >>>  list(full.country.value_counts().index)
years = range(2010, 2020)

# Dictionary to store data for all indicators
indicator_data = {indicator: {} for indicator in indicators}

# Fetch the indicator data
for indicator, code in indicators.items():
    for country in countries:
        for year in years:
            indicator_data[indicator][(country, year)] = get_indicator_data(country, year, code)

# Function to add indicator features to the DataFrame
def add_indicator_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  # Extract year from the date
    
    # Add each indicator as a new feature
    for indicator in indicators:
        df[indicator] = df.apply(lambda row: indicator_data[indicator].get((row['country'], row['year']), None), axis=1)
    
    return df

# :)
add_indicator_features(full)



full["BOT"]= full["Exports as Percentage of GDP"] - full["Imports as Percentage of GDP"]
full.drop(["Exports as Percentage of GDP", "Imports as Percentage of GDP"], axis = 1, inplace = True)


full.head()


def full_date_FE(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['month_store'] = df['month'].astype(str) + "_" + df['store']
    df['month_product'] = df['month'].astype(str) + "_" + df['product']
    df['year_centered'] = df['year'] - df['year'].min()
    df['year_gdp_interaction'] = df['year_centered'] * df['gdp']
    # new incorp.
    df['year_tradeBalance_int'] = df['year_centered'] * df['BOT']
    df['year_FinalConsumption'] = df["year_centered"] * df['Final Consumption Expenditure']

    df['Quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)
    
    df['Quarter'] = df['Quarter'].astype('str')
    df['month'] = df['month'].astype('str')
    df['week_of_year'] = df['week_of_year'].astype('str')
    
    return df

full = full_date_FE(full)
full


def get_holiday_name(country_code, date_obj):
    country_holiday = holidays.CountryHoliday(country_code, years=date_obj.year)
    return country_holiday.get(date_obj)

def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_2  
    except KeyError:
        print(f"Unknown Country: {country_name}")
        return None

def get_holiday_for_row(row):
    country_code = get_country_code(row['country'])
    if country_code is None:
        return 'Unknown Country'
    
    try:
        date_obj = row['date']
    except ValueError:
        print(f"Invalid Date: {row['date']}")
        return 'Invalid Date'
    
    return get_holiday_name(country_code, date_obj)

full["Holidays_names"] = full.apply(get_holiday_for_row, axis = 1) 
full.Holidays_names.fillna("None", inplace = True)
# train_data['holidays_name'] = train_data.apply(get_holiday_for_row, axis=1)


from sklearn.preprocessing import LabelEncoder

def label_encoding(df):
    
    cat_columns = [col for col in full.columns if full[col].dtype == "O"]
    
    label_encoders = {}
    
    for col in cat_columns:
        label_encoder = LabelEncoder()
        df[f'{col}_codificado'] = label_encoder.fit_transform(df[col])
        label_encoders[col] = label_encoder  

    # print(df)
    
    # # ReconversiÃ³n (opcional)
    # print("\nReconversiÃ³n de la columna 'Color_codificado':")
    # print(label_encoders['Color'].inverse_transform(df['Color_codificado']))

    return df, label_encoders

full, encoders = label_encoding(full)


cat_columns = [col for col in full.columns if full[col].dtype == "O"]
full.drop(cat_columns, axis=1, inplace=True)
full.info()


full.drop('date',axis=1,inplace=True)


full.nunique()


train_size = train.shape[0]

X_train = full.iloc[:train_size] 
X_test = full.iloc[train_size:]


y_log = np.log1p(y)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Split datainto training set and test set
x_train, x_val, y_train, y_val = train_test_split(X_train, y_log, test_size=0.10, random_state=42, shuffle = True)


# import optuna
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_absolute_percentage_error
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split


# # x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# def objective(trial):
#   
#     params = {
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'rmse',
#         'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
#         'max_depth': trial.suggest_int('max_depth', 3, 20),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 10.0),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
#         'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-2, 100.0),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
#         'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
#         'gamma': trial.suggest_loguniform('gamma', 1e-4, 10.0),
#         'seed': 42,
#         'verbose': -1
#     }

#   
#     model = lgb.LGBMRegressor(**params)

#     
#     model.fit(
#         x_train, y_train,
#         eval_set=[(x_train, y_train), (x_val, y_val)],
#         eval_metric='rmse',
#         callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
#     )

#     
#     y_pred = model.predict(x_val, num_iteration=model.best_iteration_)
#     mape_score = mean_absolute_percentage_error(y_val, y_pred)
    
#     return mape_score


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50, timeout=600)


# print("Mejores parÃ¡metros:", study.best_params)


# print("Mejor MAPE:", study.best_value)



import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1620,
    'learning_rate': 0.004896522917845993,
    'max_depth': 16,
    'reg_alpha': 0.7724417012092296,
    'lambda_l2': 0.01,
    'min_child_samples': 20,
    'min_child_weight': 57,
    'colsample_bytree': 0.9091343258597363,
    'subsample': 0.7,
    'seed': 42,
    'gamma': 0.0033278145178718306,
    'verbose': -1,
    'device': 'cpu'
}

model = lgb.LGBMRegressor(**lgbm_params)


callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=True)]

# Entrenar el modelo
model.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],  # Dos conjuntos
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=30)]
)

# MAPE
y_pred = model.predict(x_val, num_iteration=model.best_iteration_)
mape_score = mean_absolute_percentage_error(y_val, y_pred)
print(f"MAPE: {mape_score:.4f}")



results = model.evals_result_
plt.figure(figsize=(10, 6))
plt.plot(results['training']['rmse'], label='Training', color='violet')
plt.plot(results['valid_1']['rmse'], label='Validation', color='orange', ls = "--")
plt.xlabel('Iter.')
plt.ylabel('RMSE')
plt.title('Training vs Validation')
plt.legend()
plt.grid()
plt.show()


# Importance of features

importance = model.feature_importances_
features = X_train.columns


importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(importance_df)


plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Importance of features')
plt.xlabel('LGBM model: Importance ')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



y_TEST_pred = model.predict(X_test, num_iteration=model.best_iteration_)
y_TEST_pred_reverted = np.expm1(y_TEST_pred)


# 98550 
submission = pd.DataFrame()
submission["id"] = X_test.index
submission["num_sold"] = y_TEST_pred_reverted
# submission.to_csv("submission.csv", index=False)
submission.to_csv('submission.csv', index=False)
submission


full.to_csv("full_stickersales.csv")

