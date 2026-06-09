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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
sub_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("data shape :",sub_data.shape)


print(train_data.columns)
print(test_data.columns)
print(sub_data.columns)


train_data.isna().sum()


print(train_data['product'].unique())
print(train_data['store'].unique())
print(train_data['country'].unique())


# 각 나라, 가게, 제품별 행 개수 계산
grouped = train_data.groupby(['country', 'store', 'product']).size().reset_index(name='num_rows')

# 결과 출력
print(grouped.head(100))


# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def plot_missing_sales_data(df):
    sales_counts = df.groupby(["country", "store", "product"])['num_sold'].count().rename("num_rows")
    missing_sales_data = sales_counts.loc[sales_counts != 2557].reset_index()
    num_plots = len(missing_sales_data)
    fig_rows = num_plots
    fig, axes = plt.subplots(fig_rows, 1, figsize=(20, fig_rows * 4))

    if fig_rows == 1:
        axes = [axes]

    for i, (country, store, product) in enumerate(missing_sales_data[["country", "store", "product"]].values):
        if i >= fig_rows:
            break
        product_data = df.loc[
            (df["country"] == country) &
            (df["store"] == store) &
            (df["product"] == product)
        ]

        missing_dates = product_data.loc[product_data["num_sold"].isna()]

        if product_data["num_sold"].notna().any():
            sns.lineplot(data=product_data, x="date", y="num_sold", ax=axes[i], color="skyblue")

            for missing_date in missing_dates["date"]:
                axes[i].axvline(missing_date, color='red', linestyle='-', linewidth=1, alpha=0.7, label='Missing Date')
                
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, color='red', transform=axes[i].transAxes)

        axes[i].set_title(f"{country} - {store} - {product}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Date", fontsize=12)
        axes[i].set_ylabel("Number Sold", fontsize=12)
        axes[i].set_xticks([])  

    # Adjust layout
    plt.subplots_adjust(top=0.95)

plot_missing_sales_data(train_data)


import matplotlib.pyplot as plt
import seaborn as sns

#Generate separate graphs for each store.
sns.set_palette("pastel")

def plot_store(df):
    
    for store in df["store"].unique():
        store_data = df[df["store"] == store]  
        
        products = store_data["product"].unique()
        num_products = len(products)

        fig_rows = (num_products + 1) // 2  
        fig, axes = plt.subplots(fig_rows, 2, figsize=(20, 5 * fig_rows), sharex=True, sharey=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        
        for i, prod in enumerate(products):   
            product_data = store_data[store_data["product"] == prod]
            avg_sales = product_data["num_sold"].mean()          
            sns.lineplot(data=product_data, x="date", y="num_sold", hue="country", ax=axes[i])
            axes[i].set_title(f"Store: {store} - Product: {prod} (Avg Sales: {avg_sales:.2f})", fontsize=14, fontweight="bold")
            axes[i].set_xticks([])
            
        for j in range(len(products), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

plot_store(train_data)



country_weights = train_data.groupby("country")["num_sold"].sum() / train_data["num_sold"].sum()
country_ratio = (
    train_data.groupby(["date", "country"])["num_sold"].sum()
    .div(train_data.groupby("date")["num_sold"].sum())
    .reset_index()
)

plt.figure(figsize=(20, 10))
ax = sns.lineplot(data=country_ratio, x="date", y="num_sold", hue="country")
ax.set_ylabel("Proportion of sales")
ax.set_title("Proportion of Sales by Country Over Time")
ax.set_xticks([])  
plt.show()


train_data['date'] = pd.to_datetime(train_data['date'])
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['weekday'] = train_data['date'].dt.weekday

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['weekday'] = test_data['date'].dt.weekday

train_data['day_of_year'] = train_data['date'].dt.dayofyear  
train_data['sin_day_of_year'] = np.sin(2 * np.pi * train_data['day_of_year'] / 365) 
train_data['cos_day_of_year'] = np.cos(2 * np.pi * train_data['day_of_year'] / 365) 

train_data['day_sin'] = np.sin(2 * np.pi * train_data['day'] / 31)  
train_data['day_cos'] = np.cos(2 * np.pi * train_data['day'] / 31)

train_data['month_sin'] = np.sin(2 * np.pi * train_data['month'] / 12)  
train_data['month_cos'] = np.cos(2 * np.pi * train_data['month'] / 12)  

train_data['weekday_sin'] = np.sin(2 * np.pi * train_data['weekday'] / 7) 
train_data['weekday_cos'] = np.cos(2 * np.pi * train_data['weekday'] / 7)  

train_data['group'] = (train_data['year'] - 2020) * 48 + train_data['month'] * 4 + train_data['day'] 

test_data['day_of_year'] = test_data['date'].dt.dayofyear  
test_data['sin_day_of_year'] = np.sin(2 * np.pi * test_data['day_of_year'] / 365)  
test_data['cos_day_of_year'] = np.cos(2 * np.pi * test_data['day_of_year'] / 365)  

test_data['day_sin'] = np.sin(2 * np.pi * test_data['day'] / 31)  
test_data['day_cos'] = np.cos(2 * np.pi * test_data['day'] / 31)

test_data['month_sin'] = np.sin(2 * np.pi * test_data['month'] / 12)  
test_data['month_cos'] = np.cos(2 * np.pi * test_data['month'] / 12)  

test_data['weekday_sin'] = np.sin(2 * np.pi * test_data['weekday'] / 7) 
test_data['weekday_cos'] = np.cos(2 * np.pi * test_data['weekday'] / 7) 

test_data['group'] = (test_data['year'] - 2020) * 48 + test_data['month'] * 4 + test_data['day']

print(test_data.head())


#In my case, dropping missing values slightly improves the model's performance.
#I'd appreciate it if you could share any good ideas.
# train_data = train_data.dropna() 

yearly_country_ratio = (
    train_data.groupby(["year", "country"])["num_sold"].sum()
    .div(train_data.groupby("year")["num_sold"].sum())
    .reset_index()
)
yearly_country_ratio.head()

# Function to calculate weighted ratio for each country
def calculate_weighted_ratio(country, target_country, now_year, yearly_country_ratio, reference_data):
    """
    Calculate the weighted ratio of num_sold for a specific country
    """
    target_ratio = yearly_country_ratio[
        (yearly_country_ratio['country'] == target_country) & 
        (yearly_country_ratio['year'] == now_year)
    ]['num_sold'].values[0]
    
    country_ratio = yearly_country_ratio[
        (yearly_country_ratio['country'] == country) & 
        (yearly_country_ratio['year'] == now_year)
    ]['num_sold'].values[0]

    ref_num_sold = reference_data[reference_data['country'] == country]['num_sold'].values[0]
   
    return ref_num_sold * (target_ratio / country_ratio)

# Filling missing values
for date in tqdm(train_data["date"].unique()):  # Iterate over each unique date
    # Extract rows where num_sold is missing
    missing_data = train_data[(train_data["date"] == date) & train_data["num_sold"].isna()]
    
    # Extract rows where num_sold is not missing for the given date
    reference_data = train_data[(train_data["date"] == date) & train_data["num_sold"].notna()]
    
    # Filter yearly country ratio for the given year
    yearly_country_data = yearly_country_ratio[yearly_country_ratio['year'] == reference_data['year'].values[0]]

    for i, row in missing_data.iterrows():
        # Extract country, store, and product for the current missing value
        missing_country = row["country"]
        missing_store = row["store"]
        missing_product = row["product"]

        # Filter data for the same date, product, and store but different country
        relevant_reference_data = reference_data[
            (reference_data["store"] == missing_store) &
            (reference_data["product"] == missing_product) &
            (reference_data["country"] != missing_country)
        ]
        
        # Calculate country ratios
        countries = ['Singapore', 'Italy', 'Finland']
        now_year = reference_data['year'].values[0]
        
        # Compute the weighted average
        filled_value = sum(
            calculate_weighted_ratio(country, missing_country, now_year, yearly_country_data, relevant_reference_data)
            for country in countries
        ) / len(countries)

        # Fill the missing value
        train_data.loc[i, "num_sold"] = int(filled_value)



import holidays
def add_holiday_flag(data, country_col, date_col, holidays_dict):
    data['holiday_flag'] = 0
    for country, holiday_list in holidays_dict.items():
        data.loc[data[country_col] == country, 'holiday_flag'] = data[date_col].isin(holiday_list).astype(int)
    return data

country_codes = dict(zip(np.sort(train_data.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
holiday_dict = {country: holidays.country_holidays(code, years=range(2010, 2020)) for country, code in country_codes.items()}

train_data = add_holiday_flag(train_data, 'country', 'date', holiday_dict)
test_data = add_holiday_flag(test_data, 'country', 'date', holiday_dict)


import seaborn as sns
import matplotlib.pyplot as plt

holiday_avg = train_data.groupby(['country', 'holiday_flag'])['num_sold'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='country', y='num_sold', hue='holiday_flag', data=holiday_avg)  # 색상 팔레트 설정
plt.title('Average num_sold by Country and Holiday Flag', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=14)
plt.ylabel('Average num_sold', fontsize=14)
plt.xticks(rotation=45, fontsize=12) 
plt.yticks(fontsize=12)  
plt.legend(title='Holiday Flag ', title_fontsize=14, loc='upper right', frameon=False, labelspacing=1.2, fontsize=12)
plt.tight_layout()
plt.show()



#Label Encoding
for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])  
 
print("train_data head:\n", train_data[['country', 'store', 'product']].head())
print("test_data head:\n", test_data[['country', 'store', 'product']].head())


columns_to_convert = ["country", "store", "product", "year", "month", "day", "weekday", "holiday_flag"]
train_data[columns_to_convert] = train_data[columns_to_convert].astype("category")
test_data[columns_to_convert] = test_data[columns_to_convert].astype("category")


X = train_data.drop(['id', 'date', 'day_of_year', 'num_sold'], axis=1)
y = np.log1p(train_data['num_sold'])
X_test = test_data.drop(['id', 'date', 'day_of_year'], axis=1)


X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X.info()


import lightgbm as lgb
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error

# K-Fold
n_splits = 5 
seed = 43
try_num = 5  # Number of Optuna trials
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Optuna objective function
def objective(trial):
    bagging_freq = trial.suggest_int('bagging_freq', 1, 7)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.1)
    num_leaves = trial.suggest_int('num_leaves', 31, 100)
    max_depth = trial.suggest_int('max_depth', 6, 16)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.3, 1.0)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.3, 1.0)
    n_estimators = trial.suggest_int('n_estimators', 1000, 2000)
    min_child_samples = trial.suggest_int('min_child_samples', 10, 100)  # Added parameter

    mape_scores = []
    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        
        # LGBM 
        model = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            metric='mape',
            random_state=seed,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_child_samples=min_child_samples, 
            verbose=-1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
        y_pred = model.predict(X_valid)
        mape = mean_absolute_percentage_error(y_valid, y_pred)
        mape_scores.append(mape)
    
    return np.mean(mape_scores)

# Optuna 
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=try_num) 

print("Best hyperparameters: ", study.best_params)
print("Best MAPE: ", study.best_value)

best_params = study.best_params

final_model = lgb.LGBMRegressor(
    objective='regression',
    metric='mape',
    random_state=seed,
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    feature_fraction=best_params['feature_fraction'],
    bagging_fraction=best_params['bagging_fraction'],
    bagging_freq=best_params['bagging_freq'],
    min_child_samples=best_params['min_child_samples'],
    verbose=-1
)

final_model.fit(X, y)
y_pred_test = final_model.predict(X_test)
y_pred_test = np.expm1(y_pred_test).round()

sub_data['num_sold'] = y_pred_test
sub_data.to_csv('predictions.csv', index=False)

