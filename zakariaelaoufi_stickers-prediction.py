import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


stickers = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


stickers['date'] = pd.to_datetime(stickers['date'])
stickers.drop('id',axis=1, inplace= True)
stickers.head()


stickers.describe().T


stickers.info()


stickers.isna().sum()


def get_day_month_year(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'] >= 5
    df['is_weekend'] = df['weekday'].map(lambda x: 1 if x == True else 0)
    df.drop('date',axis=1,inplace= True)
    return df


stickers_with_days = get_day_month_year(stickers)


stickers_with_days.info()


def missing_value_proportions(df):
    for col in df.columns:
        prop = df[col].isna().sum()/len(stickers)*100
        print(f'{col}: {round(prop,2)}%')


missing_value_proportions(stickers_with_days)


# The proportion of missing values in the data is less than 5%. 
# Therefore, we can delete them without significantly affecting the overall dataset.


nb_rows_bef = len(stickers_with_days)


stickers_with_days = stickers_with_days.dropna(axis=0).reset_index(drop=True)


nb_rows_aff = len(stickers_with_days)


print(f'nb of rows before {nb_rows_bef} and after {nb_rows_aff}')
print(f'{nb_rows_bef - nb_rows_aff} rows deleted')


stickers_with_days.isna().sum()


def plot_multiple_boxplots(data):
    # Select only numeric columns
    numeric_columns = data.select_dtypes(include=['float64']).columns

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data[numeric_columns])
    plt.title('Outlier Detection')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_multiple_boxplots(stickers_with_days)


def remove_outliers_iqr(data):
    # Select only numeric columns
    numeric_columns = data.select_dtypes(include=['float64']).columns
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = data.copy()
    
    for col in numeric_columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove rows with values outside the bounds
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned


clean_stickers = remove_outliers_iqr(stickers_with_days)


print(f'{stickers.shape[0]-clean_stickers.shape[0]} rows deleted')


plot_multiple_boxplots(clean_stickers)


clean_stickers.describe().T


pip install pycountry


import requests
import pycountry

# This function return the abbreviation of a country 
def get_country_abbreviation(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_2
    except AttributeError:
        return "Country not found"


def get_gdp_per_capita_to_df(df):
    data = df.copy()
    # get years and countries exist within the data
    years = data['year'].unique().tolist()
    countries = data['country'].unique().tolist()
    
    # fill the dic with the correspondent abbreviation for a country
    country_abv = {}
    for country in countries:
        country_abv[country] = get_country_abbreviation(country)
    
    # get the GDP for each country by year
    gdp_per_capita = {}
    for abv in country_abv:
        gdp_per_capita[abv] = {}
        
        for year in years:
            url = f'https://api.worldbank.org/v2/country/{country_abv[abv]}/indicator/NY.GDP.PCAP.CD?date={year}&format=json'
            resp = requests.get(url)
            
            if resp.status_code == 200:
                gdp_info = resp.json()
                gdp_year = gdp_info[1][0]['value']
                gdp_per_capita[abv][year] = round(gdp_year,2)
            else:
                print(f'error {resp.status_code}')
    
    # add GDP column to the data
    data['GDP'] = data.apply(lambda row: gdp_per_capita[row['country']][row['year']], axis=1)
    return data


stickers_with_gdp = get_gdp_per_capita_to_df(clean_stickers)


stickers_with_gdp.head()


stickers_with_gdp.info()


def country_label_encoded(data):
    country_list = data.groupby('country').agg({'GDP': 'mean', 'num_sold': 'mean'}).sort_values(by=['GDP','num_sold'], ascending=True).reset_index()['country'].tolist()
    country_mapping = {country: idx for idx, country in enumerate(country_list)}
    stickers_with_gdp['country'] = stickers_with_gdp['country'].map(country_mapping)
    return country_mapping


country_order = country_label_encoded(stickers_with_gdp)


country_order


stickers_with_gdp.head()


obj_col = stickers_with_gdp.select_dtypes(include='object').columns
print(obj_col)


print(stickers_with_gdp['store'].unique())
print(stickers_with_gdp['product'].unique())


X = stickers_with_gdp.drop('num_sold', axis=1)
y = stickers_with_gdp['num_sold']


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='if_binary'), obj_col)
    ], remainder='passthrough', verbose_feature_names_out=False
).set_output(transform="pandas")

transformed_X_stickers = transformer.fit_transform(X)


transformed_X_stickers.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)


col_to_delete = ['store_Discount Stickers','product_Holographic Goose']
transformed_X_stickers.drop(col_to_delete, axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(transformed_X_stickers, y, test_size=0.05, random_state=42)


rf_regressor.fit(X_train, y_train)


rf_regressor.oob_score_


rf_regressor.score(X_test, y_test)


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

rmse = mean_squared_error(y_test, rf_regressor.predict(X_test))
rmsep = mean_absolute_percentage_error(y_test, rf_regressor.predict(X_test)) 


rmse, rmsep


test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')  


test_df_date_decomposition = get_day_month_year(test_df.drop('id',axis=1))
test_df_date_decomposition.head()


test_df_gdp = get_gdp_per_capita_to_df(test_df_date_decomposition)


test_df_gdp['country'] = test_df_gdp['country'].map(country_order)


test_df_gdp.head()


transformed_test_df = transformer.transform(test_df_gdp) 
transformed_test_df.drop(col_to_delete, axis=1, inplace=True)


transformed_test_df.head()


pred = rf_regressor.predict(transformed_test_df)


sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub['num_sold'] = pred
sub.to_csv('submission.csv', index=False)

