import numpy as np 
import pandas as pd 
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import requests
import holidays


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.head()


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

def add_gdp_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  
    df['gdp'] = df.apply(lambda row: gdp_data.get((row['country'], row['year']), None), axis=1)
    return df


def add_holiday_feature(df):
    country_holidays = {
        'Canada': holidays.CountryHoliday('CA'),
        'Finland': holidays.CountryHoliday('FI'),
        'Italy': holidays.CountryHoliday('IT'),
        'Kenya': holidays.CountryHoliday('KE'),
        'Norway': holidays.CountryHoliday('NO'),
        'Singapore': holidays.CountryHoliday('SG')
    }

    df['date'] = pd.to_datetime(df['date'])

    df['is_holiday'] = df.apply(
        lambda row: row['date'] in country_holidays.get(row['country'], []), axis=1
    )
    return df



train = add_holiday_feature(train)
test = add_holiday_feature(test)
train = add_gdp_feature(train)
test = add_gdp_feature(test)


train.columns


train.head()


train = train.dropna(subset=['num_sold'])

for df in [train, test]:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['month_store'] = df['month'].astype(str) + "_" + df['store']
    df['month_product'] = df['month'].astype(str) + "_" + df['product']



train['num_sold_log'] = np.log1p(train['num_sold'])  

X = train.drop(columns=['id', 'num_sold', 'num_sold_log', 'date'])
y = train['num_sold_log']

categorical_features = ['country', 'store', 'product', 'month_country', 'month_store', 'month_product']
numerical_features = ['gdp', 'year', 'month', 'day', 'day_of_week', 'is_weekend','is_holiday']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])



tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y), 1):
    print(f"Fold {fold}:")
    print(f"  Training years: {train['year'].iloc[train_idx].unique()}")
    print(f"  Validation years: {train['year'].iloc[val_idx].unique()}")
    print("-" * 40)

final_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_percentage_error')
final_mape = -np.mean(final_scores)

print(f"Final MAPE: {final_mape:.4f}")


model.fit(X, y)

test['num_sold'] = np.expm1(model.predict(test.drop(columns=['id', 'date'])))


submission = test[['id', 'num_sold']]
submission.to_csv("submission.csv", index=False)


coefficients = model.named_steps['regressor'].coef_

cat_columns = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)

feature_names = numerical_features + list(cat_columns) 

coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})


coeff_df['Abs_Coefficient'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False)

top_coeff_df = coeff_df.head(15)

plt.figure(figsize=(15, 12))
plt.barh(top_coeff_df['Feature'], top_coeff_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Coefficients of Linear Regression Model')
plt.show()



residuals = np.expm1(y) - np.expm1(model.predict(X))
plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y), residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual num_sold")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


