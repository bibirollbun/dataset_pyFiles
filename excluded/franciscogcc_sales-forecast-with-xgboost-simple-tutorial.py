import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import itertools
from sklearn.metrics import mean_absolute_percentage_error
import optuna
import xgboost as xgb


import warnings
warnings.filterwarnings('ignore')


libs = [np,pd,sns,holidays,optuna,xgb]

def list_versions(libs):
    for lib in libs:
        print(f'{lib.__name__} {lib.__version__}')

list_versions(libs)


path_code = os.getcwd()
path = os.path.abspath(os.path.join(path_code, os.pardir))
path_data = f'{path}\\data'


train = pd.read_csv(path_data + '/train.csv')
X_test = pd.read_csv(path_data + '/test.csv')

# submission dataframe; num_sold will be populated in the end
final = pd.DataFrame({'id':X_test.id,'num_sold':[None]*len(X_test)})


# Drop id column
X_test = X_test.drop(['id'], axis=1)
train = train.drop(['id'], axis=1)


train.head()


train.shape


train.dtypes


for column in ['country','store','product']:
    if train[column].dtype == 'object':
        print(f'{train[column].value_counts()} \n')


train.describe()


# change data type of column 'date' to datetime
def set_date_type(df,col,format):
    '''
    Takes a column of a dataframe and changes its data type to datetime64
    '''

    if df[col].dtype != 'datetime64[ns]':
        df[col] = pd.to_datetime(df[col], format = format)

set_date_type(train,'date','%Y-%m-%d')
set_date_type(X_test,'date','%Y-%m-%d')


grouped_data_country = train.groupby(['country'])['num_sold'].sum().reset_index()

plt.figure(figsize=(6, 4))
sns.set_palette("bright")
sns.barplot(
    data=grouped_data_country, 
    x='country', 
    y='num_sold', 
    hue='country', 
    errorbar=None,
    dodge=False
)

plt.title('num_sold by Country', fontsize=12)
plt.xlabel('')
plt.ylabel('Number Sold', fontsize=10)
plt.xticks(rotation=45, fontsize=9)

plt.tight_layout()
plt.show()


grouped_data = train.groupby(['store', 'country', 'product'])['num_sold'].sum().reset_index()

plt.figure(figsize=(8, 5))
sns.set_palette("colorblind")
sns.barplot(
    data=grouped_data, 
    x='product', 
    y='num_sold', 
    hue='store', 
    errorbar=None
)

plt.title('num_sold by Store for Each Product - All Countries', fontsize=12)
plt.xlabel('Product', fontsize=10)
plt.ylabel('Number Sold', fontsize=10)
plt.xticks(rotation=45, fontsize=9)
plt.legend(title='Store', loc='upper right', prop={'size': 7})

plt.tight_layout()
plt.show()


sns.set_palette("colorblind")

# Create a FacetGrid using sns.catplot
g = sns.catplot(
    data=grouped_data, 
    x="product", 
    y="num_sold", 
    hue="store", 
    col="country", 
    kind="bar", 
    col_wrap=3,  
    height=5, 
    aspect=1.5, 
    dodge=True
)

sns.move_legend(g,"upper right",frameon=True)
g.set_titles("{col_name}")  
g.set_axis_labels("Product", "Number Sold")   

g.map(plt.grid, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

g.figure.subplots_adjust(hspace=0.4, wspace=0.3)  
g.figure.tight_layout()  # Ensure everything fits without overlapping
plt.show()


g = sns.barplot(
    grouped_data[grouped_data['country']=='Kenya'],
    x="product", 
    y="num_sold", 
    hue="store"
)

plt.grid(axis='y', linestyle='--', alpha=0.7, color='blue') 

plt.title('Kenya', fontsize=12)
plt.xlabel('Product', fontsize=10)
plt.ylabel('Number Sold', fontsize=10)
plt.xticks(rotation=45, fontsize=9)
plt.legend(title='Store', loc='upper right', prop={'size': 8})

g.figure.subplots_adjust(hspace=0.4, wspace=0.3)
g.figure.tight_layout()
plt.show()


# Extract Year-Month from the 'date' column
train['year_month'] = train['date'].dt.to_period('M')
X_test['year_month'] = X_test['date'].dt.to_period('M')

# Filter data for Country
finland_data = train[train['country'] == 'Finland']

grouped_data = finland_data.groupby(['store', 'product', 'year_month'])['num_sold'].sum().reset_index()
stores = grouped_data['store'].unique()

sns.set_theme(style="whitegrid")

for store in stores:
    plt.figure(figsize=(12, 6))
    store_data = grouped_data[grouped_data['store'] == store]
    
    # Loop through each combination of product and plot the time series line for this store
    for product, group in store_data.groupby('product'):
        plt.plot(group['year_month'].astype(str), group['num_sold'], label=f"Product: {product}")
    
    # Extract the unique months from the data for better display
    months_to_show = store_data['year_month'].dt.month.unique()
    months_to_show = [month for month in months_to_show if month in [1, 4, 7, 10]]

    # Convert the months back to period format (Year-Month) to match the x-axis
    months_to_show_period = store_data[store_data['year_month'].dt.month.isin(months_to_show)]['year_month'].astype(str).unique()
    
    plt.title(f"Sales for {store} in Finland")
    plt.ylabel("Number Sold")
    plt.xticks(months_to_show_period, rotation=45)
    plt.legend(title='Product')
    plt.tight_layout()
    plt.show()


print(f'\n Train Set: \n {train.isna().sum()} \n \n Test Set: \n {X_test.isna().sum()}')


train[train['num_sold'].isna()==True]['country'].value_counts()


missing_values = train[train['num_sold'].isna()==True]
not_missing_values = train[~train['num_sold'].isna()==True]

missing_values.shape


missing_values['store'].value_counts()


missing_values['product'].value_counts()


missing_values[missing_values['country']=='Canada']['product'].value_counts()


def check_missing_values(train, missing_values, not_missing_values):
    '''
    Takes the training set 'train', its matrix of missing values and non-NaN values,
    and checks the months with missing and available data for each country, store, and product.
    '''
    # Get unique values for year_month, country, store, and product
    unique_year_month = train['year_month'].unique()
    unique_countries = ['Canada', 'Kenya']
    unique_stores = train['store'].unique()
    unique_products = train['product'].unique()

    # Create all combinations of year_month, country, store, and product
    year_month_country_store_product_pairs = pd.DataFrame(
        itertools.product(unique_year_month, unique_countries, unique_stores, unique_products),
        columns=['year_month', 'country', 'store', 'product']
    )

    # For the missing months
    missing_months = (
        missing_values
        .groupby(['year_month', 'country', 'store', 'product'])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [unique_year_month, unique_countries, unique_stores, unique_products],
                names=['year_month', 'country', 'store', 'product']
            ),
            fill_value=0
        )
        .reset_index(name='missing_values')
    )

    # For the months with data
    available_months = (
        not_missing_values
        .groupby(['year_month', 'country', 'store', 'product'])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [unique_year_month, unique_countries, unique_stores, unique_products],
                names=['year_month', 'country', 'store', 'product']
            ),
            fill_value=0
        )
        .reset_index(name='available_values')
    )

    # Merge the unique pairs with the missing counts
    result = year_month_country_store_product_pairs.merge(
        missing_months,
        on=['year_month', 'country', 'store', 'product'],
        how='left'
    ).merge(
        available_months,
        on=['year_month', 'country', 'store', 'product'],
        how='left'
    ).fillna(0)  # Fill NaN values with 0

    # Calculate the percentage of missing values
    result['%_missing'] = (
        result['missing_values'] * 100 /
        (result['missing_values'] + result['available_values'])
    ).fillna(0)  # Handle division by zero

    return result

# Example usage
result = check_missing_values(train, missing_values, not_missing_values)
result.head()


result['%_missing'].value_counts().sort_index(ascending=False).sort_index(ascending=False)


# Filter combinations with 100% missing values
missing_combinations = result[result['%_missing'] == 100][['year_month', 'country', 'store', 'product']]

# Find matching rows in train and set 'num_sold' to 0
train.loc[
    train.set_index(['year_month', 'country', 'store', 'product']).index.isin(
        missing_combinations.set_index(['year_month', 'country', 'store', 'product']).index
    ),
    'num_sold'
] = 0

train.head()


# Remove values that were set to 0 since they will affect the model's performance
train = train[train['num_sold'] != 0]


median_values = (
    train[train['num_sold'].isna()==False]
    .groupby(['country', 'store', 'product', 'year_month'])['num_sold']
    .median()
    .reset_index()
)

median_values.rename(columns={'num_sold': 'median_num_sold'}, inplace=True)

train = train.merge(median_values, on=['country', 'store', 'product', 'year_month'], how='left')

train.head()


train['num_sold'] = train['num_sold'].fillna(train['median_num_sold'])

train[train['num_sold'].isna()==True]


train = train.drop('median_num_sold',axis=1)


train[train.duplicated()]


X_train = train.drop(columns=['num_sold'])
y_train = train['num_sold']


# get column with day of the week
def week_day(df,col,new_col):
    '''
    creates a new column with the week day (0 to 6) from a column with dates
    '''
    df[new_col] = df[col].dt.weekday

week_day(X_train,'date','week_day')
week_day(X_test,'date','week_day')


# Create a column that flags holidays

# Create a dictionary of holiday objects for each unique country
unique_countries = pd.concat([train['country'], X_test['country']]).unique()
holiday_objects = {country: holidays.CountryHoliday(country) for country in unique_countries}


def is_holiday(row):
    '''
    creates a new binary column that flags dates which are holidays (based on date and country columns)
    '''

    country_code = row['country']
    date = row['date']

    if country_code in holiday_objects:
        return 1 if date in holiday_objects[country_code] else 0
    else:
        return 0  # Default to 0 if country code is not supported (not the case)


X_train['is_holiday'] = X_train.apply(is_holiday, axis=1)
X_test['is_holiday'] = X_test.apply(is_holiday, axis=1)


gdp = pd.read_csv(path_data + '/gdp_per_capita.csv')
gdp = gdp[gdp['Country'].isin(train['country'].unique())]
gdp = gdp.filter(regex="^201[0-9]$").join(gdp['Country']).reset_index(drop=True)
gdp = pd.melt(gdp, id_vars=["Country"], var_name="year", value_name="gdp")
gdp["year"] = pd.to_numeric(gdp["year"])
gdp.rename(columns={'Country':'country'},inplace=True)

gdp.head()


def add_gdp(df,gdp):
    '''
    Adds column gdp to the train and test dataframes
    '''

    df['year'] = df['date'].dt.year

    df = df.merge(gdp, left_on=['country', 'year'], right_on=['country', 'year'], how='left')

    df = df.drop(columns=['year']) 

    return df


X_train = add_gdp(X_train,gdp)
X_test = add_gdp(X_test,gdp)

X_train['gdp'] = X_train['gdp'].round(2)
X_test['gdp'] = X_test['gdp'].round(2)


def date_to_sin_cos(df, date_col):
    '''
    Creates 2 new columns from an existing datetime column. the new columns are the sin and the cos of the day of the year
    '''
    df[f'{date_col}_sin'] = np.sin(2*np.pi*df[date_col].dt.dayofyear/365)
    df[f'{date_col}_cos'] = np.cos(2*np.pi*df[date_col].dt.dayofyear/365)

date_to_sin_cos(X_train,'date')
date_to_sin_cos(X_test,'date')


X_train['year_month'] = X_train['year_month'].dt.month
X_test['year_month'] = X_test['year_month'].dt.month

X_train.rename(columns={'year_month': 'month'}, inplace=True)
X_test.rename(columns={'year_month': 'month'}, inplace=True)


X_val = X_train[X_train['date']>='2015-01-01']
X_train = X_train[X_train['date']<'2015-01-01']

y_val = y_train[y_train.index.isin(X_val.index)]
y_train = y_train[y_train.index.isin(X_train.index)]


X_train_date = X_train['date']
X_test_date = X_test['date']
X_val_date = X_val['date']

X_train.drop(columns=['date'],inplace=True)
X_test.drop(columns=['date'],inplace=True)
X_val.drop(columns=['date'],inplace=True)


X_train


categorical_columns = ['country','store','product']
for col in categorical_columns:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')


def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),  
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),  
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),  
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),  
        'gamma': trial.suggest_uniform('gamma', 0, 5),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  
        'enable_categorical': True,  
        'use_label_encoder': False
    }

    # Create DMatrix for training and validation sets with enable_categorical=True
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)    

    # Train the model 
    model = xgb.train(param, dtrain, num_boost_round=param['n_estimators'])

    # Predict on the validation set
    y_pred = model.predict(dval)

    mape = mean_absolute_percentage_error(y_val, y_pred)

    return mape  

# Set up the Optuna study and start the tuning process
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)


# best_params1 = study.best_params
best_params1 = {'max_depth': 10, 'learning_rate': 0.013666044271808698, 'n_estimators': 500, 'subsample': 0.9843675292451406, 'colsample_bytree': 0.901523476961581, 'gamma': 0.24494912077894382, 'min_child_weight': 9} # Results of my grid search

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)    
dtest = xgb.DMatrix(X_test, enable_categorical=True)    

final_model = xgb.train(best_params1, dtrain, num_boost_round=best_params1['n_estimators'])


y_train_pred = final_model.predict(dtrain)
y_val_pred = final_model.predict(dval)
y_test_pred = final_model.predict(dtest)

y_train_pred = pd.Series(y_train_pred, index=y_train.index)
y_val_pred = pd.Series(y_val_pred, index=y_val.index)
y_test_pred = pd.Series(y_test_pred)


mean_absolute_percentage_error(y_train,y_train_pred), mean_absolute_percentage_error(y_val,y_val_pred)


final['num_sold'] = y_test_pred 
final.head()


final.to_csv(path_data + '/submission.csv',index=False)


X_train['date'] = X_train_date
X_val['date'] = X_val_date
X_test['date'] = X_test_date

X_train['num_sold'] = y_train
X_val['num_sold'] = y_val

X_train['num_sold_pred'] = y_train_pred
X_val['num_sold_pred'] = y_val_pred
X_test['num_sold_pred'] = y_test_pred



selected_country = "Italy"
selected_store = "Discount Stickers"
selected_product = "Holographic Goose"

train_filtered = X_train[(X_train['country'] == selected_country) & 
                         (X_train['product'] == selected_product) & 
                         (X_train['store'] == selected_store)]

val_filtered = X_val[(X_val['country'] == selected_country) & 
                     (X_val['product'] == selected_product) & 
                     (X_val['store'] == selected_store)]

test_filtered = X_test[(X_test['country'] == selected_country) & 
                       (X_test['product'] == selected_product) & 
                       (X_test['store'] == selected_store)]


plt.figure(figsize=(12, 6))

# Plot num_sold - the actual values

plt.scatter(train_filtered['date'], train_filtered['num_sold'], label='Train', color='blue',alpha=0.3)
plt.scatter(val_filtered['date'], val_filtered['num_sold'], label='Validation', color='orange',alpha=0.3)


# Plot num_sold_pred - the model predictions

# plt.scatter(train_filtered['date'], train_filtered['num_sold_pred'], label='Train', color='black',alpha=0.3)
plt.scatter(val_filtered['date'], val_filtered['num_sold_pred'], label='Validation Pred', color='red',alpha=0.3)
plt.scatter(test_filtered['date'], test_filtered['num_sold_pred'], label='Test Pred', color='green',alpha=0.3)


plt.title(f'Results for {selected_country} - {selected_product} - {selected_store}')
plt.legend()
plt.show()


importance = final_model.get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
sorted_importance

