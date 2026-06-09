import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
df2_test=pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
df_new=pd.read_csv('/kaggle/input/used-car-price-prediction-dataset/used_cars.csv')


df_new['milage'] = df_new['milage'].str.replace(' mi.', '', regex=False)
df_new['milage'] = df_new['milage'].str.replace(',', '', regex=False)
df_new['milage'] = df_new['milage'].astype(int)
df_new['price'] = df_new['price'].str.replace('$', '', regex=False)
df_new['price'] = df_new['price'].str.replace(',', '', regex=False)
df_new['price'] = df_new['price'].astype(int)
valid_brands = df_train['brand'].unique()
filtered_df_new = df_new[df_new['brand'].isin(valid_brands)]
df_combined = pd.concat([df_train, filtered_df_new], ignore_index=True)
df_train=df_combined


def accident_col(df):
    df['accident_dummy'] = df['accident'].apply(lambda x: 0 if x == 'NONE REPORTED' else (1 if x == 'AT LEAST 1 ACCIDENT OR DAMAGE REPORTED' else None))
    return df


def transmission_col(df):
    def classify_transmission(transmission):
        if 'M/T' in transmission or 'MT' in transmission or 'MANUAL' in transmission:
            return 'M/T'
        elif 'A/T' in transmission or 'AT' in transmission or 'AUTOMATIC' in transmission:
            return 'A/T'
        else:
            return 'OTHER'
    
    df['transmission'] = df['transmission'].apply(classify_transmission)
    return df


def ext_and_int_col(df):
    df['ext_col'] = df['ext_col'].replace({
        r'.*BLACK.*': 'BLACK',
        r'.*WHITE.*': 'WHITE',
        r'.*GRAY.*': 'GRAY',
        r'.*SILVER.*': 'SILVER',
        r'.*BLUE.*': 'BLUE',
        r'.*RED.*': 'RED',
    }, regex=True)

    df['ext_col'] = df['ext_col'].apply(lambda x: x if x in ['BLACK', 'WHITE', 'GRAY', 'SILVER', 'BLUE', 'RED'] else 'OTHERS')

    df['int_col'] = df['int_col'].replace({
        r'.*BLACK.*': 'IBLACK',
        r'.*WHITE.*': 'IWHITE',
        r'.*GRAY.*': 'IGRAY',
        r'.*SILVER.*': 'ISILVER',
        r'.*BLUE.*': 'IBLUE',
        r'.*RED.*': 'IRED',
        r'.*GREEN.*': 'IGREEN',
        r'.*BEIGE.*': 'IBEIGE',
        r'.*ORANGE.*': 'IORANGE',
    }, regex=True)

    df['int_col'] = df['int_col'].apply(lambda x: x if x in ['IBLACK', 'IBEIGE', 'IGRAY'] else 'IOTHERS')

    return df


def fuel_type_col(df):
    df['fuel_type'] = df['fuel_type'].replace({'HYBRID': 'HYBRID', 'PLUG-IN HYBRID': 'HYBRID','NOT SUPPORTED':'OTHER','–':'OTHER'})
    return df


def engine_col(df):
    # Create hp column
    df['hp'] = df['engine'].str.extract(r'(\d+\.\d+)HP').astype(float, errors='ignore')

    # Create engine displacement
    df['engine displacement'] = df['engine'].str.extract(r'(\d+\.\d+)L')
    df['engine displacement'] = df['engine displacement'].fillna(df['engine'].str.extract(r'(\d+\.\d+)LITER')[0])
    df['engine displacement'] = df['engine displacement'].astype(float, errors='ignore')

    # Create cylinder
    df['cylinder'] = df['engine'].str.extract(r'(\d+) CYLINDER')
    df['cylinder'] = df['cylinder'].fillna(df['engine'].str.extract(r'V(\d)')[0])
    df['cylinder'] = df['cylinder'].astype('Int64', errors='ignore')

    # Create fuel type
    df['fuel'] = df['engine'].str.extract(r'(GASOLINE|DIESEL|ELECTRIC|HYBRID)')
    df['fuel_type'] = df['fuel_type'].combine_first(df['fuel'])

    # Is it V type or not
    df['is_v_engine'] = df['engine'].str.contains(r'V\d+', case=False, na=False)

    # Create turbo
    df['turbo'] = df['engine'].str.contains('TWIN TURBO', case=False, na=False)
    
    # Create dohc
    df['dohc'] = df['engine'].str.contains('DOHC', case=False, na=False)
    return df


def fill_na(df, bugatti_avg_hp_external=1300):
    df['hp'] = df.groupby('brand')['hp'].transform(lambda x: x.fillna(x.mean()))
    df.loc[(df['brand'] == 'BUGATTI') & (df['hp'].isnull()), 'hp'] = bugatti_avg_hp_external

    most_common_fuel = df.groupby('brand')['fuel_type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    most_common_displacement = df.groupby('brand')['engine displacement'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    most_common_cylinder = df.groupby('brand')['cylinder'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

    df['fuel_type'] = df.apply(
        lambda row: most_common_fuel[row['brand']] if pd.isna(row['fuel_type']) else row['fuel_type'],
        axis=1
    )

    df['engine displacement'] = df.apply(
        lambda row: most_common_displacement[row['brand']] if pd.isna(row['engine displacement']) else row['engine displacement'],
        axis=1
    )

    df['cylinder'] = df.apply(
        lambda row: most_common_cylinder[row['brand']] if pd.isna(row['cylinder']) else row['cylinder'],
        axis=1
    )

    df['cylinder'] = df.apply(
        lambda row: most_common_cylinder[row['brand']] if pd.isna(row['cylinder']) else row['cylinder'],
        axis=1
    )
    df['accident_dummy'].fillna(0, inplace=True)
    
    if df['cylinder'].isnull().any():
        most_common_cylinder_value = df['cylinder'].mode()[0]
        df['cylinder'].fillna(most_common_cylinder_value, inplace=True)

    return df


def one_hot_encoding(df):
    df= pd.get_dummies(df, columns=['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col'], drop_first=False)
    return df



def age_and_usage(df):
    current_year = 2024
    df['age'] = current_year - df['model_year']

    df['annual_km'] = df.apply(
        lambda row: row['milage'] / row['age'] if row['age'] > 0 else row['milage'],
        axis=1
    )
    
    return df


def model_create(df, target='price', model_type='lightgbm'):
    X = df.drop([target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if  model_type == 'xgboost':
        model = XGBRegressor()
    elif model_type == 'lightgbm':
        model = LGBMRegressor(verbosity=-1)
    else:
        raise ValueError("Change it. ")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


def bayesian_hyperparameter_optimization(X_train, y_train, model_type):
    if model_type == 'lightgbm':
        model = LGBMRegressor(verbosity=-1)
        search_spaces = {
            'n_estimators': (30, 300),
            'learning_rate': (0.01, 0.2, 'log-uniform'),
            'num_leaves': (10, 100),
            'max_depth': (-1, 15),
        }
        
    elif model_type == 'xgboost':
        model = XGBRegressor()
        search_spaces = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'max_depth': (3, 15),
            'min_child_weight': (1, 10)
        }
        
    else:
        raise ValueError("Change it.")
    
    opt = BayesSearchCV(model, search_spaces, n_iter=32, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,random_state=42)
    opt.fit(X_train, y_train)

    return opt.best_estimator_, opt.best_params_


def change_outlier(df):
    Q1 = df['price'].quantile(0.01)
    Q3 = df['price'].quantile(0.99)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    df.loc[df['price'] < lower_bound, 'price'] = lower_bound
    df.loc[df['price'] > upper_bound, 'price'] = upper_bound
    
    return df


def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return rmse


def all_processing(df, bugatti_avg_hp_external=1300):
    df = df.drop(columns=['id', 'model', 'clean_title'])

    df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)

    df = accident_col(df)

    df = transmission_col(df)

    df = ext_and_int_col(df)

    df = fuel_type_col(df)

    df = engine_col(df)

    df = fill_na(df, bugatti_avg_hp_external)

    df= one_hot_encoding(df)
    df= age_and_usage(df)
    #df=change_outlier(df)
    df = df.drop(columns=['fuel', 'engine', 'accident'], )

    return df


df_train=all_processing(df_train)


model, X_train, X_test, y_train, y_test, y_pred = model_create(df_train, target='price', model_type='lightgbm')


best_model, best_params = bayesian_hyperparameter_optimization(X_train, y_train, model_type='lightgbm')
print("The best model:", best_model)
print("The best parameters:", best_params)


y_pred_best = best_model.predict(X_test)


calculate_rmse(y_test,y_pred_best)


df2_test_processed = all_processing(df2_test)


df2_test_processed['brand_POLESTAR'] = False
df2_test_processed['brand_SMART'] = False


train_columns = df_train.columns.tolist()
df2_test_processed = df2_test_processed.reindex(columns=train_columns, fill_value=False)


X_test = df2_test_processed.drop(columns=['price'], errors='ignore')


y_pred = best_model.predict(X_test)


results = pd.DataFrame({
    'price': y_pred
})


df2_test_processed['id'] = range(188533, 188533 + len(df2_test_processed))


df_results = pd.DataFrame({
    'id': df2_test_processed['id'],
    'price': y_pred
})


df_results.to_csv('c:\\Users\\PC\\Downloads\\SubmissionPrediction.csv', index=False)

