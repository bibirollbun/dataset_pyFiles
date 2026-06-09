import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import supplemental_english as sup


# Calculate SMAPE
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Convert REGION_CODE mapping
def convert_dict(input_dict):
    new_dict = {}
    for key, value_list in input_dict.items():
        for item in value_list:
            new_dict[item] = key
    return new_dict

reg_codes = convert_dict(sup.REGION_CODES)

# Convert GOV_INFO mapping
def convert_dict2(input_dict):
    df = pd.DataFrame()
    for key, value in input_dict.items():
        new_row = pd.DataFrame({'c': [key[0]], 'rg': [key[2]],
                                'rl': [int(key[1][0])], 'rh': [int(key[1][1])],
                                'gov_name': [value[0]],
                                'gov_buy': [bool(value[1])],
                                'gov_adv': [bool(value[2])],
                                'buy_sig': [int(value[3])]})
        df = pd.concat([df, new_row], ignore_index=True)
    df = df.drop_duplicates(subset=['c','rg'])
    return df

gc = convert_dict2(sup.GOVERNMENT_CODES)
gc['is_gov'] = True


def fe(data):
    data = data.drop(data.columns[0], axis=1)
    data['c'] = data['plate'].str[0] + data['plate'].str[4:6]
    data['h1'] = data['plate'].str[0]
    data['h2'] = data['plate'].str[4]
    data['h3'] = data['plate'].str[5]
    data['n'] = data['plate'].str[1:4].astype(int)
    data['rg'] = data['plate'].str[6:]
    data['loc'] = data['rg'].map(reg_codes)
    data = pd.merge(data, gc, on=['c', 'rg'], how='left')

    with pd.option_context("future.no_silent_downcasting", True):
        data['is_gov'] = data['is_gov'].fillna(False).infer_objects(copy=False)
        data['gov_buy'] = data['gov_buy'].fillna(False).infer_objects(copy=False)
        data['gov_adv'] = data['gov_adv'].fillna(False).infer_objects(copy=False)
        data['buy_sig'] = data['buy_sig'].fillna(0).infer_objects(copy=False)

    data.loc[(data['n'] < data['rl']) | (data['n'] > data['rh']),'is_gov'] = False
    data.loc[(data['n'] < data['rl']) | (data['n'] > data['rh']),'gov_buy'] = False
    data.loc[(data['n'] < data['rl']) | (data['n'] > data['rh']),'gov_adv'] = False
    data.loc[(data['n'] < data['rl']) | (data['n'] > data['rh']),'buy_sig'] = 0
    
    data['year'] = pd.to_datetime(data['date']).dt.year
    data['month'] = pd.to_datetime(data['date']).dt.month
    
    data = data[['c','n','loc','is_gov','gov_buy','gov_adv','buy_sig','year','price']]
    return data


data = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
data = fe(data)
data['price'] = data['price'].astype(int)
cat_lst = ["c","loc"]
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(data[cat_lst])
one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(cat_lst))
X = pd.concat([data.drop(cat_lst, axis=1).drop(['price'], axis=1), one_hot_df], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)


rf_model = RandomForestRegressor(n_estimators=50, min_samples_split=2, ccp_alpha=5, random_state=24)
rf_model.fit(X_train, y_train)
y_test_pred = rf_model.predict(X_test)
evl = smape(y_test, y_test_pred)
print("Model Performance: {}".format(evl))


df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
X = fe(df)
cat_lst = ["c","loc"]
one_hot_encoded = encoder.transform(X[cat_lst])
one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(cat_lst))
X = pd.concat([X.drop(cat_lst, axis=1).drop(['price'], axis=1), one_hot_df], axis=1)

df['price'] = rf_model.predict(X).astype(int)
df = df[['id', 'price']]
df.to_csv('/kaggle/working/output.csv', index=False)

