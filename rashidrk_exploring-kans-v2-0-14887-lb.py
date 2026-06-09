# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install git+https://github.com/wkqian06/pykan.git -q
from kan import KAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, cross_val_score
import requests
import torch
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/train-filled/TrainFilled.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

train.head()


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

# Apply to train and test datasets
train = add_gdp_feature(train)
test = add_gdp_feature(test)


combined = pd.concat([train[['year', 'gdp']], test[['year', 'gdp']]])
gdp_total = combined.groupby('year')['gdp'].sum().reset_index().rename(columns={'gdp': 'total_gdp'})

train = train.merge(gdp_total, on='year', how='left')
train['gdp_ratio'] = train['gdp'] / train['total_gdp']

test = test.merge(gdp_total, on='year', how='left')
test['gdp_ratio'] = test['gdp'] / test['total_gdp']

global_product_ratio = train.groupby('product')['num_sold'].mean() / train['num_sold'].mean()
global_store_ratio = train.groupby('store')['num_sold'].mean() / train['num_sold'].mean()
global_country_ratio = train.groupby('country')['num_sold'].mean() / train['num_sold'].mean()


overall_product_ratio = global_product_ratio.mean()
overall_store_ratio = global_store_ratio.mean()
overall_country_ratio = global_country_ratio.mean()

for df in [train, test]:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['month_store'] = df['month'].astype(str) + "_" + df['store']
    df['month_product'] = df['month'].astype(str) + "_" + df['product']
    df['year_normalized'] = df['year'] - 2010
    
    df['product_ratio'] = df['product'].map(global_product_ratio)
    df['store_ratio'] = df['store'].map(global_store_ratio)
    df['country_ratio'] = df['country'].map(global_country_ratio)
    
    df['product_ratio'].fillna(overall_product_ratio, inplace=True)
    df['store_ratio'].fillna(overall_store_ratio, inplace=True)
    df['country_ratio'].fillna(overall_country_ratio, inplace=True)


train['num_sold_log'] = np.log1p(train['num_sold'])



train.head()


X = train.drop(columns=['id', 'num_sold', 'num_sold_log', 'date'])
y = train['num_sold_log']

categorical_features = ['country', 'store', 'product', 'month', 'day_of_week', 'is_weekend']
numerical_features = ['gdp_ratio', 'year_normalized', 'product_ratio', 'store_ratio', 'country_ratio']


categorical_transformer = OneHotEncoder(handle_unknown='ignore')  
numerical_transformer = StandardScaler()  


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


test.head()


train_preprocessed = preprocessor.fit_transform(X)
test_preprocessed = preprocessor.transform(test.drop(columns = ["id", "date"]))


train_preprocessed.shape, test_preprocessed.shape


train_preprocessed = train_preprocessed.toarray()

val_count = int(train_preprocessed.shape[0] * 0.1)
train_X = train_preprocessed[: -val_count]
train_y = y.values[: -val_count]

val_X = train_preprocessed[-val_count :]
val_y = y.values[-val_count :]


# Converting data to Torch tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
train_input = torch.tensor(train_X, dtype=torch.float32).to(device)
train_label = torch.tensor(train_y, dtype=torch.float32).to(device)
val_input = torch.tensor(val_X, dtype=torch.float32).to(device)
val_label = torch.tensor(val_y, dtype=torch.float32).to(device)
test_input = torch.tensor(test_preprocessed.toarray(), dtype=torch.float32).to(device)


def train_mape():
    preds = model(dataset['train_input'])
    return mean_absolute_percentage_error(preds.detach().cpu().numpy(), dataset['train_label'].cpu())

def val_mape():
    preds = model(dataset['test_input'])
    return mean_absolute_percentage_error(preds.detach().cpu().numpy(), dataset['test_label'].cpu())


# Create model
model = KAN(width=[40, 1], grid=200, k=3, noise_scale=0.1, 
            sp_trainable=False, sb_trainable=False, base_fun='silu')
model.to(device)


train_X.shape, train_label.shape


def calculate_mape(true_values, pred_values):
    return mean_absolute_percentage_error(true_values, pred_values)


skip_range = 10000  # The model will learn in chunks of 10000 training datasets

train_model_preds = []
test_model_preds = []

# Preprocess labels
train_label = train_label.unsqueeze(-1)
val_label = val_label.unsqueeze(-1)

best_score = 100000

for i in range(0, len(train_X), skip_range):
    end_index = min(i + skip_range, len(train_X))

    dataset = {
        'train_input': train_input[i:end_index],
        'train_label': train_label[i:end_index],
        'test_input': val_input,
        'test_label': val_label
    }

    results = model.fit(
        dataset,
        metrics=(train_mape, val_mape),
        opt="LBFGS",
        steps=200,
        loss_fn=torch.nn.MSELoss(),
        update_grid=False
    )

    val_pred = model(val_input).detach().cpu().numpy()
    val_score = calculate_mape(val_label.detach().cpu().numpy(), val_pred)
    print(f"Chunk {i},VAL MAPE score {val_score}")
    
    if val_score < best_score:
        best_score = val_score
        test_preds = model(test_input)
        test_model_preds.append(test_preds.detach().cpu().numpy())



def testing_model(X, y):
    preds = model(X)
    return mean_absolute_percentage_error(preds.detach().cpu().numpy(), y.cpu())


model.prune()


model.plot(scale = 4)


# testing on validation data
testing_model(val_input, val_label)


total_preds = len(test_model_preds)
test_model_preds = np.array(test_model_preds).reshape(total_preds, 98550).T
test_model_preds.shape


avg_preds = test_model_preds[:, -2:].mean(axis = 1)
avg_preds


avg_preds.shape


test['num_sold_log'] = avg_preds
test['num_sold'] = np.expm1(test['num_sold_log']).clip(lower=0).round()

submission = test[['id', 'num_sold']]
submission.to_csv("submission.csv", index=False)


submission







