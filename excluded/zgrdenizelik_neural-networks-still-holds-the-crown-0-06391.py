# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#!pip install git+https://github.com/wkqian06/pykan.git -q
#from kan import KAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, cross_val_score
import requests
import torch
import torch.nn as nn #!
import torch.optim as optim #!
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


'''
# Create model
model = KAN(width=[40, 1], grid=200, k=3, noise_scale=0.1, 
            sp_trainable=False, sb_trainable=False, base_fun='zero')
model.to(device)
'''


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(40, 20)  
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  
        return x


model = NeuralNet()
model.to(device)


train_X.shape, train_label.shape


train_X


skip_range = 10000
num_steps = 100

loss_fn = nn.MSELoss()
optimizer = optim.LBFGS(model.parameters())

model_preds = []


train_input = torch.tensor(train_X, dtype=torch.float32).to(device)
train_label = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1).to(device)  
val_input = torch.tensor(val_X, dtype=torch.float32).to(device)
val_label = torch.tensor(val_y, dtype=torch.float32).unsqueeze(-1).to(device)  
test_input = torch.tensor(test_preprocessed.toarray(), dtype=torch.float32).to(device)



for step in range(num_steps):
    def closure():
        optimizer.zero_grad()
        predictions = model(train_input)
        loss = loss_fn(predictions, train_label)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    
    if (step + 1) % skip_range == 0:
        print(f'Step [{step + 1}/{num_steps}]')


model.eval()
with torch.no_grad():
    
    train_predictions = model(train_input)
    train_loss = loss_fn(train_predictions, train_label)
    
   
    val_predictions = model(val_input)
    val_loss = loss_fn(val_predictions, val_label)
    
    
    test_predictions = model(test_input)
    
    print(f'\nFinal Training Loss: {train_loss.item():.4f}')
    print(f'Final Validation Loss: {val_loss.item():.4f}')
    
    
    train_predictions = train_predictions.cpu().numpy()
    val_predictions = val_predictions.cpu().numpy()
    test_predictions = test_predictions.cpu().numpy()

model.train()


def mean_absolute_percentage_error_c(y_true, y_pred):
    """
    Calculate MAPE,
    Handles division by zero by excluding those cases
    """
    y_true = y_true.flatten()  
    y_pred = y_pred.flatten()  
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


train_mape = mean_absolute_percentage_error(
    train_label.detach().cpu().numpy(),
    train_predictions
)
print(f'Training MAPE: {train_mape:.2f}%')


val_mape = mean_absolute_percentage_error(
    val_label.detach().cpu().numpy(),
    val_predictions
)
print(f'Validation MAPE: {val_mape:.2f}%')


train_actual = train_y  
val_actual = val_y      

def plot_preds_vs_actuals(preds, actuals, title):
    preds = preds.flatten()
    actuals = actuals.flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, preds, alpha=0.5, label="Predictions vs Actuals", s=10)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
             color='red', linestyle='--', label="Perfect Fit")
    plt.xlabel("Actuals")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot training predictions
plot_preds_vs_actuals(train_predictions, train_y, "Training Predictions vs Actuals")

# Plot validation predictions
plot_preds_vs_actuals(val_predictions, val_y, "Validation Predictions vs Actuals")


'''
def testing_model(X, y):
    preds = model(X)
    return mean_absolute_percentage_error(preds.detach().cpu().numpy(), y.cpu())
'''


#model.prune()


#model.plot(scale = 4)


# testing on validation data
#testing_model(val_input, val_label)


# testing on validation data
#testing = model(test_input)


# Ensure test_predictions is a NumPy array (no need to detach or convert again)
test['num_sold_log'] = test_predictions
test['num_sold'] = np.expm1(test['num_sold_log']).clip(lower=0).round()

# Create submission DataFrame
submission = test[['id', 'num_sold']]

# Save to CSV
submission.to_csv("submission.csv", index=False)


submission




