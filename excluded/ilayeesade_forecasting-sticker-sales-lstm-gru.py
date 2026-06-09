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


%%capture
!pip install tradingeconomics


from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, cross_val_score, train_test_split
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tradingeconomics as te
import scipy.stats as stats
from scipy.special import inv_boxcox
from sklearn.impute import SimpleImputer
from tqdm import tqdm


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

train.head()


# Function to fetch GDP per capita - BLOCK IS COPIED FROM : "Neural Networks Still holds the Crown [0.06391]"
# I believe CCI would be more indicative, might add it.
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


train.head()


train['country'].nunique(), train['store'].nunique(), train['product'].nunique()


train['country'].unique()


train['date'].dt.dayofweek.unique()


def add_date_features(df):
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekend'] = (train['dayofweek'] == 0) | (train['dayofweek'] == 6)
    df['day'] = df['date'].dt.day
    return df


train = add_date_features(train)
test = add_date_features(test)


train.columns[train.isnull().any()]


train.columns


train['mean_num_sold'] = train.dropna().groupby(
    ['country', 'store', 'product']
)['num_sold'].transform('mean')


train['num_sold'] = train['num_sold'].fillna(train['mean_num_sold'])
train['num_sold'] = train['num_sold'].fillna(train['num_sold'].mean())


train['num_sold'].value_counts(dropna=False)


stats.skew(train['num_sold'])


train['num_sold_log'] = np.log1p(train['num_sold'])


stats.skew(train['num_sold_log'])


train['num_sold_box'], lambda_val = stats.boxcox(train['num_sold'])


stats.skew(train['num_sold_box'])


train.columns


categorical_cols = ['country', 'store', 'product', 'month', 'weekend']
numerical_cols = ['gdp', 'day', 'dayofweek', 'year']


to_pipeline = pd.concat([train[categorical_cols + numerical_cols], test])


numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(sparse_output=True, drop='first'))
])

# Combine both pipelines into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

transformed_data = preprocessor.fit_transform(to_pipeline)


num_columns = numerical_cols
cat_columns = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)

num_columns = list(num_columns)
cat_columns = list(cat_columns)
all_columns = num_columns + cat_columns

if hasattr(transformed_data, "toarray"):
    transformed_data = transformed_data.toarray()
final_df = pd.DataFrame(transformed_data, columns=all_columns)


transformed_data.shape


final_df.head()


p_train = final_df.loc[:train.shape[0]-1]
p_test = final_df.loc[train.shape[0]:]


p_train.shape, p_test.shape


scaler = StandardScaler()


p_train['num_sold_box_scaled'] = scaler.fit_transform(np.array(train['num_sold_box']).reshape(-1, 1))


p_train.head()


p_train.columns


X = torch.tensor(np.array(p_train.drop(['num_sold_box_scaled'], axis=1)))


y = torch.tensor(np.array(p_train['num_sold_box_scaled']))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69, shuffle=True)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(Model, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.apply(init_weights)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, input_size)
        out, h_n = self.rnn(x)
        output = self.fc1(out[-1])
        output = self.fc2(self.act(output))
        return output


24, 72


X_train.shape


184104 / 8


X_train.shape


arr = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])


arr.view(-1, 2, 4)


ar = torch.tensor([1,2,3,4,5,6,7,8])


ar.view(-1, 4)[:, -1]


def create_sliding_window(X, y, seq_len):
    n, input_size = X.shape
    seqX = torch.stack([X[i:i + seq_len] for i in range(n - seq_len + 1)])
    seqY = torch.stack([y[i + seq_len - 1] for i in range(n - seq_len + 1)])
    return seqX.permute(1, 0, 2).to(torch.float32), seqY.to(torch.float32)


input_size = 27
hidden_size = 64
output_size = 1
batch_size = 128
seq_len = 30
epochs = 25
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-1)

seqX, seqY = create_sliding_window(X_train, y_train, seq_len)
valX, valY = create_sliding_window(X_test, y_test, seq_len)

vl_max = 20
for epoch in range(epochs):
    
    epoch_loss = 0
    num_batches = seqX.size(1) // batch_size
    val_nb_batches = valX.size(1) // batch_size

    progress_bar = tqdm(np.random.default_rng().permutation(num_batches-1), desc=f"Epoch [{epoch+1}/{epochs}]")
    lossi = 0
    model.train()
    for batch_idx in progress_bar:
        
        sampled_X = seqX[:, batch_idx:(batch_idx+batch_size)].to(device)  # Shape: (seq_len, batch_size, input_size)
        sampled_Y = seqY[batch_idx:(batch_idx+batch_size)].view(-1, 1).to(device)       # Shape: (batch_size, 1)

        optimizer.zero_grad()
        output = model(sampled_X)
        loss = criterion(output, sampled_Y) 
        epoch_loss += loss.item()  
        loss.backward()  
        optimizer.step()
        # scheduler.step()

        lossi = 0.9 * lossi + 0.1 * loss.item()
        progress_bar.set_postfix({'Batch Loss': lossi})

    progress_bar = tqdm(np.random.default_rng().permutation(val_nb_batches-1), desc=f"Epoch [{epoch+1}/{epochs}]")
    lossi = 0 
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx in progress_bar:
            sampled_X = valX[:, batch_idx:(batch_idx+batch_size)].to(device)  # Shape: (seq_len, batch_size, input_size)
            sampled_Y = valY[batch_idx:(batch_idx+batch_size)].view(-1, 1).to(device)       # Shape: (batch_size, 1)
            
            output = model(sampled_X)
            loss = criterion(output, sampled_Y)  
            val_loss += loss.item()
    
            lossi = 0.9 * lossi + 0.1 * loss.item()
            val_loss += loss / (val_nb_batches-1)
            progress_bar.set_postfix({'Val Loss': lossi})
    
        rmse_v = val_loss ** (1/2)
        if(rmse_v < vl_max):
            vl_max = val_loss
            torch.save(model, "model.pth")
    
    # Calculate and print the average loss for the epoch
    scheduler.step(rmse_v)
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_epoch_loss:.4f}, RMSE Val: {rmse_v:.4f}")


X_sub = torch.cat([torch.tensor(np.array(p_train))[-29:, :-1], torch.tensor(np.array(p_test))], axis=0)


p_test.shape


X_sub.shape


n, input_size = X_sub.shape
X_sub_seqs = torch.stack([X[i:i + seq_len] for i in range(n - seq_len + 1)])
Input = X_sub_seqs.permute(1, 0, 2).to(torch.float32)


with torch.no_grad():
    res = model.to('cpu')(Input)


res.shape


inv_boxcox(p_train['num_sold_box_scaled'], lambda_val)


re = scaler.inverse_transform(res)


r = inv_boxcox(re, lambda_val)


sub.iloc[:, 1] = sub.iloc[:, 1].astype(float)


sub.iloc[:, 1] = r


sub


sub.to_csv('submission.csv', index=False)

!head submission.csv


kaggle competitions submit -c playground-series-s5e1 -f submission.csv -m "Message"

