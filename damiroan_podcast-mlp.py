import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt


# Load Data
op = 1 # 0: Local, 1: Kaggle

if not op: # Local
    train_path = './data/train.csv'
    test_path = './data/test.csv'
    sub_path = './data/sample_submission.csv'
    save_path = './data/submission.csv'
else:  # Kaggle
    train_path = '/kaggle/input/playground-series-s5e4/train.csv'
    test_path = '/kaggle/input/playground-series-s5e4/test.csv'
    sub_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'    
    save_path = '/kaggle/working/submission.csv'
    

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd.read_csv(sub_path)


# Check NaN
print(df_train.isnull().sum())
print("=====================================")
print(df_test.isnull().sum())

df_train.head()


# Check the type of string data
comparion_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in comparion_cols:
    re = np.array_equal(
        np.sort(df_train[col].unique()), 
        np.sort(df_test[col].unique())
    )
    print(f"{col} : {re}")


# Drop Feature 
drop_cols = ['Episode_Title']
df_train = df_train.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)

# Fill NaN
def fill_NaN(df, target_col, group_col, method):
    df[target_col] = df[target_col].fillna(df.groupby(group_col)[target_col].transform(method))

nan_list = [
    ['Episode_Length_minutes','Podcast_Name','mean'],
    ['Number_of_Ads', 'Podcast_Name','mean'], 
    ['Guest_Popularity_percentage', 'Podcast_Name','mean']
]

for l in nan_list:
    fill_NaN(df_train, l[0], l[1], l[2])
    fill_NaN(df_test, l[0], l[1], l[2])


print(df_train.isnull().sum())
print("=====================================")
print(df_test.isnull().sum())
df_train.head()


# Encoding - One-hot
onehot_cols = ['Podcast_Name', 'Genre', 'Publication_Time', 'Publication_Day', 'Episode_Sentiment']
df_train = pd.get_dummies(df_train, columns=onehot_cols, drop_first=False)
df_test = pd.get_dummies(df_test, columns=onehot_cols, drop_first=False)
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

print(df_train.columns.size == df_test.columns.size)
df_train


# split Train and Val
X = df_train.drop(columns=['Listening_Time_minutes','id']).to_numpy()
y = df_train['Listening_Time_minutes'].to_numpy()
X_pred = df_test.drop(columns=['Listening_Time_minutes','id']).to_numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# Check Normalize
original_data = X_train.copy()

qt_uniform = QuantileTransformer(n_quantiles=10000, output_distribution='uniform')
qt_uniform_data = qt_uniform.fit_transform(X_train)

qt_normal = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
qt_normal_data = qt_normal.fit_transform(X_train)

pt_yeo = PowerTransformer(method='yeo-johnson', standardize=True)
pt_yeo_data = pt_yeo.fit_transform(X_train)

# visualization
fig, axes = plt.subplots(4, 3, figsize=(15, 10))

datasets = [original_data, qt_uniform_data, qt_normal_data, pt_yeo_data]
titles = ['Original', 'Quantile (uniform)', 'Quantile (normal)', 'Power (yeo-johnson)']

for row in range(4):
    for col in range(3):    
        ax = axes[row, col]
        ax.hist(datasets[row][:, col], bins=1000, color='steelblue')
        ax.set_xlim(datasets[row][:, col].min(), datasets[row][:, col].max())
        if row == 0:
            ax.set_title(f'Feature {col}')
        if col == 0:
            ax.set_ylabel(titles[row])

plt.tight_layout()
plt.show()


# Normalize
X_scaler = QuantileTransformer(
    n_quantiles=10000,
    subsample=10000,
    output_distribution='uniform',
)
y_scaler = QuantileTransformer(
    n_quantiles=10000,
    subsample=10000,
    output_distribution='uniform',
)

X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_val_scaled = X_scaler.transform(X_val)
X_pred_scaled = X_scaler.transform(X_pred)

y_scaler.fit(y_train.reshape(-1, 1))
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

# Check Normalize
data_list = [
    X_train_scaled[:, 0],
    X_val_scaled[:, 0],
    X_pred_scaled[:, 0],
    y_train_scaled,
    y_val_scaled
]

titles = ['X_train (col 0)', 'X_val (col 0)', 'X_pred (col 0)', 'y_train', 'y_val']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    axes[i].hist(data_list[i], bins=1000, color='steelblue')
    axes[i].set_title(titles[i])
    axes[i].set_xlim(min(data_list[i]), max(data_list[i]))  # x축 범위 고정 (optional)

plt.tight_layout()
plt.show()


mlp = MLPRegressor(
    solver='adam',
    activation='tanh',
    hidden_layer_sizes=[100, 100, 100],      
    max_iter=5000,                     
    early_stopping=True,    
    tol=1e-4,
    n_iter_no_change=10,
    validation_fraction=0.1,
    learning_rate='adaptive',         
    learning_rate_init=0.001,          
    alpha=0.001,
    random_state=0                      
)
mlp.fit(X_train_scaled, y_train_scaled)

y_train_pred = y_scaler.inverse_transform(mlp.predict(X_train_scaled).reshape(-1, 1)).ravel()
y_val_pred = y_scaler.inverse_transform(mlp.predict(X_val_scaled).reshape(-1, 1)).ravel()
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
print(np.sqrt(mean_squared_error(y_val, y_val_pred)))


X_scaler.fit(X)
X_scaled = X_scaler.transform(X)
X_pred_scaled = X_scaler.transform(X_pred)

y_scaler.fit(y.reshape(-1, 1))
y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

mlp.fit(X_scaled, y_scaled)
y_pred = y_scaler.inverse_transform(mlp.predict(X_pred_scaled).reshape(-1, 1)).ravel()
df_sub['Listening_Time_minutes'] = y_pred
df_sub.to_csv(save_path, index=False)




