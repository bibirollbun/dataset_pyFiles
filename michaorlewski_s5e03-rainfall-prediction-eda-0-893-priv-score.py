import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

train_data.head()


fig, axes = plt.subplots(1, 2)
fig.set_size_inches(12, 6)

axes[0].set_title('Before')
sns.lineplot(x=train_data.index, y=train_data.day, ax=axes[0])

# adjust the day based on id
train_data['day'] = (train_data.index) % 365 + 1
test_data['day'] = (test_data.index) % 365 + 1

axes[1].set_title('After')
sns.lineplot(x=train_data.index, y=train_data.day, ax=axes[1])


train_data.info()


train_data.describe()


pd.Series(train_data.isna().sum(), name='missing values count')


plt.figure(figsize=(12, 8))
for i, col in enumerate(train_data.columns):
    plt.subplot(4, 3, i+1)
    sns.histplot(data=train_data, x=col)

plt.tight_layout()
plt.show()


train_data['rainfall'].value_counts()


train_data.columns


plt.figure(figsize=(12, 8))
for i, col in enumerate(train_data.drop(['day', 'rainfall'], axis=1).columns):
    plt.subplot(5, 2, i+1)
    sns.scatterplot(data=train_data, x=train_data.index, y=col, hue='rainfall', size=1)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True)


train_data = train_data.rename(columns={'temparature': 'temperature'})
test_data = test_data.rename(columns={'temparature': 'temperature'})

# there is only 1 missing value across the entire dataset in the winddirection column
test_data['winddirection'] = test_data['winddirection'].ffill()

# convert winddirection to a categorical feature
bins = [-22.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

train_data['winddirection_cat'] = pd.cut(train_data['winddirection'], bins=bins, labels=labels,
                                         include_lowest=True, ordered=False).astype('object')
test_data['winddirection_cat'] = pd.cut(test_data['winddirection'], bins=bins, labels=labels,
                                         include_lowest=True, ordered=False).astype('object')

train_data = train_data.drop('winddirection', axis=1)
test_data = test_data.drop('winddirection', axis=1)

train_data.head()


train_data['temprange'] = train_data['maxtemp'] - train_data['mintemp']
test_data['temprange'] = test_data['maxtemp'] - test_data['mintemp']

train_data = train_data.drop(['mintemp', 'maxtemp'], axis=1)
test_data = test_data.drop(['mintemp', 'maxtemp'], axis=1)


train_data['month'] = pd.cut(train_data['day'], bins=12, labels=[i for i in range(1, 13)])
test_data['month'] = pd.cut(test_data['day'], bins=12, labels=[i for i in range(1, 13)])

train_data['year'] = (train_data.index + 1) // 365 + 1
test_data['year'] = (test_data.index + 1) // 365 + 1


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

rainfall_per_month = train_data.groupby('month')['rainfall'].sum()
axes[0].set_title('Number of rainfalls per month')
axes[0].set_ylabel('rainfalls')
sns.barplot(x=rainfall_per_month.index, y=rainfall_per_month.values, ax=axes[0])

rainfall_per_year = train_data.groupby('year')['rainfall'].sum()
axes[1].set_title('Number of rainfalls per year')
axes[1].set_ylabel('rainfalls')
sns.barplot(x=rainfall_per_year.index, y=rainfall_per_year.values, ax=axes[1])


# drop year column as it's not useful
train_data = train_data.drop('year', axis=1)
test_data = test_data.drop('year', axis=1)


from statsmodels.graphics.tsaplots import plot_pacf

features = ['pressure', 'temperature', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'windspeed']

fig, axes = plt.subplots(4, 2, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(features):
    plot_pacf(train_data[feature], lags=7, ax=axes[i])
    axes[i].set_title(f'Partial Autocorrelation - {feature}')

plt.tight_layout()
plt.show()


# merge train and test sets to properly calculate shift values
test_index = test_data.index
test_data['rainfall'] = np.nan
full_data = pd.concat([train_data, test_data])

lagged =  ['pressure', 'temperature', 'dewpoint']

for feature in lagged:
    for i in range(3):
        full_data[f'{feature}_{i}'] = full_data[feature].shift(i+1)


features = ['pressure', 'temperature', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
window_sizes = [7, 30]

moving_means = []
for feature in features:
    for window_size in window_sizes:
        moving_means.append(f'{feature}_mean_{window_size}')
        full_data[f'{feature}_mean_{window_size}'] = full_data[feature].rolling(
            window=window_size
        ).mean()


sns.scatterplot(x=full_data.index, y=full_data['pressure_mean_7'])


def month_to_season(m):
    if m in [12, 1, 2]:
        return 'winter'
    elif m in [3, 4, 5]:
        return 'spring'
    elif m in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


full_data['season'] = full_data['month'].map(month_to_season)


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# artificially create timeseries as a temporary index
start_date = '2020-01-01'
date_range = pd.date_range(start=start_date, periods=len(full_data), freq='D')

full_data_og_index = full_data.index
full_data.index = date_range

fourier = CalendarFourier(freq="A", order=4)

dp = DeterministicProcess(
    index=full_data.index,
    constant=False,
    order=2,
    seasonal=False,
    additional_terms=[fourier],
    drop=True,
)

full_data = pd.concat([full_data, dp.in_sample()], axis=1)


# split data back into train and test sets
full_data.index = full_data_og_index

train_data = full_data[~full_data.index.isin(test_index)].dropna()
test_data = full_data[full_data.index.isin(test_index)].drop('rainfall', axis=1)


X = train_data.drop(['rainfall'], axis=1)
y = train_data['rainfall']

X.shape, y.shape

num_columns = X.select_dtypes(exclude=['object']).columns
cat_columns = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)
], remainder='passthrough')

tscv = TimeSeriesSplit()


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

param_grid = {'model__C': [0.01, 0.1, 1],
              'model__penalty': ['elasticnet'],
              'model__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
              'model__solver': ['liblinear', 'saga'],
              'model__max_iter': [250, 500, 750, 1000], 
              'model__tol': [1e-5, 1e-4, 1e-3]}
clf = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=tscv, n_jobs=-1, verbose=True)
clf.fit(X, y)

print(clf.best_params_)
best_logreg = clf.best_estimator_['model']

pipeline_logreg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_logreg)
])

scores = cross_validate(pipeline_logreg, X, y, scoring='roc_auc', cv=tscv, return_train_score=True, n_jobs=-1)
print(f'Mean ROC AUC score (train, test): {scores["train_score"].mean()}, {scores["test_score"].mean()}')


pipeline_logreg.fit(X, y)
predictions_logreg = pipeline_logreg.predict_proba(test_data)[:, 1]

submission = pd.DataFrame({'id': test_data.index, 'rainfall': predictions_logreg})
submission.to_csv('/kaggle/working/submission_logreg.csv', index=None)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


X = train_data.drop(['rainfall'], axis=1)
y = train_data['rainfall']

num_columns = X.select_dtypes(exclude=['object']).columns
cat_columns = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)
], remainder='passthrough')

X_train = preprocessor.fit_transform(X)
X_test = preprocessor.transform(test_data)
y_train = y.to_numpy()
X_train.shape, y_train.shape, X_test.shape


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train.dtype, y_train.dtype, X_test.dtype


class RainfallDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RainfallDataset(X_train, y_train)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss

def validate_model(model, dataloader, loss_fn, set_name):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    total_loss /= num_batches
    correct /= size
    acc = 100 * correct
    print(f"Validation ({set_name}): Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>4f}")
    return acc, total_loss



loss_fn = nn.CrossEntropyLoss()
n_splits = 4
tscv = TimeSeriesSplit(n_splits=n_splits)
num_epochs = 20
learning_rate = 0.01
batch_size = 1

train_acc_scores = []
train_losses = []
valid_acc_scores = []
valid_losses = []

for i, (train_idx, valid_idx) in enumerate(tscv.split(X_train)):
    print(f'Split {i+1}')
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    
    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size)

    for epoch in range(num_epochs):
        epoch_loss = train_model(model, train_loader, loss_fn, optimizer)
        print(f'Epoch: {epoch+1}/{num_epochs}, avg loss: {epoch_loss/len(train_loader):.4f}')

        if (epoch+1) % 5 == 0 or epoch == num_epochs-1:
            train_acc, train_loss = validate_model(model, train_loader, loss_fn, 'train')
            valid_acc, valid_loss = validate_model(model, valid_loader, loss_fn, 'valid')

            if epoch == num_epochs-1:
                train_acc_scores.append(train_acc)
                train_losses.append(train_loss)
                valid_acc_scores.append(valid_acc)
                valid_losses.append(valid_loss)

        scheduler.step()    
    print('-------------------------------------')


print(f'Mean accuracy (train, valid): {np.mean(train_acc_scores):.2f}, {np.mean(valid_acc_scores):.2f}')
print(f'Mean loss (train, valid): {np.mean(train_losses):.4f}, {np.mean(valid_losses):.4f}')


model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

for epoch in range(num_epochs):
    train_model(model, train_loader, loss_fn, optimizer)
    scheduler.step()


predictions_dl = []
model.eval()
with torch.no_grad():
    for X in X_test:
        output = model(X)
        predictions_dl.append(output[1].item())

predictions_dl = np.array(predictions_dl)


submission = pd.DataFrame({'id': test_data.index, 'rainfall': predictions_dl})
submission.to_csv('/kaggle/working/submission_dl.csv', index=None)


averaged_predictions = (predictions_logreg + predictions_dl) / 2
submission = pd.DataFrame({'id': test_data.index, 'rainfall': averaged_predictions})
submission.to_csv('/kaggle/working/submission.csv', index=None)

