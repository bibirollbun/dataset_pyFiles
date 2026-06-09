import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import VotingRegressor
from sklearnex import patch_sklearn
from joblib import Parallel, delayed
import os, warnings, tqdm

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/directional-forecasting-cryptocurrencies/train.csv')
test = pd.read_csv('/kaggle/input/directional-forecasting-cryptocurrencies/test.csv')


train.head()


train.isna().any()


test.isna().any()


train.nunique()


train['target'].value_counts().plot(kind='bar')


def plot_correlation(df, cols): #Plot feature correlation
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    sns.heatmap(df[cols].corr(), vmin=0, vmax=1, annot=True, cmap='PiYG', ax=ax)
    plt.show()


plot_correlation(train, cols=['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 
                              'taker_buy_quote_volume', 'target'])


def plot_distribution(df, columns):
    ncols = 4
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14 * ncols, 12 * nrows))

    for ax, col in zip(axes.flat ,columns):
        ax.hist(df[col], bins=20, label=col)
        ax.set_title(f"{col} data distribution", fontweight='bold', fontsize=34)
        ax.legend(loc='best', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=34)

    for j in range(len(columns), len(axes.flat)):
        fig.delaxes(axes.flat[j])

    plt.tight_layout(pad=3.0)
    plt.show()


plot_distribution(train, columns=['open', 'high', 'low', 'close', 'volume','quote_asset_volume', 
                                  'number_of_trades','taker_buy_base_volume','taker_buy_quote_volume'])


def split_data(df):
    train_size = int(len(df) * 0.75)
    test_size = int(len(df) * 0.2)
    return df[:train_size], df[train_size:train_size+test_size], df[train_size+test_size:]


class TemporalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_temporal_(self, df):
        df = df.copy()
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('date_time').sort_index(ascending=True)
        df['hour'] = df.index.hour
        df['min'] = df.index.minute
        df['1_hour_change'] = df['close'] - df['open'].shift(freq='30min')
        df['volatility_change'] = df['high'].rolling('30min', min_periods=30).max() - df['low']
        return df.dropna()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.get_temporal_(X)

class RollingHistory(BaseEstimator, TransformerMixin):
    def __init__(self, window, cols, test=None):
        self.cols = cols
        self.window = window
        self.test = test

    def rolling_window_(self, data, window, cols, target=True):
        df = data[cols].values 
        size = df.shape[0] - window + 1
        emb = df.shape[1]
        
        if size <= 0:
            return np.empty((0, len(cols) * window))
    
        inputs = np.lib.stride_tricks.sliding_window_view(df, 
                                                          (window, emb), 
                                                          axis=(0, 1)).reshape(size, window * emb)
        if target:
            targets = data['target'].values[window-1:]
            return inputs, targets
            
        return inputs
    
    
    def fit(self, X, y=None):
        if self.test is not None and not self.test.empty:
            self.test = TemporalFeatures().fit_transform(self.test)
        return self
        
    def transform(self, X):
        X_sorted = X.sort_values(by='date_time')  # Sort once globally
        if self.test is not None:
            df = pd.concat([X_sorted, self.test], axis=0, join='inner', ignore_index=False)
            windowed_data = self.rolling_window_(df, self.window, self.cols, target=False)
            return windowed_data
            
        windowed_data, targets = self.rolling_window_(X_sorted, self.window, self.cols, target=True)
        return windowed_data, targets


def make_data_pipeline(columns, window, test=None): #Construct Pipeline
    pipe =  Pipeline([
        ('temporal_inputs', TemporalFeatures()),
        ('sliding_window', RollingHistory(window=window, cols=columns, test=test))
    ])
    return pipe


cols = ['open', 'close', 'volume', 'number_of_trades', 'hour', 'min', '1_hour_change', 'volatility_change']
data_pipeline = make_data_pipeline(columns=cols, window=5)


windowed_data, targets = data_pipeline.fit_transform(train)


def make_lgb(params): # Construct pipeline
    pipe = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 10))),
        ('model', lgb.LGBMClassifier(**params))
    ])
    
    return pipe


def evaluate_model(train, params, window, cols, n_splits=5, shuffle=True, random_state=11):
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    data_pipeline = make_data_pipeline(columns=cols, window=window)
    windowed_data, targets = data_pipeline.fit_transform(train)
    train_data, test_data, _ =  split_data(np.hstack((windowed_data, targets.reshape(-1,1))))

    cv_scores = []
    test_scores = []
    for train_index, test_index in tqdm.tqdm(kfold.split(train_data), desc="Evaluation model...."):
        X_train, X_test, y_train, y_test = train_data[train_index, :-1], train_data[test_index, :-1], train_data[train_index, -1], train_data[test_index, -1]
        model = make_lgb(params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        test_preds = model.predict(test_data[:, :-1])
        cv_scores.append(f1_score(y_test, preds))
        test_scores.append(f1_score(test_data[:, -1], test_preds))

    avg_cv_score = np.mean(cv_scores)
    avg_test_score = np.mean(test_scores)
    input_shape = windowed_data.shape[-1]
    print(f"(Features: {cols}, Window: {window}) - Validation Score: {avg_test_score:.4f} - CV Score: {avg_cv_score:.4f}\nInput Shape: {input_shape}")

    return cv_scores, test_scores


params = {"boosting_type": 'gbdt',"num_leaves": 77, "max_depth": 7, "colsample_bytree": 0.45, "learning_rate": 0.45,
          'min_child_samples': 20,'min_split_gain': 0.45, "n_estimators": 200,"verbose": -1, "objective": "binary",
          'random_state':11}
cols = ['1_hour_change','volatility_change', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'hour', 'min'] #volatility_change

cv_scores, test_scores = evaluate_model(train, params=params, window=18, cols=cols, n_splits=5, shuffle=True, random_state=11)


def get_temporal_(df):
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('date_time').sort_index(ascending=True)
    df['hour'] = df.index.hour
    df['min'] = df.index.minute
    df['1_hour_change'] = df['close'] - df['open'].shift(freq='30min')
    df['volatility_change'] = df['high'].rolling('30min', min_periods=30).max() - df['low']
    return df.dropna()


updated_data = get_temporal_(train)


plot_distribution(updated_data, columns=['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 
                                         'taker_buy_base_volume', 'taker_buy_quote_volume', 'volatility_change', '1_hour_change'])


plot_correlation(updated_data, cols=['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 
                                     'taker_buy_base_volume', 'taker_buy_quote_volume', 'hour', 'min', 'volatility_change', 
                                     '1_hour_change','target'])


def generate_predictions(train, test, params, window, cols, n_splits=5, shuffle=True, random_state=11):
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    train_pipeline = make_data_pipeline(columns=cols, window=window)
    test_pipeline = make_data_pipeline(columns=cols, window=window, test=test)
    windowed_data, targets = train_pipeline.fit_transform(train)
    test_data = test_pipeline.fit_transform(train)[-len(test):]
    assert len(test) == len(test_data)
    
    all_predictions = []
    for train_index, _ in tqdm.tqdm(kfold.split(windowed_data), desc="generating predictions...."):
        X_train, y_train = windowed_data[train_index], targets[train_index]
        model = make_lgb(params)
        model.fit(X_train, y_train)
        
        preds = model.predict(test_data)
        all_predictions.append(preds)
    all_predictions = np.array(all_predictions)
    
    return (np.mean(all_predictions, axis=0) > 0.5).astype(int)


params = {"boosting_type": 'gbdt',"num_leaves": 77, "max_depth": 7, "colsample_bytree": 0.45, "learning_rate": 0.45,
          'min_child_samples': 20,'min_split_gain': 0.45, "n_estimators": 200,"verbose": -1, "objective": "binary",
          'random_state':11}
cols = ['1_hour_change','volatility_change', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'hour', 'min'] #volatility_change

predictions = generate_predictions(train=train, test=test, params=params, window=18, cols=cols, n_splits=5, shuffle=True, random_state=11)


submission = pd.DataFrame(predictions, columns=['target']).reset_index(names='row_id')


submission.to_csv('submission.csv', index=False)




