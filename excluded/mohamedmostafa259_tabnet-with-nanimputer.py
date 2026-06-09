# !pip uninstall -y numpy scikit-learn xgboost
# !pip install numpy==1.24.4 scikit-learn==1.1.3 xgboost==1.7.6 --force-reinstall
# !pip install verstack


!pip install verstack


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from verstack import NaNImputer


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_train.head()


df_train.info()


df_train.duplicated().sum()


df_train.drop(columns=['id'], inplace=True)


df_train.hist(bins=50, figsize=(10, 8))
plt.tight_layout()
plt.show()


for col in df_train.select_dtypes(include=np.number).columns:
	plt.hist(np.log1p(df_train[col]), bins=50)
	plt.title(f'log_{col}')
	plt.show()


df_train.describe()


df_train[df_train['Listening_Time_minutes'] > df_train['Episode_Length_minutes']]


# df_train = df_train[df_train['Episode_Length_minutes'] != 0]


df_train[
    (df_train['Host_Popularity_percentage'] > 100) |
    (df_train['Host_Popularity_percentage'] < 0) |
    (df_train['Guest_Popularity_percentage'] > 100) |
    (df_train['Guest_Popularity_percentage'] < 0)
]


df_train[
    (df_train['Number_of_Ads'] < 0) |
    (df_train['Listening_Time_minutes'] < 0)
]


df_train['Number_of_Ads'].value_counts(dropna=False)


df_train[df_train['Number_of_Ads'] > 10]


for col in df_train.select_dtypes(include=[object]):
	print('-'*30, col, '-'*30)
	print(df_train[col].value_counts(dropna=False))
	print('\n')


df_train.isna().sum()


df_train[df_train['Guest_Popularity_percentage'].isna()]


# pd.set_option('display.max_rows', None)
class FeatureEng(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		X = X.copy()

		print('FeatureEng-5', X.isna().sum().sum())
        
		X['Total_pop'] = np.where(
			X['Guest_Popularity_percentage'] == 0, 
			X['Host_Popularity_percentage'], 
			X['Host_Popularity_percentage'] * X['Guest_Popularity_percentage']
		)

		print('FeatureEng-4', X.isna().sum().sum())
        
		X['pure_len'] = np.where(
			X['Number_of_Ads'] == 0.00,
			X['Episode_Length_minutes'],  # if Number_of_Ads == 0 → just return episode_len
			X['Episode_Length_minutes'] / X['Number_of_Ads']  # else → do the division
		)
        
		X['Have_Ads'] = X['Number_of_Ads'] > 0
		X['Ads_One'] = X['Number_of_Ads'] == 1
		X['Ads_Two'] = X['Number_of_Ads'] == 2
		X['Ads_Three'] = X['Number_of_Ads'] == 3
		X['Ads_Four'] = X['Number_of_Ads'] == 4
		X['Ads_atleast_One'] = X['Number_of_Ads'] >= 1
		X['Ads_atleast_Two'] = X['Number_of_Ads'] >= 2
		X['Ads_atleast_Three'] = X['Number_of_Ads'] >= 3

		print('FeatureEng-3', X.isna().sum().sum())
        
		# assuming that each Ad takes around one minute on average
		X['Ad_density'] = (X['Number_of_Ads'] * 1.00) / X['Episode_Length_minutes']
		X['Ad_density'] = X['Ad_density'].fillna(X['Ad_density'].median())
		X['content_minutes'] = X['Episode_Length_minutes'] * (1 - X['Ad_density'])  

		print('FeatureEng-2', X.isna().sum().sum())
        
		X['is_weekend'] = (X['Publication_Day'] == 'Sunday') | (X['Publication_Day'] == 'Saturday') 

		print('FeatureEng-1', X.isna().sum().sum())

		hour_mapping = {
			'Morning': 9,
			'Afternoon': 14,  # 2 PM
			'Evening': 19,    # 7 PM
			'Night': 23       # 11 PM
		}
		ampm_mapping = {
			'Morning': 'AM',
			'Afternoon': 'PM',
			'Evening': 'PM',
			'Night': 'PM'
		}
		
		print('FeatureEng0', X.isna().sum().sum())
        
		X['Publication_Hour'] = X['Publication_Time'].map(hour_mapping)
		X['Publication_AMPM'] = X['Publication_Time'].map(ampm_mapping)
		X['title_len'] = X['Episode_Title'].apply(len)
		X['episode_num'] = X['Episode_Title'].apply(lambda title: int(title[8:]))

		print('FeatureEng1', X.isna().sum().sum())
        
        # I think bin_size=4 is the most informative from the visualizations above
		bin_size = 4
		X[f'episode_num_binned_{bin_size}'] = pd.cut(X['episode_num'], 
                                                     range(0, 101, bin_size), 
                                                     labels=[f'episode{col}_to_{col+bin_size}' for col in range(0, 101-bin_size, bin_size)]).astype(object)
        
		print('FeatureEng2', X.isna().sum().sum())
        
		sentiment_map = {
			'Negative': -1, 
			'Neutral': 0, 
			'Positive': 1
		}
		X['sentiment_score'] = X['Episode_Sentiment'].map(sentiment_map)
		X['Host_sentiment_impact'] = X['sentiment_score'] * X['Host_Popularity_percentage']
		X['Guest_sentiment_impact'] = X['sentiment_score'] * X['Guest_Popularity_percentage']
        
		print('FeatureEng3', X.isna().sum().sum())
		
		print(f"FeatureEng: Output shape {X.shape}")
		
		return X


X_train = df_train.drop(columns=['Listening_Time_minutes'])
y_train = df_train['Listening_Time_minutes']


one_hot_cat = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 
               'Episode_Sentiment', 'Publication_AMPM', 'episode_num_binned_4']
scale_num = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
             'Guest_Popularity_percentage', 'Number_of_Ads', 'Total_pop', 
             'pure_len', 'Ad_density', 'content_minutes', 'Publication_Hour', 
             'title_len', 'episode_num', 'sentiment_score', 'Host_sentiment_impact', 
             'Guest_sentiment_impact']
others = ['is_weekend', 'Has_Guest'] 


class NaNImputerWrapper(BaseEstimator, TransformerMixin):
	def __init__(self, train_sample_size=30_000, verbose=True):
		self.train_sample_size = train_sample_size
		self.verbose = verbose
		self.imputer = NaNImputer(self.train_sample_size, self.verbose)

	def fit(self, X, y=None):
		self.upper_and_lower_limits = []
		for col in X.select_dtypes(include=np.number).columns:
			col_stats = X[col].describe()
			q3, q1 = col_stats['75%'], col_stats['25%'] 
			iqr = q3-q1
			upper_limit = q3 + 1.5 * iqr
			lower_limit = q1 - 1.5 * iqr
			self.upper_and_lower_limits.append((lower_limit, upper_limit))

		return self

	def transform(self, X):
		print(f"NaNImputerWrapper: Input shape {X.shape}")
		
		# set outlier in all cols to np.nan
		for idx, col in enumerate(X.select_dtypes(include=np.number).columns):
			X.loc[X[col] > self.upper_and_lower_limits[idx][1], col] = np.nan
			X.loc[X[col] < self.upper_and_lower_limits[idx][0], col] = np.nan

		X['Has_Guest'] = X['Guest_Popularity_percentage'].notna().astype(int)
		X['Guest_Popularity_percentage'] = X['Guest_Popularity_percentage'].fillna(0)

		result = self.imputer.impute(X)
		print(f"NaNImputerWrapper: Output shape {result.shape}")
		print('NanImputerWrapper', result.isna().sum().sum())
		return result


class TopImportanceFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model, threshold=0.95):
        self.model = model
        self.threshold = threshold
        
    def fit(self, X, y=None):
        print(type(X))
        print(X.dtype)
        X = np.asarray(X, dtype='float64')
        print('TopImportanceFeatures1', np.isnan(X).sum().sum())
        print(f"TopImportanceFeatures: Input X shape {X.shape}, y shape {None if y is None else y.shape}")
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        cumulative_importances = importances[sorted_idx].cumsum()
        cutoff_idx = np.searchsorted(cumulative_importances, self.threshold) + 1
        self.top_features_ = sorted_idx[:cutoff_idx]
        return self
    
    def transform(self, X):
        print(f"TopImportanceFeatures transform: Input shape {X.shape}")
        if hasattr(X, "toarray"):  # For sparse
            X = X.toarray()
        else:
            X = np.asarray(X, dtype='float64')  # Coerce all types to float
    
        print(f"TopImportanceFeatures transform: Output shape {X[:, self.top_features_].shape}")
        print('TopImportanceFeatures NaNs:', np.isnan(X[:, self.top_features_]).sum())
        return X[:, self.top_features_]


pd.reset_option('display.max_rows')


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2)


!pip install pytorch_tabnet
!pip install torch


X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

encoder_scaler_transformer = ColumnTransformer([
	('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), one_hot_cat), 
    ('scaler', StandardScaler(), scale_num), 
    ('rem', 'passthrough', others)
])

preprocessor_nn = Pipeline([
	('imputer', NaNImputerWrapper(train_sample_size=X_train_nn.shape[0])),
	('feature_eng', FeatureEng()),
	('encoder_scaler', encoder_scaler_transformer)
	# ('selector', TopImportanceFeatures(XGBRegressor(random_state=42), threshold=0.95))
])

X_train_preprocessed_nn = preprocessor_nn.fit_transform(X_train_nn)
X_val_preprocessed_nn = preprocessor_nn.transform(X_val_nn)


X_train_preprocessed_nn = np.asarray(X_train_preprocessed_nn).astype(np.float32)
X_val_preprocessed_nn = np.asarray(X_val_preprocessed_nn).astype(np.float32)
y_train_nn = np.asarray(y_train_nn).astype(np.float32)
y_val_nn = np.asarray(y_val_nn).astype(np.float32)


from pytorch_tabnet.tab_model import TabNetRegressor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tabnet_model = TabNetRegressor(
    device_name=device,
    n_d=64,  
    n_a=64,  
    n_steps=5,  
    gamma=1.5,  
    n_independent=2, 
    n_shared=2,  
    lambda_sparse=1e-3,  
    momentum=0.3,
    clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(mode="min", 
                         patience=5,
                         min_lr=1e-5,
                         factor=0.5),
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    verbose=1
)

X_train_np = X_train_preprocessed_nn
y_train_np = y_train_nn.reshape(-1, 1)
X_valid_np = X_val_preprocessed_nn
y_valid_np = y_val_nn.reshape(-1, 1)

tabnet_model.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_train_np, y_train_np), (X_valid_np, y_valid_np)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128
)


X_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
# test_id = X_test['id']
X_test.drop(columns=['id'], inplace=True)


X_test_preprocessed_nn = preprocessor_nn.transform(X_test).astype('float32')
y_pred_tabnet = tabnet_model.predict(X_test_preprocessed_nn)


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sample_submission['Listening_Time_minutes'] = y_pred_tabnet
sample_submission.to_csv('submission.csv', index=False)

