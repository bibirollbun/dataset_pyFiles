# !pip uninstall -y numpy scikit-learn xgboost
# !pip install numpy==1.24.4 scikit-learn==1.1.3 xgboost==1.7.6 --force-reinstall
# !pip install verstack


!pip install verstack


import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from verstack import NaNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import copy


df_original = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


# df_original.drop(columns=['id'], inplace=True)
df_train.drop(columns=['id'], inplace=True)


X_train = df_train.drop(columns=['Listening_Time_minutes'])
y_train = df_train['Listening_Time_minutes']


# 1 = 'original', 0 = 'artificial'
df_original['source'] = 1
df_train['source'] = 0


df_combined = pd.concat([df_original, df_train], axis=0).drop('Listening_Time_minutes', axis=1)
X_combined = df_combined.drop('source', axis=1)
y_combined = df_combined['source']


df_original = df_original[df_original['Listening_Time_minutes'].notna()]
X_original = df_original.drop(columns=['Listening_Time_minutes'])
y_original = df_original['Listening_Time_minutes']


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


one_hot_cat = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time',
               'Episode_Sentiment', 'Publication_AMPM', 'episode_num_binned_4']
scale_num = ['Episode_Length_minutes', 'Host_Popularity_percentage',
             'Guest_Popularity_percentage', 'Number_of_Ads', 'Total_pop',
             'pure_len', 'Ad_density', 'content_minutes', 'Publication_Hour',
             'title_len', 'episode_num', 'sentiment_score', 'Host_sentiment_impact',
             'Guest_sentiment_impact']
others = ['is_weekend', 'Has_Guest']


cat_encoder = ColumnTransformer([
	('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), one_hot_cat),
    ('rem', 'passthrough', scale_num + others)
])


class NaNImputerWrapper1(BaseEstimator, TransformerMixin):
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

		# these features may distinguish between the original and artificial datasets
		for idx, col in enumerate(X.select_dtypes(include=np.number).columns):
			X[f'{col}_outlier'] = (X[col] > self.upper_and_lower_limits[idx][1]) | (X[col] < self.upper_and_lower_limits[idx][0])

		X['Has_Guest'] = X['Guest_Popularity_percentage'].notna().astype(int)
		X['Guest_Popularity_percentage'] = X['Guest_Popularity_percentage'].fillna(0)

		result = self.imputer.impute(X)
		print(f"NaNImputerWrapper: Output shape {result.shape}")
		print('NanImputerWrapper', result.isna().sum().sum())
		return result


preprocessor1 = Pipeline([
	('imputer', NaNImputerWrapper1(train_sample_size=X_combined.shape[0])),
    ('feature_eng', FeatureEng()),
	('encoder_onehot', cat_encoder)
])


X_combined_preprocessed = preprocessor1.fit_transform(X_combined)


xgb_params = {
    'n_estimators': 500,
    'random_state': 42,
    'max_depth': 14,
    'learning_rate': 0.04,
    'min_child_weight': 10,
    #'early_stopping_rounds': 150,
    'colsample_bytree': 0.8,
    'n_jobs': -1
    #'subsample': 0.9,
}


source_model = XGBClassifier(**xgb_params)
source_model.fit(X_combined_preprocessed, y_combined)


class NaNImputerWrapper2(BaseEstimator, TransformerMixin):
	def __init__(self, train_sample_size=30_000, verbose=True):
		self.train_sample_size = train_sample_size
		self.verbose = verbose
		self.imputer = NaNImputer(self.train_sample_size, self.verbose)

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		print(f"NaNImputerWrapper: Input shape {X.shape}")

		X['Has_Guest'] = X['Guest_Popularity_percentage'].notna().astype(int)
		X['Guest_Popularity_percentage'] = X['Guest_Popularity_percentage'].fillna(0)

		result = self.imputer.impute(X)
		print(f"NaNImputerWrapper: Output shape {result.shape}")
		print('NanImputerWrapper', result.isna().sum().sum())
		return result


preprocessor2 = Pipeline([
	('imputer', NaNImputerWrapper2(train_sample_size=X_original.shape[0])),
    ('feature_eng', FeatureEng()),
	('encoder_onehot', cat_encoder)
])


X_original_preprocessed = preprocessor2.fit_transform(X_original)


poly_ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=5))
])


original_model = poly_ridge_pipeline.fit(X_original_preprocessed, y_original)


preprocessor3 = Pipeline([
	('imputer', NaNImputerWrapper2(train_sample_size=X_train.shape[0])),
    ('feature_eng', FeatureEng()),
	('encoder_onehot', cat_encoder)
])


X_train_preprocessed = preprocessor3.fit_transform(X_train)


poly_ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=5))
])


# we cannot use the original array as it's very large and this raises a memory error
from sklearn.model_selection import train_test_split
X_sampled, _, y_sampled, _ = train_test_split(X_train_preprocessed, y_train, train_size=0.3, random_state=42)
train_model = poly_ridge_pipeline.fit(X_sampled, y_sampled)


X_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
# test_id = X_test['id']
X_test.drop(columns=['id'], inplace=True)


X_test_preprocessed1 = preprocessor1.transform(X_test)
X_test_preprocessed2 = preprocessor2.transform(X_test)
X_test_preprocessed3 = preprocessor3.transform(X_test)


y_pred_src_proba = source_model.predict_proba(X_test_preprocessed1)
y_pred_test_originalModel = original_model.predict(X_test_preprocessed2)
y_pred_test_trainModel = train_model.predict(X_test_preprocessed3)


y_pred_test = y_pred_src_proba[:, 0] * y_pred_test_trainModel + y_pred_src_proba[:, 1] * y_pred_test_originalModel


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sample_submission['Listening_Time_minutes'] = y_pred_test
sample_submission.to_csv('submission.csv', index=False)

