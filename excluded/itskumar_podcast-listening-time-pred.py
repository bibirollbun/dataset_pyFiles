import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')


train.shape, test.shape


train.head()


test.tail()


train.info()


train.describe()


train = train[train['Episode_Length_minutes'] > 0]
train[train['Episode_Length_minutes'] == 0].shape


train.hist(bins=20, figsize=(15, 10));


import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sns.pairplot(train.dropna(), diag_kind='kde', plot_kws={'alpha': 0.6});


num_cols = [col for col in train.columns if train[col].dtype != 'object']  # Numerical columns

plt.figure(figsize=(12, 10)) 

# Loop through numerical columns and plot boxplots
for i, col in enumerate(num_cols, 0):
    plt.subplot(2, 3, i+1)
    train.boxplot(column=col)
    plt.title(f'Boxplot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout() 

plt.show();


train['Podcast_Name'].value_counts(ascending=False).head(10)


top_names = train['Podcast_Name'].value_counts(ascending=False).head(10).index
filtered_df = train[train['Podcast_Name'].isin(top_names)]

sns.boxplot(data=filtered_df, x='Podcast_Name', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.title('Listening Time by Top Podcast');


plt.figure(figsize=(12, 6))
sns.countplot(data=train, y='Genre', order=train['Genre'].value_counts().index)
plt.title('Episode Count by Genre')
plt.show();


plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='Genre', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.title('Listening Time Distribution by Genre')
plt.show();


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='Publication_Day', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title('Episodes Published by Day')
plt.xticks(rotation=45)
plt.show();


plt.figure(figsize=(10, 5))
sns.boxplot(data=train, x='Publication_Day', y='Listening_Time_minutes', 
            order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title('Listening Time by Publication Day')
plt.show();


train['Publication_Time'].value_counts()


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='Publication_Time', order=['Morning', 'Afternoon', 'Evening', 'Night'])
plt.title('Episode Count by Time of Day')
plt.ylabel('Number of Episodes')
plt.xlabel('Publication Time')
plt.show();


plt.figure(figsize=(8, 5))
sns.boxplot(data=train, x='Publication_Time', y='Listening_Time_minutes', order=['Morning', 'Afternoon', 'Evening', 'Night'])
plt.title('Listening Time by Publication Time')
plt.xlabel('Publication Time')
plt.ylabel('Listening Time (minutes)')
plt.show();


plt.figure(figsize=(8, 5))
sns.barplot(data=train, x='Publication_Time', y='Number_of_Ads', order=['Morning', 'Afternoon', 'Evening', 'Night'])
plt.title('Average Number of Ads by Publication Time')
plt.xlabel('Publication Time')
plt.ylabel('Average Ads')
plt.show();


train['Episode_Sentiment'].value_counts()


plt.figure(figsize=(6, 4))
sns.boxplot(data=train, x='Episode_Sentiment', y='Listening_Time_minutes')
plt.title('Listening Time by Episode Sentiment')
plt.show();


plt.figure(figsize=(6, 4))
sns.barplot(data=train, x='Episode_Sentiment', y='Number_of_Ads')
plt.title('Average Number of Ads by Sentiment')
plt.show();


train.isna().sum()


test.isna().sum()


train['Episode_Length_minutes'] = pd.to_numeric(train['Episode_Length_minutes'], errors='coerce')
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())

test['Episode_Length_minutes'] = pd.to_numeric(test['Episode_Length_minutes'], errors='coerce')
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean())


train['Number_of_Ads'] = pd.to_numeric(train['Number_of_Ads'], errors='coerce')
test['Number_of_Ads'] = pd.to_numeric(test['Number_of_Ads'], errors='coerce')


train = train.dropna(subset=['Number_of_Ads'])
test = test.dropna(subset=['Number_of_Ads'])


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"{column} - Removing outliers outside [{lower_bound}, {upper_bound}]")
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]



train_cleaned = remove_outliers_iqr(train, 'Number_of_Ads')
train_cleaned = remove_outliers_iqr(train_cleaned, 'Episode_Length_minutes')


train_cleaned.shape, test.shape


plt.figure(figsize=(12, 8))
sns.heatmap(train_cleaned.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm');


# Feature Engineering

train_cleaned['Ads_per_minute'] = train_cleaned['Number_of_Ads'] / train_cleaned['Episode_Length_minutes']
train_cleaned['Overall_popularity'] = train_cleaned['Host_Popularity_percentage'] + train_cleaned['Guest_Popularity_percentage']
train_cleaned['Popularity_diff'] = train_cleaned['Host_Popularity_percentage'] - train_cleaned['Guest_Popularity_percentage']
train_cleaned['Popularity_Ratio'] = train_cleaned['Guest_Popularity_percentage'] / (train_cleaned['Host_Popularity_percentage'] + 1e-3)
train_cleaned['Is_weekend'] = train_cleaned['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
train_cleaned['Episode_No'] = train_cleaned['Episode_Title'].str.extract(r'(\d+)').astype(float)
train_cleaned['Lenght_Bucket'] = pd.cut(train_cleaned['Episode_Length_minutes'],
                                        bins=[0, 30, 60, 90, 200],
                                        labels=['Short', 'Medium', 'Long', 'Very_Long'])

test['Ads_per_minute'] = test['Number_of_Ads'] / test['Episode_Length_minutes']
test['Overall_popularity'] = test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']
test['Popularity_diff'] = test['Host_Popularity_percentage'] - test['Guest_Popularity_percentage']
test['Popularity_Ratio'] = test['Guest_Popularity_percentage'] / (test['Host_Popularity_percentage'] + 1e-3)
test['Is_weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
test['Episode_No'] = test['Episode_Title'].str.extract(r'(\d+)').astype(float)
test['Lenght_Bucket'] = pd.cut(test['Episode_Length_minutes'],
                                        bins=[0, 30, 60, 90, 200],
                                        labels=['Short', 'Medium', 'Long', 'Very_Long'])


train_cleaned['Lenght_Bucket'] = train_cleaned['Lenght_Bucket'].map({'Short': 1, 'Medium': 2, 'Long': 3, 'Very_Long': 4})
test['Lenght_Bucket'] = test['Lenght_Bucket'].map({'Short': 1, 'Medium': 2, 'Long': 3, 'Very_Long': 4})


train_cleaned.select_dtypes(include=['number']).corr()['Listening_Time_minutes'].sort_values(ascending=False)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = train_cleaned.drop(columns=['Listening_Time_minutes']), train_cleaned['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=.25,
                                                  random_state=777
                                                 )


X_train.shape, X_val.shape


import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=50)


class SentimentMapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        return np.vectorize(mapping.get)(X)#.reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        return ['score']

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        text = text.lower()                      # Lowercase
        text = text.strip()                      # Trim whitespace
        text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
        text = re.sub(r'\d+', '', text)          # Remove digits
        text = re.sub(r'\s+', ' ', text)         # Normalize whitespace
        return text
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return ['cleaned_text']
        return input_features


text_pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('count_vector', vectorizer)
])


sc = StandardScaler()
ordinal_encoder = OrdinalEncoder(categories=[['Morning', 'Afternoon', 'Evening', 'Night']])
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

ordinal_features = ['Publication_Time']
nominal_features = ['Genre', 'Publication_Day',]
numeric_features = [c for c in X_train.select_dtypes(include=['number']).columns]

preprocessor = ColumnTransformer(transformers=[
    ('ord', ordinal_encoder, ordinal_features),
    ('onehot', one_hot_encoder, nominal_features),
    ('sentiment', SentimentMapper(), ['Episode_Sentiment']),
    ('num', sc, numeric_features),
])

preprocessor_with_names = ColumnTransformer(transformers=[
    ('ord', ordinal_encoder, ordinal_features),
    ('onehot', one_hot_encoder, nominal_features),
    ('sentiment', SentimentMapper(), ['Episode_Sentiment']),
    ('num', sc, numeric_features),
    ('podcast_name', text_pipeline, 'Podcast_Name') 
])


features = ordinal_features+nominal_features+numeric_features+['Episode_Sentiment']

X_train_scaled = preprocessor.fit_transform(X_train[features])
X_val_scaled = preprocessor.transform(X_val[features])

feature_names = preprocessor.get_feature_names_out()

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_names)
X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=feature_names)


features = ordinal_features+nominal_features+numeric_features+['Episode_Sentiment', 'Podcast_Name']

X_train_with_names = preprocessor_with_names.fit_transform(X_train[features])
X_val_with_names = preprocessor_with_names.transform(X_val[features])

feature_names = preprocessor_with_names.get_feature_names_out()

X_train_with_names = pd.DataFrame(X_train_with_names, index=X_train.index, columns=feature_names)
X_val_with_names = pd.DataFrame(X_val_with_names, index=X_val.index, columns=feature_names)


X_train_scaled.shape, X_val_scaled.shape


X_train_with_names.shape, X_val_with_names.shape


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_model_1 = LinearRegression()
lr_model_1.fit(X_train_scaled, y_train)
y_pred = lr_model_1.predict(X_val_scaled)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_model_2 = LinearRegression()
lr_model_2.fit(X_train_with_names, y_train)
y_pred = lr_model_2.predict(X_val_with_names)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(criterion='friedman_mse', max_depth=8,)
dt_model.fit(X_train_scaled, y_train)
y_pred = dt_model.predict(X_val_scaled)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


from sklearn.tree import DecisionTreeRegressor

dt_model_2 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=8,)
dt_model_2.fit(X_train_with_names, y_train)
y_pred = dt_model_2.predict(X_val_with_names)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


# from sklearn.ensemble import RandomForestRegressor

# rf_model = RandomForestRegressor(n_estimators=200, random_state=777,)
# rf_model.fit(X_train_scaled, y_train)
# y_pred = rf_model.predict(X_val_scaled)

# print("MAE:", mean_absolute_error(y_val, y_pred))
# print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


from catboost import CatBoostRegressor

cat_model_1 = CatBoostRegressor(iterations=200, verbose=0)
cat_model_1.fit(X_train_scaled, y_train)
y_pred = cat_model_1.predict(X_val_scaled)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


from catboost import CatBoostRegressor

cat_features = ordinal_features+nominal_features+['Episode_Sentiment', 'Podcast_Name']
cat_model = CatBoostRegressor(cat_features=cat_features, iterations=300,
                              verbose=0)
cat_model.fit(X_train[features], y_train, early_stopping_rounds=30)
y_pred = cat_model.predict(X_val[features])

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=700,          
    learning_rate=0.05,        
    max_depth=6,               
    gamma=0.1,                 
    min_child_weight=3,        
    subsample=0.8,                 
    # reg_alpha=0.5,             # L1 regularization (sparse models)
    # reg_lambda=1.0,            # L2 regularization (stability)
    eval_metric='rmse',
    random_state=666,
    enable_categorical=True,
    n_jobs=-1                  
)
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_val_scaled)
# xgb_model.fit(X_train_with_names, y_train)
# y_pred = xgb_model.predict(X_val_with_names)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('RMSE:', mean_squared_error(y_val, y_pred, squared=False))


import xgboost as xgb

fig, ax = plt.subplots(figsize=(8, 12))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=25, ax=ax)
plt.show();


test_scaled = preprocessor.transform(test[features])
test_scaled = pd.DataFrame(test_scaled, index=test.index, columns=preprocessor.get_feature_names_out())

predictions = xgb_model.predict(test_scaled)
# predictions = cat_model.predict(test[features])

submission = pd.DataFrame({'id': test.index, 'Listening_Time_minutes': predictions})


submission.to_csv('submission.csv', index=False)




