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


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


print('data preparing...')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print('\nEDA:')
print(f'train_shape: {train_data.shape}')
print(f'test_shape: {test_data.shape}')
print('\ntrain_info:')
print(train_data.info())
print('\ntrain_describtion:')
print(train_data.describe())

# check lack
print('\nlack:')
print(train_data.isnull().sum())

# 数据预处理
print('\npreprogress...')

# 合并训练集和测试集进行预处理
test_data['Listening_Time_minutes'] = np.nan  # 添加目标列以便合并
all_data = pd.concat([train_data, test_data], axis=0)


# Feature Engineering

# 1. handle missing values
# use median
all_data['Episode_Length_minutes'].fillna(all_data['Episode_Length_minutes'].median(), inplace=True)
all_data['Guest_Popularity_percentage'].fillna(all_data['Guest_Popularity_percentage'].median(), inplace=True)

# 2. making new feature
# an interesting feature
all_data['Podcast_Name_Length'] = all_data['Podcast_Name'].apply(lambda x: len(str(x)))

# episode number feature
def extract_episode_number(title):
    try:
        return int(''.join(filter(str.isdigit, str(title))))
    except:
        return 0

all_data['Episode_Number'] = all_data['Episode_Title'].apply(extract_episode_number)

# feature 2
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
all_data['Publication_Time_Encoded'] = all_data['Publication_Time'].map(time_mapping)

# feature 3
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
all_data['Publication_Day_Encoded'] = all_data['Publication_Day'].map(day_mapping)

# feature 4
sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
all_data['Episode_Sentiment_Encoded'] = all_data['Episode_Sentiment'].map(sentiment_mapping)

# split train set and test set
train_processed = all_data[all_data['Listening_Time_minutes'].notna()].copy()
test_processed = all_data[all_data['Listening_Time_minutes'].isna()].copy()

# choose features
numerical_features = [
    'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',
    'Number_of_Ads', 'Episode_Number', 'Podcast_Name_Length',
    'Publication_Time_Encoded', 'Publication_Day_Encoded', 'Episode_Sentiment_Encoded'
]

categorical_features = ['Genre']


X = train_processed[numerical_features + categorical_features]
y = train_processed['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model Training and evaluation 
print('\nModel Training and evaluation...')

# RF
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
rf_val_r2 = r2_score(y_val, rf_val_pred)

print(f'RF_RMSE: {rf_val_rmse:.4f}')
print(f'RF_R²: {rf_val_r2:.4f}')

# GBR
gbr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

gbr_model.fit(X_train, y_train)
gbr_val_pred = gbr_model.predict(X_val)
gbr_val_rmse = np.sqrt(mean_squared_error(y_val, gbr_val_pred))
gbr_val_r2 = r2_score(y_val, gbr_val_pred)

print(f'GBR_RMSE: {gbr_val_rmse:.4f}')
print(f'GBR_R²: {gbr_val_r2:.4f}')

#choose better model
if gbr_val_rmse < rf_val_rmse:
    final_model = gbr_model
    print('\nGBR is the best model')
else:
    final_model = rf_model
    print('\nRF is the best model')

# use all data to train model
final_model.fit(X, y)



# feature importance
if isinstance(final_model.named_steps['regressor'], RandomForestRegressor):
    feature_importances = final_model.named_steps['regressor'].feature_importances_
    
    
    preprocessor = final_model.named_steps['preprocessor']
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_features)
    
    # creatr feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importances)],
        'Importance': feature_importances
    })
    
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print('\nFeature_importance:')
    print(importance_df.head(10))

# predict in test set
print('\npredicting in test set...')
X_test = test_processed[numerical_features + categorical_features]
test_predictions = final_model.predict(X_test)

# create submission
submission = pd.DataFrame({
    'id': test_processed['id'],
    'Listening_Time_minutes': test_predictions
})

# download it 
submission.to_csv('submission_YinbiAyasa.csv', index=False)
print('\n提交文件已保存为 YinbiAyasa.csv')

