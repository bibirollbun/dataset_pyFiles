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


import numpy as np
import pandas as pd
import lightgbm as lgb
import ast # 文字列をPythonのオブジェクトとして評価するためのライブラリ
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter



train_df = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
print("Data loaded successfully!")



print("\n--- 2. Feature Engineering ---")


all_df = pd.concat([train_df.drop(['revenue'], axis=1), test_df], axis=0)


def parse_json(column_str):
    if isinstance(column_str, str):
        try:
            return ast.literal_eval(column_str)
        except:
            return []
    return []

json_columns = ['belongs_to_collection', 'genres', 'production_companies', 
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

for col in json_columns:
    all_df[col] = all_df[col].apply(parse_json)

print("Creating basic features...")
all_df['has_collection'] = all_df['belongs_to_collection'].apply(lambda x: 1 if x else 0)
all_df['has_homepage'] = all_df['homepage'].notna().astype(int)
all_df['has_tagline'] = all_df['tagline'].notna().astype(int)

print("Creating numerical features...")
all_df['budget'] = all_df['budget'].replace(0, np.nan) # 0を欠損値として扱う
# budgetとruntimeの欠損値をrelease_yearの中央値で埋める
all_df['budget'] = all_df['budget'].fillna(all_df['budget'].median())
all_df['runtime'] = all_df['runtime'].fillna(all_df['runtime'].median())


print("Creating count features from JSON...")
for col in ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']:
    all_df[f'num_{col}'] = all_df[col].apply(len)

print("Extracting Team & Background features...")


def get_names_from_list(data_list, key='name', top_n=5):
    if not isinstance(data_list, list): return ['Unknown'] * top_n
    names = [item.get(key, 'Unknown') for item in data_list]
    return (names + ['Unknown'] * top_n)[:top_n]

def get_job_name(crew_list, job_title):
    if not isinstance(crew_list, list): return 'Unknown'
    return next((member['name'] for member in crew_list if member.get('job') == job_title), 'Unknown')

all_df['first_genre'] = all_df['genres'].apply(lambda x: x[0]['name'] if x else 'Unknown')
all_df['production_company'] = all_df['production_companies'].apply(lambda x: x[0]['name'] if x else 'Unknown')
all_df['director'] = all_df['crew'].apply(get_job_name, job_title='Director')
all_df['writer'] = all_df['crew'].apply(get_job_name, job_title='Writer')
all_df['producer'] = all_df['crew'].apply(get_job_name, job_title='Producer')
all_df['lead_actor_name'] = all_df['cast'].apply(lambda x: x[0]['name'] if x else 'Unknown')

print("Creating text features...")
all_df['overview_word_count'] = all_df['overview'].str.split().str.len().fillna(0)
all_df['title_char_count'] = all_df['title'].str.len().fillna(0)

print("Encoding all categorical features...")
categorical_cols = [
    'first_genre', 'production_company', 'director', 'writer', 'producer',
    'lead_actor_name', 'original_language'
]
for col in categorical_cols:
    all_df[f'{col}_code'], _ = pd.factorize(all_df[col])

train_processed = all_df[:len(train_df)]
test_processed = all_df[len(train_df):]


print("\n--- 3. Preparing for training ---")
# 使用する特徴量をリストで定義
features = [
    # 基本数値
    'budget', 'popularity', 'runtime',
    # フラグ
    'has_collection', 'has_homepage', 'has_tagline',
    # カウント
    'num_genres', 'num_production_companies', 'num_production_countries',
    'num_spoken_languages', 'num_Keywords', 'num_cast', 'num_crew',
    # テキスト
    'overview_word_count', 'title_char_count',
    # カテゴリカル
    'first_genre_code', 'production_company_code', 'director_code', 'writer_code', 'producer_code',
    'lead_actor_name_code', 'original_language_code'
]

X_train = train_processed[features]
X_test = test_processed[features]
y_train_log = np.log1p(train_df['revenue'])

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)
print(f"Using {len(features)} features for training.")


print("\n--- 4. Training model ---")
params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 3000,
    'learning_rate': 0.003, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
}
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train_log)


print("\n--- 5. Making predictions ---")
log_predictions = model.predict(X_test)
final_predictions = np.expm1(log_predictions)

submission = pd.DataFrame({
    'id': test_df['id'],
    'revenue': final_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
print(submission.head())


print("\n--- 6. Feature Importances ---")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(20)) # 上位20個を表示

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

