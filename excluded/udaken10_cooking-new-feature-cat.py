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


train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')


target_df = train_df['HOMELESS_RATE']


all_data = pd.concat([train_df, test_df], axis=0)


def create_new_features(data):

    try:
        # 元のデータファイルを読み込みます
        df = data
        print("元のデータファイルを正常に読み込みました。")

        # --- 1. 年齢層の集約 ---
        # 18歳未満と18-24歳を合計して「若年層の割合」を作成
        df['YOUTH_AND_YOUNG_ADULT_PCT'] = df['AGE_U18_PCT'] + df['AGE_18_24_PCT']

        # --- 2. リスク要因のカテゴリー化 ---
        # 25-34歳の割合を「低・中・高」の3つのレベルに分類
        df['YOUNG_ADULT_LEVEL'] = pd.qcut(df['AGE_25_34_PCT'],
                                         q=3,
                                         labels=[1,2,3],
                                         duplicates='drop')

        # --- 3. 複数のリスクの掛け合わせ ---
        # 孤立しがちな男性と人種的マイノリティの二重のリスクを表現
        df['NONFAMILY_MALE_X_BLACK_INTERACTION'] = df['NONFAMILY_SINGLE_MALE_PCT'] * df['RACE_BLACK_NH_PCT']

        # --- 4. 社会的安定性に関する比率 ---
        # 家族世帯と非家族世帯の比率を計算し、コミュニティの安定性を表現
        non_family_sum = df['NONFAMILY_SINGLE_MALE_PCT'] + df['NONFAMILY_SINGLE_FEMALE_PCT'] + df['MULTI_PERSON_NONFAMILY_HH_PCT']
        epsilon = 1e-6  # ゼロでの割り算を防ぐための微小な値
        df['FAMILY_VS_NONFAMILY_RATIO'] = df['FAMILY_HH_TOTAL'] / (non_family_sum + epsilon)

        print("新しい4つの特徴量を正常に作成しました。")

        # 新しく作成した特徴量の内容を確認
        print("\n--- 作成された特徴量（最初の5行） ---")
        new_columns = [
            'YOUTH_AND_YOUNG_ADULT_PCT', 'YOUNG_ADULT_LEVEL',
            'NONFAMILY_MALE_X_BLACK_INTERACTION', 'FAMILY_VS_NONFAMILY_RATIO'
        ]
        print(df[new_columns].head())

        # 特徴量を追加したデータフレームを新しいCSVファイルとして保存
        output_filename = 'train_with_features.csv'
        df.to_csv(output_filename, index=False)
        print(f"\n全ての特徴量を含む新しいファイル '{output_filename}' を保存しました。")

        return df

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりませんでした。ファイルパスを確認してください。")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None


df_featured = create_new_features(all_data)


new_train_df = df_featured.iloc[:len(train_df)]
new_test_df = df_featured.iloc[len(train_df):]


new_df_train = df_featured[:len(train_df)].drop(['ID', 'HOMELESS_RATE'], axis = 1)
new_df_test = df_featured[len(train_df):].drop(['ID', 'HOMELESS_RATE'], axis = 1)

new_df_train.columns.values


features = ['RACE_BLACK_NH_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT', 'AGE_25_34_PCT','FAMILY_HH_CHILD_LT18_PCT', 'VETERAN_POP_PCT', 'FAMILY_HH_TOTAL', 'YOUTH_AND_YOUNG_ADULT_PCT',
       'YOUNG_ADULT_LEVEL', 'NONFAMILY_MALE_X_BLACK_INTERACTION',
       'FAMILY_VS_NONFAMILY_RATIO'] 


new_df_train = new_df_train[features]
new_df_test = new_df_test[features]


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from catboost import Pool




X_train, X_val, y_train, y_val = train_test_split(new_df_train, target_df, test_size=0.2, random_state=42)

train_Pool = Pool(X_train, y_train, cat_features=['YOUNG_ADULT_LEVEL'])
val_Pool = Pool(X_val, y_val, cat_features=['YOUNG_ADULT_LEVEL'])

model_catboost = CatBoostRegressor(iterations=100,
                                   learning_rate=0.05,
                                   depth=7,
                                   l2_leaf_reg=3,
                                   eval_metric='RMSE',
                                   random_seed=42,
                                   verbose=False,
                                   )

model_catboost.fit(train_Pool,
                   eval_set=val_Pool,
                   use_best_model=True,
                   early_stopping_rounds=100)

model_catboost.predict(val_Pool)

RMSE = np.sqrt(mean_squared_error(y_val, model_catboost.predict(val_Pool)))
print(RMSE)


prediction = model_catboost.predict(new_df_test)
prediction


sub['HOMELESS_RATE'] = prediction
sub.to_csv('submission.csv', index=False)




