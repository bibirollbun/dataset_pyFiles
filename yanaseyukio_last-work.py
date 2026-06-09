import numpy as np
import pandas as pd

# データの読み込み
train = pd.read_csv("../input/playground-series-s4e4/train.csv")
test = pd.read_csv("../input/playground-series-s4e4/test.csv")

#trainから目的変数を分離
train_x = train.drop(['Rings'], axis=1)
train_y = train['Rings']

#学習データからidを除去
train_x = train_x.drop(['id'], axis=1)
train_id = train['id']

#テストデータからidを除去
test_x = test.drop(['id'], axis=1)
test_id = test['id']

#データからSexを除去
train_x = train_x.drop(['Sex'], axis=1)
train_sex = train['Sex']
test_x = test_x.drop(['Sex'], axis=1)
test_sex = test['Sex']

print(train_x)
print(test_x)


#すべての変数で割り算して割合を算出
col_name = list(train_x.columns)

for i_col in range(len(col_name)):
    feature_name = col_name[i_col]
    for feature_next_name in col_name[i_col+1:]:
        raito_feature_name = f'ratio_{feature_name}_to_{feature_next_name}'
        train_x[raito_feature_name] = train_x[feature_name] / train_x[feature_next_name]
        test_x[raito_feature_name] = test_x[feature_name] / test_x[feature_next_name]

        #infを0に置き換え
        train_x[raito_feature_name] = train_x[raito_feature_name].replace([np.inf, -np.inf], 0.0)
        test_x[raito_feature_name] = test_x[raito_feature_name].replace([np.inf, -np.inf], 0.0)



# 7個目までの特徴の間の相関係数ヒートマップ
import matplotlib.pyplot as plt
import seaborn as sns

_, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train_x.iloc[:, :7].corr(), annot=True, ax=ax)


# 7個目以降の特徴の間の相関係数ヒートマップ
import matplotlib.pyplot as plt
import seaborn as sns

_, ax = plt.subplots(figsize=(12, 11))
sns.heatmap(train_x.iloc[:, 7:].corr(), annot=True, ax=ax)


#特徴量の選定
col_name = list(train_x.columns)

for c_col in range(len(col_name)):
    if col_name[c_col] in {'Length', 'Diameter', 'Height', 'Whole weight', 
                           'Whole weight.1', 'Whole weight.2', 'Shell weight', 
                           'ratio_Whole weight_to_Whole weight.1', 
                           'ratio_Whole weight.1_to_Shell weight'}:
        pass
    else:
        train_x = train_x.drop([col_name[c_col]], axis=1)
        test_x = test_x.drop([col_name[c_col]], axis=1)

print(train_x)


from sklearn.preprocessing import OneHotEncoder

col = 'Sex'
#Sexを結合
train_x[col] = train_sex
test_x[col] = test_sex

#ワンホットエンコーディング
oeTrain = pd.get_dummies(train_x[col], prefix='Sex', drop_first=False, dtype='int')
train_x = pd.concat([train_x, oeTrain], axis=1)
oeTest = pd.get_dummies(test_x[col], prefix='Sex', drop_first=False, dtype='int')
test_x = pd.concat([test_x, oeTest], axis=1)

#データからSexを除去
train_x = train_x.drop([col], axis=1)
train_sex = train[col]
test_x = test_x.drop([col], axis=1)
test_sex = test[col]

print(train_x)


import matplotlib.pyplot as plt
train_x.hist(bins=100, figsize=(15, 10), color='blue', grid=True)
plt.show()


from sklearn.preprocessing import QuantileTransformer

num_cols = ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 
            'Whole weight.2', 'Shell weight', 'ratio_Whole weight_to_Whole weight.1', 
            'ratio_Whole weight.1_to_Shell weight']

# 学習データに基づいて複数列のRankGaussによる変換を定義
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 変換後のデータで各列を置換
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])

# 変換後の学習データのヒストグラムを算出/表示
train_x.hist(bins=100, figsize=(15, 10), color='red', grid=True, label='pandas')
plt.show()


import lightgbm as lgb

lgb_train = lgb.Dataset(train_x, train_y)
lgb_param = {'seed':71}
num_round = 100

model = lgb.train(lgb_param, lgb_train, num_boost_round=num_round)

pred = model.predict(test_x)
submission = pd.DataFrame({'id':test_id, 'Ring':pred})
submission.to_csv("submision.csv", index=False)

print(submission)

