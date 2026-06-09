import numpy as np
import pandas as pd

train_data = pd.read_csv("/kaggle/input/california-homelessness-prediction-challenge/train.csv")

train_data.head()


train_data.describe()


train_data.isnull().sum()


train_data.info()


# Column names
print("  \n".join(train_data.columns))


# Total count of null values per column
train_data.isnull().sum()


# Total count of zeros per column
(train_data == 0).sum()


for col in train_data.columns:
    print(col , train_data[col].value_counts())
    print(col  ,train_data[col].unique())


correlations = train_data.iloc[: , 1:].corr()
correlations["HOMELESS_RATE"].sort_values()


cols = ["TOTAL_HOUSEHOLDS_PCT",
        "FAMILY_HH_TOTAL", 
        "FAMILY_HH_CHILD_LT18_PCT",
        "FAMILY_MEMBERS_UNDER_18_PCT" ,
        "AGE_U18_PCT",
        "AGE_55_59_PCT",
        "VETERAN_POP_PCT",
        "RACE_BLACK_NH_PCT",
        "AGE_35_44_PCT",
        "NONVETERAN_POP_PCT",
        "AGE_25_34_PCT"]

for col in cols:
    correlations = train_data.iloc[: , 1:].corr()
    print(col)
    print(correlations[col].sort_values())
    print('====================================================')


from sklearn.ensemble import RandomForestRegressor

X = train_data.iloc[: , 2:]
y = train_data.iloc[: ,1]

model = RandomForestRegressor(random_state = 42)
model.fit(X ,  y)
importances = pd.Series(model.feature_importances_ , index=X.columns).sort_values(ascending=False)
print(importances.head(10))
print(len(train_data))


importances[:10]


test_data = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')


target_ser = train_data['HOMELESS_RATE']


from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.feature_extraction.text import TfidfVectorizer


train_id = train_data['ID']
test_id = test_data['ID']


train_id_df = pd.DataFrame(train_id)
test_id_df = pd.DataFrame(test_id)


train_id_df.shape


all_id = pd.concat([train_id, test_id], axis = 0).reset_index()
print(len(all_id))  


all_id_df = all_id.drop('index', axis = 1)
type(all_id_df)


vectorizer = TfidfVectorizer()


vectorizer.fit(all_id_df['ID'])


train_X = vectorizer.transform(train_id_df['ID'])

test_X = vectorizer.transform(test_id_df['ID'])


df_train_X = pd.DataFrame(train_X.toarray(), columns = vectorizer.get_feature_names_out())
df_test_X = pd.DataFrame(test_X.toarray(), columns = vectorizer.get_feature_names_out())



df_train_X


train_data_df = pd.concat([train_data, df_train_X], axis = 1)
test_data_df = pd.concat([test_data, df_test_X], axis = 1)


train_data_df.shape


test_data_df.shape


train_data_df.head()


test_data_df.head()


new_train_data = train_data_df.drop('ID', axis = 1)
new_test_data = test_data_df.drop('ID', axis = 1)




import numpy as np


k = 10
kf = KFold(
    n_splits=k,      # 分割数（kの値）
    shuffle=True,    # データをシャッフルするか（通常はTrue）
    random_state=42  # シャッフルの乱数シード（再現性確保のため）
)

# 評価スコアを保存するためのリスト
scores = []


X = new_train_data.drop('HOMELESS_RATE', axis = 1)
y = new_train_data['HOMELESS_RATE']

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# k=10 に設定
k = 10 
kf = KFold(
    n_splits=k,      # 分割数
    shuffle=True,    # データのシャッフルは必ずTrueにする
    random_state=42  # 再現性のためのシード
)

# 評価スコア（例: RMSE）を保存するリスト
rmse_scores = [] 

# k回繰り返しのループ
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"--- Fold {fold+1}/{k} ---")
    
    # 1. データの分割
    # k-foldが生成したインデックスを使ってデータを訓練用と検証用に分ける
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 2. モデルの学習と予測
    model = RandomForestRegressor(random_state=42) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 3. 評価指標の計算
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} の RMSE: {rmse:.4f}")

# 4. 最終結果の集計
print("-" * 30)
print(f"全 {k} 回の平均 RMSE: {np.mean(rmse_scores):.4f}")
print(f"全 {k} 回の RMSE 標準偏差: {np.std(rmse_scores):.4f}")


train_data_df.shape


test_data_df.shape


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# グループ情報: 10個のグループ（地域IDなど）が存在すると仮定
# GroupKFoldは、このグループが訓練とテストで重複しないように分割します。
groups = np.repeat(np.arange(10), 10) # [0, 0, ..., 9, 9] (10回ずつ)

# 未知データ (X_unknown): 最終提出用のデータ
X_unknown = pd.DataFrame(np.random.rand(50, 5), columns=[f'feature_{i}' for i in range(5)])

# 評価スコアを保存するリスト
scores = []














data_string = train_data['ID'].unique()

data_string


data_string = """
array(['SC_05', 'RV_08', 'SC_10', 'AL_12', 'SAC_04', 'OC_32', 'SF_03',
       'OC_19', 'OC_25', 'RV_07', 'SD_13', 'SC_02', 'RV_24', 'RV_01',
       'RV_05', 'SB_05', 'OC_34', 'AL_05', 'SD_21', 'SF_11', 'SD_02',
       'SB_14', 'OC_04', 'SB_33', 'OC_11', 'AL_10', 'SF_04', 'OC_14',
       'OC_27', 'SB_13', 'OC_13', 'SF_05', 'SB_27', 'SF_06', 'AL_08',
       'OC_16', 'SF_01', 'OC_01', 'LA_07', 'SAC_03', 'SC_13', 'RV_12',
       'SB_04', 'RV_23', 'SB_15', 'RV_25', 'SF_09', 'AL_01', 'OC_09',
       'RV_18', 'SD_03', 'SO_10', 'OC_31', 'SD_11', 'RV_27', 'RV_28',
       'SC_06', 'SC_08', 'AL_04', 'RV_06', 'OC_12', 'RV_20', 'OC_08',
       'SAC_05', 'AL_15', 'LA_04', 'SF_10', 'OC_30', 'SD_14', 'SO_05',
       'SB_30', 'SC_09', 'RV_03', 'OC_21', 'RV_11', 'SC_07', 'SD_18',
       'SO_08', 'SD_12', 'SO_06', 'OC_05', 'RV_22', 'SB_39', 'SB_03',
       'OC_03', 'OC_23', 'SB_35', 'OC_07', 'SC_03', 'SC_04', 'OC_24',
       'RV_14', 'SB_40', 'SAC_06', 'LA_05', 'SC_15', 'OC_02', 'RV_04',
       'SO_02', 'SB_20', 'RV_17', 'SF_08', 'SB_16', 'SAC_01', 'OC_17',
       'SD_08', 'SF_02', 'RV_13', 'SC_11', 'LA_13', 'SD_06', 'SD_10',
       'SD_05', 'LA_11', 'SB_02', 'SB_17', 'SB_31', 'SD_17', 'LA_03',
       'AL_09', 'LA_06', 'SD_16', 'LA_12', 'OC_15', 'LA_08', 'AL_07',
       'RV_16', 'OC_20', 'LA_14', 'RV_10'], dtype=object)
"""


import re
# 'SC_05' 形式の文字列だけを抽出する
data_list = re.findall(r"'([A-Z]+_[0-9]+)'", data_string)




# 2. リスト内包表記を使ってプレフィックスを抽出
# item.split('_')[0] で '_’の前を取得
prefixes = [item.split('_')[0] for item in data_list]

# 3. set() に変換して重複を排除
unique_prefixes = set(prefixes)

print("--- プレフィックスのセット ---")
print(unique_prefixes)
print(f"\nユニークなプレフィックスの数: {len(unique_prefixes)}")


# 'item_id' 列からプレフィックス（グループID）を抽出する例
# X_train['item_id'] の中身が 'SC_05', 'RV_08' などであると仮定
X['Group_ID'] =  train_data_df['ID']

# グループ情報として Group_ID 列を配列として取得
# groups = X['Group_ID'].values
groups =['OC', 'RV', 'SD', 'SO', 'SC', 'LA', 'SF', 'SB', 'AL', 'SAC']


X.shape


X


# X や y と同じ行数（130行）を持つことを確認
groups = X['Group_ID'].values 
# 確認
print(len(X))       # -> 130
print(len(y))       # -> 130
print(len(groups))  # -> 130 であることを確認


X.shape, y.shape


y


groups


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- データの準備 (GroupKFoldに必要な X, y, groups を定義) ---

# 1. グループID配列の作成 (X, y と同じ 130行分)
# train_data_df['ID'] ('SC_05'など) から郡コード('SC'など) のプレフィックスを抽出します
groups = train_data_df['ID'].str.split('_').str[0].values # -> 130要素の配列

# 2. 特徴量 X と 目的変数 y の定義
# ID列とターゲット列 ('HOMELESS_RATE') は特徴量Xから除外します
X = train_data_df.drop(['ID', 'HOMELESS_RATE'], axis=1) 
y = train_data_df['HOMELESS_RATE']

# 評価スコアを保存するためのリスト
scores = []

# --- GroupKFoldの実行 ---

n_splits = 10
gkf = GroupKFold(n_splits=n_splits)

print(f"--- GroupKFold (n_splits={n_splits}) の処理を開始 ---")

# 2. GroupKFoldのループ処理
# groups=groups が、上で作成した130要素の郡コード配列を使用
for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups)):
    
    # テストセットに含まれる郡コードを確認 (必ず1つだけになるはず)
    test_group = np.unique(groups[test_index])
    
    # データの分割
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 3. モデルの学習と評価
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    scores.append(rmse)
    
    # 評価結果の表示
    print(f"Fold {fold+1}: テスト郡 = {test_group[0]}, RMSE = {rmse:.4f}")

# 4. 最終結果の集計
print("-" * 30)
print(f"【CV結果】全 {n_splits} 回の平均 RMSE: {np.mean(scores):.4f}")


final_model = RandomForestRegressor(random_state=42)

print("--- 全訓練データで最終モデルを学習中 ---")
final_model.fit(train_data_df.drop(['ID', 'HOMELESS_RATE'], axis = 1), train_data_df['HOMELESS_RATE'])
print("学習完了。")

# ----------------------------------------------------
# 2. テストデータの前処理
# ----------------------------------------------------

# !!! 重要 !!!
# 訓練データ（X_train_full）に行ったすべての前処理（欠損値処理、カテゴリ変数エンコーディングなど）
# を、このテストデータ（test_data_df）にも全く同じ方法で適用する必要があります。

# 【例】訓練データとテストデータの前処理を分離する
X_test_processed = test_data_df.drop('ID', axis = 1)

print("テストデータの前処理が完了しました。")

# ----------------------------------------------------
# 3. ターゲットの予測
# ----------------------------------------------------

# 前処理済みのテストデータに対して予測を実行
predictions = final_model.predict(X_test_processed)

# ----------------------------------------------------
# 4. 提出ファイルの作成
# ----------------------------------------------------

# コンペティションの要求に合わせて、ID列と予測値の列を持つDataFrameを作成
submission_df = pd.DataFrame({
    # test_data_df が持つID列名を指定してください (例: 'ID', 'County_ID'など)
    'ID': test_data_df['ID'], 
    'Target': predictions 
})

# CSVファイルとして保存（index=Falseで余計な行番号を出力しない）
submission_df.to_csv('submission.csv', index=False)

print("-" * 30)
print("予測が完了し、submission.csv が作成されました。")
print(f"予測結果（最初の5件）:\n{submission_df.head()}")




