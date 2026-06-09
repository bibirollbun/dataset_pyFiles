import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

rseed = 71


# ----------------------------------------------
# データ読み込み
# ----------------------------------------------
DATA_DIR = "/kaggle/input/forest-cover-type-prediction"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# データ確認用
# print(train)
# print(test)

# 元テーブルからID列と目的変数をコピー
train_id = train["Id"].copy()
test_id  = test["Id"].copy()
train_y  = train["Cover_Type"].copy()


# EDA用にデータフレームのコピーを作成（元のデータフレームを汚さないため）
df_eda = train.copy()

# =============================================================================
# 特徴量の定義と前処理 (EDA用)
# =============================================================================
# 特徴量のグループ分け
numerical_features = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]
wilderness_features = [col for col in df_eda.columns if 'Wilderness_Area' in col]
soil_features = [col for col in df_eda.columns if 'Soil_Type' in col]

# One-Hotエンコードされたカテゴリカル特徴量を元の単一カラムに戻す
# これにより、グループごとの集計や可視化が容易になります。
df_eda['Wilderness_Area'] = df_eda[wilderness_features].idxmax(axis=1)
df_eda['Soil_Type'] = df_eda[soil_features].idxmax(axis=1)


# =============================================================================
# 1. 目的変数（Cover_Type）の分布を確認
# =============================================================================
print("\n--- 1. 目的変数（Cover_Type）の分布 ---")
# ここでは df_eda を使用します
cover_type_counts = df_eda['Cover_Type'].value_counts().reset_index()
cover_type_counts.columns = ['Cover_Type', 'Count']
fig = px.bar(cover_type_counts, x='Cover_Type', y='Count',
            title="目的変数（Cover_Type）の分布",
            color='Cover_Type',
            labels={'Cover_Type': '森林被覆タイプ', 'Count': 'サンプル数'})
fig.show()


# =============================================================================
# 2. 数値特徴量と目的変数の関係
# =============================================================================
print("\n--- 2. 数値特徴量と目的変数の関係 (インタラクティブな箱ひげ図) ---")
for col in numerical_features:
    fig = px.box(df_eda, x='Cover_Type', y=col, color='Cover_Type',
                    title=f"Cover_Typeごとの {col} の分布",
                     labels={'Cover_Type': '森林被覆タイプ'})
    fig.show()


# =============================================================================
# 3. カテゴリ特徴量と目的変数の関係
# =============================================================================
print("\n--- 3. カテゴリ特徴量と目的変数の関係 (インタラクティブな棒グラフ) ---")
# Wilderness_Area vs Cover_Type
fig = px.histogram(df_eda, x='Wilderness_Area', color='Cover_Type',
                    barmode='group',
                    title="Wilderness_AreaごとのCover_Type分布",
                    labels={'Wilderness_Area': '原生自然地域', 'Cover_Type': '森林被覆タイプ'})
fig.show()

# Soil_Type vs Cover_Type (種類が多いため、上位10種に絞って表示)
top_10_soil_types = df_eda['Soil_Type'].value_counts().nlargest(10).index
df_top_soil = df_eda[df_eda['Soil_Type'].isin(top_10_soil_types)]
    
fig = px.histogram(df_top_soil, x='Soil_Type', color='Cover_Type',
                    barmode='group',
                    title="Soil_TypeごとのCover_Type分布 (上位10種)",
                    category_orders={'Soil_Type': top_10_soil_types},
                    labels={'Soil_Type': '土壌タイプ', 'Cover_Type': '森林被覆タイプ'})
fig.update_xaxes(tickangle=45)
fig.show()


# =============================================================================
# 4. 数値特徴量間の相関
# =============================================================================
print("\n--- 4. 数値特徴量間の相関 (インタラクティブなヒートマップ) ---")
correlation_matrix = df_eda[numerical_features].corr()
fig = px.imshow(correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r', # 赤-白-青のカラースケール
                title="数値特徴量間の相関ヒートマップ")
fig.show()


train_x = train.copy().drop(columns=["Cover_Type"])
test_x = test.copy()
# ID列を削除
train_x = train_x.drop(columns=["Id"])
test_x  = test_x.drop(columns=["Id"])


# ----------------------------------------------
# 特徴量作成(特徴量エンジニアリング)
# ----------------------------------------------
train_x["Euclidean_Distance_to_Hydrology"] = np.sqrt(train_x.Horizontal_Distance_To_Hydrology**2 + train_x.Vertical_Distance_To_Hydrology**2)
test_x["Euclidean_Distance_to_Hydrology"] = np.sqrt(test_x.Horizontal_Distance_To_Hydrology**2 + test_x.Vertical_Distance_To_Hydrology**2)
train_x["EVDtH"] = train_x.Elevation - train_x.Vertical_Distance_To_Hydrology
test_x["EVDtH"] = test_x.Elevation - test_x.Vertical_Distance_To_Hydrology
train_x["EHDtH"] = train_x.Elevation - train_x.Horizontal_Distance_To_Hydrology * 0.2
test_x["EHDtH"] = test_x.Elevation - test_x.Horizontal_Distance_To_Hydrology * 0.2
train_x["Hydro_Fire_1"] = train_x.Horizontal_Distance_To_Hydrology + train_x.Horizontal_Distance_To_Fire_Points
test_x["Hydro_Fire_1"] = test_x.Horizontal_Distance_To_Hydrology + test_x.Horizontal_Distance_To_Fire_Points
train_x["Hydro_Fire_2"] = abs(train_x.Horizontal_Distance_To_Hydrology - train_x.Horizontal_Distance_To_Fire_Points)
test_x["Hydro_Fire_2"] = abs(test_x.Horizontal_Distance_To_Hydrology - test_x.Horizontal_Distance_To_Fire_Points)
train_x["Hydro_Road_1"] = abs(train_x.Horizontal_Distance_To_Hydrology + train_x.Horizontal_Distance_To_Roadways)
test_x["Hydro_Road_1"] = abs(test_x.Horizontal_Distance_To_Hydrology + test_x.Horizontal_Distance_To_Roadways)
train_x["Hydro_Road_2"] = abs(train_x.Horizontal_Distance_To_Hydrology - train_x.Horizontal_Distance_To_Roadways)
test_x["Hydro_Road_2"] = abs(test_x.Horizontal_Distance_To_Hydrology - test_x.Horizontal_Distance_To_Roadways)
train_x["Fire_Road_1"] = abs(train_x.Horizontal_Distance_To_Fire_Points + train_x.Horizontal_Distance_To_Roadways)
test_x["Fire_Road_1"] = abs(test_x.Horizontal_Distance_To_Fire_Points + test_x.Horizontal_Distance_To_Roadways)
train_x["Fire_Road_2"] = abs(train_x.Horizontal_Distance_To_Fire_Points - train_x.Horizontal_Distance_To_Roadways)
test_x["Fire_Road_2"] = abs(test_x.Horizontal_Distance_To_Fire_Points - test_x.Horizontal_Distance_To_Roadways)
train_x["EHiElv"] = train_x.Horizontal_Distance_To_Roadways * train_x.Elevation
test_x["EHiElv"] = test_x.Horizontal_Distance_To_Roadways * test_x.Elevation
train_x["EViElv"] = train_x.Vertical_Distance_To_Hydrology * train_x.Elevation
test_x["EViElv"] = test_x.Vertical_Distance_To_Hydrology * test_x.Elevation
# train_x["Soil_Type12_32"] = train_x.Soil_Type32 + train_x.Soil_Type12
# test_x["Soil_Type12_32"] = test_x.Soil_Type32 + test_x.Soil_Type12
# train_x["Soil_Type23_22_32_33"] = train_x.Soil_Type23 + train_x.Soil_Type22 + train.Soil_Type32 + train_x.Soil_Type33
# test_x["Soil_Type23_22_32_33"] = test_x.Soil_Type23 + test_x.Soil_Type22 + test_x.Soil_Type32 + test_x.Soil_Type33
# train_x['binned_elevation'] = np.floor(train_x['Elevation'] / 50.0)
# test_x['binned_elevation'] = np.floor(test_x['Elevation'] / 50.0)
# train_x = train_x.drop([ "Soil_Type7", "Soil_Type15"], axis=1)
# test_x = test_x.drop(["Soil_Type7", "Soil_Type15"], axis=1)

# # 水源、発火点、道路までの水平距離をリストにまとめる
# dist_cols = ["Horizontal_Distance_To_Hydrology", 
#              "Horizontal_Distance_To_Fire_Points", 
#              "Horizontal_Distance_To_Roadways"]
# # 平均値と標準偏差を新しい特徴量として追加
# train_x["mean_dist"] = train_x[dist_cols].mean(axis=1)
# train_x["std_dist"] = train_x[dist_cols].std(axis=1)
# test_x["mean_dist"] = test_x[dist_cols].mean(axis=1)
# test_x["std_dist"] = test_x[dist_cols].std(axis=1)

# # 度をラジアンに変換してからsin, cosを計算
# rad = np.deg2rad(train_x["Aspect"])
# train_x["Aspect_sin"] = np.sin(rad)
# train_x["Aspect_cos"] = np.cos(rad)
# rad = np.deg2rad(test_x["Aspect"])
# test_x["Aspect_sin"] = np.sin(rad)
# test_x["Aspect_cos"] = np.cos(rad)

# # 水平距離が0の場合に備えて、微小な値(1e-6)を足して0除算を回避
# # 勾配 (Vertical / Horizontal)
# train_x["Hydrology_slope"] = train_x.Vertical_Distance_To_Hydrology / (train_x.Horizontal_Distance_To_Hydrology + 1e-6)
# test_x["Hydrology_slope"] = test_x.Vertical_Distance_To_Hydrology / (test_x.Horizontal_Distance_To_Hydrology + 1e-6)
# # 標高と何かの比率 (上で mean_dist を作成した場合)
# train_x["Elevation_over_mean_dist"] = train_x.Elevation / (train_x.mean_dist + 1e-6)
# test_x["Elevation_over_mean_dist"] = test_x.Elevation / (test_x.mean_dist + 1e-6)

train_x["Horizontal_Distance_To_Roadways_Log"] = np.log1p(train_x.Horizontal_Distance_To_Roadways)
test_x["Horizontal_Distance_To_Roadways_Log"] = np.log1p(test_x.Horizontal_Distance_To_Roadways)

# train_x["Road+Fire+Hydro"] = train_x.Horizontal_Distance_To_Roadways + train_x.Horizontal_Distance_To_Fire_Points + train_x.Horizontal_Distance_To_Hydrology
# test_x["Road+Fire+Hydro"] = test_x.Horizontal_Distance_To_Roadways + test_x.Horizontal_Distance_To_Fire_Points + test_x.Horizontal_Distance_To_Hydrology
# train_x["Ele+Road+Fire+Hydro"] = train_x.Elevation + train_x.Horizontal_Distance_To_Roadways + train_x.Horizontal_Distance_To_Fire_Points + train_x.Horizontal_Distance_To_Hydrology
# test_x["Ele+Road+Fire+Hydro"] = test_x.Elevation + test_x.Horizontal_Distance_To_Roadways + test_x.Horizontal_Distance_To_Fire_Points + test_x.Horizontal_Distance_To_Hydrology
train_x['Ele+road'] = train_x.Elevation + train_x.Horizontal_Distance_To_Roadways
test_x['Ele+road'] = test_x.Elevation + test_x.Horizontal_Distance_To_Roadways
train_x['Ele-road'] = train_x.Elevation - train_x.Horizontal_Distance_To_Roadways
test_x['Ele-road'] = test_x.Elevation - test_x.Horizontal_Distance_To_Roadways
train_x['Ele+fire'] = train_x.Elevation + train_x.Horizontal_Distance_To_Fire_Points
test_x['Ele+fire'] = test_x.Elevation + test_x.Horizontal_Distance_To_Fire_Points
train_x['Ele-fire'] = train_x.Elevation - train_x.Horizontal_Distance_To_Fire_Points
test_x['Ele-fire'] = test_x.Elevation - test_x.Horizontal_Distance_To_Fire_Points
train_x['Ele+hydro'] = train_x.Elevation + train_x.Horizontal_Distance_To_Hydrology
test_x['Ele+hydro'] = test_x.Elevation + test_x.Horizontal_Distance_To_Hydrology
train_x['Ele-hydro'] = train_x.Elevation - train_x.Horizontal_Distance_To_Hydrology
test_x['Ele-hydro'] = test_x.Elevation - test_x.Horizontal_Distance_To_Hydrology

# train_x['asp+3am'] = train_x.Aspect + train_x.Hillshade_3pm
# test_x['asp+3am'] = test_x.Aspect + test_x.Hillshade_3pm


# Mapping soil type to ELU code
ELU_CODE = {
    1:2702,2:2703,3:2704,4:2705,5:2706,6:2717,7:3501,8:3502,9:4201,
    10:4703,11:4704,12:4744,13:4758,14:5101,15:5151,16:6101,17:6102,
    18:6731,19:7101,20:7102,21:7103,22:7201,23:7202,24:7700,25:7701,
    26:7702,27:7709,28:7710,29:7745,30:7746,31:7755,32:7756,33:7757,
    34:7790,35:8703,36:8707,37:8708,38:8771,39:8772,40:8776
}


# Encode soil type ordinally
def categorical_encoding(input_df):
    data = input_df.copy()
    data['Soil_Type'] = 0
    for i in range(1,41):
        data['Soil_Type'] += i*data[f'Soil_Type{i}']
    return data

# Encode soil type
train_x = categorical_encoding(train_x)
test_x = categorical_encoding(test_x)

# Original soil features
soil_features = [f'Soil_Type{i}' for i in range(1,41)]

def climatic_zone(input_df):
    df = input_df.copy()
    df['Climatic_Zone'] = input_df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[0])
    )
    return df

# Climatic Zone
train_x = climatic_zone(train_x)
test_x = climatic_zone(test_x)

def geologic_zone(input_df):
    df = input_df.copy()
    df['Geologic_Zone'] = input_df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[1])
    )
    return df

# Geologic Zone
train_x = geologic_zone(train_x)
test_x = geologic_zone(test_x)

def surface_cover(input_df):
    # Group IDs
    no_desc = [7,8,14,15,16,17,19,20,21,23,35]
    stony = [6,12]
    very_stony = [2,9,18,26]
    extremely_stony = [1,22,24,25,27,28,29,30,31,32,33,34,36,37,38,39,40]
    rubbly = [3,4,5,10,11,13]

    # Create dictionary
    surface_cover = {i:0 for i in no_desc}
    surface_cover.update({i:1 for i in stony})
    surface_cover.update({i:2 for i in very_stony})
    surface_cover.update({i:3 for i in extremely_stony})
    surface_cover.update({i:4 for i in rubbly})
    
    # Create Feature
    df = input_df.copy()
    df['Surface_Cover'] = input_df['Soil_Type'].apply(
        lambda x: surface_cover[x]
    )
    return df

# Surface Cover
train_x = surface_cover(train_x)
test_x = surface_cover(test_x)

# def rock_size(input_df):
    
#     # Group IDs
#     no_desc = [7,8,14,15,16,17,19,20,21,23,35]
#     stones = [1,2,6,9,12,18,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40]
#     boulders = [22]
#     rubble = [3,4,5,10,11,13]

#     # Create dictionary
#     rock_size = {i:0 for i in no_desc}
#     rock_size.update({i:1 for i in stones})
#     rock_size.update({i:2 for i in boulders})
#     rock_size.update({i:3 for i in rubble})
    
#     df = input_df.copy()
#     df['Rock_Size'] = input_df['Soil_Type'].apply(
#         lambda x: rock_size[x]
#     )
#     return df

# # Rock Size
# train_x = rock_size(train_x)
# test_x = rock_size(test_x)

# def soiltype_interactions(data):
#     df = data.copy()
            
#     # Important Soil Types
#     df['Soil_12_32'] = df['Soil_Type32'] + df['Soil_Type12']
#     df['Soil_Type23_22_32_33'] = df['Soil_Type23'] + df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']
    
#     # Soil Type Interactions
#     df['Soil29_Area1'] = df['Soil_Type29'] + df['Wilderness_Area1']
#     df['Soil3_Area4'] = df['Wilderness_Area4'] + df['Soil_Type3']
    
#     #  New Feature Interactions
#     df['Climate_Area2'] = df['Wilderness_Area2']*df['Climatic_Zone'] 
#     df['Climate_Area4'] = df['Wilderness_Area4']*df['Climatic_Zone'] 
#     df['Rock_Area1'] = df['Wilderness_Area1']*df['Rock_Size']    
#     df['Rock_Area3'] = df['Wilderness_Area3']*df['Rock_Size']  
#     df['Surface_Area1'] = df['Wilderness_Area1']*df['Surface_Cover'] 
#     df['Surface_Area2'] = df['Wilderness_Area2']*df['Surface_Cover']   
#     df['Surface_Area4'] = df['Wilderness_Area4']*df['Surface_Cover'] 
    
#     # Fill NA
#     df.fillna(0, inplace = True)
    
#     return df
    
# # Soiltype Interactions
# train_x = soiltype_interactions(train_x)
# test_x = soiltype_interactions(test_x)

# Drop original soil features
train.drop(columns = soil_features, inplace = True)
test.drop(columns = soil_features, inplace = True)


# ------------------------------
# XGBoostの学習・推論・submit（ベースライン）
# ------------------------------
# 学習用にクラスラベルを修正（０〜６に対応させる）
train_y = train_y - 1

model = XGBClassifier(objective="multi:softmax", num_class=7, random_state=rseed)
model.fit(train_x, train_y)
preds = model.predict(test_x)

print("\n--- 学習済みモデルの性能評価 ---")
train_preds = model.predict(train_x)
train_preds_proba = model.predict_proba(train_x)
accuracy = accuracy_score(train_y, train_preds)
logloss = log_loss(train_y, train_preds_proba)
print(f"訓練データに対する正解率 (Accuracy): {accuracy:.4f}")
print(f"訓練データに対するLogLoss: {logloss:.4f}")


# 提出用にクラスラベルを修正（１〜７に対応させる）
preds = preds + 1

submission = pd.DataFrame({
    "Id": test_id,
    "Cover_Type":  preds
})

submission.to_csv("submission.csv", index=False)

