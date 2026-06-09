import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,GroupKFold
from sklearn.metrics import roc_auc_score
from itertools import product
import time
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train_extra=pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
train_extra["id"] = range(len(train_extra))


train


train_extra


train_extra.columns = train_extra.columns.str.replace(' ', '')
train_extra = train_extra[train_extra.columns].copy()
train_extra['rainfall'] = train_extra['rainfall'].map({'no': 0, 'yes': 1})
train_extra['humidity']=train_extra['humidity'].astype(float)
train_extra['cloud']=train_extra['cloud'].astype(float)
train_features=list(train)
train_extra=train_extra[train_features]

train = pd.concat([train, train_extra], axis=0, ignore_index=True)
train = train.drop_duplicates()
train.shape
train


train['year_group'] = train['id']//365 

# 可视化rainfall目标变量的分布（0-1二分类变量）
plt.figure(figsize=(10, 6))
plt.bar([0, 1], [
    (train['rainfall'] == 0).sum(),
    (train['rainfall'] == 1).sum()
], color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Rainfall distribution')
plt.xlabel('Rainfall value')
plt.ylabel('frequency')
plt.xticks([0, 1], ['(0)', '(1)'])
plt.grid(True, alpha=0.3)
plt.show()


def split_by_date(df, dates):
    df_start, df_end = dates
    df = df[(df['date'] >= df_start) & (df['date'] <= df_end)].reset_index(drop=True)
    return df


def create_lag_features(df, columns_to_lag, lag_periods):

    df_copy = df.copy()
    
    for column in columns_to_lag:
        for lag in lag_periods:
            df_copy[f'lag{lag}_{column}'] = df_copy[column].shift(lag)
    
    return df_copy

def create_diff_features(df, columns_to_diff, window_size=5):
    #lag diff features

    df_copy = df.copy()
    

    for col in columns_to_diff:

        for lag in range(1, window_size):

            df_copy[f'{col}_diff_lag{lag}'] = df_copy[col].diff(lag).fillna(0)
    
    return df_copy



def compute_imbalances(df_, columns, prefix = ''):

    df = df_.copy()
    for col1, col2 in combinations(columns, 2):

        # 按字典顺序排序列，确保一致的排序
        col1, col2 = sorted([col1, col2])

        # 直接计算不平衡，无需创建临时差异列
        total = df[col1] + df[col2]
        imbalance_column_name = f'{col1}_{col2}_imb{prefix}'

        # 确保不会除以零
        df[imbalance_column_name] = (df[col1] - df[col2]).divide(total, fill_value=np.nan)

    return df


def create_interaction_features(df, features):

    df = df.copy()


    interaction_features = []

    for i, c1 in enumerate(features):
        for j, c2 in enumerate(features[i+1:]):
            new_feature_name = f"{c1}_{c2}"            
            df[new_feature_name] = df[c1] * df[c2]

            interaction_features.append(new_feature_name)
    
    print(f"There are {len(interaction_features)} interaction features:")
    
    return df

def create_cumsum_features(df, columns_to_compute):
    df_copy = df.copy()
    
    # Group by 'stock_id' and 'date_id' for cumulative sum calculation
    grouped = df_copy.groupby(["month"])
    
    # Calculate cumulative sum for each column within each group
    for column in columns_to_compute:
        cumsum_col_name = f'{column}_cumsum'
        df_copy[cumsum_col_name] = grouped[column].cumsum()
    return df_copy

def convert_day_to_time_features(df):
    df_copy = df.copy()
    
    # 保留原始的day列
    df_copy['original_day'] = df_copy['day']
    
    # 创建连续的day列（1-2190）
    if 'id' in df_copy.columns:
        df_copy['continuous_day'] = df_copy['id'] + 1

    
    return df_copy



def aggregated_features_dic(df):
    global_feats = {}
    
    # 定义要聚合的列
    columns_to_aggregate = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                           'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    
    df['month'] = df['day'] // 30 + 1 
    groupby_cols = ['month']
    
    def q25(x):
        return x.quantile(0.25)
    
    def q75(x):
        return x.quantile(0.75)
    
    # 定义聚合操作
    aggregations = ['mean', 'median', 'std', 'min', 'max', q25, q75]
    
    # 对每列执行聚合
    for column in columns_to_aggregate:
        for agg in aggregations:
            # 定义聚合函数名
            if callable(agg):
                func_name = agg.__name__
            else:
                func_name = agg
            
            # 执行聚合
            agg_series = df.groupby(groupby_cols)[column].agg(agg)
            # 创建新特征名并添加到字典
            new_feature_name = f"{func_name}_{column}"
            global_feats[new_feature_name] = agg_series
    
    # 创建组合特征
    global_feats["ptp_temp"] = df.groupby(groupby_cols)["maxtemp"].max() - df.groupby(groupby_cols)["mintemp"].min()
    global_feats["median_temp"] = df.groupby(groupby_cols)["maxtemp"].median() + df.groupby(groupby_cols)["mintemp"].median()
    global_feats['std_temp']    =  df.groupby(groupby_cols)["maxtemp"].std() + df.groupby(groupby_cols)["mintemp"].std()
 
    
    return global_feats

aggregated_dic = aggregated_features_dic(train)
aggregated_dic = aggregated_features_dic(test)
def map_global(df, dict):
    df_ = df.copy()
    # 确保df有month列
    df_['month'] = df_['day'] // 30 + 1
    
    for key, value in dict.items():
        # 将全局特征映射回原始数据框
        df_[f"global_{key}"] = df_['month'].map(value.to_dict())
    
    return df_



#allcolumns = [ 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
RMV = ['rainfall', 'id','bucket']
FEATURES = [ 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Imbalance feature
columns_tem_imb         = ['mintemp', 'maxtemp','temparature']

columns_weather_imb = ['pressure', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine']
# Lag feature
num_of_feature_lags = 3

feature_lags        = list(range(1,num_of_feature_lags+1))

pure_lag_features = ['pressure', 'humidity','cloud']

diff_lags           = [1, 2, 3, 6, 12, 18, 24]

def features_engineering(df):


    ## temperature related
    df["avg_temp_eng"] = (df["maxtemp"] + df["mintemp"]) / 2
    df['Dewpoint_diff_eng'] = df['temparature'] - df['dewpoint']

    ## sunshine, cloud amount
    df['Sunshine_per_hour_eng'] = df['sunshine'] / 24
    df['Cloud_per_hour_eng'] = df['cloud'] / 24
    df['Cloud_Humidity_ratio_eng'] = df['cloud'] / (df['humidity'] + 1e-5)
    df['Cloud_Sunshine_ratio_eng'] = df['cloud'] / (df['sunshine'] + 1e-5)

    ## wind related
    df['Wind_x_eng'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['Wind_y_eng'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    ## others
    df['sunshine_percentage_eng'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['cloud_percentage_eng'] = df['cloud'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['weather_index_eng'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    df['Temp_Ratio_eng'] = df['temparature'] / df['maxtemp'].max()

    # df['humidity_index'] = df['dewpoint'] / df['maxtemp']
    df['High_Cloud_Cover_eng'] = (df['cloud'] > 60).astype(int)
    df['High_Humidity_eng'] = (df['humidity'] > 75).astype(int)

    # wet-bulb temperature
    def calc_wet_bulb(T, RH):
        return T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) - 4.686035

    df['wet_bulb_temp_eng'] = calc_wet_bulb(df['temparature'], df['humidity'])

    # saturated vapor pressure
    def calc_saturation_vapor_pressure(temp):
        return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))

    df['e_s_temp_eng'] = calc_saturation_vapor_pressure(df['temparature'])
    df['e_s_dewpoint_eng'] = calc_saturation_vapor_pressure(df['dewpoint'])

    # vapor pressure deficit
    df['vapor_pressure_deficit_eng'] = df['e_s_temp_eng'] - df['e_s_dewpoint_eng']

    df.fillna(method='bfill', inplace=True)
    
    return df




def feature_pipeline(df):

    if df.empty:
        return pd.DataFrame()
    
    #--------------- connected ------------
    df = features_engineering(df)
    
    # Imbalance feature
    df = compute_imbalances(df, columns_tem_imb, prefix = '_temp_')
    df = compute_imbalances(df, columns_weather_imb, prefix = '_weather_')
    # eng_features       = [feature for feature in df.columns if "_eng" in feature]
    # imb_features_all   = [feature for feature in df.columns if "_imb_" in feature]
    # imb_features_temp  = [feature for feature in df.columns if "_temp_" in feature]
    # imb_features_weather  = [feature for feature in df.columns if "_weather_" in feature]
    
    # #--------------- connected ------------
    #cumsum_features
    #cumsum_features = eng_features + imb_features_weather +columns_weather_imb
    # df = create_cumsum_features(df, cumsum_features)
    
    
    #df = map_global(df,aggregated_dic)

    
    
    # #interactonFE
    #df = create_interaction_features(df, FEATURES)
    
    # # diff_lag
    #columns_to_difflag     =  eng_features + FEATURES
    #df = create_diff_features(df, columns_to_difflag)
    #df = create_lag_features(df, pure_lag_features,feature_lags)
    # NaN
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ADDFeatures = [c for c in df.columns if c not in ['rainfall', 'id'] and c not in FEATURES]
    print("Done...")
    return df, FEATURES , ADDFeatures


train, FEATURES , ADDFeatures = feature_pipeline(train)
test,  FEATURES , ADDFeatures = feature_pipeline(test)
print(f"Added {len(ADDFeatures)} features pools:")
print(ADDFeatures)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


from scipy.stats import rankdata
ADD = ['day', 'month', 'avg_temp_eng', 'Dewpoint_diff_eng', 'Sunshine_per_hour_eng', 'Cloud_per_hour_eng', 'Cloud_Humidity_ratio_eng', 'Cloud_Sunshine_ratio_eng', 'Wind_x_eng', 'Wind_y_eng', 'sunshine_percentage_eng', 'cloud_percentage_eng', 'weather_index_eng', 'Temp_Ratio_eng', 'High_Cloud_Cover_eng', 'High_Humidity_eng', 'wet_bulb_temp_eng', 'e_s_temp_eng', 'e_s_dewpoint_eng', 'vapor_pressure_deficit_eng', 'maxtemp_mintemp_imb_temp_', 'mintemp_temparature_imb_temp_', 'maxtemp_temparature_imb_temp_', 'pressure_temparature_imb_weather_', 'dewpoint_pressure_imb_weather_', 'humidity_pressure_imb_weather_', 'cloud_pressure_imb_weather_', 'pressure_sunshine_imb_weather_', 'dewpoint_temparature_imb_weather_', 'humidity_temparature_imb_weather_', 'cloud_temparature_imb_weather_', 'sunshine_temparature_imb_weather_', 'dewpoint_humidity_imb_weather_', 'cloud_dewpoint_imb_weather_', 'dewpoint_sunshine_imb_weather_', 'cloud_humidity_imb_weather_', 'humidity_sunshine_imb_weather_', 'cloud_sunshine_imb_weather_']
#skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# 5折交叉验证预测
FOLDS = 6
Gkf = GroupKFold(n_splits=FOLDS) 


oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(Gkf.split(train,groups=train.year_group)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index, FEATURES + ADD]
    y_train = train.loc[train_index, "rainfall"]
    x_valid = train.loc[test_index, FEATURES + ADD]
    y_valid = train.loc[test_index, "rainfall"]
    x_test = test[FEATURES + ADD].copy()

    model = XGBClassifier(
            device="cuda",
            max_depth=1, 
            learning_rate=0.05, 
            n_estimators=300,        
            colsample_bytree=0.9,
            subsample=0.9,
            eval_metric="auc",
            early_stopping_rounds=100,
            alpha=1,
            gamma=0,
            min_child_weight=1,
            random_state=42
    )
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )
    
    # INFER OOF
    oof_xgb[test_index] = model.predict_proba(x_valid)[:,1]
    # INFER TEST
    pred_xgb += model.predict_proba(x_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from sklearn.metrics import roc_auc_score
true = train.rainfall.values
m = roc_auc_score(true, oof_xgb)
print(f"XGBoost CV Score AUC = {m:.6f}")


feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES+ADD,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  
plt.show()


from scipy.stats import rankdata

sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = rankdata( pred_xgb ) 
sub.rainfall = rankdata( sub.rainfall ) / len(sub)

sub['rainfall'] = sub['rainfall']
# 保存结果
sub.to_csv(f"submission_single-xgb.csv",index=False)




