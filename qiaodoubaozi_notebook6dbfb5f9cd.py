import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from geopy.distance import geodesic

# 加载数据
df_train = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv")

# 检查缺失值
missing = df_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)

def handle_data(df, drop_missing_threshold=0.3):
    def preprocess(train):
        train[['street', 'city', 'postcode']] = train['fullAddress'].str.rsplit(', ', expand = True, n = 2)
        regions_map = {
            'Leytonstone':1, 'Walthamstow':1, 'Leyton':1, 'Stratford':1, 'Chingford':1,
            'Forest Gate':1, 'Woodford Green':1, # East London
            'London':2, # Central London
            'Southgate':3, 'Wembley':3,'Edmonton':3, 'Palmers Green':3, # North London
            'Blackheath':4, 'Woolwich':4, 'Kidbrooke':4, 'Charlton':4, 'Abbey Wood':4,
            'Greenwich':4, 'Eltham':4,  'Deptford':4, # South East London
            'Wimbledon':5, 'Raynes Park':5, 'Colliers Wood':5, # South West London
            'Acton':6, 'West Ealing':6, 'Ealing':6, 'Hanwell':6, 'Chiswick':6, 'Park Royal':6, # West London
            'Plumstead':7,'Bromley':7 # South London
            }
        train['affluence'] = train['city'].map(regions_map)
        
        outcode_map = {
            'W1B': 4, 'W1C': 4, 'W1D': 4, 'W1F': 4, 'W1G': 4, 'W1H': 4, 'W1J': 4, 'W1K': 4, 'W1S': 4, 'W1T': 4,
            'W1U': 4, 'W1W': 4, 'SW1A': 4, 'SW1E': 4, 'SW1H': 4, 'SW1P': 4, 'SW1V': 4, 'SW1W': 4, 'SW1X': 4, 'SW1Y': 4,
            'WC1A': 4, 'WC1B': 4, 'WC1E': 4, 'WC1H': 4, 'WC1N': 4, 'WC1R': 4, 'WC1V': 4, 'WC1X': 4,'WC2A': 4, 'WC2B': 4,
            'WC2E': 4, 'WC2H': 4, 'WC2N': 4, 'WC2R': 4, 'W10':4, # Most Expensive (Prime Central London)
            'EC1A': 3, 'EC1M': 3, 'EC1N': 3, 'EC1R': 3, 'EC1V': 3, 'EC1Y': 3, 'EC2A': 3, 'EC2M': 3, 'EC2N': 3, 'EC2R': 3,
            'EC2V': 3, 'EC2Y': 3, 'EC3A': 3, 'EC3M': 3, 'EC3N': 3, 'EC3R': 3, 'EC3V': 3, 'EC4A': 3, 'EC4M': 3, 'EC4R': 3,
            'EC4V': 3, 'EC4Y': 3, 'SW3': 3, 'SW5': 3, 'SW6': 3, 'SW7': 3, 'SW10': 3, 'SW11': 3, 'W2': 3, 'W8': 3,
            'W9': 3, 'W11': 3, 'W14': 3, # Expensive (Central London and Prime Areas)
            'N1': 2, 'N2': 2, 'N3': 2, 'N4': 2, 'N5': 2, 'N6': 2, 'N7': 2, 'N8': 2, 'N10': 2, 'N11': 2,
            'N12': 2, 'N13': 2, 'N14': 2, 'N15': 2, 'N16': 2, 'N19': 2, 'N20': 2, 'N21': 2, 'N22': 2,'NW1': 2,
            'NW2': 2, 'NW3': 2, 'NW5': 2, 'NW6': 2, 'NW8': 2, 'NW10': 2, 'NW11': 2, 'SE1': 2, 'SE10': 2, 'SE11': 2,
            'SE15': 2, 'SE16': 2, 'SE21': 2, 'SE22': 2, 'SE24': 2, 'SW2': 2, 'SW4': 2, 'SW8': 2, 'SW9': 2, 'SW12': 2,
            'SW13': 2, 'SW14': 2, 'SW15': 2, 'SW16': 2, 'SW17': 2, 'SW18': 2, 'SW19': 2, 'SW20': 2, 'W3': 2, 'W4': 2,
            'W5': 2, 'W6': 2, 'W7': 2, 'W12': 2, 'W13': 2, # Moderately Expensive (Outer Central London and Suburban Prime Areas)
            'E1': 1, 'E2': 1, 'E3': 1, 'E4': 1, 'E5': 1, 'E6': 1, 'E7': 1, 'E8': 1, 'E9': 1, 'E10': 1,
            'E11': 1, 'E12': 1, 'E13': 1, 'E14': 1, 'E15': 1, 'E16': 1, 'E17': 1, 'E18': 1, 'E1W': 1,'IG8': 1,
            'N9': 1, 'N17': 1, 'N18': 1, 'NW4': 1, 'NW7': 1, 'NW9': 1,'SE2': 1, 'SE3': 1, 'SE4': 1, 'SE5': 1,
            'SE6': 1, 'SE7': 1, 'SE8': 1, 'SE9': 1, 'SE12': 1, 'SE13': 1, 'SE14': 1, 'SE17': 1, 'SE18': 1, 'SE19': 1,
            'SE20': 1, 'SE23': 1, 'SE25': 1, 'SE26': 1, 'SE27': 1, 'SE28': 1 # Less Expensive (Outer London and Suburban Areas)
            }
        
        train['urban'] = train['outcode'].map(outcode_map)
        
        city_center = (51.5074, -0.1278)  # London
        train['distance_to_center'] = train.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), city_center).km, axis=1
        )
        
        train['rooms_per_sqM'] = (train['bedrooms'] + train['livingRooms'] + train['bathrooms']) / train['floorAreaSqM']
        
        train['sale_month_sin'] = np.sin(2 * np.pi * train['sale_month'] / 12)
        train['sale_month_cos'] = np.cos(2 * np.pi * train['sale_month'] / 12)
        
        train['age_of_property'] = 2025 - train['sale_year']
        return train
    
    df = preprocess(df)
    
    # 删除缺失率过高的特征
    missing_ratio = df.isnull().mean()
    high_missing_features = missing_ratio[missing_ratio > drop_missing_threshold].index
    df = df.drop(columns=high_missing_features)
    print(f"已删除高缺失特征: {list(high_missing_features)}")

    # 填充剩余缺失值
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # 数值型特征：用中位数填充（更抗异常值）
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median() if df[col].skew() > 1 else df[col].mean()
                # df[col].fillna(fill_value, inplace=True)
                df[col] = df[col].fillna(fill_value)
                print(f"数值型特征 [{col}] 填充值: {fill_value:.2f}")
            # 类别型特征：用众数填充
            else:
                mode_value = df[col].mode()[0]
                # df[col].fillna(mode_value, inplace=True)
                df[col] = df[col].fillna(mode_value)
                print(f"类别型特征 [{col}] 填充众数: {mode_value}")
                
    
    df['postcode'] = df['postcode'].str.split(' ').str[1]
    
    # 标签编码
    labelencoder = LabelEncoder()
    
    # 获取所有 object 类型列（排除 tenure）
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'tenure' in object_cols:
        object_cols.remove('tenure')  # 确保 tenure 列不被标签编码
    
    # 对其他 object 列执行标签编码
    for col in object_cols:
        df[col] = labelencoder.fit_transform(df[col])
    
    # 对 tenure 列执行独热编码（假设 tenure 是分类列）
    df = pd.get_dummies(df, columns=['tenure'], prefix='tenure')

    print(df.head())
    return df

# 执行处理
df_train = handle_data(df_train, drop_missing_threshold=0.3)

# 保存为CSV文件（无索引列）
df_train.to_csv("pre.csv", index=False)

features = [
            # 'fullAddress', 
            'postcode',
            # 'country',  # 都相同
            'outcode',
            'latitude',
            'longitude',
            'bathrooms',
            'bedrooms',
            'floorAreaSqM',
            'livingRooms',
            # 'tenure',
            'tenure_Feudal',
            'tenure_Freehold',
            'tenure_Leasehold',
            'tenure_Shared',
            'propertyType',
            'currentEnergyRating',
            'sale_month',
            'sale_year',
            'age_of_property','sale_month_cos', 'sale_month_sin', 'distance_to_center','urban','affluence'
]



# 选择特征用于训练模型
X = df_train[features]
y = df_train['price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

# # XGBoost模型
# xg_reg = xgb.XGBRegressor(objective ='reg:absoluteerror', colsample_bytree = 0.3, learning_rate = 0.3,
#                 max_depth = 5, alpha = 10, n_estimators = 300)

# # 训练模型
# xg_reg.fit(X_train, y_train)

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 5,  # 限制树深度
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1
}

# 使用早停法
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_test, y_test)
model = lgb.train(params, lgb_train, 
                 valid_sets=[lgb_val],
                 callbacks=[lgb.early_stopping(100)])

# 预测
preds = model.predict(X_test)

# 评估 MAE
mae = mean_absolute_error(y_test, preds)
print("MAE: %f" % (mae))

# 定义参数网格
# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [50, 100, 200],
#     'num_leaves': [10, 20, 30],
#     'max_depth': [5, 10, 15],
#     'min_data_in_leaf': [10, 20, 30],
#     'feature_fraction': [0.6, 0.8, 1.0],
#     'bagging_fraction': [0.6, 0.8, 1.0],
#     'lambda_l1': [0.1, 0.2, 0.3],
#     'lambda_l2': [0.1, 0.2, 0.3]
# }

# # 初始化 LGBMRegressor
# lgb_model = lgb.LGBMRegressor(objective='regression', metric='mae', boosting_type='gbdt', verbose=-1)

# # 初始化 GridSearchCV
# grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# # 进行网格搜索
# grid_search.fit(X_train, y_train)

# # 输出最优参数
# print("Best parameters found: ", grid_search.best_params_)

# # 使用最优参数训练模型
# best_model = grid_search.best_estimator_
# best_model.fit(X_train, y_train)

# # 预测
# preds = best_model.predict(X_test)

# # 评估 MAE
# mae = mean_absolute_error(y_test, preds)
# print("MAE: %f" % (mae))


# 加载测试集
df_test = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv")

# 执行处理
df_test = handle_data(df_test, drop_missing_threshold=0.3)
    
# 预测
X_test_external = df_test[features]
test_preds = model.predict(X_test_external)

submission = pd.DataFrame({
    "ID": df_test["ID"],
    "price": test_preds
})

# 保存为CSV文件（无索引列）
submission.to_csv("submission.csv", index=False)
print('submission has done')

