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


sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
train_data=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



sample_submission



train_data


test_data


'''
import matplotlib.pyplot as plt
import seaborn as sns

# 查看每个数值型特征的分布
numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 查看每个分类型特征的分布
categorical_columns = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train_data[col])
    plt.title(f'Distribution of {col}')
    plt.show()
    '''


import pandas as pd
import numpy as np
from scipy import stats

def calculate_distribution_metrics(dataframe):
    """
    计算数据集中每个特征的分布指标
    
    参数:
    dataframe (pd.DataFrame): 输入的数据集
    
    返回:
    metrics_dict (dict): 每个特征的分布指标字典
    """
    metrics_dict = {}
    
    for column in dataframe.columns:
        data = dataframe[column].dropna().values  # 去除缺失值
        
        # 集中趋势指标
        mean = np.mean(data) if len(data) > 0 else np.nan
        median = np.median(data) if len(data) > 0 else np.nan
        
        # 处理 mode
        if len(data) > 0:
            mode_result = stats.mode(data)
            if isinstance(mode_result[0], np.ndarray):
                mode = mode_result[0][0]
            else:
                mode = mode_result[0]
        else:
            mode = np.nan
        
        # 离散程度指标
        range_val = np.ptp(data) if len(data) > 0 else np.nan
        iqr = stats.iqr(data) if len(data) > 0 else np.nan
        variance = np.var(data) if len(data) > 0 else np.nan
        std_dev = np.std(data) if len(data) > 0 else np.nan
        cv = std_dev / mean if (len(data) > 0 and mean != 0) else np.nan
        
        # 分布形状指标
        skewness = stats.skew(data) if len(data) > 0 else np.nan
        kurtosis = stats.kurtosis(data) if len(data) > 0 else np.nan
        
        # 百分位数和四分位数
        percentiles = np.percentile(data, [25, 50, 75]) if len(data) > 0 else [np.nan, np.nan, np.nan]
        
        # 存储结果
        metrics_dict[column] = {
            'mean': mean,
            'median': median,
            'mode': mode,
            'range': range_val,
            'iqr': iqr,
            'variance': variance,
            'std_dev': std_dev,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'percentiles_25%': percentiles[0],
            'percentiles_50%': percentiles[1],
            'percentiles_75%': percentiles[2]
        }
    
    return metrics_dict

def save_metrics_to_file(metrics_dict, txt_filename="distribution_metrics.txt", csv_filename="distribution_metrics.csv"):
    """
    将分布指标保存到文本文件和CSV文件中
    
    参数:
    metrics_dict (dict): 每个特征的分布指标字典
    txt_filename (str): 输出的文本文件名
    csv_filename (str): 输出的CSV文件名
    """
    # 保存到文本文件
    with open(txt_filename, "w") as txt_file:
        for feature, metrics in metrics_dict.items():
            txt_file.write(f"Feature: {feature}\n")
            txt_file.write(f"  Mean: {metrics['mean']:.2f}\n")
            txt_file.write(f"  Median: {metrics['median']:.2f}\n")
            txt_file.write(f"  Mode: {metrics['mode']:.2f}\n")
            txt_file.write(f"  Range: {metrics['range']:.2f}\n")
            txt_file.write(f"  IQR: {metrics['iqr']:.2f}\n")
            txt_file.write(f"  Variance: {metrics['variance']:.2f}\n")
            txt_file.write(f"  Std Dev: {metrics['std_dev']:.2f}\n")
            txt_file.write(f"  CV: {metrics['cv']:.2f}\n")
            txt_file.write(f"  Skewness: {metrics['skewness']:.2f}\n")
            txt_file.write(f"  Kurtosis: {metrics['kurtosis']:.2f}\n")
            txt_file.write(f"  Percentiles (25%): {metrics['percentiles_25%']:.2f}\n")
            txt_file.write(f"  Percentiles (50%): {metrics['percentiles_50%']:.2f}\n")
            txt_file.write(f"  Percentiles (75%): {metrics['percentiles_75%']:.2f}\n")
            txt_file.write("\n")
    
    # 保存到CSV文件
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.to_csv(csv_filename)



calculate_distribution_metrics(train_data)


train_data


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 假设 train_data 和 test_data 已经被加载


# 确保 winddirection 是数值型
train_data['winddirection'] = train_data['winddirection'].astype(float)
test_data['winddirection'] = test_data['winddirection'].astype(float)
print(1)

# 处理缺失值（如果有）
train_data['winddirection'].fillna(train_data['winddirection'].mean(), inplace=True)
test_data['winddirection'].fillna(test_data['winddirection'].mean(), inplace=True)
print(2)
'''
# 对day进行归一化处理
scaler_day = MinMaxScaler()
train_data['day'] = scaler_day.fit_transform(train_data[['day']])
test_data['day'] = scaler_day.transform(test_data[['day']])
'''
'''
# 对pressure进行标准化处理
scaler_pressure = StandardScaler()
train_data['pressure'] = scaler_pressure.fit_transform(train_data[['pressure']])
test_data['pressure'] = scaler_pressure.transform(test_data[['pressure']])
'''
# 对maxtemp进行对数转换和归一化处理
train_data['maxtemp'] = np.clip(train_data['maxtemp'], a_min=0, a_max=None)  # 确保非负
test_data['maxtemp'] = np.clip(test_data['maxtemp'], a_min=0, a_max=None)
'''
train_data['maxtemp'] = np.log1p(train_data['maxtemp'])
test_data['maxtemp'] = np.log1p(test_data['maxtemp'])
scaler_maxtemp = MinMaxScaler()
train_data['maxtemp'] = scaler_maxtemp.fit_transform(train_data[['maxtemp']])
test_data['maxtemp'] = scaler_maxtemp.transform(test_data[['maxtemp']])
'''
# 对temparature进行对数转换和归一化处理
train_data['temparature'] = np.clip(train_data['temparature'], a_min=0, a_max=None)
test_data['temparature'] = np.clip(test_data['temparature'], a_min=0, a_max=None)
'''
train_data['temparature'] = np.log1p(train_data['temparature'])
test_data['temparature'] = np.log1p(test_data['temparature'])
scaler_temparature = MinMaxScaler()
train_data['temparature'] = scaler_temparature.fit_transform(train_data[['temparature']])
test_data['temparature'] = scaler_temparature.transform(test_data[['temparature']])
'''
# 对mintemp进行对数转换和归一化处理
train_data['mintemp'] = np.clip(train_data['mintemp'], a_min=0, a_max=None)
test_data['mintemp'] = np.clip(test_data['mintemp'], a_min=0, a_max=None)
'''
train_data['mintemp'] = np.log1p(train_data['mintemp'])
test_data['mintemp'] = np.log1p(test_data['mintemp'])
scaler_mintemp = MinMaxScaler()
train_data['mintemp'] = scaler_mintemp.fit_transform(train_data[['mintemp']])
test_data['mintemp'] = scaler_mintemp.transform(test_data[['mintemp']])
'''
# 对dewpoint进行对数转换和归一化处理
train_data['dewpoint'] = np.clip(train_data['dewpoint'], a_min=0, a_max=None)
test_data['dewpoint'] = np.clip(test_data['dewpoint'], a_min=0, a_max=None)
'''
train_data['dewpoint'] = np.log1p(train_data['dewpoint'])
test_data['dewpoint'] = np.log1p(test_data['dewpoint'])
scaler_dewpoint = MinMaxScaler()
train_data['dewpoint'] = scaler_dewpoint.fit_transform(train_data[['dewpoint']])
test_data['dewpoint'] = scaler_dewpoint.transform(test_data[['dewpoint']])
'''

'''
# 对humidity进行标准化处理
scaler_humidity = StandardScaler()
train_data['humidity'] = scaler_humidity.fit_transform(train_data[['humidity']])
test_data['humidity'] = scaler_humidity.transform(test_data[['humidity']])
'''

# 对cloud进行对数转换和归一化处理
train_data['cloud'] = np.clip(train_data['cloud'], a_min=0, a_max=None)
test_data['cloud'] = np.clip(test_data['cloud'], a_min=0, a_max=None)
'''
train_data['cloud'] = np.log1p(train_data['cloud'])
test_data['cloud'] = np.log1p(test_data['cloud'])
scaler_cloud = MinMaxScaler()
train_data['cloud'] = scaler_cloud.fit_transform(train_data[['cloud']])
test_data['cloud'] = scaler_cloud.transform(test_data[['cloud']])
'''
# 对sunshine进行对数转换和归一化处理
train_data['sunshine'] = np.clip(train_data['sunshine'], a_min=0, a_max=None)
test_data['sunshine'] = np.clip(test_data['sunshine'], a_min=0, a_max=None)
'''
train_data['sunshine'] = np.log1p(train_data['sunshine'])
test_data['sunshine'] = np.log1p(test_data['sunshine'])
scaler_sunshine = MinMaxScaler()
train_data['sunshine'] = scaler_sunshine.fit_transform(train_data[['sunshine']])
test_data['sunshine'] = scaler_sunshine.transform(test_data[['sunshine']])
'''
# 对winddirection进行正弦和余弦转换
train_data['winddirection_rad'] = np.radians(train_data['winddirection'])
test_data['winddirection_rad'] = np.radians(test_data['winddirection'])

# 计算正弦和余弦值
train_data['winddirection_sin'] = np.sin(train_data['winddirection_rad'])
train_data['winddirection_cos'] = np.cos(train_data['winddirection_rad'])

test_data['winddirection_sin'] = np.sin(test_data['winddirection_rad'])
test_data['winddirection_cos'] = np.cos(test_data['winddirection_rad'])

# 删除中间列（可选）
'''
train_data = train_data.drop(columns=['winddirection_rad'])
test_data = test_data.drop(columns=['winddirection_rad'])
'''

# 打印转换后的结果
print(train_data[['winddirection_sin', 'winddirection_cos']].head())
print(test_data[['winddirection_sin', 'winddirection_cos']].head())

# 对windspeed进行对数转换和归一化处理
train_data['windspeed'] = np.clip(train_data['windspeed'], a_min=0, a_max=None)
test_data['windspeed'] = np.clip(test_data['windspeed'], a_min=0, a_max=None)
'''
train_data['windspeed'] = np.log1p(train_data['windspeed'])
test_data['windspeed'] = np.log1p(test_data['windspeed'])
scaler_windspeed = MinMaxScaler()
train_data['windspeed'] = scaler_windspeed.fit_transform(train_data[['windspeed']])
'''


import pandas as pd
import numpy as np

# 假设 train_data 和 test_data 是已经加载的数据集
# 如果还没有加载数据，可以使用以下代码加载
# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')

def assign_season(day):
    # 假设 day 是一年中的第几天（从1到365或366）
    if 1 <= day <= 31:  # 1月
        return 'winter'
    elif 32 <= day <= 59:  # 2月
        return 'winter'
    elif 60 <= day <= 90:  # 3月
        return 'spring'
    elif 91 <= day <= 120:  # 4月
        return 'spring'
    elif 121 <= day <= 151:  # 5月
        return 'spring'
    elif 152 <= day <= 181:  # 6月
        return 'summer'
    elif 182 <= day <= 212:  # 7月
        return 'summer'
    elif 213 <= day <= 243:  # 8月
        return 'summer'
    elif 244 <= day <= 273:  # 9月
        return 'autumn'
    elif 274 <= day <= 303:  # 10月
        return 'autumn'
    elif 304 <= day <= 334:  # 11月
        return 'autumn'
    else:  # 12月
        return 'winter'
        
# 为训练集和测试集创建季节特征
train_data['season'] = train_data['day'].apply(assign_season)
test_data['season'] = test_data['day'].apply(assign_season)

# 对训练集和测试集的季节特征分别进行独热编码，指定 dtype 为 int
# 确保独热编码的列名一致
seasons = ['spring', 'summer', 'autumn', 'winter']
train_season_dummies = pd.get_dummies(train_data['season'], prefix='season', dtype=int)
test_season_dummies = pd.get_dummies(test_data['season'], prefix='season', dtype=int)

# 检查测试集中是否缺少某些季节的独热编码列（如果训练集中有而测试集中没有的季节）
for season in seasons:
    column_name = f'season_{season}'
    if column_name not in test_season_dummies.columns:
        test_season_dummies[column_name] = 0

# 检查训练集中是否缺少某些季节的独热编码列（如果测试集中有而训练集中没有的季节）
for season in seasons:
    column_name = f'season_{season}'
    if column_name not in train_season_dummies.columns:
        train_season_dummies[column_name] = 0

# 确保列的顺序一致
train_season_dummies = train_season_dummies[sorted(train_season_dummies.columns)]
test_season_dummies = test_season_dummies[sorted(test_season_dummies.columns)]

# 将独热编码后的季节特征合并到训练集和测试集中
train_data = pd.concat([train_data, train_season_dummies], axis=1)
test_data = pd.concat([test_data, test_season_dummies], axis=1)

# 删除原始的季节列
train_data.drop('season', axis=1, inplace=True)
test_data.drop('season', axis=1, inplace=True)

# 查看数据集的前几行，确认新特征是否添加成功
print("训练集前5行：")
print(train_data.head())
print("\n测试集前5行：")
print(test_data.head())


train_data



'''
import matplotlib.pyplot as plt
import seaborn as sns

# 查看每个数值型特征的分布
numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 查看每个分类型特征的分布
categorical_columns = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train_data[col])
    plt.title(f'Distribution of {col}')
    plt.show()
    '''


train_data


import pandas as pd
import numpy as np
from scipy import stats

def calculate_distribution_metrics(dataframe):
    """
    计算数据集中每个特征的分布指标
    
    参数:
    dataframe (pd.DataFrame): 输入的数据集
    
    返回:
    metrics_dict (dict): 每个特征的分布指标字典
    """
    metrics_dict = {}
    
    for column in dataframe.columns:
        # 确保数据是数值型
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            data = dataframe[column].dropna().values  # 去除缺失值
            
            # 集中趋势指标
            mean = np.mean(data) if len(data) > 0 else np.nan
            median = np.median(data) if len(data) > 0 else np.nan
            
            # 处理 mode
            if len(data) > 0:
                mode_result = stats.mode(data)
                if isinstance(mode_result[0], np.ndarray):
                    mode = mode_result[0][0]
                else:
                    mode = mode_result[0]
            else:
                mode = np.nan
            
            # 离散程度指标
            range_val = np.ptp(data) if len(data) > 0 else np.nan
            iqr = stats.iqr(data) if len(data) > 0 else np.nan
            variance = np.var(data) if len(data) > 0 else np.nan
            std_dev = np.std(data) if len(data) > 0 else np.nan
            cv = std_dev / mean if (len(data) > 0 and mean != 0) else np.nan
            
            # 分布形状指标
            skewness = stats.skew(data) if len(data) > 0 else np.nan
            kurtosis = stats.kurtosis(data) if len(data) > 0 else np.nan
            
            # 百分位数和四分位数
            percentiles = np.percentile(data, [25, 50, 75]) if len(data) > 0 else [np.nan, np.nan, np.nan]
            
            # 存储结果
            metrics_dict[column] = {
                'mean': mean,
                'median': median,
                'mode': mode,
                'range': range_val,
                'iqr': iqr,
                'variance': variance,
                'std_dev': std_dev,
                'cv': cv,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'percentiles_25%': percentiles[0],
                'percentiles_50%': percentiles[1],
                'percentiles_75%': percentiles[2]
            }
        else:
            print(f"Feature {column} is not numeric, skipping...")
    
    return metrics_dict

def print_distribution_metrics(metrics_dict):
    """
    打印数据分布评价指标
    
    参数:
    metrics_dict (dict): 每个特征的分布评价指标字典
    """
    for feature, metrics in metrics_dict.items():
        print(f"Feature: {feature}")
        print(f"  Mean: {metrics['mean']:.2f}")
        print(f"  Median: {metrics['median']:.2f}")
        print(f"  Mode: {metrics['mode']:.2f}")
        print(f"  Range: {metrics['range']:.2f}")
        print(f"  IQR: {metrics['iqr']:.2f}")
        print(f"  Variance: {metrics['variance']:.2f}")
        print(f"  Std Dev: {metrics['std_dev']:.2f}")
        print(f"  CV: {metrics['cv']:.2f}")
        print(f"  Skewness: {metrics['skewness']:.2f}")
        print(f"  Kurtosis: {metrics['kurtosis']:.2f}")
        print(f"  Percentiles (25%): {metrics['percentiles_25%']:.2f}")
        print(f"  Percentiles (50%): {metrics['percentiles_50%']:.2f}")
        print(f"  Percentiles (75%): {metrics['percentiles_75%']:.2f}")
        print("\n")

distribution_metrics = calculate_distribution_metrics(train_data)
print_distribution_metrics(distribution_metrics)


train_data


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 train_data 已经被加载

# 选择数值型特征
numeric_features = train_data.select_dtypes(include=[np.number])

# 计算相关性矩阵
correlation_matrix = numeric_features.corr()

# 绘制相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Feature Correlation Heatmap")
plt.show()


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 假设 train_data 和 test_data 已经被加载

# 选择需要合并的特征
features = ['maxtemp', 'temparature', 'mintemp', 'dewpoint']
train_features = train_data[features]
test_features = test_data[features]

# 特征标准化
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# 应用 PCA
pca = PCA(n_components=1)  # 保留一个主成分
train_pca = pca.fit_transform(train_features_scaled)
test_pca = pca.transform(test_features_scaled)

# 将主成分添加到数据集中
train_data['temperature_pca'] = train_pca
test_data['temperature_pca'] = test_pca

# 打印解释的方差比例
print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 train_data 已经被加载

# 选择数值型特征
numeric_features = train_data.select_dtypes(include=[np.number])

# 计算相关性矩阵
correlation_matrix = numeric_features.corr()

# 绘制相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Feature Correlation Heatmap")
plt.show()


test_data


train_data


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 假设 train_data 和 test_data 已经被加载

# 对 temperature_pca 进行归一化处理
scaler_temperature_pca = MinMaxScaler()
train_data['temperature_pca'] = scaler_temperature_pca.fit_transform(train_data[['temperature_pca']])
test_data['temperature_pca'] = scaler_temperature_pca.transform(test_data[['temperature_pca']])

# 打印处理后的数据
print(train_data[['temperature_pca']].head())
print(test_data[['temperature_pca']].head())


train_data


# 定义要选择的特征
#features = ['cloud', 'humidity', 'dewpoint', 'winddirection_sin', 'winddirection_cos', 'windspeed', 'temperature_pca', 'sunshine','temparature']

# 从训练集中选择特征和目标变量
X_train = train_data.drop(columns=['rainfall'])
y_train = train_data['rainfall']

# 从测试集中选择特征
X_test = test_data

# 打印选择的特征和目标变量
print("训练集特征:")
print(X_train.head())
print("\n测试集特征:")
print(X_test.head())
print("\n目标变量（训练集）:")
print(y_train.head())
sample_submission


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint

# 定义逻辑回归模型及其超参数网格
logistic = LogisticRegression(random_state=42, max_iter=1500)
logistic_param_dist = {
    'C': uniform(0.01, 10),
    'penalty': ['l2']
}

# 定义 XGBoost 模型及其超参数网格
xgb = XGBClassifier(random_state=42)
xgb_param_dist = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 6),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3)
}

# 定义 LightGBM 模型及其超参数网格
lgbm = LGBMClassifier(random_state=42)
lgbm_param_dist = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 6),
    'num_leaves': randint(20, 40),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3)
}


train_data


# 使用随机搜索为每个模型调优
def tune_model(model, param_dist, X_train, y_train):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# 调优并训练每个模型
best_logistic = tune_model(logistic, logistic_param_dist, X_train, y_train)
best_xgb = tune_model(xgb, xgb_param_dist, X_train, y_train)
best_lgbm = tune_model(lgbm, lgbm_param_dist, X_train, y_train)



# 创建投票分类器
voting_clf = VotingClassifier(
    estimators=[
        ('lr', best_logistic),
        ('xgb', best_xgb),
        ('lgbm', best_lgbm)
    ],
    voting='soft'  # 使用软投票以获取概率
)

# 训练投票分类器
voting_clf.fit(X_train, y_train)

# 对测试集进行预测，获取概率
test_probabilities = voting_clf.predict_proba(X_test)[:, 1]  # 获取正类的概率



# 创建一个 DataFrame 来存储测试集的预测概率
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_probabilities
})

# 保存为 CSV 文件
submission.to_csv('submission.csv', index=False)

print("测试集预测概率已保存为 'submission.csv'。")

# 查看提交结果的前几行
print("\n提交结果的前5行：")
print(submission.head())


submission




