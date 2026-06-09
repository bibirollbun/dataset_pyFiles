!pip -q install calplot


pip install pytabkit


import numpy as np 
import pandas as pd
import datetime as dt
import seaborn as sns
from colorama import Style, Fore
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, SplineTransformer, FunctionTransformer
from category_encoders import OneHotEncoder, TargetEncoder
from datetime import datetime
from lightgbm import LGBMRegressor
from scipy.optimize import differential_evolution, minimize
from xgboost import XGBRegressor
from pytabkit import RealMLP_TD_Regressor
from pytabkit import TabM_D_Regressor
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline, Pipeline
import gc
from scipy.signal import periodogram
from scipy.stats import kurtosis
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator, TransformerMixin
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import PredictionErrorDisplay, mean_absolute_error
import os
from sklearn.kernel_approximation import Nystroem
import plotly_express as px
import calplot
from sklearn import set_config
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
set_config(transform_output='pandas')
plt.style.use('ggplot')


from IPython.display import display


sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv',parse_dates=['date'])
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv',parse_dates=['date'])
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv',parse_dates=['date'])
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


sales_train = pd.merge(sales_train, inventory, how='left', on =['unique_id','warehouse'])
sales_test = pd.merge(sales_test, inventory, how='left', on =['unique_id','warehouse'])


sales_train = pd.merge(sales_train, calendar, how='left', on =['date','warehouse'])
sales_test = pd.merge(sales_test, calendar, how='left', on =['date','warehouse'])


for df in [sales_train,sales_test]:
    df.set_index('date',inplace=True)


np.setdiff1d(sales_train.columns,sales_test.columns)


sales_train.drop(['availability'], axis=1, inplace=True)
sales_train.sort_values(['date','warehouse'],inplace=True)


cat_cols = list(sales_test.select_dtypes(include='O'))
num_cols = list(sales_test._get_numeric_data())
target = 'sales'
initial_features = list(sales_test.columns)


for c in cat_cols:    
    A = sales_train[c].fillna('None').astype(str).unique()
    B = sales_test[c].fillna('None').astype(str).unique()
    C = np.setdiff1d(B,A)
    if C.size>0:
        print(C)
        sales_train.iloc[~sales_train[c].isin(C), c ] = 'None'
    sales_train[c] = sales_train[c].astype('category')
    sales_test[c] = sales_test[c].astype('category')    


sales_train.loc[sales_train.sales.isnull(),:].reset_index().groupby(['warehouse'],observed=False). \
agg(size=('warehouse','size'),
    min_date=('date','min'),
    max_date=('date','max'),
    days = ('date', lambda x: x.max() - x.min()),
    split_date=('date', lambda x: list(np.unique(np.unique(x.dt.strftime('%Y-%m-%d'))))) 
   ).dropna()


sales_train['sales'] = sales_train['sales'].fillna(0)
sales_train['total_orders'] = sales_train['total_orders'].fillna(0)
sales_train['sell_price_main'] = sales_train['sell_price_main'].interpolate()


weight_map = test_weights.set_index('unique_id')['weight'].to_dict()


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


class DropColsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        self.cols = cols
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.drop(self.cols,axis=1)



class CreateTimeFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()                
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        df['week'] = df.index.isocalendar().week
        df['weekend'] = df.index.weekday // 5
        df['semiweekly'] = np.where(df.index.weekday <3,0,1)    
        df['year_sin'] = np.sin(df['year'] / 1 * 2 * np.pi)
        df['year_cos'] = np.cos(df['year'] / 1 * 2 * np.pi)
        df['month_sin'] = np.sin(df['month'] / 12 * 2 * np.pi)
        df['month_cos'] = np.cos(df['month'] / 12 * 2 * np.pi)
        

        return df
    


ctf = CreateTimeFeatures()
sales_train = ctf.fit_transform(sales_train).copy()  
sales_test = ctf.fit_transform(sales_test).copy() 

my_index = sales_train.index
my_index_ts = sales_test.index

agg_df = sales_train.reset_index().groupby(['name'],observed=False).agg(days_in_sale=('date','nunique'),
                                                                   purchase_interval=('date',lambda x: (x.max() - x.min()).days)                                                               
                                                                  ).reset_index()
sales_train = sales_train.merge(agg_df[['name', 'days_in_sale', 'purchase_interval']], on='name', how='left')
sales_test = sales_test.merge(agg_df[['name', 'days_in_sale', 'purchase_interval']], on='name', how='left')
sales_train.set_index(my_index,inplace=True)
  
sales_train.loc[:,'date_diff'] = sales_train.reset_index().groupby('name',observed=False)['date'].diff().dt.days.fillna(0).reset_index()['date'].values
sales_train['gap'] = sales_train['date_diff'] > 1
sales_train['gap_group'] = sales_train.groupby(['name'],observed=False)['gap'].cumsum()
agg_df = sales_train.groupby(['name', 'gap_group'],observed=False)['date_diff'].max().reset_index()
agg_df = agg_df.groupby('name',observed=False)['date_diff'].max().rename('days_without_sale')
sales_train = sales_train.merge(agg_df, on='name', how='left')
sales_test = sales_test.merge(agg_df, on='name', how='left')

sales_train.set_index(my_index,inplace=True)
sales_test.set_index(my_index_ts,inplace=True)



# 选出所有的折扣列
discount_cols = ["type_0_discount", "type_1_discount", "type_2_discount", 
                 "type_3_discount", "type_4_discount", "type_5_discount", "type_6_discount"]

# 计算最大折扣（如果有负值，意味着无折扣，需要替换为 0）
sales_train["max_discount"] = sales_train[discount_cols].max(axis=1)
sales_test["max_discount"] = sales_test[discount_cols].max(axis=1)

# 确保所有负折扣值转换为 0（负值意味着没有折扣）
sales_train["max_discount"] = sales_train["max_discount"].clip(lower=0)
sales_test["max_discount"] = sales_test["max_discount"].clip(lower=0)
target_column = 'sales'  

# 选择要编码的类别特征
category_cols = ["name", "holiday_name", "L1_category_name_en", "L2_category_name_en", "L3_category_name_en", "L4_category_name_en"]

# 初始化 TargetEncoder
encoder = TargetEncoder(
    cols=category_cols,
    handle_unknown='value',  # 测试集新类别用训练集全局均值代替
    smoothing=10,            # 平滑系数，防止过拟合
    min_samples_leaf=5       # 最小样本数，增加鲁棒性
)

# 对训练集进行拟合和转换
sales_train[category_cols] = encoder.fit_transform(
    sales_train[category_cols], 
    sales_train[target_column]  # 需要传入目标变量
)

# 对测试集进行转换（使用训练集的统计量）
sales_test[category_cols] = encoder.transform(
    sales_test[category_cols]
)
from sklearn.preprocessing import LabelEncoder

# 选择要编码的类别特征
category_cols = ['warehouse', 'L1_category_name_en']

# 逐列进行 Label Encoding
label_encoders = {}
for col in category_cols:
    le = LabelEncoder()
    sales_train[col] = le.fit_transform(sales_train[col])
    sales_test[col] = le.transform(sales_test[col])  # 确保 test 数据集用相同编码
    label_encoders[col] = le

sales_train['gap'] = sales_train['gap'].astype(int)


features = ['unique_id', 'name','total_orders','warehouse',
            'product_unique_id','max_discount','sell_price_main',
            'gap_group','week']
##这部分和syq那部分的catboost模型选择的是一样的输入特征


##这部分就是训练模型（保持和syq一样的参数）&保存模型
seed = 2025
import joblib


X_train = sales_train[features]
y_train = sales_train['sales']

cb_params = {
    'grow_policy'        : 'Lossguide',
    'task_type'          : 'GPU',
    'iterations'         : 800,
    'bagging_temperature': 0.5,
    'learning_rate'      : 0.1,
    'max_leaves'         : 128,
    'max_depth'          : 12,
    'l2_leaf_reg'        : 1.25,
    'min_data_in_leaf'   : 24,
    'verbose'            : 0,
    'border_count'       : 256,
  
} 


model = make_pipeline(
    # TargetEncoder(cols=['name', 'holiday_name', 'L2_category_name_en',
    #                     'L3_category_name_en', 'L4_category_name_en']), 
    # OneHotEncoder(cols=['warehouse', 'L1_category_name_en']),
    CatBoostRegressor(**cb_params, random_state=seed)
)

# 模型训练
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, 'catboost_model_pipeline.pkl')

# 加载模型（之后可用于预测）
model = joblib.load('catboost_model_pipeline.pkl')





pip install stable-baselines3[extra] gym


##导入一些包
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.sac.policies import MlpPolicy
import torch

# 设备检查
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#后续用src和ppo方法都是用这个表
##这里限定unique_ID=12是因为，每个商品的策略都不一样，跑模型只能一个商品一个商品跑,这里用12这个商品来跑
sales_subset_12=sales_train[sales_train['unique_id'] == 12]
sales_subset_12 =sales_subset_12[['unique_id', 'name', 'total_orders', 'warehouse','product_unique_id', 'sales', 'sell_price_main',  'holiday', 'shops_closed','winter_school_holidays', 'school_holidays', 'weekend', 'gap_group', 'week','semiweekly', 'year_sin', 'month_sin', 'max_discount']]


                              
##total_orders：该 Rohlik 仓库的历史订单数量
##sales:销量
##sell_price_main原价
##holiday：是否为节假日（0/1）
##shops_closed：是否为大多数商店关闭的节假日
##winter_school_holidays：是否为冬季学校假期
##school_holidays：是否为学校假期
##weekend：指示该日期是否为周末。返回值是一个二值化的特征
##semiweekly：这是一个二值化特征，表示该日期是否属于星期一到星期三的工作周。具体来说：如果日期是星期一到星期三（weekday < 3），则为 0，表示前半周。如果日期是星期四或星期五（weekday >= 3），则为 1，表示后半周。
##year_sin：使用正弦函数对年份进行周期性转换，目的是让年份能够表现出周期性特征（例如季节性或年度模式）。np.sin(df['year'] / 1 * 2 * np.pi) 使得年份的转换映射到 -1 到 1 的范围。该特征通过正弦波的方式，捕捉到年度周期性。
##year_cos：与 year_sin 相对，使用余弦函数对年份进行周期性转换。它和 year_sin 一起，提供了对周期性数据的更好表示，常用于时间周期特征建模。
##month_sin：使用正弦函数对月份进行周期性转换，将月份数据映射到 -1 到 1 的范围。month_sin 特征帮助模型理解不同月份之间的季节性关系，例如，夏季和冬季的销售模式可能不同。
##month_cos：与 month_sin 配合使用，使用余弦函数对月份进行周期性转换。它和 month_sin 一起，帮助模型理解月份之间的周期性模式。
##purchase_interval-销售最早日期和最晚日期之间的天数差某商品在销售期间的购买间隔
##gap-是否有销售间隔超过1天的情况
##max_discount-折扣

sales_subset_12.head(10)



from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, smoothing_window=10):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0
        self.smoothing_window = smoothing_window

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        if rewards is not None:
            self.current_rewards += sum(rewards)
        return True

    def _on_rollout_end(self):
        self.episode_rewards.append(self.current_rewards)
        self.current_rewards = 0

    def _on_training_end(self):
        smoothed_rewards = self._smooth_rewards(self.episode_rewards, self.smoothing_window)

        plt.figure(figsize=(12, 5))
        plt.plot(self.episode_rewards, label='Original Reward', alpha=0.5, color='gray')
        plt.plot(smoothed_rewards, label=f'Smoothed Reward', color='blue', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode (Original vs Smoothed)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _smooth_rewards(self, rewards, window):
        if len(rewards) < window:
            return rewards
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        return smoothed

# -------------------------
# 定义强化学习环境
# -------------------------

class DiscountOptimizationEnv(gym.Env):
    def __init__(self, data, seed=None):  # 添加 seed 参数
        super(DiscountOptimizationEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.index = 0

        # 加载训练好的销量预测模型
        self.model = joblib.load("catboost_model_pipeline.pkl")

        # 特征列表（模型输入用）
        self.model_features = ['unique_id', 'name', 'total_orders', 'warehouse',
                               'product_unique_id', 'max_discount', 'sell_price_main',
                               'gap_group', 'week']

        # 动作空间：折扣比例（0~0.5）
        self.action_space = spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32)

        # 状态空间：你定义的10维状态向量
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)
        # 标准化处理：让状态空间的每一维都近似 0 均值、单位方差，利于训练
        # STATE_COLS = [
        #     'total_orders', 'sell_price_main', 'holiday', 'shops_closed',
        #     'winter_school_holidays', 'school_holidays', 'weekend',
        #     'semiweekly', 'year_sin', 'month_sin'
        # ]
        # STATE_COLS = [
        #    'unique_id', 'name', 'total_orders', 'warehouse',
        #                        'product_unique_id', 'sell_price_main',
        #                        'gap_group', 'week'
        # ]
        # self.scaler = StandardScaler()
        # self.data[STATE_COLS] = self.scaler.fit_transform(self.data[STATE_COLS])

        # 设置种子
        self.np_random, seed = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        self.index = 0
        # 设置随机种子
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # 设置种子
        obs = self._get_state(self.index)  # 获取当前的状态
        info = {}  # 返回的附加信息（空字典）
        return obs, info  # 返回状态和信息字典


    def _get_state(self, index):
        row = self.data.iloc[index]
        return np.array([
            row['total_orders'],
            row['sell_price_main'],
            row['holiday'],
            row['shops_closed'],
            row['winter_school_holidays'],
            row['school_holidays'],
            row['weekend'],
            row['semiweekly'],
            row['year_sin'],
            row['month_sin']
            # row['unique_id'], 
            # row['name'], 
            # row['total_orders'], 
            # row['warehouse'],           
            # row['product_unique_id'],
            # row['sell_price_main'],
            # row['gap_group'],
            # row['week']

            
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.index].copy()
        discount = float(np.clip(action[0], 0, 0.5))  # 折扣比例限制在 0 ~ 0.5
        discounted_price = row['sell_price_main'] * (1 - discount)
        # 输出折扣后的价格
        # print(f"Discount: {discount}, Discounted Price: {discounted_price}")
        
        # 更新 max_discount 字段为当前动作的折扣比例
        row['max_discount'] = discount
    
        # 构建模型预测所需输入
        model_input = row[self.model_features].to_frame().T  # 注意需要变成 DataFrame 的形式
    
        # 预测销量
        predicted_sales = self.model.predict(model_input)[0]
    
        # 输出预测的销量
        # print(f"Predicted Sales: {predicted_sales}")
    
        # 计算 reward：折后价 × 预测销量
        reward = discounted_price * predicted_sales
        # print(f"Reward (before scaling): {reward}")
        reward = np.log1p(discounted_price * predicted_sales)
        reward=reward*50
        # reward /= 100.0  # reward 缩放
        # print(f"Reward (after scaling): {reward}")
    
        self.index += 1
        done = self.index >= len(self.data)
    
        # 添加 truncated 判断：这里我们假设没有步数限制
        truncated = False  # 你可以根据实际情况调整这个条件
    
        next_state = np.zeros(10) if done else self._get_state(self.index)
    
        # 返回的元组需要包含五个值
        return next_state, reward, done, truncated, {}



# -------------------------
# 训练模型（使用 SAC）
# -------------------------
seed_value = 2025

env = DummyVecEnv([lambda: DiscountOptimizationEnv(sales_subset_12, seed=seed_value)])

model = SAC(
    MlpPolicy,
    env,
    verbose=1,
    device=device,
    batch_size=1024,
    learning_rate=3e-4,  # 降低学习率
    learning_starts=1000,
    buffer_size=10000,
    train_freq=8,
    ent_coef="0.05" 
)


callback = RewardLoggingCallback(smoothing_window=50)
model.learn(total_timesteps=4000, callback=callback)


model.save("discount_optimization_model")


from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, smoothing_window=10):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0
        self.smoothing_window = smoothing_window

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        if rewards is not None:
            self.current_rewards += sum(rewards)
        return True

    def _on_rollout_end(self):
        self.episode_rewards.append(self.current_rewards)
        self.current_rewards = 0

    def _on_training_end(self):
        smoothed_rewards = self._smooth_rewards(self.episode_rewards, self.smoothing_window)

        plt.figure(figsize=(12, 5))
        plt.plot(self.episode_rewards, label='Original Reward', alpha=0.5, color='gray')
        plt.plot(smoothed_rewards, label=f'Smoothed Reward', color='blue', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode (Original vs Smoothed)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _smooth_rewards(self, rewards, window):
        if len(rewards) < window:
            return rewards
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        return smoothed

# -------------------------
# 定义强化学习环境
# -------------------------

class DiscountOptimizationEnv(gym.Env):
    def __init__(self, data, seed=None):  # 添加 seed 参数
        super(DiscountOptimizationEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.index = 0

        # 加载训练好的销量预测模型
        self.model = joblib.load("catboost_model_pipeline.pkl")

        # 特征列表（模型输入用）
        self.model_features = ['unique_id', 'name', 'total_orders', 'warehouse',
                               'product_unique_id', 'max_discount', 'sell_price_main',
                               'gap_group', 'week']

        # 动作空间：折扣比例（0~1）
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 状态空间：你定义的10维状态向量
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)
        # 标准化处理：让状态空间的每一维都近似 0 均值、单位方差，利于训练
        # STATE_COLS = [
        #     'total_orders', 'sell_price_main', 'holiday', 'shops_closed',
        #     'winter_school_holidays', 'school_holidays', 'weekend',
        #     'semiweekly', 'year_sin', 'month_sin'
        # ]
        # STATE_COLS = [
        #    'unique_id', 'name', 'total_orders', 'warehouse',
        #                        'product_unique_id', 'sell_price_main',
        #                        'gap_group', 'week'
        # ]
        # self.scaler = StandardScaler()
        # self.data[STATE_COLS] = self.scaler.fit_transform(self.data[STATE_COLS])

        # 设置种子
        self.np_random, seed = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        self.index = 0
        # 设置随机种子
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # 设置种子
        obs = self._get_state(self.index)  # 获取当前的状态
        info = {}  # 返回的附加信息（空字典）
        return obs, info  # 返回状态和信息字典


    def _get_state(self, index):
        row = self.data.iloc[index]
        return np.array([
            row['total_orders'],
            row['sell_price_main'],
            row['holiday'],
            row['shops_closed'],
            row['winter_school_holidays'],
            row['school_holidays'],
            row['weekend'],
            row['semiweekly'],
            row['year_sin'],
            row['month_sin']
            # row['unique_id'], 
            # row['name'], 
            # row['total_orders'], 
            # row['warehouse'],           
            # row['product_unique_id'],
            # row['sell_price_main'],
            # row['gap_group'],
            # row['week']

            
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.index].copy()
        discount = float(np.clip(action[0], 0, 0.5))  # 折扣比例限制在 0 ~ 0.5
        discounted_price = row['sell_price_main'] * (1 - discount)
        # 输出折扣后的价格
        # print(f"Discount: {discount}, Discounted Price: {discounted_price}")
        
        # 更新 max_discount 字段为当前动作的折扣比例
        row['max_discount'] = discount
    
        # 构建模型预测所需输入
        model_input = row[self.model_features].to_frame().T  # 注意需要变成 DataFrame 的形式
    
        # 预测销量
        predicted_sales = self.model.predict(model_input)[0]
    
        # 输出预测的销量
        # print(f"Predicted Sales: {predicted_sales}")
    
        # 计算 reward：折后价 × 预测销量
        reward = discounted_price * predicted_sales
        # print(f"Reward (before scaling): {reward}")
        reward = np.log1p(discounted_price * predicted_sales)
        reward /= 5  # 视情况缩放
        # reward /= 100.0  # reward 缩放
        # print(f"Reward (after scaling): {reward}")
    
        self.index += 1
        done = self.index >= len(self.data)
    
        # 添加 truncated 判断：这里我们假设没有步数限制
        truncated = False  # 你可以根据实际情况调整这个条件
    
        next_state = np.zeros(10) if done else self._get_state(self.index)
    
        # 返回的元组需要包含五个值
        return next_state, reward, done, truncated, {}



from stable_baselines3 import PPO  # 改为 PPO

# 保持你的 seed、env 构造和 callback 不变
seed_value = 2025
env = DummyVecEnv([lambda: DiscountOptimizationEnv(sales_subset_12, seed=seed_value)])
callback = RewardLoggingCallback(smoothing_window=50)

# 初始化 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    batch_size=1024,
    learning_rate=3e-5,
    n_steps=2048,         # 每次更新使用多少步数据，默认2048，可按数据量适当调整
    n_epochs=10,          # 每次更新的训练轮数
    gamma=0.99,           # 折扣因子
    clip_range=0.2        # PPO 剪切范围
)

# 训练模型
model.learn(total_timesteps=1000000, callback=callback)

# 保存模型
model.save("discount_optimization_model_ppo1")


