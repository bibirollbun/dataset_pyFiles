import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway  #ANOVA检验：判断多个独立组的均值差异是否由随机误差引起
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from bayes_opt import BayesianOptimization , SequentialDomainReductionTransformer
import gc
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
sns.set(style="whitegrid", font=['SimHei'])


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


test.head()


plt.figure(figsize=(6,4))
sns.histplot(train['accident_risk'] , kde = True ,color= 'orange')
plt.show()


train['accident_risk'].describe()



#列类型
print(f'train cols types : \n{train.dtypes.value_counts()}')


train.isnull().sum()


train.duplicated().sum()


num_feat = train.drop(['id','accident_risk'],axis=1).select_dtypes(include='number').columns
str_feat = train.select_dtypes(exclude='number').columns  #8

# 为每个特征指定不同的颜色
feature_colors = ['coral', 'skyblue', 'lightgreen', 'gold', 
                 'violet', 'orange', 'lightcoral', 'lightblue']

fig , ax = plt.subplots(2 , 4 , figsize = (12 , 8 ))
axes = ax.ravel()

for idx , col in enumerate(str_feat) :
    data = train.groupby(col)['accident_risk'].mean().sort_values()
    data.plot(kind = 'bar' , color = feature_colors[idx], edgecolor = 'black' , alpha = 0.8 ,ax = axes[idx])
    axes[idx].set_title(f'Avg Accident Risk by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Avg Accident Risk')
    axes[idx].tick_params(axis='x', rotation=45)  #倾斜45°
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


for col in str_feat :
    if train[col].dtype != 'bool' :# 跳过布尔型特征
        # 按特征类别分组提取事故风险数据
        groups = [train[train[col] == val]['accident_risk'].values for val in train[col].unique()]

         # 执行ANOVA检验
        f_stat, p_value = f_oneway(*groups)
        print(f'{col} - F-stat : {f_stat:8.2f} , P-value : {p_value:.2e} ')


fig , ax = plt.subplots(2 , 2 , figsize = (12 , 8 ))
axes = ax.ravel()
num_feature_colors = ['coral', 'skyblue', 'lightgreen', 'gold']

for idx , col in enumerate(num_feat) :
    if col == 'curvature' : 
        num_group = train.groupby(pd.cut(train[col], bins=7) , observed=False)['accident_risk'].mean().sort_index()  #分组再看不同的影响
    else :
        num_group = train.groupby(col)['accident_risk'].mean().sort_index()
        
    num_group.plot(kind = 'bar' , color = num_feature_colors[idx] , ax=axes[idx] , alpha = 0.8)
    axes[idx].tick_params(axis='x', rotation=45)

    # 显示水平（x轴）方向的网格线，从而去除竖直（y轴）方向的网格线
    axes[idx].grid(False , axis ='x') 
    axes[idx].set_title(f'Avg Accident Risk by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Avg Accident Risk')
    
plt.tight_layout()
plt.show()


heat_feat = list(num_feat) + ['accident_risk']
corr_matrix = train[heat_feat].corr()

sns.heatmap(corr_matrix , 
            annot= True, # 显示数值
            cmap='coolwarm',   # 颜色方案
            center=0,          # 中心点为0
            square=True)       # 正方形单元格

plt.show()


train['is_night'] = (train['lighting'] == 'night').astype(int)
train['hight_curvature'] = (train['curvature'] > 0.5).astype(int)
train['hight_speed_limit'] = (train['speed_limit'] >= 60).astype(int)
train['num_reported_accidents'] = np.where(train['num_reported_accidents'].between(3,6),1,0)
train['pow2_curvature'] = train['curvature']**2  #curvature对结果影响明显 对该列多做一些变化
train['pow3_curvature'] = train['curvature']**3
train['speed_squared'] = train['speed_limit'] ** 2
train['bins_curvature'] = pd.cut(train['curvature'] , bins = [0 , 0.3 , 0.6 , 1] ,
                                 labels= [0 ,1 ,2 ] , include_lowest= True)

train['high_risk_combo'] = ((train['curvature'] > 0.5) & (train['speed_limit'] >= 60)).astype(int)

train['weather_lighting_risk'] = (
    ((train['weather'] == 'foggy') | (train['weather'] == 'rainy')) &
    ((train['lighting'] == 'dim') | (train['lighting'] == 'night'))
).astype(int)

train['is_bad_weather'] = train['weather'].isin(['foggy', 'rainy']).astype(int)
train['is_peak_time'] = train['time_of_day'].isin(['morning', 'evening']).astype(int)
train['is_weekend'] = train['holiday'].astype(int)

train['danger_score'] = (
        (train['curvature'] > 0.6).astype(int) +
        (train['speed_limit'] >= 60).astype(int) +
        train['is_bad_weather'] +
        train['is_night'] +
        (train['num_reported_accidents'] >= 2).astype(int)
    )
train['accidents_per_lane'] = train['num_reported_accidents'] / (train['num_lanes'] + 1)



test['is_night'] = (test['lighting'] == 'night').astype(int)
test['hight_curvature'] = (test['curvature'] > 0.5).astype(int)
test['hight_speed_limit'] = (test['speed_limit'] >= 60).astype(int)
test['num_reported_accidents'] = np.where(test['num_reported_accidents'].between(3,6),1,0)
test['pow2_curvature'] = test['curvature']**2  #curvature对结果影响明显 对该列多做一些变化
test['pow3_curvature'] = test['curvature']**3
test['speed_squared'] = test['speed_limit'] ** 2
test['bins_curvature'] = pd.cut(test['curvature'] , bins = [0 , 0.3 , 0.6 , 1] ,
                                labels= [0 ,1 ,2 ] , include_lowest= True)
test['high_risk_combo'] = ((test['curvature'] > 0.5) & (test['speed_limit'] >= 60)).astype(int)

test['weather_lighting_risk'] = (
    ((test['weather'] == 'foggy') | (test['weather'] == 'rainy')) &
    ((test['lighting'] == 'dim') | (test['lighting'] == 'night'))
).astype(int)
test['is_bad_weather'] = test['weather'].isin(['foggy', 'rainy']).astype(int)
test['is_peak_time'] = test['time_of_day'].isin(['morning', 'evening']).astype(int)
test['is_weekend'] = test['holiday'].astype(int)


test['danger_score'] = (
        (test['curvature'] > 0.6).astype(int) +
        (test['speed_limit'] >= 60).astype(int) +
        test['is_bad_weather'] +
        test['is_night'] +
        (test['num_reported_accidents'] >= 2).astype(int)
    )
test['accidents_per_lane'] = test['num_reported_accidents'] / (test['num_lanes'] + 1)



# 创建多项式特征转换器，设置度为2且只生成交互项（不包括x1^2, x2^2这样的单项） 不用全部都交互 这里四列都比较有意义可以尝试
poly = PolynomialFeatures(degree=4, interaction_only=True, include_bias=False)  

train_num_feat_Poly = poly.fit_transform(train[num_feat])
test_num_feat_Poly = poly.fit_transform(test[num_feat])

interaction_names = poly.get_feature_names_out(num_feat)

train_interactions = pd.DataFrame(train_num_feat_Poly, columns=interaction_names ,index=train.index)
test_interactions = pd.DataFrame(test_num_feat_Poly, columns=interaction_names ,index=test.index)

# 筛选出交互项（特征名中包含空格，表示由多个原始特征相乘而成）
interaction_cols = [col for col in interaction_names if ' ' in col]

# 从交互特征DataFrame中只选取这些交互项列
train_interactions_only = train_interactions[interaction_cols]
test_interactions_only = test_interactions[interaction_cols]

train_with_interactions = pd.concat([train, train_interactions_only], axis=1)
test_with_interactions = pd.concat([test, test_interactions_only], axis=1)

train_with_interactions.head()


def feature_combine(df , combine_col = ['lighting' , 'weather' , 'hight_curvature' , 'hight_speed_limit']) :
    df = df.copy()
    
    num_cols = len(combine_col)
    
    for idx , col in enumerate(combine_col) :
        
        if col not in df.columns :
            raise ValueError(f'DataFrame not include {col}, check the col name')
        
        for i in range(idx+1 , num_cols) :
            
            sub_col = combine_col[i]
            if sub_col not in df.columns :
                raise ValueError(f'DataFrame not include {col}, check the col name')
            
            new_col_names = f'c_{col}_{sub_col}'  #加个标识c_开头

            df[new_col_names] = df[col].astype(str) + '_' + df[sub_col].astype(str) 

    return df

train_combine = feature_combine(train_with_interactions)
test_combine = feature_combine(test_with_interactions)



comb_cols = list(train_combine.columns[train_combine.columns.str.startswith('c_')])
bool_feat = train_combine.select_dtypes(include='bool').columns  #布尔型本就是True/False 这里不需要转换
str_feat = list(set(str_feat) - set(bool_feat)) + ['bins_curvature']

# 1. 为训练集和测试集添加来源标识，便于后续拆分
train_combine['data_type'] = 'train'
test_combine['data_type'] = 'test'

# 2. 纵向合并两个数据集
combined_data = pd.concat([train_combine, test_combine], ignore_index=True)

# 3. 对合并后的数据统一进行独热编码
combined_encoded = pd.get_dummies(combined_data, columns=comb_cols + str_feat)
print("合并编码后的列名:", combined_encoded.columns.tolist())

# 4. 利用之前添加的标识列，将数据重新拆分为训练集和测试集
train_encoded = combined_encoded[combined_encoded['data_type'] == 'train'].drop(['data_type' ] , axis=1)
test_encoded = combined_encoded[combined_encoded['data_type'] == 'test'].drop(['data_type' , 'accident_risk'], axis=1)

train_encoded.head()


X = train_encoded.drop(['id','accident_risk'] , axis =1 )
y = train_encoded['accident_risk']
X_test = test_encoded.drop('id' , axis =1)

X_train , X_valid , y_train , y_valid = train_test_split(X , y , test_size=0.25 , random_state=42)



del train_with_interactions , test_with_interactions , train , test , train_encoded , test_encoded
gc.collect()


# 基础随机森林模型
rf_base = RandomForestRegressor(
    n_estimators=144,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features = 1 ,
    random_state=42
)

# 训练基础模型
rf_base.fit(X_train, y_train)

# 在验证集上评估
y_valid_pred = rf_base.predict(X_valid)

# 计算评估指标
mse = mean_squared_error(y_valid, y_valid_pred)
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

print("基础模型性能:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")


final_test = rf_base.predict(X_test)

final_test = np.clip(final_test , 0 ,1)  #范围应该在 0-1
pd.Series(final_test).describe()


submission = sample_submission.copy()
submission['accident_risk'] = final_test

submission.to_csv('submission.csv', index=False)




