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


train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_df = pd.merge(train,train_id,on='TransactionID',how='left')


def amt_eng(df):

    # 创建副本以避免修改原DataFrame
    result_df = df.copy()
    
    # 添加金额对数列（处理0和负值的情况）
    # 使用log1p处理可能为0的值，即log(1+x)
    result_df['TransactionAmt_log'] = np.log1p(result_df['TransactionAmt'])
    
    # 添加金额小数列（将金额转换为小数格式）
    result_df['TransactionAmt_decimal'] = result_df['TransactionAmt'] % 1
    
    return result_df


def id_split(df):
    # 复制DataFrame避免修改原数据
    result_df = df.copy()
    
    # 拆分设备信息
    device_split = result_df['DeviceInfo'].str.split('/', expand=True)
    result_df['device_name'] = device_split[0]
    result_df['device_version'] = device_split[1] if len(device_split.columns) > 1 else None
    
    # 拆分操作系统信息
    id_30_split = result_df['id_30'].str.split(' ', expand=True)
    result_df['OS_id_30'] = id_30_split[0]
    result_df['version_id_30'] = id_30_split[1] if len(id_30_split.columns) > 1 else None
    
    # 拆分浏览器信息
    id_31_split = result_df['id_31'].str.split(' ', expand=True)
    result_df['browser_id_31'] = id_31_split[0]
    result_df['version_id_31'] = id_31_split[1] if len(id_31_split.columns) > 1 else None
    
    # 拆分屏幕分辨率
    id_33_split = result_df['id_33'].str.split('x', expand=True)
    result_df['screen_width'] = id_33_split[0]
    result_df['screen_height'] = id_33_split[1] if len(id_33_split.columns) > 1 else None
    
    # 提取id字段的值 - 安全地处理可能缺失冒号的情况
    result_df['id_34'] = result_df['id_34'].str.split(':').str[-1]  # 取最后一个部分
    result_df['id_23'] = result_df['id_23'].str.split(':').str[-1]  # 取最后一个部分
    
    # 设备名称标准化
    device_mapping = {
        'Samsung': ['SM', 'SAMSUNG', 'GT-'],
        'Motorola': ['Moto G', 'Moto', 'moto'],
        'LG': ['LG-'],
        'RV': ['rv:'],
        'Huawei': ['HUAWEI', 'ALE-', '-L'],
        'ZTE': ['Blade', 'BLADE'],
        'Linux': ['Linux'],
        'Sony': ['XT'],
        'HTC': ['HTC'],
        'Asus': ['ASUS']
    }
    
    # 应用设备名称映射
    for new_name, patterns in device_mapping.items():
        for pattern in patterns:
            mask = result_df['device_name'].str.contains(pattern, na=False)
            result_df.loc[mask, 'device_name'] = new_name
    
    # 将稀有设备名称归为"Others"
    device_counts = result_df['device_name'].value_counts()
    rare_devices = device_counts[device_counts < 200].index
    result_df.loc[result_df['device_name'].isin(rare_devices), 'device_name'] = "Others"
    
    return result_df


#电子邮件域名分组映射&电子邮件后缀提取和标准化
def process_email(df, p_email_col='P_emaildomain', r_email_col='R_emaildomain'):
    
    # 步骤1: 域名分组映射
    domain_mapping = {
        'google': ['gmail.com', 'googlemail.com'],
        'microsoft': ['hotmail.com', 'outlook.com', 'live.com', 'msn.com'],
        'yahoo': ['yahoo.com', 'yahoo.co.jp', 'yahoo.co.uk', 'ymail.com'],
        'apple': ['icloud.com', 'me.com', 'mac.com'],
        'ao1': ['ao1.com'],
        'att': ['att.net', 'sbcglobal.net'],
    }
    
    # 美国常见后缀
    us_suffixes = ['com', 'net', 'edu', 'org', 'gov', 'mil', 'us']
    
    # 反转映射
    reverse_map = {}
    for service, domains in domain_mapping.items():
        for domain in domains:
            reverse_map[domain] = service
    
    # 处理每个电子邮件列
    for col in [p_email_col, r_email_col]:
        # 处理空值
        domain_clean = df[col].fillna('')
        
        # 步骤1: 创建bin特征
        df[f'{col}_bin'] = domain_clean.str.lower().map(reverse_map).fillna('other')
        
        # 步骤2: 创建suffix特征
        suffix = domain_clean.str.lower().str.split('.').str[-1]
        df[f'{col}_suffix'] = suffix.apply(lambda x: 'us' if x in us_suffixes else x)
        df[f'{col}_suffix'] = df[f'{col}_suffix'].replace('', 'other')
    
    return df


def time_feature_extraction(df):
    # 复制DataFrame避免修改原数据
    result_df = df.copy()
    
    # 从TransactionDT计算星期几
    result_df['Transaction_day_of_week'] = ((result_df['TransactionDT'] / (3600 * 24) - 1) % 7).astype(int)
    
    # 从TransactionDT计算小时
    result_df['Transaction_hour'] = ((result_df['TransactionDT'] / 3600) % 24).astype(int)
    
    return result_df


import pandas as pd
import numpy as np
import gc

# FREQUENCY ENCODE - 单数据集版本
def encode_FE(df, cols):
    """
    频率编码 - 单数据集版本
    """
    for col in cols:
        vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1  # 为缺失值预留
        nm = col + '_FE'
        df[nm] = df[col].map(vc)
        df[nm] = df[nm].astype('float32')
        print(nm, ', ', end='')
    return df

# LABEL ENCODE - 单数据集版本  
def encode_LE(df, cols, verbose=True):
    """
    标签编码 - 单数据集版本
    """
    for col in cols:
        encoded_series, uniques = df[col].factorize(sort=True)
        nm = col
        
        if encoded_series.max() > 32000:
            df[nm] = encoded_series.astype('int32')
        else:
            df[nm] = encoded_series.astype('int16')
            
        if verbose: 
            print(nm, ', ', end='')
    
    x = gc.collect()
    return df

# GROUP AGGREGATION MEAN AND STD - 优化版本
def encode_AG(df, main_columns, uids, aggregations=['mean'], fillna=True, usena=False):
    """
    分组聚合统计 - 优化版本
    """
    # 创建数据副本，避免修改原始数据
    result_df = df.copy()
    
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = f"{main_column}_{col}_{agg_type}"
                
                # 准备数据
                temp_data = result_df[[col, main_column]].copy()
                
                # 处理缺失值
                if usena:
                    temp_data[main_column] = temp_data[main_column].replace(-1, np.nan)
                
                # 分组聚合计算
                agg_values = temp_data.groupby(col)[main_column].agg(agg_type)
                
                # 映射回原数据
                result_df[new_col_name] = result_df[col].map(agg_values).astype('float32')
                
                # 填充缺失值
                if fillna:
                    result_df[new_col_name] = result_df[new_col_name].fillna(-1)
                
                print(f"'{new_col_name}', ", end='')
    
    return result_df

# COMBINE FEATURES - 单数据集版本
def encode_CB(df, col_pairs):
    """
    特征组合 - 单数据集版本
    col_pairs: 需要组合的特征对列表，如 [('col1', 'col2'), ('col3', 'col4')]
    """
    for col1, col2 in col_pairs:
        nm = f"{col1}_{col2}"
        df[nm] = df[col1].astype(str) + '_' + df[col2].astype(str)
        
        # 对新组合的特征进行标签编码
        encoded_series, _ = df[nm].factorize(sort=True)
        if encoded_series.max() > 32000:
            df[nm] = encoded_series.astype('int32')
        else:
            df[nm] = encoded_series.astype('int16')
            
        print(nm, ', ', end='')
    
    return df

# GROUP AGGREGATION NUNIQUE - 单数据集版本
def encode_AG2(df, main_columns, uids):
    """
    分组唯一值计数 - 单数据集版本
    """
    for main_column in main_columns:  
        for col in uids:
            # 计算每个分组中主列的唯一值数量
            mapping_dict = df.groupby(col)[main_column].nunique().to_dict()
            
            new_col_name = f"{col}_{main_column}_ct"
            df[new_col_name] = df[col].map(mapping_dict).astype('float32')
            
            print(new_col_name, ', ', end='')
    
    return df


# 同步增加一些特征
def create_card_addr_features(df):
    """
    创建基于card和addr的特征组合和聚合特征
    
    """
    # 创建数据副本
    result_df = df.copy()
    
    # 1. 频率编码
    result_df = encode_FE(result_df, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
    
    # 2. 特征组合
    result_df = encode_CB(result_df, [('card1', 'addr1')])
    result_df = encode_CB(result_df, [('card1_addr1', 'P_emaildomain')])
    
    # 3. 频率编码组合特征
    result_df = encode_FE(result_df, ['card1_addr1', 'card1_addr1_P_emaildomain'])
    
    # 4. 分组聚合统计
    result_df = encode_AG(result_df, 
                         ['TransactionAmt', 'D9', 'D11'], 
                         ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], 
                         ['mean', 'std'], 
                         usena=True)
    
    return result_df


import datetime
def dt_m(df):
    # 复制DataFrame避免修改原数据
    result_df = df.copy()
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    result_df['DT_M'] = result_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    return result_df


def add_uid(df):
    # 复制DataFrame避免修改原数据
    result_df = df.copy()
    #uid为用户+地址+天数
    result_df['day'] = result_df.TransactionDT / (24*60*60) #这里可能跟前面的一些处理矛盾，不过我理解应该其实也只包含dt信息问题不大
    result_df['uid'] = result_df.card1_addr1.astype(str)+'_'+np.floor(result_df.day-result_df.D1).astype(str)
    
    return result_df


def create_uid_features(df, uid_col='uid'):
    """
    创建基于UID的聚合特征
    """
    import numpy as np
    
    # 创建数据副本
    result_df = df.copy()
    
    # 1. 频率编码
    result_df = encode_FE(result_df, [uid_col])
    
    # 2. 数值型特征的聚合统计
    # TransactionAmt, D系列特征的均值和标准差
    result_df = encode_AG(result_df, 
                         ['TransactionAmt','D4','D9','D10','D15'], 
                         [uid_col], 
                         ['mean','std'], 
                         fillna=True, usena=True)
    
    # C系列特征的均值（除C3外）
    c_cols = ['C'+str(x) for x in range(1,15) if x != 3]
    result_df = encode_AG(result_df, c_cols, [uid_col], ['mean'], fillna=True, usena=True)
    
    # M系列特征的均值
    m_cols = ['M'+str(x) for x in range(1,10)]
    
    # 只加这一行：在处理M系列之前转换object类型为数值
    for col in m_cols:
        if col in result_df.columns and result_df[col].dtype == 'object':
            result_df[col] = pd.factorize(result_df[col])[0].astype('float32')
    
    result_df = encode_AG(result_df, m_cols, [uid_col], ['mean'], fillna=True, usena=True)
    
    # C14的标准差
    result_df = encode_AG(result_df, ['C14'], [uid_col], ['std'], fillna=True, usena=True)
    
    # 3. 分类特征的唯一值计数
    # 第一组分类特征
    result_df = encode_AG2(result_df, ['P_emaildomain','dist1','DT_M','id_02','TransactionAmt_decimal'], [uid_col])
    
    # 第二组分类特征
    result_df = encode_AG2(result_df, ['C13','V314'], [uid_col])
    
    # 第三组分类特征
    result_df = encode_AG2(result_df, ['V127','V136','V309','V307','V320'], [uid_col])
    
    # 4. 创建新特征
    # 这里存在对于时间间隔的判断，可能到时候需要解释一下
    if 'D1' in result_df.columns and 'D15' in result_df.columns:
        result_df['outsider15'] = (np.abs(result_df.D1 - result_df.D15) > 3).astype('int8')
        print('outsider15')
    else:
        print("警告: D1或D15列不存在，跳过创建outsider15特征")
    return result_df


def apply_all_feature_engineering(df):
    """
    应用所有特征工程函数，按照指定顺序
    
    """
    # 按照你提供的顺序依次调用所有特征工程函数
    df = amt_eng(df)
    df = id_split(df)
    df = process_email(df)
    df = time_feature_extraction(df)
    df = create_card_addr_features(df)
    df = dt_m(df)
    df = add_uid(df)
    df = create_uid_features(df, uid_col='uid')
    
    return df


train_df_model=apply_all_feature_engineering(train_df)


train_df_model.shape


train_df_model = train_df_model.drop(columns=['DT_M','D6','D7','D8','D9','D12','D13','D14','C3','M5','id_08','id_33','card4','id_07','id_14','id_21','id_30','id_32','id_34'] + ['id_'+str(x) for x in range(22,28)])
# 切分训练集和验证集
y_train = train_df_model['isFraud'].copy()
X_train = train_df_model.drop(columns=['isFraud'])
idxT = X_train.index[:3*len(X_train)//4]
idxV = X_train.index[3*len(X_train)//4:]
cols = list( X_train.columns )


from bayes_opt import BayesianOptimization

def xgb_bayesian_optimization(X_train, y_train, idxT, idxV, cols):
    """
    使用贝叶斯优化调参
    """
    # 复制数据以避免修改原始数据
    X_train_processed = X_train.copy()
    
    # 自动识别并编码object类型的列
    object_cols = X_train_processed[cols].select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"正在编码 {len(object_cols)} 个分类特征: {list(object_cols)}")
        
        # 使用LabelEncoder编码分类特征
        label_encoders = {}
        for col in object_cols:
            if col in cols:  # 确保该列在使用的特征中
                le = LabelEncoder()
                # 合并训练和验证集的数据来拟合encoder，确保编码一致性
                combined_data = pd.concat([
                    X_train_processed.loc[idxT, col], 
                    X_train_processed.loc[idxV, col]
                ])
                le.fit(combined_data)
                X_train_processed[col] = le.transform(X_train_processed[col])
                label_encoders[col] = le
                print(f"已编码: {col}")
    
    def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda):
        """贝叶斯优化的目标函数"""
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_estimators': 1000,
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 27,
            'early_stopping_rounds': 100  # 移到这里
        }
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train_processed.loc[idxT, cols], 
            y_train[idxT],
            eval_set=[(X_train_processed.loc[idxV, cols], y_train[idxV])],
            verbose=False  # 移除early_stopping_rounds参数
        )
        
        return clf.best_score
    
    # 定义参数范围
    pbounds = {
        'max_depth': (6, 12),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.6, 0.95),
        'colsample_bytree': (0.3, 0.8),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 2)
    }
    
    # 贝叶斯优化
    optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=27)
    optimizer.maximize(init_points=5, n_iter=15)
    
    print("最佳参数:", optimizer.max)
    
    # 用最佳参数训练最终模型
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = 2000
    best_params['early_stopping_rounds'] = 100  # 添加早停参数
    
    final_clf = xgb.XGBClassifier(**best_params)
    h = final_clf.fit(
        X_train_processed.loc[idxT, cols], 
        y_train[idxT],
        eval_set=[(X_train_processed.loc[idxV, cols], y_train[idxV])],
        verbose=50
    )
    
    return final_clf, h, label_encoders


clf, h, label_encoders= xgb_bayesian_optimization(X_train, y_train, idxT, idxV, cols)


# 获取特征重要性
feature_importance = clf.feature_importances_

# 创建重要性DataFrame
importance_df = pd.DataFrame({
    'feature': cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 60 最重要特征:")
print(importance_df.head(60))

# 绘制特征重要性图
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.barh(importance_df.head(60)['feature'], importance_df.head(60)['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()
plt.show()







