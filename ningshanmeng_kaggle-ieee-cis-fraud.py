# 在Kaggle中运行前需先安装依赖
!pip install psutil seaborn --quiet
#EDA过程
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np,gc # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


cols_t = ['TransactionID', 'TransactionDT', 'TransactionAmt',
       'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
       'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
       'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
       'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
       'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
       'M5', 'M6', 'M7', 'M8', 'M9']
cols_v = ['V'+str(x) for x in range(1,340)]; types_v = {}
for c in cols_v: types_v[c] = 'float32'
train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',usecols=cols_t+['isFraud']+cols_v,dtype=types_v)


nans_df = train.isna()
nans_groups={}
i_cols = ['V'+str(i) for i in range(1,340)]
for col in train.columns:
    cur_group = nans_df[col].sum()
    try:
        nans_groups[cur_group].append(col)
    except:
        nans_groups[cur_group]=[col]
del nans_df; x=gc.collect()

for k,v in nans_groups.items():
    print('####### NAN count =',k)
    print(v)


Vc = ['dayr','isFraud','TransactionAmt','card1','addr1','D1n','D11n']
Vs = nans_groups[279287]
Vtitle = 'V1 - V11, D11'


def make_plots(Vs):
    col = 4
    row = len(Vs)//4+1
    plt.figure(figsize=(20,row*5))
    idx = train[~train[Vs[0]].isna()].index
    for i,v in enumerate(Vs):
        plt.subplot(row,col,i+1)
        n = train[v].nunique()
        x = np.sum(train.loc[idx,v]!=train.loc[idx,v].astype(int))
        y = np.round(100*np.sum(train[v].isna())/len(train),2)
        t = 'int'
        if x!=0: t = 'float'
        plt.title(v+' has '+str(n)+' '+t+' and '+str(y)+'% nan')
        plt.yticks([])
        h = plt.hist(train.loc[idx,v],bins=100)
        if len(h[0])>1: plt.ylim((0,np.sort(h[0])[-2]))
    plt.show()
make_plots(Vs)


def make_corr(Vs,Vtitle=''):
    cols = ['TransactionDT'] + Vs
    plt.figure(figsize=(15,15))
    sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
    if Vtitle!='': plt.title(Vtitle,fontsize=14)
    else: plt.title(Vs[0]+' - '+Vs[-1],fontsize=14)
    plt.show()
make_corr(Vs,Vtitle)


grps = [[1],[2,3],[4,5],[6,7],[8,9],[10,11]]
def reduce_group(grps,c='V'):
    use = []
    for g in grps:
        mx = 0; vx = g[0]
        for gg in g:
            n = train[c+str(gg)].nunique()
            if n>mx:
                mx = n
                vx = gg
            #print(str(gg)+'-'+str(n),', ',end='')
        use.append(vx)
        #print()
    print('Use these',use)
reduce_group(grps)


Vs = nans_groups[314]
make_corr(Vs)


v =  [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]
v += [96, 98, 99, 104]
v += [107, 108, 111, 115, 117, 120, 121, 123]
v += [124, 127, 129, 130, 136]
v += [138, 139, 142, 147, 156, 162]
v += [165, 160, 166]
v += [178, 176, 173, 182]
v += [187, 203, 205, 207, 215]
v += [169, 171, 175, 180, 185, 188, 198, 210, 209]
v += [218, 223, 224, 226, 228, 229, 235]
v += [240, 258, 257, 253, 252, 260, 261]
v += [264, 266, 267, 274, 277]
v += [220, 221, 234, 238, 250, 271]
v += [294, 284, 285, 286, 291, 297]
v += [303, 305, 307, 309, 310, 320]
v += [281, 283, 289, 296, 301, 314]
v += [332, 325, 335, 338]

print('Reduced set has',len(v),'columns')


cols = ['TransactionDT'] + ['V'+str(x) for x in v]
train2 = train[cols].sample(frac=0.2)
plt.figure(figsize=(15,15))
sns.heatmap(train2[cols].corr(), cmap='RdBu_r', annot=False, center=0.0)
plt.title('V1-V339 REDUCED',fontsize=14)
plt.show()


cols = ['TransactionDT'] + ['V'+str(x) for x in range(1,340)]
train2 = train[cols].sample(frac=0.2)
plt.figure(figsize=(15,15))
sns.heatmap(train2[cols].corr(), cmap='RdBu_r', annot=False, center=0.0)
plt.title('V1-V339 ALL',fontsize=14)
plt.show()


def make_plots2(Vs):
    col = 4
    row = len(Vs)//4+1
    plt.figure(figsize=(20,row*5))
    for i,v in enumerate(Vs):
        plt.subplot(row,col,i+1)
        idx = train[~train[v].isna()].index
        n = train[v].nunique()
        x = np.sum(train.loc[idx,v]!=train.loc[idx,v].astype(int))
        y = np.round(100*np.sum(train[v].isna())/len(train),2)
        t = 'int'
        if x!=0: t = 'float'
        plt.title(v+' has '+str(n)+' '+t+' and '+str(y)+'% nan')
        plt.yticks([])
        h = plt.hist(train.loc[idx,v],bins=100)
        if len(h[0])>1: plt.ylim((0,np.sort(h[0])[-2]))
    plt.show()
make_plots2(['C'+str(x) for x in range(1,15)])


cols = ['TransactionDT'] + ['D'+str(x) for x in range(1,16)]
plt.figure(figsize=(15,15))
sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('D1-D15')
plt.show()


Ms = ['M'+str(x) for x in range(1,10)]
mp = {'F':0,'T':1,'M0':0,'M1':1,'M2':2}
for c in Ms: train[c] = train[c].map(mp)


cols = ['TransactionDT'] + Ms
plt.figure(figsize=(15,15))
sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('M1-M9')
plt.show()


train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_id = pd.merge(train_id,train[['TransactionID','TransactionDT']],on='TransactionID',how='left')
ids = ['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,39)]
for c in ids: print (c,train_id[c].unique()[:10])


booln = ['id_12','id_15','id_16','id_27','id_28','id_29','id_35','id_36','id_37','id_38']
cats = ['id_23','id_30','id_31','id_33','id_34']
mp = {'Unknown':0,'NotFound':1,'Found':2,'New':3,'F':0,'T':1}
for c in booln: train_id[c] = train_id[c].map(mp)


def make_plots2(Vs):
    col = 4
    row = len(Vs)//4+1
    plt.figure(figsize=(20,row*5))
    for i,v in enumerate(Vs):
        plt.subplot(row,col,i+1)
        idx = train_id[~train_id[v].isna()].index
        n = train_id[v].nunique()
        x = np.sum(train_id.loc[idx,v]!=train_id.loc[idx,v].astype(int))
        y = np.round(100*np.sum(train_id[v].isna())/len(train_id),2)
        t = 'int'
        if x!=0: t = 'float'
        plt.title(v+' has '+str(n)+' '+t+' and '+str(y)+'% nan')
        plt.yticks([])
        h = plt.hist(train_id.loc[idx,v],bins=100)
        if len(h[0])>1: plt.ylim((0,np.sort(h[0])[-2]))
    plt.show()
make_plots2([x for x in ids if x not in cats])


cols = ['TransactionDT'] + [x for x in ids if x not in cats]
plt.figure(figsize=(15,15))
sns.heatmap(train_id[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
plt.title('ID1-ID38')
plt.show()


# 在Kaggle中运行前需先安装依赖
!pip install psutil seaborn --quiet
 
import numpy as np
import pandas as pd
import os
import gc
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import psutil


import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 新增：设置可视化风格
sns.set_style("whitegrid")

# path
path_train_transaction = "../input/ieee-fraud-detection/train_transaction.csv"
path_train_id = "../input/ieee-fraud-detection/train_identity.csv"
path_test_transaction = "../input/ieee-fraud-detection/test_transaction.csv"
path_test_id = "../input/ieee-fraud-detection/test_identity.csv"
path_sample_submission = '../input/ieee-fraud-detection/sample_submission.csv'
path_submission = 'sub_xgb_95.csv'


# 版本控制开关（保持原有逻辑）
BUILD95 = False
BUILD96 = True



str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5',
            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']


cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
        'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
        'M5', 'M6', 'M7', 'M8', 'M9']

# V COLUMNS 
v = [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]
v += [107, 108, 111, 115, 117, 120, 121, 123]  # maybe group, no NAN
v += [124, 127, 129, 130, 136]  # relates to groups, no NAN

# LOTS OF NAN BELOW
v += [138, 139, 142, 147, 156, 162]  # b1
v += [165, 160, 166]  # b1
v += [178, 176, 173, 182]  # b2
v += [187, 203, 205, 207, 215]  # b2
v += [169, 171, 175, 180, 185, 188, 198, 210, 209]  # b2
v += [218, 223, 224, 226, 228, 229, 235]  # b3
v += [240, 258, 257, 253, 252, 260, 261]  # b3
v += [264, 266, 267, 274, 277]  # b3
v += [220, 221, 234, 238, 250, 271]  # b3

v += [294, 284, 285, 286, 291, 297]  # relates to grous, no NAN
v += [303, 305, 307, 309, 310, 320]  # relates to groups, no NAN
v += [281, 283, 289, 296, 301, 314]  # relates to groups, no NAN
# v += [332, 325, 335, 338] # b4 lots NAN

cols += ['V' + str(x) for x in v]
dtypes = {}
for c in cols + ['id_0' + str(x) for x in range(1, 10)] + ['id_' + str(x) for x in range(10, 34)]:
    dtypes[c] = 'float32'
for c in str_type:
    dtypes[c] = 'category'


print("load data...")
X_train = pd.read_csv(path_train_transaction, index_col="TransactionID", dtype=dtypes, usecols=cols + ["isFraud"])
train_id = pd.read_csv(path_train_id, index_col="TransactionID", dtype=dtypes)
X_train = X_train.merge(train_id, how="left", left_index=True, right_index=True)

X_test = pd.read_csv(path_test_transaction, index_col="TransactionID", dtype=dtypes, usecols=cols)
test_id = pd.read_csv(path_test_id, index_col="TransactionID", dtype=dtypes)
X_test = X_test.merge(test_id, how="left", left_index=True, right_index=True)

# target
y_train = X_train["isFraud"]
del train_id, test_id, X_train["isFraud"]

print("X_train shape:{}, X_test shape:{}".format(X_train.shape, X_test.shape))




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc

# 转换时间特征
for i in range(1, 16):
    if i in [1, 2, 3, 5, 9]:
        continue
    X_train["D" + str(i)] = X_train["D" + str(i)] - X_train["TransactionDT"] / np.float32(60 * 60 * 24)
    X_test["D" + str(i)] = X_test["D" + str(i)] - X_test["TransactionDT"] / np.float32(60 * 60 * 24)

# 频率编码函数
def encode_FE(df1, df2, cols):
    for col in cols:
        # 列存在性检查
        if col not in df1.columns or col not in df2.columns:
            print(f"警告：列 {col} 不存在于数据集中，已跳过频率编码")
            continue
            
        # 自动类型转换
        if df1[col].dtype == 'float16' or df2[col].dtype == 'float16':
            print(f"警告：列 {col} 包含float16类型，已转换为float32")
            df1[col] = df1[col].astype('float32')
            df2[col] = df2[col].astype('float32')
        
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1  # 处理缺失值
        nm = col + "FE"
        
        
        df1[nm] = df1[col].map(vc).astype("float32")
        df2[nm] = df2[col].map(vc).astype("float32")
        print(f"已完成频率编码：{col}")

# 标签编码函数
def encode_LE(col, train=X_train, test=X_test, verbose=True):
    if col not in train.columns or col not in test.columns:
        print(f"警告：列 {col} 不存在于数据集中，已跳过标签编码")
        return
    
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = pd.factorize(df_comb)
    nm = col
    if df_comb.max() > 32000:
        train[nm] = df_comb[: len(train)].astype("float32")
        test[nm] = df_comb[len(train):].astype("float32")
    else:
        train[nm] = df_comb[: len(train)].astype("float16")
        test[nm] = df_comb[len(train):].astype("float16")
    del df_comb
    gc.collect()
    if verbose:
        print(col)

# 组合特征函数
def encode_CB(col1, col2, df1=X_train, df2=X_test):
    nm = col1 + '_' + col2
    # 检查列是否存在
    if col1 not in df1.columns or col2 not in df1.columns:
        print(f"警告：列 {col1} 或 {col2} 不存在于训练集，已跳过组合特征创建")
        return
    if col1 not in df2.columns or col2 not in df2.columns:
        print(f"警告：列 {col1} 或 {col2} 不存在于测试集，已跳过组合特征创建")
        return
    
    
    df1[nm] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str) + '_' + df2[col2].astype(str)
    encode_LE(nm, verbose=False)
    print(f"已创建组合特征：{nm}")

# 聚合特征函数
def encode_AG(main_columns, uids, aggregations=["mean"], df_train=X_train, df_test=X_test, fillna=True, usena=False):
    for main_column in main_columns:
        for col in uids:
            # 动态生成新列名
            new_columns = [f"{main_column}_{col}_{agg}" for agg in aggregations]
            
            temp_df = pd.concat([df_train[[col, main_column]], df_test[[col, main_column]]])
            if usena and main_column in temp_df.columns:
                temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan

            # 添加类型检查
            if temp_df[col].dtype == 'float16':
                temp_df[col] = temp_df[col].astype('float32')
                
            agg_df = temp_df.groupby(col)[main_column].agg(aggregations).reset_index()
            
            # 确保列名匹配
            agg_df.columns = [col] + new_columns
            
            # 添加映射字典类型检查
            if agg_df[col].dtype == 'float16':
                agg_df[col] = agg_df[col].astype('float32')
                
            map_dict = agg_df.set_index(col)[new_columns].to_dict(orient="index")
            
            # 展开映射字典
            for i, agg in enumerate(aggregations):
                new_column = new_columns[i]
                # 修复FutureWarning
                df_train[new_column] = df_train[col].map(lambda x: map_dict.get(x, {}).get(agg, np.nan)).astype("float32")
                df_test[new_column] = df_test[col].map(lambda x: map_dict.get(x, {}).get(agg, np.nan)).astype("float32")
                
                if fillna:
                    # 修复FutureWarning
                    df_train[new_column] = df_train[new_column].fillna(-1)
                    df_test[new_column] = df_test[new_column].fillna(-1)
            print(f"已创建聚合特征：{new_columns}")

print("开始特征工程...")
# 创建cents特征
X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
print('cents特征创建完成')

# 频率编码
encode_FE(X_train, X_test, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])

# 组合特征创建
encode_CB('card1', 'addr1')
encode_CB('card1_addr1', 'P_emaildomain')

# 验证组合特征
required_columns = ['card1_addr1', 'card1_addr1_P_emaildomain']
for col in required_columns:
    if col not in X_train.columns:
        print(f"警告：训练集缺失列 {col}")
    if col not in X_test.columns:
        print(f"警告：测试集缺失列 {col}")

# 频率编码组合特征
encode_FE(X_train, X_test, ['card1_addr1', 'card1_addr1_P_emaildomain'])

# 聚合特征（修复后的版本）
encode_AG(['TransactionAmt', 'D9', 'D11'], 
          ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], 
          ['mean', 'std'], usena=False)

# 标签编码（字符串类型列）
str_type = ['ProductCD', 'M4', 'P_emaildomain', 'R_emaildomain', 'card1', 'card2']  # 示例字符串列
available_str_cols = [col for col in str_type if col in X_train.columns and col in X_test.columns]
missing_str_cols = [col for col in str_type if col not in X_train.columns or col not in X_test.columns]

print("\n存在的字符串列：", available_str_cols)
print("缺失的字符串列：", missing_str_cols)

for col in available_str_cols:
    encode_LE(col, X_train, X_test)



#确保保留目标变量
cols =[c for c in cols if c !='isFraud']

#移除其他指定列
for c in ['C3','M5','id_08','id_33']:
    if c in cols:
        cols.remove(c)
for c in ['card4','id_07','id_14','id_21','id_30','id_32','id_34']:
    if c in cols:
        cols.remove(c)
for c in ['id_'+str(x)for x in range(22,28)]:
    if c in cols:
        cols.remove(c)
        
#新增：确保列在两个数据集中都存在
valid_cols =[]
for col in cols:
    if col in X_train.columns and col in X_test.columns:
        valid_cols.append(col)
    else:
        print(f"警告：列{col}不同时存在于训练集和测试集，已移除")
cols =valid_cols

print('NOW USING THE FOLLOWING',len(cols),'FEATURES.')



#划分训练集和验证集
idxT =X_train.index[:3 *len(X_train)//4]
idxV =X_train.index[3 *len(X_train)//4:]

print(X_train.info())

#修改后的类型转换(添加异常处理)
str_type =['ProductCD','M4','P_emaildomain','R_emaildomain','card1','card2']
available_str_cols =[col for col in str_type if col in X_train.columns and col in X_test.columns]

for col in available_str_cols:
    try:
         #先尝试转换为整型
         X_train[col]=pd.to_numeric(X_train[col],errors='coerce')
         X_test[col]=pd.to_numeric(X_test[col],errors='coerce')

         #检查转换后的数据类型
         if X_train[col].dtype.kind in'iuf':
             print(f"成功转换列{col}为数值类型")
         else:
              #如果转换失败则回退到标签编码
              print(f"列{col}包含非数值内容，使用标签编码")
              encode_LE(col,X_train,X_test,verbose=False)
             
    except Exception as e:
        print(f"处理列{col}时出错：{str(e)}")
        print(f"列{col}示例值：{X_train[col].unique()[:5]}")
        print(f"列{col}回退到标签编码")
        encode_LE(col,X_train,X_test,verbose=False)

print("after transform:")
print(X_train.info())


#安全填充缺失值
for col in cols:
     if col not in X_train.columns or col not in X_test.columns:
         print(f"警告：跳过不存在于两个数据集中的列{col}")
         continue
         
     if X_train[col].dtype.name =='category':
          #处理分类列：先添加缺失值类别
          if -1 not in X_train[col].cat.categories:
              X_train[col]=X_train[col].cat.add_categories(-1)
          if -1 not in X_test[col].cat.categories:
              X_test[col]=X_test[col].cat.add_categories(-1)
          #填充缺失值
          X_train[col]=X_train[col].fillna(-1).astype('category')
          X_test[col]=X_test[col].fillna(-1).astype('category')
     else:
          #处理数值列
          X_train[col]=X_train[col].fillna(-1).astype('float32')
          X_test[col]=X_test[col].fillna(-1).astype('float32')
import xgboost as xgb
print("XGBoost version:",xgb.__version__)

# 临时恢复标签列进行分析
analysis_df = X_train.copy()
analysis_df['isFraud'] = y_train.values

# 时间转换（直接在analysis_df上操作）
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
analysis_df['DT_M'] = analysis_df['TransactionDT'].apply(
    lambda x: (START_DATE + datetime.timedelta(seconds=x))
)
analysis_df['DT_M'] = (analysis_df['DT_M'].dt.year - 2017) * 12 + analysis_df['DT_M'].dt.month
 
#时间特征创建
X_train =X_train.copy()
#先转换为datetime类型
X_train['TransactionDT']=pd.to_datetime(X_train['TransactionDT'],unit='s')
X_train['DT_M']=(
    (X_train['TransactionDT']-START_DATE)
    .dt.days //30 +1 #保持整数月份计算
)
X_test =X_test.copy()
X_test['TransactionDT']=pd.to_datetime(X_test['TransactionDT'],unit='s')
X_test['DT_M']=(
    (X_test['TransactionDT']-START_DATE)
    .dt.days //30 +1
)



#新增：恢复标签列用于分析
analysis_df =X_train.copy()
#假设y_train是已定义的目标变量(需要确保实际存在)
if 'y_train'in locals():
    analysis_df['isFraud']=y_train.values
else:
    raise NameError("需要先定义y_train目标变量")




# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import gc
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import platform

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def reduce_mem_usage(df):
    """内存优化函数（修复版：跳过datetime类型）"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type.kind in ['M', 'm']:
            continue
            
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'[Kaggle] Memory usage reduced from {start_mem:.2f}MB to {end_mem:.2f}MB ({1 - end_mem/start_mem:.1%} reduction)')
    return df

def auto_detect_start_date(df, time_col='TransactionDT'):
    """自动检测数据中的最小时间戳作为基准日期"""
    min_time = pd.to_datetime(df[time_col].min(), unit='s')
    return min_time - pd.DateOffset(months=3)

def check_gpu_availability():
    """检测GPU可用性"""
    try:
        xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
        return True
    except xgb.XGBoostError:
        return False


GPU_AVAILABLE = check_gpu_availability()
TREE_METHOD = 'gpu_hist' if GPU_AVAILABLE else 'hist'
print(f"Detected tree method: {TREE_METHOD} (GPU available: {GPU_AVAILABLE})")

KAGGLE_PATH = '/kaggle/input/ieee-fraud-detection/'
OUTPUT_PATH = '/kaggle/working/'
SAMPLE_SUB = f'{KAGGLE_PATH}sample_submission.csv'
TRAIN_DATA = f'{KAGGLE_PATH}train_transaction.csv'
TEST_DATA = f'{KAGGLE_PATH}test_transaction.csv'

print('[Kaggle] Loading data...')
X_train = pd.read_csv(TRAIN_DATA)
X_test = pd.read_csv(TEST_DATA)
y_train = X_train['isFraud']
X_train.drop('isFraud', axis=1, inplace=True)

# 内存优化
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)

# 时间特征处理
START_DATE = auto_detect_start_date(X_train)
X_train = pd.concat([
    X_train,
    pd.DataFrame({
        'TransactionDT': pd.to_datetime(X_train['TransactionDT'], unit='s'),
        'day': (pd.to_datetime(X_train['TransactionDT'], unit='s') - START_DATE).dt.days.astype(int),
        'DT_M': ((pd.to_datetime(X_train['TransactionDT'], unit='s') - START_DATE).dt.days // 30 + 1).astype(int)
    })
], axis=1)

X_test = pd.concat([
    X_test,
    pd.DataFrame({
        'TransactionDT': pd.to_datetime(X_test['TransactionDT'], unit='s'),
        'day': (pd.to_datetime(X_test['TransactionDT'], unit='s') - START_DATE).dt.days.astype(int),
        'DT_M': ((pd.to_datetime(X_test['TransactionDT'], unit='s') - START_DATE).dt.days // 30 + 1).astype(int)
    })
], axis=1)

# 预处理数据
def preprocess_data(train, test):
    remove_cols = ['C3', 'M5', 'id_08', 'id_33', 'card4', 'id_07', 'id_14', 
                  'id_21', 'id_30', 'id_32', 'id_34'] + [f'id_{i}' for i in range(22,28)]
    
    valid_remove = [c for c in remove_cols if c in train.columns]
    train.drop(columns=valid_remove, inplace=True)
    test.drop(columns=valid_remove, inplace=True)

    valid_cols = []
    for col in train.columns:
        if col in test.columns:
            valid_cols.append(col)
        else:
            print(f"警告：列 {col} 不同时存在于训练集和测试集，已移除")
    
    def encode_LE(col, train, test, verbose=True):
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined.astype(str).fillna('Missing'))
        train[col] = le.transform(train[col].astype(str).fillna('Missing'))
        test[col] = le.transform(test[col].astype(str).fillna('Missing'))
        if verbose:
            print(f"编码 {col}: {len(le.classes_)} 类")
        return le

    str_type = ['ProductCD', 'M4', 'P_emaildomain', 'R_emaildomain', 'card1', 'card2']
    available_str_cols = [col for col in str_type if col in valid_cols]
    
    le_dict = {}
    for col in available_str_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le_dict[col] = encode_LE(col, train, test)
    
    for col in valid_cols:
        if col in ['TransactionDT', 'DT_M', 'day', 'uid']:
            continue
            
        if col in ['ProductCD', 'M4', 'P_emaildomain', 'R_emaildomain', 'card1', 'card2']:
            train[col] = train[col].astype(int)
            test[col] = test[col].astype(int)
        else:
            train[col] = pd.to_numeric(train[col], errors='coerce').fillna(-1).astype('float32')
            test[col] = pd.to_numeric(test[col], errors='coerce').fillna(-1).astype('float32')
    
    return train, test, valid_cols

X_train, X_test, cols = preprocess_data(X_train, X_test)

# 特征选择逻辑
cols = list(X_train.columns)
remove_cols = ['TransactionDT'] + [f'D{i}' for i in range(6,15)]  # 移除D6-D14
remove_cols += ['C3', 'M5', 'id_08', 'id_33', 'card4']
remove_cols += ['id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']
remove_cols += [f'id_{i}' for i in range(22,28)]

for c in remove_cols:
    if c in cols:
        cols.remove(c)

essential_cols = ['ProductCD', 'M4', 'P_emaildomain', 'R_emaildomain', 
                 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']
for c in essential_cols:
    if c not in cols:
        cols.append(c)

print('NOW USING THE FOLLOWING', len(cols), 'FEATURES.')
print(np.array(cols))

# 创建分析用数据集
analysis_df = pd.concat([
    X_train.assign(isFraud=y_train.values),
    X_test.assign(isFraud=np.nan)
])

# 特征工程改进
def create_uid(df):
    if 'D1' in df.columns:
        df = df.copy()
        df['D1'] = df['D1'].clip(lower=0)
        df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        df['day_diff'] = (df['day'] - df['D1']).clip(lower=0).astype(int)
        df['uid'] = (df['card1_addr1'] + '_' + 
                    df['day_diff'].astype(str)).apply(lambda x: hash(x) % 10000)
    return df

X_train = create_uid(X_train)
X_test = create_uid(X_test)

# 最终特征列确认
cols = [c for c in cols if c not in ['TransactionDT', 'DT_M', 'day']]

# 模型训练配置
def train_model():
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    groups = X_train['DT_M']
    class_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
    skf = GroupKFold(n_splits=6)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train, groups=groups)):
        print(f'\n[Kaggle] Fold {fold+1} withholding month {X_train.iloc[val_idx]["DT_M"].iloc[0]}')
        
        X_t, X_v = X_train[cols].iloc[train_idx], X_train[cols].iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric='auc',
            scale_pos_weight=class_weight,
            tree_method=TREE_METHOD,
            enable_categorical=True,
            n_jobs=-1,
            random_state=42
        )
        
        h = clf.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=100,
            early_stopping_rounds=200
        )
        
        oof[val_idx] = clf.predict_proba(X_v)[:, 1]
        preds += clf.predict_proba(X_test[cols])[:, 1] / skf.n_splits
        fold_scores.append(h.best_score)
        
        del X_t, X_v, y_t, y_v, h, clf
        gc.collect()
    
    return oof, preds

# 执行训练
warnings.filterwarnings('ignore')
oof, preds = train_model()

# 结果保存
print(f'\n[Kaggle] Final OOF AUC: {roc_auc_score(y_train, oof):.5f}')
sample_submission = pd.read_csv(SAMPLE_SUB)
sample_submission['isFraud'] = preds
sample_submission.to_csv(f'{OUTPUT_PATH}submission.csv', index=False)
print(f'[Kaggle] Submission saved to {OUTPUT_PATH}submission.csv')

# 内存最终清理
del X_train, X_test, y_train
gc.collect()

