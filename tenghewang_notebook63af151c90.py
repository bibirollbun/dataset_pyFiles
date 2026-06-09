import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv(r"../input/GiveMeSomeCredit/cs-training.csv",index_col=0)
data.head()


# 查看形状
data.shape


# 查看数据集中每个字段的信息
data.info()


# 查看缺失值、异常值（最大最小）、分布情况。
data.describe()


# 检查数据中重复值的存在情况
data.duplicated().sum()


# 过滤重复值 inplace=True ，直接对原对象修改。
# data.drop_duplicates(inplace=True)
data = data.drop_duplicates()
# 检查过滤效果
data.duplicated().sum()


# 删除重复值后，需要重新恢复索引
data.index = range(data.shape[0])

# 最后在检查一下数据的信息，看看索引有没有正确恢复
data.info()


# 检查数据集中每一个字段的数据缺失情况
data.isnull().sum()


data.isnull().sum()/data.shape[0]


# 可以使用均值来填补NumberOfDependents字段的缺失值
# a = data['NumberOfDependents'].mean()
# data['NumberOfDependents'].fillna(a,inplace=True)


# 查看数据情况，选用已知列[0,1,2,3,4,6,7,8,9] 来预测 [5] 和 [10]
data.info()


from sklearn.ensemble import RandomForestRegressor

def fill_data(data_df,col_name,target_col):
  known = data_df[data_df[col_name].notnull()]
  unknown = data_df[data_df[col_name].isnull()]

  x_train = known.iloc[:,[0,1,2,3,4,6,7,8,9]]
  y_train = known.iloc[:,target_col]

  x_test = unknown.iloc[:,[0,1,2,3,4,6,7,8,9]]

  # 学习器太大比较慢
  clf = RandomForestRegressor(n_estimators=10,random_state=0)

  pred = clf.fit(x_train,y_train)
  pred = clf.predict(x_test)
  return pred

# 生成缺失值
pred_MonthlyIncome = fill_data(data,'MonthlyIncome',5)
pred_NumberOfDependents = fill_data(data,'NumberOfDependents',10)

# 填充缺失值 MonthlyIncome  NumberOfDependents
data.loc[data['MonthlyIncome'].isnull(),'MonthlyIncome'] = pred_MonthlyIncome
data.loc[data['NumberOfDependents'].isnull(),'NumberOfDependents'] = pred_NumberOfDependents

#再次检查缺失值的情况
data.info()


# 描述性统计
data.describe([0.01,0.1,0.25,.5,.75,.9,.99]).T


# age字段中异常值的处理
# 首先，我们可以查看一下年龄为0的人有多少
(data['age']==0).sum()


# 对于这样的异常值我们直接过滤掉即可
data = data[data['age'] != 0]


# 接下来处理NumberOfTime30-59DaysPastDueNotWorse，NumberOfTimes90DaysLate，NumberOfTime60-89DaysPastDueNotWorse
# 这三个字段中都存在非常大的异常值
data['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()


# 可以看到96和98明显属于异常值,共有225个。我们把他们全部过滤掉
data = data[data.loc[:,'NumberOfTime30-59DaysPastDueNotWorse']<90]


# 观察NumberOfTimes90DaysLate字段
data['NumberOfTimes90DaysLate'].value_counts()


# 观察NumberOfTime60-89DaysPastDueNotWorse字段
data['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()


# 这说明这三个字段的异常值来自于相同的记录。当我们进行针对其中一个进行过滤后，其余两个的异常值就解决了
data.info()


# 过滤数据之后需要恢复索引
data.index = range(data.shape[0])
data.info()


data['SeriousDlqin2yrs'].value_counts().plot(kind='pie')


import imblearn
from imblearn.over_sampling import SMOTE

# 我们采用过采样法来在原始数据集上进行采样，以得到一组均衡的数据
# 数据准备
X = data.iloc[:,1:]
y = data['SeriousDlqin2yrs']

# 合成少数类过采样技术 ， 随机种子42
sm = SMOTE(random_state=42)
X,y = sm.fit_sample(X,y)

# 绘制饼状图

y = pd.DataFrame(y)
y.value_counts().plot(kind='pie')


X.shape


data.shape


from sklearn.model_selection import train_test_split

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_train,X_vali,Y_train,Y_vali = train_test_split(X,y,test_size=0.3)


Y_train


# 将用于训练模型的数据的特征和标签合并在一起
model_data = pd.concat([Y_train,X_train],axis=1)
model_data


# 数据集在拆分的时候会被打乱顺序，所以我们在合并之后还是要恢复索引
model_data.index = range(model_data.shape[0])
model_data.columns = data.columns


model_data.head()


# 将用于验证模型的特征和标签合并在一起
vali_data = pd.concat([Y_vali,X_vali],axis=1)
# 恢复索引
vali_data.index = range(vali_data.shape[0])
vali_data.columns = data.columns


model_data['qcut'], updown = pd.qcut(model_data['age'],retbins=True,q=20)

# 对y为0的客户age进行分箱。
count_y0 = model_data[model_data['SeriousDlqin2yrs']==0].groupby(by='qcut').count()['SeriousDlqin2yrs']
# 对y为1的客户age进行分箱。
count_y1 = model_data[model_data['SeriousDlqin2yrs']==1].groupby(by='qcut').count()['SeriousDlqin2yrs']

# *zip 是zip 的反向操作
num_bins = [*zip(updown,updown[1:],count_y0,count_y1)]

columns = ['min','max','count_0','count_1']
df = pd.DataFrame(num_bins,columns=columns)
df.head()


updown


updown[1:]


count_y0


import scipy
def graph_for_best_bin(DF,X,Y,n=5,q=20,graph=True):
  """
  自动最优分箱函数，基于卡方检验
  
  参数：
  DF:需要输入的数据
  X:需要分箱的列名
  Y:分箱数据对应的标签
  n:保留分箱的个数
  q:初始分箱的个数
  graph:是否要画出IV图像
  
  区间为前开后闭
  
  """
  DF = DF[[X,Y]].copy()
  bins_df = 0
  
  # 先把数据分成q箱
  DF['qcut'], bins = pd.qcut(DF[X],retbins=True,q=q,duplicates='drop')
  count_y0 = model_data[DF[Y]==0].groupby(by='qcut').count()[Y]
  count_y1 = model_data[DF[Y]==1].groupby(by='qcut').count()[Y]
  num_bins = [*zip(bins,bins[1:],count_y0,count_y1)]
  
  # 如果某个箱子里有0，就进行合并。保证每个箱子中正负样本的数量都不为0
  for i in range(q):
    if 0 in num_bins[0][2:]:
      num_bins[0:2] = [(num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2]+num_bins[1][2],
                num_bins[0][3]+num_bins[1][3]
                )]
      continue
          
    for i in range(len(num_bins)):
      if 0 in num_bins[i][2:]:
        num_bins[i-1:i+1] = [(num_bins[i-1][0],
                    num_bins[i][1],
                    num_bins[i-1][2]+num_bins[i][2],
                    num_bins[i-1][3]+num_bins[i][3]
                    )]
        break
    else:
      break
  
  # 计算woe函数
  def get_woe(num_bins):
    columns = ['min','max','count_0','count_1']
    df = pd.DataFrame(num_bins,columns=columns)

    df['total'] = df.count_0 + df.count_1   #一个箱中样本的总数量
    df['percentage'] = df.total / df.total.sum()    #一个箱中的样本数占全部样本总数的比例
    df['bad_rate'] = df.count_1 / df.total  
    df['good%'] = df.count_0 / df.count_0.sum()
    df['bad%'] = df.count_1 / df.count_1.sum()
    df['woe'] = np.log(df['good%']/df['bad%'])
    return df
  
  # 计算IV值的函数
  def get_iv(df):
    rate = df['good%'] - df['bad%']
    iv = np.sum(rate * df.woe)
    return iv

  # 利用卡方值合并箱体，合并完毕之后计算woe与IV值。
  IV = []
  axisx = []

  while len(num_bins) > n:
    pvs = []
    # 获得num_bins两两之间的卡方值
    for i in range(len(num_bins)-1):
      x1 = num_bins[i][2:]
      x2 = num_bins[i+1][2:]
      pv = scipy.stats.chi2_contingency([x1,x2])[1]
      pvs.append(pv)

    # 合并p值最大的两组
    i = pvs.index(max(pvs))
    num_bins[i:i+2] = [( num_bins[i][0],
                num_bins[i+1][1],
                num_bins[i][2]+num_bins[i+1][2],
                num_bins[i][3]+num_bins[i+1][3])]
    bins_df = get_woe(num_bins)
    axisx.append(len(num_bins))
    IV.append(get_iv(bins_df))
  
  # 绘图
  if graph:
    plt.figure()
    plt.plot(axisx,IV)

    plt.xticks(axisx)
    plt.xlabel('number of box')
    plt.ylabel('IV value')
    plt.show()

  return bins_df


for i in model_data.columns[1:-1]:
  print(i)
  graph_for_best_bin(model_data,i,'SeriousDlqin2yrs',n=2,q=20)


#对于不能够进行自动分箱的特征，我们需要观察这些特征，手动设置这些特征的区间划分
hand_bins = {
    'NumberOfTime30-59DaysPastDueNotWorse':[0,1,2,13],
    'NumberOfTimes90DaysLate':[0,1,2,17],
    'NumberRealEstateLoansOrLines':[0,1,2,4,54],
    'NumberOfTime60-89DaysPastDueNotWorse':[0,1,2,8],
    'NumberOfDependents':[0,1,2,3]
}

#为了保证区间全覆盖，我们使用np.inf来替换区间的最大值，使用-np.inf来替换区间的最小值
hand_bins = {k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()}


hand_bins


# 对于能够使用函数进行自动分箱的特征，我们已经找到了他们的最佳分箱个数
auto_col_bins = {
    'RevolvingUtilizationOfUnsecuredLines':6,
    'age':5,
    'DebtRatio':4,
    'MonthlyIncome':3,
    'NumberOfOpenCreditLinesAndLoans':5
}

# 利用我们找到的最佳分箱个数来找到相应的区间划分
bins_of_col = {}

for col in auto_col_bins:
  bins_df = graph_for_best_bin(model_data,
                col,
                'SeriousDlqin2yrs',
                n=auto_col_bins[col],
                q=20,
                graph=False)
  
  # 将区间上界的集合与区间下界的集合合并，并且按照从小到大进行排序
  bins_list = sorted(set(bins_df['min']).union(bins_df['max']))
  
  # 为保证区间覆盖
  bins_list[0],bins_list[-1] = -np.inf, np.inf
  bins_of_col[col] = bins_list


# Python字典中的update方法：将字典2中的键值对更新到字典1中
bins_of_col.update(hand_bins)
bins_of_col


#一个新的get_woe函数，用于按照之前的区间划分进行分箱
#为每个分箱计算相应的woe值，用于之后的实际分箱之后的映射
def get_woe(df,col,y,bins):
  df = df[[col,y]].copy()
  df['cut'] = pd.cut(df[col],bins)
  bins_df = df.groupby('cut')[y].value_counts().unstack()
  woe = bins_df['woe'] = np.log((bins_df[0]/bins_df[0].sum()) / (bins_df[1]/bins_df[1].sum()))
  
  return woe


#将所有特征进行get_woe操作，然后，将计算出的结果存储到一个字典中
woeall = {}
for col in bins_of_col:
  woeall[col] = get_woe(model_data,col,'SeriousDlqin2yrs',bins_of_col[col])

woeall


# 将所有的woe值映射到原始数据中

# 由于不希望覆盖掉原本的数据，所以我们提前创建一个原始数据索引相同的DataFrame
model_woe = pd.DataFrame(index=model_data.index)

# 对所有特征进行分箱与woe映射操作
for col in bins_of_col:
  model_woe[col] = pd.cut(model_data[col],bins_of_col[col]).map(woeall[col])


model_woe


# 将标签补充到数据中
model_woe['SeriousDlqin2yrs'] = model_data['SeriousDlqin2yrs']
model_woe.head()


# 处理完训练集还要对测试集进行处理
vali_woe = pd.DataFrame(index=vali_data.index)

# 对测试集的所有特征进行分箱与woe映射操作
for col in bins_of_col:
  vali_woe[col] = pd.cut(vali_data[col],bins_of_col[col]).map(woeall[col])

# 将标签补充到数据中
vali_woe['SeriousDlqin2yrs'] = vali_data['SeriousDlqin2yrs']





vali_woe.head()


# 准备训练数据
x = model_woe.iloc[:,:-1]
y = model_woe.iloc[:,-1]

# 准备测试数据
vali_x = vali_woe.iloc[:,:-1]
vali_y = vali_woe.iloc[:,-1]


# 建立逻辑回归模型
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression()
lr.fit(x,y)


# 模型评估--正确率
lr.score(vali_x,vali_y)


from sklearn.metrics import confusion_matrix

# 混淆矩阵
vali_y_test = lr.predict(vali_x)
confusion_matrix(vali_y,vali_y_test)



(36936-4912)/36936
(35051-6677)/35051


# 模型评估--ROC曲线
# import scikitplot as skplt  # 这个包在google colab会出现问题，于是用sklearn替换。
from sklearn.metrics import roc_curve,plot_roc_curve

# 某个标签为正例和负例的概率
# vali_proba_df = pd.DataFrame(lr.predict_proba(vali_x))

# skplt.metrics.plot_roc(vali_y,
#             vali_proba_df,
#             plot_micro=False,
#             figsize=(6,6),
#             plot_macro=False
#             )
display = plot_roc_curve(lr, vali_x, vali_y)
print('type(display):',type(display))
plt.show()



B = 20 / np.log(2)
A = 600 + B * np.log(1/60)


# 基础分
base_score = A - B * lr.intercept_
base_score


# 我们可以通过循环，将所有特征的评分卡内容全部一次性的写入一个本地文件ScoreData.csv


# 将评分结果存入字典
data_dict={}
data_dict['base_score'] = base_score[0]

# file = "/content/ScoreData.csv"
# with open(file,"w") as fdata:
#   fdata.write("base_score,{}\n".format(base_score))



for i,col in enumerate(x.columns):
  score = woeall[col] * (-B*lr.coef_[0][i])
  # score.name = "Score"
  # score.index.name = col
  # score.to_csv(file,header=True,mode="a")
  item = {
      'key':score.index.to_list(),
      'value':score.to_list()
      }
  data_dict[col] = item


# 上面的操作将区放入key中，将区间所对应的值放入value中
print(data_dict['DebtRatio'])
data_dict['base_score']






# 先看一下我们所拥有的资源
data_dict
len(data_dict)


vali_x
vali_y
len(vali_x.iloc[1,:])


# 看眼第一行数据形式
vali_x.iloc[1,:]


# vali_x.iloc[1,:]['RevolvingUtilizationOfUnsecuredLines']


# 先编写一个测试函数，查看函数是否可行。
def pred_credit_score_test(data):
  score_num = data_dict['base_score']   # 总分
  for index,value in enumerate(data_dict['RevolvingUtilizationOfUnsecuredLines']['key']):
    if data['RevolvingUtilizationOfUnsecuredLines'] in value:
      score_num+=data_dict['RevolvingUtilizationOfUnsecuredLines']['value'][index]
      print(value)
      print(data['RevolvingUtilizationOfUnsecuredLines'])
      print(data_dict['RevolvingUtilizationOfUnsecuredLines']['value'][index])
      break
  print(data_dict['RevolvingUtilizationOfUnsecuredLines'])
pred_credit_score_test(vali_x.iloc[1,:])


# 查看函数是否可行。
def pred_credit_score(data):
  score_num = data_dict['base_score']
  col_name_list = ['RevolvingUtilizationOfUnsecuredLines',
            'age' ,
            'DebtRatio' ,
            'MonthlyIncome' ,
            'NumberOfOpenCreditLinesAndLoans' ,
            'NumberOfTime30-59DaysPastDueNotWorse' ,
            'NumberOfTimes90DaysLate'   ,
            'NumberRealEstateLoansOrLines'     ,
            'NumberOfTime60-89DaysPastDueNotWorse' ,
            'NumberOfDependents']
  for col_name in col_name_list:
    for index,value in enumerate(data_dict[col_name]['key']):
      if data[col_name] in value:
        score_num+=data_dict[col_name]['value'][index]
        break
  return score_num


# 喂入数据进行模拟
# 选取一行数据
user1 = vali_x.iloc[1,:]
user1_score = pred_credit_score(user1)
print(user1)
print('user1信用总分：',user1_score)




# 选取500到1000行的数据进行统计得分。
betch_data1 = vali_x.iloc[500:1000,:]
for i in range(len(betch_data1)):
  item = vali_x.iloc[i,:]
  user_score = pred_credit_score(item)
  print(i,user_score)



from joblib import dump, load
import os

model = lr
model_path = 'ckpt/lr/'




# 如果路径存在则直接加载模型，如果路径不存在。则训练模型后保存。
if os.path.exists(model_path):
  model = load(model_path+'/lr_ckpt.joblib')
else:
  model.fit(x,y)
  # 已经保存到本地。
  os.makedirs(model_path)
  dump(model,model_path+'/lr_ckpt.joblib')


model.score(vali_x,vali_y)


model2 = LogisticRegression()
model2 = load(model_path+'/lr_ckpt.joblib')
model2.score(vali_x,vali_y)


plot_roc_curve(model2, vali_x, vali_y)


data_kaggle = pd.read_csv(r"../input/GiveMeSomeCredit/cs-test.csv",index_col=0)
data_kaggle.head()


data_kaggle.isnull().sum()


# 生成缺失值
def fill_data_kaggle(data_df,col_name,target_col):
  known = data_df[data_df[col_name].notnull()]
  unknown = data_df[data_df[col_name].isnull()]

  x_train = known.iloc[:,[1,2,3,4,6,7,8,9]]
  y_train = known.iloc[:,target_col]

  x_test = unknown.iloc[:,[1,2,3,4,6,7,8,9]]

  # 学习器太大比较慢
  clf = RandomForestRegressor(n_estimators=10,random_state=0)

  pred = clf.fit(x_train,y_train)
  pred = clf.predict(x_test)
  return pred

pred_MonthlyIncome_kaggle = fill_data_kaggle(data_kaggle,'MonthlyIncome',5)
pred_NumberOfDependents_kaggle = fill_data_kaggle(data_kaggle,'NumberOfDependents',10)

# 填充缺失值 MonthlyIncome  NumberOfDependents
data_kaggle.loc[data_kaggle['MonthlyIncome'].isnull(),'MonthlyIncome'] = pred_MonthlyIncome_kaggle
data_kaggle.loc[data_kaggle['NumberOfDependents'].isnull(),'NumberOfDependents'] = pred_NumberOfDependents_kaggle


data_kaggle.info()


data_kaggle.describe([0.01,0.1,0.25,.5,.75,.9,.99]).T


# 处理完训练集还要对测试集进行处理
kaggle_woe =pd.DataFrame()# pd.DataFrame(index=vali_data.index)

# 对测试集的所有特征进行分箱与woe映射操作
for col in bins_of_col:
  kaggle_woe[col] = pd.cut(data_kaggle[col],bins_of_col[col]).map(woeall[col])


kaggle_woe.head()


kaggle_pred = model2.predict(kaggle_woe)

kaggle_pred[10:20]


# data_kaggle.loc[data_kaggle['SeriousDlqin2yrs'].isnull(),'SeriousDlqin2yrs']

result_data = data_kaggle.copy()
result_data.loc[result_data['SeriousDlqin2yrs'].isnull(),'SeriousDlqin2yrs'] = kaggle_pred

result_data.iloc[10:20,:]


result_data.to_csv('result_data.csv')

# ../input/GiveMeSomeCredit/cs-training.csv


result = result_data.iloc[:,0]

result.name = 'Probability'

# rename(column={'SeriousDlqin2yrs':'Probability'},inplace=True)

result

result.to_csv('result_data2.csv',index_label='Id')




