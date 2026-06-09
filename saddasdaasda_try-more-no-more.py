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


import pandas as pd
from sklearn.preprocessing import StandardScaler
# 导入数据
data2 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

# 转化为数据框
df2 = pd.DataFrame(data2)
ID1=df2['ID']
df2=df2.drop('ID',axis=1)
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 导入数据
data1 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')

# 转化为数据框
df1 = pd.DataFrame(data1)
df1=df1.drop('ID',axis=1)
# 排除目标列 'target'
features_df = df1.drop(columns=['PCOS'])

# 筛选分类型特征名称列表
category = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

# 筛选数值型特征名称列表
numeric = features_df.select_dtypes(exclude=['object', 'category']).columns.tolist()



# 执行Z标准化

# 更改数据类型
df1['PCOS']=df1['PCOS'].map({'No':0,'Yes':1})
df1['PCOS'] = df1['PCOS'].astype('category')
missingindex=df1[df1.isnull().any(axis=1)].index
df1.iloc[missingindex]
df1['Exercise_Type']=df1['Exercise_Type'].replace('No Exercise','No')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Cardio (e.g., running, cycling, swimming)','Cardio')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Cardio (e.g.','Cardio')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Flexibility and balance (e.g., yoga, pilates)','Flexible')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Strength training (e.g., weightlifting, resistance exercises)','Strength')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)','Cardio_strength')

df1['Exercise_Type']=df1['Exercise_Type'].replace('Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)','Cardio_flex')
df1['Exercise_Type']=df1['Exercise_Type'].replace('High-intensity interval training (HIIT)','HIIT')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)','Cardio_str_flex')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)','Str_flex')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Flexibility and balance (e.g., yoga, pilates), None','Flexible')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Cardio (e.g., running, cycling, swimming), None','Cardio')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Strength training','Strength')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Strength training (e.g.','Strength')
df1['Exercise_Type']=df1['Exercise_Type'].replace('Flexibility and balance (e.g.','Flexible')





df1['Exercise_Type'].value_counts()
df1['Conception_Difficulty']=df1['Conception_Difficulty'].replace('Yes, diagnosed by a doctor','Yes')
df1=df1.drop(df1.loc[df1['Conception_Difficulty']=='No, Yes, not diagnosed by a doctor'].index)
df1['Conception_Difficulty'].value_counts()
df2['Conception_Difficulty'].value_counts()
columns1=df1.columns.to_list()
columns1
columns2=df2.columns.to_list()
columns2
df2['Exercise_Type']=df2['Exercise_Type'].replace('No Exercise','No')
df2['Exercise_Type']=df2['Exercise_Type'].replace('Strength (e.g.','Strength')
df2['Exercise_Type']=df2['Exercise_Type'].replace('Strength training','Strength')
df2['Exercise_Type']=df2['Exercise_Type'].replace('Flexibility and balance (e.g.','Flexible')
df2['Exercise_Type']=df2['Exercise_Type'].replace('Strength (e.g.','Strength')
df2['Exercise_Type']=df2['Exercise_Type'].replace('Strength training (e.g.','Strength')






df2['Exercise_Type'].value_counts()
df1['Age']=df1['Age'].replace('30-25','25-30')
df1['Age']=df1['Age'].replace('15-20','Less than 20')
df1['Age']=df1['Age'].replace('Less than 20-25','Less than 20')


df1['Age'].value_counts()
df2['Age']=df2['Age'].replace('Less than 20-25','Less than 20')
df2['Age']=df2['Age'].replace('Less than 20)','Less than 20')
df2['Age']=df2['Age'].replace('30-30','30-35')
df2['Age']=df2['Age'].replace('25-25','20-25')
df2['Age']=df2['Age'].replace('45-49','45 and above')
df2['Age']=df2['Age'].replace('50-60','45 and above')

df2['Age']=df2['Age'].replace('22-25','20-25')
df2['Age']=df2['Age'].replace('20','20-25')


df2['Age'].value_counts()
df1['Hormonal_Imbalance']=df1['Hormonal_Imbalance'].replace('Yes Significantly','Yes')
df1=df1.drop(df1[df1['Hormonal_Imbalance']=='No, Yes, not diagnosed by a doctor'].index)
df1=df1.drop(df1[df1['Hirsutism']=='No, Yes, not diagnosed by a doctor'].index)

df1['Exercise_Frequency']=df1['Exercise_Frequency'].replace('Less than usual','Rarely')
df2[df2['Exercise_Frequency']=='30-35']
df2['Exercise_Frequency']=df2['Exercise_Frequency'].replace('1/2 Times a Week','1-2 Times a Week')
df2['Exercise_Frequency']=df2['Exercise_Frequency'].replace('Daily','6-8 Times a Week')
df2['Exercise_Frequency']=df2['Exercise_Frequency'].replace('Somewhat','Rarely')

df2['Exercise_Frequency']=df2['Exercise_Frequency'].replace('30-35','Rarely')
df2['Exercise_Frequency']=df2['Exercise_Frequency'].replace('Less than 6-8 Times a Week','3-4 Times a Week')

df1['Exercise_Duration']=df1['Exercise_Duration'].replace('20 minutes','Less than 30 minutes')
df1['Exercise_Duration']=df1['Exercise_Duration'].replace('30 minutes','30 minutes to 1 hour')
df1['Exercise_Duration']=df1['Exercise_Duration'].replace('45 minutes','30 minutes to 1 hour')
df1['Exercise_Duration']=df1['Exercise_Duration'].replace('More than 30 minutes','30 minutes to 1 hour')
df1['Exercise_Duration']=df1['Exercise_Duration'].replace('Less than 6 hours','Not Applicable')


df1['Exercise_Duration'].value_counts()

df2['Exercise_Duration']=df2['Exercise_Duration'].replace('45 minutes','30 minutes to 1 hour')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('30 minutes','30 minutes to 1 hour')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('40 minutes','30 minutes to 1 hour')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('20 minutes','Less than 30 minutes')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('Less than 20 minutes','Less than 30 minutes')



df2['Exercise_Duration']=df2['Exercise_Duration'].replace('Not Much','Not Applicable')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('3-4 Times a Week','Not Applicable')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('1-2 Times a Week','Not Applicable')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('Less than 6 hours','Not Applicable')
df2['Exercise_Duration']=df2['Exercise_Duration'].replace('6-8 hours','30 minutes to 1 hour')


df2['Sleep_Hours']=df2['Sleep_Hours'].replace('6-8 Times a Week','6-8 hours')
df2['Sleep_Hours']=df2['Sleep_Hours'].replace('20 minutes','6-8 hours')
df2['Sleep_Hours']=df2['Sleep_Hours'].replace('6-12 hours','9-12 hours')




df2['Sleep_Hours'].value_counts()


df1.isnull().sum()


for i in columns1:
    mode_value = df1[i].mode().values[0]
    df1[i]=df1[i].fillna(mode_value)
    


for i in columns2:
    mode_value = df2[i].mode().values[0]
    df2[i]=df2[i].fillna(mode_value)
    


df2.isnull().sum()


columns1


df1.dtypes


columns1version1=['Age',
 'PCOS',
 'Hormonal_Imbalance',
 'Hyperandrogenism',
 'Hirsutism',
 'Conception_Difficulty',
 'Insulin_Resistance',
 'Exercise_Frequency',
 'Exercise_Type',
 'Exercise_Duration',
 'Sleep_Hours',
 'Exercise_Benefit']



for i in columns1version1:
    df1[i]=df1[i].astype('category')


columns2version1=['Age',
 'Hormonal_Imbalance',
 'Hyperandrogenism',
 'Hirsutism',
 'Conception_Difficulty',
 'Insulin_Resistance',
 'Exercise_Frequency',
 'Exercise_Type',
 'Exercise_Duration',
 'Sleep_Hours',
 'Exercise_Benefit']


for i in columns2version1:
    df2[i]=df2[i].astype('category')


df3=df1[columns1version1]
df4=df2[columns2version1]


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 假设df3是你的数据集
# 示例数据创建


# 分离特征列和目标列
X = df3.drop('PCOS', axis=1)
y = df3['PCOS']

# 对特征列进行独热编码
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 创建决策树分类器
dt_classifier = DecisionTreeClassifier(random_state=42)

# 在训练集上训练模型
dt_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = dt_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率: {accuracy:.2f}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

# 假设df3是你的数据集
# 示例数据创建

# 分离特征列和目标列
X = df3.drop('PCOS', axis=1)
y = df3['PCOS']

# 对特征列进行独热编码
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 定义要调整的参数范围
param_grid = {
    'n_estimators': [1000],  # 树的棵数
    'max_depth': [3]  # 树的最大深度
}

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(random_state=42)

# 使用网格搜索来寻找最佳参数组合
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_rf_classifier = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_rf_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)

print(f"模型的准确率: {accuracy:.2f}")
print(f"模型的精确率: {precision:.2f}")
print(f"模型的 F1 分数: {f1:.2f}")
print(f"最佳参数组合: {grid_search.best_params_}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

# 假设df3是你的数据集
# 示例数据创建

# 分离特征列和目标列
X = df3.drop('PCOS', axis=1)
y = df3['PCOS']

# 对特征列进行独热编码
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 定义要调整的参数范围
param_grid = {
    'n_estimators': [300],  # 树的棵数
    'max_depth': [2],  # 树的最大深度
    'learning_rate': [0.1]  # 学习率
}

# 创建 XGBoost 分类器
xgb_classifier = XGBClassifier(random_state=42)

# 使用网格搜索来寻找最佳参数组合
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_xgb_classifier = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_xgb_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)

print(f"模型的准确率: {accuracy:.2f}")
print(f"模型的精确率: {precision:.2f}")
print(f"模型的 F1 分数: {f1:.2f}")
print(f"最佳参数组合: {grid_search.best_params_}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

# 假设df3是你的训练数据集
# 示例数据创建

# 分离特征列和目标列
X = df3.drop('PCOS', axis=1)
y = df3['PCOS']

# 对特征列进行独热编码，设置 handle_unknown='ignore' 以处理新类别
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 定义要调整的参数范围
param_grid = {
    'n_estimators': [1000],  # 树的棵数
    'max_depth': [3],  # 树的最大深度
    'learning_rate': [0.1]  # 学习率
}

# 创建 XGBoost 分类器
xgb_classifier = XGBClassifier(random_state=42)

# 使用网格搜索来寻找最佳参数组合
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_xgb_classifier = grid_search.best_estimator_

# 获取最佳模型
best_xgb_classifier = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_xgb_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)

print(f"模型的准确率: {accuracy:.2f}")
print(f"模型的精确率: {precision:.2f}")
print(f"模型的 F1 分数: {f1:.2f}")
print(f"最佳参数组合: {grid_search.best_params_}")


# 假设 df4 是新的测试数据集

# 使用训练集的编码器对新测试集进行编码
X4_encoded = encoder.transform(df4)

# 在新测试集上进行预测概率
y_pred_proba = best_xgb_classifier.predict_proba(X4_encoded)
positive_class_proba = pd.Series(y_pred_proba[:, 1], index=df4.index)




data10={
    'ID':ID1,
    'PCOS':positive_class_proba 
}
df10=pd.DataFrame(data10)
df10.to_csv('submission.csv',index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

# 假设df3是你的训练数据集
# 示例数据创建

# 分离特征列和目标列
X = df1.drop('PCOS', axis=1)
y = df1['PCOS']

# 对特征列进行独热编码，设置 handle_unknown='ignore' 以处理新类别
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 定义要调整的参数范围
param_grid = {
    'n_estimators': [1000],  # 树的棵数
    'max_depth': [3],  # 树的最大深度
    'learning_rate': [0.1]  # 学习率
}

# 创建 XGBoost 分类器
xgb_classifier = XGBClassifier(random_state=42)

# 使用网格搜索来寻找最佳参数组合
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_xgb_classifier = grid_search.best_estimator_

# 获取最佳模型
best_xgb_classifier = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_xgb_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)

print(f"模型的准确率: {accuracy:.2f}")
print(f"模型的精确率: {precision:.2f}")
print(f"模型的 F1 分数: {f1:.2f}")
print(f"最佳参数组合: {grid_search.best_params_}")



# 假设 df4 是新的测试数据集

# 使用训练集的编码器对新测试集进行编码
X4_encoded = encoder.transform(df2)

# 在新测试集上进行预测概率
y_pred_proba = best_xgb_classifier.predict_proba(X4_encoded)
positive_class_proba = pd.Series(y_pred_proba[:, 1], index=df2.index)




data11={
    'ID':ID1,
    'PCOS':positive_class_proba 
}
df11=pd.DataFrame(data11)
df11.to_csv('submission1.csv',index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

# 假设 df1 是你的训练数据集

# 分离特征列和目标列
X = df1.drop('PCOS', axis=1)
y = df1['PCOS']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义要调整的参数范围
param_grid = {
    'iterations': [300],  # 树的棵数
    'depth': [5],  # 树的最大深度
    'learning_rate': [0.1]  # 学习率
}

# 创建 CatBoost 分类器
catboost_classifier = CatBoostClassifier(random_seed=42, silent=True)

# 使用网格搜索来寻找最佳参数组合
grid_search = GridSearchCV(catboost_classifier, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train, cat_features=columns2version1)

# 获取最佳模型
best_catboost_classifier = grid_search.best_estimator_

# 获取最佳模型
best_xgb_classifier = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_xgb_classifier.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred)
# 计算 F1 分数
f1 = f1_score(y_test, y_pred)

print(f"模型的准确率: {accuracy:.2f}")
print(f"模型的精确率: {precision:.2f}")
print(f"模型的 F1 分数: {f1:.2f}")
print(f"最佳参数组合: {grid_search.best_params_}")




# 假设 df2 是新的测试数据集

# 在新测试集上进行预测概率
y_pred_proba = best_catboost_classifier.predict_proba(df2)
positive_class_proba = pd.Series(y_pred_proba[:, 1], index=df2.index)

print("新测试集正类的预测概率结果:")
print(positive_class_proba)

