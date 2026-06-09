import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 


train_data=pd.read_csv("/kaggle/input/predict-loan-default/train.csv")
test_data=pd.read_csv("/kaggle/input/predict-loan-default/test.csv")
sub_data=pd.read_csv("/kaggle/input/predict-loan-default/sample_submmission.csv")


train_data


print(pd.crosstab(train_data['STATE'],train_data['Risk_Flag']))
#生成交叉表，得到对应州和对应违约情况的样本数量


pd.crosstab(train_data['STATE'],train_data['Risk_Flag']).plot(kind='bar', stacked=True, colormap='viridis', figsize=(20,6))
                                                                                                                    
plt.title('Risk Flags Distribution by Profession')
plt.xlabel('STATE')
plt.ylabel('Count')
plt.legend(title="Risk Flags")
plt.xticks(rotation=45)
plt.show()#生成柱状图，直观看到不同州的贷款违约分布情况


from category_encoders.target_encoder import TargetEncoder

#对职业和城市这两个分类特征进行目标编码
encoder = TargetEncoder(cols=['Profession', 'CITY'])

# 让编码器学习每个职业/城市类别对应的违约概率均值
encoder.fit(train_data[['Profession', 'CITY']], train_data['Risk_Flag'])

# 用拟合好的编码器对训练集&测试集的'Profession'和'CITY'列进行转换，将分类值替换为对应的违约概率均值
train_data[['Profession', 'CITY']] = encoder.transform(train_data[['Profession', 'CITY']])
test_data[['Profession', 'CITY']] = encoder.transform(test_data[['Profession', 'CITY']]) 

#对于测试集中出现的训练集中没有的职业/城市类别,用训练集对应列的均值来填充这些缺失值
test_data[['Profession', 'CITY']] = test_data[['Profession', 'CITY']].fillna(train_data[['Profession', 'CITY']].mean())


train_data


test_data.head(20)


[train_data[col].unique() for col in train_data.columns if len(train_data[col].unique())<5]
#识别类别型特征：筛选训练数据中不同取值数量少于 5 个的列，并获取这些列的唯一值


from category_encoders.target_encoder import TargetEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder

# 针对婚姻状况、房屋所有权、车辆所有权这三个分类列，使用目标编码与留一编码
target_enc = TargetEncoder(cols=['Married.Single', 'House_Ownership', 'Car_Ownership'])
loo_enc = LeaveOneOutEncoder(cols=['Married.Single', 'House_Ownership', 'Car_Ownership'])

# 基于训练集的分类列和目标列'Risk_Flag'，学习每个类别对应的目标变量均值，并将训练集的分类列转换为数值
train_data_encoded = target_enc.fit_transform(train_data[['Married.Single', 'House_Ownership', 'Car_Ownership']], train_data['Risk_Flag'])
#复用训练集学到的编码规则，对测试集的分类列进行转换
test_data_encoded = target_enc.transform(test_data[['Married.Single', 'House_Ownership', 'Car_Ownership']]) 

# 用训练集编码结果的均值填充测试集中在训练集未见过的类别
test_data_encoded.fillna(train_data_encoded.mean(), inplace=True)

# 编码结果回写
train_data[['Married.Single', 'House_Ownership', 'Car_Ownership']] = train_data_encoded
test_data[['Married.Single', 'House_Ownership', 'Car_Ownership']] = test_data_encoded


test_data.head(20)


#绘制数值型特征列（收入、年龄、工作经验、当前工作年限、当前住房年限）的箱线图
numeric_columns = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=test_data[col])
    plt.title(f"{col}")
plt.tight_layout()
plt.show()


#识别训练数据中数值型特征的异常值
from scipy.stats import zscore

z_scores = train_data[numeric_columns].apply(zscore) 
outliers = (z_scores.abs() > 3).any(axis=1) 

print(f"Number of outliers: {outliers.sum()}")
df_outliers = train_data[outliers] 
df_outliers.head()


train_data['Risk_Flag'].value_counts().unique#训练集Risk_Flag列中每个取值的出现次数


test_data['Income_per_Experience'] = test_data['Income'] / (test_data['Experience'] + 1) 
train_data['Income_per_Experience'] = train_data['Income'] / (train_data['Experience'] + 1) # +1避免除0
# 年龄与工作年限的差值（反映工作开始时间）
train_data['Age_minus_Experience'] = train_data['Age'] - train_data['Experience']
test_data['Age_minus_Experience'] = test_data['Age'] - test_data['Experience']


test_data.head(20)


#划分机器学习任务中的特征和目标变量
X_train=train_data.drop(columns=['Id','STATE','Risk_Flag'])
y_train= train_data['Risk_Flag']
X_test=test_data.drop(columns=['Id','STATE'])


X_test.info()


import catboost as cb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 识别类别特征
cat_features = ['Married.Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY']

# 将类别特征列转换为字符串类型，缺失值填充为Missing
for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("Missing")
    X_test[col] = X_test[col].astype(str).fillna("Missing")

# 筛选出非类别特征的数值列，对大于 0 的数值进行对数转换
num_features = [col for col in X_train.columns if col not in cat_features]
for col in num_features:
    X_train[col] = X_train[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    X_test[col] = X_test[col].apply(lambda x: np.log1p(x) if x > 0 else 0)

# 将训练集拆分为训练子集和验证子集
X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 定义 CatBoost 模型
cat_model = cb.CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function='Logloss',
    cat_features=cat_features,
    eval_metric='F1',
    early_stopping_rounds=150,
    auto_class_weights='Balanced',
    l2_leaf_reg=10,
    bagging_temperature=1.0,
    max_bin=64,
    verbose=200
)

# 模型训练（带早停）
cat_model.fit(
    X_train_split, y_train_split,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=150,
    verbose=200
)
# 寻找最优阈值
y_valid_proba = cat_model.predict_proba(X_valid)[:, 1]
best_threshold = 0.5  # Default

for threshold in np.arange(0.1, 0.9, 0.05):
    y_valid_pred = (y_valid_proba > threshold).astype(int)
    score = f1_score(y_valid, y_valid_pred)
    if score > f1_score(y_valid, (y_valid_proba > best_threshold).astype(int)):
        best_threshold = threshold

print("Best Threshold:", best_threshold)

# 用最优阈值对测试集预测
y_test_pred = (cat_model.predict_proba(X_test)[:, 1] > best_threshold).astype(int)


submission = pd.DataFrame({
    'Id': sub_data['Id'],  
    'Risk_Flag': y_test_pred  
})

submission.to_csv("submission.csv", index=False)

print("✅ We're number one!")

