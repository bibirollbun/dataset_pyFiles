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
import numpy as np
import itertools
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv(r"/kaggle/input/playground-series-s4e7/train.csv")
df_test = pd.read_csv(r'/kaggle/input/playground-series-s4e7/test.csv')


df_train.head()



# 设置显示选项，去掉科学计数法
pd.set_option('display.float_format', '{:.2f}'.format)

# 查看数据的描述性统计
df_train.describe()


df_train.shape


df_train.info()


df_test.info()


Gender_mapping = {'Male':1,'Female':0}
Vehicle_Age_mapping = {'< 1 Year':0, '1-2 Year':1, '> 2 Years':2}
Vehicle_Damage_mapping = {'Yes':1, 'No':0}


df_train['Gender'] = df_train['Gender'].map(Gender_mapping)
df_train['Vehicle_Age'] = df_train['Vehicle_Age'].map(Vehicle_Age_mapping)
df_train['Vehicle_Damage'] = df_train['Vehicle_Damage'].map(Vehicle_Damage_mapping)

df_test['Gender'] = df_test['Gender'].map(Gender_mapping)
df_test['Vehicle_Age'] = df_test['Vehicle_Age'].map(Vehicle_Age_mapping)
df_test['Vehicle_Damage'] = df_test['Vehicle_Damage'].map(Vehicle_Damage_mapping)


df_train.head(100)








df_train["Response"].value_counts().plot.pie(autopct='%1.1f%%', title="Pie Chart")
plt.show()



#计算相关系数矩阵
correlation_matrix = df_train.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()




numerical_features = ['Age', 'Region_Code', 'Annual_Premium', 'Vintage']

# 绘制数值特征的分布
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    df_train[feature].plot(kind='hist', bins=30, alpha=0.6, color='blue', edgecolor='black')
    
    # 绘制核密度估计（KDE）曲线
    df_train[feature].plot(kind='density', color='red')
    
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 假设你的 train 数据集已经存在
# train = pd.read_csv("your_data.csv")  # 你需要加载你的数据集

numerical_features = ['Age', 'Region_Code', 'Annual_Premium', 'Vintage']

# 绘制数值特征的分布
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图，设置适当的 bin 数量
    df_train[feature].plot(kind='hist', bins=20, alpha=0.6, color='blue', edgecolor='black', density=True)
    
    # 绘制核密度估计（KDE）曲线，设置平滑度
    df_train[feature].plot(kind='density', color='red', lw=2)
    
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()



fig, ax = plt.subplots(ncols=5, figsize=(20,8))
for i, col in enumerate(['Gender', 'Driving_License','Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']):
    n_bins =df_train[col].unique().shape[0]
    sns.histplot(df_train[col], color="gold", kde=True, bins=n_bins,
                 label='Train', ax=ax[i], legend=True)
    sns.histplot(df_test[col], color="crimson", kde=True, bins=n_bins,
                 label='Test', ax=ax[i], legend=True)
    #ax[i].title.set_text(col+" by Survived")
    ax[i].set_title("DISTRIBUTION OF {} IN TRAIN & TEST".format(col.upper())
                    , x=0.0, y=1.01, ha='left', fontweight=100, fontfamily='Lato', size=5)
    ax[i].legend(loc='upper left')


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(22, 8))

# 确保列名和参数拼写正确
kde = sns.kdeplot(x='Age', data=df_train, cut=0, hue='Response',
                  fill=True, palette='plasma_r')

# 设置刻度
kde.xaxis.set_major_locator(ticker.MultipleLocator(1))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

# 设置标题
fig.suptitle("AGE BY RESPONSE", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=39)

plt.show()



fig = plt.figure(figsize=(22,8))
hist = sns.histplot(df_train['Age'], color="springgreen", kde=True, bins=50, label='Train')
hist = sns.histplot(df_test['Age'], color="gold", kde=True, bins=50, label='Test')

title = fig.suptitle("DISTRIBUTION OF AGE IN TRAIN & TEST", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=39)

hist.xaxis.set_major_locator(ticker.MultipleLocator(1))
hist.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.legend()
plt.show()


fig = plt.figure(figsize=(22,8))
hist = sns.histplot(df_train['Age'], color="springgreen", kde=True, bins=50, label='Train')
hist = sns.histplot(df_test['Age'], color="gold", kde=True, bins=50, label='Test')

title = fig.suptitle("DISTRIBUTION OF AGE IN TRAIN & TEST", x=0.125, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=39)

hist.xaxis.set_major_locator(ticker.MultipleLocator(1))
hist.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.legend()
plt.show()





fig = plt.figure(figsize=(22,8))
kde = sns.kdeplot(x="Annual_Premium", data=df_train, cut=0, hue="Response", fill=True, legend=True, palette="mako_r")

kde.xaxis.set_major_locator(ticker.MultipleLocator(20000))
kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.suptitle("Annual_Premium BY RESPONSE", x=0.125, y=1.01
            , ha='left',fontweight=100, fontfamily='Lato', size=39);


fig = plt.figure(figsize=(20,8))
kde = sns.kdeplot(x="Annual_Premium", data=df_train, cut=0, clip=[15000,100000], hue="Response", fill=True, legend=True, palette="mako_r")

# 设置X轴的刻度间隔和范围
kde.xaxis.set_major_locator(ticker.MultipleLocator(5000))  # 根据数据密度调整这个值
kde.set_xlim([0, 100000])  # 设置X轴的显示范围

kde.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.suptitle("Annual_Premium BY RESPOMSE - CLIPPED TO REMOVE OUTLIERS", x=0.12, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=37)



fig = plt.figure(figsize=(20,8))
dist = sns.histplot(df_train[(df_train.Annual_Premium > 2630.00) & (df_train.Annual_Premium <=100000)]['Annual_Premium'],
                    color="gold", kde=True, bins=50, label='Train')
dist = sns.histplot(df_test[(df_test.Annual_Premium > 2630.00) & (df_test.Annual_Premium <=100000)]['Annual_Premium'],
                    color="crimson", kde=True, bins=50, label='Test')

title = fig.suptitle("DISTRIBUTION OF FARE IN TRAIN & TEST", x=0.12, y=1.01, ha='left',
             fontweight=100, fontfamily='Lato', size=37)

dist.xaxis.set_major_locator(ticker.MultipleLocator(5000))
dist.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.legend()
plt.show()



X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]


'''
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report
from bayes_opt import BayesianOptimization

# 初始化分类器XGBOOST分类器
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='auc')

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=666)

# 对原始数据使用交叉验证并生成报告
y_pred_original = cross_val_predict(xgb_clf, X, y, cv=kf)
print("Original Dataset Classification Report")
print(classification_report(y, y_pred_original))
'''


"""
# 贝叶斯优化函数
def xgb_eval(n_estimators, learning_rate, max_depth, colsample_bytree):
    params = {
        "n_estimators": int(round(n_estimators)),
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'colsample_bytree': colsample_bytree,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False
    }
    cv_result = cross_val_score(
        XGBClassifier(**params),
        X,
        y,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy'
    ).mean()
    return cv_result

# 贝叶斯优化
xgb_bo = BayesianOptimization(
    f=xgb_eval,
    pbounds={
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'colsample_bytree': (0.5, 1.0)
    },
    random_state=42
)

# 优化过程
xgb_bo.maximize(init_points=5, n_iter=30)

# 最优参数
best_params = xgb_bo.max['params']

# 格式化输出
print("Best XGBOOST Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value:.4f}")
"""


from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


#训练最优XGBOOST
xgb_tuned = XGBClassifier(random_state =42,
                         eval_metric = 'auc',
                         max_depth = 10,
                         n_estimators = 173,
                         learning_rate = 0.1295 ,
                         colsample_bytree = 0.5811  )

xgb_tuned.fit(X, y)


# 确保你得到的是预测的概率值
probabilities = xgb_tuned.predict_proba(df_test)

# 如果是二分类问题，获取第二类的概率
# 对于多分类任务，`probabilities[:, 1]` 代表的是第二类的概率
submission = pd.DataFrame({'id': df_test['id'], 'Response': probabilities[:, 1]})

# 创建输出文件
submission.to_csv('submission.csv', index=False)

# 提示保存成功
print("Your submission was successfully saved!")



submission


'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#生成混淆矩阵
conf_mat = confusion_matrix(y, y_pred_original)

#绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for XGBoost Classifier')
plt.show()
'''


'''import shap
#解释器实列化
explainer = shap.TreeExplainer(xgb_tuned)

#计算三个核心值
shap_values = explainer(df_test)  #打印的每个样本对于不同分类的各自的SHAP值， 索引是样本数量
shap_values2 = explainer.shap_values(df_test) #打印的是表格数据中每个特征对应的SHAP值，输出的数据格式和表格数据完全格式相同，索引是分类
shap_interaction_values = explainer.shap_intercation_values(df_test)
'''



'''
#绘制摘要图
shap.summary_plot(shap_values, df_test, plot_type='bar')
'''



'''
#不设置图片的的类型的话默认是群峰图
shap.summary_plot(shap_values, df_test)
'''


'''
#特征交互图
shap.summary_plot(shap_interaction_values, df_test)
'''



'''
#计算力图,力图是针对每一个样本的，不过可以力图叠加
shap.plots.force(shap_values[14)
shap.plot.force(shap_values[:20])
'''


'''
#瀑布图其实和力图是有点同质化的
shap.plots.waterfall(shap_values[14])
'''


'''
#特征交互影像图
shap.dependence_plot('I', shap_values2, df_test, interaction_index='H')
'''


'''
# 创建 shap.Explanation 对象
shap_expanation = shap.Explanation(values=shap_values[10:30],
                                  base_values=explainer.expected_value,
                                  data=df_test, feature_names=df_test.colums)
# 绘制热图
shap.plots.heatmap(shap_explantion)
'''





