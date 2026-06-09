import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')

print(f'训练集：{train_df.shape}')
print(f'测试集：{test_df.shape}')

train_df.head()


train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# cat_col 
train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})


# BMI
train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)


# Age
bins = [19, 39, 59, 79]
labels = ['young', 'adult', 'senior']

train_df['Age_Quantile'] = pd.cut(train_df['Age'], bins=bins, labels=labels)
test_df['Age_Quantile'] = pd.cut(test_df['Age'], bins=bins, labels=labels)

train_df


# 基础代谢率
"""
男性：BMR = 10 * Weight + 6.25 * Height - 5 * Age + 5
女性：BMR = 10 * Weight + 6.25 * Height - 5 * Age - 161
"""
def calculate_bmr(row):
    if row['Sex'] == 'male': # 男性
        return 10 * row['Weight'] - 6.25 * row['Height'] - 5 * row['Age'] + 5
    else:
        return 10 * row['Weight'] - 6.25 * row['Height'] - 5 * row['Age'] - 161
    
train_df['BMR'] = train_df.apply(calculate_bmr, axis=1)
test_df['BMR'] = test_df.apply(calculate_bmr, axis=1)


# 运动强度
train_df['Intensity'] = train_df['Heart_Rate'] * train_df['Duration']
test_df['Intensity'] = test_df['Heart_Rate'] * train_df['Duration']
train_df


# 心率和体重比
train_df['HR_Temp'] = train_df['Heart_Rate'] / train_df['Body_Temp']
test_df['HR_Temp'] = test_df['Heart_Rate'] / train_df['Body_Temp']
train_df


train_df['Age_Quantile'] = train_df['Age_Quantile'].map({'young': 0, 'adult': 1, 'senior': 2})
test_df['Age_Quantile'] = test_df['Age_Quantile'].map({'young': 0, 'adult': 1, 'senior': 2})
train_df


train_df['Age_Quantile'] = train_df['Age_Quantile'].astype(int)
test_df['Age_Quantile'] = test_df['Age_Quantile'].astype(int)


X = train_df.drop('Calories', axis=1)
y = train_df['Calories']
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 6),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 0.5),
    'reg_lambda': uniform(0, 1)
}

# 初始化 XGBRegressor，并启用类别特征支持（如果需要）
xgb = XGBRegressor(enable_categorical=True)

# 使用 RandomizedSearchCV 进行随机搜索
random_search = RandomizedSearchCV(estimator=xgb,
                                  param_distributions=param_dist,
                                  n_iter=50,  
                                  scoring='neg_mean_squared_error',
                                  cv=5,
                                  verbose=1,
                                  n_jobs=-1)

# 拟合数据
random_search.fit(X, y)

# 输出得分
print("Best parameters found: ", random_search.best_params_)
print("Best RMSE: ", np.sqrt(-random_search.best_score_))


best_model = random_search.best_estimator_
sample = pd.read_csv(r'/kaggle/input/playground-series-s5e5/sample_submission.csv')
pred = best_model.predict(test_df)
pred[pred < 0] = 0 
sub = pd.DataFrame({
    'id':sample.id,
    'Calories':pred
})
sub.to_csv('submission_1.csv', index=False)




