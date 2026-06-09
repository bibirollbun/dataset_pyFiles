import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
print("ok")



def preprocess(df):
    # Convert date and extract features
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    
    # Feature engineering
    df['age'] = df['sale_year'] - df['year_built']
    df['renovated'] = np.where(df['year_reno'] > 0, 1, 0)
    df['years_since_reno'] = np.where(df['renovated'], df['sale_year'] - df['year_reno'], 0)
    df['total_baths'] = df['bath_full'] + 0.75*df['bath_3qtr'] + 0.5*df['bath_half']
    df['total_value'] = df['land_val'] + df['imp_val']
    df['living_area'] = df['sqft'] + df['sqft_fbsmt']

    return df
    
print("ok")


# load data
train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

train = preprocess(train)
test= preprocess(test)

cat_cols = ['sale_date','join_status', 'city', 'zoning','subdivision','submarket','sale_warning']
df_combined = pd.concat([train[cat_cols], test[cat_cols]], axis=0)

# fill null
for col in ['sale_nbr','subdivision','submarket']:
    train[col] = train[col].fillna(train[col].mode()[0])
for col in ['sale_nbr','subdivision','submarket']:
    test[col] = test[col].fillna(test[col].mode()[0])
    
# encoding cat_cols 
label_encoders = {}
for column in cat_cols:
    le = LabelEncoder()
    le.fit_transform(df_combined[column])
    label_encoders[column] = le

for column in cat_cols:
    train[column] = label_encoders[column].transform(train[column])
    test[column] = label_encoders[column].transform(test[column])




# train and test data
X = train.drop(["sale_price", "id"], axis=1)
y = train["sale_price"]
X_test = test.drop("id", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("ok")


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=True):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)

print("ok")


# XGB params
model_lower = XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.05,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    enable_categorical=True
)
model_upper = XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.95,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    enable_categorical=True

)

#model_lower.fit(X_train, y_train)
#model_upper.fit(X_train, y_train)

model_lower.fit(X, y)
model_upper.fit(X, y)

# val evaluate
val_pred_lower = model_lower.predict(X_val)
val_pred_upper = model_upper.predict(X_val)

coverage = np.mean((y_val >= val_pred_lower) & (y_val <= val_pred_upper))
print(f"Coverage: {coverage * 100:.2f}%") 

# Average Interval Width
interval_width = np.mean(val_pred_upper - val_pred_lower)
print(f"Average Interval Width: {interval_width:.2f}")

print("ok")


# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
 
# ===== LightGBM 模型 =====
# lgbm_lower = LGBMRegressor(
#     objective="quantile",
#     alpha=0.05,  # 5% 分位数
#     n_estimators=1500,
#     learning_rate=0.05,
#     max_depth=-1,
#     random_state=42,
#    # categorical_feature="auto"
#     min_data_in_leaf=20,  # 增加最小叶子样本数
#     # 不限制深度（或设为更大值）
#     min_gain_to_split=0.1,  # 增加最小分割增益
# )
#  base_params = dict(
#     boosting_type="gbdt",
#     num_leaves=128,
#     learning_rate=0.03,
#     n_estimators=3_000,
#     subsample=0.8,
#     subsample_freq=1,
#     colsample_bytree=0.75,
#     reg_alpha=0.5,
#     reg_lambda=10,
#     min_child_samples=50,
#     random_state=2025,
#     verbose=-1
# )
# lgbm_upper = LGBMRegressor(
#     objective="quantile",
#     alpha=0.95,  # 95% 分位数
#     n_estimators=1500,
#     learning_rate=0.05,
#     max_depth=-1,
#     random_state=42,
#     #categorical_feature="auto"
#         min_data_in_leaf=20,  # 增加最小叶子样本数
#     # 不限制深度（或设为更大值）
#     min_gain_to_split=0.1,  # 增加最小分割增益
# )


lgbm_lower = LGBMRegressor(
    boosting_type="gbdt",
    num_leaves=128,
    objective="quantile",
    alpha=0.05,  # 5% 分位数
    n_estimators=3_000,
    learning_rate=0.03,
    max_depth=-1,
    random_state=42,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=10,
    min_child_samples=50,
   # categorical_feature="auto"
    min_data_in_leaf=20,  # 增加最小叶子样本数
    # 不限制深度（或设为更大值）
    min_gain_to_split=0,  # 增加最小分割增益
)

lgbm_upper = LGBMRegressor(
    objective="quantile",
    boosting_type="gbdt",
    num_leaves=128,
    alpha=0.95,  # 95% 分位数
    n_estimators=3_000,
    learning_rate=0.03,
    max_depth=-1,
    random_state=42,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.5,
    reg_lambda=10,
    min_child_samples=50,
   # categorical_feature="auto"
    min_data_in_leaf=20,  # 增加最小叶子样本数
    # 不限制深度（或设为更大值）
    min_gain_to_split=0,  # 增加最小分割增益
)

print("Training LightGBM models...")
#lgbm_lower.fit(X_train, y_train)
#lgbm_upper.fit(X_train, y_train)
lgbm_lower.fit(X, y)
lgbm_upper.fit(X, y)
# ===== 验证集预测 =====

# LightGBM 预测
lgbm_val_lower = lgbm_lower.predict(X_val)
lgbm_val_upper = lgbm_upper.predict(X_val)
 

# LightGBM 评估
lgbm_coverage = np.mean((y_val >= lgbm_val_lower) & (y_val <= lgbm_val_upper))
lgbm_width = np.mean(lgbm_val_upper - lgbm_val_lower)
 

print("\n===== LightGBM Results =====")
print(f"Coverage: {lgbm_coverage * 100:.2f}%")
print(f"Average Interval Width: {lgbm_width:.2f}")
 
print("\nok")





##### from catboost import CatBoostRegressor
import numpy as np

# ===== CatBoost 模型 =====
cat_lower = CatBoostRegressor(
    loss_function="Quantile:alpha=0.05",  # 5% 分位数回归目标
    learning_rate=0.05,                  # 学习率
    iterations=6500,                     # 树的数量（配合早停）
    random_seed=42,                      # 固定随机种子
    verbose=800,                         # 每800次迭代打印日志
    grow_policy="Depthwise",             # 深度优先生长（更精细的树结构）
    min_data_in_leaf=1000,               # 叶子最小数据量（防过拟合）
    l2_leaf_reg=100,                     # L2 正则化系数
    od_type="IncToDec",                  # 过拟合检测器类型（递增到递减）
    od_pval=0.1,                         # 检测阈值
)

cat_upper = CatBoostRegressor(
    loss_function="Quantile:alpha=0.95",  # 5% 分位数回归目标
    learning_rate=0.05,                  # 学习率
    iterations=6500,                     # 树的数量（配合早停）
    random_seed=42,                      # 固定随机种子
    verbose=800,                         # 每800次迭代打印日志
    grow_policy="Depthwise",             # 深度优先生长（更精细的树结构）
    min_data_in_leaf=1000,               # 叶子最小数据量（防过拟合）
    l2_leaf_reg=100,                     # L2 正则化系数
    od_type="IncToDec",                  # 过拟合检测器类型（递增到递减）
    od_pval=0.1,                         # 检测阈值
)

print("Training CatBoost models...")
#cat_lower.fit(X_train, y_train)
#cat_upper.fit(X_train, y_train)
cat_lower.fit(X, y)
cat_upper.fit(X, y)

# ===== 验证集预测 =====
cat_val_lower = cat_lower.predict(X_val)
cat_val_upper = cat_upper.predict(X_val)

# ===== 评估 =====
cat_coverage = np.mean((y_val >= cat_val_lower) & (y_val <= cat_val_upper))
cat_width = np.mean(cat_val_upper - cat_val_lower)

print("\n===== CatBoost Results =====")
print(f"Coverage: {cat_coverage * 100:.2f}%")
print(f"Average Interval Width: {cat_width:.2f}")

print("\nok")


cat_score=winkler_score(y_val,cat_val_lower,cat_val_upper)

print(f"cat :{cat_score}")
print("ok")


lgbm_score=winkler_score(y_val,lgbm_val_lower,lgbm_val_upper)

print(f"lgbm :{lgbm_score}")#lgbm :(271685.361112528, 0.9014)

print("ok")


xgb_score=winkler_score(y_val,val_pred_lower,val_pred_upper)

print(f"xgb :{xgb_score}")
print("ok")


#only use cat and lgbm
#end_lower=(cat_val_lower-8000+lgbm_val_lower-5000+val_pred_lower-5000)/3
#end_upper=(cat_val_upper+8000+lgbm_val_upper+5000+val_pred_upper+5000)/3
#end_lower=(cat_val_lower+lgbm_val_lower+val_pred_lower)/3
#end_upper=(cat_val_upper+lgbm_val_upper+val_pred_upper)/3
end_lower=(cat_val_lower+lgbm_val_lower)/2
end_upper=(cat_val_upper+lgbm_val_upper)/2

score=winkler_score(y_val,end_lower,end_upper)

print(f"means：{score}")#means：(273520.8924108883, 0.911125)

print("ok")


xgb_test_lower = model_lower.predict(X_test)
xgb_test_upper = model_upper.predict(X_test)

cat_test_lower = cat_lower.predict(X_test)
cat_test_upper = cat_upper.predict(X_test)

lgbm_test_lower = lgbm_lower.predict(X_test)
lgbm_test_upper = lgbm_upper.predict(X_test)
print("ok")


# predic test
#test_pred_lower=(cat_test_lower-8000+lgbm_test_lower-5000+xgb_test_lower-5000)/3
#test_pred_upper=(cat_test_upper+8000+lgbm_test_upper+5000+xgb_test_upper+5000)/3
test_pred_lower=(cat_test_lower+lgbm_test_lower)/2
test_pred_upper=(cat_test_upper+lgbm_test_upper)/2


# save submission file
submission = pd.DataFrame({
    "id": test["id"],
    "pi_lower": test_pred_lower,
    "pi_upper": test_pred_upper
})
submission.to_csv("submission.csv", index=False)
print("Submission file generated: submission.csv")

plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_pred_lower, color="blue", alpha=0.3, label="Lower Bound")
plt.scatter(y_val, val_pred_upper, color="red", alpha=0.3, label="Upper Bound")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "k--", label="Perfect Prediction")
plt.xlabel("True Sale Price")
plt.ylabel("Predicted Bounds")
plt.title("Prediction Intervals on Validation Set")
plt.legend()
plt.show()
submission.head()




