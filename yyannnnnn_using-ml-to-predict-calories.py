import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_squared_log_error
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 載入資料集
train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# 增加BMI變數
train_data['BMI'] = train_data['Weight']/((train_data['Height']/100)**2)
test_data['BMI'] = test_data['Weight']/((test_data['Height']/100)**2)


train_data.head(5)


# 檢視訓練集和測試集之列數和欄數
print(train_data.shape)
print(test_data.shape)

# 檢視訓練集是否有遺失值
print(train_data.isnull().sum()) # 無遺失值
print(test_data.isnull().sum())


X_train = train_data.drop(columns = ['Calories'])
Y_train = train_data['Calories']

# 將提供之訓練集分割20%驗證集
X_train_data, X_val_data, Y_train_data, Y_val_data = train_test_split(X_train,Y_train,test_size = 0.2, random_state = 20250524)


X_train_data.head(5)


Y_train_data.head(5)


print(X_train_data.shape)
print(Y_train_data.shape)
print(X_val_data.shape)
print(Y_val_data.shape)


# 新的訓練集完整資料
NEW_train_data = X_train_data.copy()
NEW_train_data['Calories'] = Y_train_data.copy()
NEW_train_data.head(5)


NEW_train_data.describe()


sns.histplot(Y_train_data,kde = True)


# log1p轉換
Y_train_data_log1p = np.log1p(Y_train_data)
sns.histplot(Y_train_data_log1p, kde = True)


# box-cox 轉換
from scipy import stats

target_transformed, lambda_ = stats.boxcox(Y_train_data)

sns.histplot(target_transformed, kde = True)


NEW_train_data = pd.get_dummies(NEW_train_data,columns = ['Sex'],drop_first = True)
NEW_train_data.corr()

plt.figure(figsize = (12,8))
sns.heatmap(NEW_train_data.corr(),annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


X_train_data_vif = X_train_data.copy().drop(columns = ['Sex'])



def drop_high_vif_features(X, threshold=10.0, verbose=True):
    """
    自動移除 VIF 過高的變數，直到所有變數的 VIF < threshold
    參數:
        X         : 原始特徵資料（DataFrame）
        threshold : VIF 閾值（預設為10）
        verbose   : 是否印出每次移除的欄位與 VIF 表
    回傳:
        cleaned_X : 已移除高 VIF 的特徵資料
        dropped   : 被移除的欄位名稱列表
    """
    X = X.copy()
    X = X.select_dtypes(include=['float64', 'int64'])  # 避免 bool/object 問題
    dropped = []

    while True:
        X_const = add_constant(X)
        vif = pd.DataFrame()
        vif["Feature"] = X_const.columns
        vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

        max_vif = vif.loc[vif['Feature'] != 'const', 'VIF'].max()
        max_feature = vif.loc[vif['VIF'] == max_vif, 'Feature'].values[0]

        if max_vif > threshold:
            X = X.drop(columns=max_feature)
            dropped.append(max_feature)
            if verbose:
                print(f"❌ Dropped '{max_feature}' with VIF = {max_vif:.2f}")
        else:
            if verbose:
                print("\n✅ Final VIF table:")
                print(vif)
            break

    return X, dropped



X_cleaned, removed_vars = drop_high_vif_features(X_train_data_vif)

print("\n 移除的欄位:", removed_vars)


NEW_train_data


NEW_train_data['Sex_male'].value_counts()


sns.scatterplot(data = NEW_train_data,x = "Height",y = "Weight", hue = "Sex_male")
plt.title("Height vs Weight by Sex")
plt.show()
# plt.scatter(train_data['Height'],train_data['Weight'],c = train_data['Sex_code'])


sns.scatterplot(data = NEW_train_data,x = "Duration",y = "Heart_Rate", hue = "Body_Temp",size = "Body_Temp",style = "Sex_male",sizes = (20,200))
plt.title("Duration vs Heart_rate with Body_Temp by Sex")
plt.show()


sns.scatterplot(data = NEW_train_data,x = "Duration",y = "Heart_Rate", hue = "Calories",size = "Calories",style = "Sex_male",sizes = (20,200))
plt.title("Duration vs Calories with Heart_Rate by Sex")
plt.show()


sns.scatterplot(data = NEW_train_data,x = "Age",y = "Duration", hue = "Calories",size = "Calories",style = "Sex_male",sizes = (20,200))
plt.title("Age vs Duration with Calories by Sex")
plt.show()


sns.scatterplot(data = NEW_train_data,x = "Age",y = "BMI", hue = "Calories",size = "Calories",style = "Sex_male",sizes = (20,200))
plt.title("Age vs BMI with Calories by Sex")
plt.show()


sns.scatterplot(data = NEW_train_data,x = "Body_Temp",y = "Calories", hue = "Sex_male")
plt.title("Body_Temp vs Calories by Sex")
plt.show()


X_train_data = pd.get_dummies(X_train_data,columns = ['Sex'],drop_first = True)
X_val_data = pd.get_dummies(X_val_data,columns = ['Sex'],drop_first = True)
test_data = pd.get_dummies(test_data,columns = ['Sex'],drop_first = True)


NEW_X_train_data = X_train_data.drop(['Weight','Heart_Rate'],axis = 1)
NEW_test_data = test_data.drop(['Weight','Heart_Rate'],axis = 1)
NEW_X_train_data.head(5)


def rmsle (y_true,y_pred):
    y_pred = np.maximum(0,y_pred)
    return np.sqrt(mean_squared_log_error(y_true,y_pred))


linear_regr_model = LinearRegression()

# MSE
rmsle_scorer = make_scorer(rmsle, greater_is_better = False)

# 5-fold 交叉驗證
k_fold = KFold(n_splits = 5, shuffle = True, random_state = 20250524)

cv_mse_scores = cross_val_score(
    linear_regr_model,
    NEW_X_train_data,
    Y_train_data,
    scoring = rmsle_scorer,
    cv = k_fold
)

print("5-fold CV MSE", -cv_mse_scores.mean()) # 轉成正的mse
print("5-fold CV MSE 各折分數", -cv_mse_scores)


# 訓練模型
linear_regr_model.fit(NEW_X_train_data,Y_train_data)

#預測測試集
y_test_pred = linear_regr_model.predict(NEW_test_data)

y_test_pred = np.maximum(0,y_test_pred)
y_test_pred


test_data


submission = pd.DataFrame({
    'id' : test_data['id'],
    'Calories' : y_test_pred
})

submission.to_csv('/kaggle/working/submission.csv',index = False)


