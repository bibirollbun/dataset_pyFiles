import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head()


df.info()


df.drop(columns=['id'], inplace=True)


df.isnull().sum()


df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2


df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})


for col in df.columns:
    plt.scatter(df[col], df['Calories'])
    plt.xlabel(col)
    plt.ylabel('Calories')
    plt.show()


sns.countplot(x=pd.cut(df['Weight'], bins=20, labels=False), data=df,hue='Sex')
plt.show()


sns.lineplot(x='Height',y='Weight',data=df)
plt.show()


sns.lineplot(x='Duration',y='Heart_Rate',data=df)
plt.show()


sns.lineplot(x='Duration',y='Body_Temp',data=df)
plt.show()


sns.lineplot(x='Heart_Rate',y='Body_Temp',data=df)
plt.show()


for col in df.columns:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()


from scipy.stats import mstats

features = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']
for col in features:
    df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])


df.corr()


y = df['Calories']
X = df.drop(['Calories'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.feature_selection import SelectKBest, chi2

# 使用卡方检验选择前 2 个最佳特征
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X_train, y_train)
print("选择后的特征形状：", X_new.shape)
print("每个特征的得分：", selector.scores_)
print("是否被选择：", selector.get_support())

# 输出每个特征的得分
scores = pd.Series(selector.scores_, index=X.columns)
scores = scores.sort_values(ascending=False)
print("卡方检验得分最高的特征：\n", scores.head(10))


df.drop(columns=['BMI', 'Sex', 'Weight', 'Height'], inplace=True)


y = df['Calories']
X = df.drop(['Calories'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
    
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

results = []

for name, model in models.items():
    y_train_log = np.log1p(y_train)
    model.fit(X_train_scaled, y_train_log)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MSE': mse,
        'R²': r2
    })
results_df = pd.DataFrame(results)
print(results_df)




