import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))




df1 = pd.read_csv("/kaggle/input/playground-series-s3e5/train.csv")
df1.head()


df2 = pd.read_csv("/kaggle/input/playground-series-s3e5/test.csv")
df2.head()


df1.shape, df2.shape


df = pd.concat([df1,df2])


df.sample(4)


df.info()


df.describe().T


df.isnull().sum()


plt.figure(figsize=(20,12))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu");


df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
df['sulfur_dioxide_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-5)
df['alcohol_density_ratio'] = df['alcohol'] / df['density']
df['sugar_sulfur_ratio'] = df['residual sugar'] / (df['total sulfur dioxide'] + 1e-5)
df['alcohol_level'] = pd.cut(df['alcohol'], bins=[0, 9, 11, 14], labels=['Low', 'Medium', 'High'])
df['acid_density_interaction'] = df['total_acidity'] * df['density']
df['inverse_pH'] = 1 / (df['pH'] + 1e-5)


d ={"Low":1, "Medium":2, "High":3}
df['alcohol_level'] = df['alcohol_level'].map(d)


abs(df.corr(numeric_only=True)["quality"].sort_values(ascending=False))


df["quality"].value_counts()


train=df[:2056]
test=df[2056:]


train["quality"] = train["quality"].astype(int)


x = train.drop(["Id","quality"], axis=1)
y = train[["quality"]]
test = test.drop(["Id","quality"], axis=1)


scaler = MinMaxScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.12, random_state=8)


y_train = y_train.astype(int)
y_test = y_test.astype(int)


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(max_iter=1000),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, verbosity=0),
    "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.14, depth=6, verbose=0)
}

for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds_rounded = preds.round().astype(int)  
    acc = accuracy_score(y_test, preds_rounded)  
    print(f"ðŸ”¹ {name} Accuracy (rounded preds): {acc:.4f}")


best_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, verbosity=0)
best_model.fit(x, y)


test_preds = best_model.predict(test)
test_preds_rounded = test_preds.round().astype(int)

submission = pd.DataFrame({
    "Id": df["Id"][2056:],
    "quality": test_preds_rounded
})
submission.to_csv("submission.csv", index=False)
submission.sample(10)




