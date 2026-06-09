from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


import pandas as pd


train = pd.read_csv('/kaggle/input/age-estimation-from-health-metrics-2025/train.csv')
train


train.info()


train.describe().T


import seaborn as sns
import matplotlib.pyplot as plt


plt.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), linewidths=.01, cmap="coolwarm", annot=True)
plt.title("Тепловая карта взаимокорреляций Пирсона для признаков")
plt.show()


# Корреляции всех признаков с Age
correlations = train.corr()["Age"].drop("Age").sort_values(ascending=False)

# Порог для отбора информативных признаков
threshold = 0.01
selected_features = correlations[correlations.abs() >= threshold].index.tolist()

# Визуализация
plt.figure(figsize=(12, len(selected_features) * 4))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(len(selected_features), 2, i)
    sns.regplot(data=train, x=feature, y="Age", line_kws={"color": "red"})
    plt.title(f"{feature} vs Age (corr = {correlations[feature]:.2f})")

plt.tight_layout()
plt.show()


y=train['Age']
X=train.drop(['Age'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


svr_reg = SVR()
svr_reg.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error
y_pred = svr_reg.predict(X_test)
mean_absolute_error(y_pred, y_test)


test = pd.read_csv('/kaggle/input/age-estimation-from-health-metrics-2025/test.csv')
X_test = test
y_pred = svr_reg.predict(X_test)
submission = pd.read_csv('/kaggle/input/age-estimation-from-health-metrics-2025/sample_submission.csv')
submission['Age'] = y_pred.round(0).astype(int)
submission.to_csv('submission.csv', index = False)

