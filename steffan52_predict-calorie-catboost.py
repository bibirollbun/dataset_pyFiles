import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

train.head()


train2 = train.copy()


train = train.drop(columns='id')


train.info()


train.describe()


test.describe()


NUMS = train.columns.to_list()[1:]
NUMS


for i in NUMS:
    sns.histplot(data=train, x=i, bins=30, kde=True)
    plt.show()


train = pd.get_dummies(train, columns=['Sex'])
test = pd.get_dummies(test, columns=['Sex'])


X = train.drop(columns='Calories')
y = train['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


lin_reg = LinearRegression()
y_pred = lin_reg.fit(X_train, y_train).predict(X_test)
print('На тесте',mean_squared_error(y_test, y_pred))


dt = DecisionTreeRegressor(max_depth=11, min_samples_leaf=500, min_samples_split=2000)
y_pred = dt.fit(X_train, y_train).predict(X_test)
print('На тесте',mean_squared_error(y_test, y_pred))


rf = RandomForestRegressor(max_depth=10, min_samples_leaf=500, min_samples_split=2000, n_estimators=100)
y_pred = rf.fit(X_train, y_train).predict(X_test)
print('На тесте',mean_squared_error(y_test, y_pred))


cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, l2_leaf_reg=0.3, max_depth=7, min_data_in_leaf=1000, verbose=100)
y_pred = cat_model.fit(X_train, y_train).predict(X_test)
print('На тесте',mean_squared_error(y_test, y_pred))


comp_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, l2_leaf_reg=0.3, max_depth=7, min_data_in_leaf=1000, verbose=100, cat_features=['Sex'])


X_train = train2.drop(columns=['Calories', 'id'])
X_test = test.drop(columns='id')

y_train = train2['Calories']


y_pred = comp_model.fit(X_train, y_train).predict(X_test)


comp_predict = pd.DataFrame(
    {
        'id': test['id'],
        'Calories': y_pred
    }
)
comp_predict.head()


# comp_predict.to_csv('data/comp.csv', index=False)

