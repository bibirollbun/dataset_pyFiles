import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
data.isna().sum()


df = data.dropna()
[(i, df[i].unique().shape) for i in df.iloc[:, 1:]]


[(i, data[i].unique().shape) for i in data.where(data['num_sold']==np.nan).dropna(how='all').iloc[:, 1:-1]]


X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X['yyyy'] = [i.split('-')[0] for i in X['date']]
X['mm'] = [i.split('-')[1] for i in X['date']]
X['dd'] = [i.split('-')[2] for i in X['date']]
X = X.iloc[:, 1:]


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
X = oe.fit_transform(X)
X


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
y_train = sc_y.fit_transform(np.array(y_train).reshape(-1,1))
y_test = sc_y.transform(np.array(y_test).reshape(-1,1))


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

reg = XGBRegressor(n_estimators=500, max_depth=7, reg_lambda=0.2, alpha=1)
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
mean_absolute_percentage_error(y_test, y_pred)


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

test['yyyy'] = [i.split('-')[0] for i in test['date']]
test['mm'] = [i.split('-')[1] for i in test['date']]
test['dd'] = [i.split('-')[2] for i in test['date']]
test=test.iloc[:, 2:]
test.head()


newoe = OrdinalEncoder()
test = newoe.fit_transform(test)
test = sc_x.transform(test)
test


nums = sc_y.inverse_transform(reg.predict(test).reshape(-1,1))

sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub['num_sold'] = nums
sub.to_csv('submission.csv', index=False)

