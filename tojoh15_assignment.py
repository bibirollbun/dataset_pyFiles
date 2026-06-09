import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.metrics import mean_squared_error


# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
# Note that we don't have any y_test! This way you cannot "cheat"!
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')


# Initialize
model = tree.DecisionTreeRegressor()

# Fit
model.fit(x_train, y_train)


y_test_hat = model.predict(x_test)
y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(x_test))),
    'Predicted': y_test_hat.reshape(-1),
})


# After you make your predictions, you should submit them on the Kaggle webpage for our competition.
# You may also (and I recommend you do it) send your code to me (at tsdj@sam.sdu.dk).
# Then I can provide feecback if you'd like (so ask away!).

# Below is a small check that your output has the right type and shape
assert isinstance(y_test_hat_pd, pd.DataFrame)
assert all(y_test_hat_pd.columns == ['Id', 'Predicted'])
assert len(y_test_hat_pd) == 200

# If you pass the checks, the file is saved.
y_test_hat_pd.to_csv('y_test_hat.csv', index=False)

