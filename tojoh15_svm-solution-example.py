import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# Load data (must be in same folder as this file, which it will be if you simply unzip the assignment).
# Note that we don't have any y_test! This way you cannot "cheat"!
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')


# Let's scale our data
scaler = StandardScaler()
scaler.fit(x_train)

z_train = scaler.transform(x_train)
z_test = scaler.transform(x_test)


# Look at column-wise mean, variance, min, max
print(z_train.mean(axis=0))
print(z_train.var(axis=0))
print(z_train.min(axis=0))
print(z_train.max(axis=0))


# Now let's define our search space and optimize hyperparameters
grid_search_space = { # See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    'kernel': ['poly', 'rbf'], # For polynomial kernel, default is cubic, could also tune that
    'C': [0.1, 1, 10, 25], # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1, 5, 10], # Kernel coefficient
    'epsilon': [0.01, 0.1, 1.0], # Size of epsilon-tube with no penalty
}

# Define our model
model = svm.SVR()

# Optimize hyperparameters using cross-validation
cv = GridSearchCV(model, grid_search_space, scoring='neg_mean_squared_error', n_jobs=-1)
cv.fit(z_train, y_train)


# Let's look at the best hyperparameters
print(cv.best_params_)


# Train final model
best_model = cv.best_estimator_
best_model.fit(z_train, y_train)


y_test_hat = best_model.predict(z_test)
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

