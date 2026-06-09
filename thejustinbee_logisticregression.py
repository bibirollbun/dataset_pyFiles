import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

train_data.head()


from sklearn.preprocessing import StandardScaler


X = train_data[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',\
    'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]

y= train_data[['rainfall']] 

X_train, X_test, y_train, y_test  = train_test_split(X,y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model2 = LinearRegression()
model.fit(X_train, y_train)
model2.fit(X_train, y_train)

prediction  = model.predict(X_test)
prediction2 = model2.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
print('Accuracy of linear regression classifier on test set: {:.2f}'.format(model2.score(X_test, y_test)))


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model
log_reg = LogisticRegression(max_iter=2000)

# Define the parameter grid (Make sure 'l1' is only used with solvers that support it)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],          # Regularization strength
    'penalty': ['l1', 'l2'],               # Regularization type
    'solver': ['liblinear', 'lbfgs', 'saga']  # Solvers
}

# Adjust the parameter grid to only use 'l1' with 'liblinear' and 'saga'
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # 'l2' works with 'lbfgs', 'liblinear', and 'saga'
    'solver': ['liblinear', 'lbfgs', 'saga']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit the grid search to your data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and the best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_log_reg = grid_search.best_estimator_
print("Test Set Accuracy:", best_log_reg.score(X_test_scaled, y_test))

test_data.fillna(0, inplace=True)
test = test_data[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',\
    'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]
test = scaler.transform(test)
print(test)

prediction = best_log_reg.predict(test)
result = pd.DataFrame({'id': test_data['id'], 'prediction': prediction})
result.to_csv('submission.csv', index=False)




