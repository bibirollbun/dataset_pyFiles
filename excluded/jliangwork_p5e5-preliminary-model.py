import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting for pandas dataframes

from sklearn.model_selection import KFold # split data into folds


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_w_source = train.copy()
test_w_source = test.copy()
train_w_source['Source'] = 'Train'
test_w_source['Source'] = 'Test'
test_w_source['Calories'] = -1.0
combined_data = pd.concat([train_w_source,test_w_source])


# we will compute the rmlse error even outside of the PrelimModel class, so we define it here
def rmsle(y, y_pred):

    # assume that y and y_pred are 1-D numpy arrays
    y = np.array(y)
    y_pred = np.array(y_pred)

    logy = np.log1p(y)
    logy_pred = np.log1p(y_pred)
    diff = logy_pred - logy
    mean = np.mean(diff**2)

    return np.sqrt(mean)


class PrelimModel:

    def __init__(self, params = np.array([0, 0, 0, 0, 0, 0, 0, 0])):
        self.params = params

    def update(self, delta = np.array([0, 0, 0, 0, 0, 0, 0, 0])):
        self.params = self.params + delta

    def print_params(self):
    
        a = self.params[0]
        b = self.params[1]
        lH = self.params[2]
        kH = self.params[3]
        h0 = self.params[4]
        lT = self.params[5]
        kT = self.params[6]
        t0 = self.params[7]
        
        print("--- Model Parameters ---")
        print(f"  a  : {a:.4f}") # Added formatting for readability
        print(f"  b  : {b:.4f}")
        print(f"  LH : {lH:.4f}")
        print(f"  kH : {kH:.4f}")
        print(f"  H0 : {h0:.4f}")
        print(f"  LT : {lT:.4f}")
        print(f"  kT : {kT:.4f}")
        print(f"  T0 : {t0:.4f}")
        print("------------------------")

    def predict(self, X):
        # assuming that X is a 2D numpy array with columns D, H, and T in that order

        X = np.array(X)
        
        a = self.params[0]
        b = self.params[1]
        lH = self.params[2]
        kH = self.params[3]
        h0 = self.params[4]
        lT = self.params[5]
        kT = self.params[6]
        t0 = self.params[7]
        
        linear_term = b + a * X[: , 0]
        h_term = lH / (1 + np.exp(-kH * (X[: , 1] - h0)))
        t_term = lT / (1 + np.exp(-kT * (X[: , 2] - t0)))

        return linear_term + h_term + t_term

    def rmsle_grad(self, X, y):

        X = np.array(X)
        y = np.array(y)
        y_hat = self.predict(X)
        error = rmsle(y, y_hat)

        if (error == 0):
            return np.array([0, 0, 0, 0, 0, 0, 0, 0])

        a = self.params[0]
        b = self.params[1]
        lH = self.params[2]
        kH = self.params[3]
        h0 = self.params[4]
        lT = self.params[5]
        kT = self.params[6]
        t0 = self.params[7]

        dyda = X[ : , 0]
        dydb = 1
        dydlH = 1 / (1 + np.exp(-kH * (X[ : , 1] - h0)))
        dydkH = (X[ : , 1] - h0) * lH * np.exp(-kH * (X[ : , 1] - h0)) / (1 + np.exp(-kH * (X[ : , 1] - h0)))**2
        dydh0 = -lH * kH * np.exp(-kH * (X[ : , 1] - h0)) / (1 + np.exp(-kH * (X[ : , 1] - h0)))**2
        dydlT = 1 / (1 + np.exp(-kT * (X[ : , 2] - t0)))
        dydkT = (X[ : , 2] - t0) * lT * np.exp(-kT * (X[ : , 2] - t0)) / (1 + np.exp(-kT * (X[ : , 2] - t0)))**2
        dydt0 = -lT * kT * np.exp(-kT * (X[ : , 2] - t0)) / (1 + np.exp(-kT* (X[ : , 2] - t0)))**2

        scaling = max(1, (1 / 2) * (1 / error))
        diff = 2*(np.log1p(y_hat) - np.log1p(y))*(1 / (1 + y_hat))

        dEda = scaling * (np.mean(diff * dyda))
        dEdb = scaling * (np.mean(diff * dydb))
        dEdlH = scaling * (np.mean(diff * dydlH))
        dEdkH = scaling * (np.mean(diff * dydkH))
        dEdh0 = scaling * (np.mean(diff * dydh0))
        dEdlT = scaling * (np.mean(diff * dydlT))
        dEdkT = scaling * (np.mean(diff * dydkT))
        dEdt0 = scaling * (np.mean(diff * dydt0))

        return np.array([dEda, dEdb, dEdlH, dEdkH, dEdh0, dEdlT, dEdkT, dEdt0])
    
    def fit(self, X, y, iterations = 10000, alpha = 0.01, error_tolerance = 0.00001):

        error = 1000000
        
        for i in range(iterations):
        
            gradient = self.rmsle_grad(X, y) # method doesn't exist yet
            self.update(delta = -alpha*gradient)
            predictions = self.predict(X)
            new_error = rmsle(predictions, y)
            
            if (np.abs(error-new_error) < error_tolerance):
                print("Early stop")
                break
            else:
                error = new_error


X = train[['Duration', 'Heart_Rate', 'Body_Temp']].copy()
y = train['Calories'].copy()


model = PrelimModel()


model.fit(X, y)
predictions = model.predict(X)
print(rmsle(predictions, y))
print(np.mean(np.abs(predictions - y)))


np.mean(np.abs(model.predict(X) - y))


model = PrelimModel()
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
errors = np.array([])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print(f"The RMSLE in this split is {rmsle(y_val, y_pred)}")
    print(f"The mean absolute error in this split is {np.mean(np.abs(y_pred - y_val))}")
    errors = np.append(errors, rmsle(y_val, y_pred))


print(np.mean(errors))
print(np.std(errors))


model = PrelimModel()
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
errors = np.array([])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train, alpha = 0.1)
    y_pred = model.predict(X_val)

    print(f"The RMSLE in this split is {rmsle(y_val, y_pred)}")
    print(f"The mean absolute error in this split is {np.mean(np.abs(y_pred - y_val))}")
    errors = np.append(errors, rmsle(y_val, y_pred))


model = PrelimModel()
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
errors = np.array([])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train, alpha = 1)
    y_pred = model.predict(X_val)

    print(f"The RMSLE in this split is {rmsle(y_val, y_pred)}")
    print(f"The mean absolute error in this split is {np.mean(np.abs(y_pred - y_val))}")
    errors = np.append(errors, rmsle(y_val, y_pred))


model = PrelimModel()
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
errors = np.array([])

for train_index, val_index in kf.split(X, y):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train, y_train, alpha = 2)
    y_pred = model.predict(X_val)

    print(f"The RMSLE in this split is {rmsle(y_val, y_pred)}")
    print(f"The mean absolute error in this split is {np.mean(np.abs(y_pred - y_val))}")
    errors = np.append(errors, rmsle(y_val, y_pred))


train_w_prelim_predictions = train.copy()
train_w_prelim_predictions['Prediction'] = -1.0
train_w_prelim_predictions['Residual'] = -1.0
train_w_prelim_predictions['Squared Logarithmic Error'] = - 1.0


model = PrelimModel()
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)

for train_index, val_index in kf.split(X, y):

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train, alpha = 1)
    y_pred = model.predict(X_val)

    train_w_prelim_predictions.loc[val_index, 'Prediction'] = y_pred


train_w_prelim_predictions['Residual'] = train_w_prelim_predictions['Calories'] - train_w_prelim_predictions['Prediction']
train_w_prelim_predictions['Squared Logarithmic Error'] = (np.log1p(train_w_prelim_predictions['Calories']) - np.log1p(train_w_prelim_predictions['Prediction']))**2


sns.scatterplot(data = train_w_prelim_predictions, x = 'Prediction', y = 'Residual')
plt.title("Residuals vs. Prediction")
plt.show()


sns.scatterplot(data = train_w_prelim_predictions, x = 'Prediction', y = 'Squared Logarithmic Error')
plt.title("Squared Logarithmic Error vs. Prediction")
plt.show()


sns.histplot(data = train_w_prelim_predictions, x = 'Residual')
plt.title("Histogram of Residuals")
plt.show()


sns.boxplot(data = train_w_prelim_predictions, x = 'Residual')
plt.title("Boxplot of Residuals")
plt.show()


sns.histplot(data = train_w_prelim_predictions, x = 'Squared Logarithmic Error')
plt.title("Histogram of Squared Logarithmic Error")
plt.show()


sns.boxplot(data = train_w_prelim_predictions, x = 'Squared Logarithmic Error')
plt.title("Boxplot of Squared Logarithmic Error")
plt.show()


np.sqrt(np.mean(train_w_prelim_predictions[train_w_prelim_predictions['Squared Logarithmic Error'] < train_w_prelim_predictions['Squared Logarithmic Error'].quantile(0.9)]['Squared Logarithmic Error']))


model = PrelimModel()
model.fit(X, y)


model.print_params()


model = PrelimModel(params = np.array([1, 1, 1, 1, 1, 1, 1, 1]))
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
errors = np.array([])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train, alpha = 1)
    y_pred = model.predict(X_val)

    print(f"The RMSLE in this split is {rmsle(y_val, y_pred)}")
    print(f"The mean absolute error in this split is {np.mean(np.abs(y_pred - y_val))}")
    errors = np.append(errors, rmsle(y_val, y_pred))


model = PrelimModel(params = np.array([1, 1, 50, 1, 50, 50, 1, 50]))
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
errors = np.array([])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train, alpha = 1)
    y_pred = model.predict(X_val)

    print(f"The RMSLE in this split is {rmsle(y_val, y_pred)}")
    print(f"The mean absolute error in this split is {np.mean(np.abs(y_pred - y_val))}")
    errors = np.append(errors, rmsle(y_val, y_pred))


model = PrelimModel(params = np.array([1, 1, 50, 0.001, 50, 50, 0.001, 50]))
model.fit(X, y, alpha = 1)


model.print_params()

