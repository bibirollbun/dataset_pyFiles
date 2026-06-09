# ğŸ“¦ Standard Libraries
import numpy as np
import pandas as pd

# ğŸ“Š Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# âš™ï¸� Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# ğŸ¤– Machine Learning Models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# ğŸ”® Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {mse:.2f}")
    print(f"RÂ² Score: {r2:.2f}")
    return rmse, r2



train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())


# 4. Prepare Data for Modeling


# Drop non-numeric ID column
X = train.drop(['field_id', 'yield'], axis=1)
y = (train['yield']-np.mean(train['yield']))/np.std(train['yield'])
X_test = test.drop(['field_id'], axis=1)

scale=StandardScaler()

X = scale.fit_transform(X)
X_test = scale.transform(X_test)


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Baseline Model

model = SVR(kernel='rbf')
model.fit(X_train, y_train)

rs = 42
np.random.seed(rs)
# Parameters


# Validation predictions and RMSE
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")


rs = 42
np.random.seed(rs)
# Parameters
low = 3500.2      # lower bound of the range (inclusive)
high = 5000     # upper bound of the range (exclusive)
size = (len(X_test),1) # desired shape 

# Generate data
uniform_data = np.random.uniform(low=low, high=high, size=size)


predict = (model.predict(X_test)*np.std(train['yield']))+np.mean(train['yield'])
submission['yield'] =predict

submission.head()


submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")







