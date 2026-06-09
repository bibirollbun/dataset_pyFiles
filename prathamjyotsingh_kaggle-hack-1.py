import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
test=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
sub=pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/sample_submission.csv')
train


print(train.shape)
print(test.shape)


train.isnull().sum()


train.columns


train.dtypes


sub


test


idcol=test.id
test=test.drop("id",axis=1)
test


idcol


X=train.drop("target",axis=1)
y=train.target


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import pandas as pd

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

X_train_scaled = standard_scaler.fit_transform(X_train_split)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

X_val_scaled = standard_scaler.transform(X_val_split)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)

X_test_scaled = standard_scaler.transform(test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=test.columns)


print(f"Training shape : {X_train_scaled.shape}")
print(f"Validation shape : {X_val_scaled.shape}")
print(f"Test shape : {X_test_scaled.shape}")




# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [3000],
#     'learning_rate': [0.01],
#     'max_depth': [8,9,10],
#     'subsample': [0.8],
#     'colsample_bytree': [0.6,0.7],
#     'gamma': [1],
# }

# # Initialize the base XGBRegressor model
# xgb_model = XGBRegressor(random_state=42)

# # Set up GridSearchCV with R² scoring
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring='r2',  # Use R² score for optimization
#     verbose=2,
#     n_jobs=-1  # Use all available cores
# )

# # Fit the grid search on the training data
# grid_search.fit(X_train_clean, y_train_clean)


# # Retrieve the best model
# best_xgb_model = grid_search.best_estimator_
# print("Best parameters found: ", grid_search.best_params_)




from xgboost import XGBRegressor
from sklearn.metrics import r2_score

best_xgb_model = XGBRegressor(
    colsample_bytree=0.7,
    gamma=1,
    learning_rate=0.01,
    max_depth=9,
    n_estimators=3750,
    reg_alpha=0,
    reg_lambda=0.1,
    subsample=0.8,
    random_state=42
)

best_xgb_model.fit(X_train_scaled, y_train_split)

y_pred_val = best_xgb_model.predict(X_val_scaled)

r2_xgb = r2_score(y_val_split, y_pred_val)

print(f"Optimized XGBoost R² Score: {r2_xgb:.4f}")



y_test_pred = best_xgb_model.predict(X_test_scaled)

if y_test_pred.ndim > 1:
    y_test_pred = y_test_pred.flatten()
submission = pd.DataFrame({
    'id': idcol,    
    'target': y_test_pred 
})

submission.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv.")




