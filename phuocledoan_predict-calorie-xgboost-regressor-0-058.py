import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from xgboost import XGBRegressor


data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


data.info()


data.head()


data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
test_data['Sex'] = test_data['Sex'].map({'female': 0, 'male': 1})


data.head()


corr = data.drop(columns=['id', 'Sex']).corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


X = data.drop(columns=['id', 'Calories'])
y = data['Calories']

X_test = test_data.drop(columns=['id'])  


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# import os
# import optuna

# def objective(trial):
#     param_dist = {
#         "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5, log=True),
#         "max_depth": trial.suggest_int("max_depth", 5, 10),
#         "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.1),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.1),
#         "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 400]),
#     }
    
#     model = XGBRegressor(
#         objective='reg:squarederror',
#         random_state=42,
#         tree_method='hist',
#         device='cuda',
#         n_jobs=-1,
#         **param_dist
#     )
    
#     # Use cross_val_predict to get predictions across folds
#     kf = KFold(n_splits=3, shuffle=True, random_state=42)
#     y_pred = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=-1)
#     y_pred = np.clip(y_pred, 0, None)
    
#     # Compute MSLE
#     score = mean_squared_log_error(y_train, y_pred)
    
#     return score

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# # Print best results
# print("Best Parameters:", study.best_params)
# print("Best MSLE:", study.best_value)

# Best Parameters: {'learning_rate': 0.052273638058659554, 'max_depth': 10, 'min_child_weight': 1.9893286562051629, 'subsample': 0.9997183856932446, 'colsample_bytree': 0.8611479144257395, 'reg_alpha': 0.04688615811762002, 'reg_lambda': 0.05259963820449194, 'n_estimators': 200}
# Best MSLE: 0.0037121778336770203


xgboost_regressor = XGBRegressor(
    learning_rate = 0.052273638058659554, 
    max_depth = 10, 
    min_child_weight = 1.9893286562051629, 
    subsample = 0.9997183856932446, 
    colsample_bytree = 0.8611479144257395, 
    reg_alpha = 0.04688615811762002, 
    reg_lambda = 0.05259963820449194, 
    n_estimators = 200,
  
    random_state = 42,
    tree_method='hist',
    device='cuda',
    n_jobs=-1,
)

xgboost_regressor.fit(X_train, y_train)


y_val_pred = xgboost_regressor.predict(X_val)
print(f'Validation RMSLE: {np.sqrt(mean_squared_log_error(y_val, y_val_pred)):.4f}')


y_pred = xgboost_regressor.predict(X_test)


submission = pd.DataFrame({
    'id': test_data['id'],
    'Calories': y_pred
})


submission.to_csv('submission.csv', index=False)


submission

