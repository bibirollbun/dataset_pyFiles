import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import optuna


test= pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
train= pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
sample_submission= pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


cat_cols = train.select_dtypes(include=['object']).columns

X = train.drop(columns=["accident_risk"])
y = train["accident_risk"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 100, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'od_type': 'Iter',
        'od_wait': 50,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': False,
        'task_type':'CPU'

        
    } 
    
    model = CatBoostRegressor(**params)
    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        cat_features=['road_type', 'lighting', 'weather', 'time_of_day'],
        use_best_model=True,
        early_stopping_rounds=10
    )
    
    preds = model.predict(x_test)
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_trial)
print("Best params:")
print(study.best_params)


# use catboost for final model training and prediction

final_model = CatBoostRegressor(
    iterations=434,
    learning_rate=0.17160769945341836,
    depth=8,
    l2_leaf_reg= 0.0009939625228595733,
    border_count= 191,
    random_strength= 0.0012842098407259207,
    bagging_temperature=0.18988359959381565,
    loss_function='RMSE',
    
    cat_features=['road_type', 'lighting', 'weather', 'time_of_day']
)

final_model.fit(X,y)


pred= final_model.predict(test)
sample_submission["accident_risk"]= pred
sample_submission.to_csv("submission.csv", index= False)


import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = final_model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

