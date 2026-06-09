import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import optuna


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_data_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
train_data = pd.concat([train_data,train_data_extra],axis=0)
train_data.info()


train_data.dropna()


def clean_data(data):
    data['Brand'] = data['Brand'].map({'Jansport': 0, 'Under Armour': 1, 'Nike': 2, 'Adidas': 3, 'Puma':4})
    
    data['Material'] = data['Material'].map({'Leather': 0, 'Canvas': 1, 'Nylon': 2, 'Polyester': 3})
    
    data['Style'] = data['Style'].map({'Tote': 0, 'Messenger': 1, 'Backpack': 2})
    
    data['Size'] = data['Size'].map({'Small': 0, 'Medium': 1, 'Large': 2})
    
    data['Laptop Compartment'] = data['Laptop Compartment'].map({'No': 0, 'Yes': 1})
    
    data['Waterproof'] = data['Waterproof'].map({'No': 0, 'Yes': 1})
    
    data['Color'] = data['Color'].map({'Black':0, 'Green':1, 'Red':2, 'Blue':3, 'Gray':4, 'Pink':5})

    return data

train_data = clean_data(train_data)


display(train_data)


df_train = train_data.drop(columns=['Price','id'])
df_train_value = train_data['Price']

X_train, X_test, y_train, y_test = train_test_split(df_train, df_train_value, test_size=0.1, random_state=42)
train_pool = Pool(data=X_train, label=y_train)



def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'verbose': 0
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Melhores parâmetros:", study.best_params)


best_params ={'iterations': 941, 'depth': 6, 'learning_rate': 0.05924786636688097, 'l2_leaf_reg': 6.313873403056393e-05, 'verbose':0}
model = CatBoostRegressor(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)  # RMSE



feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure()
plt.bar(feature_importance['feature'],feature_importance['importance'], color='blue', lw=2)

feature_importance


test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
ids = test_data['id']
test_data = test_data.drop(columns=['id'])
test_data = clean_data(test_data)


y_pred = model.predict(test_data)

submission_df = pd.DataFrame()
submission_df['id'] = ids
submission_df['price'] = y_pred


submission_df.to_csv("submission.csv", index=False)


