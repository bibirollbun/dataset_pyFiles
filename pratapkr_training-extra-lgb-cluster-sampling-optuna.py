import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
data_extra.head(5)


data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
data.head(5)


# data = pd.concat([data_train, data_extra])


data = data.drop('id', axis =1 )


data


y = data['Price']


data.shape


data.columns


for col in data.columns:
    print(col, data[col].nunique())


for col in data.columns:
    print(col, data[col].isnull().sum()/3994318)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgbm

def impute_with_lgbm(df, target_col):
    
    df = df.copy()
    
    train_data = df[df[target_col].notna()]
    test_data = df[df[target_col].isna()]
    
    if test_data.empty:
        print("No missing values to impute.")
        return df
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    print (df.columns)

    if df[target_col].dtypes == 'float64':
        model = lgbm.LGBMRegressor()

    else:
        model = lgbm.LGBMClassifier()
    
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    df.loc[df[target_col].isna(), target_col] = y_pred
    
    return df






# median = data["Price"].median()


# data["Price"] = data["Price"] - median


# data["Price"]


sns.scatterplot(data = data.sample(200), x = 'Weight Capacity (kg)', y = 'Price')


# sns.kdeplot(data=data, x="Weight Capacity (kg)")


print(
    data['Weight Capacity (kg)'].corr(data['Price']),
    data['Weight Capacity (kg)'].corr(data['Compartments'])
)


data = impute_with_lgbm(data, "Weight Capacity (kg)")


categorical_cols = [ col for col in data.columns if data[col].nunique() < 8 ]
categorical_cols


data['Brand'].nunique()


sns.countplot(data = data, x = data['Brand'])


data = impute_with_lgbm(data, "Brand")


data['Material'].nunique()


sns.countplot(data = data, x = data['Material'])


data = impute_with_lgbm(data, "Material")


data['Size'].nunique()


sns.countplot(data = data, x = "Size")


data = impute_with_lgbm(data, "Size")


data["Laptop Compartment"].nunique()


sns.countplot(data = data, x = data['Laptop Compartment'])


data.groupby('Laptop Compartment')['Compartments'].describe()


data = impute_with_lgbm(data, "Laptop Compartment")


data["Waterproof"].nunique()


sns.countplot(data = data, x = data['Waterproof'])


pd.crosstab(data['Waterproof'], data['Material'])



data = impute_with_lgbm(data, "Waterproof")


data["Style"].nunique()


sns.countplot(data = data, x = data['Style'])


data = impute_with_lgbm(data, "Style")


data["Color"].nunique()


sns.countplot(data = data, x = data['Color'])


data = impute_with_lgbm(data, "Color")


data['Size'] = data['Size'].map({
    "New":0,
    "Small":1,
    "Medium":2,
    "Large":3
})


data = pd.get_dummies(data, columns=['Brand',
 'Material',
 'Laptop Compartment',
 'Waterproof',
 'Style',
 'Color']
, drop_first=True, dtype = 'float32')

data





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X,y = data.drop("Price",axis = 1), data["Price"]


print(X.shape, y.shape)


# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# print(X_train.shape, y_train.shape)


import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),       
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1), 
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),          
        'max_depth': trial.suggest_int('max_depth', 3, 15),                 
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),  
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),         
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0) 
    }

    lgbm = LGBMRegressor(**params)

    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return mse

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

study = optuna.create_study(direction='minimize') 
study.optimize(objective, n_trials=100)

best_params = study.best_params
mse = study.best_value

print(f"Best hyperparameters: {study.best_params}")
print(f"Best MSE: {study.best_value}")


lgb = LGBMRegressor(**best_params)
lgb.fit(X_train, y_train)


mse = mean_squared_error(y_test, lgb.predict(X_test))
print (np.sqrt(mse))


import time

def impute_data(df):
    df.drop('id', axis = 1, inplace = True)

    columns_to_impute = [
        "Weight Capacity (kg)", "Color", "Style", "Waterproof",
        "Laptop Compartment", "Size", "Material", "Brand"
    ]

    for col in columns_to_impute:
        df = impute_with_lgbm(df, col)
        # time.sleep(200)
    
    # data = impute_with_lgbm(data, "Weight Capacity (kg)")
    # data = impute_with_lgbm(data, "Color")
    # data = impute_with_lgbm(data, "Style")
    # data = impute_with_lgbm(data, "Waterproof")
    # data = impute_with_lgbm(data, "Laptop Compartment")
    # data = impute_with_lgbm(data, "Size")
    # data = impute_with_lgbm(data, "Material")
    # data = impute_with_lgbm(data, "Brand")

    return df

def encode_cat_cols(df):
    df['Size'] = df['Size'].map({
        "New":0,
        "Small":1,
        "Medium":2,
        "Large":3
    })
    df = pd.get_dummies(df, columns=['Brand',
         'Material',
         'Laptop Compartment',
         'Waterproof',
         'Style',
         'Color']
        , drop_first=True, dtype = 'float32')

    return df


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data.head(5)


test_data = impute_data(test_data)


test_data = encode_cat_cols(test_data)


res = lgb.predict(test_data)


res.shape


submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


submission["Price"] = res


submission["Price"] = submission["Price"]


submission.to_csv("submission.csv", index = False)







