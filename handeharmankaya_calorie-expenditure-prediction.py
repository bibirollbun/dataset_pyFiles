import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


df_train.head()


df_train.shape


df_train.info()


df_train.isnull().sum()


df_test.head()


df_test.shape


df_test.info()


df_test.isnull().sum()


submission.head()


df_train.describe()


train = df_train.drop('id', axis=1).copy()


sns.histplot(train['Calories'], kde=True, color='salmon')
plt.title('Calories Distribution');


sns.boxplot(x=train['Sex'], y=train['Calories'], palette='pastel')
plt.title('Calories Distribution Based on Gender');


train['Sex'] = train['Sex'].map({'male': 0,'female': 1})


train.info()


sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap='RdBu');


sns.scatterplot(data=train.sample(2000), x='Duration', y='Calories', hue='Sex', alpha=0.6, cmap='plasma')
plt.title('Duration vs. Calories')
plt.tight_layout();


#Body Mass Index
train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
df_test["BMI"] = df_test["Weight"] / ((df_test["Height"] / 100) ** 2)

#BMR (Basal Metabolic Rate)
train['BMR'] = (10 * train['Weight']) + (6.25 * train['Height']) - (5 * train['Age'])
df_test['BMR'] = (10 * df_test['Weight']) + (6.25 * df_test['Height']) - (5 * df_test['Age'])

#Duration relative to heart rate (Effort)
train["Effort"] = train["Duration"] * train["Heart_Rate"]
df_test["Effort"] = df_test["Duration"] * df_test["Heart_Rate"]

#Temperature Difference
train['Temp_Diff'] = train['Body_Temp'] - 37.0
df_test['Temp_Diff'] = df_test['Body_Temp'] - 37.0

# Exercise intensity: interaction between heart rate and body temperature
train["Intensity"] = train["Heart_Rate"] * train["Body_Temp"]
df_test["Intensity"] = df_test["Heart_Rate"] * df_test["Body_Temp"]

# Body temperature per minute of exercise
train["Temp_per_Minute"] = train["Body_Temp"] / train["Duration"].replace(0, 1)
df_test["Temp_per_Minute"] = df_test["Body_Temp"] / df_test["Duration"].replace(0, 1)


train.head()


df_test['Sex'] = df_test['Sex'].map({'male': 0,'female': 1})


df_test.head()


x=train.drop('Calories', axis=1)
y=train['Calories']
test=df_test.drop('id', axis=1)


from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import plot_tree, ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

def algo_test(x,y):
        print("Model tanımlanıyor...")
        L=LinearRegression()
        R=Ridge()
        Lass=Lasso()
        E=ElasticNet()
        sgd=SGDRegressor()
        ETR=ExtraTreeRegressor()
        GBR=GradientBoostingRegressor()
        ada=AdaBoostRegressor()
        xgb=XGBRegressor()
        
        algos=[L,R,Lass,E,sgd,ETR,GBR,ada,xgb]
        algo_names=['Linear','Ridge','Lasso','ElasticNet','SGD','Extra Tree','Gradient Boosting',
                    'AdaBoost','XGBRegressor']
    
        print("Veri ölçeklendiriliyor...")
        x=MinMaxScaler().fit_transform(x)
                
        print("Veri kümesi eğitim ve test kümelerine ayrılıyor...")
        x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.20,random_state=42)
        
        r_squared= []
        rmse= []
        mae= []

        print("Sonuçları saklamak için DataFrame oluşturuluyor...")
        result=pd.DataFrame(columns=['R_Squared','RMSE','MAE'],index=algo_names)


        print("Modeller eğitiliyor ve test ediliyor...")
        for algo, name in zip(algos, algo_names): 
            print(f"{name} modeli eğitiliyor...")
            p=algo.fit(x_train,y_train).predict(x_test)
            r_squared.append(r2_score(y_test,p))
            rmse.append(mean_squared_error(y_test,p)**.5)
            mae.append(mean_absolute_error(y_test,p))
        
        print("Sonuçlar DataFrame'e yerleştiriliyor...")
        result.R_Squared=r_squared
        result.RMSE=rmse
        result.MAE=mae

        print("Sonuçlar sıralanıyor...")
        rtable=result.sort_values('R_Squared',ascending=False)

        print("İşlem tamamlandı!")
        return rtable


algo_test(x,y)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

def smart_algo_test():
    print("Modeller hazırlanıyor...")
    
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
        ("XGBoost (GPU)", XGBRegressor(n_estimators=1000,learning_rate=0.05,
                                       tree_method='hist',device='cuda',random_state=42)),
        ("LightGBM (GPU)", LGBMRegressor(n_estimators=1000,learning_rate=0.05,
                                         device='gpu',random_state=42,verbose=-1)),
        ("CatBoost (GPU)", CatBoostRegressor(iterations=1000,learning_rate=0.05,
                                             task_type="GPU",devices='0',verbose=0,random_state=42))]

    results = []
    
    print(f"{'Model':<20} | {'RMSE':<10} | {'R2 Score':<10}")
    
    for name, model in models:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results.append([name, rmse, r2])
            print(f"{name:<20} | {rmse:.4f}     | {r2:.4f}")
        except Exception as e:
            print(f"{name} hatası: {str(e)}")
            
    return pd.DataFrame(results, columns=["Model", "RMSE", "R2"]).sort_values("RMSE")


smart_algo_test()


import optuna
import xgboost as xgb

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method': 'hist',
        'device': 'cuda',       
        'random_state': 42,
        'n_jobs': -1}
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)],verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

print("Optuna optimizasyonu başlıyor...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) 
print("En İyi Parametreler:", study.best_params)
print("En İyi RMSE:", study.best_value)


best_params = {'n_estimators': 1348,'learning_rate': 0.013113900053516404,'max_depth': 8,
               'subsample': 0.8939637965060047,'colsample_bytree': 0.8539190344172591,'min_child_weight': 1,
               'tree_method': 'hist','device': 'cuda','random_state': 42,'n_jobs': -1}

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(x, y)


final_preds = final_model.predict(test)

submission = pd.DataFrame()
submission['id'] = df_test['id']  
submission['Calories'] = final_preds

submission.to_csv('submission.csv', index=False)


import joblib
import json

joblib.dump(final_model, 'xgboost_model.pkl') 

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')


from catboost import CatBoostRegressor

xgb_preds = final_preds 

cat_params = {'iterations': 2000,'learning_rate': 0.05,'depth': 8,'loss_function': 'RMSE',
              'task_type': 'GPU','devices': '0','verbose': 0,'random_seed': 42}

cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(x, y)

cat_preds = cat_model.predict(test)
ensemble_preds = (xgb_preds * 0.5) + (cat_preds * 0.5)

submission_ensemble = pd.DataFrame()
submission_ensemble['id'] = df_test['id']
submission_ensemble['Calories'] = ensemble_preds

submission_ensemble.to_csv('submission_v1.csv', index=False)

