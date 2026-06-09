import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train_df['source'] = 'train'
extra_df['source'] = 'extra'
train = pd.concat([train_df, extra_df], axis=0, ignore_index=True)


train.head()


train.shape


train.info()


train.isnull().sum()


test_df.head()


test_df.shape


test_df.info()


test_df.isnull().sum()


sns.histplot(train['Price'], kde=True, bins=50, color='skyblue')
plt.show()
print(f"Skewness: {train['Price'].skew()}")


missing_vals = train.isnull().sum()
missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)

sns.barplot(x=missing_vals.index, y=missing_vals.values, palette='pastel')
plt.xticks(rotation=45)
plt.title('Missing Value Counts')
plt.show()
print((train.isnull().sum()/len(train) * 100).sort_values(ascending=False))


missing_values = test_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

sns.barplot(x=missing_values.index, y=missing_values.values, palette='pastel')
plt.xticks(rotation=45)
plt.title('Missing Value Counts')
plt.show()
print((test_df.isnull().sum()/len(train) * 100).sort_values(ascending=False))


categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, col in enumerate(['Brand', 'Material', 'Size']):
    sns.boxplot(data=train, x=col, y='Price', ax=axes[i], palette='pastel')
    axes[i].set_title(f'{col} vs Price')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


numeric_cols = train.select_dtypes(include=[np.number])

sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='plasma', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


df_all = pd.concat([train, test_df], axis=0, ignore_index=True)


#Imputation
categorical = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof', 'Size']
numeric = ['Compartments', 'Weight Capacity (kg)']

df_all[categorical] = df_all[categorical].fillna('Unknown')
df_all[numeric] = df_all[numeric].fillna(df_all[numeric].median())

#Encoding
size = {'Small': 0, 'Medium': 1, 'Large': 2, 'Unknown': -1}
df_all['Size_Encoded'] = df_all['Size'].map(size)

le = LabelEncoder()
encode_cols = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
for col in encode_cols:
    df_all[col + '_Encoded'] = le.fit_transform(df_all[col].astype(str))

#Deleting original columns
df_all_processed = df_all.drop(columns=categorical)

train_processed = df_all_processed[df_all_processed['Price'].notnull()].copy()
test_processed = df_all_processed[df_all_processed['Price'].isnull()].drop(columns=['Price']).copy()


train_processed.head()


test_processed.head()


test_processed.drop(columns=['source'], inplace=True)


test_processed.head()


sns.heatmap(train_processed.drop(columns=['source']).corr(), annot=True, fmt='.2f', cmap='plasma')
plt.title('Correlation Matrix')
plt.show()


x = train_processed.drop(columns=['Price', 'id', 'source']) 
y = train_processed['Price']


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import time

def algo_test_(x, y):
    print("Veri kümesi ayrılıyor...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'XGBoost (Fast)': XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,
                                       tree_method='hist',device="cuda",n_jobs=-1,early_stopping_rounds=50)}

    results = []

    print(f"Toplam {len(models)} model eğitilecek...")
    
    for name, model in models.items():
        try: 
            start_time = time.time()
            print(f"--> {name} eğitiliyor...", end=" ")

            if name == 'XGBoost (Fast)':
                model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
            else:
                model.fit(x_train, y_train)
            
            p = model.predict(x_test)
            
            r2 = r2_score(y_test, p)
            rmse = mean_squared_error(y_test, p, squared=False) 
            mae = mean_absolute_error(y_test, p)
                
            duration = time.time() - start_time
            
            print(f"Tamamlandı ({duration:.2f} sn) | RMSE: {rmse:.4f}")
                
            results.append({'Model': name, 'R_Squared': r2, 'RMSE': rmse, 'MAE': mae})
                
        except Exception as e:
            print(f"\nHATA: {name} çalışırken hata oluştu: {e}")

    df_results = pd.DataFrame(results).sort_values(by='RMSE', ascending=True) 
    
    return df_results


algo_test_(x,y)


import optuna

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method': 'hist',
        'device': 'cuda', 
        'n_jobs': -1,
        'random_state': 42}
    
    model = XGBRegressor(**params)
    
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)],early_stopping_rounds=50,verbose=False)
    
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(study.best_params)
print(f"En İyi RMSE: {study.best_value}")


best_params = {'n_estimators': 1762,
    'learning_rate': 0.033620889442177546,
    'max_depth': 7,
    'subsample': 0.7635960540356912,
    'colsample_bytree': 0.5974845023278984,
    'min_child_weight': 7,
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42}

final_model = XGBRegressor(**best_params)
final_model.fit(x, y)

x_test_final = test_processed.drop(columns=['id'])
test_preds = final_model.predict(x_test_final)


submission = pd.DataFrame({'id': test_processed['id'],'Price': test_preds})
submission.to_csv('submission_optuna_v1.csv', index=False)


import joblib

df_all = pd.concat([train, test_df], axis=0, ignore_index=True)

medians = {
    'Compartments': df_all['Compartments'].median(),
    'Weight Capacity (kg)': df_all['Weight Capacity (kg)'].median()}
df_all['Compartments'] = df_all['Compartments'].fillna(medians['Compartments'])
df_all['Weight Capacity (kg)'] = df_all['Weight Capacity (kg)'].fillna(medians['Weight Capacity (kg)'])

categorical_cols = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof', 'Size']
df_all[categorical_cols] = df_all[categorical_cols].fillna('Unknown')

encoders = {} 

size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2, 'Unknown': -1}
df_all['Size_Encoded'] = df_all['Size'].map(size_mapping)

encode_cols = ['Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
for col in encode_cols:
    le = LabelEncoder()
    df_all[col + '_Encoded'] = le.fit_transform(df_all[col].astype(str))
    encoders[col] = le 
    
x = df_all[df_all['Price'].notnull()].drop(columns=['Price', 'id', 'source'] + categorical_cols)
y = df_all[df_all['Price'].notnull()]['Price']

final_model = XGBRegressor(n_estimators=1762, learning_rate=0.033, max_depth=7,subsample=0.76, 
                           colsample_bytree=0.59, min_child_weight=7,n_jobs=-1)
final_model.fit(x, y)

artifacts = {'model': final_model,'encoders': encoders,'size_mapping': size_mapping,'medians': medians}

joblib.dump(artifacts, 'backpack_model.joblib')

