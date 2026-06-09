%%capture 
!pip install lazypredict


import pandas as pd
import numpy as np
import missingno
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler , PowerTransformer , RobustScaler 
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split , cross_val_score , KFold

from lazypredict.Supervised import LazyRegressor
import optuna
from sklearn.linear_model import Ridge, Lasso, LinearRegression 
from sklearn.ensemble import  ExtraTreesRegressor , StackingRegressor , GradientBoostingRegressor, RandomForestRegressor 
from xgboost import XGBRegressor
import catboost as ctb
import lightgbm as lgbm
from IPython.display import display


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(f'Shape of Train data : {train_df.shape}')
print(f'Shape of Test data  : {test_df.shape}')


train_df.head()


train_df.describe()


combined_data = pd.concat((train_df,test_df), axis = 0)
combined_data


print(combined_data.dtypes)


categorical_feature = train_df.select_dtypes(include='object').columns.tolist()

numerical_feature = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

numerical_feature = [col for col in numerical_feature if col != 'id' and col != 'Calories']

# Print hasilnya
print(f'Number of Categorical Feature : {len(categorical_feature)}')
print(f'Number of Numerical Feature   : {len(numerical_feature)}')



n_cols = len(numerical_feature) // 2 + len(numerical_feature) % 2
fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(5 * n_cols, 12))

# Flatten axes biar mudah di-loop
axes = axes.flatten()

# Plot distribusi untuk setiap fitur numerik
for i, feature in enumerate(numerical_feature):
    sns.histplot(train_df[feature], kde=True, ax=axes[i], color='darkblue')
    sns.histplot(test_df[feature], kde= True, ax=axes[i], color='gold')
    axes[i].set_ylabel('Count')

# Hilangkan axes kosong jika ada
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


body temp sama age keknya harus dinormalin, sisanya cari outlier trus standardscale?


sns.histplot(data=train_df, x='Calories')


# Check Features Correlation

numeric_data = pd.DataFrame()

for feature in numerical_feature:
    numeric_data[feature] = train_df[feature]

corr_data = numeric_data.corr(method='pearson')

plt.figure(figsize=(5,5))
sns.heatmap(data= corr_data, cmap='coolwarm', annot=True, fmt='.2f')


# Check Calories Correlation
numeric_data['Calories'] = train_df['Calories'] 

corr_data = numeric_data.corr(method='pearson')
corr_data = corr_data[['Calories']]       


plt.figure(figsize=(7,10))
sns.heatmap(data=corr_data, cmap='coolwarm', annot=True, fmt='.2f')


plt.figure(figsize=(12,8))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


train_df.isnull().sum()


combined_data['BMI'] = combined_data['Weight'] / (combined_data['Height'] / 100) ** 2
# combined_data = combined_data.drop(['Height', 'Weight'], axis = 1)
combined_data


combined_data.drop(['id'], axis = 1, inplace=True)
combined_data


combined_data = pd.get_dummies(combined_data).reset_index(drop=True)

combined_data


new_train_data = combined_data.iloc[:len(train_df), :]
new_test_data  = combined_data.iloc[len(train_df):, :]

x_train = new_train_data.drop(labels=['Calories'], axis=1)

y_train = np.log1p(new_train_data['Calories'])

x_test = new_test_data.drop(labels=['Calories'], axis=1)


x_train.shape , y_train.shape, x_test.shape


sns.histplot(data=y_train)


# BUILDING LAZY PREDICT TO FIND BEST MODEL

x_train_lazy , x_test_lazy , y_train_lazy , y_test_lazy = train_test_split(x_train, y_train, test_size=0.2, random_state=12, shuffle=True)


custom_regressors = [
    RandomForestRegressor,
    LinearRegression,
    GradientBoostingRegressor,
    XGBRegressor,
    lgbm.LGBMRegressor,
    ctb.CatBoostRegressor
]

lazy_model = LazyRegressor(verbose=0, random_state=12, regressors=custom_regressors)
train_lazy, test_lazy = lazy_model.fit(x_train_lazy, x_test_lazy, y_train_lazy, y_test_lazy)
test_lazy


import time

def calculate_rmsle(y_true, y_pred):
    # Pastikan nilai non-negatif untuk RMSLE
    y_true = np.expm1(y_true)  # Kembalikan dari log1p ke skala asli
    y_pred = np.expm1(y_pred)  # Kembalikan dari log1p ke skala asli
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))

x_train_lazy, x_test_lazy, y_train_lazy, y_test_lazy = train_test_split(
    x_train, y_train, test_size=0.2, random_state=12, shuffle=True
)

# Inisialisasi model sesuai dengan LazyRegressor Anda
models = {
    'CatBoostRegressor': ctb.CatBoostRegressor(verbose=0, random_state=12),
    'XGBRegressor': XGBRegressor(random_state=12),
    'LGBMRegressor': lgbm.LGBMRegressor(random_state=12),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=12),
    'RandomForestRegressor': RandomForestRegressor(random_state=12),
    'LinearRegression': LinearRegression()
}

# List untuk menyimpan hasil
results = []

# Latih dan evaluasi setiap model
for name, model in models.items():
    # Catat waktu mulai
    start_time = time.time()
    
    # Latih model
    model.fit(x_train_lazy, y_train_lazy)
    
    # Hitung waktu pelatihan
    training_time = time.time() - start_time
    
    # Prediksi
    y_train_pred = model.predict(x_train_lazy)
    y_test_pred = model.predict(x_test_lazy)
    
    # Hitung metrik
    r2_train = r2_score(y_train_lazy, y_train_pred)
    r2_test = r2_score(y_test_lazy, y_test_pred)
    rmsle = calculate_rmsle(y_test_lazy, y_test_pred)
    
    # Simpan hasil
    results.append({
        'Model': name,
        'R2 Train': round(r2_train, 2),
        'R2 Test': round(r2_test, 2),
        'RMSLE': round(rmsle, 2),
        'Training Time (s)': round(training_time, 2)
    })

# Buat DataFrame dan tampilkan dalam format yang diminta
df_results = pd.DataFrame(results)
print(df_results[['Model', 'R2 Train', 'R2 Test', 'RMSLE', 'Training Time (s)']].to_string(index=False))


import optuna
import lightgbm as lgbm
from sklearn.model_selection import cross_val_score
import numpy as np

# Fungsi objektif untuk Optuna
def lgbm_objective(trial):
    # Tentukan rentang nilai hyperparameter
    n_estimators = trial.suggest_int('n_estimators', 50, 2000)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 16)
    num_leaves = trial.suggest_int('num_leaves', 20, 150)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)

    # Deklarasi model LGBMRegressor dengan hyperparameter
    lgbm_model = lgbm.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=12,
        verbose=-1  # Suppress output dari LightGBM
    )


    # Evaluasi dengan cross-validation menggunakan neg_root_mean_squared_error
    score = cross_val_score(
        estimator=lgbm_model,
        X=x_train,
        y=y_train,
        scoring='neg_root_mean_squared_error',
        cv=5
    )

    return score.mean()

    
    ''' {'n_estimators': 1768, 'learning_rate': 0.03281522251757935, 
    'max_depth': 10, 'num_leaves': 93, 'min_child_samples': 22, 
    'colsample_bytree': 0.6549407479683254, 
    'reg_alpha': 2.4000724475343205, 'reg_lambda': 3.0705544650117753}'''

# Buat studi Optuna
#study = optuna.create_study(direction='maximize')

# Mulai optimasi hyperparameter
#study.optimize(func=lgbm_objective, n_trials=100)

# Tampilkan hasil terbaik
#print(f'Best Hyperparameter: {study.best_params}')
#print(f'Best Score: {study.best_value}')

# Ambil parameter terbaik
#lgbm_best_params = study.best_params


lgbm_best_param = {'n_estimators': 1768, 'learning_rate': 0.03281522251757935, 
    'max_depth': 10, 'num_leaves': 93, 'min_child_samples': 22, 
    'colsample_bytree': 0.6549407479683254, 
    'reg_alpha': 2.4000724475343205, 'reg_lambda': 3.0705544650117753}

lgbm_new = lgbm.LGBMRegressor(random_state = 12, **lgbm_best_param)
lgbm_new.fit(x_train, y_train)


plt.figure(figsize=(3,5))

features_importance = pd.DataFrame({
    'feature' : lgbm_new.booster_.feature_name(),
    'coefficient' : lgbm_new.feature_importances_
})

features_importance = features_importance.sort_values(by='coefficient', ascending= True)
plt.barh(y= features_importance['feature'], width= features_importance['coefficient'], color='skyblue')


# PREDICT MODEL AND SAVE SUBMISSION

y_pred = np.expm1(lgbm_new.predict(x_test))

submission = pd.read_table(r'/kaggle/input/playground-series-s5e5/sample_submission.csv', delimiter=',')   

output = pd.DataFrame(y_pred, columns=['Calories'])
output = pd.concat([submission.iloc[:,0] , output], axis=1)
output.rename(columns={output.columns[0] : 'id'}, inplace=True)
output['id'] = output['id'].astype(int)

# SAVE SUBMISSION
output.to_csv('submission.csv', index=False)

