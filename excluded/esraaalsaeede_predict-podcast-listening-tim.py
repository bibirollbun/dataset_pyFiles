import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_train.head()


df_train.describe()


df_test.head()


df_sub.head()


df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)



df_train.shape,df_test.shape


df_train.isnull().sum()


df_test.isnull().sum()


df_train.shape,df_test.shape,df_sub.shape


df_train.corr()


df_train.dtypes


print("Publication_Time",df_train['Publication_Time'].unique())
print("_________________")
print("Episode_Sentiment",df_train['Episode_Sentiment'].unique())
print("_________________")
print("Podcast_Name",df_train['Podcast_Name'].unique())
print("_________________")
print("Publication_Day",df_train['Publication_Day'].unique())



common_cols = df_train.columns.intersection(df_test.columns)

numerical_cols = df_train[common_cols].select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_train[common_cols].select_dtypes(include=['object']).columns

df_train[numerical_cols] = df_train[numerical_cols].fillna(df_train[numerical_cols].median())
df_test[numerical_cols] = df_test[numerical_cols].fillna(df_train[numerical_cols].median())
df_train[categorical_cols] = df_train[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
df_test[categorical_cols] = df_test[categorical_cols].apply(lambda x: x.fillna(df_train[x.name].mode()[0]))


common_cols = df_train.columns.intersection(df_test.columns)
categorical_cols = df_train[common_cols].select_dtypes(include=['object']).columns


y = df_train['Listening_Time_minutes'] 


df_train = pd.read_csv('/kaggle/input/interstellarspacetri/datap/train.csv')
df_test  = pd.read_csv('/kaggle/input/interstellarspacetri/datap/test.csv')


X = df_train
X_test = df_test


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# LightGBM parameters
lgbm_params = {
    'n_estimators': 1000,
    'max_depth': 30,
    'learning_rate': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    # 'n_jobs': -1,
}

# KFold cross-validation
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []
test_preds = np.zeros(len(X_test)) 

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]   
    
    # Initialize LightGBM model
    model = LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)    
    
    # Validation predictions and score
    val_pred = model.predict(X_val)
    score = rmse(y_val, val_pred)
    scores.append(score)
    
    # Averaging test predictions across folds
    test_preds += model.predict(X_test) / n_splits      
    
    print(f"Fold {fold + 1} RMSE: {score:.4f}")

# Final cross-validation results
print(f'Optimized Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')



df_sub.head()


df_sub['Listening_Time_minutes'] = test_preds


df_sub.to_csv('submission3.csv', index=False)


df_sub.head()


df_sub['Listening_Time_minutes'].hist()




