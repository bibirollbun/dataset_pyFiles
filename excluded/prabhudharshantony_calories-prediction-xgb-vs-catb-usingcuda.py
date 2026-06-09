# ==== Basic Libs ====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==== Preprocessing, ML models and performance evaluation ====
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_log_error

# ==== Remove Warnings ====
import warnings
warnings.simplefilter("ignore")


# ==== Importing Traning Data ====
df = pd.read_csv('train.csv')
df


# ==== Column id won't be used for traning, hence remove and checking for null values ====
df = df.drop(columns= ['id'])
df.isna().sum()


# ==== Preprocessing ====
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


# ==== Checking for Correaltion/ Linearity ====
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot= True, fmt= '0.2f')
plt.title('Correlation Matrix')
plt.show()


# ==== Create traning and testing splits ====
x = df.drop(columns=['Calories'])
y = df['Calories']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state= 42, test_size=0.2 )


# ==== Checking Two Models and picking the one with the least Root Mean Squared Log Error ====

# ==== XGBoost, using CUDA Btw ====
XG = XGBRegressor(
    tree_method='gpu_hist',  # Use GPU
    predictor='gpu_predictor',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    eval_metric='logloss',
    use_label_encoder=False,
    objective='reg:gamma'
)
XG.fit(xtrain,ytrain)
xg_ypred = XG.predict(xtest)


# ==== CatBoost. Again, using CUDA Btw ====
y_train_log = np.log1p(ytrain)  # log(y + 1), step to ensure no negative values during prediction

Cat = CatBoostRegressor(
    iterations=1500,
    depth=6,
    learning_rate=0.1,
    loss_function='RMSE',
    task_type='GPU',
    devices='0',
    verbose=100
)
Cat.fit(xtrain,y_train_log)
cat_ypred_log = Cat.predict(xtest)
cat_ypred = np.expm1(cat_ypred_log) # Converting ypred back


# ==== Function to check performance ====
def evaluation(ytest,ypred):
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytest, ypred)
    rmsle = root_mean_squared_log_error(ytest, ypred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"RMSLE: {rmsle}")

print('XGBosst Evaluation: \n')
evaluation(ytest,xg_ypred)
print('***************************** \n')

print('CatBoot Evaluation: \n')
evaluation(ytest,cat_ypred)


# ==== CATBoost performs better ====
    # Played around with iterration and depth values and found out this worked best for me
    
# ==== Let's import Test data =====
df_test = pd.read_csv('test.csv')

# ==== LabelEncode, Features(x), Saving id for the submission.csv ====
df_test['Sex'] = le.fit_transform(df_test['Sex'])

x_test_main = df_test.drop(columns=['id'])

id = df_test['id']




# Honestly, don't know why predictions are wayy off when i use x y directly, so doing this instead
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state= 42, test_size=0.1 )

y_train_log = np.log1p(ytrain)  # log(y + 1), step to ensure no negative values during prediction

Cat.fit(xtrain,y_train_log)
cat_ypred_log = Cat.predict(x_test_main)

cat_ypred = np.expm1(cat_ypred_log) # Converting ypred back


# ====== Final Predict ======
cat_ypred_log = Cat.predict(x_test_main)
ypred = np.expm1(cat_ypred_log)


final_csv = pd.DataFrame({
    'id':id,
    'Calories' : cat_ypred
})
final_csv


# ==== Check for any negative values ====
filtered_df = final_csv[final_csv['Calories'] < 0]
filtered_df


#==== Save csv ====
final_csv.to_csv('submission.csv', index= False)

