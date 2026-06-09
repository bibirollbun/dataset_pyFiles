import pandas as pd
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

file_paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        file_paths.append(os.path.join(dirname, filename))

sample_submission, train_file, test_file = file_paths
rst = 42


def dataset_summary(df, file_name="DataFrame"):
    
    print(f"\n{'-'*20} FILE NAME {'-'*20}")
    print(file_name)
    
    print(f"\n{'-'*20} HEAD {'-'*20}")
    print(df.head())
    
    print(f"\n{'-'*20} MISSING VALUES {'-'*20}")
    print(df.isna().sum())
    
    print(f"\n{'-'*20} DATA TYPES {'-'*20}")
    print(df.dtypes)
    
    print(f"\n{'-'*20} DESCRIBE {'-'*20}")
    print(df.describe(include="all"))
    
    print(f"\n{'-'*20} INFO {'-'*20}")
    df.info()


train_data = pd.read_csv(train_file) 
test_data = pd.read_csv(test_file)
dataset_summary(train_data, "train Dataset")
dataset_summary(test_data,"Test Dataset")
print("Dropping id")
test_id = test_data[test_data.columns[0]]
test_data = test_data.drop(columns=[test_data.columns[0]])
train_data = train_data.drop(columns=[train_data.columns[0]])
print("Fixing Missing Values")
missing_col = list(train_data.isna().sum()[train_data.isna().sum() > 0].index)
print("Missing Columns")
print(missing_col)
num_cols = train_data[missing_col].select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_data[missing_col].select_dtypes(include=['object']).columns
# train data missing solved 
num_imputer = SimpleImputer(strategy="median")
train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
#  test data missing solved 
test_data[num_cols] = num_imputer.transform(test_data[num_cols])
test_data[cat_cols] = cat_imputer.transform(test_data[cat_cols])

print(f"\n{'-'*20} MISSING TRAIN {'-'*20}")
print(train_data.isna().sum())
print(f"\n{'-'*20} MISSING TEST {'-'*20}")
print(test_data.isna().sum())
missing_col = list(train_data.isna().sum()[train_data.isna().sum() > 0].index)
if(len(missing_col) == 0):
    print("NO MISSING VLAUE")

# converting into num from cat
cat_cols = list(train_data.dtypes[train_data.dtypes == 'object'].index)
print("Categrical columns")
print(cat_cols)
train_data = pd.get_dummies(train_data,columns=cat_cols,drop_first=True)
test_data = pd.get_dummies(test_data,columns=cat_cols,drop_first=True)
print("No Missing and All Num and Float Converted")
# we dont need scalling in node models like random forest/ catboost/lighgbm/xgboost
X = train_data.drop(columns=['price'])
y = train_data['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=rst)
test_data = test_data.reindex(columns=X_train.columns,fill_value=0)

models = [
    RandomForestRegressor(n_estimators=150,random_state=rst),
    LGBMRegressor(n_estimators=150,random_state=rst),
    XGBRegressor(n_estimators=150,random_state=rst),
    CatBoostRegressor(n_estimators=150,random_state=rst,verbose=0)
]
model = None
best_r2 = -9999
print("Tranning Started....")
print("-"*40)
for m in models:
    print(f"MODEL : {m.__class__.__name__}")
    m.fit(X_train,y_train)
    y_pred = m.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE : {rmse}")
    print(f"r2 : {r2}")
    print("-"*40)
    if r2 > best_r2:
        best_r2 = r2
        model = m

print(f"Best Model : {model.__class__.__name__}")
print(f"Best r2 : {best_r2}")

test_pred = model.predict(test_data)
sample_sub = pd.read_csv(sample_submission)
print(sample_sub.head())
submission = pd.DataFrame({
    sample_sub.columns[0]: test_id,     
    sample_sub.columns[1]: test_pred   
})

submission.to_csv("submission.csv",index=False)
print("✅ All Done")

