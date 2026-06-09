import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train.head()



cols=train.columns
for col in cols:
    print(f'{[col]} : : : {train[col].isnull().sum()}')


%%time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer,KNNImputer
from sklearn.model_selection import cross_val_score,train_test_split,RandomizedSearchCV,KFold
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,r2_score
target='Listening_Time_minutes'
train.drop(columns=['id'],inplace=True,errors='ignore')
x=train.drop(target,axis=1)
y=train[target]
kf= KFold(n_splits=5,random_state=42,shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)
num_cols=x.select_dtypes(include=(['int64','float64'])).columns.tolist()
cat_cols=x.select_dtypes(include=(['object'])).columns.tolist()
prepro = make_column_transformer(
    (make_pipeline(IterativeImputer()), num_cols),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder()), cat_cols)
)
lg = lgb.LGBMRegressor(
    n_estimators=1000,         
    learning_rate=0.05,        
    max_depth=-1,               
    num_leaves=31,             
    subsample=0.8,             
    colsample_bytree=0.8,       
    reg_alpha=0.1,        
    reg_lambda=0.1,            
    min_child_samples=20,    
    random_state=42,
    objective='regression',
    metric='rmse'
)
model=make_pipeline(prepro,lg)
print(cross_val_score(model,x,y, cv=kf))
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f'MSE: {mean_squared_error(y_test,y_pred) :.2f}')
print(f'R2 score {r2_score(y_test,y_pred) * 100 :.2f}')
rmsc=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'RMSC = {rmsc :.4f}')


import joblib

# Save the trained model to a file
joblib.dump(model, 'lgbm_model.pkl')

print("Model saved successfully!")

# Load the saved model from file
loaded_model = joblib.load('lgbm_model.pkl')

# Now you can use the loaded model for predictions
y_pred_loaded = loaded_model.predict(x_test)
print(f"Predictions using loaded model: {y_pred_loaded[:10]}")



# Never done this step :)
# Re-load the 'id' column from original test file
test_ids = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')['id']

# Generate predictions for the competition test set
test_predictions = model.predict(test)

# Prepare submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})

# Save to CSV for submission
submission.to_csv('submission.csv', index=False)





