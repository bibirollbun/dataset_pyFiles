import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train.head()


# For train
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean(),inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean(),inplace=True)
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(),inplace=True)
train.drop(columns=['id'], inplace=True,errors='ignore')
# For test
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mean(),inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean(),inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(),inplace=True)
test.drop(columns=['id'], inplace=True) 


num_cols=train.select_dtypes(include=(['int64','float64'])).columns.tolist()
cat_cols=train.select_dtypes(include=(['object'])).columns.tolist()


sns.set(style='whitegrid')
for i in num_cols:
    plt.figure(figsize=(6,5))
    sns.histplot(train[i],kde=True,color='blue',edgecolor='black')
    plt.xlabel(f'Distribution of {i}')
    plt.ylabel('Frequency')
    plt.show()


sns.set(style='whitegrid')
cat_cols_modified=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for i in cat_cols_modified:
    plt.figure(figsize=(6,5))
    sns.histplot(train[i],kde=True,color='blue',edgecolor='black')
    plt.xlabel(f'Distribution of {i}')
    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.show()


from sklearn.model_selection import cross_val_score,train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,r2_score
target='Listening_Time_minutes'
train.drop(columns=['id'],inplace=True,errors='ignore')
x=train.drop(target,axis=1)
y=train[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_cols=x.select_dtypes(include=(['int64','float64'])).columns.tolist()
cat_cols=x.select_dtypes(include=(['object'])).columns.tolist()
num_pipeline=Pipeline([('Impute',SimpleImputer(strategy='mean')),
                       ('scaler',StandardScaler())])

cat_pipeline=Pipeline([('scaler',OneHotEncoder(handle_unknown='ignore',drop='first',
                                              sparse_output=False))])

col_transformer=ColumnTransformer([('num',num_pipeline,num_cols),
                        ('cat',cat_pipeline,cat_cols)])

lg=lgb.LGBMRegressor(
        n_iter=3000,
        max_depth=100,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
        random_state=69,
        subsample=0.8,
        subsample_freq=1)


    
model=Pipeline([('pre',col_transformer),
               ('lg',lg)])

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f'MSE: {mean_squared_error(y_test,y_pred) :.2f}')
print(f'R2 score {r2_score(y_test,y_pred) * 100 :.2f}')
rmsc=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'RMSC = {rmsc :.4f}')
for actual,pred in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {actual :.2f}   | Predicted: {pred :.2f}')


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





