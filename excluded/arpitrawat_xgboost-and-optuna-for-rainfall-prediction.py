import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_org = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


train.shape, test.shape


train.info()


test.info()


train.head(5)


plt.plot(train['day'])


# plots for train set 
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

columns = train.drop(columns=['id', 'day', 'rainfall']).columns 

for ax, col in zip(axes.flat, columns):  
    sns.scatterplot(x=train['day'], y=train[col], ax=ax, hue=train['rainfall'])
    ax.set_title(f"day vs {col}") 

plt.tight_layout()  
plt.show()



# plots for test set
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

columns = test.drop(columns=['id', 'day']).columns  

for ax, col in zip(axes.flat, columns): 
    sns.scatterplot(x=test['day'], y=test[col], ax=ax)
    ax.set_title(f"day vs {col}")  

plt.tight_layout()  
plt.show()



corr = train[train.drop(columns='id').columns].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


# Feature Engineering inspired from https://www.kaggle.com/code/muhammadqasimshabbir/854scorebinary-prediction-with-a-rainfall-dataset
def feature_engineering(df):
    
    # Convert 'day' to datetime
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    
    # Extract temporal features
    df['month'] = df['day'].dt.month
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Temperature features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']
    
    # Dew point depression
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    
    # Wind chill factor (simplified version)
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)
    
    # Interaction features
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    
    # Rolling statistical features
    df['rolling_temp_mean'] = df['avg_temp'].rolling(window=7).mean()
    df['rolling_wind_mean'] = df['windspeed'].rolling(window=7).mean()
    df['rolling_humidity_mean'] = df['humidity'].rolling(window=7).mean()


    # Lag features
    df['temp_lag_1'] = df['avg_temp'].shift(1)
    df['humidity_lag_1'] = df['humidity'].shift(1)
    df['windspeed_lag_1'] = df['windspeed'].shift(1)

    
    
    # Pressure-Temperature interaction
    df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
    # Wind-Speed-Temperature interaction
    df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']

    
    # Sunshine-Cloud interaction
    df['sunshine_cloud_interaction'] = df['sunshine'] * df['cloud']
    
    
    # Season feature
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                      'Summer' if 6 <= x <= 8 else
                                      'Autumn' if 9 <= x <= 11 else 'Winter')

    # Binary encoding for season
    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    # Drop original 'day' column
    df.drop(columns=['day'], inplace=True)
    
    return df



train_org.columns = train_org.columns.str.strip()
train_org['rainfall'] = train_org['rainfall'].str.lower().map({'yes': 1, 'no': 0})
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


column_order = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']

train = train[column_order]  
train_org = train_org[column_order]  
train = pd.concat([train, train_org], axis=0, ignore_index=True)

# Remove 'rainfall' correctly from column_order
column_order_without_rainfall = [col for col in column_order if col != 'rainfall']
test = test[column_order_without_rainfall]



train = feature_engineering(train)
test = feature_engineering(test)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train.drop(columns=['rainfall']),train['rainfall'],test_size=0.2)


import xgboost as xgb

# Define best parameters obtained from Optuna
Params = {
    'n_estimators': 57,
    'max_depth': 18,
    'learning_rate': 0.24244887641503957,
    'subsample': 0.6160718780228461,
    'colsample_bytree': 0.7452106714919773,
    'gamma': 3.043026485920432,
    'reg_alpha': 8.928697499352786,
    'reg_lambda': 2.29959889992905
}

# Initialize the XGBoost Classifier with best parameters
best_model = xgb.XGBClassifier(**Params, use_label_encoder=False, eval_metric="logloss")

# Fit the model on training data
best_model.fit(X_train, y_train)



from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')

# Print results
print(f"ROC AUC Scores: {cv_scores}")
print(f"Mean ROC AUC Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

y_prob = best_model.predict_proba(X_test)[:, 1]  # Take probability of class 1
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score on Validation Set: {roc_auc:.4f}")


predictions=best_model.predict(test)
submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.rainfall=predictions
submission.to_csv("submission.csv", index=False)




