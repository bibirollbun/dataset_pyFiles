from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


df.columns


df= df.drop(columns=['Genre','Publication_Day', 'Publication_Time','id', 'Podcast_Name','Episode_Title'])



# Option 1: Fill with median (robust)
# df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean(), inplace=True)
df=df.dropna()
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean(), inplace=True)
df['Number_of_Ads'].fillna(0, inplace=True)  # Assuming 0 ads if missing



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# df['Podcast_Name'] = le.fit_transform(df['Podcast_Name'])
# df['Genre'] = le.fit_transform(df['Genre'])
# df['Publication_Day'] = le.fit_transform(df['Publication_Day'])
# df['Publication_Time'] = le.fit_transform(df['Publication_Time'])
# df['Episode_Title'] = le.fit_transform(df['Episode_Title'])
# df['Episode_Sentiment'] = le.fit_transform(df['Episode_Sentiment'])

df = pd.get_dummies(df)
df = df.astype(int)



df


X = df.drop(columns=['Listening_Time_minutes'])
y = df['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import VotingRegressor
import xgboost as xgb

# Define 5 slightly different XGBoost models
model1 = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=4,
    min_child_weight=3,
    random_state=42
)

model2 = xgb.XGBRegressor(
    n_estimators=220,
    learning_rate=0.08,
    max_depth=9,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.3,
    reg_lambda=5,
    min_child_weight=2,
    random_state=43
)

model3 = xgb.XGBRegressor(
    n_estimators=180,
    learning_rate=0.12,
    max_depth=11,
    subsample=0.75,
    colsample_bytree=0.7,
    reg_alpha=0.7,
    reg_lambda=3,
    min_child_weight=4,
    random_state=44
)

model4 = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.09,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.85,
    reg_alpha=0.4,
    reg_lambda=6,
    min_child_weight=3,
    random_state=45
)

model5 = xgb.XGBRegressor(
    n_estimators=210,
    learning_rate=0.11,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.6,
    reg_lambda=4,
    min_child_weight=2,
    random_state=46
)

estimators=[
        ('xgb1', model1),
        ('xgb2', model2),
        ('xgb3', model3),
        ('xgb4', model4),
        ('xgb5', model5)
    ]


    



for name, model in estimators:
    model.fit(X_train, y_train)  # Train on 80%
    y_pred = model.predict(X_test)  # Predict on 20%
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(name, "RMSE:", round(rmse, 2))



# Voting Regressor
voting_regressor = VotingRegressor(estimators=estimators)
voting_regressor.fit(X_train, y_train)

y_pred_voting = voting_regressor.predict(X_test)
rmse_voting = np.sqrt(mean_squared_error(y_test, y_pred_voting))
print("Voting Regressor RMSE:", round(rmse_voting, 5))








