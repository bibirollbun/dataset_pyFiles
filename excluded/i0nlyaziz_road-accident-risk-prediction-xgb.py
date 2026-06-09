import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# import the necessary libraries


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# load the data


test_ids = test['id']

# save the id column


train.head()


test.head()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columnns = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']

for i in columnns :
  train[i] = encoder.fit_transform(train[i])

# preprocess the train data


train.drop(columns=['id'],inplace=True)

# drop the unnecessary column


train.isnull().sum().sum()

# check for missing values


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columnns = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']

for i in columnns :
  test[i] = encoder.fit_transform(test[i])

# preprocess the test data


test.drop(columns=['id'],inplace=True)

# drop the unnecessary column


test.isnull().sum().sum()

# check for missing values


road = train['road_type'].value_counts()

Labels = ['Highway','Rural','Urban']

plt.figure(figsize=(8,8))
plt.pie(road,autopct='%1.1f%%',labels=Labels)
plt.title("Road Type Distribution")
plt.show()

# explore the data with pieplot


plt.figure(figsize=(8,8))
sns.countplot(x='num_lanes',data=train)
plt.title("Distribution of the Number of Lanes")
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.histplot(x='curvature',data=train)
plt.title("Road Curvature Distribution")
plt.show()

# explore the data with histplot


plt.figure(figsize=(8,8))
sns.countplot(x='speed_limit',data=train)
plt.title("Speed Limit Distribution")
plt.show()

# explore the data with countplot


light = train['lighting'].value_counts()

Labels = ['dim','daylight','night']

plt.figure(figsize=(8,8))
plt.pie(road,autopct='%1.1f%%',labels=Labels)
plt.title("Distribution of Lighting Conditions")
plt.show()

# explore the data with pieplot


weather = train['weather'].value_counts()

Labels = ['foggy','clear','rainy']

plt.figure(figsize=(8,8))
plt.pie(road,autopct='%1.1f%%',labels=Labels)
plt.title("weather Distribution")
plt.show()

# explore the data with pieplot


plt.figure(figsize=(8,8))
sns.countplot(x='time_of_day',data=train)
plt.xticks(ticks=[0,1,2],
           labels=['afternoon','evening','morning'])
plt.title('Time of the Day Distribution')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='road_signs_present',data=train)
plt.xticks(ticks=[0,1],
           labels=['False','True'])
plt.title('Road Signs Present')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='public_road',data=train)
plt.xticks(ticks=[0,1],
           labels=['False','True'])
plt.title('Public Road')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='holiday',data=train)
plt.xticks(ticks=[0,1],
           labels=['False','True'])
plt.title('Holiday')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='school_season',data=train)
plt.xticks(ticks=[0,1],
           labels=['False','True'])
plt.title('School Season')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.heatmap(train.corr(),annot=True)
plt.title('Heat Map Correlation')
plt.show()

# explore the data with heatmap


x = train.drop(columns=['accident_risk'],axis=1)
y = train['accident_risk']


from sklearn.model_selection import train_test_split

x_train , x_valid , y_train , y_valid = train_test_split(x,y,test_size=0.3,random_state=42)

# split the data


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

xgb_model = XGBRegressor(random_state=42, verbosity=0)

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3, 5]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=50,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

xgb_random.fit(x_train, y_train)

print("Best XGB params:", xgb_random.best_params_)
print("Best XGB score:", xgb_random.best_score_)

# use RandomizedSearchCV to find the best hyperparameters


from xgboost import XGBRegressor


Model = XGBRegressor(
     subsample=0.8,
    reg_lambda=10,
    reg_alpha=0,
    n_estimators=100,
    min_child_weight=1,
    max_depth=7,
    learning_rate=0.1,
    gamma=0,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

Model.fit(x_train,y_train)

# train the model


y_pred = Model.predict(x_valid)


from sklearn.metrics import r2_score

accuracy = r2_score(y_valid,y_pred)
print(f""" The R2 : {accuracy :.2f}""")

# evaluate the model with r2 score


from sklearn.metrics import mean_absolute_error

accuracy = mean_absolute_error(y_valid,y_pred)
print(f""" The MAE : {accuracy :.2f}""")

# evaluate the model with MAE


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
print(f"The RMSE : {rmse:.2f}")

# evaluate the model with RMSE


print(f"""Train Score : {Model.score(x_train,y_train) * 100:.2f}%""")
print(f"""Valid Score : {Model.score(x_valid,y_valid) * 100:.2f}%""")

# display training and valid accuracy


predictions = Model.predict(test)


submission = pd.DataFrame({
    "id": test_ids,
    "accedint_risk": predictions
})

submission.to_csv("submission.csv", index=False)


submission

