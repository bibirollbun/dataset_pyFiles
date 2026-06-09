import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head()


train.shape


train.info()


train.isnull().sum()


train.describe()


test.head()


test.shape


test.info()


test.isnull().sum()


test.describe()


sns.histplot(train['Listening_Time_minutes'], kde=True, bins=50, color='orange');


sns.scatterplot(x=train['Episode_Length_minutes'], y=train['Listening_Time_minutes'], alpha=0.3, color='orange')
plt.title('Episode Length vs. Listening Time')
plt.plot([0, 200], [0, 200], 'r--'); # x=y 


sns.boxplot(x=train['Genre'], y=train['Listening_Time_minutes'], palette='pastel')
plt.xticks(rotation=45)
plt.title('Listening Time based on Podcast Genre');


sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='plasma', fmt=".2f")
plt.title('Correlation Matrix');


#Imputation
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(0)

train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(-1)
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(-1)

train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train.groupby('Genre')['Episode_Length_minutes'].transform('median'))
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test.groupby('Genre')['Episode_Length_minutes'].transform('median'))
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median())

#Encoding
categorical_cols = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
for col in categorical_cols:
    le = LabelEncoder()
    all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(all_values)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train_ids = train['id']
test_ids = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


train.head()


test.head()


x = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)


import xgboost as xgb

model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=1000,learning_rate=0.05,max_depth=6,
                         random_state=42,n_jobs=-1)

model.fit(x_train, y_train,eval_set=[(x_test, y_test)],early_stopping_rounds=50,verbose=100)

preds = model.predict(x_test)
rmse = mean_squared_error(y_test, preds, squared=False) 
r2 = r2_score(y_test, preds)

print(f"Validation RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

#Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10, height=0.5)
plt.show()


final_predictions = model.predict(test)
final_predictions = np.maximum(final_predictions, 0)

submission = pd.DataFrame({'id': test_ids,'Listening_Time_minutes': final_predictions})
submission.to_csv('submission.csv', index=False)


import joblib

genre_medians = train.groupby('Genre')['Episode_Length_minutes'].median().to_dict()
global_median = train['Episode_Length_minutes'].median()

encoders = {}
categorical_cols = ['Podcast_Name', 'Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment']

for col in categorical_cols:
    le = LabelEncoder()
    all_values = pd.concat([pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')[col], 
                            pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')[col]], axis=0).astype(str)
    le.fit(all_values)
    encoders[col] = le

bundle = {'model': model,'encoders': encoders,'genre_medians': genre_medians,'global_median': global_median}

joblib.dump(bundle, 'model_assets.pkl')

