import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head()


train.shape


train.isnull().sum()


train.duplicated().sum()


train.info()


train.describe()


sns.set_style('darkgrid')
plt.figure(figsize=(8,5))
sns.histplot(train['Listening_Time_minutes'], bins=30, kde=True, color='purple')
plt.title("Distribution of Podcast Listening Time")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Frequency")
plt.show()


train['Genre'].value_counts().plot(kind='pie' , shadow=True, autopct='%.2f')
plt.title('Distribution of Genre', fontsize=14)
plt.show()


print(train['Publication_Day'].value_counts())
print(train['Publication_Time'].value_counts())



plt.figure(figsize=(8,5))
sns.countplot(x=train['Publication_Day'], hue=train['Publication_Time'],palette='magma')
plt.title('Publication Day vs. Publication Time', fontsize=14)
plt.legend()
plt.show()


train.hist(figsize=(14,9))
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x=train['Episode_Length_minutes'], y=train['Listening_Time_minutes'], alpha=0.7)
plt.title("Listening Time vs. Episode Length", fontsize=14)
plt.xlabel("Episode Length (minutes)")
plt.ylabel("Listening Time (minutes)")
plt.show()


train.boxplot(figsize=(8,6))
plt.title('Check Outliers', fontsize=14)
plt.xticks(rotation=50)
plt.show()


test.head()


test.shape


test.isnull().sum()


test.info()


plt.figure(figsize=(10, 5))
sns.countplot(x="Genre", data=train, palette="summer", order=train["Genre"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Podcast Episode Count by Genre", fontsize=14)
plt.xlabel("Genre")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="Number_of_Ads", data=test, palette="viridis")
plt.title("Distribution of Ads in Podcast Episodes", fontsize=14)
plt.xlabel("Number of Ads")
plt.ylabel("Count")
plt.xticks(rotation=50)
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="Publication_Day", data=train, palette="mako", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Podcast Publication Frequency by Day", fontsize=14)
plt.xlabel("Day of the Week")
plt.ylabel("Number of Episodes")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="Episode_Sentiment", data=train, palette="YlOrBr", order=train["Episode_Sentiment"].value_counts().index)
plt.title("Episode Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


test.hist(figsize=(14,9))
plt.show()


 # Train Data
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean(),inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean(),inplace=True)
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(),inplace=True)
train.drop(columns=['id'], inplace=True)

# Test Data
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mean(),inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean(),inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(),inplace=True)
test.drop(columns=['id'], inplace=True)


train.isnull().sum()


test.isnull().sum()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score


# Define Taret var
X = train.drop(columns='Listening_Time_minutes')
y = train['Listening_Time_minutes']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)



num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])


preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


cat_pipeline=Pipeline([('Impute',SimpleImputer(strategy='most_frequent')),
                       ('scaler',OneHotEncoder(handle_unknown='ignore',drop='first',
                                              sparse_output=False))])

col_transformer=ColumnTransformer([('num',num_pipeline,num_cols),
                        ('cat',cat_pipeline,cat_cols)])



lg =lgb.LGBMRegressor(n_estimators=6000,
    max_depth=15,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=6,
    n_jobs=-1,
    verbose=-1)
model=Pipeline([('pre',col_transformer),
               ('lg',lg)])

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test,y_pred) :.2f}')
print(f'R2 score {r2_score(y_test,y_pred) * 100 :.2f}')
rmsc=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'RMSC = {rmsc :.4f}')
for actual,pred in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {actual :.2f}   | Predicted: {pred :.2f}')


test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_ids = test_df['id']

test = test_df.drop(columns=['id'])
test_predictions = model.predict(test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created: 'submission.csv'")

