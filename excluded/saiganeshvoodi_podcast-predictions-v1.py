import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")



# Load datasets
train= pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


train.info()


# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
sns.histplot(train["Listening_Time_minutes"], bins=30, kde=True)
plt.title("Distribution of Listening Time Minutes")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(train["Listening_Time_minutes"])
plt.title("Boxplot of Listening Time Minutes")
plt.show()



print("Train:\n",train.isnull().sum())
print("Test:\n",test.isnull().sum())


# Preprocess
def preprocess_data(train, test):
    train['is_train'] = 1
    test['is_train'] = 0
    test['Listening_Time_minutes'] = np.nan
    full_data = pd.concat([train, test], axis=0)
    full_data.drop(['Podcast_Name', 'Episode_Title'], axis=1, inplace=True)
    full_data['Episode_Length_minutes'] = full_data['Episode_Length_minutes'].fillna(full_data['Episode_Length_minutes'].median())
    full_data['Guest_Popularity_percentage'] = full_data['Guest_Popularity_percentage'].fillna(full_data['Guest_Popularity_percentage'].median())
    full_data['Number_of_Ads'] = full_data['Number_of_Ads'].fillna(full_data['Number_of_Ads'].mode()[0])
    cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for col in cat_cols:
        le = LabelEncoder()
        full_data[col] = le.fit_transform(full_data[col].astype(str))
    train_clean = full_data[full_data['is_train'] == 1].drop('is_train', axis=1)
    test_clean = full_data[full_data['is_train'] == 0].drop(['is_train', 'Listening_Time_minutes'], axis=1)
    return train_clean, test_clean


# Prepare data
train_clean, test_clean = preprocess_data(train, test)

X = train_clean.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train_clean['Listening_Time_minutes']
X_test = test_clean.drop(['id'], axis=1)


# Train individual ensemble models
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
et_model = ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42)



# Train models
rf_model.fit(X, y)
gb_model.fit(X, y)
et_model.fit(X, y)



rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)
et_preds = et_model.predict(X_test)


# Voting Regressor (Ensemble)
voting_model = VotingRegressor(estimators=[
    ('rf', rf_model),
    ('gb', gb_model),
    ('et', et_model)
])
voting_model.fit(X, y)
voting_preds = voting_model.predict(X_test)


# Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': voting_preds
})
submission.to_csv("submission.csv", index=False)


submission.isnull().sum()

