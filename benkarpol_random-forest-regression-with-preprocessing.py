import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder




# Load data
df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv') 

# Quick look
print(df.head())
print(df.info())
print(df.isnull().sum())



df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)
df['Number_of_Ads'].fillna(df['Number_of_Ads'].mode()[0], inplace=True)



label_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])



X = df.drop(['id', 'Listening_Time_minutes'], axis=1)
y = df['Listening_Time_minutes']




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.model_selection import RandomizedSearchCV



param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}



rf = RandomForestRegressor(random_state=42)

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,  # Try 20 random combinations
    cv=3,       # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1   # Use all CPU cores
)

rf_random.fit(X_train, y_train)



best_rf = rf_random.best_estimator_

y_pred = best_rf.predict(X_val)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


test_df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Number_of_Ads'].fillna(df['Number_of_Ads'].mode()[0], inplace=True)


for col in label_cols:
    test_df[col] = le.fit_transform(test_df[col])



mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")


X_test = test_df.drop(['id'], axis=1)
test_predictions = best_rf.predict(X_test)


submission = pd.DataFrame({'id': test_df['id'], 'Listening_Time_minutes': test_predictions})
submission.to_csv('submission.csv', index=False)

