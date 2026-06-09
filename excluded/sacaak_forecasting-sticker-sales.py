import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Some warnings are ignored.
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df.head()


df.shape


df.info()


df.describe().T


(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


# Since the ratio is less than 5%, we can neglect missing values.

df = df.drop(df[df['num_sold'].isnull()].index, axis=0)


df.columns


test_df.columns


df['country'].value_counts()


df = pd.get_dummies(df, columns=['country'], prefix=['country'], dtype=int, drop_first=True)
test_df = pd.get_dummies(test_df, columns=['country'], prefix=['country'], dtype=int, drop_first=True)


df['store'].value_counts()


# Since spaces in values may arise problems, we will replace them with "_"

df['store'] = df['store'].str.replace(' ', '_')
test_df['store'] = test_df['store'].str.replace(' ', '_')


df = pd.get_dummies(df, columns=['store'], prefix=['store'], dtype=int, drop_first=True)
test_df = pd.get_dummies(test_df, columns=['store'], prefix=['store'], dtype=int, drop_first=True)


df['product'].value_counts()


df['product'] = df['product'].str.replace(' ', '_')
test_df['product'] = test_df['product'].str.replace(' ', '_')


df = pd.get_dummies(df, columns=['product'], prefix=['product'], dtype=int, drop_first=True)
test_df = pd.get_dummies(test_df, columns=['product'], prefix=['product'], dtype=int, drop_first=True)


df.columns = df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()


df['date'] = pd.to_datetime(df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


df.info()


df['year'] = df['date'].dt.year
test_df['year'] = test_df['date'].dt.year


df['month'] = df['date'].dt.month
test_df['month'] = test_df['date'].dt.month


def assign_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn


df['season'] = df['date'].apply(assign_season)
test_df['season'] = test_df['date'].apply(assign_season)


df = df.drop(columns=['date'])
test_df = test_df.drop(columns=['date'])


df.head()


X = df.iloc[:, 2:]
y = df["num_sold"]


from sklearn.model_selection import train_test_split


X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=17)
rf_model.fit(X_train, y_train)


from sklearn.model_selection import RandomizedSearchCV

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

rf_best_random = RandomizedSearchCV(rf_model, rf_params, cv=3, n_iter=5, n_jobs=-1, verbose=True, random_state=17)
rf_best_random.fit(X_train, y_train)

rf_final = rf_model.set_params(**rf_best_random.best_params_, random_state=17).fit(X_train, y_train)


from sklearn.metrics import mean_squared_error

y_pred = rf_final.predict(X_validation)

rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
print("RMSE:", rmse)


from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_validation, y_pred)


predictions = rf_final.predict(test_df.drop(columns=['id']))  # 'id' sütununu çıkarın

submission_df = test_df[['id']].copy()

submission_df['num_sold'] = predictions

submission_df.to_csv('submission.csv', index = False)

