!pip install autogluon


import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_train


df_train.info()


# Dropping `id` due to redundency
df_train.drop('id', axis=1, inplace=True)


for col in df_train.columns:
    unqs = df_train[col].unique()
    print(col, unqs if len(unqs)<15 else '[...]', len(unqs))


# Replacing nans in `Number_of_Ads` with 0
df_train.loc[df_train['Number_of_Ads'].isna(), 'Number_of_Ads'] = 0


# Label encode categorical columns

lencoder = LabelEncoder()

df_train['Publication_Day'] = lencoder.fit_transform(df_train['Publication_Day'])
df_train['Publication_Time'] = lencoder.fit_transform(df_train['Publication_Time'])
df_train['Episode_Sentiment'] = lencoder.fit_transform(df_train['Episode_Sentiment'])


df_train['Episode_Title'].unique()


# Convert `Episode_Title` to int data type
df_train['Episode_Title'] = df_train['Episode_Title'].str.replace('Episode ', '').astype(int)


df_train.duplicated().sum()


df_train.isna().sum()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df_train, x='Episode_Title', ax=ax[0])
ax[0].set_title('Distribution of Episode_Title')

sns.scatterplot(df_train, x='Episode_Title', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Episode_Title vs Listening_Time_minutes')

print("Correlation: ", df_train['Episode_Title'].corr(df_train['Listening_Time_minutes']))
print("Skewness: ", df_train['Episode_Title'].skew())
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(df_train, x='Episode_Length_minutes', ax=ax[0])
ax[0].set_title('Boxplot of Episode_Length_minutes')

sns.scatterplot(df_train, x='Episode_Length_minutes', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Episode_Length_minutes vs Listening_Time_minutes')

print("Correlation: ", df_train['Episode_Length_minutes'].corr(df_train['Listening_Time_minutes']))
print("Correlation: ", df_train['Episode_Length_minutes'].skew())
plt.show()


# Remove outliers
df_train = df_train[~(df_train['Episode_Length_minutes']>150)]


# Impute missing values with Linear regression

not_null = df_train[df_train['Episode_Length_minutes'].notnull()]
is_null = df_train[df_train['Episode_Length_minutes'].isnull()]

model = LinearRegression()
model.fit(not_null[['Listening_Time_minutes']], not_null['Episode_Length_minutes'])

predicted = model.predict(is_null[['Listening_Time_minutes']])
df_train.loc[df_train['Episode_Length_minutes'].isnull(), 'Episode_Length_minutes'] = predicted


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(df_train['Genre'].value_counts().reset_index(), x='Genre', y='count', ax=ax[0])
ax[0].set_title('Distribution of Genre')

sns.boxplot(df_train, x='Genre', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Boxplots of Genre with Listening_Time_minutes')

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df_train, x='Host_Popularity_percentage', ax=ax[0])
ax[0].set_title('Distribution of Host_Popularity_percentage')

sns.scatterplot(df_train, x='Host_Popularity_percentage', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Host_Popularity_percentage vs Listening_Time_minutes')

print("Correlation: ", df_train['Host_Popularity_percentage'].corr(df_train['Listening_Time_minutes']))
print("Skewness: ", df_train['Host_Popularity_percentage'].skew())
plt.show()


# Remove outliers
df_train = df_train[(df_train['Host_Popularity_percentage']>=20) & (df_train['Host_Popularity_percentage']<=100)]


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(df_train['Publication_Day'].value_counts().reset_index(), x='Publication_Day', y='count', ax=ax[0])
ax[0].set_title('Distribution of Publication_Day')

sns.boxplot(df_train, x='Publication_Day', y='Listening_Time_minutes', ax=ax[1])
ax[0].set_title('Boxplot of Publication_Day with Listening_Time_minutes')

print("Correlation: ", df_train['Publication_Day'].corr(df_train['Listening_Time_minutes']))
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(df_train['Publication_Time'].value_counts().reset_index(), x='Publication_Time', y='count', ax=ax[0])
ax[0].set_title('Distribution of Publication_Time')

sns.boxplot(df_train, x='Publication_Time', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Boxplot of Publication_Time with Listening_Time_minutes')

print("Correlation: ", df_train['Publication_Time'].corr(df_train['Listening_Time_minutes']))
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df_train, x='Guest_Popularity_percentage', ax=ax[0])
ax[0].set_title('Distribution of Guest_Popularity_percentage')

sns.scatterplot(df_train, x='Guest_Popularity_percentage', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Guest_Popularity_percentage vs Listening_Time_minutes')

print("Correlation: ", df_train['Guest_Popularity_percentage'].corr(df_train['Listening_Time_minutes']))
print("Skewness: ", df_train['Guest_Popularity_percentage'].skew())
plt.show()


# Remove outliers
df_train = df_train[~(df_train['Guest_Popularity_percentage']>100)]


# Imputing null values with median
df_train.loc[df_train['Guest_Popularity_percentage'].isna(), 'Guest_Popularity_percentage'] = df_train['Guest_Popularity_percentage'].median()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(df_train['Number_of_Ads'].value_counts().reset_index(), x='Number_of_Ads', y='count', ax=ax[0])
ax[0].set_title('Distribution of Number_of_Ads')

sns.boxplot(df_train, x='Number_of_Ads', y='Listening_Time_minutes', ax=ax[1])
ax[1].set_title('Boxplots of Number_of_Ads with Listening_Time_minutes')

print("Correlation: ", df_train['Number_of_Ads'].corr(df_train['Listening_Time_minutes']))
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(df_train['Episode_Sentiment'].value_counts().reset_index(), x='Episode_Sentiment', y='count', ax=ax[0])
ax[0].set_title('Distribution of Episode_Sentiment')

sns.boxplot(df_train, x='Episode_Sentiment', y='Listening_Time_minutes', ax=ax[1])
ax[0].set_title('Boxplots of Episode_Sentiment with Listening_Time_minutes')

print("Correlation: ", df_train['Episode_Sentiment'].corr(df_train['Listening_Time_minutes']))
plt.show()


print("Skewness: ", df_train['Listening_Time_minutes'].skew())

sns.histplot(df_train['Listening_Time_minutes'])
plt.show()


plt.figure(figsize=(12, 7))
sns.heatmap(df_train.select_dtypes(exclude='object').corr(), annot=True, cmap='coolwarm')
plt.show()


def chi_square_test(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    results = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):  # Avoid duplicate pairs
            col1, col2 = categorical_cols[i], categorical_cols[j]
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append((col1, col2, chi2, p))
    
    results_df = pd.DataFrame(results, columns=["Feature 1", "Feature 2", "Chi2 Score", "p-value"])
    correlated_features = results_df[results_df["p-value"] < 0.05]
    
    return correlated_features

chi_square_results = chi_square_test(df_train.drop(columns=['Listening_Time_minutes'], axis=1))
print(chi_square_results)


# Remove `Podcast_Name`
df_train.drop('Podcast_Name', axis=1, inplace=True)


# One-Hot Encoding
df_train = pd.get_dummies(df_train, dtype='int', drop_first=True)


# Reset index
df_train.reset_index(inplace=True, drop=True)


def regression_pipeline(df, model, param_distributions, n_iter=20):
    X = df.drop(columns=['Listening_Time_minutes'])
    y = df['Listening_Time_minutes']

    numeric_features = X.columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Best Parameters: {random_search.best_params_}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    return best_model, random_search.best_params_

# Define hyperparameter distributions
param_distributions = {
    "Linear Regression": {},
    # "Random Forest": {'regressor__n_estimators': [50, 100, 200], 'regressor__max_depth': [None, 10, 20]},
    # "Gradient Boosting": {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]},
    "XGBoost": {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]},
    # "Support Vector Regression": {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']},
    # "K-Nearest Neighbors": {'regressor__n_neighbors': [3, 5, 7]},
    # "Decision Tree": {'regressor__max_depth': [None, 10, 20], 'regressor__min_samples_split': [2, 5, 10]}
}

# Instantiate models
models = {
    "Linear Regression": LinearRegression(),
    # "Random Forest": RandomForestRegressor(),
    # "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    # "Support Vector Regression": SVR(),
    # "K-Nearest Neighbors": KNeighborsRegressor(),
    # "Decision Tree": DecisionTreeRegressor()
}


# Train and evaluate all models
for name, model in models.items():
    print(f"Running {name} with Hyperparameter Tuning...")
    best_model, best_params = regression_pipeline(df_train, model, param_distributions[name])


def autogluon_regression_pipeline(df, target='Listening_Time_minutes'):
    # Split data
    train_data, test_data = train_test_split(df, test_size=0.2)

    # Train using AutoGluon
    predictor = TabularPredictor(label=target, eval_metric='rmse').fit(
        train_data=train_data,
        time_limit=600,
        presets='best_quality',
    )

    # Evaluate on test data
    performance = predictor.evaluate(test_data)

    # Predictions
    y_test = test_data[target]
    y_pred = predictor.predict(test_data.drop(columns=[target]))

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    return predictor


predictor = autogluon_regression_pipeline(df_train, target='Listening_Time_minutes')
predictor


predictor.leaderboard()





# Read test data
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_test


# Perform transformation operations

df_test.loc[df_test['Number_of_Ads'].isna(), 'Number_of_Ads'] = 0

df_test['Publication_Day'] = lencoder.fit_transform(df_test['Publication_Day'])
df_test['Publication_Time'] = lencoder.fit_transform(df_test['Publication_Time'])
df_test['Episode_Sentiment'] = lencoder.fit_transform(df_test['Episode_Sentiment'])

df_test['Episode_Title'] = df_test['Episode_Title'].str.replace('Episode ', '').astype(int)
df_test = pd.get_dummies(df_test, dtype='int', drop_first=True)


# Prediction

y_pred = predictor.predict(df_test.drop('id', axis=1))
y_pred


submission = pd.DataFrame({'id': df_test['id'], 'Listening_Time_minutes': y_pred})
submission


# Save submission
submission.to_csv('submission.csv', index=False)

