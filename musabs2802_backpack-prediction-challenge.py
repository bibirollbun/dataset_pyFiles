import pandas as pd

import numpy as np
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

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df


df.info()


# Remove id column
df.drop('id', axis=1, inplace=True)


for col in df.columns:
    unqs = df[col].unique()
    print(col, unqs if len(unqs) < 10 else '[...]', len(unqs))


df.isna().sum()


df.fillna({'Brand': 'Other'}, inplace=True)
df.fillna({'Material': 'Other'}, inplace=True)
df.fillna({'Size': 'Other'}, inplace=True)
df.fillna({'Laptop Compartment': 'Other'}, inplace=True)
df.fillna({'Waterproof': 'Other'}, inplace=True)
df.fillna({'Style': 'Other'}, inplace=True)
df.fillna({'Color': 'Other'}, inplace=True)
df.fillna({'Weight Capacity (kg)': df['Weight Capacity (kg)'].median()}, inplace=True)


# Brand
pd.merge(df['Brand'].value_counts(), df['Brand'].value_counts(normalize=True), on='Brand')


# Material
pd.merge(df['Material'].value_counts(), df['Material'].value_counts(normalize=True), on='Material')


# Size
pd.merge(df['Size'].value_counts(), df['Size'].value_counts(normalize=True), on='Size')


# Laptop Compartment
pd.merge(df['Laptop Compartment'].value_counts(), df['Laptop Compartment'].value_counts(normalize=True), on='Laptop Compartment')


# Waterproof
pd.merge(df['Waterproof'].value_counts(), df['Waterproof'].value_counts(normalize=True), on='Waterproof')


# Style
pd.merge(df['Style'].value_counts(), df['Style'].value_counts(normalize=True), on='Style')


# Color
pd.merge(df['Color'].value_counts(), df['Color'].value_counts(normalize=True), on='Color')


# Weight Capacity (kg)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(df['Weight Capacity (kg)'], ax=ax[0])
sns.kdeplot(df['Weight Capacity (kg)'], ax=ax[1])

ax[0].title.set_text('Box Plot - Weight Capacity (kg)')
ax[1].title.set_text('KDE Plot - Weight Capacity (kg)')
plt.show()


# Price

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(df['Price'], ax=ax[0])
sns.kdeplot(df['Price'], ax=ax[1])

ax[0].title.set_text('Box Plot - Price')
ax[1].title.set_text('KDE Plot - Price')
plt.show()


df.info()


# Brand, Material, Size, Compartments, Laptop Compartment, Waterproof, Style, Color

fig, ax = plt.subplots(3, 3, figsize=(12, 15))

cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for i, col in enumerate(cols):
    row, col_index = divmod(i, 3)
    sns.boxplot(data=df, x=col, y="Price", ax=ax[row, col_index])
    ax[row, col_index].set_title(f'Box Plot - {col}')

plt.tight_layout()
plt.show()


sns.scatterplot(df, x='Weight Capacity (kg)', y='Price')


sns.heatmap(df[df.select_dtypes(include=['int', 'float']).columns].corr(), annot=True)


df = pd.get_dummies(df, columns=['Brand', 'Material', 'Style', 'Color'], dtype='int')


encoder = LabelEncoder()
df['Size'] = encoder.fit_transform(df['Size'])
df['Laptop Compartment'] = encoder.fit_transform(df['Laptop Compartment'])
df['Waterproof'] = encoder.fit_transform(df['Waterproof'])

df


def regression_pipeline(df, model, param_distributions, n_iter=20):
    X = df.drop(columns=['Price'])
    y = df['Price']

    numeric_features = X.columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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
    # "XGBoost": {'regressor__n_estimators': [50, 100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.2]},
    # "Support Vector Regression": {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']},
    # "K-Nearest Neighbors": {'regressor__n_neighbors': [3, 5, 7]},
    # "Decision Tree": {'regressor__max_depth': [None, 10, 20], 'regressor__min_samples_split': [2, 5, 10]}
}

# Instantiate models
models = {
    "Linear Regression": LinearRegression(),
    # "Random Forest": RandomForestRegressor(),
    # "Gradient Boosting": GradientBoostingRegressor(),
    # "XGBoost": XGBRegressor(),
    # "Support Vector Regression": SVR(),
    # "K-Nearest Neighbors": KNeighborsRegressor(),
    # "Decision Tree": DecisionTreeRegressor()
}


# Train and evaluate all models
for name, model in models.items():
    print(f"Running {name} with Hyperparameter Tuning...")
    best_model, best_params = regression_pipeline(df, model, param_distributions[name])





df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_test


df_test.fillna({'Brand': 'Other'}, inplace=True)
df_test.fillna({'Material': 'Other'}, inplace=True)
df_test.fillna({'Size': 'Other'}, inplace=True)
df_test.fillna({'Laptop Compartment': 'Other'}, inplace=True)
df_test.fillna({'Waterproof': 'Other'}, inplace=True)
df_test.fillna({'Style': 'Other'}, inplace=True)
df_test.fillna({'Color': 'Other'}, inplace=True)
df_test.fillna({'Weight Capacity (kg)': df['Weight Capacity (kg)'].median()}, inplace=True)


df_test = pd.get_dummies(df_test, columns=['Brand', 'Material', 'Style', 'Color'], dtype='int')


encoder = LabelEncoder()
df_test['Size'] = encoder.fit_transform(df_test['Size'])
df_test['Laptop Compartment'] = encoder.fit_transform(df_test['Laptop Compartment'])
df_test['Waterproof'] = encoder.fit_transform(df_test['Waterproof'])


predictions = best_model.predict(df_test.drop('id', axis=1))
predictions


submission_df = pd.DataFrame({'id': df_test['id'], 'Price': predictions})
submission_df


submission_df.to_csv('submission.csv', index=False)

