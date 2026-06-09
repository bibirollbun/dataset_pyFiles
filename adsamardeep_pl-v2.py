import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


def feature_engineering(df):
    # Extract hour and minute from Publication_Time and handle missing values
    df['Publication_Hour'] = df['Publication_Time'].str.extract('(\d+):').astype(float)
    df['Publication_Minute'] = df['Publication_Time'].str.extract(':(\d+)').astype(float)
    median_hour = df['Publication_Hour'].median()
    median_minute = df['Publication_Minute'].median()
    df['Publication_Hour'] = df['Publication_Hour'].fillna(median_hour).fillna(0).astype(int)
    df['Publication_Minute'] = df['Publication_Minute'].fillna(median_minute).fillna(0).astype(int)

    # Create interaction features
    df['Podcast_Genre'] = df['Podcast_Name'] + '_' + df['Genre']

    # Polynomial features for numerical columns
    poly = PolynomialFeatures(degree=2, include_bias=False)
    num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
    poly_features = poly.fit_transform(df[num_cols].fillna(df[num_cols].median()))
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(num_cols), index=df.index)

    # Ensure unique column names
    poly_df.columns = [f'poly_{col}' for col in poly_df.columns]
    df = pd.concat([df, poly_df], axis=1)

    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)



# Preprocessing
categorical_features = [
    'Podcast_Name', 'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment', 'Podcast_Genre'
]

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in categorical_features:
    train_df[col] = ordinal_encoder.fit_transform(train_df[col].astype(str).values.reshape(-1, 1))
    test_df[col] = ordinal_encoder.transform(test_df[col].astype(str).values.reshape(-1, 1))

    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

train_df['Episode_Num'] = train_df['Episode_Title'].str[8:].astype('category')
test_df['Episode_Num'] = test_df['Episode_Title'].str[8:].astype('category')

train_df.drop(columns=['Episode_Title'], inplace=True)
test_df.drop(columns=['Episode_Title'], inplace=True)



# Define preprocessor
numerical_features = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Hour', 'Publication_Minute']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])

# Define model pipeline with stacking
estimators = [
    ('lgb', lgb.LGBMRegressor(random_state=42)),
    ('xgb', xgb.XGBRegressor(random_state=42)),
    ('cat', CatBoostRegressor(random_state=42, verbose=0))
]

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=lgb.LGBMRegressor(random_state=42)
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', stacking_regressor)
])



# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'regressor__lgb__n_estimators': [500, 1000],
    'regressor__lgb__learning_rate': [0.01, 0.03, 0.05],
    'regressor__lgb__num_leaves': [31, 50, 100],
    'regressor__lgb__max_depth': [-1, 10, 20],
    'regressor__lgb__subsample': [0.7, 0.8, 0.9],
    'regressor__lgb__colsample_bytree': [0.7, 0.8, 0.9],
    'regressor__xgb__n_estimators': [500, 1000],
    'regressor__xgb__learning_rate': [0.01, 0.03, 0.05],
    'regressor__xgb__max_depth': [3, 6, 9],
    'regressor__xgb__subsample': [0.7, 0.8, 0.9],
    'regressor__xgb__colsample_bytree': [0.7, 0.8, 0.9],
    'regressor__cat__iterations': [500, 1000],
    'regressor__cat__learning_rate': [0.01, 0.03, 0.05],
    'regressor__cat__depth': [6, 8, 10],
    'regressor__cat__l2_leaf_reg': [1, 3, 5]
}

random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)



# Train model
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']

random_search.fit(X, y)

# Best model
best_model = random_search.best_estimator_



# Predictions
test_preds = best_model.predict(test_df)

# Submission
submission = pd.DataFrame({
    'id': sample_submission['id'],
    'Listening_Time_minutes': test_preds
})
submission.to_csv('submission.csv', index=False)

print(submission.head())


