pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")

# Select features and target
features = [
    'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment',
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads'
]
target = 'Listening_Time_minutes'

X = df[features]
y = df[target]

# Define columns by type
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                  'Guest_Popularity_percentage', 'Number_of_Ads']

# Define preprocessors
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)





from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter space for XGBoost
param_distributions = {
    'regressor__n_estimators': randint(50, 300),
    'regressor__max_depth': randint(3, 10),
    'regressor__learning_rate': uniform(0.01, 0.2),
    'regressor__subsample': uniform(0.6, 0.4),
    'regressor__colsample_bytree': uniform(0.6, 0.4)
}

# Create the full pipeline again (from earlier)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, verbosity=0))
])

# Setup RandomizedSearchCV
search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=30,  # Number of parameter settings sampled
    cv=3,       # 3-fold cross-validation
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the search
search.fit(X_train, y_train)

# Best model
print("Best R^2 Score from CV:", search.best_score_)
print("Best Parameters:", search.best_params_)
print("✅ Best cross-validation R² score: {:.4f}".format(search.best_score_))




# Load test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_ids = test_df['id']

# Select features used in training
X_test = test_df[features]

# Generate predictions
preds = model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created!")



submission

