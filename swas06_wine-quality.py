# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv("/kaggle/input/chydv-hackathon-2025/train.csv")
df_test = pd.read_csv("/kaggle/input/chydv-hackathon-2025/test.csv")


df_train.shape,df_test.shape


df_train.head(3)


from xgboost import XGBRegressor
from sklearn.metrics import cohen_kappa_score,make_scorer
# Define features and target
X = df_train.drop(columns=['quality'])
y = df_train['quality']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Preprocessing
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Define models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(objective='reg:squarederror')
}

# Function to calculate QWK
def quadratic_weighted_kappa(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int)  # Convert regression output to integer categories
    return cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')

# Loop through models
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate using QWK
    qwk_score = quadratic_weighted_kappa(y_test, y_pred)
    print(f"{name}: QWK = {qwk_score:.4f}")


from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
model = XGBRegressor(objective='reg:squarederror')

# Define QWK scoring function
def quadratic_weighted_kappa(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int)  # Convert regression output to integer categories
    return cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')

# Create custom scorer for cross-validation
qwk_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
qwk_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=qwk_scorer)

# Print results
print(f"QWK Scores: {qwk_scores}")
print(f"Mean QWK Score: {np.mean(qwk_scores):.4f}")

