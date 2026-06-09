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


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error  # ✅ Added missing import
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import train_test_split


train_set = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv",index_col = 'id')
test_set = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")



train_set.info()


test_set.info()


train_set.head()


print(train_set.shape)
test_set.shape


X = train_set.drop(columns = ['Price'])
y = train_set['Price'].copy()
# Create categorical mask
categorical_mask =X.dtypes=='object'

# Extract categorical columns
categorical_columns = X.columns[categorical_mask].tolist()
print("Categorical Columns:\n",categorical_columns)
# Extract non-categorical columns
numerical_columns = X.columns[~categorical_mask].tolist()
print("Numerical Columns:\n",numerical_columns)



# Preprocessing
scaler= StandardScaler()
impute = SimpleImputer(strategy = 'most_frequent')
one_hot = OneHotEncoder(handle_unknown = 'ignore')
# Categorical pipeline: Imputting missing values and apply one-hot encoding
categorical_pipeline = Pipeline(steps=[
                                       ('imputer',impute),
                                       ('one_hot',one_hot)
                                      ]
                               )
# Numerical pipeline: Imputing missing values and scale features
numerical_pipeline = Pipeline(steps=[
                    ('imputer',impute),
                    ('scaler',scaler)
                                    ]
                             )

# Combine both pipelines into ColumnTransformer
preprocessor = ColumnTransformer(
    transformers = [
        ("num",numerical_pipeline, numerical_columns),
        ("cat",categorical_pipeline,categorical_columns)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state= 42)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)


def random_search(estimator, params):
    
        randomized_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=5,  # Increased for better tuning
            scoring='neg_mean_squared_error',
            verbose=1,
            cv=3,  # Increased cross-validation
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        )
        
        return  randomized_search
# Parameter grid
param_grid = {
    'learning_rate': np.arange(0.05, 1, 0.05),
    'max_depth': np.arange(3, 10, 1),
    'n_estimators': np.arange(50, 200, 50)
}

# Define model
reg_xgb = xgb.XGBRegressor(n_jobs=2, objective='reg:squarederror', random_state=500)

# Run Randomized Search and XGB Regressor

reg_xgb_model = random_search(reg_xgb, param_grid)
# Fit Model to training set
reg_xgb_model.fit(X_train_scaled, y_train)
best_model = reg_xgb_model.best_estimator_


y_pred = best_model.predict(X_test_scaled)
print("Voting Regressor Validation RMSE: ",np.sqrt(mean_squared_error(y_test, y_pred)))


prepropressed_test = preprocessor.fit_transform(test_set)
combined_test_preds = best_model.predict(prepropressed_test)
submission = pd.DataFrame({'id':test_set.index,
                            "Price":combined_test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully")


# Display the first few rows of the submission file
display(submission.head(10))

