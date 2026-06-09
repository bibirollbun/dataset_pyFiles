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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', encoding='utf-8', engine='python')
df_validation = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')



df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_validation.replace(r'^\s*$', np.nan, regex=True, inplace=True)


pd.reset_option("display.max_info_columns")
pd.reset_option("display.max_info_rows")
print(df_train.columns)
print(df_train.shape)


df.info()


df_train.info()


df_validation.info()


df_train_cleaned = df_train.dropna()
df_train_cleaned.info()


missing_counts = df_train_cleaned.isnull().sum()
print(missing_counts)


df_validation_cleaned = df_validation.dropna()
df_validation_cleaned.info()


categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numeric_cols = ['Compartments', 'Weight Capacity (kg)']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)


XX_train = df_train_cleaned.drop(columns=['id', 'Price'])
YY_train = df_train_cleaned['Price']


XX_validation = df_validation_cleaned.drop(columns=['id', 'Price'])
YY_validation = df_validation_cleaned['Price']


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=42))
])


model.fit(XX_train, YY_train)


YY_pred = model.predict(XX_validation)
rmse = np.sqrt(mean_squared_error(YY_validation, YY_pred))
print("RMSE on the test set:", rmse)


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)),
        ('catboost', CatBoostRegressor(n_estimators=300, learning_rate=0.05, depth=6, random_state=42, verbose=0))
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_model)
])

stacking_pipeline.fit(XX_train, YY_train)
YY_pred_stacking = stacking_pipeline.predict(XX_validation)
rmse_stacking = np.sqrt(mean_squared_error(YY_validation, YY_pred_stacking))
print("Stacking Model RMSE:", rmse_stacking)



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rmse_stacking = np.sqrt(mean_squared_error(YY_validation, YY_pred_stacking))
mae_stacking = mean_absolute_error(YY_validation, YY_pred_stacking)
mse_stacking = mean_squared_error(YY_validation, YY_pred_stacking)
r2_stacking = r2_score(YY_validation, YY_pred_stacking)
mape_stacking = np.mean(np.abs((YY_validation - YY_pred_stacking) / YY_validation)) * 100

# Print Results
print("Stacking Model Performance:")
print(f"RMSE  : {rmse_stacking:.4f}")
print(f"MAE   : {mae_stacking:.4f}")
print(f"MSE   : {mse_stacking:.4f}")
print(f"R²    : {r2_stacking:.4f}")
print(f"MAPE  : {mape_stacking:.2f}%")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_test.info()


# Drop 'id' from test data as it's not a feature
XX_test = df_test.drop(columns=['id'])

# Predict prices using the trained stacking pipeline
YY_pred_test = stacking_pipeline.predict(XX_test)

# Create a DataFrame to store predictions
df_test_predictions = df_test[['id']].copy()
df_test_predictions['Predicted_Price'] = YY_pred_test

# Save to CSV (optional)
df_test_predictions.to_csv("test_predictions.csv", index=False)

# Display the first few predictions
df_test_predictions.head()





from lightgbm import LGBMRegressor

model_lgbm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1))
])

model_lgbm.fit(XX_train, YY_train)
YY_pred_lgbm = model_lgbm.predict(XX_validation)
rmse_lgbm = np.sqrt(mean_squared_error(YY_validation, YY_pred_lgbm))
print("LightGBM RMSE:", rmse_lgbm)



from catboost import CatBoostRegressor

model_catboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(n_estimators=500, learning_rate=0.05, depth=7, random_state=42, verbose=100))
])

model_catboost.fit(XX_train, YY_train)
YY_pred_catboost = model_catboost.predict(XX_validation)
rmse_catboost = np.sqrt(mean_squared_error(YY_validation, YY_pred_catboost))
print("CatBoost RMSE:", rmse_catboost)



from xgboost import XGBRegressor

model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1))
])

model_xgb.fit(XX_train, YY_train)
YY_pred_xgb = model_xgb.predict(XX_validation)
rmse_xgb = np.sqrt(mean_squared_error(YY_validation, YY_pred_xgb))
print("XGBoost RMSE:", rmse_xgb)



import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numeric_cols = ['Compartments', 'Weight Capacity (kg)']

# Preprocessing for categorical data (One-Hot Encoding without Standardization for numeric columns)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine categorical transformations (no numerical transformation)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough'  # Keeps numerical columns as they are
)

# Fit the preprocessor only on features (excluding 'id' and 'Price')
df_train_features = df_train.drop(columns=['id', 'Price'])
df_validation_features = df_validation.drop(columns=['id', 'Price'])

# Apply transformations
XX_train = preprocessor.fit_transform(df_train_features)
XX_validation = preprocessor.transform(df_validation_features)

# Convert transformed data into DataFrame
XX_train = pd.DataFrame(XX_train)
XX_validation = pd.DataFrame(XX_validation)

# Extract target variable (Price)
YY_train = df_train['Price'].values
YY_validation = df_validation['Price'].values

# Define Hyperparameter Grid for each model
param_grid_xgb = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

param_grid_lgbm = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

param_grid_catboost = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

# Tune XGBoost
xgb_model = XGBRegressor(random_state=42)
xgb_search = RandomizedSearchCV(xgb_model, param_grid_xgb, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_search.fit(XX_train, YY_train)
best_xgb = xgb_search.best_estimator_

# Tune LightGBM
lgbm_model = LGBMRegressor(random_state=42)
lgbm_search = RandomizedSearchCV(lgbm_model, param_grid_lgbm, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
lgbm_search.fit(XX_train, YY_train)
best_lgbm = lgbm_search.best_estimator_

# Tune CatBoost
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
catboost_search = RandomizedSearchCV(catboost_model, param_grid_catboost, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
catboost_search.fit(XX_train, YY_train)
best_catboost = catboost_search.best_estimator_

# Print Best Parameters
print("Best XGBoost Parameters:", xgb_search.best_params_)
print("Best LightGBM Parameters:", lgbm_search.best_params_)
print("Best CatBoost Parameters:", catboost_search.best_params_)


# Define Optimized Stacking Model
stacking_model_2 = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('catboost', best_catboost)
    ],
    final_estimator=Ridge(alpha=1.0)
)

# Train Stacking Model
stacking_model_2.fit(XX_train, YY_train)

# Make Predictions
YY_pred_stacking_2 = stacking_model_2.predict(XX_validation)

# Calculate RMSE
rmse_stacking = np.sqrt(mean_squared_error(YY_validation, YY_pred_stacking_2))
mae_stacking = mean_absolute_error(YY_validation, YY_pred_stacking_2)
mse_stacking = mean_squared_error(YY_validation, YY_pred_stacking_2)
r2_stacking = r2_score(YY_validation, YY_pred_stacking_2)
mape_stacking = np.mean(np.abs((YY_validation - YY_pred_stacking_2) / YY_validation)) * 100

# Print Results
print("Stacking Model Performance:")
print(f"RMSE  : {rmse_stacking:.4f}")
print(f"MAE   : {mae_stacking:.4f}")
print(f"MSE   : {mse_stacking:.4f}")
print(f"R²    : {r2_stacking:.4f}")
print(f"MAPE  : {mape_stacking:.2f}%")








import optuna
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
encoder = LabelEncoder()

for col in categorical_cols:
    XX_train[col] = encoder.fit_transform(XX_train[col])
    XX_validation[col] = encoder.transform(XX_validation[col])

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
    }
    model = XGBRegressor(**params, random_state=42, n_jobs=-1, enable_categorical=True)
    model.fit(XX_train, YY_train)
    preds = model.predict(XX_validation)
    return np.sqrt(mean_squared_error(YY_validation, preds))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best Params:", study.best_params_)



print("Best Params:", study.best_params)


from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Encode categorical features
df_encoded = df_train_cleaned.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_train_cleaned[col])

# Apply f_regression for regression tasks
reg_selector = SelectKBest(score_func=f_regression, k='all')  # Select best features
reg_selector.fit(df_encoded[categorical_cols], df_train_cleaned['Price'])

# Print feature scores
reg_scores = reg_selector.scores_
feature_scores = pd.Series(reg_scores, index=categorical_cols).sort_values(ascending=False)
print(feature_scores)



import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']

# **New Faster Stacking Model**
stacking_model = StackingRegressor(
    estimators=[
        ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)),  # Faster
        ('extra', ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1))  # Non-boosting fast model
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1  # Parallel processing
)

# Create the final Stacking Pipeline
stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply feature transformation
    ('stacking', stacking_model)      # Apply Stacking model
])

# Fit the model
stacking_pipeline.fit(XX_train, YY_train)

# Make predictions
YY_pred_stacking = stacking_pipeline.predict(XX_validation)

# Compute RMSE
rmse_stacking = np.sqrt(mean_squared_error(YY_validation, YY_pred_stacking))
print("Optimized Stacking Model RMSE:", rmse_stacking)



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']

# Preprocessing pipeline for categorical and numerical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())  # Normalize numerical features
])

# Apply column transformers
#preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', numerical_transformer, numerical_cols),
#        ('cat', categorical_transformer, categorical_cols)
#    ])

# Step 1: Train Initial Regressor (LightGBM)
regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lgbm', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
])

# Fit the regression model
regressor.fit(XX_train, YY_train)

# Predict using the initial model
YY_pred_regressor = regressor.predict(XX_validation)

# Step 2: Convert Predictions to Categories (Classification Step)
binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
# Convert Pandas Series to NumPy array before reshaping
YY_class_train = binner.fit_transform(YY_train.values.reshape(-1, 1)).ravel()
YY_class_pred = binner.transform(YY_pred_regressor.reshape(-1, 1)).ravel()

# Step 3: Fine-Tune with Neural Network
fine_tuner = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])

# Fit the fine-tuner model using categorized targets
fine_tuner.fit(XX_train, YY_class_train)

# Predict final outputs
YY_pred_fine_tuned = fine_tuner.predict(XX_validation)

# Convert classification bins back to continuous values
YY_final_pred = binner.inverse_transform(YY_pred_fine_tuned.reshape(-1, 1)).ravel()

# Calculate RMSE
rmse_final = np.sqrt(mean_squared_error(YY_validation, YY_final_pred))
print("Fine-Tuned Model RMSE:", rmse_final)






import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']

# Preprocessing pipeline for categorical columns (uses sparse matrices)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # Use sparse matrices
])

# Preprocessing pipeline for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Fill missing numerical values with mean
])

# Apply preprocessing to both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# **New Faster Stacking Model**
stacking_model = StackingRegressor(
    estimators=[
        ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)),  # Faster
        ('catboost', CatBoostRegressor(iterations=100, learning_rate=0.1, depth=4, random_state=42, verbose=0)),  # Faster alternative to XGBoost
        ('histgb', HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=4, random_state=42))  # Fastest boosting method in sklearn
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1  # Parallel processing
)

# Create the final Stacking Pipeline
stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply feature transformation
    ('stacking', stacking_model)      # Apply Stacking model
])

# Fit the model
stacking_pipeline.fit(XX_train, YY_train)

# Make predictions
YY_pred_stacking = stacking_pipeline.predict(XX_validation)

# Compute RMSE
rmse_stacking = np.sqrt(mean_squared_error(YY_validation, YY_pred_stacking))
print("Optimized Stacking Model RMSE:", rmse_stacking)






import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']

# Convert categorical columns to 'category' dtype to avoid unnecessary OneHotEncoding
for col in categorical_cols:
    XX_train[col] = XX_train[col].astype('category')
    XX_validation[col] = XX_validation[col].astype('category')

# Preprocessing pipeline for categorical columns (uses sparse matrices)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # Use sparse matrices
])

# Preprocessing pipeline for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Fill missing numerical values with mean
])

# Apply preprocessing to both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# **New Faster Stacking Model**
stacking_model = StackingRegressor(
    estimators=[
        ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)),  # Faster
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, tree_method='hist')),  # Faster
        ('extra', ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1))  # Non-boosting fast model
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1  # Parallel processing
)

# Create the final Stacking Pipeline
stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply feature transformation
    ('stacking', stacking_model)      # Apply Stacking model
])

# Fit the model
stacking_pipeline.fit(XX_train, YY_train)

# Make predictions
YY_pred_stacking = stacking_pipeline.predict(XX_validation)

# Compute RMSE
rmse_stacking = np.sqrt(mean_squared_error(YY_validation, YY_pred_stacking))
print("Optimized Stacking Model RMSE:", rmse_stacking)






import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
correlation_matrix = df_train_cleaned[numeric_cols].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()



from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Encode categorical features
df_encoded = df_train_cleaned.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_train_cleaned[col])

# Compute Mutual Information scores
mi_scores = mutual_info_regression(df_encoded[categorical_cols], df_train_cleaned['Price'])

# Convert to pandas Series for better readability
feature_scores = pd.Series(mi_scores, index=categorical_cols).sort_values(ascending=False)
print("Feature Importance using Mutual Information:\n", feature_scores)



import matplotlib.pyplot as plt
import seaborn as sns   

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
axes = axes.flatten()
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for i, var in enumerate(categorical_cols):
    sns.boxplot(x=var, y='Price', data=df_train_cleaned, ax=axes[i])
    axes[i].set_title(f'Price by {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Price')

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Adjust layout
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

# Copy the dataset
df_train_cleaned_copy = df_train_cleaned.copy()

# Define categorical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                    'Waterproof', 'Style', 'Color']

# Convert categorical variables to 'category' type
for var in categorical_cols:
    df_train_cleaned_copy[var] = df_train_cleaned_copy[var].astype('category')

# Initialize correlation results dictionary
correlation_results = {}

for var in categorical_cols:
    num_unique = df_train_cleaned_copy[var].nunique()  # Count unique values
    
    if num_unique < 2:
        print(f"Skipping {var}: not enough unique values ({num_unique})")
        continue

    # Encode categorical variable
    df_train_cleaned_copy[var] = df_train_cleaned_copy[var].cat.codes

    # Apply Point-Biserial Correlation for Binary Categorical Variables
    if num_unique == 2:  
        correlation, _ = stats.pointbiserialr(df_train_cleaned_copy[var], df_train_cleaned_copy['Price'])
        correlation_results[var] = correlation
    else:
        # Apply Mutual Information for Multi-Class Categorical Variables
        mi_score = mutual_info_regression(df_train_cleaned_copy[[var]], df_train_cleaned_copy['Price'], discrete_features=True)
        correlation_results[var] = mi_score[0]  # Extract the MI value

# Sort results in descending order
sorted_correlations = sorted(correlation_results.items(), key=lambda x: x[1], reverse=True)



# Check if correlations exist
if not sorted_correlations:
    print("No valid categorical variables found for correlation.")
else:
    variables, correlations = zip(*sorted_correlations)

    # Plotting the correlations
    plt.figure(figsize=(20, 12))
    bars = plt.bar(variables, correlations, color=['#FF5733' if c > 0 else '#6890F0' for c in correlations])

    # Adding value annotations on bars
    for bar, corr in zip(bars, correlations):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02 if yval > 0 else yval - 0.02, 
                 f'{corr:.4f}', ha='center', va='bottom' if yval > 0 else 'top')

    # Final plot adjustments
    plt.title('Correlation with Price (Categorical Variables)')
    plt.xlabel('Categorical Variables')
    plt.ylabel('Correlation Score')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



numeric_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']
correlation_matrix = df_train_cleaned[numeric_cols].corr()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Convert categorical variables to 'category' type
for var in categorical_cols:
    df_train_cleaned_copy[var] = df_train_cleaned_copy[var].astype('category')


#  Pairwise Correlations Among Categorical Variables
categorical_encoded = df_train_cleaned_copy[categorical_cols].apply(lambda x: x.cat.codes)
categorical_corr_matrix = categorical_encoded.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(categorical_corr_matrix, annot=True, cmap='coolwarm', fmt=".5f", linewidths=0.5)
plt.title('Correlation Matrix for Categorical Features (Encoded)')
plt.show()



import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt

# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                    'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']

# Convert categorical columns to category dtype
for var in categorical_cols:
    df_train_cleaned_copy[var] = df_train_cleaned_copy[var].astype('category')

# Encode categorical variables for correlation analysis
categorical_encoded = df_train_cleaned_copy[categorical_cols].apply(lambda x: x.cat.codes)

# Compute Pearson correlation for numerical columns
numerical_corr_matrix = df_train_cleaned_copy[numerical_cols].corr()

# Compute correlations between categorical and numerical variables
correlation_results = {}

for cat_var in categorical_cols:
    num_unique = df_train_cleaned_copy[cat_var].nunique()

    # If binary categorical, use Point-Biserial Correlation
    if num_unique == 2:
        correlation, _ = stats.pointbiserialr(df_train_cleaned_copy[cat_var].cat.codes, 
                                              df_train_cleaned_copy['Price'])
    else:
        # Use Mutual Information for multi-class categorical variables
        correlation = mutual_info_regression(df_train_cleaned_copy[[cat_var]].astype(int), 
                                             df_train_cleaned_copy['Price'], 
                                             discrete_features=True)[0]

    correlation_results[cat_var] = correlation

# Convert categorical-numerical correlations to DataFrame
cat_num_corr_df = pd.DataFrame(correlation_results.items(), columns=['Variable', 'Correlation'])
cat_num_corr_df.set_index('Variable', inplace=True)

# Compute correlation between categorical variables using Cramér’s V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))

categorical_corr_matrix = categorical_encoded.corr(method=cramers_v)

# Combine everything into a single correlation heatmap
combined_corr = pd.concat([numerical_corr_matrix, cat_num_corr_df.T], axis=1)
combined_corr = pd.concat([combined_corr, categorical_corr_matrix], axis=0)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_corr, annot=True, cmap="coolwarm", fmt=".5f")
plt.title("Correlation Matrix: Numerical & Categorical Variables")
plt.show()



import pandas as pd
import scipy.stats as stats

# Function to calculate ANOVA F-statistic
def anova_f_stat(df, categorical_cols, numerical_cols):
    results = {}
    for cat in categorical_cols:
        for num in numerical_cols:
            groups = [df[num][df[cat] == cat_value] for cat_value in df[cat].unique()]
            f_stat, p_value = stats.f_oneway(*groups)  # ANOVA test
            results[(cat, num)] = p_value  # Store p-value
    return pd.DataFrame(results.items(), columns=["Variable Pair", "P-Value"])

# Run ANOVA on your dataset
anova_results = anova_f_stat(df_train_cleaned_copy, categorical_cols, numerical_cols)

# Filter significant correlations (p-value < 0.05)
anova_results[anova_results["P-Value"] < 0.05]



from sklearn.feature_selection import mutual_info_regression

df_train_cleaned_copy = df_train_cleaned.copy()
# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                    'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']

# Function to compute mutual information
def compute_mutual_info(df, categorical_cols, numerical_cols):
    results = {}
    for cat in categorical_cols:
        for num in numerical_cols:
            mi_score = mutual_info_regression(df[[cat]].astype('category').apply(lambda x: x.cat.codes),
                                              df[num])
            results[(cat, num)] = mi_score[0]  # Store MI score
    return pd.DataFrame(results.items(), columns=["Variable Pair", "MI Score"])

# Run Mutual Information analysis
mi_results = compute_mutual_info(df_train_cleaned_copy, categorical_cols, numerical_cols)

# Sort results by MI score
mi_results.sort_values(by="MI Score", ascending=False)





