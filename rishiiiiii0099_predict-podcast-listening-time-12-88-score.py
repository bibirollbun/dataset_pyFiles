import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



train_df= pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train_df


test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_df


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission 


import pandas as pd

for label, column in train_df.items():
    if pd.api.types.is_numeric_dtype(column):  # Correct function
        print(label)
        print(f"Here is the Numerical Column Data:{train_df[label].value_counts()}")
        print("--"*40)



import pandas as pd

for label, column in train_df.items():
    if pd.api.types.is_object_dtype(column):  # Correct function
        print(label)
        print(f"Here is the Object Column Data:{train_df[label].value_counts()}")
        print("--"*40)



train_df.drop("id",axis=1,inplace=True)


train_df.head()


train_df.isnull().sum()


train_df.info()


train_df.describe()



num_columns = train_df.select_dtypes(include=['number'])
num_columns


correc_matrix=num_columns.corr()



# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correc_matrix, annot=True, cmap='mako', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize relationships
sns.pairplot(num_columns)
plt.suptitle('Pair Plot of Numeric Columns', y=1.02)
plt.show()



!pip install --upgrade scikit-learn 
!pip install --upgrade xgboost


for label,col in train_df.items():
    if pd.api.types.is_numeric_dtype(col):
        print(label)


for label,col in train_df.items():
    if pd.api.types.is_object_dtype(col):
        print(label)


train_df2= pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import xgboost as xgb  # Ensure XGBoost is installed with GPU support

# Define Categorical and Numerical Columns
cat_cols = ["Podcast_Name","Episode_Title","Genre","Publication_Day","Publication_Time","Episode_Sentiment"]
num_cols = ["Episode_Length_minutes","Host_Popularity_percentage","Guest_Popularity_percentage","Number_of_Ads"]


# Define Transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

processor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Define XGBRegressor wrapper for compatibility
class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.regressor = xgb.XGBRegressor(tree_method="gpu_hist", **kwargs)  # Use GPU acceleration

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    def __sklearn_is_fitted__(self):  # Add this method for compatibility
        return hasattr(self.regressor, "best_iteration")


# Data Preparation
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']
X_test = test_df.copy()

# Make copies of datasets
train_df1 = train_df.copy()
test_df1 = test_df.copy()

if "id" in test_df.columns:
    test_df.drop("id", axis=1, inplace=True)

# Define Model Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", processor),
    ("regressor", XGBRegressorWrapper(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42))
])

# K-Fold Cross-Validation and Training
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
oof_predictions = np.zeros(len(X))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    model_pipeline.fit(X_train_cv, y_train_cv)
    preds = model_pipeline.predict(X_valid_cv)
    oof_predictions[valid_index] = preds
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)
    print(f"Fold {fold} - RMSE: {rmse:.4f}")

print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
oof_df = pd.DataFrame({'id': train_df2['id'], 'oof_predictions': oof_predictions})
oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")


# Train on Full Dataset and Predict on Test Data
model_pipeline.fit(X, y)
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(test_df)
test_preds = model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'id': test_df1['id'], 'Listening_Time_minutes': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



# viewing submission file 

pd.read_csv("/kaggle/working/submission_20250401_140511.csv")


train_df["Podcast_Name"].value_counts()


test_df2=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_df2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import xgboost as xgb  

# Define Categorical and Numerical Columns
cat_cols = ["Podcast_Name","Episode_Title","Genre","Publication_Day","Publication_Time","Episode_Sentiment"]
num_cols = ["Episode_Length_minutes","Host_Popularity_percentage","Guest_Popularity_percentage","Number_of_Ads"]

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

processor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# XGBRegressor wrapper for compatibility
class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.regressor = xgb.XGBRegressor(**kwargs)  

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    def __sklearn_is_fitted__(self):  
        return hasattr(self.regressor, "best_iteration")

# Data Preparation
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']
X_test = test_df.copy()

# Make copies
train_df1 = train_df.copy()
test_df1 = test_df.copy()

if "id" in test_df.columns:
    test_df.drop("id", axis=1, inplace=True)

#  Model Pipeline with Parameters
# after tuning with optuna 50 iteration
best_params = {
    'n_estimators': 700,
    'learning_rate': 0.04977701442672517,
    'max_depth': 10,
    'subsample': 0.9680686178465127,
    'colsample_bytree': 0.7498305024755628,
    'gamma': 1.2302022583556316,
    'reg_alpha': 0.000633131342156743,
    'reg_lambda': 0.12530534784145997,
    'random_state': 42,
    'device': 'cuda'
}

final_model_pipeline = Pipeline(steps=[
    ("preprocessor", processor),
    ("regressor", XGBRegressorWrapper(**best_params))
])

# K-Fold Cross-Validation and Training
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
oof_predictions = np.zeros(len(X))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    final_model_pipeline.fit(X_train_cv, y_train_cv)
    preds = final_model_pipeline.predict(X_valid_cv)
    oof_predictions[valid_index] = preds
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)
    print(f"Fold {fold} - RMSE: {rmse:.4f}")

print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
oof_df = pd.DataFrame({'id': train_df1['id'], 'oof_predictions': oof_predictions})
oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset and Predict on Test Data
final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(test_df)
test_preds = final_model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"final_model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Final trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'id': test_df2['id'], 'Listening_Time_minutes': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")




