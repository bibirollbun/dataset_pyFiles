import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head()


test.head()


# Drop id columns for both train and test data
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


train.shape


# Check the variables for each data type
for dtype, cols in train.dtypes.groupby(train.dtypes):
    col_names = list(cols.index)
    print(f"Type: {dtype} | Total: {len(col_names)}")
    print(list(cols.index))
    print()


train.describe()


# Check null values in the train dataset
nulls = False

for col in col_names:
    if train[col].isnull().sum() != 0:
        print(f"{col}: {train[col].isnull.sum()}")
        nulls = True
        
if not nulls:
    print("There aren't any null values in the train dataset.")


train.nunique()


# Check for duplicated rows and drop them
print(f"The number of duplicates is: {train.duplicated().sum()}\n")
train.drop_duplicates()


plt.figure(figsize=(10,8))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap="PiYG", fmt=".2f", linewidths=0.5)
plt.title("Corelatii intre variabile", fontsize=14, fontweight="bold")
plt.show()


print(f"Data type for curvature is: {train['curvature'].dtype}")
print(f"Data type for speed_limit is: {train['speed_limit'].dtype}")


#colors = ["skyblue", "salmon", "lightgreen", "orange", "violet", "gold", "cyan", "magenta"]
ax = train.hist(figsize=(10,10), bins=20, color="magenta", edgecolor="black")
for axes_row in ax:
    for a in axes_row:
        a.grid(False)
plt.show()


features = train.iloc[:,:-1]
target = train.iloc[:,-1]


# Split the columns based on data type
categorical_cols = features.select_dtypes(include=['object']).columns
numeric_cols = features.select_dtypes(include=['int64', 'float64', 'bool']).columns

# Preprocessing for the two types of cols using Pipelines
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Use ColumnTransformer to apply the pipelines
preprocessor = ColumnTransformer(transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)])


# Base models
xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42)
lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=7, random_state=42, verbose=0)
cat = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=7, random_state=42, verbose=0)

# Meta_model 
meta_model = LassoCV(alphas=np.logspace(-4,2,50), cv=5, random_state=42)

# Stacking model
stacking_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stack', StackingRegressor(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('cat', cat)
        ],
        final_estimator=meta_model,
        n_jobs=-1,
        passthrough=True
    ))
])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
stacking_model.fit(X_train, y_train)

# Get the preds
stacking_preds = stacking_model.predict(X_test)


# Calculate scores
rmse = mean_squared_error(y_test, stacking_preds, squared=False)
r2 = r2_score(y_test, stacking_preds)

print("RMSE: ", rmse)
print("R2 Score: ", r2)


assert all(features.columns == test.columns)
y_preds = stacking_model.predict(test)

submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = y_preds
submission.head()


submission.to_csv('submission.csv', index=False)

