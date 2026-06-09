from google.colab import drive
drive.mount('/content/drive')


#!unzip "/content/drive/MyDrive/Forecasting Stricker Sales/playground-series-s5e1.zip" -d "/content/drive/MyDrive/Forecasting Stricker Sales#"


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from google.colab import drive
drive.mount('/content/drive')



train_data = pd.read_csv("/content/drive/MyDrive/Forecasting Stricker Sales/train.csv")
test_data=pd.read_csv('/content/drive/MyDrive/Forecasting Stricker Sales/test.csv')


train_data.isnull().sum()


train_data.shape


train_data.info()


train_data.describe()



# check value counts for nymeric columns

for label,content in train_data.items():
  if pd.api.types.is_numeric_dtype(content):
    print(f'Value Counts For Column Numeric:{label}')
    print(train_data[label].value_counts())
    print("--"*40)


# check value counts for object columns
for label,content in train_data.items():
  if pd.api.types.is_object_dtype(content):
    print(f'The Value Count For Objects:{label}')
    print(train_data[label].value_counts())
    print("--"*40)


train_data


# make a copy of our dataframe
df=train_data.copy()


avg_sales=train_data.groupby('country')['num_sold'].mean()
avg_sales


avg_sales.plot(kind='bar',
               figsize=(10,6),
               xlabel='Country',
               ylabel='Avg Sales Per Day By Country wise',
               color='purple')
plt.show()


total_sales=train_data.groupby('country')['num_sold'].sum()
total_sales


total_sales_df = total_sales.reset_index()
total_sales_df.columns = ["Country", "num_sold"]
total_sales_df


import plotly.express as px
fig1 = px.choropleth(total_sales_df,
                    locations="Country",
                    locationmode='country names',
                    color="num_sold",
                    hover_name="Country",
                    title="Sales Distribution by Country (in Units Sold)",
                    color_continuous_scale=px.colors.sequential.Plasma)

fig1.update_layout(
    coloraxis_colorbar=dict(
        title="Total Units Sold (in millions)"
    ),
    title_font_size=20
)
fig1.show()


store = train_data.groupby('store')['num_sold'].sum().reset_index()
store.columns = ["Store", "num_sold"]
store



store.plot(x='Store', y='num_sold',
           kind='barh',
           color='firebrick',
           xlabel='Store',
           ylabel='Num Sold',
           title='Sales Distribution as per Stores')
plt.show()


df_viz = train_data.groupby(['store', 'product'])['num_sold'].sum().reset_index()
df_viz.columns = ['Store', 'Product', 'num_sold']
df_viz = df_viz.sort_values('Store')

# Create a bar chart
fig = px.bar(df_viz,x='Store',y='num_sold', color='Product',title="Sales Distribution By Products And Stores")
fig.update_traces(textposition='auto',textfont_size=20)
fig.update_layout(barmode='group')
fig.show()



import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

# Feature Engineering
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Drop rows with missing target
train_data = train_data.dropna(subset=['num_sold'])
print("Train shape after dropping missing target:", train_data.shape)

# Create date-based features
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['dayofweek'] = train_data['date'].dt.dayofweek

test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['dayofweek'] = test_data['date'].dt.dayofweek

# Split Features & Target (UNSORTED for final training)
X = train_data.drop(columns=['id', 'date', 'num_sold'])
y = train_data['num_sold']

X_test = test_data.drop(columns=['id', 'date'])

# Build scikit-learn Pipeline
cat_cols = ['country', 'store', 'product']
num_cols = ['year', 'month', 'dayofweek']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
])


# Generate Timestamp

timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Time-based Cross Validation with OOF Predictions
# Sort the training data by date and reset the index
train_data_sorted = train_data.sort_values(by='date').reset_index(drop=True)

# Build X_sorted, y_sorted from the RE-INDEXED DataFrame
X_sorted = train_data_sorted.drop(columns=['id', 'date', 'num_sold'])
y_sorted = train_data_sorted['num_sold']

# Prepare OOF array
oof_predictions = np.zeros(len(train_data_sorted))

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_index, valid_index) in enumerate(tscv.split(X_sorted), 1):
    # Split
    X_train_cv, X_valid_cv = X_sorted.iloc[train_index], X_sorted.iloc[valid_index]
    y_train_cv, y_valid_cv = y_sorted.iloc[train_index], y_sorted.iloc[valid_index]

    # Fit
    model_pipeline.fit(X_train_cv, y_train_cv)
    preds = model_pipeline.predict(X_valid_cv)

    # Store OOF predictions
    oof_predictions[valid_index] = preds

    # Compute MAPE
    mape = np.mean(np.abs((y_valid_cv - preds) / y_valid_cv))
    scores.append(mape)
    print(f"Fold {fold} MAPE: {mape:.2%}")

print("TimeSeriesSplit MAPE Scores:", scores)
print("Average MAPE:", np.mean(scores))


# Save OOF Predictions


# Match OOF predictions to the correct IDs from the re-indexed DataFrame
oof_df = pd.DataFrame({
    'id': train_data_sorted['id'],
    'oof_num_sold': oof_predictions
})

oof_filename = f"oof_predictions_m01_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset & Predict on Test

model_pipeline.fit(X, y)
test_preds = model_pipeline.predict(X_test)

# Define the directory and ensure it exists
model_dir = "/content/drive/MyDrive/model"
os.makedirs(model_dir, exist_ok=True)

# Create the full file path
model_filename = f"model_01_{timestamp_str}.pkl"
model_filepath = os.path.join(model_dir, model_filename)

# Save the model
joblib.dump(model_pipeline, model_filepath)
print(f"Trained model saved as {model_filepath}")


# Submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': test_preds
})

# Define the directory and ensure it exists
submission_dir = "/content/drive/MyDrive/Submission File"
os.makedirs(submission_dir, exist_ok=True)

# Create the full file path
submission_filename = f"sub_m01_{timestamp_str}.csv"
submission_filepath = os.path.join(submission_dir, submission_filename)

# Save the submission file
submission.to_csv(submission_filepath, index=False)
print(f"Submission saved as {submission_filepath}")


#load our model
import joblib

# Load the model
model = joblib.load('/content/drive/MyDrive/model/model_01_20250111_033814.pkl')




model.get_params()


from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import datetime
import numpy as np
import os
import joblib
import pandas as pd

# MAPE scoring function
def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# MAPE
mape_scorer = make_scorer(mape, greater_is_better=False)  # Lower MAPE is better

# Timestamp for file naming
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Parameter grid for RandomForestRegressor
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [10, 20, 50, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 5],
    'regressor__max_features': ['sqrt', 'log2', None],
}

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# GridSearchCV with TimeSeriesSplit and MAPE scoring
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring=mape_scorer,  # Use MAPE as the scoring metric
    n_jobs=-1,
    verbose=3
)

# Fit the grid search to the sorted training data iam going to try with 10000 samples to save time
grid_search.fit(X_sorted[:10000], y_sorted[:10000])

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Score (MAPE): {-grid_search.best_score_:.2f}%")  # Convert to positive percentage

# Save the model to the model directory
model_dir = "/content/drive/MyDrive/model"
os.makedirs(model_dir, exist_ok=True)

model_filename = f"tuned_model_01_{timestamp_str}.pkl"
model_filepath = os.path.join(model_dir, model_filename)

joblib.dump(grid_search, model_filepath)
print(f"Trained model saved as {model_filepath}")

# Train on the full dataset using the best parameters
best_model_pipeline = grid_search.best_estimator_
best_model_pipeline.fit(X, y)

# Predict on test set
test_preds = best_model_pipeline.predict(X_test)

# Save test predictions to submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': test_preds
})

# Save submission file to submission directory
submission_dir = "/content/drive/MyDrive/Submission File"
os.makedirs(submission_dir, exist_ok=True)

submission_filename = f"tuned_sub_m01_{timestamp_str}.csv"
submission_filepath = os.path.join(submission_dir, submission_filename)

submission.to_csv(submission_filepath, index=False)
print(f"Submission file saved as {submission_filepath}")



import datetime
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import joblib
from lightgbm import LGBMRegressor

# Feature Engineering
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Drop rows with missing target
train_data = train_data.dropna(subset=['num_sold'])
print("Train shape after dropping missing target:", train_data.shape)

# Create date-based features
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['dayofweek'] = train_data['date'].dt.dayofweek

test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['dayofweek'] = test_data['date'].dt.dayofweek

# Split Features & Target (UNSORTED for final training)
X = train_data.drop(columns=['id', 'date', 'num_sold'])
y = train_data['num_sold']

X_test = test_data.drop(columns=['id', 'date'])

# Build scikit-learn Pipeline
cat_cols = ['country', 'store', 'product']
num_cols = ['year', 'month', 'dayofweek']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# LGBMRegressor
model_pipeline1 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(n_estimators=50, random_state=42))  # LGBMRegressor here
])

# Generate Timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Time-based Cross Validation with OOF Predictions
# Sort the training data by date and reset the index
train_data_sorted = train_data.sort_values(by='date').reset_index(drop=True)

# Build X_sorted, y_sorted from the RE-INDEXED DataFrame
X_sorted = train_data_sorted.drop(columns=['id', 'date', 'num_sold'])
y_sorted = train_data_sorted['num_sold']

# Prepare OOF array
oof_predictions1 = np.zeros(len(train_data_sorted))

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_index, valid_index) in enumerate(tscv.split(X_sorted), 1):
    # Split
    X_train_cv, X_valid_cv = X_sorted.iloc[train_index], X_sorted.iloc[valid_index]
    y_train_cv, y_valid_cv = y_sorted.iloc[train_index], y_sorted.iloc[valid_index]

    # Fit
    model_pipeline1.fit(X_train_cv, y_train_cv)
    preds1 = model_pipeline1.predict(X_valid_cv)

    # Store OOF predictions
    oof_predictions1[valid_index] = preds1

    # Compute MAPE
    mape = np.mean(np.abs((y_valid_cv - preds1) / y_valid_cv))
    scores.append(mape)
    print(f"Fold {fold} MAPE: {mape:.2%}")

print("TimeSeriesSplit MAPE Scores:", scores)
print("Average MAPE:", np.mean(scores))

# Save OOF Predictions
oof_df1 = pd.DataFrame({
    'id': train_data_sorted['id'],
    'oof_num_sold': oof_predictions1
})

oof_filename1 = f"oof_predictions_m01_{timestamp_str}.csv"
oof_df1.to_csv(oof_filename1, index=False)
print(f"OOF predictions saved as {oof_filename1}")

# Train on Full Dataset & Predict on Test
model_pipeline1.fit(X, y)
test_preds = model_pipeline1.predict(X_test)

# Define the directory and ensure it exists
model_dir = "/content/drive/MyDrive/model"
os.makedirs(model_dir, exist_ok=True)

# Create the full file path
model_filename1 = f"lgbmmodel_01_{timestamp_str}.pkl"
model_filepath1 = os.path.join(model_dir, model_filename)

# Save the model
joblib.dump(model_pipeline1, model_filepath1)
print(f"Trained model saved as {model_filepath1}")


# Submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': test_preds
})

# Define the directory and ensure it exists
submission_dir = "/content/drive/MyDrive/Submission File"
os.makedirs(submission_dir, exist_ok=True)

# Create the full file path
submission_filename = f"Lgbm_m01_{timestamp_str}.csv"
submission_filepath = os.path.join(submission_dir, submission_filename)

# Save the submission file
submission.to_csv(submission_filepath, index=False)
print(f"Submission saved as {submission_filepath}")




from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import datetime
import numpy as np
import os
import joblib
import pandas as pd

# MAPE scoring function
def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# MAPE
mape_scorer = make_scorer(mape, greater_is_better=False)  # Lower MAPE is better

# Timestamp for file naming
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Parameter grid for LGBMRegressor
param_grid = {'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [5, 10, 20, -1],
    'regressor__num_leaves': [31, 50, 100],
    'regressor__subsample': [0.7, 0.8, 1.0],
    'regressor__colsample_bytree': [0.7, 0.8, 1.0]
}

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# GridSearchCV with TimeSeriesSplit and MAPE scoring
grid_search1 = GridSearchCV(
    estimator=model_pipeline1,
    param_grid=param_grid,
    cv=tscv,
    scoring=mape_scorer,  # Use MAPE as the scoring metric
    n_jobs=-1,
    verbose=3
)

# Fit the grid search to the sorted training data iam going to try with 10000 samples to save time
grid_search1.fit(X_sorted[:10000], y_sorted[:10000])

# Best parameters and score
print("Best Parameters:", grid_search1.best_params_)
print(f"Best CV Score (MAPE): {-grid_search1.best_score_:.2f}%")  # Convert to positive percentage

# Save the model to the model directory
model_dir = "/content/drive/MyDrive/model"
os.makedirs(model_dir, exist_ok=True)

model_filename = f"tuned_model_02_{timestamp_str}.pkl"
model_filepath = os.path.join(model_dir, model_filename)

joblib.dump(grid_search1, model_filepath)
print(f"Trained model saved as {model_filepath}")

# Train on the full dataset using the best parameters
best_model_pipeline = grid_search1.best_estimator_
best_model_pipeline.fit(X, y)

# Predict on test set
test_preds = best_model_pipeline.predict(X_test)

# Save test predictions to submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': test_preds
})

# Save submission file to submission directory
submission_dir = "/content/drive/MyDrive/Submission File"
os.makedirs(submission_dir, exist_ok=True)

submission_filename = f"tuned_sub_m01_{timestamp_str}.csv"
submission_filepath = os.path.join(submission_dir, submission_filename)

submission.to_csv(submission_filepath, index=False)
print(f"Submission file saved as {submission_filepath}")



# Best parameters and score
print("Best Parameters:", grid_search1.best_params_)
print(f"Best CV Score (MAPE): {-grid_search1.best_score_:.2f}%")




