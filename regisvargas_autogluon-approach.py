pip install autogluon


import pandas as pd



# Load the file

file_path = "/kaggle/input/playground-series-s5e2/train.csv"

df = pd.read_csv(file_path)



# Display basic info and first few rows

df.info(), df.head()


df['Price'] = df['Price'].astype(float)


# Filter out extreme outliers in 'Price' (using IQR method)

Q1 = df['Price'].quantile(0.25)

Q3 = df['Price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR



# Filter data within the acceptable range

df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]


from autogluon.tabular import TabularPredictor

# Define target variable
target = "Price"
save_path = "autogluon_models"  # Directory to save models

# Train AutoGluon model
predictor = TabularPredictor(label=target, problem_type="regression", eval_metric="rmse", path=save_path).fit(
    df, 
    presets="best_quality",  # Can use "medium_quality_faster_train" for speed
   # hyperparameters={
   #     "LR": {},  # Enables Linear Regression
   #     "GBM": {},  # Includes LightGBM (default)
   #     "RF": {}    # Includes Random Forest
   # },
    time_limit=36000  # Training time limit in seconds
)

# Evaluate model
performance = predictor.evaluate(df)

# Make predictions on new data
predictions = predictor.predict(df.drop(columns=[target]))


print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)


predictor.feature_importance(df)


predictor.leaderboard()


test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
#test_data.iloc[:, 1:]  # Keeps all rows, but only columns from index 1 onward



y_pred = predictor.predict(test_data)

submission = pd.DataFrame({
    'id': test_data['id'],
    'Price': y_pred
})
print(submission.head())
submission.to_csv('submission.csv', index=False)

