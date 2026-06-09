#pip install h2o pandas


# Import necessary libraries
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Initialize H2O
h2o.init()

# Load your training dataset (with prices)
train_data = pd.read_csv('/kaggle/input/bp-competition-concat/KC_BP_ConcatFile2.csv')  # Your original dataset with prices
h2o_train = h2o.H2OFrame(train_data)

# Identify predictors and response
x = h2o_train.columns
y = 'Price'  # Your target variable
x.remove(y)

# Run AutoML to train models
aml = H2OAutoML(max_models=20,
                max_runtime_secs=10000,
                seed=42,
                sort_metric='RMSE')

# Train the models
aml.train(x=x, y=y, training_frame=h2o_train)

# Get the best model
best_model = aml.leader
print("Best model:", best_model.model_id)

# Load your new dataset (without prices)
new_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')  # Your new dataset without prices
h2o_new = h2o.H2OFrame(new_data)

# Make predictions
predictions = best_model.predict(h2o_new)
print("\nPredictions for new data:")
print(predictions.head())

# Convert predictions to pandas dataframe
predictions_df = predictions.as_data_frame()

# Combine predictions with original data
result = pd.concat([new_data, predictions_df], axis=1)

# Save results to CSV
result.to_csv('predictions_output.csv', index=False)
print("\nPredictions saved to 'predictions_output.csv'")

# Optional: Print model performance on training data
performance = best_model.model_performance(h2o_train)
print("\nBest Model Performance on Training Data:")
print(f"RMSE: {performance.rmse()}")

# Save the model for future use
model_path = h2o.save_model(model=best_model, path="./best_model", force=True)
print(f"Model saved to: {model_path}")

# Shutdown H2O
h2o.cluster().shutdown()

