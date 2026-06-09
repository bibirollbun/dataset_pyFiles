import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

train_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv')
test_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv')
sample_solution_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv')


X_train = train_df.drop(columns=["id", "price"])
y_train = train_df["price"].copy()  # target ('price')

full_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ["days_left", "duration"]),
    ('one_hot', OneHotEncoder(handle_unknown="ignore"), ["class", "stops", "airline", "flight", "source_city", "destination_city"]),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ["arrival_time", "departure_time"])
])
X_prepared = full_pipeline.fit_transform(X_train)

ML_model = RandomForestRegressor()
ML_model.fit(X_prepared, y_train)


# Prepare test dataset
X_final_test = test_df.drop(columns=['id'])  # No 'price' column in test set
X_final_test_prepared = full_pipeline.transform(X_final_test)

# Make predictions
y_final_predicted = ML_model.predict(X_final_test_prepared)

y_final_predicted = np.maximum(y_final_predicted, 0)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],  # Keep the original IDs
    'price': y_final_predicted  # Predicted values
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv!")




