import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Load dataset
df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Remove 'id' column and store test IDs
df_train.drop(columns=['id'], inplace=True)
test_ids = df_test['id']
df_test.drop(columns=['id'], inplace=True)

# Fill missing values with most frequent values
df_train.fillna(method='ffill', inplace=True)
df_test.fillna(method='ffill', inplace=True)

# Feature classification
cat_features = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
num_features = [col for col in df_train.columns if col not in cat_features + ['Price']]

# Label encoding for categorical variables
label_dict = {}
for feature in cat_features:
    if feature in df_train and feature in df_test:
        encoder = LabelEncoder()
        df_train[feature] = encoder.fit_transform(df_train[feature].astype(str))
        df_test[feature] = encoder.transform(df_test[feature].astype(str))
        label_dict[feature] = encoder

# Features and target
X_features = df_train.drop(columns=['Price'])
Y_target = df_train['Price']

# Train-validation split
X_train_set, X_val_set, Y_train_set, Y_val_set = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

# Handling missing values by replacing with mean
data_imputer = SimpleImputer(strategy='mean')
X_train_set[num_features] = data_imputer.fit_transform(X_train_set[num_features])
X_val_set[num_features] = data_imputer.transform(X_val_set[num_features])
df_test[num_features] = data_imputer.transform(df_test[num_features])

# Scaling numerical data using RobustScaler
scaler_tool = RobustScaler()
X_train_set[num_features] = scaler_tool.fit_transform(X_train_set[num_features])
X_val_set[num_features] = scaler_tool.transform(X_val_set[num_features])
df_test[num_features] = scaler_tool.transform(df_test[num_features])

# Regression models
model_variants = {
    "SVM": SVR(kernel='rbf'),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
}

# Model training and evaluation
test_results = {}
for model_title, model in model_variants.items():
    model.fit(X_train_set, Y_train_set)
    predicted_values = model.predict(X_val_set)
    validation_error = mean_absolute_error(Y_val_set, predicted_values)
    print(f"{model_title} MAE: {validation_error:.3f}")
    test_results[model_title] = model.predict(df_test)

# Combine predictions using mean ensemble
final_output = np.mean(np.array(list(test_results.values())), axis=0)

# Save final predictions
output_dataframe = pd.DataFrame({"id": test_ids, "Price": final_output})
output_dataframe.to_csv("/kaggle/working/submission_v2.csv", index=False)

print("Predictions stored in submission_v2.csv")

