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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings


warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train_data = pd.concat([train, train_extra], axis=0, ignore_index=True)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
Y_train = train_data[["Price"]]



# Encode categorical features
categorical_features = ['Brand', 'Material', 'Size', 'Style', 'Color']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(train_data[categorical_features])
categorical_data = pd.DataFrame(categorical_encoded, index=train_data.index, columns=encoder.get_feature_names_out(categorical_features))



# Encode categorical features for test data
categorical_encoded_test = encoder.fit_transform(test_data[categorical_features])
categorical_data_test = pd.DataFrame(categorical_encoded_test, columns=encoder.get_feature_names_out(categorical_features))


numerical_features = ['id','Compartments', 'Weight Capacity (kg)']
binary_features = ['Laptop Compartment', 'Waterproof']



test_data[binary_features] = test_data[binary_features].replace({'Yes': 1, 'No': 0})



train_data[binary_features] = train_data[binary_features].replace({'Yes': 1, 'No': 0})
test_data[binary_features] = test_data[binary_features].replace({'Yes': 1, 'No': 0})



X_test = pd.concat([test_data[numerical_features + binary_features], categorical_data_test], axis=1)


X_train = pd.concat([train_data[numerical_features + binary_features], categorical_data], axis=1)
X_test = pd.concat([test_data[numerical_features + binary_features], categorical_data_test], axis=1)


display(X_train)


X_train.drop('id', axis =1)


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Use IterativeImputer (more intelligent but still efficient)
imputer = IterativeImputer(max_iter=10, random_state=42)

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

print("Imputation complete using IterativeImputer!")



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Use IterativeImputer (more intelligent but still efficient)
imputer = IterativeImputer(max_iter=10, random_state=42)

X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

print("Imputation complete using IterativeImputer!")



X_train.to_csv("/kaggle/working/imputated_train_data.csv", index=False)
X_test.to_csv("/kaggle/working/imputated_test_data.csv", index=False)



import pandas as pd
X_train = pd.read_csv('/kaggle/input/train-and-test-data/imputated_train_data.csv')
X_test = pd.read_csv('/kaggle/input/train-and-test-data/imputated_test_data.csv')


missing_count_per_column = X_test.isnull().sum()
print(missing_count_per_column)



missing_count_per_column = X_train.isnull().sum()
print(missing_count_per_column)



scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Ensure your data is in NumPy array format
X_train = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
Y_train = Y_train.to_numpy() if hasattr(Y_train, 'to_numpy') else Y_train
X_test = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


import tensorflow as tf

# Initialize the MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")



from sklearn.ensemble import RandomForestRegressor
n_estimators=150
# Optimized RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=150,             # More trees improve stability
    max_depth=20,                 # Limits depth to prevent overfitting
    min_samples_split=5,          # Requires at least 5 samples to split (prevents overfitting)
    min_samples_leaf=2,           # At least 2 samples per leaf (stabilizes predictions)
    max_features='sqrt',          # Uses sqrt(features) for best splits
    bootstrap=True,               # Enables bootstrap sampling (diversity)
    n_jobs=-1,                    # Uses all CPU cores for parallel processing
    random_state=42
)




with strategy.scope():
    model = RandomForestRegressor(
            n_estimators= 80,            # Number of trees in the forest
            max_depth=None,              # Maximum depth of the tree (None allows full depth)
            min_samples_split=2,         # Minimum samples required to split an internal node
            min_samples_leaf=1,          # Minimum samples required to be a leaf node
            max_features='sqrt',         # Number of features to consider when looking for the best split
            bootstrap=True,              # Whether bootstrap samples are used when building trees
            random_state=42             
        )


import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the model
Y_pred = model.predict(X_val)
r2 = r2_score(Y_val, Y_pred)

print(f"R2 Score: {r2:.4f}")






model_path = f"random_forest_model_with{n_estimators}.pkl"
print(model_path)


# Save the trained model
model_path = f"random_forest_model_with{n_estimators}.pkl"
joblib.dump(model, model_path)


rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
print(f"RMSE loss:, {rmse:.4f}")


predictions = model.predict(X_test)
df = pd.DataFrame(predictions, columns=["Predictions"])

# Save DataFrame to an Excel file
file_path = "/kaggle/working/model_predictions.xlsx"
df.to_excel(file_path, index=False)

print(f"Excel file saved as {file_path}")


display(test_data["id"])




