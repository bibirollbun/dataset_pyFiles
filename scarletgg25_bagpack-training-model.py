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

# Load datasets
df2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df1 = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")

# Adjust the ID in df2
df2["id"] += df1["id"].max()

# Concatenate both datasets
df = pd.concat([df1, df2], ignore_index=True)




# Check for null values
print(df.isnull().sum())
print(f"Total rows: {df.shape[0]}")



for col in ["Brand", "Material", "Size", "Style", "Waterproof", "Laptop Compartment"]:
    df[col].fillna(df[col].mode()[0], inplace=True)
df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].median(), inplace=True)



print(df.isnull().sum())


df.drop(columns=["Color"], inplace=True)



print(df.isnull().sum())


from sklearn.preprocessing import LabelEncoder

# List of categorical columns
cat_cols = ["Brand", "Material", "Size", "Style", "Waterproof", "Laptop Compartment"]

# Apply Label Encoding
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])



df_sampled = df.sample(frac=0.1, random_state=42)  # Take 10% of data (400,000 rows)



# Select features (excluding 'Price' because it's the target)
X = df_sampled.drop(columns=["Price", "id"])  # Remove 'id' if not useful
y = df_sampled["Price"]  # Target variable



from sklearn.model_selection import train_test_split

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100,n_jobs=-1,max_depth=8, random_state=42)
model.fit(X_train, y_train)



from sklearn.metrics import mean_squared_error
import numpy as np
y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)



import pandas as pd

# Assuming X_test has an 'id' column, else use range(len(y_pred))
submission = pd.DataFrame({
    "id": X_test.index,  # Replace with actual ID column if available
    "Predicted_Price": y_pred
})

# Save the CSV file
submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved successfully!")


