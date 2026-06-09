import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


train_csv = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_csv.head()


train_csv.isnull().sum()


test_csv.isnull().sum()


train_csv.drop(columns=['id'],inplace=True)


train_csv.duplicated().value_counts()


train_csv = train_csv.drop_duplicates()


num_cols =  train_csv.select_dtypes(include="number").columns.tolist()
cat_cols = train_csv.select_dtypes(exclude="number").columns.tolist()
num_cols.remove("accident_risk")

print(f"categorical columns : {cat_cols}")
print(f"numerical columns : {num_cols}")


train_csv.head()


for cat in cat_cols:
    print(f"Unique of {cat} is : ",train_csv[cat].unique())


train_csv = pd.get_dummies(train_csv, columns=cat_cols, prefix=cat_cols)

# Identify the new dummy columns created
# They start with the original column name and an underscore
dummy_cols = [col for col in train_csv.columns if any(col.startswith(c + '_') for c in cat_cols)]

# Convert the new dummy columns to integer type (int)
train_csv[dummy_cols] = train_csv[dummy_cols].astype(int)



train_csv.head()


X = train_csv.drop(columns=['accident_risk'])
y = train_csv['accident_risk']


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_transformed = ss.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed,y,test_size = 0.2, random_state=42)


X_train.shape


from keras import models, layers


model = models.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),  
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1) 
])


model.compile(optimizer="adam", loss="mse", metrics=["mae"])


model.fit(X_train, y_train, batch_size=64, epochs=10)


preds = model.predict(X_test)


model.evaluate(X_test, y_test)


test_arr = y_test.to_numpy()


filename = 'finalized_keras_model.keras'
print(f"\n4. Saving the model to: {filename}")
model.save(filename)
print("Model saved successfully!")


import tensorflow as tf


filename = 'finalized_keras_model.keras'

# 1. Load the saved model
loaded_model = tf.keras.models.load_model(filename)
print(f"Model successfully loaded from {filename}")


test_csv.head()


test_csv.drop(columns=['id'],inplace=True)


num_cols =  test_csv.select_dtypes(include="number").columns.tolist()
cat_cols = test_csv.select_dtypes(exclude="number").columns.tolist()

print(f"categorical columns : {cat_cols}")
print(f"numerical columns : {num_cols}")


test_csv = pd.get_dummies(test_csv, columns=cat_cols, prefix=cat_cols)

# Identify the new dummy columns created
# They start with the original column name and an underscore
dummy_cols = [col for col in test_csv.columns if any(col.startswith(c + '_') for c in cat_cols)]

# Convert the new dummy columns to integer type (int)
test_csv[dummy_cols] = test_csv[dummy_cols].astype(int)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
test_transformed = ss.fit_transform(test_csv)


predictions = loaded_model.predict(test_transformed)


test_csv = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


id_column = test_csv['id']


flat_predictions = predictions.flatten()

# Create the new DataFrame
submission_df = pd.DataFrame({
    'id': test_csv['id'],
    # You can rename this to match your target variable, e.g., 'accident_risk'
    'prediction': flat_predictions 
})


output_filename = 'submission.csv'

# Save the DataFrame to CSV. index=False is crucial as you don't want 
# the DataFrame index (0, 1, 2, ...) included in the file.
submission_df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created and saved the submission file: {output_filename}")








