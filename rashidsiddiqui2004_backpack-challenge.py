import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df.head()


x, y = df.shape
print(f"There are {x} rows and {y} fields in the dataset.")


df.describe()


print(df.isna().sum()) 


print(df.isna().sum().sum())


df['Brand'].unique()


 for column in df.columns:
    if df[column].dtype == "object":
        mode_value = df[column].mode()[0] 
        df[column] = df[column].fillna(mode_value)


print(df.isna().sum()) 


# Find the mean of the 'Weight Capacity (kg)' column
mean_weight_capacity = df['Weight Capacity (kg)'].mean() 

mean_weight_capacity = round(mean_weight_capacity, 5)

df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(mean_weight_capacity)


print(df.isna().sum()) 


df = df.drop(['id'], axis=1)


for column in df.columns:
    if df[column].dtype == "object":
        print(df[column].value_counts())


for feature in df.columns:
    if feature!= 'Price':
        plt.figure(figsize=(6,6))  
        plt.scatter(df[feature], df['Price'])
        plt.xlabel(feature)
        plt.ylabel('Price')
        plt.title(f'Scatter Plot of Price vs. {feature}')
        plt.show()


from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le 


df.head()


def plot_coorelation_matrix(): 
    corr_matrix = df.corr()
     
    plt.figure(figsize=(8,8))
     
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                linewidths=0.5, 
                annot_kws={"size": 8}) 
     
    plt.title('Correlation Matrix', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

plot_coorelation_matrix()


# for all categorical fields
 
categorical_fields = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof','Style','Color']

for field in categorical_fields:
    sns.boxplot(x=field, y='Price', data=df)
    plt.xlabel(field)
    plt.ylabel('Price')
    plt.title(f"Price Distribution by {field}") 
    plt.show()


df['Price'] = round(df['Price'],3)


print(df['Price'].min())
print(df['Price'].max())


# Find lowest and highest priced bags
lowest_price_bags = df[df['Price'] <= df['Price'].min() + 10.0]
highest_price_bags = df[df['Price'] >= df['Price'].max() - 10.0]

 
def count_features(df):
    feature_counts = {}
    for column in categorical_fields:
        feature_counts[column] = df[column].value_counts().to_dict()
    return feature_counts

# Count features in lowest and highest priced bags
lowest_price_features = count_features(lowest_price_bags)
highest_price_features = count_features(highest_price_bags)
 
for feature in categorical_fields:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(lowest_price_features[feature].keys(), lowest_price_features[feature].values())
    plt.title(f"{feature} in Lowest Price Bags")
    plt.subplot(1, 2, 2)
    plt.bar(highest_price_features[feature].keys(), highest_price_features[feature].values())
    plt.title(f"{feature} in Highest Price Bags")
    plt.show()
 
print("Lowest Price Bags:")
for feature, counts in lowest_price_features.items():
    most_common = max(counts, key=counts.get)
    print(f"Most common {feature}: {most_common}")

print("\nHighest Price Bags:")
for feature, counts in highest_price_features.items():
    most_common = max(counts, key=counts.get)
    print(f"Most common {feature}: {most_common}")


df['All_in_one'] = df['Laptop Compartment'] + df['Waterproof']
df = df.drop(['Laptop Compartment', 'Waterproof'], axis=1)


plot_coorelation_matrix()


df.sample(10)


X = df.drop(['Price'], axis=1)
y = df['Price']


from sklearn.model_selection import train_test_split


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42) 


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Create the model
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)), 
    Dropout(0.2), 
    Dense(64, activation='relu'),
    Dropout(0.2), 
    Dense(32, activation='relu'),
    Dense(1) 
])

# Compile the model
optimizer = Adam(learning_rate=0.001)  # Use Adam optimizer with a learning rate
model.compile(optimizer=optimizer, loss='mse') 

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with Early Stopping
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)



# Evaluate the model
train_loss = model.evaluate(X_train, y_train)
val_loss = model.evaluate(X_val, y_val)
test_loss = model.evaluate(X_test, y_test)

print(f"Train Loss: {train_loss}")
print(f"Validation Loss: {val_loss}")
print(f"Test Loss: {test_loss}")


predictions = model.predict(X_test)

# Calculate RMSE on test data
rmse = mean_squared_error(y_test, predictions, squared=False) 

print(f"Root Mean Squared Error (RMSE): {rmse}") 


test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


test_data.sample(5)


bag_ids = test_data['id']
test_data.drop(['id'], axis=1, inplace=True)


test_data.shape


test_data.isna().sum()


mean_weight_capacity = test_data['Weight Capacity (kg)'].mean() 

mean_weight_capacity = round(mean_weight_capacity, 5)

test_data['Weight Capacity (kg)'] = test_data['Weight Capacity (kg)'].fillna(mean_weight_capacity)

for column in test_data.columns:
    if test_data[column].dtype == "object":
        mode_value = test_data[column].mode()[0] 
        test_data[column] = test_data[column].fillna(mode_value)


test_data.isna().sum()


for column in test_data.select_dtypes(include=['object']).columns:
    le = label_encoders[column] 
    test_data[column] = le.transform(test_data[column])


test_data['All_in_one'] = test_data['Laptop Compartment'] + test_data['Waterproof']
test_data = test_data.drop(['Laptop Compartment', 'Waterproof'], axis=1)


# Make predictions
predictions = model.predict(test_data)
print(predictions)


predictions_list = list(prediction[0] for prediction in predictions)


pred = {
    "id": bag_ids,
    "Price": predictions_list
}


submission = pd.DataFrame(pred)


submission.head()


submission.to_csv("predictions.csv", index=False)

