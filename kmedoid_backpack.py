import pandas as pd

# Load train and test data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Display first few rows
train_df.head()





# Check missing values
print(train_df.isnull().sum())

# Summary statistics
print(train_df.describe())



cat_columns = ['Brand', 'Material', 'Size', 'Style', 'Color']

for col in cat_columns:
    train_df[col] = train_df[col].fillna("Unknown")
    test_df[col] = test_df[col].fillna("Unknown")



binary_columns = ["Laptop Compartment", "Waterproof"]

for col in binary_columns:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])



train_df["Weight Capacity (kg)"] = train_df["Weight Capacity (kg)"].fillna(train_df["Weight Capacity (kg)"].median())
test_df["Weight Capacity (kg)"] = test_df["Weight Capacity (kg)"].fillna(test_df["Weight Capacity (kg)"].median())



print(train_df.isnull().sum())
print(test_df.isnull().sum())



train_df = pd.get_dummies(train_df, columns=['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])
test_df = pd.get_dummies(test_df, columns=['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_features = ['Weight Capacity (kg)', 'Compartments']
train_df[num_features] = scaler.fit_transform(train_df[num_features])
test_df[num_features] = scaler.transform(test_df[num_features])



X = train_df.drop(columns=['id', 'Price'])  # Features
y = train_df['Price']  # Target

X_test = test_df.drop(columns=['id'])  # Test features (no target)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate model
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Baseline RMSE: {rmse:.4f}')



# Get feature columns used in training (excluding 'id' and 'Price')
train_features = [col for col in train_df.columns if col not in ['id', 'Price']]

# Ensure test dataset has the same features
X_test = test_df[train_features]  

# Make predictions
predictions = model.predict(X_test)

# Print first 10 predictions
print(predictions[:10])



submission = pd.DataFrame({'id': test_df['id'], 'Price': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")





