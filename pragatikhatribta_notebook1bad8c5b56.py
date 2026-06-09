import pandas as pd  
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()




print("Train shape:", train.shape)
print("Test shape:", test.shape)


print("\nTrain Info:")
print(train.info())


print("\nMissing values in Train Data:")
print(train.isnull().sum())



import seaborn as sns  
import matplotlib.pyplot as plt  

 
sns.countplot(x=train['rainfall'])  
plt.title("Rainfall Distribution (Target Variable)")  
plt.show()



train.describe()



import seaborn as sns
import matplotlib.pyplot as plt


corr = train.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



from sklearn.model_selection import train_test_split

# Select features (drop target column)
X = train.drop(columns=['rainfall'])  # Features (independent variables)
y = train['rainfall']  # Target (dependent variable)

# Split into training (80%) and validation (20%) sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Check shapes
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_valid.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create and train the model
model = LogisticRegression(max_iter=1000)  # Increased iterations
model.fit(X_train, y_train)


# Make predictions on validation set
y_pred = model.predict(X_valid)

# Check accuracy
accuracy = accuracy_score(y_valid, y_pred)
print("Validation Accuracy:", accuracy)



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased iterations
model.fit(X_train_scaled, y_train)

# Validate the model
y_valid_pred = model.predict(X_valid_scaled)
accuracy = accuracy_score(y_valid, y_valid_pred)

print("Validation Accuracy:", accuracy)



print("Missing values in test data:\n", test.isnull().sum())



test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())



test_scaled = scaler.transform(test)



test_predictions = model.predict(test_scaled)



import pandas as pd

# Create a submission DataFrame
submission = pd.DataFrame({'Id': test.index, 'rainfall': test_predictions})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("✅ Submission file 'submission.csv' created successfully!")



submission.to_csv('/kaggle/working/submission.csv', index=False)





