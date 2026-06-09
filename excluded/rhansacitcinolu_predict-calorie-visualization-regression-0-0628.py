import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Veri çerçevelerini birleştirin
df = pd.concat([train_df, test_df], ignore_index=True)


# first 5 rows
df.head()


# last 2 rows
df.tail(2)


# check the number of rows and columns
df.shape


# Check general information about data
df.info()


# statistics 
df.describe() 


# check empty data
df.isnull().sum()


plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20)
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Age', y='Weight', data=df)
plt.title('Weight by Age Group')
plt.xticks(rotation=45)
plt.show()


# 3. Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='Duration', y='Heart_Rate', data=df)
plt.title('Heart Rate over Duration')
plt.show()



# 4. Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Calories', data=df)
plt.title('Average Calories Burned by Gender')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Height', y='Weight', data=df)
plt.title('Height vs Weight')
plt.show()


plt.figure(figsize=(10, 6))
df['Sex'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()


# Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '61+'])


# Body Mass Index (BMI)
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)


# Exercise Intensity
df['Intensity'] = df['Heart_Rate'] / df['Duration']


# Interaction between important features
df['Height_Weight_Interaction'] = df['Height'] * df['Weight']
df['Age_Duration_Interaction'] = df['Age'] * df['Duration']


plt.figure(figsize=(10, 6))
sns.boxplot(x='Age_Group', y='Weight', data=df)
plt.title('Weight by Age Group')
plt.show()


# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True, linewidths=.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x='Age_Group', y='Weight', data=df)
plt.title('Weight Distribution by Age Group')
plt.show()


plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Duration', y='Calories', hue='Age_Group')
plt.title('Calories Burned by Age Group over Duration')
plt.show()


# Create a box plot for Body Temperature by Age Group
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age_Group', y='Body_Temp', data=df)
plt.title('Body Temperature by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Body Temperature (°C)')
plt.xticks(rotation=45)
plt.show()


from sklearn.preprocessing import StandardScaler


# Perform normalization on continuous variables
df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Intensity', 'BMI', 'Height_Weight_Interaction', 'Age_Duration_Interaction']] = StandardScaler().fit_transform(
    df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Intensity', 'BMI', 'Height_Weight_Interaction', 'Age_Duration_Interaction']]
)


# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Age_Group', 'Sex'], drop_first=True)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score 


# Separate features and target variable
x = df.drop(['id', 'Calories'], axis=1)  # Replace 'target_variable' with the actual target variable name
y = df['Calories']  # Only for training data

# Split into training and test sets
x_train = x[:len(train_df)]  # Training data
y_train = train_df['Calories']  # Training target variable
x_test = x[len(train_df):]  # Test data

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model with training data
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Add predicted results to the test data
test_df['Predicted_Calories'] = y_pred

# Print results
print(test_df[['id', 'Predicted_Calories']])


from sklearn.preprocessing import LabelEncoder


# Convert categorical data to numerical data
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])

# Separate features and target variable
X = train_df.drop(['id', 'Calories'], axis=1)  # Remove 'id' and 'Calories' columns
y = train_df['Calories']  # Target variable

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model with training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the performance on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

# Print results
print(f'Mean Squared Error for Validation Set: {mse_val}')
print(f'R² Score for Validation Set: {r2_val}')




