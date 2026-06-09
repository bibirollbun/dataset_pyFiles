import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Load datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')



train_data.head()


test_data.head()


train_data.shape


test_data.shape


train_data.isnull().sum()


train_data.info()


f= plt.figure(figsize=(12,4))

ax=f.add_subplot(121)
sns.distplot(train_data['Height'],bins=50,color='r',ax=ax)
ax.set_title('Distribution of  Height')

ax=f.add_subplot(122)
sns.distplot(np.log10(train_data['Weight']),bins=40,color='b',ax=ax)
ax.set_title('Distribution of Weight')
ax.set_xscale('log');


f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.scatterplot(x='Age',y='Calories',data=train_data,palette='magma',hue='Sex',ax=ax)
ax.set_title('Scatter plot of Charges vs age')




# Pairplot of a subset of features
sns.pairplot(train_data[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(train_data['Duration'], bins=30, kde=True)
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()



# Distribution of Heart Rate
plt.figure(figsize=(8, 6))
sns.histplot(train_data['Heart_Rate'], bins=30, kde=True)
plt.title('Distribution of Heart Rate')
plt.xlabel('Heart_Rate')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8,7))
sns.histplot(train_data['Body_Temp'], bins=30, kde=True)
plt.title('Distribution of Body Temperature')
plt.xlabel('Body_Temp')
plt.ylabel('Frequency')
plt.show()



label_enc = LabelEncoder()
train_data['Sex'] = label_enc.fit_transform(train_data['Sex'])
test_data['Sex'] = label_enc.transform(test_data['Sex'])


train_data.head()


test_data.head()


# Define features and target
features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target = 'Calories'




# Split training data
X = train_data[features]
y = train_data[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R^2 Score:", r2_score(y_val, y_pred))



# Redefine features and target
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target = 'Calories'

# Prepare training and test data
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]



# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)



# Make predictions on the test set
predictions = model.predict(X_test)


# Create the submission file
submission_df = pd.DataFrame({'id': test_data['id'], 'Calories': predictions})

# Save the submission file
submission_df.to_csv('Mysubmission.csv', index=False)



print("Submission file created successfully!")
print(submission_df.head())


