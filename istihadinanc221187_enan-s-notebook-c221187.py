import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load training data
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Label encoding
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Feature-target split
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for CNN input: (samples, features, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

# One-hot encode the labels (for binary, optional)
y_cat = to_categorical(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')  # for binary classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)
print(f"CNN Validation Accuracy: {accuracy_score(y_val_classes, y_pred_classes):.2f}")

# Load and preprocess test data
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
X_test = scaler.transform(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict on test data
y_test_pred = model.predict(X_test)
solution['satisfaction'] = np.argmax(y_test_pred, axis=1)
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])

# Output preview
print(solution[['id', 'satisfaction']].head())


# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)


solution.head()


import pandas as pd


train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")


print(train_data.head())


print("Shape:", train_data.shape)


print(train_data.dtypes)



train_data.head(20)


missing_values = train_data.isnull().sum()
print("Missing values in each column: ")
print(missing_values)


# Summary statistics
print(train_data.describe())

# Categorical summary
print(train_data.describe(include='object'))


train_data['Arrival Delay in Minutes'] = train_data['Arrival Delay in Minutes'].fillna(train_data['Arrival Delay in Minutes'].median())




train_data.head(10)


missing_values = train_data.isnull().sum()
print("Missing values in each column: ")
print(missing_values)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Show first few rows
train_data.head()



# Fill missing numerical values with median
train_data['Arrival Delay in Minutes'] = train_data['Arrival Delay in Minutes'].fillna(train_data['Arrival Delay in Minutes'].median())




missing_values = train_data.isnull().sum()
print("Missing values in each column: ")
print(missing_values)



train_data['Total Delay'] = train_data['Departure Delay in Minutes'] + train_data['Arrival Delay in Minutes']



train_data.head(10)


train_data['Distance Category'] = pd.cut(
    train_data['Flight Distance'],
    bins=[0, 1000, 2000, 3000, 5000],
    labels=['Short', 'Medium', 'Long', 'Very Long']
)



train_data.head(10)


service_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]

train_data['Service Score Avg'] = train_data[service_cols].mean(axis=1)



from sklearn.preprocessing import LabelEncoder

label_encoders = {}
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le



plt.figure(figsize=(10, 6))
sns.boxplot(x='satisfaction', y='Service Score Avg', data=train_data)
plt.title('Service Score by Satisfaction Level')
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns
numeric_data = train_data.select_dtypes(include='number')

# Plot the correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()




plt.figure(figsize=(6, 6))
train_data['satisfaction'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Overall Satisfaction Distribution')
plt.ylabel('')  # Hide y-label
plt.show()



plt.figure(figsize=(6, 6))
train_data['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink'])
plt.title('Passenger Gender Distribution')
plt.ylabel('')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Display columns to identify suitable ones for bar plots
train_data.columns



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Identify categorical columns (object or category types)
categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns

# Plot bar charts for each categorical column
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train_data, x=col, order=train_data[col].value_counts().index)
    plt.title(f'Count of each category in {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("submission.csv", index=False)

