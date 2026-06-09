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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Label Encoding
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Feature and target
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
y_cat = to_categorical(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define better DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 2 classes: satisfied, not satisfied
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)
print(f"Validation Accuracy: {accuracy_score(y_val_classes, y_pred_classes):.4f}")


# Load test set
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# Apply label encoders
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# Prepare test features
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
X_test = scaler.transform(X_test)

# Predict
y_test_pred = model.predict(X_test)
solution['satisfaction'] = np.argmax(y_test_pred, axis=1)
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])

# Show output
print(solution[['id', 'satisfaction']].head())




