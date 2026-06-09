import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1. Load the training dataset
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# 2. Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# 3. Define features and target variable
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# 4. Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 5. Feature scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 6. Reshape for CNN: (samples, features, 1)
X_cnn = np.expand_dims(X.values, axis=2)

# 7. One-hot encode the target variable
y_cat = to_categorical(y)

# 8. Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_cnn, y_cat, test_size=0.2, random_state=42, stratify=y
)

# 9. Build the CNN model
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(64, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 10. Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# 11. Validate the model
y_val_pred = model.predict(X_val)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)
y_val_true_labels = np.argmax(y_val, axis=1)
print(f"Validation Accuracy: {accuracy_score(y_val_true_labels, y_val_pred_labels):.2f}")

# 12. Load the test dataset
solution = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")

# 13. Encode categorical variables in test data
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        solution[col] = label_encoders[col].transform(solution[col])

# 14. Select features for prediction
X_test = solution.drop(columns=['Unnamed: 0', 'id'], errors='ignore')

# 15. Handle missing values in test data
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 16. Feature scaling for test data
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 17. Reshape for CNN
X_test_cnn = np.expand_dims(X_test.values, axis=2)

# 18. Make predictions
test_pred = model.predict(X_test_cnn)
solution['satisfaction'] = np.argmax(test_pred, axis=1)

# 19. Map predictions back to original labels
solution['satisfaction'] = label_encoders['satisfaction'].inverse_transform(solution['satisfaction'])




# Rename the 'id' column to 'ID' and save the predictions to Submission.csv
solution.rename(columns={'id': 'ID'}, inplace=True)
solution[['ID', 'satisfaction']].to_csv("Newsubmission.csv", index=False)


solution.head()

