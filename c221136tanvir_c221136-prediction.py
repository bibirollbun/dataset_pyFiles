import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




train_df = pd.read_csv('/kaggle/input/dataset/train_dataset.csv')



# Impute missing values in 'Arrival Delay in Minutes' with median
imputer = SimpleImputer(strategy='median')
train_df['Arrival Delay in Minutes'] = imputer.fit_transform(train_df[['Arrival Delay in Minutes']])




# Label Encoding for target
label_encoder = LabelEncoder()
train_df['satisfaction'] = label_encoder.fit_transform(train_df['satisfaction'])  # satisfied = 1, neutral/dissatisfied = 0

# Encode features
categorical_cols = train_df.select_dtypes(include='object').columns
train_df[categorical_cols] = train_df[categorical_cols].apply(label_encoder.fit_transform)



X = train_df.drop('satisfaction', axis=1)
y = train_df['satisfaction']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can tune kernel, C, gamma
svm_model.fit(X_train_scaled, y_train)



y_pred = svm_model.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


