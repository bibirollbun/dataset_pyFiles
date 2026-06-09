import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.head()


train.info()


train['Personality'] = train['Personality'].replace({'Extrovert': 1, 'Introvert': 0})


train.head()


# Drop rows with any null values in training and test sets
train = train.dropna()
test = test.dropna()


train.info()


train.head()


train['Stage_fear'] = train['Stage_fear'].replace({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})


test['Stage_fear'] = test['Stage_fear'].replace({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


# Separate features and target
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

# Impute missing values with mean (numerical only)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


num_features = X.shape[1]
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    BatchNormalization(), Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.2),
    Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['AUC']
)


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


y_val_pred_proba = model.predict(X_val)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)

report_dict = classification_report(y_val, y_val_pred, output_dict=True)

report_df = pd.DataFrame(report_dict).transpose()

# Plot heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:2, :3], annot=True, fmt=".2f", cmap="Blues")
plt.title("Classification Report Heatmap")
plt.ylabel("Class")
plt.xlabel("Metric")
plt.show()



print("\nClassification Report on Validation Set:")
print(classification_report(y_val, y_val_pred))


y_test_pred_proba = model.predict(X_test_scaled)
y_test_pred_binary = (y_test_pred_proba > 0.5).astype(int)


submission = test.copy()
submission['Personality'] = y_test_pred_binary
submission = submission[['id', 'Personality']]
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv successfully!")


# Convert predicted labels to strings for submission
submission['Personality'] = submission['Personality'].map({1: 'extravert', 0: 'introvert'})


submission.head(20)

