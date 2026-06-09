import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


df=pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')


df.head()


df.info()


df.isnull().sum()


df.shape


#target distribution
plt.figure(figsize=(20, 8))
sns.countplot(data=df, x='NObeyesdad', palette='coolwarm')
plt.title('Distribution of Target (Obesity level)', fontsize=16, fontweight='bold')
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.show()


#gender vs obesity categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='NObeyesdad', hue='Gender', palette='coolwarm')
plt.title('Obesity Categories by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Obesity Categories')
plt.ylabel('Count')
plt.legend(title='Gender', loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#age distribution across obesity categories
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='NObeyesdad', y='Age', palette='Set3')
plt.title('Age Distribution by Obesity Category', fontsize=16, fontweight='bold')
plt.xlabel('Obesity Categories')
plt.ylabel('Age')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#weight vs height
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Height', y='Weight', hue='NObeyesdad', palette='tab10', s=100)
plt.title('Weight vs Height by Obesity Categories', fontsize=16, fontweight='bold')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(title='Obesity Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#family history vs obesity categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='NObeyesdad', hue='family_history_with_overweight', palette='husl')
plt.title('Impact of Family History on Obesity Categories', fontsize=16, fontweight='bold')
plt.xlabel('Obesity Categories')
plt.ylabel('Count')
plt.legend(title='Family History', loc='upper right')
plt.tight_layout()
plt.show()


#correlation matrix
numeric_cols = df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_cols.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Spectral", 
            vmin=-1, vmax=1, linewidths=0.5, linecolor="black", 
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
plt.title("Numeric Feature Correlation Heatmap", fontsize=18, fontweight="bold", color="darkblue")
plt.xticks(fontsize=12, rotation=45, ha="right", color="darkgreen")
plt.yticks(fontsize=12, rotation=0, color="darkgreen")
plt.tight_layout()
plt.show()


#preprocessing the training data
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')

x = train.drop(columns=['id', 'NObeyesdad'])
y = train['NObeyesdad']

#one-hot encoding categorical features in training data
x = pd.get_dummies(x, drop_first=True)


#label encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=7)

#splitting the dataset into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x, y_categorical, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#building the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

#compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#training the model
history = model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=64)


model.summary()


#accuracy and loss Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
axes[0].set_title('Model Accuracy', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Training Loss', color='blue')
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
axes[1].set_title('Model Loss', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


#evaluating the model
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}")


#converting one-hot encoded y_test back to class labels
y_test_classes = np.argmax(y_test, axis=1)
predicted_classes = np.argmax(model.predict(x_test_scaled), axis=1)
#classification report
report = classification_report(y_test_classes, predicted_classes, target_names=label_encoder.classes_)
print(report)


#confusion matrix
cm = confusion_matrix(y_test_classes, predicted_classes)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5, linecolor='black', cbar=True, annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()


#preprocessing test data
submission = pd.DataFrame({'id': test['id']})  # Keep the 'id' column for submission
test = test.drop(columns=['id'])

#one-hot encoding the test data and align columns
test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=x_train.columns, fill_value=0)

#scaling
test_scaled = scaler.transform(test)


#making predictions on test data
predictions = model.predict(test_scaled)

#converting probabilities to class labels
predictions = np.argmax(predictions, axis=1)

#mapping numeric predictions back to original class names
predicted_labels = label_encoder.inverse_transform(predictions)


#create submission file
submission['NObeyesdad'] = predicted_labels
submission.to_csv('submission.csv', index=False)

