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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.shape


df.info()


#target distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Target', palette='coolwarm')
plt.title('Distribution of Academic Outcomes', fontsize=16, fontweight='bold')
plt.xlabel('Outcome (Graduate, Enrolled, Dropout)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.show()


#admission grade by outcome
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Target', y='Admission grade', palette='viridis')
plt.title('Admission Grade vs. Academic Outcomes', fontsize=16, fontweight='bold')
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Admission Grade', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#curricular units approved (1st semester) vs. outcomes
sem1 = ['Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)']
outcome_group = df.groupby('Target')[sem1].mean().reset_index()

outcome_group.plot(x='Target', kind='bar', stacked=True, colormap='coolwarm', figsize=(10, 6))
plt.title('1st Semester Performance by Outcome', fontsize=16, fontweight='bold')
plt.xlabel('Academic Outcome', fontsize=12)
plt.ylabel('Average Number of Units', fontsize=12)
plt.tight_layout()
plt.show()


#scholarship status and outcomes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Target', hue='Scholarship holder', palette='Set2')
plt.title('Scholarship Holders and Academic Outcomes', fontsize=16, fontweight='bold')
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Scholarship Holder', loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#correlation heatmap
temp_df = df.copy()
temp_df['Target'] = LabelEncoder().fit_transform(temp_df['Target'])
plt.figure(figsize=(30, 20))
sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


#preprocessing the training data
train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')

x = train.drop(columns=['id', 'Target'])
y = train['Target']

#label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y, num_classes=3)


#splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#defining the input shape
input_shape = x_train_scaled.shape[1]


#building the deep learning classification model
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


#callbacks for optimization
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]

#training the model
history = model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=64, callbacks=callbacks, verbose=0)


predictions = model.predict(x_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)


#converting one-hot encoded y_test back to class labels for classification report
y_test_classes = np.argmax(y_test, axis=1)

#generating the classification report
report = classification_report(y_test_classes, predicted_classes, target_names=label_encoder.classes_)
print(report)


#confusion matrix
cm = confusion_matrix(y_test_classes, predicted_classes)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()


model.summary()


#accuracy and loss graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()


#preprocessing test data
submission = pd.DataFrame({'id': test['id']})
test = test.drop(columns=['id'])

test = test.reindex(columns=x_train.columns, fill_value=0)

#scaling
test_scaled = scaler.transform(test)


#making predictions on test data
predictions = model.predict(test_scaled)

#converting probabilities to class labels
predictions = np.argmax(predictions, axis=1)

#mapping numeric predictions back to original class names
predicted_labels = label_encoder.inverse_transform(predictions)


#creating the submission file
submission['Target'] = predicted_labels
submission.to_csv('submission.csv', index=False)

