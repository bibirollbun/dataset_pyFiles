import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


df=pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')


df.head()


df.shape


df.info()


#smoking status distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='smoking', palette='coolwarm')
plt.title('Distribution of Smoking Status', fontsize=16)
plt.xlabel('Smoking Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Non-Smoker', 'Smoker'])
plt.show()


#age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, bins=30, color='blue', alpha=0.7)
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


#height vs. weight colored by smoking status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='height(cm)', y='weight(kg)', hue='smoking', palette='coolwarm', alpha=0.7)
plt.title('Height vs. Weight Colored by Smoking Status', fontsize=16)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Weight (kg)', fontsize=12)
plt.legend(title='Smoking Status')
plt.show()


#average blood pressure by smoking status
bp_means = df.groupby('smoking')[['systolic', 'relaxation']].mean().reset_index()
bp_means.plot(x='smoking', kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Average Blood Pressure by Smoking Status', fontsize=16)
plt.xlabel('Smoking Status', fontsize=12)
plt.ylabel('Blood Pressure (mmHg)', fontsize=12)
plt.legend(['Systolic', 'Relaxation'], title='Metric')
plt.show()


#correlation heatmap
plt.figure(figsize=(20, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()


#average health parameters by smoking status
health_means = df.groupby('smoking')[['fasting blood sugar', 'HDL', 'LDL']].mean().reset_index()
health_means.plot(x='smoking', kind='bar', figsize=(10, 6), colormap='coolwarm')
plt.title('Average Health Parameters by Smoking Status', fontsize=16)
plt.xlabel('Smoking Status', fontsize=12)
plt.ylabel('Average Values', fontsize=12)
plt.legend(['Fasting Blood Sugar', 'HDL', 'LDL'], title='Metric')
plt.show()


#adding new features
df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
df['WHR'] = df['waist(cm)'] / df['height(cm)']
df['Metabolic_Health'] = (df['fasting blood sugar'] + df['triglyceride'] + df['Cholesterol'] - df['HDL'] + df['LDL']) / 4
df['Hemoglobin_Height_Ratio'] = df['hemoglobin'] / df['height(cm)']


x = df.drop(columns=['smoking', 'id'])
y = df['smoking']

#normalizing feature data for the model
scaler = StandardScaler()
x = scaler.fit_transform(x)

#splitting dataset into training, validation, and testing sets
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


#applying SMOTE for oversampling
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

#creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(x_resampled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


#training the model
history = model.fit(x_resampled, y_resampled,validation_data=(x_val, y_val),epochs=100,
    batch_size=16, callbacks=[early_stopping],verbose=0)


model.summary()


#predicting probabilities and evaluating performance
y_pred_prob = model.predict(x_test)
y_pred_class = (y_pred_prob > 0.5).astype(int)


#classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_class))


#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Smoker', 'Smoker'], yticklabels=['Non-Smoker', 'Smoker'])
plt.title('Confusion Matrix After Oversampling', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()


#ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve for Predicted Probabilities After Oversampling', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right')
plt.show()


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

