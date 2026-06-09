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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('/kaggle/input/mentally-stability-of-the-person/train.csv')


df.head()


df.info()


df.isnull().sum()


df.shape


#depression rates by gender graph
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
bar_plot = sns.countplot(data=df, x='Gender', hue='Depression', palette='pastel')
plt.title('Depression Rates by Gender', fontsize=24, fontweight='bold')
plt.xlabel('Gender', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Depression', labels=['No', 'Yes'], frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in bar_plot.patches:
    bar_plot.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()


#correlation matrix heatmap
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(15, 10))
correlation_matrix = numeric_df.corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix Heatmap', fontsize=24, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()


#impact of family history on depression graph
family_history_data = df.groupby(['Family History of Mental Illness', 'Depression']).size().unstack()
family_history_data.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#FF9999', '#66B3FF'])

plt.title('Impact of Family History on Depression', fontsize=24, fontweight='bold')
plt.xlabel('Family History of Mental Illness', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Depression Status', labels=['No', 'Yes'], frameon=True)
plt.tight_layout()
plt.show()


#depression rates by age group
bins = [17, 24, 34, 44, 54, 64, 100]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
temp_df = df.copy()
temp_df['Age Group'] = pd.cut(temp_df['Age'], bins=bins, labels=labels)
age_depression_data = temp_df.groupby(['Age Group', 'Depression']).size().unstack()

age_depression_data.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#FF9999', '#66B3FF'])
plt.title('Depression Rates by Age Group', fontsize=24)
plt.xlabel('Age Group', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend(title='Depression Status', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


#removing columns with more than 70% missing values
df.dropna(thresh=len(df) * 0.3, axis=1, inplace=True)


#deleting the columns that we think will not be needed in the model
df.drop(['id', 'Name'], axis=1, inplace=True)


#filling the columns with sparse data using the mode
df[['Dietary Habits', 'Degree', 'Financial Stress']] = df[['Dietary Habits', 'Degree', 'Financial Stress']].fillna(df.mode().iloc[0])


#filling the missing data in the 'Work Pressure' and 'Job Satisfaction' columns using the KNN Imputer.
imputer = KNNImputer(n_neighbors=5)
filled_values = imputer.fit_transform(df[['Work Pressure', 'Job Satisfaction']])

df[['Work Pressure', 'Job Satisfaction']] = filled_values


#filling the missing 'Profession' values based on the distribution of the existing professions.
distribution = df['Profession'].value_counts(normalize=True)
missing_count = df['Profession'].isnull().sum()

if missing_count > 0:
    random_choices = np.random.choice(distribution.index, size=missing_count, p=distribution.values)
    
    df.loc[df['Profession'].isnull(), 'Profession'] = random_choices


df.isnull().sum()


#converting categorical variables to numeric and separating the target variable 'Depression'
df = pd.get_dummies(df, drop_first=True)

x = df.drop(columns=['Depression'])
y = df['Depression']


#splitting the data into training and test sets, normalizing numerical features
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(x_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)


model.summary()


predictions = model.predict(x_test_scaled)
predicted_classes = (predictions > 0.5).astype('int32')


#classification report
report = classification_report(y_test, predicted_classes, target_names=["No Depression (0)", "Depression (1)"])
print(report)


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


#confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)
cm_df = pd.DataFrame(cm, index=["True Negative (0)", "True Positive (1)"], columns=["Predicted Negative (0)", "Predicted Positive (1)"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Values')
plt.xlabel('Predicted Values')
plt.show()


#preprocessing test data
test = pd.read_csv('/kaggle/input/mentally-stability-of-the-person/test.csv')

submission = pd.DataFrame({'id': test['id']})
test = test.drop(columns=['id'])
test = test.reindex(columns=x_train.columns, fill_value=0)

#scaling
test_scaled = scaler.transform(test)


#making predictions on test data
binary_predictions = (model.predict(test_scaled) > 0.35).astype(int)


#creating the submission file
submission['Depression'] = binary_predictions
submission.to_csv('submission.csv', index=False)

