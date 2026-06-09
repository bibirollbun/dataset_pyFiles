# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Set visual style
plt.style.use('seaborn')
sns.set_palette('viridis')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



train.head()


train.shape


train.info()


test.shape


test.info()


# Personality distribution with emoji
plt.figure(figsize=(8,5))
ax = sns.countplot(x='Personality', data=train)
plt.title('Introvert vs Extrovert Distribution ', fontsize=14)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)


# Add percentage labels
total = len(train)
for p in ax.patches:
    percentage = f'{100*p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width()/2
    y = p.get_height() + 200
    ax.annotate(percentage, (x, y), ha='center')
plt.show()


# Feature distributions with personality comparison
plt.figure(figsize=(15,10))


# Plot 1: Time spent alone
plt.subplot(2,2,1)
sns.boxplot(x='Personality', y='Time_spent_Alone', data=train)
plt.title('⏳ Time Spent Alone Comparison')


# Plot 2: Social events
plt.subplot(2,2,2)
sns.boxplot(x='Personality', y='Social_event_attendance', data=train)
plt.title(' Social Event Attendance')
plt.tight_layout()
plt.show()


# Plot 3: Friends circle
plt.subplot(2,2,3)
sns.boxplot(x='Personality', y='Friends_circle_size', data=train)
plt.title(' Friends Circle Size')
plt.tight_layout()
plt.show()


# Plot 4: Stage fear
plt.subplot(2,2,4)
sns.countplot(x='Stage_fear', hue='Personality', data=train)
plt.title('Stage Fear Analysis')
plt.tight_layout()
plt.show()


# Simple feature engineering
train['Social_Ratio'] = train['Social_event_attendance']/(train['Time_spent_Alone']+1)
test['Social_Ratio'] = test['Social_event_attendance']/(test['Time_spent_Alone']+1)



# First identify numerical columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()



# Only fill numerical columns with median
train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols] = test[num_cols].fillna(test[num_cols].median())


# For categorical columns, fill with mode
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('Personality')  # Exclude target variable



for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)



# Verify no missing values remain
print("Missing values in train:", train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())



le = LabelEncoder()
train['Drained_after_socializing'] = le.fit_transform(train['Drained_after_socializing'])
test['Drained_after_socializing'] = le.transform(test['Drained_after_socializing'])
train['Stage_fear'] = le.fit_transform(train['Stage_fear'])
test['Stage_fear'] = le.transform(test['Stage_fear'])



X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality'].map({'Introvert':0, 'Extrovert':1})
X_test = test.drop('id', axis=1)



scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Train SVC
model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(X_train, y_train)



y_pred = model.predict(X_val)
print(f" Accuracy: {accuracy_score(y_val, y_pred):.4f}")


# Confusion matrix with style
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_val, y_pred), 
            annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Introvert', 'Extrovert'],
            yticklabels=['Introvert', 'Extrovert'])
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Classification report
print("\n Classification Report:")
print(classification_report(y_val, y_pred))


# Make predictions
test_preds = model.predict(X_test)
test_preds = ['Introvert' if x == 0 else 'Extrovert' for x in test_preds]




# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds
})

# Show sample with style
print("\n Sample Submission:")
display(submission.head(10))

# Save
submission.to_csv('submission4444.csv', index=False)




