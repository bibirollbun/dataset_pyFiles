import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/introvert-extrovert-dataset/train.csv')
test = pd.read_csv('/kaggle/input/introvert-extrovert-dataset/test.csv')



print("Train shape:", train.shape)
train.head()



import warnings
warnings.filterwarnings("ignore")



train.isnull().sum()



import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train, x='Personality', palette='Set2')
plt.title('Introvert vs Extrovert Count')
plt.show()



numerical = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 'Post_frequency']

for col in numerical:
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()



for col in numerical:
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} by Personality')
    plt.show()



categorical = ['Stage_fear', 'Drained_after_socializing']

for col in categorical:
    sns.countplot(data=train, x=col, hue='Personality')
    plt.title(f'{col} vs Personality')
    plt.show()



sns.heatmap(train[numerical].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()



print(train.columns)
print(test.columns)



from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Encode categorical columns
train_encoded = train.copy()
test_encoded = test.copy()
label_encoder = LabelEncoder()

for col in train_encoded.columns:
    if train_encoded[col].dtype == 'object':
        train_encoded[col] = label_encoder.fit_transform(train_encoded[col])
        if col in test_encoded.columns:
            test_encoded[col] = label_encoder.transform(test_encoded[col])

# Features & target
X_train = train_encoded.drop(columns=['id', 'Personality'])
y_train = train_encoded['Personality']
X_test = test_encoded.drop(columns=['id'])

# Fill missing values
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Save to submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': predictions
})
submission.to_csv('/kaggle/working/submission.csv', index=False)


