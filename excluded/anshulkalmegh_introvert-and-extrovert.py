import numpy as np
import pandas as pd 

df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


x = df.drop(['id', 'Personality'], axis = 1)
y = df['Personality']


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size','Post_frequency']
for col in numeric_cols:
    x[col] = x[col].fillna(x[col].median())

category_cols = ['Stage_fear', 'Drained_after_socializing']
for col in category_cols:
    x[col] = x[col].fillna(x[col].mode()[0])

x[category_cols] = x[category_cols].replace({'Yes' : 1, 'No' : 0})
y = y.replace({'Introvert' : 0, 'Extrovert' : 1})



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0)
# x_train.isna().sum()

model = RandomForestClassifier(n_estimators = 200, max_depth = 10, min_samples_split = 5, min_samples_leaf = 2, random_state = 0)
model.fit(x_train, y_train)




from sklearn.metrics import accuracy_score, confusion_matrix
y_predicted = model.predict(x_val)
print("Accuracy:",accuracy_score(y_val, y_predicted) * 100)
print("Confusion:",confusion_matrix(y_val, y_predicted))


feature_importance = pd.Series(model.feature_importances_, index = x_train.columns)
feature_importance.sort_values(ascending = False)


# Load new dataset
new_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
ids = new_data['id']
new_data = new_data.drop(columns = ['id'])
# Preprocessing: fill missing values
# Numeric columns
numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']
for col in numeric_cols:
    new_data[col] = new_data[col].fillna(new_data[col].median())

# Categorical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    new_data[col] = new_data[col].fillna(new_data[col].mode()[0])

# Encode categorical columns if you did encoding for training
new_data['Stage_fear'] = new_data['Stage_fear'].map({'Yes': 1, 'No': 0})
new_data['Drained_after_socializing'] = new_data['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# Predictions

predictions = model.predict(new_data)

# Map numeric predictions to labels
pred_labels = ['Introvert' if p == 0 else 'Extrovert' for p in predictions]

# Optional: create a DataFrame with IDs and predictions
output = pd.DataFrame({'id': ids, 'Personality': pred_labels})
print(output.head())
output.to_csv('predictions.csv', index=False)


