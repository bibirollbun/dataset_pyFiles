import pandas as pd
import os

# Load a few example tables
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
notebooks = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

# Display some rows from each
print("Users:")
display(users.head())

print("Notebooks:")
display(notebooks.head())

print("Competitions:")
display(competitions.head())



users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
notebooks = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')



user_activity = notebooks['AuthorUserId'].value_counts()
active_users = user_activity[user_activity > 20].index
users['IsActive'] = users['Id'].isin(active_users)



merged = notebooks.merge(users, left_on='AuthorUserId', right_on='Id', suffixes=('_notebook', '_user'))
model_data = merged[['UserName', 'Country', 'PerformanceTier', 'IsActive']].dropna()



from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Encode categorical
le_country = LabelEncoder()
model_data['CountryEncoded'] = le_country.fit_transform(model_data['Country'])

# Split
X = model_data[['CountryEncoded', 'PerformanceTier']]
y = model_data['IsActive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()



import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Top Influential Features")
plt.show()



import pandas as pd

# Example: A dummy prediction output (adjust according to your project)
submission = pd.DataFrame({
    'Id': [1, 2, 3],  # Replace with actual Ids if needed
    'Prediction': [0, 1, 1]  # Replace with your model's predictions
})

# Save as CSV file (required for submission)
submission.to_csv('submission.csv', index=False)


