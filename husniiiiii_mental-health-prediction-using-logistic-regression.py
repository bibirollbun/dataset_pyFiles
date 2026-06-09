# Step 1: Import Pandas and Load Dataset
import pandas as pd

df_train = pd.read_csv("/kaggle/input/depressed-people/train.csv")


# Step 2: Encode Categorical Features
from sklearn.preprocessing import LabelEncoder

encoding_columns = ['Gender', 'Sleep Duration', 'Dietary Habits',
                    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
                    'Depression']

for column in encoding_columns:
    label_encoder = LabelEncoder()
    df_train[column] = label_encoder.fit_transform(df_train[column])
    print("Column: ", column)
    print("Classes: ", label_encoder.classes_)

    mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    print("Mapping (original → encoded):", mapping)


# Step 3: Remove Index Column - No Need For Training
df_train.drop('index', axis=1, inplace=True)

df_train.head()


# Step 4: Feature Engineering For Training
features = df_train.drop('Depression', axis=1)
targets = df_train['Depression']


# Step 5: Train Test Split For Training
from sklearn.model_selection import train_test_split

train_features,test_features, train_targets, test_targets = train_test_split(features,
                                                                             targets, 
                                                                             test_size=0.2, 
                                                                             random_state=42)


# Step 6: Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(train_features, train_targets)


# Step 7: Evaluate Performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

predictions = model.predict(test_features)
accuracy = accuracy_score(test_targets, predictions) * 100
precision = precision_score(test_targets, predictions) * 100
recall =  recall_score(test_targets, predictions) * 100
f1_score = f1_score(test_targets, predictions) * 100

print("Model Performance:")
print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1 Score : {f1_score:.2f}%")


# Step 8: Hyper Parameter Tuning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

lr = LogisticRegression(max_iter=1000)

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(train_features, train_targets)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

best_lr = grid_search.best_estimator_
predictions = best_lr.predict(test_features)
print("\nClassification Report:\n", classification_report(test_targets, predictions))


# Step 9: Load test.csv File
df_test = pd.read_csv("/kaggle/input/depressed-people/test.csv")


# Step 10: Encode Categorical Features
from sklearn.preprocessing import LabelEncoder

encoding_columns = ['Gender', 'Sleep Duration', 'Dietary Habits',
                    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

for column in encoding_columns:
    label_encoder = LabelEncoder()
    df_test[column] = label_encoder.fit_transform(df_test[column])
    print("Column: ", column)
    print("Classes: ", label_encoder.classes_)

    mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    print("Mapping (original → encoded):", mapping)


# Step 11: Remove Index Column Save It Seperately For Submission File Use
indexes = df_test['index']
df_test.drop('index', axis=1, inplace=True)
df_test.head()


# Step 12: Use test.csv Data As Features For The Model To Predict
final_predictions = best_lr.predict(df_test)


# Step 13: Convert The Encoded Predicted Value As YES , NO Lables As Expected In Submission Format
reverse_encoded_labels_to_orginal = ['Yes' if pred == 1 else 'No' for pred in final_predictions]


# Step 14: Create A DataFrame Containing Only Index And Depression Column,
#          Add The Indexes Saved Earlier And Orginal Predicted Value To DataFrame,
submission = pd.DataFrame({
    'index': indexes,
    'Depression': reverse_encoded_labels_to_orginal
    
})


# Step 15: Create A CSV File For Submission
submission.to_csv("submission.csv", index=False)

