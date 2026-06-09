import zipfile
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def load_json_from_zip(zip_path, json_filename):
    with zipfile.ZipFile(zip_path,'r') as z:
        with z.open(json_filename,'r') as f:
            return json.load(f)


train_data= load_json_from_zip('/kaggle/input/whats-cooking/train.json.zip', "train.json")
test_data= load_json_from_zip('/kaggle/input/whats-cooking/test.json.zip', "test.json")


train_df= pd.DataFrame(train_data)
test_df= pd.DataFrame(test_data)

train_df['ingredients_str'] = train_df['ingredients'].apply(lambda x: ' '.join(x))
test_df['ingredients_str'] = test_df['ingredients'].apply(lambda x: ' '.join(x))


label_encoder= LabelEncoder()
y= label_encoder.fit_transform(train_df['cuisine'])


tfidf= TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features= 5000)
X= tfidf.fit_transform(train_df['ingredients_str'])
X_test = tfidf.transform(test_df['ingredients_str'])


# Split train data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Model training and evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=200),
    'Naive Bayes': MultinomialNB()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    print(f"{name}: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model


final_model = best_model  # or best_model from earlier
print(best_model)


test_predictions = final_model.predict(X_test)
test_predictions_cuisine = label_encoder.inverse_transform(test_predictions)


# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'cuisine': test_predictions_cuisine
})

submission.to_csv('submission.csv', index=False)




