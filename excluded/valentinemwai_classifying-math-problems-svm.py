import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


sub=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv")
train=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")


train.head()


print(train.isnull().sum())
print(test.isnull().sum())


plt.figure(figsize=(12,6))
sns.countplot(x='label', data=train)
plt.xticks(rotation=45)
plt.title("Topics Distribution")
plt.show()


#Clearing the text
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#", "", text)        # Remove hashtags symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Keep only letters
    return text.lower()

train['Question'] = train['Question'].apply(clean_text)


#Key words from the Questions Column
from wordcloud import WordCloud
all_text = " ".join(train['Question'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Keyword Cloud")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split
X_train, X_test, y_train, y_test = train_test_split(
    train['Question'], train['label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)




from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Evaluate model using classification report (for classification tasks)
print(classification_report(y_test, y_pred))


import xgboost as xgb
model = xgb.XGBClassifier()

#Training the model on the training data
model.fit(X_train, y_train)

#Making predictions on the test set
predictions = model.predict(X_test)

#Calculating accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))


from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


import lightgbm as lgb
lgbm = lgb.LGBMClassifier(objective='multiclass',learning_rate= 0.05560401538609363, 
                          num_leaves=25,  max_depth= 15, min_child_samples= 13, 
                  subsample= 0.8356622223967285, colsample_bytree= 0.9465492777484568, 
                  lambda_l1=0.009726408203084654, lambda_l2= 3.854959178372731e-08, 
                  n_estimators= 418)

# 4. Train the model
lgbm.fit(X_train, y_train)

# 5. Predict
y_pred = lgbm.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


test.info()


#Clearing the text
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#", "", text)        # Remove hashtags symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Keep only letters
    return text.lower()

test['Question'] = test['Question'].apply(clean_text)


test_new = test[['Question']]
test_new


# 4. Make Predictions on the Test Set
test_new = vectorizer.transform(test['Question'])
test_new


test_preds = svm.predict(test_new)
test_preds


# Submission
submission = pd.DataFrame({"id": test["id"], "label": test_preds})
submission.to_csv("submission.csv", index=False)


submission.head()

