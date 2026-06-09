import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


module_path = "/kaggle/input/daigt-drcat-final/train_v3_drcat_01.csv"
data = pd.read_csv(module_path, sep=',', quotechar='"', encoding='utf-8', on_bad_lines='warn').sample(n=25000)
data = data.dropna(how='any',axis=0)
kdata = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
kdata = data.dropna(how='any',axis=0)
kdata.head()


X = data['text'] + kdata['text']
y = data['label'] + kdata['label']


vectorizer = TfidfVectorizer()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
vX_train = vectorizer.fit_transform(X_train).toarray()
vX_test = vectorizer.transform(X_test).toarray()


# LINEAR REGRESSION

model = LinearRegression()
model.fit(vX_train, y_train)
'''
predictions = model.predict(v_test_essays)

predictions = predictions.clip(0, 1)

submission_df = pd.DataFrame({
    'id': test_essays['id'],
    'generated': predictions.round(2)
})

submission_df.to_csv('submission.csv', index=False)
'''

print("Linear Regression scored:", model.score(vX_test, y_test))


# LOGISTIC REGRESSION

model = LogisticRegression()
model.fit(vX_train, y_train)
'''
predictions = model.predict(v_test_essays)

predictions = predictions.clip(0, 1)

submission_df = pd.DataFrame({
    'id': test_essays['id'],
    'generated': predictions.round(2)
})

submission_df.to_csv('submission.csv', index=False)
'''
print("Logistic Regression scored:", model.score(vX_train, y_train))


from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
'''
vectorizer2 = TfidfVectorizer(max_features=1000)
X_train = vectorizer2.fit_transform(X_train).toarray()
X_test = vectorizer2.transform(X_test).toarray()


svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

svm_configs = {
    "Linear": SVC(kernel='linear'),
    "RBF": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Poly": SVC(kernel='poly', degree=3, coef0=1, C=1.0)
}

for name, model in svm_configs.items():
    model.fit(X_train_svd, y_train)
    preds = model.predict(X_test_svd)
    
    print(f"\n{name} SVM score on train data: {model.score(X_train_svd, y_train):.4f}")
    
    test_data[f'predicted_label_{name.lower()}'] = preds

clf = SVC(kernel='poly', degree=3, coef0=1, C=1.0)
clf.fit(X_train_svd, y_train)
predictions = clf.predict(X_test_svd)

submission_df = pd.DataFrame({
    'id': test["id"],
    'generated': predictions
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
'''


gnb = GaussianNB()
y_pred = gnb.fit(vX_train, y_train).predict(vX_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (vX_test.shape[0], (y_test != y_pred).sum()))
print(gnb.score(vX_test, y_test))


test_text1 = "Artificial Intelligence, or AI, has started to change many parts of our lives, including how we learn. In education, AI is having a big impact by making it easier for students and teachers to achieve their goals in new and effective ways. First of all, AI helps personalize education. Not all students learn in the same way or at the same speed, and AI can help by creating learning programs that match each student’s unique needs. For example, some math apps use AI to find out what a student is struggling with and then give them extra practice on those topics. This makes learning more efficient and can help students build confidence in their abilities. AI is also making education more accessible. For students who have disabilities, AI can provide tools such as speech-to-text for those who have trouble writing, or audio books for those who have difficulty reading. AI can also translate materials into different languages, making it easier for students who speak other languages to keep up in class. By breaking down barriers, AI helps create a more equal learning environment. Moreover, teachers are also benefiting from AI. With AI handling tasks like grading quizzes or giving feedback on essays, teachers have more time to help students individually and focus on creating engaging lessons. AI can even help teachers find new teaching methods and resources that fit their students' needs. Of course, there are some concerns about AI in education. Some people worry about privacy because AI needs a lot of data to work well. Others are afraid that if we rely too much on AI, teachers could become less important. However, most experts agree that AI should be used to help and support teachers, not to replace them. In conclusion, AI is bringing big changes to education. It is helping students learn in ways that are tailored just for them, making learning more accessible, and giving teachers powerful new tools. As long as we use AI carefully and responsibly, it can help make education better for everyone."
test_text = "I think AI has impacted education quite a bit, however the study of education remains very relevant."
text_test = vectorizer.transform([test_text]).toarray()

print(gnb.predict(text_test))


mnb = MultinomialNB()
y_pred = mnb.fit(vX_train, y_train).predict(vX_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (vX_test.shape[0], (y_test != y_pred).sum()))
print(mnb.score(vX_test, y_test))


clf = DecisionTreeClassifier()
y_pred = clf.fit(vX_train, y_train).predict(vX_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (vX_test.shape[0], (y_test != y_pred).sum()))
print(clf.score(vX_test, y_test))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


svm = SVC(kernel='linear', probability=True)
lr = LogisticRegression()
mnb = MultinomialNB()

clf = StackingClassifier(
    estimators=[('mnb', mnb), ('svm', svm), ('lr', lr)], final_estimator=LogisticRegression()
)
'''
clf.fit(vX_train, y_train)
y_pred = clf.predict(vX_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (vX_test.shape[0], (y_test != y_pred).sum()))
print(clf.score(vX_test, y_test))
'''

