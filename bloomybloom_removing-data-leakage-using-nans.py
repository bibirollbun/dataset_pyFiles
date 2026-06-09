!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv',index_col = 0)
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col=0)


sample_submission.shape


train.shape


test.shape


print(train.isnull().sum())


print(test.isnull().sum())


train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train['Personality'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})


test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})


train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)


train['Stage_fear'] = train['Stage_fear'].fillna(0.5)
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna(0.5)


test['Stage_fear'] = test['Stage_fear'].fillna(0.5)
test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna(0.5)


print(train.isnull().sum())


print(test.isnull().sum())


train.info()


train.describe()


test.info()


test.describe()


sns.countplot(data=train, x='Personality', palette='Set2')
plt.title("Class Distribution on the Train Set: Personality Types")
plt.xlabel("Personality Type")
plt.ylabel("Count")
plt.show()


X = train.drop('Personality', axis=1)
y = train['Personality']


df = X.copy()
df['Personality'] = y


df_unique = df.drop_duplicates(subset=X.columns)


X_unique = df_unique.drop('Personality', axis=1)
y_unique = df_unique['Personality']


X_train, X_val, y_train, y_val = train_test_split(
    X_unique, y_unique, 
    test_size=0.2, 
    stratify=y_unique, 
    random_state=42
)


train_data = X_train  
val_data = X_val     

duplicates = val_data.merge(train_data, how='inner')
print(f"Duplicates between train and val: {len(duplicates)}")


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_resampled, y_train_resampled)

ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_train_resampled, y_train_resampled)

sgd_classifier = SGDClassifier()
sgd_classifier.fit(X_train_resampled, y_train_resampled)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_resampled, y_train_resampled)

random_forest = RandomForestClassifier()
random_forest.fit(X_train_resampled, y_train_resampled)

gradient_boosting = GradientBoostingClassifier()
gradient_boosting.fit(X_train_resampled, y_train_resampled)

hist_gradient_boosting = HistGradientBoostingClassifier()
hist_gradient_boosting.fit(X_train_resampled, y_train_resampled)

svc = SVC()
svc.fit(X_train_resampled, y_train_resampled)

linear_svc = LinearSVC()
linear_svc.fit(X_train_resampled, y_train_resampled)

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train_resampled, y_train_resampled)

multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_resampled, y_train_resampled)

knn = KNeighborsClassifier()
knn.fit(X_train_resampled, y_train_resampled)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_resampled, y_train_resampled)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_resampled, y_train_resampled)

mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train_resampled, y_train_resampled)


y_pred_logistic_regression = logistic_regression.predict(X_val)
y_pred_ridge_classifier = ridge_classifier.predict(X_val)
y_pred_sgd_classifier = sgd_classifier.predict(X_val)
y_pred_decision_tree = decision_tree.predict(X_val)
y_pred_random_forest = random_forest.predict(X_val)
y_pred_gradient_boosting = gradient_boosting.predict(X_val)
y_pred_hist_gradient_boosting = hist_gradient_boosting.predict(X_val)
y_pred_svc = svc.predict(X_val)
y_pred_linear_svc = linear_svc.predict(X_val)
y_pred_gaussian_nb = gaussian_nb.predict(X_val)
y_pred_multinomial_nb = multinomial_nb.predict(X_val)
y_pred_knn = knn.predict(X_val)
y_pred_lda = lda.predict(X_val)
y_pred_qda = qda.predict(X_val)
y_pred_mlp = mlp.predict(X_val)


print("Logistic Regression:")
print(classification_report(y_val, y_pred_logistic_regression))

print("Ridge Classifier:")
print(classification_report(y_val, y_pred_ridge_classifier))

print("SGD Classifier:")
print(classification_report(y_val, y_pred_sgd_classifier))

print("Decision Tree:")
print(classification_report(y_val, y_pred_decision_tree))

print("Random Forest:")
print(classification_report(y_val, y_pred_random_forest))

print("Gradient Boosting:")
print(classification_report(y_val, y_pred_gradient_boosting))

print("HistGradient Boosting:")
print(classification_report(y_val, y_pred_hist_gradient_boosting))

print("SVC:")
print(classification_report(y_val, y_pred_svc))

print("Linear SVC:")
print(classification_report(y_val, y_pred_linear_svc))

print("Gaussian NB:")
print(classification_report(y_val, y_pred_gaussian_nb))

print("Multinomial NB:")
print(classification_report(y_val, y_pred_multinomial_nb))

print("KNN:")
print(classification_report(y_val, y_pred_knn))

print("LDA:")
print(classification_report(y_val, y_pred_lda))

print("QDA:")
print(classification_report(y_val, y_pred_qda))

print("MLP:")
print(classification_report(y_val, y_pred_mlp))


print("Logistic Regression:")
ConfusionMatrixDisplay.from_estimator(logistic_regression, X_val, y_val)

print("Ridge Classifier:")
ConfusionMatrixDisplay.from_estimator(ridge_classifier, X_val, y_val)

print("SGD Classifier:")
ConfusionMatrixDisplay.from_estimator(sgd_classifier, X_val, y_val)

print("Decision Tree:")
ConfusionMatrixDisplay.from_estimator(decision_tree, X_val, y_val)

print("Random Forest:")
ConfusionMatrixDisplay.from_estimator(random_forest, X_val, y_val)

print("Gradient Boosting:")
ConfusionMatrixDisplay.from_estimator(gradient_boosting, X_val, y_val)

print("HistGradient Boosting:")
ConfusionMatrixDisplay.from_estimator(hist_gradient_boosting, X_val, y_val)

print("SVC:")
ConfusionMatrixDisplay.from_estimator(svc, X_val, y_val)

print("Linear SVC:")
ConfusionMatrixDisplay.from_estimator(linear_svc, X_val, y_val)

print("Gaussian NB:")
ConfusionMatrixDisplay.from_estimator(gaussian_nb, X_val, y_val)

print("Multinomial NB:")
ConfusionMatrixDisplay.from_estimator(multinomial_nb, X_val, y_val)

print("KNN:")
ConfusionMatrixDisplay.from_estimator(knn, X_val, y_val)

print("LDA:")
ConfusionMatrixDisplay.from_estimator(lda, X_val, y_val)

print("QDA:")
ConfusionMatrixDisplay.from_estimator(qda, X_val, y_val)

print("MLP:")
ConfusionMatrixDisplay.from_estimator(mlp, X_val, y_val)


X_test = test.copy()
X_test


print("Logistic Regression:")
y_test_preds_logistic_regression = logistic_regression.predict(X_test)
print(y_test_preds_logistic_regression)

print("Ridge Classifier:")
y_test_preds_ridge_classifier = ridge_classifier.predict(X_test)
print(y_test_preds_ridge_classifier)

print("SGD Classifier:")
y_test_preds_sgd_classifier = sgd_classifier.predict(X_test)
print(y_test_preds_sgd_classifier)

print("Decision Tree:")
y_test_preds_decision_tree = decision_tree.predict(X_test)
print(y_test_preds_decision_tree)

print("Random Forest:")
y_test_preds_random_forest = random_forest.predict(X_test)
print(y_test_preds_random_forest)

print("Gradient Boosting:")
y_test_preds_gradient_boosting = gradient_boosting.predict(X_test)
print(y_test_preds_gradient_boosting)

print("HistGradient Boosting:")
y_test_preds_hist_gradient_boosting = hist_gradient_boosting.predict(X_test)
print(y_test_preds_hist_gradient_boosting)

print("SVC:")
y_test_preds_svc = svc.predict(X_test)
print(y_test_preds_svc)

print("Linear SVC:")
y_test_preds_linear_svc = linear_svc.predict(X_test)
print(y_test_preds_linear_svc)

print("Gaussian NB:")
y_test_preds_gaussian_nb = gaussian_nb.predict(X_test)
print(y_test_preds_gaussian_nb)

print("Multinomial NB:")
y_test_preds_multinomial_nb = multinomial_nb.predict(X_test)
print(y_test_preds_multinomial_nb)

print("KNN:")
y_test_preds_knn = knn.predict(X_test)
print(y_test_preds_knn)

print("LDA:")
y_test_preds_lda = lda.predict(X_test)
print(y_test_preds_lda)

print("QDA:")
y_test_preds_qda = qda.predict(X_test)
print(y_test_preds_qda)

print("MLP:")
y_test_preds_mlp = mlp.predict(X_test)
print(y_test_preds_mlp)



X_test['Personality_logistic_regression'] = y_test_preds_logistic_regression
X_test['Personality_logistic_regression'] = X_test['Personality_logistic_regression'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_ridge_classifier'] = y_test_preds_ridge_classifier
X_test['Personality_ridge_classifier'] = X_test['Personality_ridge_classifier'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_sgd_classifier'] = y_test_preds_sgd_classifier
X_test['Personality_sgd_classifier'] = X_test['Personality_sgd_classifier'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_decision_tree'] = y_test_preds_decision_tree
X_test['Personality_decision_tree'] = X_test['Personality_decision_tree'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_random_forest'] = y_test_preds_random_forest
X_test['Personality_random_forest'] = X_test['Personality_random_forest'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_gradient_boosting'] = y_test_preds_gradient_boosting
X_test['Personality_gradient_boosting'] = X_test['Personality_gradient_boosting'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_hist_gradient_boosting'] = y_test_preds_hist_gradient_boosting
X_test['Personality_hist_gradient_boosting'] = X_test['Personality_hist_gradient_boosting'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_svc'] = y_test_preds_svc
X_test['Personality_svc'] = X_test['Personality_svc'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_linear_svc'] = y_test_preds_linear_svc
X_test['Personality_linear_svc'] = X_test['Personality_linear_svc'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_gaussian_nb'] = y_test_preds_gaussian_nb
X_test['Personality_gaussian_nb'] = X_test['Personality_gaussian_nb'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_multinomial_nb'] = y_test_preds_multinomial_nb
X_test['Personality_multinomial_nb'] = X_test['Personality_multinomial_nb'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_knn'] = y_test_preds_knn
X_test['Personality_knn'] = X_test['Personality_knn'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_lda'] = y_test_preds_lda
X_test['Personality_lda'] = X_test['Personality_lda'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_qda'] = y_test_preds_qda
X_test['Personality_qda'] = X_test['Personality_qda'].map({0: 'Extrovert', 1: 'Introvert'})

X_test['Personality_mlp'] = y_test_preds_mlp
X_test['Personality_mlp'] = X_test['Personality_mlp'].map({0: 'Extrovert', 1: 'Introvert'})

X_test


predicted_labels_logistic_regression = X_test['Personality_logistic_regression']
predicted_labels_ridge_classifier = X_test['Personality_ridge_classifier']
predicted_labels_sgd_classifier = X_test['Personality_sgd_classifier']
predicted_labels_decision_tree = X_test['Personality_decision_tree']
predicted_labels_random_forest = X_test['Personality_random_forest']
predicted_labels_gradient_boosting = X_test['Personality_gradient_boosting']
predicted_labels_hist_gradient_boosting = X_test['Personality_hist_gradient_boosting']
predicted_labels_svc = X_test['Personality_svc']
predicted_labels_linear_svc = X_test['Personality_linear_svc']
predicted_labels_gaussian_nb = X_test['Personality_gaussian_nb']
predicted_labels_multinomial_nb = X_test['Personality_multinomial_nb']
predicted_labels_knn = X_test['Personality_knn']
predicted_labels_lda = X_test['Personality_lda']
predicted_labels_qda = X_test['Personality_qda']
predicted_labels_mlp = X_test['Personality_mlp']


true_labels = sample_submission['Personality']

accuracy_logistic_regression = accuracy_score(predicted_labels_logistic_regression, true_labels)
print(f"Model Accuracy_logistic_regression: {accuracy_logistic_regression * 100:.2f}%")

accuracy_ridge_classifier = accuracy_score(predicted_labels_ridge_classifier, true_labels)
print(f"Model Accuracy_ridge_classifier: {accuracy_ridge_classifier * 100:.2f}%")

accuracy_sgd_classifier = accuracy_score(predicted_labels_sgd_classifier, true_labels)
print(f"Model Accuracy_sgd_classifier: {accuracy_sgd_classifier * 100:.2f}%")

accuracy_decision_tree = accuracy_score(predicted_labels_decision_tree, true_labels)
print(f"Model Accuracy_decision_tree: {accuracy_decision_tree * 100:.2f}%")

accuracy_random_forest = accuracy_score(predicted_labels_random_forest, true_labels)
print(f"Model Accuracy_random_forest: {accuracy_random_forest * 100:.2f}%")

accuracy_hist_gradient_boosting = accuracy_score(predicted_labels_hist_gradient_boosting, true_labels)
print(f"Model Accuracy_hist_gradient_boosting: {accuracy_hist_gradient_boosting * 100:.2f}%")

accuracy_svc = accuracy_score(predicted_labels_svc, true_labels)
print(f"Model Accuracy_svc: {accuracy_svc * 100:.2f}%")

accuracy_linear_svc = accuracy_score(predicted_labels_linear_svc, true_labels)
print(f"Model Accuracy_linear_svc: {accuracy_linear_svc * 100:.2f}%")

accuracy_gaussian_nb = accuracy_score(predicted_labels_gaussian_nb, true_labels)
print(f"Model Accuracy_gaussian_nb: {accuracy_gaussian_nb * 100:.2f}%")

accuracy_multinomial_nb = accuracy_score(predicted_labels_multinomial_nb, true_labels)
print(f"Model Accuracy_multinomial_nb: {accuracy_multinomial_nb * 100:.2f}%")

accuracy_knn = accuracy_score(predicted_labels_knn, true_labels)
print(f"Model Accuracy_knn: {accuracy_knn * 100:.2f}%")

accuracy_lda = accuracy_score(predicted_labels_lda, true_labels)
print(f"Model Accuracy_lda: {accuracy_lda * 100:.2f}%")

accuracy_qda = accuracy_score(predicted_labels_qda, true_labels)
print(f"Model Accuracy_qda: {accuracy_qda * 100:.2f}%")

accuracy_mlp = accuracy_score(predicted_labels_mlp, true_labels)
print(f"Model Accuracy_mlp: {accuracy_mlp * 100:.2f}%")


submission = X_test[['Personality_knn']].copy()
submission.reset_index(inplace=True) 

submission.rename(columns={'id': 'id', 'Personality_knn': 'Personality'}, inplace=True)


submission.to_csv('submission.csv', index=False)

print("Submission file saved successfully!")

