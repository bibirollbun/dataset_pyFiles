import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             matthews_corrcoef, confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score)


df=pd.read_csv("/kaggle/input/playground-series-s3e26/train.csv")
df


df.drop(['N_Days'],axis=1,inplace=True)


#df.drop(['id'],axis=1,inplace=True)


# Count the missing values
print("Missing values:\n", df.isnull().sum())


df


df['Status'].value_counts()


plt.figure(figsize=(10,6))
sns.countplot(x='Status', data=df, order=df['Status'].value_counts().index);


columns=['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema','Status']#
le=LabelEncoder()
for i in columns:
    df[i]=le.fit_transform(df[i])

df


x = df.drop('Status', axis=1)
y = df['Status']


from imblearn.over_sampling import SVMSMOTE
from collections import Counter
sm = SVMSMOTE(random_state=55)
xx, yy = sm.fit_resample(x,y)


plt.figure(figsize=(10,6))
sns.countplot(x=yy,data=df)
print('Resampled dataset shape %s' % Counter(yy))


# split testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=20) 


# Feature importance using XGBoost

rf = XGBClassifier(random_state=42)
rf.fit(X_train, y_train)  # Use X_train_strat and y_train_strat

# Get feature importances
feature_importances = rf.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(13, 6))
ax = sns.barplot(x='Importance', y='Feature', data=importance_df)

# Annotate the bars with the numerical importance values
for index, value in enumerate(importance_df['Importance']):
    ax.text(value, index, f'{value:.4f}', va='center', ha='left')

plt.title('Feature Importance from XGBoost')
plt.show()



# Logistic Regression Classifier
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

# Make predictions
predictions_lr = model_lr.predict(X_test)

# Calculate accuracy for training and testing sets
train_accuracy = accuracy_score(y_train, model_lr.predict(X_train))
test_accuracy = accuracy_score(y_test, predictions_lr)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Classification report
report = classification_report(y_test, predictions_lr)
print("Classification Report:\n", report)

# performance metrics
accuracy = accuracy_score(y_test, predictions_lr)
precision = precision_score(y_test, predictions_lr, average='weighted')
recall = recall_score(y_test, predictions_lr, average='weighted')
f1 = f1_score(y_test, predictions_lr, average='weighted')
mcc = matthews_corrcoef(y_test, predictions_lr)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print('*******************************************')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions_lr)

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.set(font_scale=2.5)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':30}, fmt=".0f", square=True, cmap=plt.cm.Blues, linewidths=0.8)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Logistic Regression')
plt.tight_layout()
plt.show()


# SVM Classifier
model_svm = SVC(kernel='rbf', probability=True)  # Using RBF kernel
model_svm.fit(X_train, y_train)

# Make predictions
predictions_svm = model_svm.predict(X_test)

# training and testing sets
train_accuracy = accuracy_score(y_train, model_svm.predict(X_train))
test_accuracy = accuracy_score(y_test, predictions_svm)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

#classification report
report = classification_report(y_test, predictions_svm)
print("Classification Report:\n", report)

# performance metrics
accuracy = accuracy_score(y_test, predictions_svm)
precision = precision_score(y_test, predictions_svm, average='weighted')
recall = recall_score(y_test, predictions_svm, average='weighted')
f1 = f1_score(y_test, predictions_svm, average='weighted')
mcc = matthews_corrcoef(y_test, predictions_svm)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print('*******************************************')

# Confusion Matrix 
conf_matrix = confusion_matrix(y_test, predictions_svm)

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.set(font_scale=2.5)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':30}, fmt=".0f", square=True, cmap=plt.cm.Blues, linewidths=0.8)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix SVM')
plt.tight_layout()
plt.show()



# XGBoost Classifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

# Make predictions
predictions_xgb = model_xgb.predict(X_test)

# training and testing sets
train_accuracy = accuracy_score(y_train, model_xgb.predict(X_train))
test_accuracy = accuracy_score(y_test, predictions_xgb)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Generate and print classification report
report = classification_report(y_test, predictions_xgb)
print("Classification Report:\n", report)

# performance metrics
accuracy = accuracy_score(y_test, predictions_xgb)
precision = precision_score(y_test, predictions_xgb, average='weighted')
recall = recall_score(y_test, predictions_xgb, average='weighted')
f1 = f1_score(y_test, predictions_xgb, average='weighted')
mcc = matthews_corrcoef(y_test, predictions_xgb)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print('*******************************************')

# Confusion Matrix 
conf_matrix = confusion_matrix(y_test, predictions_xgb)

# Plot confusion matrix
plt.figure(figsize=(12,8))
sns.set(font_scale= 2.5)
sns.heatmap(conf_matrix, annot = True, annot_kws={'size':30}
            ,fmt = ".0f", square = True, cmap = plt.cm.Blues,linewidths=0.8)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix XGB ')
plt.tight_layout()


# AdaBoost Classifier
model_ada = AdaBoostClassifier(n_estimators=50)
model_ada.fit(X_train, y_train)

# Make predictions
predictions_ada = model_ada.predict(X_test)

# training and testing sets
train_accuracy = accuracy_score(y_train, model_ada.predict(X_train))
test_accuracy = accuracy_score(y_test, predictions_ada)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# classification report
report = classification_report(y_test, predictions_ada)
print("Classification Report:\n", report)

# Calculate and print various performance metrics
accuracy = accuracy_score(y_test, predictions_ada)
precision = precision_score(y_test, predictions_ada, average='weighted')
recall = recall_score(y_test, predictions_ada, average='weighted')
f1 = f1_score(y_test, predictions_ada, average='weighted')
mcc = matthews_corrcoef(y_test, predictions_ada)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print('*******************************************')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions_ada)


# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.set(font_scale=2.5)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':30}, fmt=".0f", square=True, cmap=plt.cm.Blues, linewidths=0.8)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix AdaBoost')
plt.tight_layout()
plt.show()


# K-Nearest Neighbors Classifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)

# Make predictions
predictions_knn = model_knn.predict(X_test)

# accuracy for training and testing sets
train_accuracy = accuracy_score(y_train, model_knn.predict(X_train))
test_accuracy = accuracy_score(y_test, predictions_knn)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Generate and print classification report
report = classification_report(y_test, predictions_knn)
print("Classification Report:\n", report)


# performance metrics
accuracy = accuracy_score(y_test, predictions_knn)
precision = precision_score(y_test, predictions_knn, average='weighted')
recall = recall_score(y_test, predictions_knn, average='weighted')
f1 = f1_score(y_test, predictions_knn, average='weighted')
mcc = matthews_corrcoef(y_test, predictions_knn)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print('*******************************************')

# Confusion Matrix 
conf_matrix = confusion_matrix(y_test, predictions_knn)

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.set(font_scale=2.5)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':30}, fmt=".0f", square=True, cmap=plt.cm.Blues, linewidths=0.8)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix KNN')
plt.tight_layout()
plt.show()


test=pd.read_csv("/kaggle/input/playground-series-s3e26/test.csv")
test


test.drop(['N_Days'],axis=1,inplace=True)


 test


from sklearn.preprocessing import LabelEncoder

c = ['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']
#l_encoders = {}  # Dictionary to store encoders for each column

le=LabelEncoder()
for i in c:
    test[i]=le.fit_transform(test[i])


test



# Make predictions
predictions_xgb = model_xgb.predict(X_test)

# training and testing sets
train_accuracy = accuracy_score(y_train, model_xgb.predict(X_train))
test_accuracy = accuracy_score(y_test, predictions_xgb)


prob = model_xgb.predict_proba(test)  
prob


submission = pd.DataFrame(prob, columns=['Status_C', 'Status_D', 'Status_CL'])
submission['id'] = test['id']
submission = submission[['id', 'Status_C', 'Status_D', 'Status_CL']]
submission.to_csv('submission.csv', index=False)
submission

