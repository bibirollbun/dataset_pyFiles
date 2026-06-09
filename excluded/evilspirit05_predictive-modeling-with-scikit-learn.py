import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline


col_names=[f"Col {i}" for i in range(1, 41)]
df=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv",names=col_names)
label=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv",names=["label"])



df.head()


df.shape


df.isnull().sum()


df.info()


df.describe()


df=pd.concat([df,label],axis=1)



df.head()


corr = df.corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(20, 8))
sns.heatmap(corr, annot=True, cmap='gnuplot2', fmt='.2f', linewidths=0.5)
plt.show()


sns.countplot(x='label', data=df)
plt.show()


plt.scatter(df['Col 1'], df['Col 2'])
plt.xlabel('Col 1')
plt.ylabel('Col 2')
plt.title('Scatter Plot between Col 1 and Col 2')
plt.show()


plt.figure(figsize=(20,5))
sns.violinplot(data=df, inner="quart")
plt.xticks(rotation=45)
plt.show()


df.plot(kind='box', figsize=(15, 8), vert=False)
plt.show()



import missingno as msno
msno.matrix(df)
plt.show()


# correlation_matrix = df.corr()
# correlation_with_label = correlation_matrix['label'].sort_values(ascending=False)
# print(correlation_with_label.head(15))

# plt.figure(figsize=(20, 5))
# sns.barplot(x=correlation_with_label.index, y=correlation_with_label.values, palette='coolwarm')
# plt.title('Feature Correlation with Label')
# plt.xticks(rotation=90)
# plt.show()


df.head()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = df.iloc[:, :-1]  # All columns except the last column ('label')
y = df['label']      # The target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

# Initialize and train RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")



from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Predict and evaluate
y_pred_gb = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")



from sklearn.ensemble import AdaBoostClassifier

# Initialize and train AdaBoostClassifier
ab = AdaBoostClassifier(n_estimators=100, random_state=42)
ab.fit(X_train, y_train)

# Predict and evaluate
y_pred_ab = ab.predict(X_test)
ab_accuracy = accuracy_score(y_test, y_pred_ab)
print(f"AdaBoost Accuracy: {ab_accuracy:.4f}")



import xgboost as xgb

# Initialize and train XGBoost
xg = xgb.XGBClassifier(n_estimators=100, random_state=42)
xg.fit(X_train, y_train)

# Predict and evaluate
y_pred_xg = xg.predict(X_test)
xg_accuracy = accuracy_score(y_test, y_pred_xg)
print(f"XGBoost Accuracy: {xg_accuracy:.4f}")



from catboost import CatBoostClassifier

# Initialize and train CatBoostClassifier
cb = CatBoostClassifier(iterations=5000, random_state=42, verbose=500,learning_rate=0.1)
cb.fit(X_train, y_train)

# Predict and evaluate
y_pred_cb = cb.predict(X_test)
cb_accuracy = accuracy_score(y_test, y_pred_cb)
print(f"CatBoost Accuracy: {cb_accuracy:.4f}")



# Store all accuracies
accuracies = {
    'Random Forest': rf_accuracy,
    'Gradient Boosting': gb_accuracy,
    'AdaBoost': ab_accuracy,
    'XGBoost': xg_accuracy,
    'CatBoost': cb_accuracy
}

# Plotting the comparison of model accuracies
models = list(accuracies.keys())
scores = list(accuracies.values())

plt.figure(figsize=(20, 5))
sns.barplot(x=models, y=scores, palette='Set1')
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.show()



df_test=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv",names=col_names)
df_test.reset_index(drop=True, inplace=True)
df_test.head()


df_test.shape


df_test.head()


prediction=cb.predict(df_test)
my_id = range(1, len(prediction) + 1)
my_submission=pd.DataFrame({"Id":my_id,"Solution":prediction})
my_submission.to_csv("submission.csv",index=False)


my_submission.head()


my_submission.shape

