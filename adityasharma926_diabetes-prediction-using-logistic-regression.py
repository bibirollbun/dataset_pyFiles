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


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression #LinearRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import (accuracy_score, confusion_matrix,
                            classification_report, roc_curve,
                            roc_auc_score, precision_score,
                            recall_score, f1_score)
from sklearn.metrics import ConfusionMatrixDisplay


data_path = '/kaggle/input/playground-series-s5e12/train.csv'
data = pd.read_csv(data_path)
data.head()


data.drop('id',axis = 1, inplace=True)


data.info()


data.describe()


print("Number Of Columns:", len(data.columns.tolist()))


string_data = data.select_dtypes(include=['object'])
str_cols = string_data.columns.tolist()
num_cols = data.select_dtypes(include=['number']).columns.tolist()
print("Numerical Columns:", len(num_cols))
for i in num_cols:
    print(i, end=', ')
print("\nString Columns: ", len(str_cols))
for i in str_cols:
    print(i, end = ", ")


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(15,13))
for i, col in enumerate(str_cols, 1):
    plt.subplot(3,3, i)
    sns.set(style='darkgrid')
    gen = data.groupby(col).size()
    gen.plot.bar(stacked=True)
    plt.title(col)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import OrdinalEncoder

def encoder(str_cols, df):
    data = df
    for col in str_cols:
        ord1 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Fit on a DataFrame (2D)
        ord1.fit(data[[col]])

        # Transform using the same shape
        data[col] = ord1.transform(data[[col]]).astype(int)
        
    return data


data = encoder(str_cols, data)
data.head()


plt.figure(figsize=(30,25))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Feature correlation heatmap")
plt.show()


plt.figure(figsize=(7, 5))
data['diagnosed_diabetes'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel('Diagnosation')
plt.ylabel("Values")
plt.xticks([0,1], ['True', "False"], rotation = 0)
plt.grid(axis = 'y', alpha = 0.3)
plt.show()


#Applying the model .drop('diagnosed_diabetes',axis = 1) #
X = data.drop('diagnosed_diabetes',axis = 1) #[['age', 'bmi','systolic_bp', 'ldl_cholesterol', 'triglycerides', 'family_history_diabetes', 'screen_time_hours_per_day','cholesterol_total', 'heart_rate']]
y = data['diagnosed_diabetes']

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)

model  = LogisticRegression(max_iter = 200, random_state = 42)

model.fit(X_train_scaled, y_train)


print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")


#predict 

pred = model.predict_proba(X_test_scaled)


# evaluation
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
print("Accuracy:", accuracy)
print('precision', precision)
print("F1:", f1)
print('recall:', recall)

print("\nClassification report:")
print(classification_report(y_test, pred))


# Compute and visualize the confusion matrix
cm = confusion_matrix(y_test, pred)
cmd = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['False', 'True']
)

plt.figure(figsize=(8, 6))
cmd.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()

# Explain the confusion matrix components
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix Components:")
print(f"True Positives (TP): {tp} - Correctly predicted as approved")
print(f"True Negatives (TN): {tn} - Correctly predicted as rejected")
print(f"False Positives (FP): {fp} - Incorrectly predicted as approved (Type I error)")
print(f"False Negatives (FN): {fn} - Incorrectly predicted as rejected (Type II error)")


coefs = pd.DataFrame(data=model.coef_[0],
                     index= X.columns,
                     columns=['importance']).abs().sort_values(by=['importance'],ascending=False)
coefs['importance'] /= coefs['importance'].sum()  #L1 normalization
coefs.plot(rot=60)
coefs['total_%'] = coefs['importance'].cumsum().round(3)*100
coefs


important_features = coefs[coefs['importance']*100>=0.9] #.index.to_list()
important_features


X = data[important_features.index.to_list()]
X.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.fit_transform(X_test)


k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=42)


model = LogisticRegression(max_iter = 200, random_state = 42)
model.fit(X_train_scaled, y_train)


scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')

print(f"Accuracy for each fold: {scores}")

average_accuracy = np.mean(scores) 
print(f"Average Accuracy: {average_accuracy:.2f}")


train_pred = model.predict(X_train_scaled)
accuracy = accuracy_score(train_pred, y_train)
print("Accuracy Score:", accuracy)


test_path = "/kaggle/input/playground-series-s5e12/test.csv"
test = pd.read_csv(test_path)


test.shape


test = encoder(str_cols, test)
test.head()


test.drop('id', axis = 1)


# X_test = test.drop('diagnosed_diabetes',axis = 1)  
X_test = test[important_features.index.to_list()]
x_test_scaled = scaler.fit_transform(X_test)


test_pred = model.predict(x_test_scaled)


submission_data = pd.DataFrame({
        "id": test['id'],
        "diagnosed_diabetes": test_pred
})


submission_data.head()


submission_data.to_csv('submission.csv', index = False)

