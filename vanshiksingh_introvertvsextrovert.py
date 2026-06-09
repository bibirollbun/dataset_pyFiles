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


!pip uninstall -y scikit-learn imbalanced-learn
!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0



import sklearn
import imblearn
print(sklearn.__version__)
print(imblearn.__version__)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df.head(10)


print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
df.describe()



sns.countplot(data=df, x="Personality", palette="coolwarm")
plt.title("Personality Distribution")
plt.show()

print(df["Personality"].value_counts(normalize=True))



num_cols = df.select_dtypes(include=["float64","int64"]).columns
df[num_cols].hist(bins=20,figsize=(12,10),color="#6aabd2",edgecolor="black")
plt.suptitle("Distribution of Numerical Features",fontsize=16)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
plt.title("Missing Value Map")
plt.show()


for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df,x="Personality",y=col, palette="Set2")
    plt.title(f"{col} vs Personality")
    plt.show()


cat_cols = ["Stage_fear","Drained_after_socializing"]

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df,x=col,hue="Personality",palette="Set1")
    plt.title(f"{col} by Personality")
    plt.show()




num_cols = df.select_dtypes(include=['float64','int64']).columns


plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()


df.isnull().sum()


data = df.copy()

nums_cols = data.select_dtypes(include=['float64','int64']).columns.drop('id')
data[nums_cols] = data[nums_cols].fillna(data[num_cols].median())


cat_cols = ['Stage_fear','Drained_after_socializing']
data[cat_cols] = data[cat_cols].apply(lambda x : x.fillna(x.mode()[0]))



data.isnull().sum()


data.head()
data['Personality'] = data['Personality'].map({'Introvert': 0,'Extrovert': 1})
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])


scaler = StandardScaler()
scaled_feature = scaler.fit_transform(data[num_cols])
data[num_cols] = scaled_feature


data.head()


X = data.drop(['Personality','id'],axis=1)
y = data['Personality']

X_train , X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)







# Train basic RF on imbalanced data to inspect feature importance
rf_imp = RandomForestClassifier(random_state=42)
rf_imp.fit(X_train, y_train)

# Get feature importances
importances = rf_imp.feature_importances_
feature_names = X_train.columns

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.show()



from imblearn.over_sampling import SMOTE


smote = SMOTE(random_state=42)
X_resampled ,y_resampled = smote.fit_resample(X_train,y_train)
print("Before SMOTE: ",y_train.value_counts())
print("After SMOTE: ",pd.Series(y_resampled).value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    print(f"=== {name} ===")
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print()



for name, model in models.items():
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='f1')
    print(f"{name} Cross-validated F1 Score: {scores.mean():.4f}")



from sklearn.model_selection import RandomizedSearchCV

# Step 1: Resample the data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 2: Define hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Step 3: Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Step 4: Randomized search with 5-fold CV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,  # increase if more time/compute
    cv=5,
    verbose=2,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

# Step 5: Fit
random_search.fit(X_resampled, y_resampled)

# Step 6: Best estimator
best_rf = random_search.best_estimator_
print("Best Parameters:\n", random_search.best_params_)

# Step 7: Evaluate on test set
y_pred = best_rf.predict(X_test)
print("\nTuned Random Forest Report:\n")
print(classification_report(y_test, y_pred))



# X_resampled ,y_resampled = smote.fit_resample(X_train,y_train)
# rf = RandomForestClassifier()
# rf.fit(X_resampled, y_resampled)
# y_pred = rf.predict(X_test)
# print(classification_report(y_test, y_pred))


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test_id = test['id']


test[num_cols] = test[num_cols].fillna(test[num_cols].median())
test[cat_cols] = test[cat_cols].apply(lambda x: x.fillna(test[x.name].mode()[0]))

for col in cat_cols:
    test[col] = le.transform(test[col])

test[num_cols] = scaler.transform(test[num_cols])


test.head()


test_preds = best_rf.predict(test.drop("id",axis=1))

test_preds_labels = pd.Series(test_preds).map({0:"Introvert",1:"Extrovert"})



submission = pd.DataFrame({
    "id": test_id,
    "Personality": test_preds_labels
})

# Save as CSV
submission.to_csv("submission.csv", index=False)





