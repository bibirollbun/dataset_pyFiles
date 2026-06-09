import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.metrics import classification_report


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train_df.head()


test_df.head()


for col in train_df.select_dtypes(include="object").columns:
    print("="*20)
    print(train_df[col].value_counts())


for col in train_df.select_dtypes(include="object").columns:
    print("="*50)
    print(train_df[col].value_counts().plot(kind="bar"))
    plt.show()


train_df.isna().sum()


# Check how many classes

train_df["diagnosed_diabetes"].value_counts().plot(kind='bar')
plt.show()


X = train_df.drop(columns="diagnosed_diabetes")
y = train_df["diagnosed_diabetes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42
                                                   )



class Preprocessing:

    def __init__(self):
        self.encoders = {}

    def fit(self, X_train):
        for col in X_train.select_dtypes(include="object").columns:
            le = LabelEncoder()
            le.fit(X_train[col])
            self.encoders[col] = le

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = le.transform(X[col]) 
        return X




prep = Preprocessing()
prep.fit(X_train)

X_train_enc = prep.transform(X_train)
X_test_enc = prep.transform(X_test)




X_test_enc.head()


from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer



gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}



# def tune_model(model, param_grid, X_train, y_train):
#     grid = GridSearchCV(
#         model,
#         param_grid,
#         scoring=make_scorer(f1_score),
#         cv=5,
#         n_jobs=-1
#     )
#     grid.fit(X_train, y_train)
#     print("Best parameters:", grid.best_params_)
#     print("Best F1 score:", grid.best_score_)
#     return grid.best_estimator_


model = GradientBoostingClassifier(random_state=42, n_estimators=500, 
                                   learning_rate=0.02, max_depth=20,min_samples_split=20, min_samples_leaf=20)

model.fit(X_train_enc, y_train)


y_pred = model.predict(X_test_enc)



from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

labels = ["Class 0", "Class 1"] 
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.show()


test_df.head()


x_test = prep.transform(test_df)


x_test.head()


predictions = model.predict(x_test)



sub_file = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
sub_file.head()


submission_df = pd.DataFrame({
    'id': sub_file['id'],
    'diagnosed_diabetes': predictions
})
submission_df.to_csv("submission.csv", index=False)


sub = pd.read_csv("/kaggle/working/submission.csv")


sub.head()




