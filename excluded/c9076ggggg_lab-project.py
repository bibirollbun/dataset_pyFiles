import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/decision-trees-from-scratch-2024/train.csv")

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object': 
        data[col] = le.fit_transform(data[col].astype(str))

X = data.iloc[:, :-1]   
y = data.iloc[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(12,6))
plot_tree(clf, feature_names=X.columns, filled=True)
plt.show()

test_data = pd.read_csv("/kaggle/input/decision-trees-from-scratch-2024/test.csv")

X_final = test_data.drop(columns=["id"])

preds = clf.predict(X_final)

sample = pd.read_csv("/kaggle/input/decision-trees-from-scratch-2024/sample_submission.csv")

sample.iloc[:, 1] = preds  

sample.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


