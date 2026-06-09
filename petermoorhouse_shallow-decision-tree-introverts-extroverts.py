import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


TRAIN_PATH = "/kaggle/input/playground-series-s5e7/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e7/test.csv"
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df.head()


X = train_df.drop(columns=["Personality", "id"])
y = train_df["Personality"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
    ("num", SimpleImputer(strategy="mean"), num_cols)
])


DEPTH_LIMIT = 3

classes = np.sort(y.unique())
class_to_index = {c: i for i, c in enumerate(classes)}
index_to_class = {i: c for c, i in class_to_index.items()}
y_encoded = y.map(class_to_index)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=DEPTH_LIMIT, random_state=56))
])

model.fit(X, y_encoded)


train_preds = model.predict(X)
train_accuracy = accuracy_score(y_encoded, train_preds)
print(f"Accuracy on training data: {train_accuracy:.4f}")


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    model.named_steps["classifier"],
    feature_names=model.named_steps["preprocessor"].get_feature_names_out(),
    class_names=classes,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()


X_test = test_df.drop(columns=["id"])
test_preds = model.predict(X_test)
test_labels = [index_to_class[i] for i in test_preds]


submission = pd.DataFrame({
    "id": test_df["id"],
    "Personality": test_labels
})

submission.to_csv("submission.csv", index=False)
submission.head()

