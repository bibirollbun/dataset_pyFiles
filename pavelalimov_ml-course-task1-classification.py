import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

df.head()


df.describe()


df.info()


scaler = MinMaxScaler()

df_scaled = df.copy()
df_scaled[
    [
        "age", "balance", "day", "duration"
    ]
] = scaler.fit_transform(
    df[["age", "balance", "day", "duration"]]
)

df_scaled.head()


le = LabelEncoder()


df_labeled = df_scaled.copy()
categorical_cols = df_labeled.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_labeled[col] = le.fit_transform(df_labeled[col])

df_labeled.head()


x = df_labeled.drop(columns=["y", "id"])
y = df_labeled["y"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


models_to_evaluate = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
    "SVC_poly": SVC(kernel='poly', max_iter=1000),
    "SVC_rbf": SVC(kernel='rbf', max_iter=1000),
    "SVC_linear": SVC(kernel='linear', max_iter=1000),
    "LinearSVC": LinearSVC(max_iter=1000, tol=1e-3),
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight='balanced'),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3, algorithm="brute"),
    "MLPClassifier": MLPClassifier(max_iter=1000, early_stopping=True, n_iter_no_change=10, tol=1e-3),
}


total_train_time = time.perf_counter()
for model_name, model in models_to_evaluate.items():
    print(f"Start to train model {model_name}")
    train_time = time.perf_counter()
    model.fit(x_train, y_train)
    print(f"Train took {(time.perf_counter() - train_time):.4f} (s)")
print(f"Total train time {(time.perf_counter() - total_train_time):.4f}")


for model_name, model in models_to_evaluate.items():
    print(f"Classification report for model {model_name}")
    print(classification_report(y_test, model.predict(x_test)))
    print("\n*******\n")


def get_cm(classifier):
    return confusion_matrix(y_test, classifier.predict(x_test))

plt.figure(figsize=(36, 18))
plt.suptitle("Confusion Matrixes", fontsize=24)
plt.subplots_adjust(wspace = 0.2, hspace= 0.3)


for idx, model_name in enumerate(models_to_evaluate):
    cm = get_cm(models_to_evaluate[model_name])
    plt.subplot(2, 4, idx+1)
    plt.title(model_name)
    sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()

