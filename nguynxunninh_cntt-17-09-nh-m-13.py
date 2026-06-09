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


# ========================================
# 1. Data Source
# ========================================
import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("/kaggle/input/spam-message-classification/train.csv")

# Đổi tên cột cho dễ dùng
df = df.rename(columns={"sms": "message", "# label": "label"})

print("Kích thước dữ liệu:", df.shape)
print(df["label"].value_counts())
df.head()



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải stopwords + wordnet nếu chưa có
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 1. lowercase
    text = text.lower()

    # 2. loại bỏ URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3. loại bỏ ký tự đặc biệt, số
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4. loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    # 5. tách từ
    words = text.split()

    # 6. bỏ stopwords + lemmatization
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)

# Áp dụng vào DataFrame
df["clean_message"] = df["message"].astype(str).apply(clean_text)
df.head()



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = df["clean_message"] 
y = df["label"] 
# Tách train/test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# TF-IDF 
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=5
)
X_train_vec = vectorizer.fit_transform(X_train) 
X_test_vec = vectorizer.transform(X_test) 

print("Kích thước TF-IDF:", X_train_vec.shape)


from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
import time 

models = { 
    "Naive Bayes": MultinomialNB(), 
    "KNN": KNeighborsClassifier(n_neighbors=5), 
    "Decision Tree": DecisionTreeClassifier(random_state=42) 
} 

results = {}   # cần khởi tạo trước

for name, model in models.items(): 
    start = time.time() 
    model.fit(X_train_vec, y_train) 
    y_pred = model.predict(X_test_vec) 
    end = time.time() 
    acc = (y_pred == y_test).mean() 
    
    results[name] = {
        "model": model, 
        "y_pred": y_pred, 
        "acc": acc, 
        "time": end - start
    } 
    
    print(f"{name}: Accuracy={acc:.4f}, Training time={end-start:.2f}s")



# ========================================
# 5. Evaluation
# ========================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

for name, res in results.items():
    print("="*50)
    print(f"🔹 {name}")
    print(classification_report(y_test, res["y_pred"]))
    
    cm = confusion_matrix(y_test, res["y_pred"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# ========================================
# 6. So sánh Accuracy & Training Time
# ========================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_names = np.array(list(results.keys()))
accuracies = np.array([res["acc"] for res in results.values()])
times = np.array([res["time"] for res in results.values()])

# --- Biểu đồ Accuracy ---
plt.figure(figsize=(6,4))
sns.barplot(x=model_names, y=accuracies, palette="viridis")
plt.ylim(0, 1)
plt.title("So sánh Accuracy giữa các mô hình")
plt.ylabel("Accuracy")
plt.xlabel("Model")

# Hiển thị giá trị trên cột
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

plt.show()

# --- Biểu đồ Training Time ---
plt.figure(figsize=(6,4))
sns.barplot(x=model_names, y=times, palette="mako")
plt.title("So sánh Training Time giữa các mô hình")
plt.ylabel("Thời gian (s)")
plt.xlabel("Model")

# Hiển thị giá trị trên cột
for i, v in enumerate(times):
    plt.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

plt.show()



# ========================================
# 7. Lưu mô hình tốt nhất
# ========================================
import joblib

# Chọn mô hình có Accuracy cao nhất
best_name = max(results, key=lambda x: results[x]["acc"])
best_model = results[best_name]["model"]

# Lưu model + vectorizer
joblib.dump(best_model, f"{best_name}_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print(f"✅ Đã lưu mô hình tốt nhất: {best_name}")



# ========================================
# 8. Cải thiện cân bằng dữ liệu (Random Oversampling thủ công, sparse-friendly)
# ========================================
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

y_train_arr = np.array(y_train)
X_train_arr = X_train_vec

# Đếm số mẫu từng lớp
class_counts = pd.Series(y_train_arr).value_counts()
max_count = class_counts.max()

print("📊 Trước ROS:", dict(class_counts))

# Oversampling thủ công
X_resampled = []
y_resampled = []

for cls in class_counts.index:
    X_cls = X_train_arr[y_train_arr == cls]
    y_cls = y_train_arr[y_train_arr == cls]

    n_to_add = max_count - X_cls.shape[0]

    if n_to_add > 0:
        idx = np.random.choice(X_cls.shape[0], size=n_to_add, replace=True)
        X_add = X_cls[idx]
        y_add = y_cls[idx]
        X_cls = vstack([X_cls, X_add])
        y_cls = np.concatenate([y_cls, y_add])

    X_resampled.append(X_cls)
    y_resampled.append(y_cls)

# Ghép lại
X_train_bal = vstack(X_resampled)
y_train_bal = np.concatenate(y_resampled)

print("📊 Sau ROS:", dict(pd.Series(y_train_bal).value_counts()))

# Train lại
models_bal = {
    "Naive Bayes (ROS)": MultinomialNB(),
    "KNN (ROS)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (balanced+ROS)": DecisionTreeClassifier(random_state=42, class_weight="balanced")
}

results_bal = {}

for name, model in models_bal.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_vec)
    acc = (y_pred == y_test).mean()
    results_bal[name] = acc
    print(f"{name}: Accuracy={acc:.4f}")



# ========================================
# Ghi kết quả ra file
# ========================================
output_file = "model_results.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== Kết quả mô hình trước khi cân bằng dữ liệu ===\n\n")
    for name, res in results.items():
        f.write(f"🔹 {name}\n")
        f.write(f"Accuracy: {res['acc']:.4f}\n")
        f.write(f"Training time: {res['time']:.2f}s\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, res["y_pred"]))
        f.write("\n" + "="*50 + "\n\n")

    f.write("\n=== Kết quả mô hình sau khi cân bằng dữ liệu (ROS) ===\n\n")
    for name, acc in results_bal.items():
        f.write(f"🔹 {name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("\n" + "="*50 + "\n\n")

print(f"✅ Kết quả đã được ghi ra file: {output_file}")


