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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



train_data = pd.read_csv('/kaggle/input/churn-prediction-2024/train.csv', sep=";")
test_data = pd.read_csv('/kaggle/input/churn-prediction-2024/test.csv', sep=";")


# Xem thông tin tổng quan về dữ liệu train
print(train_data.info())
print(train_data.describe())


print(train_data.isnull().sum())


# Xem thông tin tổng quan về dữ liệu test
print(test_data.info())
print(test_data.describe())


print(test_data.isnull().sum())


# Lọc các cột có kiểu số
num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

# Điền giá trị thiếu bằng trung vị của mỗi cột
train_data[num_cols] = train_data[num_cols].fillna(train_data[num_cols].median())

# Kiểm tra lại để chắc chắn không còn giá trị NaN
print(train_data[num_cols].isnull().sum().sum())


# Lọc các cột có kiểu object
cat_cols = train_data.select_dtypes(include=['object']).columns

# Điền giá trị thiếu bằng mode (giá trị phổ biến nhất) của từng cột
train_data[cat_cols] = train_data[cat_cols].fillna(train_data[cat_cols].mode().iloc[0])

# Kiểm tra lại xem còn giá trị thiếu không
print(train_data[cat_cols].isnull().sum().sum())



# Lọc các cột có kiểu số
num_cols = test_data.select_dtypes(include=['int64', 'float64']).columns

# Điền giá trị thiếu bằng trung vị của mỗi cột
test_data[num_cols] = test_data[num_cols].fillna(test_data[num_cols].median())

# Kiểm tra lại để chắc chắn không còn giá trị NaN
print(test_data[num_cols].isnull().sum().sum())


# Lọc các cột có kiểu object
cat_cols = test_data.select_dtypes(include=['object']).columns

# Điền giá trị thiếu bằng mode (giá trị phổ biến nhất) của từng cột
test_data[cat_cols] = test_data[cat_cols].fillna(test_data[cat_cols].mode().iloc[0])

# Kiểm tra lại xem còn giá trị thiếu không
print(test_data[cat_cols].isnull().sum().sum())



# Sử dụng One-Hot Encoding cho biến phân loại dùng cho cột có nhiều giá trị
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Đảm bảo hai tập train và test có cùng feature
test_data = test_data.reindex(columns=train_data.columns.drop("Churn"), fill_value=0)


# Tính ma trận tương quan
correlation_matrix = train_data.corr()

# Lấy 10 features có tương quan cao nhất với Churn
top_corr_features = correlation_matrix["Churn"].abs().sort_values(ascending=False)[1:11]
print("Top 10 features có tương quan cao với feature Churn:")
print(top_corr_features)

# Vẽ heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix[top_corr_features.index].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan của các features quan trọng")
plt.show()



selected_features = [
    "Plans_Voice mail",
    "Total day charge",
    "Customer service calls_4",
    "Customer service calls_5",
    "Customer service calls_6",
    "Plans_International, Voice mail",
    "Total eve minutes",
    "Total eve charge",
    "Total intl minutes_13.9",
    "Total charge_75.08"
]


# Lọc ra dữ liệu chỉ chứa các đặc trưng đã chọn đem đi dùng
X_train_selected = train_data[selected_features]
X_test_selected = test_data[selected_features]

# Biến mục tiêu
y_train = train_data["Churn"]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Tạo danh sách các mô hình
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Hàm đánh giá mô hình (bao gồm ROC_AUC)
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_scores = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_scores)
    except AttributeError:
        roc_auc = np.nan
        print(f"Model {model.__class__.__name__} does not have predict_proba, ROC_AUC set to NaN")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc
    }

# Chạy các mô hình và lưu kết quả
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train_scaled, y_train, X_train_scaled, y_train)
    
# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results).T
print(results_df)



plt.figure(figsize=(10, 5))
results_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.xticks(rotation=45)
plt.title("So sánh hiệu suất của các mô hình")
plt.ylabel("Score")
plt.show()



from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best Model:", best_model)

