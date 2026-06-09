import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")

test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

train_df.info()

train_df.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập style đẹp
sns.set(style="whitegrid")

# Soil Type
plt.figure(figsize=(8,4))
sns.countplot(data=train_df, x="Soil Type", order=train_df["Soil Type"].value_counts().index)
plt.title("Số lượng mỗi loại đất (Soil Type)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crop Type
plt.figure(figsize=(10,4))
sns.countplot(data=train_df, x="Crop Type", order=train_df["Crop Type"].value_counts().index)
plt.title("Số lượng mỗi loại cây trồng (Crop Type)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Fertilizer Name
plt.figure(figsize=(8,4))
sns.countplot(data=train_df, x="Fertilizer Name", order=train_df["Fertilizer Name"].value_counts().index)
plt.title("Số lượng mỗi loại phân bón (Fertilizer Name)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Phân bố {col}')
    plt.xlabel(col)
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.show()



numerical_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

outlier_summary = {}

for col in numerical_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Xác định outliers
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]

    outlier_summary[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outlier Count': len(outliers),
        'Outlier Percentage': round(len(outliers) / len(train_df) * 100, 2)
    }

# Hiển thị kết quả
import pandas as pd
outlier_df = pd.DataFrame(outlier_summary).T
print(outlier_df)



train_df.info()

train_df.head()


# Bỏ cột id
train_df = train_df.drop("id", axis=1)

# Danh sách cột
numerical_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

category_cols = ["Soil Type", "Crop Type"]

# Ép kiểu cho cột số
for col in numerical_cols:
    train_df[col] = train_df[col].astype('int8')  #gán lại
    test_df[col] = train_df[col].astype('int8')

# Ép kiểu cho cột phân loại
for col in category_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = train_df[col].astype('category') #gán lại



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Chuẩn hóa các cột số
scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.fit_transform(test_df[numerical_cols])

# One-hot encoding cho các cột phân loại
train_df = pd.get_dummies(train_df, columns=category_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=category_cols, drop_first=True)

label_encoder = LabelEncoder()
train_df["Fertilizer Name"] = label_encoder.fit_transform(train_df["Fertilizer Name"])



x_train = train_df.drop("Fertilizer Name",axis=1)
y_train = train_df["Fertilizer Name"]


from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# Thiết lập lưới siêu tham số cho LightGBM
param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10, -1],  # -1 nghĩa là không giới hạn độ sâu
    'num_leaves': [31, 50],
    'boosting_type': ['gbdt'],  # Có thể thêm 'dart' nếu muốn
}

# Khởi tạo mô hình LightGBM
lgbm = LGBMClassifier()

# GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Huấn luyện
grid_search.fit(x_train, y_train)

# In kết quả tốt nhất
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)



import numpy as np
import pandas as pd

# 1. Chuẩn bị dữ liệu test
x_test = test_df.drop("id", axis=1)

# 2. Lấy model tốt nhất đã được huấn luyện
best_model = grid_search.best_estimator_

# 3. Dự đoán xác suất các class
probas = best_model.predict_proba(x_test)

# 4. Lấy top-5 class theo xác suất giảm dần
top_5_preds = np.argsort(probas, axis=1)[:, -5:][:, ::-1]

# 5. Lấy nhãn (classes_) theo thứ tự label đã được LabelEncoder fit trước đó
labels = label_encoder.classes_  # label_encoder đã fit với train_df['Fertilizer Name']

# 6. Map chỉ số → tên class (fertilizer)
top_5_labels = np.array([[labels[i] for i in row] for row in top_5_preds])

# 7. Chỉ lấy top 3
top_3_labels = top_5_labels[:, :3]

# 8. Ghép thành chuỗi space-delimited
predictions = [' '.join(row) for row in top_3_labels]

# 9. Tạo file submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': predictions
})

# 10. Ghi ra file
submission_df.to_csv('submission.csv', index=False)
print("✅ File submission.csv đã được tạo thành công.")



result = pd.read_csv('/kaggle/working/submission.csv')
result.head()

