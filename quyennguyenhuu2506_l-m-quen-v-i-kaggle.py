# Để thực thi đoạn code sau trong ô này (cell), thí sinh có thể dùng tổ hợp phím Shift+Enter/Command+Enter, 
# hoặc chọn Run->Run current cell, hoặc bấm vào nút Excute cell (►)

# Khai báo thư viên
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Liệt kê tất cả các tệp trong thư mục Input
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Thực thi câu lệnh in ra dòng chữ "Hello World!!!"
print('Hello World!!!')


# Đọc dữ liệu có trong tập train
training_data = pd.read_csv('/kaggle/input/htqtm-demo/train.csv')


# In dữ liệu tập train ra màn hình
training_data


# In dữ liệu cột Number
training_data['Number']


# Khai báo sử dụng thuật toán Logistic Regression thư viện Sklearn
from sklearn.linear_model import LogisticRegression

# Chia dữ liệu huấn luyện thành 2 phần 
X_train = training_data[['Number']]# đặc trưng đầu vào tương ứng là các số được đưa vào mô hình
y_train = training_data['Label'] # nhãn đầu ra tương ứng là các dự đoán của mô hình


# Khởi tạo mô hình
model = LogisticRegression()
# Huấn luyện mô hình
model.fit(X_train, y_train)


# Đọc dữ liệu có trong tập test
testing_data = pd.read_csv('/kaggle/input/htqtm-demo/test.csv')


# In dữ liệu tập test ra màn hình
testing_data


# Dự đoán chẵn lẻ cho các số trong tập test với mô hình vừa huấn luyện được
y_predicted =  model.predict(testing_data)


# In ra kết quả dữ đoán
y_predicted


# Nộp kết quả dự đoán lên hệ thống thông qua việc lưu kết quả dự đoán thành file submission.csv ở thư mục /kaggle/working

# Tạo bảng thống kê kết quả 
submission = {
    'ID': [i for i in range(5)], # đánh index cho các mẫu được dự đoán trong tập test theo đúng thứ tự
    'Label': y_predicted 
}
submission = pd.DataFrame(submission)

# In bảng thống kê kết quả
print(submission)

# Lưu bảng thống kê kết quả vào file submission.csv ở thư mục /kaggle/working
submission.to_csv('submission.csv', index=False)


# Nộp bài thi như trong hướng dẫn đã công bố




