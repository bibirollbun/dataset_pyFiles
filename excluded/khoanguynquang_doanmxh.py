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


# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import os
import warnings
warnings.filterwarnings('ignore')

# Thiết lập hiển thị đồ thị
plt.style.use('ggplot')
sns.set(style='whitegrid')
%matplotlib inline


# Đường dẫn đến dữ liệu trên Kaggle
data_path = "../input/quora-insincere-questions-classification/"

# Kiểm tra các file có sẵn
print("Các file có sẵn:")
for file in os.listdir(data_path):
    print(f"- {file}")

# Đọc dữ liệu
train_df = pd.read_csv(f"{data_path}train.csv")
test_df = pd.read_csv(f"{data_path}test.csv")

# Tải các tài nguyên NLTK cần thiết
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Xem kích thước của dữ liệu
print(f"Kích thước dữ liệu train: {train_df.shape}")
print(f"Kích thước dữ liệu test: {test_df.shape}")

# Hiển thị một vài dòng đầu tiên
print("\nMột vài dòng đầu tiên trong tập train:")
display(train_df.head())

# Kiểm tra thông tin cột
print("\nThông tin về các cột trong tập train:")
train_df.info()

# Kiểm tra các giá trị bị thiếu
print("\nSố lượng giá trị bị thiếu trong tập train:")
print(train_df.isnull().sum())

# Kiểm tra phân phối nhãn
print("\nPhân phối nhãn trong tập train:")
label_dist = train_df['target'].value_counts()
print(label_dist)
print(f"Tỷ lệ câu hỏi không chân thành: {train_df['target'].mean()*100:.2f}%")

# Vẽ biểu đồ phân phối nhãn
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=train_df)
plt.title('Phân phối nhãn trong tập train')
plt.xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')
plt.ylabel('Số lượng')
for i, count in enumerate(label_dist.values):
    plt.text(i, count + 1000, f"{count} ({count/len(train_df)*100:.2f}%)", ha='center')
plt.show()


# Thêm các cột độ dài
train_df['char_length'] = train_df['question_text'].apply(len)
train_df['word_count'] = train_df['question_text'].apply(lambda x: len(str(x).split()))

# Tính toán thống kê cho độ dài
print("Thống kê về độ dài câu hỏi theo nhãn:")
length_stats = train_df.groupby('target')[['char_length', 'word_count']].agg(['mean', 'median', 'min', 'max']).round(2)
display(length_stats)

# Vẽ biểu đồ phân phối độ dài
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ phân phối số ký tự
sns.histplot(data=train_df, x='char_length', hue='target', bins=50, kde=True, ax=ax[0])
ax[0].set_title('Phân phối độ dài câu hỏi (số ký tự)')
ax[0].set_xlabel('Số ký tự')
ax[0].set_ylabel('Số lượng câu hỏi')
ax[0].set_xlim(0, 300)

# Biểu đồ phân phối số từ
sns.histplot(data=train_df, x='word_count', hue='target', bins=50, kde=True, ax=ax[1])
ax[1].set_title('Phân phối độ dài câu hỏi (số từ)')
ax[1].set_xlabel('Số từ')
ax[1].set_ylabel('Số lượng câu hỏi')
ax[1].set_xlim(0, 50)

plt.tight_layout()
plt.show()

# Boxplot so sánh độ dài theo nhãn
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(x='target', y='char_length', data=train_df, ax=ax[0])
ax[0].set_title('So sánh độ dài câu hỏi (số ký tự) theo nhãn')
ax[0].set_xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')
ax[0].set_ylabel('Số ký tự')

sns.boxplot(x='target', y='word_count', data=train_df, ax=ax[1])
ax[1].set_title('So sánh độ dài câu hỏi (số từ) theo nhãn')
ax[1].set_xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')
ax[1].set_ylabel('Số từ')

plt.tight_layout()
plt.show()


# Thêm các đặc trưng văn bản
train_df['question_marks'] = train_df['question_text'].apply(lambda x: x.count('?'))
train_df['exclamation_marks'] = train_df['question_text'].apply(lambda x: x.count('!'))
train_df['uppercase_count'] = train_df['question_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
train_df['uppercase_ratio'] = train_df['uppercase_count'] / train_df['char_length'].apply(lambda x: max(x, 1))

# Vẽ biểu đồ so sánh các đặc trưng văn bản theo nhãn
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.boxplot(x='target', y='question_marks', data=train_df, ax=axes[0, 0])
axes[0, 0].set_title('Số lượng dấu hỏi theo nhãn')
axes[0, 0].set_xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')

sns.boxplot(x='target', y='exclamation_marks', data=train_df, ax=axes[0, 1])
axes[0, 1].set_title('Số lượng dấu chấm than theo nhãn')
axes[0, 1].set_xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')

sns.boxplot(x='target', y='uppercase_count', data=train_df, ax=axes[1, 0])
axes[1, 0].set_title('Số lượng chữ in hoa theo nhãn')
axes[1, 0].set_xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')

sns.boxplot(x='target', y='uppercase_ratio', data=train_df, ax=axes[1, 1])
axes[1, 1].set_title('Tỷ lệ chữ in hoa theo nhãn')
axes[1, 1].set_xlabel('Nhãn (0: Chân thành, 1: Không chân thành)')

plt.tight_layout()
plt.show()

# Thống kê đặc trưng văn bản theo nhãn
features_stats = train_df.groupby('target')[['question_marks', 'exclamation_marks', 'uppercase_count', 'uppercase_ratio']].agg(['mean', 'median']).round(3)
print("Thống kê các đặc trưng văn bản theo nhãn:")
display(features_stats)


# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt
    text = re.sub(r'[^\w\s]', '', text)
    # Tách từ
    words = text.split()
    # Loại bỏ stopwords
    words = [word for word in words if word not in stop_words]
    return words

# Phân tách dữ liệu theo nhãn
sincere_questions = train_df[train_df['target'] == 0]['question_text']
insincere_questions = train_df[train_df['target'] == 1]['question_text']

# Lấy tất cả các từ
sincere_words = []
for question in sincere_questions:
    sincere_words.extend(preprocess_text(question))
    
insincere_words = []
for question in insincere_questions:
    insincere_words.extend(preprocess_text(question))

# Đếm các từ phổ biến
sincere_word_counts = Counter(sincere_words).most_common(20)
insincere_word_counts = Counter(insincere_words).most_common(20)

# Vẽ biểu đồ các từ phổ biến
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Từ phổ biến trong câu hỏi chân thành
sincere_df = pd.DataFrame(sincere_word_counts, columns=['word', 'count'])
sns.barplot(x='count', y='word', data=sincere_df, ax=ax[0])
ax[0].set_title('20 từ phổ biến nhất trong câu hỏi chân thành')
ax[0].set_xlabel('Số lần xuất hiện')

# Từ phổ biến trong câu hỏi không chân thành
insincere_df = pd.DataFrame(insincere_word_counts, columns=['word', 'count'])
sns.barplot(x='count', y='word', data=insincere_df, ax=ax[1])
ax[1].set_title('20 từ phổ biến nhất trong câu hỏi không chân thành')
ax[1].set_xlabel('Số lần xuất hiện')

plt.tight_layout()
plt.show()

# Tạo WordCloud
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# WordCloud cho câu hỏi chân thành
sincere_text = ' '.join(sincere_words)
wordcloud_sincere = WordCloud(width=800, height=400, background_color='white', 
                             max_words=200, contour_width=3, contour_color='steelblue')
wordcloud_sincere.generate(sincere_text)
ax[0].imshow(wordcloud_sincere, interpolation='bilinear')
ax[0].set_title('WordCloud - Câu hỏi chân thành', fontsize=15)
ax[0].axis('off')

# WordCloud cho câu hỏi không chân thành
insincere_text = ' '.join(insincere_words)
wordcloud_insincere = WordCloud(width=800, height=400, background_color='black',
                              max_words=200, contour_width=3, contour_color='firebrick', colormap='Reds')
wordcloud_insincere.generate(insincere_text)
ax[1].imshow(wordcloud_insincere, interpolation='bilinear')
ax[1].set_title('WordCloud - Câu hỏi không chân thành', fontsize=15)
ax[1].axis('off')

plt.tight_layout()
plt.show()

