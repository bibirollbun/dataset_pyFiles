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
train=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip",header=0,\
                  delimiter="\t",quoting=3)


print("Shape of the DataFrame:")
print(train.shape) 
print("\nColumn names array:")
print(train.columns.values) 
print("\nRaw output replication:") 
print(">>> train.shape")
print(train.shape)
print("\n>>> train.columns.values")
print(train.columns.values)


print("第一条影评内容：")
print(train["review"][0]) 


!pip install beautifulsoup4
from bs4 import BeautifulSoup


first_review = train["review"][0]
example1 = BeautifulSoup(first_review, "html.parser")
print("=== 原始评论 ===")
print(first_review)
print("\n=== 清洗后的评论 ===")
print(example1.get_text())


train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", delimiter="\t", quoting=3)
import re
from nltk.corpus import stopwords
from tqdm import tqdm 
def review_to_words(raw_review):
    """
    将原始评论转换为预处理文本的函数
    参数: raw_review - 原始电影评论文本
    返回: 预处理后的字符串
    """
    if pd.isna(raw_review):
        return ""
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)
print("\n=== 批量处理所有评论 ===")
num_reviews = train["review"].size
clean_train_reviews = []
# 进度条
for i in tqdm(range(num_reviews), desc="Processing reviews"):
    clean_train_reviews.append(review_to_words(train["review"][i]))
train["clean_review"] = clean_train_reviews
#  检查结果
print("\n预处理完成！前3条结果：")
for i in range(3):
    print(f"\n评论 {i+1}:")
    print(train["clean_review"][i][:200] + "...") 


!pip install scikit-learn pandas numpy
from sklearn.feature_extraction.text import CountVectorizer
def create_bag_of_words(clean_reviews, max_features=5000):
    print("Creating the bag of words...")
    
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=max_features
    )
    
    features = vectorizer.fit_transform(clean_reviews)
    return features.toarray(), vectorizer

if __name__ == "__main__":
    print("生成示例数据...")
    clean_train_reviews = [
        "this movie was amazing really loved it",
        "the worst film ever terrible acting",
        "great performance by the lead actor",
        "boring waste of time would not recommend"
    ] * 6250
    
    train_data_features, vectorizer = create_bag_of_words(clean_train_reviews)
    
    print("\n特征矩阵形状:", train_data_features.shape)
    print("词汇表前10个特征:", list(vectorizer.get_feature_names_out())[:10])
    print("\n第一条评论的词袋特征（前20个）:")
    print(train_data_features[0][:20])



vocab = vectorizer.get_feature_names_out() 
print("词汇表（前50个）:")
print(vocab[:50])
print("\n总词汇量:", len(vocab))


import numpy as np
word_counts = np.sum(train_data_features, axis=0)

print("单词出现频率统计（前50个高频词）:")
sorted_indices = np.argsort(-word_counts)  

for i in sorted_indices[:50]:
    print(f"{word_counts[i]:<6} {vocab[i]}")


from sklearn.ensemble import RandomForestClassifier
import time
print("训练随机森林...")
start_time = time.time()
forest = RandomForestClassifier(
    n_estimators=100,  
    random_state=42,   
    n_jobs=-1         
)
forest.fit(train_data_features, train["sentiment"])
elapsed_time = time.time() - start_time
print(f"训练完成！耗时: {elapsed_time:.2f}秒")
print("\n模型参数:")
print(f"- 树的数量: {forest.n_estimators}")
print(f"- 特征重要性平均得分: {forest.feature_importances_.mean():.4f}")
if hasattr(forest, 'feature_importances_'):
    top_features = np.argsort(forest.feature_importances_)[-10:][::-1]
    print("\n最重要的10个特征:")
    for idx in top_features:
        print(f"{vocab[idx]}: {forest.feature_importances_[idx]:.4f}")


import pandas as pd
import numpy as np
from tqdm import tqdm
print("读取测试数据...")
test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", delimiter="\t", quoting=3)
print(f"测试集形状: {test.shape} (应显示25000行2列)")
print("\n清洗和解析测试集评论...")
clean_test_reviews = []
for review in tqdm(test["review"], desc="Processing"):
    clean_test_reviews.append(review_to_words(review))
print("\n生成测试集特征...")
test_data_features = vectorizer.transform(clean_test_reviews).toarray()
print("\n进行预测...")
result = forest.predict(test_data_features)
print("\n创建提交文件...")
output = pd.DataFrame({
    "id": test["id"],
    "sentiment": result
})
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

print("\n提交文件已生成: Bag_of_Words_model.csv")
print("前5条预测结果:")
print(output.head())


