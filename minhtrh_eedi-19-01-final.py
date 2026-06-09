#!pip install fasttext
# !pip install sentence-transformers


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer,TFBertModel,TFBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import fasttext
import spacy
import tensorflow_hub as hub
from transformers import TFAutoModel, AutoTokenizer


df_train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

display(df_train.head())
display(df_mis.head())


print("Train: (Rows, Columns) = ", df_train.shape)
print("Test: (Rows, Columns) = ", df_test.shape)
print("Misconception_mapping: (Rows, Columns) = ", df_mis.shape)


print("Number of duplicate rows in train: ", df_train.duplicated().sum())
print("Number of duplicate rows in test: ", df_test.duplicated().sum())
print("Number of duplicate rows in misconception_mapping: ", df_mis.duplicated().sum())


display(pd.DataFrame(df_train.dtypes).T)
display(pd.DataFrame(df_mis.dtypes).T)


train_missing_ratio = df_train.isna().mean()
print(train_missing_ratio[(train_missing_ratio != 0)])


test_missing_ratio = df_test.isna().mean()
print(test_missing_ratio[(test_missing_ratio != 0)])


mis_missing_ratio = df_mis.isna().mean()
print(mis_missing_ratio[(mis_missing_ratio != 0)])


rows = []
for idx, row in df_train.iterrows():
    for option in ["A", "B", "C", "D"]:
        if option == row.CorrectAnswer:
            continue

        rows.append({
            "QuestionId_Answer": f"{row.QuestionId}_{option}",
            "ConstructId": row.ConstructId,
            "ConstructName": row.ConstructName,
            "SubjectId": row.SubjectId,
            "SubjectName": row.SubjectName,
            "QuestionText": row.QuestionText,
            "CorrectAnswerText": row[f"Answer{row.CorrectAnswer}Text"],
            "IncorrectAnswerText": row[f"Answer{option}Text"],
            "MisconceptionId": row[f"Misconception{option}Id"]
        })

df_train = pd.DataFrame(rows)


print("Train: (Rows, Columns) = ", df_train.shape)
display(df_train.head())


train_missing_ratio = df_train.isna().mean()
print(train_missing_ratio[(train_missing_ratio != 0)])


df_train = df_train.dropna(subset=['MisconceptionId'])
df_train['MisconceptionId'] = df_train['MisconceptionId'].astype(int)
df_train.reset_index(drop=True, inplace=True)
df_train.shape


print("Number of unique values ​​of ConstructName: ", df_train.ConstructName.nunique())

df_train.ConstructName.value_counts().plot(kind='bar', figsize=(4, 3))
plt.title('Value Distribution of ConstructName')
plt.xlabel('ConstructName')
plt.ylabel('Count')
plt.xticks([])
plt.show()


print("Number of unique values ​​of SubjectName: ", df_train.SubjectName.nunique())

df_train.SubjectName.value_counts().plot(kind='bar', figsize=(4, 3))
plt.title('Value Distribution of SubjectName')
plt.xlabel('SubjectName')
plt.ylabel('Count')
plt.xticks([])
plt.show()


print("Number of unique values ​​of MisconceptionId (df_train): ", df_train.MisconceptionId.nunique())

df_train.MisconceptionId.value_counts().plot(kind='bar', figsize=(4, 3))
plt.title('Value Distribution of MisconceptionId (df_train)')
plt.xlabel('MisconceptionId (df_train)')
plt.ylabel('Count')
plt.xticks([])
plt.show()


df_mis.MisconceptionId.nunique()


df_train.drop(['QuestionId_Answer', 'ConstructId', 'SubjectId'], axis=1, inplace=True)


df_train['QueryText'] = 'SubjectName: ' + df_train['SubjectName'] + '\n' + \
                        'ConstructName: ' + df_train['ConstructName'] + '\n' + \
                        'QuestionText: ' + df_train['QuestionText'] + '\n' +\
                        'CorrectAnswerText: ' + df_train['CorrectAnswerText'] + '\n' + \
                        'IncorrectAnswerText: ' + df_train['IncorrectAnswerText']

df_train.drop(['ConstructName', 'SubjectName', 'QuestionText', 'CorrectAnswerText', 'IncorrectAnswerText'], axis=1, inplace=True)


def preprocess_text(x):
    x = x.lower()                           # Chuyển văn bản thành chữ thường
    x = re.sub("@\w+", '',x)                # Xóa chuỗi bắt đầu bằng @
    x = re.sub("http\w+", '',x)             # Xóa URL
    x = re.sub(r"\\\(", " ", x)             # Xóa ký tự thoát \(
    x = re.sub(r"\\\)", " ", x)             # Xóa ký tự thoát \)
    x = re.sub(r"[ ]{1,}", " ", x)          # Thay thế nhiều khoảng trắng liên tiếp thành một khoảng trắng duy nhất.
    x = re.sub(r"\.+", ".", x)              # Nhiều dấu chấm thành một dấu chấm.
    x = re.sub(r"\,+", ",", x)              # Nhiều dấu phẩy thành một dấu phẩy.
    x = re.sub(r"\times+", "\\\\times", x)  # Thay thế times bằng \times
    x=re.sub(r"\°+","degree",x)             # Thay thế ký hiệu ° bằng từ degree
    x = x.strip()                           # Xóa khoảng trắng đầu và cuối
    return x


df_train['QueryText'] = df_train['QueryText'].apply(preprocess_text)
df_mis['MisconceptionName'] = df_mis['MisconceptionName'].apply(preprocess_text)


display(df_train.head())
display(df_mis.head())


# Tải mô hình
embedding_model = SentenceTransformer('/kaggle/input/sentencetransformersallminilml6v2')


# Chuyển đổi câu thành vector nhúng
query_embeddings = embedding_model.encode(
    df_train['QueryText'].tolist(),
    batch_size=32,
    show_progress_bar=True  # Hiển thị thanh tiến trình
)

# Chuẩn hóa các vector nhúng
query_embeddings = normalize(query_embeddings)


# Chuyển đổi câu thành vector nhúng
mis_embeddings = embedding_model.encode(
    df_mis['MisconceptionName'].tolist(),
    batch_size=32,
    show_progress_bar=True  # Hiển thị thanh tiến trình
)

# Chuẩn hóa các vector nhúng
mis_embeddings = normalize(mis_embeddings)


x_train, x_valid, y_train_misId, y_valid_misId = train_test_split(
    query_embeddings,
    df_train.MisconceptionId,
    test_size=0.15,
    random_state=42
)

#y_train_emb = mis_embeddings[[np.where(df_mis.MisconceptionId == misId)[0][0] for misId in y_train_misId]]
#y_valid_emb = mis_embeddings[[np.where(df_mis.MisconceptionId == misId)[0][0] for misId in y_valid_misId]]
misId_to_index = {misId: idx for idx, misId in enumerate(df_mis['MisconceptionId'])}
y_train_emb = mis_embeddings[[misId_to_index[misId] for misId in y_train_misId]]
y_valid_emb = mis_embeddings[[misId_to_index[misId] for misId in y_valid_misId]]


print(f"x_train shape: {x_train.shape}")
print(f"x_valid shape: {x_valid.shape}")
print(f"y_train shape: {y_train_emb.shape}")
print(f"y_valid shape: {y_valid_emb.shape}")


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(8192, activation="relu"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(y_train_emb.shape[1]),
    tf.keras.layers.BatchNormalization()
])


model.compile(
    loss=tf.keras.losses.CosineSimilarity(axis=1),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=[tf.keras.metrics.CosineSimilarity(axis=1)]
)


model.fit(
    x_train, y_train_emb,
    epochs=200,
    validation_data=(x_valid, y_valid_emb),
    batch_size=32
)


history = model.history.history
print(f"Training Cosine Similarity: {history['cosine_similarity'][-1]}")
print(f"Validation Cosine Similarity: {history['val_cosine_similarity'][-1]}")


plt.figure(figsize=(10, 5))
plt.plot(history["cosine_similarity"], label="Training Cosine Similarity")
plt.plot(history["val_cosine_similarity"], label="Validation Cosine Similarity")
plt.xlabel("Epoch")
plt.ylabel("Cosine Similarity")
plt.title("Training and Validation Cosine Similarity")
plt.legend()
plt.show()


def predict(x, mis_embeddings, df_mis, k):
    y_pred_emd = model.predict(x)
    similarities = cosine_similarity(y_pred_emd, mis_embeddings)
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    top_k_preds = np.array([[df_mis.MisconceptionId[idx] for idx in sample_indices] for sample_indices in top_k_indices])
    return top_k_preds


k = 25
y_train_pred = predict(x_train, mis_embeddings, df_mis, k)
y_valid_pred = predict(x_valid, mis_embeddings, df_mis, k)


print(f"y_train_pred shape: {y_train_pred.shape}")
print(f"y_valid_pred shape: {y_valid_pred.shape}")


def mapk(predicteds, actuals):
    def apk(predicted, actual):
        k = len(predicted)
        rel = predicted == actual
        p = 1 / np.array(range(1, k+1))
        return np.sum(rel * p)

    return np.mean([apk(predicted, actual) for predicted, actual in zip(predicteds, actuals)])


print("Score for train: ", mapk(y_train_pred, y_train_misId))


print("Score for valid: ", mapk(y_valid_pred, y_valid_misId))


rows = []
for idx, row in df_test.iterrows():
    for option in ["A", "B", "C", "D"]:
        if option == row.CorrectAnswer:
            continue

        rows.append({
            "QuestionId_Answer": f"{row.QuestionId}_{option}",
            "ConstructName": row.ConstructName,
            "SubjectName": row.SubjectName,
            "QuestionText": row.QuestionText,
            "CorrectAnswerText": row[f"Answer{row.CorrectAnswer}Text"],
            "IncorrectAnswerText": row[f"Answer{option}Text"]
        })

df_test = pd.DataFrame(rows)


display(df_test.head())


df_test['QueryText'] = 'SubjectName: ' + df_test['SubjectName'] + '\n' + \
                            'ConstructName: ' + df_test['ConstructName'] + '\n' + \
                            'QuestionText: ' + df_test['QuestionText'] + '\n' +\
                            'CorrectAnswerText: ' + df_test['CorrectAnswerText'] + '\n' + \
                            'IncorrectAnswerText: ' + df_test['IncorrectAnswerText']


df_test['QueryText'] = df_test['QueryText'].apply(preprocess_text)


x_test = embedding_model.encode(
    df_test['QueryText'].tolist(),
    batch_size=32,
    show_progress_bar=True  # Hiển thị thanh tiến trình
)

x_test = normalize(x_test)
x_test.shape


y_test_pred = predict(x_test, mis_embeddings, df_mis, k)
y_test_pred.shape


df_submission = pd.DataFrame({
    'QuestionId_Answer': df_test.QuestionId_Answer,
    'MisconceptionId': y_test_pred.tolist()
})

df_submission.MisconceptionId = df_submission.MisconceptionId.apply(lambda x: ' '.join(map(str, x)))
df_submission


df_submission.to_csv("submission.csv" ,index=False)


pd.read_csv("submission.csv")

