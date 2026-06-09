! kaggle competitions download -c quora-question-pairs


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', None)


train_data = pd.read_csv("/content/train.csv")
train_data.head()


train_data.shape


train_data.info()


train_data["is_duplicate"].value_counts()


class_counts = train_data['is_duplicate'].value_counts()


plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title("Class Distribution (Is Duplicate)")
plt.xlabel("Class (0 = Not Duplicate, 1 = Duplicate)")
plt.ylabel("Number of Samples")
plt.show()


df_0 = train_data[train_data["is_duplicate"] == 0]
df_1 = train_data[train_data["is_duplicate"] == 1]

df_0 = df_0.head(140000)
df_1 = df_1.head(140000)

train_data = pd.concat([df_0, df_1])

train_data["is_duplicate"].value_counts()


train_data.isnull().sum()


train_data.dropna(inplace=True)


print(f"Total len of words q1 {train_data['question1'].str.len().max()}")
print(f"Total len of words q2 {train_data['question2'].str.len().max()}")


train_data['q1_length'] = train_data['question1'].apply(len)
train_data['q2_length'] = train_data['question2'].apply(len)

# Plot distributions
plt.figure(figsize=(12, 6))
sns.histplot(train_data['q1_length'], label='Question 1 Length', color='blue', kde=True, bins=30)
sns.histplot(train_data['q2_length'], label='Question 2 Length', color='green', kde=True, bins=30)
plt.title("Distribution of Question Lengths")
plt.xlabel("Length of Questions")
plt.ylabel("Frequency")
plt.legend()
plt.show()


train_data[train_data["question2"].str.len() > 650].index


train_data = train_data.drop(index = [4326, 18055, 51947, 75727, 94476, 118582, 130781, 131653, 153442, 166715, 190838, 199362])
train_data.shape


def clean_text(text):
    text = str(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

train_data['question1'] = train_data['question1'].apply(clean_text)
train_data['question2'] = train_data['question2'].apply(clean_text)


train_data.head()


questions = pd.concat([train_data["question1"], train_data["question2"]])
questions = questions.drop_duplicates().reset_index(drop=True)


X = train_data[["question1", "question2"]]
y = train_data["is_duplicate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


tfidf = TfidfVectorizer(max_features = 650)

tfidf.fit(questions)

X_train_q1 = tfidf.transform(X_train['question1'])
X_train_q2 = tfidf.transform(X_train['question2'])
X_test_q1 = tfidf.transform(X_test['question1'])
X_test_q2 = tfidf.transform(X_test['question2'])


from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between questions
tfidf_q1 = tfidf.transform(train_data['question1'])
tfidf_q2 = tfidf.transform(train_data['question2'])

cosine_sim = [cosine_similarity(q1, q2)[0][0] for q1, q2 in zip(tfidf_q1, tfidf_q2)]

# Add cosine similarity to the dataset
train_data['cosine_similarity'] = cosine_sim

# Plot cosine similarity distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_data['cosine_similarity'], bins=30, kde=True, color='purple')
plt.title("Distribution of Cosine Similarity Between Questions")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.show()


# Combine the features for both questions
X_train_combined = np.hstack((X_train_q1.toarray(), X_train_q2.toarray()))
X_test_combined = np.hstack((X_test_q1.toarray(), X_test_q2.toarray()))


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    #"Support Vector Machine": SVC(kernel='linear', probability=True),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}


results = {}
for model_name, model in models.items():
    print(f"Evaluating: {model_name}")

    pipeline = Pipeline([
        ('model', model)
    ])

    if model_name == "Naive Bayes":
        pipeline.fit(X_train_combined, y_train)
        y_pred = pipeline.predict(X_test_combined)
    else:
        pipeline.fit(X_train_combined, y_train)
        y_pred = pipeline.predict(X_test_combined)

    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

print("\nModel Accuracy Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.2f}")


import pickle

for model_name, model in models.items():
    file_name = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        print(f"{model_name} saved as {file_name}")


!wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip -q glove.6B.zip


from gensim.models.keyedvectors import KeyedVectors

def load_glove_model(glove_file_path):
    word_vectors = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

glove_file_path = "/content/glove.6B.300d.txt"
glove_model = load_glove_model(glove_file_path)

print(glove_model.get('king'))


import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

def get_glove_embeddings(question, glove_model):
    words = word_tokenize(question.lower())
    vectors = []

    for word in words:
        if word in glove_model:
            vectors.append(glove_model[word])


    if len(vectors) == 0:
        return np.zeros(300)

    return np.mean(vectors, axis=0)


X_train_glove = np.array([get_glove_embeddings(q, glove_model) for q in X_train['question1']])
X_test_glove = np.array([get_glove_embeddings(q, glove_model) for q in X_test['question1']])

X_train_glove_q2 = np.array([get_glove_embeddings(q, glove_model) for q in X_train['question2']])
X_test_glove_q2 = np.array([get_glove_embeddings(q, glove_model) for q in X_test['question2']])

X_train_combined_glove = np.hstack((X_train_glove, X_train_glove_q2))
X_test_combined_glove = np.hstack((X_test_glove, X_test_glove_q2))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


model = build_model(input_dim=X_train_combined_glove.shape[1])
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

history = model.fit(
    X_train_combined_glove,
    y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test_combined_glove, y_test),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)


history = history.history

plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.show()


plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.show()


test_data = [
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("How to cook pasta?", "What is the population of India?"),
    ("How can I improve my coding skills?", "What are the best ways to get better at coding?"),
    ("What is machine learning?", "Explain the laws of thermodynamics."),
    ("What is the best smartphone in 2025?", "Which smartphone should I buy this year?")
]


expected_results = [1, 0, 1, 0, 1]


def preprocess_text(pair, glove_model):
    text1, text2 = pair
    processed_text1 = get_glove_embeddings(text1, glove_model)
    processed_text2 = get_glove_embeddings(text2, glove_model)
    return processed_text1, processed_text2

# Prediction function
def predict_duplicate(model, pair, glove_model):
    processed_text1, processed_text2 = preprocess_text(pair, glove_model)
    combined_input = np.concatenate([processed_text1, processed_text2]).reshape(1, -1)
    prediction = model.predict(combined_input)
    return 1 if prediction >= 0.5 else 0  # Binary classification threshold


for i, pair in enumerate(test_data):
    print(f"Test Pair {i+1}: {pair}")
    prediction = predict_duplicate(model, pair, glove_model)
    print(f"Predicted: {prediction}, Expected: {expected_results[i]}\n")


test_data = pd.read_csv("/content/test.csv")
test_data.head()


test_data = test_data.head(10)
test_data.shape


results = []

# Predict for each pair in the test data
for i, row in test_data.iterrows():
    pair = (row['question1'], row['question2'])
    test_id = row['test_id']
    is_duplicate = predict_duplicate(model, pair, glove_model)
    results.append([test_id, is_duplicate])


results_df = pd.DataFrame(results, columns=['test_id', 'is_duplicate'])
results_df.to_csv('sample_submission_1.csv', index=False)

print("Results saved to Sample.csv")

