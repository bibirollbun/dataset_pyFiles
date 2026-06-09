# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 套件匯入
import kagglehub
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import string
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import os
import kagglehub
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer


# import kagglehub
# kagglehub.login()

# llm_detect_ai_generated_text_path = kagglehub.competition_download('llm-detect-ai-generated-text')
# alejopaullier_daigt_external_dataset_path = kagglehub.dataset_download('alejopaullier/daigt-external-dataset')

# # Print the path to the downloaded dataset to verify it's correct
# print(alejopaullier_daigt_external_dataset_path)

# # Update the path in pd.read_csv to use the downloaded path
# extra_train = pd.read_csv(alejopaullier_daigt_external_dataset_path + "/daigt_external_dataset.csv")



# # 下載所有需要的資料集
# llm_detect_ai_generated_text_path = kagglehub.competition_download('llm-detect-ai-generated-text')
# alejopaullier_daigt_external_dataset_path = kagglehub.dataset_download('alejopaullier/daigt-external-dataset')
# narsil_daigt_misc_path = kagglehub.dataset_download('narsil/daigt-misc')
# darraghdog_hello_claude_1000_essays_from_anthropic_path = kagglehub.dataset_download('darraghdog/hello-claude-1000-essays-from-anthropic')
# thedrcat_daigt_proper_train_dataset_path = kagglehub.dataset_download('thedrcat/daigt-proper-train-dataset')
# kingki19_llm_generated_essay_using_palm_from_google_gen_ai_path = kagglehub.dataset_download('kingki19/llm-generated-essay-using-palm-from-google-gen-ai')
# carlmcbrideellis_llm_7_prompt_training_dataset_path = kagglehub.dataset_download('carlmcbrideellis/llm-7-prompt-training-dataset')
# thedrcat_daigt_v2_train_dataset_path = kagglehub.dataset_download('thedrcat/daigt-v2-train-dataset')
# jdragonxherrera_augmented_data_for_llm_detect_ai_generated_text_path = kagglehub.dataset_download('jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text')
# nbroad_daigt_data_llama_70b_and_falcon180b_path = kagglehub.dataset_download('nbroad/daigt-data-llama-70b-and-falcon180b')
# carlmcbrideellis_llm_mistral_7b_instruct_texts_path = kagglehub.dataset_download('carlmcbrideellis/llm-mistral-7b-instruct-texts')
# datafan07_daigt_gemini_pro_8_5k_essays_path = kagglehub.dataset_download('datafan07/daigt-gemini-pro-8-5k-essays')
# dardodel_4k_mixtral87b_crafted_essays_for_detect_ai_comp_path = kagglehub.dataset_download('dardodel/4k-mixtral87b-crafted-essays-for-detect-ai-comp')
# # 7-se7en-prompts 資料集
# Sentence_Transformers_from_Huggingface = kagglehub.dataset_download('erenakbulut/sentence-transformers')
# DAIGT_External_Train_Dataset = kagglehub.dataset_download("thedrcat/daigt-external-train-dataset")

# print('Data source import complete.')


# ─── 4. 讀取原始 extra_train ───
# extra_train = pd.read_csv(os.path.join(alejopaullier_daigt_external_dataset_path, "daigt_external_dataset.csv"))
# test = pd.read_csv(os.path.join(llm_detect_ai_generated_text_path, "test_essays.csv"))

# extra_train = pd.read_csv(os.path.join(alejopaullier_daigt_external_dataset_path, "daigt_external_dataset.csv"))
# test = pd.read_csv('sample_data/test_essays.csv')
extra_train = pd.read_csv("/kaggle/input/daigt-external-train-dataset/train_external_drcat_02.csv")
test = pd.read_csv("/kaggle/input/llm_detect_ai_generated_text_path/test_essays.csv")


# ─── 5. 建立各種來源的 extra_train_new ───
extra_train_1 = pd.read_csv("/kaggle/input/llm-mistral-7b-instruct-texts/Mistral7B_CME_v7.csv").drop(['prompt_id','prompt_name'], axis=1)
extra_train_2 = pd.read_csv("/kaggle/input/hello-claude-1000-essays-from-anthropic/persuade15_claude_instant1.csv").drop(['prompt_id','essay_title'],axis=1).rename(columns = {'essay_text': 'text'})
extra_train_2['generated'] = 1
extra_train_3 = pd.read_csv("/kaggle/input/daigt-data-llama-70b-and-falcon180b/llama_falcon_v3.csv").drop(['prompt_name','model'], axis=1)
extra_train_4 = pd.read_csv("/kaggle/input/llm-generated-essay-using-palm-from-google-gen-ai/LLM_generated_essay_PaLM.csv").drop(['id','prompt_id'], axis=1)

extra_train_5 = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv").rename(columns = {'label': 'generated'}).drop(['prompt_name','source', 'RDizzl3_seven'], axis=1)

extra_train_5 = extra_train_5[extra_train_5['generated'] == 0]
extra_train_5 = extra_train_5.sample(n=18149)

extra_train_6 = pd.read_csv("/kaggle/input/4k-mixtral87b-crafted-essays-for-detect-ai-comp/Mixtral8x7b_4k_essays_for_DetectAIGeneratedTextCompetition.csv").rename(columns = {'AI_Essay': 'text'}).drop(['model_Raw_output','student_id','prompt_id','index'], axis=1)
extra_train_6['generated'] = 1

extra_train_7 = pd.read_csv("/kaggle/input/daigt-proper-train-dataset/train_drcat_01.csv").rename(columns = {'label': 'generated'}).drop(['source', 'fold'], axis=1)
extra_train_8 = pd.read_csv("/kaggle/input/daigt-proper-train-dataset/train_drcat_02.csv").rename(columns = {'label': 'generated'}).drop(['essay_id','source', 'prompt', 'fold'], axis=1)
extra_train_9 = pd.read_csv("/kaggle/input/daigt-proper-train-dataset/train_drcat_03.csv").rename(columns = {'label': 'generated'}).drop(['essay_id','source', 'prompt', 'fold'], axis=1)
extra_train_10 = pd.read_csv("/kaggle/input/daigt-proper-train-dataset/train_drcat_04.csv").rename(columns = {'label': 'generated'}).drop(['essay_id','source', 'prompt', 'fold'], axis=1)

# extra_train_1 = pd.read_csv(os.path.join(carlmcbrideellis_llm_mistral_7b_instruct_texts_path, "Mistral7B_CME_v7.csv")).drop(['prompt_id','prompt_name'], axis=1)
# extra_train_2 = pd.read_csv(os.path.join(darraghdog_hello_claude_1000_essays_from_anthropic_path, "persuade15_claude_instant1.csv")).drop(['prompt_id','essay_title'],axis=1).rename(columns = {'essay_text': 'text'})
# extra_train_2['generated'] = 1

# extra_train_3 = pd.read_csv(os.path.join(nbroad_daigt_data_llama_70b_and_falcon180b_path, "llama_falcon_v3.csv")).drop(['prompt_name','model'], axis=1)

# extra_train_4 = pd.read_csv(os.path.join(kingki19_llm_generated_essay_using_palm_from_google_gen_ai_path, "LLM_generated_essay_PaLM.csv")).drop(['id','prompt_id'], axis=1)

# extra_train_5 = pd.read_csv(os.path.join(thedrcat_daigt_v2_train_dataset_path, "train_v2_drcat_02.csv")).rename(columns = {'label': 'generated'}).drop(['prompt_name','source', 'RDizzl3_seven'], axis=1)
# extra_train_5 = extra_train_5[extra_train_5['generated'] == 0]
# extra_train_5 = extra_train_5.sample(n=18149)

# extra_train_6 = pd.read_csv(os.path.join(dardodel_4k_mixtral87b_crafted_essays_for_detect_ai_comp_path, "Mixtral8x7b_4k_essays_for_DetectAIGeneratedTextCompetition.csv")).rename(columns = {'AI_Essay': 'text'}).drop(['model_Raw_output','student_id','prompt_id','index'], axis=1)
# extra_train_6['generated'] = 1

# extra_train_7 = pd.read_csv(os.path.join(thedrcat_daigt_proper_train_dataset_path, "train_drcat_01.csv")).rename(columns = {'label': 'generated'}).drop(['source', 'fold'], axis=1)
# extra_train_8 = pd.read_csv(os.path.join(thedrcat_daigt_proper_train_dataset_path, "train_drcat_02.csv")).rename(columns = {'label': 'generated'}).drop(['essay_id','source', 'prompt', 'fold'], axis=1)
# extra_train_9 = pd.read_csv(os.path.join(thedrcat_daigt_proper_train_dataset_path, "train_drcat_03.csv")).rename(columns = {'label': 'generated'}).drop(['essay_id','source', 'prompt', 'fold'], axis=1)
# extra_train_10 = pd.read_csv(os.path.join(thedrcat_daigt_proper_train_dataset_path, "train_drcat_04.csv")).rename(columns = {'label': 'generated'}).drop(['essay_id','source', 'prompt', 'fold'], axis=1)


extra_train_new = pd.concat([extra_train_1, extra_train_2, extra_train_3, extra_train_4, extra_train_5, extra_train_6, extra_train_7, extra_train_8, extra_train_9, extra_train_10], axis=0)


extra_train_student_generated = extra_train[['text']]
extra_train_student_generated['generated'] = 0
extra_train_ai_generated = extra_train[['source_text']].rename(columns = {'source_text':'text'})
extra_train_ai_generated['generated'] = 1
extra_train_nf = pd.concat([extra_train_student_generated, extra_train_ai_generated, extra_train_new], axis=0)
extra_train_f = extra_train_nf.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
print(f"最終資料筆數：{len(extra_train_f)}")
extra_train_f


# Create a temporary dataframe with counts of each category
count_df = extra_train_f['generated'].value_counts().reset_index()
count_df.columns = ['generated', 'count']

fig = px.bar(
    count_df,
    x='generated',
    y='count',
    title='Distribution of Generated Label',
    color=['#2E86AB', '#E84545'],
    color_discrete_map="identity"
)

# Customize layout for value display
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=[0, 1])
)

# Display values on top of the bars
fig.update_traces(
    texttemplate='%{y}',
    textposition='outside',
)

fig.show()


# ─── 8. 簡單視覺化：文字長度分佈 ───
print(extra_train_f['generated'].value_counts())

# 計算每筆文字的詞數長度
extra_train_f['seq_len'] = extra_train_f['text'].apply(lambda x: len(str(x).split()))

# 分組
ai_data = extra_train_f[extra_train_f['generated'] == 1.0]['seq_len']
human_data = extra_train_f[extra_train_f['generated'] == 0.0]['seq_len']

# 畫圖
plt.figure(figsize=(10, 6))
plt.hist(human_data, bins=30, alpha=0.6, label='Human', color='blue')
plt.hist(ai_data, bins=30, alpha=0.6, label='AI-generated', color='orange')

plt.title('Text Sequence Length Distribution by Label')
plt.xlabel('Sequence Length (word count)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()



# 箱型圖
extra_train_f['essay_length'] = extra_train_f['text'].apply(lambda x: len(str(x).split()))

extra_train_f['source'] = extra_train_f['generated'].map({0.0: 'Humen', 1.0: 'LLM'})

plt.figure(figsize=(10, 6))
sns.boxplot(x='source', y='essay_length', data=extra_train_f, palette=['#2980B9', '#E74C3C'])

plt.title('Comparison of Essay Lengths by Source')
plt.xlabel('Essay Source')
plt.ylabel('Essay Length (Word Count)')
plt.grid(True)
plt.show()


# Human
human_texts = extra_train_f[extra_train_f['generated'] == 0.0]['text']

all_words = ' '.join(human_texts.astype(str)).lower().split()

word_counts = Counter(all_words)
common_words = word_counts.most_common(20)  # 取前20個常見詞

common_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=common_df, x='count', y='word', palette='Blues_d')
plt.title('Top 20 Most Common Words in Humen-Written Essays')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.grid(True)
plt.show()



# AI
ai_texts = extra_train_f[extra_train_f['generated'] == 1.0]['text']

all_words = ' '.join(ai_texts.astype(str)).lower().split()

word_counts = Counter(all_words)
common_words = word_counts.most_common(20)  # 取前20個常見詞

common_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=common_df, x='count', y='word', palette='Blues_d')
plt.title('Top 20 Most Common Words in AI-Written Essays')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.grid(True)
plt.show()


# 去除stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Human
human_texts = extra_train_f[extra_train_f['generated'] == 0.0]['text']
all_words = ' '.join(human_texts.astype(str)).lower().split()
filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words]

word_counts = Counter(filtered_words)
common_words = word_counts.most_common(20)  # 前20個

common_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=common_df, x='count', y='word', palette='Blues_d')
plt.title('Top 20 Most Common Words in Humen-Written Essays (Excluding Stopwords)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.grid(True)
plt.show()


# 去除stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# AI
ai_texts = extra_train_f[extra_train_f['generated'] == 1.0]['text']
all_words = ' '.join(ai_texts.astype(str)).lower().split()
filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words]

word_counts = Counter(filtered_words)
common_words = word_counts.most_common(20)  # 前20個

common_df = pd.DataFrame(common_words, columns=['word', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=common_df, x='count', y='word', palette='Blues_d')
plt.title('Top 20 Most Common Words in AI-Written Essays (Excluding Stopwords)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.grid(True)
plt.show()


# Human
human_texts = extra_train_f[extra_train_f['generated'] == 0.0]['text']

all_text = ' '.join(human_texts.astype(str)).lower()

filtered_words = ' '.join([
    word for word in all_text.split()
    if word.isalpha() and word not in stop_words
])

# WordCloud
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(filtered_words)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Humen-Written Essays (Excluding Stopwords)", fontsize=16)
plt.show()


# AI
ai_texts = extra_train_f[extra_train_f['generated'] == 1.0]['text']

all_text = ' '.join(ai_texts.astype(str)).lower()

filtered_words = ' '.join([
    word for word in all_text.split()
    if word.isalpha() and word not in stop_words
])

# WordCloud
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(filtered_words)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of AI-Written Essays (Excluding Stopwords)", fontsize=16)
plt.show()


# 標點符號使用頻率分析
punctuation_marks = ['.', ',', '!', '?']

for mark in punctuation_marks:
    extra_train_f[f'punct_{mark}'] = extra_train_f['text'].apply(
        lambda x: str(x).count(mark) / len(str(x)) if len(str(x)) > 0 else 0
    )

extra_train_f['source'] = extra_train_f['generated'].map({0.0: 'Human', 1.0: 'LLM'})

# boxplot
plt.figure(figsize=(14, 10))
for i, mark in enumerate(punctuation_marks):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=extra_train_f, x='source', y=f'punct_{mark}', palette='Set2')
    plt.title(f"Usage of '{mark}' by Source")
    plt.xlabel('')
    plt.ylabel('Frequency per Character')

plt.suptitle("Punctuation Usage Frequency: Human vs LLM", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# TF-IDF
human_texts = extra_train_f[extra_train_f['generated'] == 0.0]['text'].astype(str)
ai_texts = extra_train_f[extra_train_f['generated'] == 1.0]['text'].astype(str)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

human_tfidf = vectorizer.fit_transform(human_texts)
ai_tfidf = vectorizer.fit_transform(ai_texts)

words = vectorizer.get_feature_names_out()

# 平均TF-IDF權重
human_avg = human_tfidf.mean(axis=0).A1
ai_avg = ai_tfidf.mean(axis=0).A1

# top 10
human_top10 = pd.DataFrame({'word': words, 'score': human_avg}).nlargest(10, 'score')
ai_top10 = pd.DataFrame({'word': words, 'score': ai_avg}).nlargest(10, 'score')

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(data=human_top10, x='score', y='word', palette='Blues_d')
plt.title("Top 10 TF-IDF Keywords in Human Essays")
plt.xlabel("TF-IDF Score")
plt.ylabel("Keyword")

plt.subplot(1, 2, 2)
sns.barplot(data=ai_top10, x='score', y='word', palette='Oranges_d')
plt.title("Top 10 TF-IDF Keywords in LLM Essays")
plt.xlabel("TF-IDF Score")
plt.ylabel("Keyword")

plt.tight_layout()
plt.show()


nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

sample_df = extra_train_f[['text', 'generated']].sample(n=3000, random_state=42).copy()

# 分析情感分數
sample_df['sentiment'] = sample_df['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

sample_df['source'] = sample_df['generated'].map({0: 'Human', 1: 'LLM'})

plt.figure(figsize=(10, 6))
sns.boxplot(data=sample_df, x='source', y='sentiment', palette='coolwarm')
plt.title('Sentiment Score Distribution: Human vs LLM')
plt.xlabel('Essay Source')
plt.ylabel('Compound Sentiment Score (-1 to 1)')
plt.grid(True)
plt.show()


#train_final = pd.concat([extra_train_f, train.drop(columns=['prompt_id', 'id'], axis=0)], axis=0).reset_index(drop=True).drop_duplicates()
train_final = extra_train_f
train_final


print(train_final['generated'].value_counts())


train_df, val_df = train_test_split(train_final, test_size=0.3, random_state=222)

batch_size = 32

raw_train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df['text'].values, train_df['generated'].values)
).batch(batch_size)

raw_val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df['text'].values, val_df['generated'].values)
).batch(batch_size)


max_features = 75000
embedding_dim = 64
sequence_length = 512

def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


# Text vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_features,
    ngrams = (3,5),
    output_mode="int",
    output_sequence_length=sequence_length,
    pad_to_max_tokens=True
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)


from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras import Model, Input

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


inputs = Input(shape=(sequence_length,), dtype="int64")
x = Embedding(max_features, embedding_dim)(inputs)
x = Bidirectional(LSTM(32, return_sequences=True))(x)
transformer_block = TransformerBlock(embedding_dim, 2, 32)
x = transformer_block(x, training=False)
x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid", name="predictions")(x)

model = Model(inputs=inputs, outputs=predictions)
model.summary()


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])

epochs = 1
model.fit(train_ds, validation_data=val_ds, epochs=epochs)


test_text = test['text'].values
vectorized_test_text = vectorize_layer(test_text)
predictions = model.predict(vectorized_test_text)
test['generated'] = predictions
print(test[['text', 'generated']])


# test[['id', 'generated']].set_index('id').to_csv("submission.csv")
test[['id', 'generated']].to_csv("submission.csv")


sample_sub = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")


sample_sub["generated"] = predictions.squeeze()   # 或 predictions[:, 0]


sample_sub.to_csv("submission.csv", index=False)


print(sample_sub.head())


print(test)

