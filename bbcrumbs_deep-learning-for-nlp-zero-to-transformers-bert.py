# Shapely 와 PyGEOS 버전 맞춰주기
!pip install --upgrade shapely pygeos


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


print(tf.__version__)


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS(device): ", strategy.num_replicas_in_sync)


train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# 기본 정보 확인
print(f"훈련데이터\n")
print(train.info())
print('---'*20)
print(f"검증 데이터\n")
print(validation.info())
print('---'*20)
print(f"테스트 데이터\n")
print(test.info())


# 문장 길이 계산
train['comment_length'] = train['comment_text'].apply(len)
print(f"문장 길이:\n{train['comment_length'].describe()}")
# 결측값 확인
print()
print(f"결측값:\n{train.isnull().sum()}")
print()
# 'toxic' 또는 목표 변수 분포 확인
print(f"target 분포:\n{train['toxic'].value_counts()}")


# 컬럼 이름 확인
print(f"column names:\n{train.columns}")
print('---'*40)
# 데이터의 첫 5줄 확인
print(f"train head:\n{train.head(30)}")
print(f"vallidation head:\n{validation.head()}")
print(f"test head:\n{test.head()}")
print('---'*40)
# 수치형 데이터의 기술 통계 확인
print(f"describe:\n{train.describe()}")


# 불필요 컬럼 삭제: 이진 레이블 예측 문제이므로 toxic에 집중하기 위해 유해 댓글의 세부 유형을 나타내는 컬럼 삭제 
train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)


# 슬라이싱
train = train.loc[:12000,:]
train.shape


# 각 댓글의 단어 수 계산 후 단어 수가 가장 많은 댓글 찾기
train['comment_text'].apply(lambda x:len(str(x).split())).max()


def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels

    Returns
    -------
    roc_auc:
        모델의 분류 성능을 평가하는 AUC 점수 반환
    '''
    # thresholds: 예측 확률에 대한 임계값. 이를 기준으로 예측이 양성 또는 음성으로 분류된다.
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, #.values가 넘파이배열 형태로 반환
                                                  stratify=train.toxic.values, # 훈련/검증 데이터 간 클래스 비율 유지
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)


# 각 문장의 단어 수(토큰 개수) 계산
comment_lengths = [len(comment.split()) for comment in xtrain]

# 히스토그램 시각화: max_length를 정하기 위해 문장 길이 분포 확인
plt.figure(figsize=(10,5))
plt.hist(comment_lengths, bins=50, edgecolor='black')
plt.xlabel('comment length(number of words)')
plt.ylabel('number of comments')
plt.title('comment length distribution')
plt.show()



# Keras Tokenizer 로 텍스트 데이터를 단어 인덱스로 변환 
# !TensorFlow 2.x 부터 최신권장문법은 tensorflow.keras (라이브러리 변경해줘야함)
token = keras.preprocessing.text.Tokenizer(num_words=None)
max_len = 1500 # 패딩할 시퀀스 길이의 기준 정의

# xtrain과 xvalid 데이터에서 나오는 모든 단어들을 기반으로 만들어진 단어 사전 학습
token.fit_on_texts(list(xtrain) + list(xvalid))
# 각 문장의 각 단어를 정수 시퀀스(인덱스 리스트)로 변환
xtrain_seq = token.texts_to_sequences(xtrain) 
xvalid_seq = token.texts_to_sequences(xvalid)

# max_len에 맞게 정수 시퀀스에 제로 패딩 추가. 기본 시퀀스의 앞에서부터 0이 붙는다.
# padding='pre' 또는 padding='post' 파라미터를 통해 시퀀스의 앞과 뒤 중 어느 위치에 패딩을 줄지 결정할 수 있다.
xtrain_pad = keras.preprocessing.sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = keras.preprocessing.sequence.pad_sequences(xvalid_seq, maxlen=max_len)

# 모델의 인풋 데이터 형식으로 변환
# word_index 는 Tokenizer 객체에서 각 단어를 고유한 정수 인덱스에 매핑하는 딕셔너리
word_index = token.word_index


xtrain_seq[:1]


# 특정 단어의 인덱스 확인
print(f"단어'love'의 인덱스:\n{word_index['love']}")
print(f"word_index의 길이:\n{len(word_index.items())}")


%%time
with strategy.scope():
    # A simpleRNN without any pretrained embeddings and one dense layer
    model = keras.models.Sequential() # sequential 모델 객체 생성
    model.add(keras.layers.Embedding(len(word_index) + 1,    # 단어 인덱스의 크기(범위)
                     300,  # 임베딩 차원:각 단어를 300차원의 실수 벡터로 변환
                     input_length=max_len)) # 입력 시퀀스 길이
    model.add(keras.layers.SimpleRNN(100))   # RNN의 출력 차원으로, 100개의 뉴런 사용
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync) #Multiplying by Strategy to run on TPU's


from sklearn.metrics import roc_auc_score

scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid))) # AUC가 높을 수록 모델의 분류 성능이 좋다. 


# 여러 모델의 AUC 점수를 비교할 수 있도록 리스트에 저장
scores_model = []
scores_model.append({'Model': 'SimpleRNN','AUC_Score': roc_auc(scores,yvalid)})


# load the GloVe vectors in a dictionary:

embeddings_index = {} # {단어(키) : 해당 단어의 300차원 벡터(numpy arrays)}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0] # 첫번째 값을 word에 저장
    coefs = np.asarray([float(val) for val in values[1:]]) # 나머지 300개의 숫자를 실수로 변환 -> numpy array 생성
    embeddings_index[word] = coefs # 딕셔너리에 단어와 벡터를 저장
f.close() # 파일 닫기 (리소스 절약)

# 임베딩된 단어 개수 출력
print('Found %s word vectors.' % len(embeddings_index))


print(embeddings_index.get("love"))


# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_matrix


%%time
with strategy.scope():
    
    # A simple LSTM with glove embeddings and one dense layer
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],  # GloVe 임베딩 로드
                     input_length=max_len,
                     trainable=False))            # 파인튜닝 미사용

    model.add(keras.layers.LSTM(100, dropout=0.3, recurrent_dropout=0.3)) 
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
model.summary()


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


scores_model.append({'Model': 'LSTM','AUC_Score': roc_auc(scores,yvalid)})


%%time
with strategy.scope():
    # GRU with glove embeddings and two dense layers
     model = keras.models.Sequential()
     model.add(keras.layers.Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
     model.add(keras.layers.SpatialDropout1D(0.3))
     model.add(keras.layers.GRU(300))
     model.add(keras.layers.Dense(1, activation='sigmoid'))

     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    
model.summary()


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


# 점수 비교를 위해 저장
scores_model.append({'Model': 'GRU','AUC_Score': roc_auc(scores,yvalid)})


scores_model


# scores_kmodel에서 모델 이름과 AUC 점수 추출
models = [model['Model'] for model in scores_model]
auc_scores = [model['AUC_Score'] for model in scores_model]

# 데이터프레임으로 변환
df = pd.DataFrame(scores_model)

# 시각화 (Seaborn을 사용한 Barplot)
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='AUC_Score', data=df, palette='Pastel1')

# 그래프 꾸미기
plt.xlabel('Model')
plt.ylabel('AUC Score')
plt.title('Comparison of Model AUC Scores')
plt.ylim(0, 1)  # AUC는 0과 1 사이의 값
plt.show()



%%time
with strategy.scope():
    # A simple bidirectional LSTM with glove embeddings and one dense layer
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    
model.summary()


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=32*strategy.num_replicas_in_sync)


model.save('/kaggle/working/my_model.h5')


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


scores_model.append({'Model': 'Bi-directional LSTM','AUC_Score': roc_auc(scores,yvalid)})


# Visualization of Results obtained from various Deep learning models
results = pd.DataFrame(scores_model).sort_values(by='AUC_Score',ascending=False)
results.style.background_gradient(cmap='Blues')


import plotly
print(plotly.__version__)


import plotly.express as px
fig = px.funnel_area(names= results.Model,
                     values= results.AUC_Score,
                     title= {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"})
fig.show(renderer='iframe')


fig = go.Figure(go.Funnelarea(
    text =results.Model,
    values = results.AUC_Score,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show(renderer='iframe') # or fig.show(renderer='iframe_connected')


# Loading Dependencies
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers

from tokenizers import BertWordPieceTokenizer


# LOADING THE DATA

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# 텍스트 데이터를 Bert 모델 입력에 적합한 정수 시퀀스로 인코딩하는 함수 
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input

    parameters
    ----------
    texts:
        인코딩할 텍스트 리스트나 배열
    tokenizer:
        토크나이저 객체
        (라이브러리 사용)
    chunk_size:
        한 번에 처리할 텍스트 청크 크기
        (기본값: 256)
    maxlen:
        최대 시퀀스 길이
        (기본값: 512)
    """
    tokenizer.enable_truncation(max_length=maxlen) # maxlen보다 길면 잘라내기
    tokenizer.enable_padding(length=maxlen)    # maxlen보다 짧으면 패딩
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()  # 메모리 효율성을 위해 텍스트를 작은 청크로 나눠 리스트 변환 
        encs = tokenizer.encode_batch(text_chunk)    # 청크 내 모든 텍스트를 한 번에 인코딩
        all_ids.extend([enc.ids for enc in encs])    # 인코딩된 각 객체에서 ID 목록을 추출해서 리스트에 추가
    
    return np.array(all_ids) # 인코딩된 ID를 NumPy 배열로 반환 [텍스트 수, maxlen] -> BERT모델의 입력 데이터로 사용


# 모델의 설정 매개변수 정의
#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192


# ipywidgets 패키지 업데이트
!pip install --upgrade ipywidgets


# jupyter 위젯 버전 충돌 오류 무시
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ipywidgets')

# 커널 재시작시 위젯 버전 다운그레이드 및 재설치
!pip uninstall -y ipywidgets
!pip install ipywidgets==7.6.5
!pip install jupyterlab-widgets==1.0.0


# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer


x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    # 최대 길이가 512인 단어 ID 입력 레이어 생성
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    # 트랜스포머(BERT) 모델의 첫번째 시퀀스 출력 
    sequence_output = transformer(input_word_ids)[0]
    # 전체 문장의 정보를 담고 있는 cls_token
    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)
    
    # 입력과 출력을 연결해 모델 생성
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# 다국어 DistillBERT 모델 기반 이진 분류 모델
%%time
with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()


n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)


sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)

