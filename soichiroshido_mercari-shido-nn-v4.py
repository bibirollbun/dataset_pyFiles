import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# 再現性確保
def seed_everything(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(42)

# 評価関数 RMSLE
def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def rmsle_score(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


# ==========================================
# 【Step 2】モデル定義 (過学習対策強化)
# ==========================================
def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model(NN_INPUT_SHAPES, X_num_shape, tfidf_dim):
    inputs_cat = []
    embeddings = []
    
    # カテゴリ
    for col, params in NN_INPUT_SHAPES.items():
        input_layer = Input(shape=(1,), name=f'input_{col}')
        # EmbeddingのDropoutを追加
        embedding_layer = Embedding(params['size'], params['dim'], input_length=1)(input_layer)
        flatten_layer = Flatten()(embedding_layer)
        inputs_cat.append(input_layer)
        embeddings.append(flatten_layer)

    # 数値
    input_num = Input(shape=(X_num_shape[1],), name='input_numerical')
    
    # TF-IDF (次元が増えたので層を深くする)
    input_tfidf = Input(shape=(tfidf_dim,), name='input_tfidf')
    
    # 入力時点でのDropout率を上げる (0.1 -> 0.2)
    tfidf_feat = Dropout(0.2)(input_tfidf) 
    
    # 圧縮層 (Dense)
    tfidf_dense = Dense(256, activation='relu', kernel_initializer='he_normal')(tfidf_feat)
    tfidf_dense = BatchNormalization()(tfidf_dense)
    tfidf_dense = Dropout(0.2)(tfidf_dense) # 追加
    
    # 全結合
    concat_layers = Concatenate()(embeddings + [input_num, tfidf_dense])
    
    # Main Dense Layers (過学習対策: Dropout率を上げ、層のサイズを調整)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(concat_layers)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x) # 0.3 -> 0.4 に強化
    
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) # 0.2 -> 0.3 に強化
    
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation='linear', name='output_price')(x)
    
    full_model = Model(inputs=inputs_cat + [input_num, input_tfidf], outputs=output)
    
    # Optimizerの学習率を少し下げる (安定化のため)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0015), 
                       loss='mse', 
                       metrics=[rmsle])
    return full_model


# データの解凍 (Kaggle環境でのみ必要)
!apt-get install p7zip > /dev/null
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z > /dev/null
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip > /dev/null

# データの読み込み (パスは環境に合わせて変更してください)
print("Loading Data...")
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test_stg2.tsv', sep='\t')

# 外れ値の除去 (価格が3ドル未満のものを除くなど)
train = train.drop(train[(train.price < 3.0)].index)

# ターゲットの対数変換
y_train = np.log1p(train['price']).values
train_ids = train['train_id'].values
test_ids = test['test_id'].values

# データ結合
nrow_train = train.shape[0]
merge: pd.DataFrame = pd.concat([train, test], sort=False).drop(['price', 'train_id', 'test_id'], axis=1)

# 欠損値処理
merge['category_name'] = merge['category_name'].fillna('missing').astype(str)
merge['brand_name'] = merge['brand_name'].fillna('missing').astype(str)
merge['item_description'] = merge['item_description'].fillna('No description yet').astype(str)
merge['item_condition_id'] = merge['item_condition_id'].astype(str)
merge['shipping'] = merge['shipping'].astype(str)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
del train, test
gc.collect()


print("Processing Text for NN (Tokenizer)...")

# テキストクリーニング関数
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"([0-9]+)000000", r"\1m", text) # 金額の省略表現など
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    # 特殊文字等の処理はKerasのTokenizerにある程度任せるが、最低限の置換を行う
    return text

merge['name'] = merge['name'].apply(preprocess_text)
merge['item_description'] = merge['item_description'].apply(preprocess_text)

# Tokenizerの学習 (NN用)
# 商品名(name)と説明文(item_description)を別々にTokenizeする
# 名前は短いので全単語、説明文は長いので頻出単語に絞るなどの戦略をとる

# Name
tok_raw_name = Tokenizer(num_words=50000) # 語彙数
tok_raw_name.fit_on_texts(merge['name'].tolist())
merge['seq_name'] = tok_raw_name.texts_to_sequences(merge['name'].tolist())

# Description
tok_raw_desc = Tokenizer(num_words=100000)
tok_raw_desc.fit_on_texts(merge['item_description'].tolist())
merge['seq_desc'] = tok_raw_desc.texts_to_sequences(merge['item_description'].tolist())

# カテゴリ変数のLabel Encoding (NN用)
le_brand = LabelEncoder()
merge['brand_name'] = le_brand.fit_transform(merge['brand_name'])

le_cat = LabelEncoder()
merge['category_name'] = le_cat.fit_transform(merge['category_name'])

le_cond = LabelEncoder()
merge['item_condition_id'] = le_cond.fit_transform(merge['item_condition_id'])

le_ship = LabelEncoder()
merge['shipping'] = le_ship.fit_transform(merge['shipping'])

print("Tokenization & Label Encoding Done.")


# パラメータ設定
MAX_NAME_SEQ = 10   # 商品名は短いので10単語
MAX_DESC_SEQ = 75   # 説明文は75単語まで見る
MAX_TEXT_VOCAB = 50000
MAX_DESC_VOCAB = 100000
MAX_BRAND_VOCAB = merge['brand_name'].max() + 1
MAX_CAT_VOCAB = merge['category_name'].max() + 1
MAX_COND_VOCAB = merge['item_condition_id'].max() + 1
MAX_SHIP_VOCAB = merge['shipping'].max() + 1

# Padding (長さを揃える)
print("Padding sequences...")
X_name_seq = pad_sequences(merge['seq_name'], maxlen=MAX_NAME_SEQ)
X_desc_seq = pad_sequences(merge['seq_desc'], maxlen=MAX_DESC_SEQ)

# NN用の入力データを辞書形式で分割
X_train_nn = {
    'name': X_name_seq[:nrow_train],
    'item_desc': X_desc_seq[:nrow_train],
    'brand': merge['brand_name'][:nrow_train].values,
    'category': merge['category_name'][:nrow_train].values,
    'item_cond': merge['item_condition_id'][:nrow_train].values,
    'shipping': merge['shipping'][:nrow_train].values,
}

X_test_nn = {
    'name': X_name_seq[nrow_train:],
    'item_desc': X_desc_seq[nrow_train:],
    'brand': merge['brand_name'][nrow_train:].values,
    'category': merge['category_name'][nrow_train:].values,
    'item_cond': merge['item_condition_id'][nrow_train:].values,
    'shipping': merge['shipping'][nrow_train:].values,
}

print("NN Data Preparation Done.")


def get_rnn_model():
    # --- Inputs ---
    name = Input(shape=[X_train_nn['name'].shape[1]], name="name")
    item_desc = Input(shape=[X_train_nn['item_desc'].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    item_cond = Input(shape=[1], name="item_cond")
    shipping = Input(shape=[1], name="shipping")

    # --- Embeddings ---
    # 商品名: 短いので重みを大きめに
    emb_name = Embedding(MAX_TEXT_VOCAB, 20)(name)
    
    # 説明文: GRUで文脈を読む
    emb_desc = Embedding(MAX_DESC_VOCAB, 60)(item_desc)
    
    # カテゴリ類
    emb_brand = Embedding(MAX_BRAND_VOCAB, 10)(brand)
    emb_cat = Embedding(MAX_CAT_VOCAB, 10)(category)
    emb_cond = Embedding(MAX_COND_VOCAB, 5)(item_cond)
    emb_ship = Embedding(MAX_SHIP_VOCAB, 5)(shipping)

    # --- RNN / CNN Layers ---
    # Name: 短いのでGRUでさらっと読む
    rnn_name = GRU(16)(emb_name)
    
    # Description: GRUで読む (ここが精度に効く)
    rnn_desc = GRU(32)(emb_desc)

    # Flatten Categorical Embeddings
    main_l = concatenate([
        Flatten()(emb_brand),
        Flatten()(emb_cat),
        Flatten()(emb_cond),
        Flatten()(emb_ship),
        rnn_name,
        rnn_desc
    ])

    # --- Dense Layers ---
    main_l = Dropout(0.1)(main_l)
    main_l = Dense(512, activation='relu')(main_l)
    main_l = Dropout(0.1)(main_l)
    main_l = Dense(256, activation='relu')(main_l)
    main_l = Dropout(0.1)(main_l)
    
    # Output
    output = Dense(1, activation="linear")(main_l)
    
    model = Model([name, item_desc, brand, category, item_cond, shipping], output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    model.compile(loss="mse", optimizer=optimizer, metrics=[rmsle])
    
    return model

model = get_rnn_model()
model.summary()


BATCH_SIZE = 1024 # 大きすぎるとRNNの学習が進みにくい場合があるので調整
EPOCHS = 3 # RNNは時間がかかるため2-3エポックで十分なことが多い

print("Training NN Model...")
# 学習データの分割 (Validation用)
X_tr, X_val, y_tr, y_val = train_test_split(
    np.arange(nrow_train), y_train, test_size=0.1, random_state=42
)

# 入力データの辞書分割関数
def get_data_split(idx):
    return {k: v[idx] for k, v in X_train_nn.items()}

model = get_rnn_model()
history = model.fit(
    get_data_split(X_tr), y_tr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(get_data_split(X_val), y_val),
    verbose=1
)

# NNの予測作成
preds_nn = model.predict(X_test_nn, batch_size=BATCH_SIZE, verbose=1)
preds_nn = preds_nn.flatten()

# Validationでのスコア確認
val_preds_nn = model.predict(get_data_split(X_val), batch_size=BATCH_SIZE, verbose=1).flatten()
print(f"NN Validation RMSLE: {rmsle_score(y_val, val_preds_nn)}")

# メモリ解放
del model, history, X_name_seq, X_desc_seq
gc.collect()


print("Preparing Data for Ridge (Sparse)...")

# Ridge用には生のテキストをベクトル化する (Bag of Words / TF-IDF)
# 注意: メモリ節約のため、fit_transform後に即座にmerge内のテキストは削除しても良い
# 名前: CountVectorizer (単語の有無が重要)
cv = CountVectorizer(min_df=10)
X_name_ridge = cv.fit_transform(merge['name'])

# 説明文: TfidfVectorizer (特徴的な単語の重み付け)
tv = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
X_desc_ridge = tv.fit_transform(merge['item_description'])

# カテゴリ系: ダミー変数化 (One-hot)
lb = LabelEncoder()
X_brand_ridge = csr_matrix(pd.get_dummies(merge['brand_name'], sparse=True).values)
X_cat_ridge = csr_matrix(pd.get_dummies(merge['category_name'], sparse=True).values)
X_cond_ridge = csr_matrix(pd.get_dummies(merge['item_condition_id'], sparse=True).values)
X_ship_ridge = csr_matrix(pd.get_dummies(merge['shipping'], sparse=True).values)

# 全結合 (Sparse Matrixとして結合)
X_ridge = hstack([X_name_ridge, X_desc_ridge, X_brand_ridge, X_cat_ridge, X_cond_ridge, X_ship_ridge]).tocsr()

# データの分割
X_train_ridge = X_ridge[:nrow_train]
X_test_ridge = X_ridge[nrow_train:]

# 不要データ削除
del X_ridge, X_name_ridge, X_desc_ridge, merge
gc.collect()

print("Training Ridge Model...")
# Ridge回帰の学習
model_ridge = Ridge(solver='sag', fit_intercept=True, random_state=42, alpha=3.0)
model_ridge.fit(X_train_ridge, y_train)

# Ridgeの予測
preds_ridge = model_ridge.predict(X_test_ridge)

# Validationでのスコア確認 (NNと同じindexを使う)
val_preds_ridge = model_ridge.predict(X_train_ridge[X_val])
print(f"Ridge Validation RMSLE: {rmsle_score(y_val, val_preds_ridge)}")


print("Ensembling...")

# 重み付け (検証データの結果を見て調整推奨だが、経験的に NN:0.6, Ridge:0.4 くらいが良い)
# NN単体で0.43-0.44、Ridge単体で0.46くらいでも、混ぜると0.41-0.42付近、
# NNが0.42までいけば、混ぜて0.40切りが見えます。

# 予測値のアンサンブル
final_preds = 0.6 * preds_nn + 0.4 * preds_ridge

# 対数を戻す (expm1)
final_preds = np.expm1(final_preds)

# 提出用データフレーム作成
submission = pd.DataFrame({
    "test_id": test_ids,
    "price": final_preds
})

# 負の値は0にする（念のため）
submission['price'] = submission['price'].apply(lambda x: 0 if x < 0 else x)

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission_ensemble.csv")

# 最終的な推定精度の表示 (Validationデータ上でのアンサンブル精度)
val_preds_ensemble = 0.6 * val_preds_nn + 0.4 * val_preds_ridge
print(f"Final Ensemble Validation RMSLE: {rmsle_score(y_val, val_preds_ensemble)}")

