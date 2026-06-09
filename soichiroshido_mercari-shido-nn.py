# --- 共通ライブラリとデータ読み込み ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras import backend as K
import warnings
import gc
warnings.filterwarnings('ignore')

# データの解凍 (Kaggle環境でのみ必要)
!apt-get install p7zip > /dev/null
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z > /dev/null
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip > /dev/null

print("データを読み込んでいます...")
# データ型を float32 に統一し、メモリを節約
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test_stg2.tsv', sep='\t')

# 全データ結合
train_len = len(train)
y_train = np.log1p(train['price']).astype(np.float32) # ターゲットの対数変換と float32 化
mercari_df = pd.concat([train.drop('price', axis=1), test], ignore_index=True)

# 欠損値処理
mercari_df['category_name'] = mercari_df['category_name'].fillna('Missing')
mercari_df['brand_name'] = mercari_df['brand_name'].fillna('Missing')
mercari_df['item_description'] = mercari_df['item_description'].fillna('No description yet')
mercari_df['name'] = mercari_df['name'].fillna('Missing')


# --- NN用カテゴリ特徴量の準備 (Label Encoding) ---
print("NN用カテゴリ特徴量を Label Encoding 中...")

# 1. カテゴリ分割 (Brand x Cat1 の交差特徴量用にも使う)
split_cats = mercari_df['category_name'].str.split('/', expand=True, n=2)
mercari_df['cat_1'] = split_cats[0].fillna('Missing')
mercari_df['cat_2'] = split_cats[1].fillna('Missing')
mercari_df['cat_3'] = split_cats[2].fillna('Missing')

# 2. Label Encoding を適用し、特徴量IDと次元数を取得
CAT_FEATS = ['brand_name', 'item_condition_id', 'shipping', 'cat_1', 'cat_2', 'cat_3']
# One-HotではなくIDとしてNNに渡すための辞書
NN_INPUT_SHAPES = {}

for col in CAT_FEATS:
    le = LabelEncoder()
    # Fit on all data, then transform to get indices (0 to N-1)
    mercari_df[col] = le.fit_transform(mercari_df[col].astype(str))
    
    # 辞書サイズ (次元数 = 最大ID + 1)
    vocab_size = mercari_df[col].max() + 1
    
    # Embedding層の出力次元 (目安: min(50, vocab_size // 10))
    embedding_dim = min(50, int(vocab_size / 10) + 1)
    
    NN_INPUT_SHAPES[col] = {'size': vocab_size, 'dim': embedding_dim}

print("【Step 0 完了】NN用データ準備完了。")


print("=== Step 1: NN向け数値特徴量とSVD特徴量の作成 ===")

# --- A. 軽量特徴量群 (Pattern 1.x) ---

# 1. Log特徴量 (テキスト長)
print("1. Log特徴量 (テキスト長) を計算中...")
name_len_log = np.log1p(mercari_df['name'].astype(str).apply(len)).astype(np.float32)
desc_len_log = np.log1p(mercari_df['item_description'].astype(str).apply(len)).astype(np.float32)

# 2. Keyword Flags
print("2. キーワードフラグを作成中...")
keywords = ['bundle', 'set', 'new', 'sealed', 'auth', 'junk', 'broken', 'missing', 'damage']
desc_lower = mercari_df['item_description'].str.lower().astype(str)
kw_flags = []
for w in keywords:
    flag = desc_lower.str.contains(w, regex=False).astype(np.float32).values
    kw_flags.append(flag)
X_kw = np.column_stack(kw_flags).astype(np.float32)

# 3. テキストの質感 (数字, 大文字率)
print("3. テキストの質感 (数字, 大文字率) を計算中...")
desc_str = mercari_df['item_description'].astype(str)
digit_log = np.log1p(desc_str.apply(lambda x: sum(c.isdigit() for c in x))).astype(np.float32)
cap_rate = desc_str.apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x)>0 else 0).astype(np.float32)

# 結合して標準化
NUM_FEATS = pd.DataFrame({
    'name_len': name_len_log,
    'desc_len': desc_len_log,
    'digit': digit_log,
    'cap': cap_rate,
})
# キーワードフラグと結合
NUM_FEATS = pd.concat([NUM_FEATS, pd.DataFrame(X_kw, columns=[f'kw_{i}' for i in range(X_kw.shape[1])])], axis=1)

# 標準化 (Scalerを学習データのみで fit)
scaler = StandardScaler()
X_num = scaler.fit_transform(NUM_FEATS.iloc[:train_len])
# テストデータにも適用
X_num_full = scaler.transform(NUM_FEATS).astype(np.float32)


# --- B. TF-IDF & SVD 特徴量 (Pattern 2) ---

print("4. 次元圧縮 (TF-IDF & SVD, 100次元) を作成中...")

# TF-IDF (Float32)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english', dtype=np.float32)
X_tfidf_tmp = tfidf.fit_transform(mercari_df['item_description'])

# SVD (100次元)
n_comp = 100
svd = TruncatedSVD(n_components=n_comp, algorithm='randomized', random_state=42)
X_svd = svd.fit_transform(X_tfidf_tmp).astype(np.float32)

# 標準化
X_svd_full = StandardScaler().fit_transform(X_svd).astype(np.float32)

del X_tfidf_tmp, tfidf, svd, X_svd, desc_lower, desc_str, NUM_FEATS
gc.collect()

print("【Step 1 完了】すべての密な数値特徴量準備完了。")


print("=== Step 2: Keras マルチインプット NN の定義 ===")

# --- 損失関数: RMSLE の代用 ---
# y_true, y_pred は log1p 変換された値として扱うため、MSE をそのまま使うのが標準的
def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# --- モデル構築 ---
def build_model(NN_INPUT_SHAPES, X_num_shape, X_svd_shape):
    
    # 1. カテゴリ入力ブランチ (Embedding)
    inputs_cat = []
    embeddings = []
    
    for col, params in NN_INPUT_SHAPES.items():
        # Input Layer (IDを直接受け取る)
        input_layer = Input(shape=(1,), name=f'input_{col}')
        
        # Embedding Layer (Sparse -> Dense)
        embedding_layer = Embedding(
            input_dim=params['size'], # 辞書サイズ
            output_dim=params['dim'], # 出力次元
            input_length=1
        )(input_layer)
        
        # Flatten (1, dim) -> (dim,)
        flatten_layer = Flatten()(embedding_layer)
        
        inputs_cat.append(input_layer)
        embeddings.append(flatten_layer)

    # 2. 数値特徴量入力ブランチ (Log/Flags/Texture)
    input_num = Input(shape=(X_num_shape[1],), name='input_numerical')
    
    # 3. SVD特徴量入力ブランチ
    input_svd = Input(shape=(X_svd_shape[1],), name='input_svd')
    
    
    # --- 結合とDense層 ---
    
    # 全てのベクトルを結合
    concat_layers = Concatenate()(embeddings + [input_num, input_svd])
    
    # 結合された層に Dropout をかけて正規化
    dense = Dropout(0.2)(concat_layers)
    
    # 隠れ層
    dense = Dense(512, activation='relu', kernel_initializer='he_normal')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(256, activation='relu', kernel_initializer='he_normal')(dense)
    dense = Dropout(0.1)(dense)
    
    # 出力層 (対数変換された価格を予測するため活性化関数なし)
    output = Dense(1, activation='linear', name='output_price')(dense)
    
    # モデル定義
    full_model = Model(inputs=inputs_cat + [input_num, input_svd], outputs=output)
    
    # コンパイル (Adamで学習)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                       loss='mse', # RMSLEの対数域でのMSE
                       metrics=[rmsle])
    
    return full_model

# モデル構築
model = build_model(NN_INPUT_SHAPES, X_num_full.shape, X_svd_full.shape)
model.summary()

print("【Step 2 完了】モデル定義完了。")


print("=== Step 3: NNのトレーニングと提出ファイル作成 ===")

# --- データの分割とNN用入力形式への変換 ---
# NNは複数の入力リストを受け取るため、データ形式を準備
def get_nn_inputs(df, X_num, X_svd):
    inputs = []
    
    # 1. カテゴリ特徴量 (ID)
    for col in CAT_FEATS:
        # NNのInput shape(1,)に合わせるため reshape
        inputs.append(df[col].values.reshape(-1, 1))
        
    # 2. 数値特徴量 (Log/Flags/Texture)
    inputs.append(X_num)
    
    # 3. SVD特徴量
    inputs.append(X_svd)
    
    return inputs

# データ分割 (Train / Validation)
X_train_full = mercari_df.iloc[:train_len]
X_train_num = X_num_full[:train_len]
X_train_svd = X_svd_full[:train_len]

# ローカル評価用の分割 (8:2)
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.2, random_state=42)
X_tr_num = X_num_full[X_tr.index]
X_val_num = X_num_full[X_val.index]
X_tr_svd = X_svd_full[X_tr.index]
X_val_svd = X_svd_full[X_val.index]


# NN入力形式に変換
inputs_tr = get_nn_inputs(X_tr, X_tr_num, X_tr_svd)
inputs_val = get_nn_inputs(X_val, X_val_num, X_val_svd)

# --- トレーニング ---
BATCH_SIZE = 2048*2 # メモリを圧迫しないサイズに設定

print(f"トレーニング開始 (BATCH_SIZE: {BATCH_SIZE}, Epochs: 20)")

history = model.fit(
    inputs_tr, 
    y_tr,
    epochs=100, # まずは5エポックで様子を見る
    batch_size=BATCH_SIZE,
    validation_data=(inputs_val, y_val),
    verbose=1
)

# --- ローカル評価 ---
preds_val = model.predict(inputs_val, batch_size=BATCH_SIZE)
score = np.sqrt(mean_squared_error(y_val, preds_val))
print(f"---------------------------------------------")
print(f"【結果確認】 Validation RMSLE: {score:.5f}")
print(f"---------------------------------------------")


# --- 提出用予測 ---
X_test = mercari_df.iloc[train_len:]
X_test_num = X_num_full[train_len:]
X_test_svd = X_svd_full[train_len:]

inputs_test = get_nn_inputs(X_test, X_test_num, X_test_svd)

print("テストデータの予測中...")
preds_test = model.predict(inputs_test, batch_size=BATCH_SIZE)
preds_test = preds_test.flatten()

# 値の補正 (Log -> 元の価格に戻す)
preds_test = np.expm1(preds_test)
preds_test = np.where(preds_test < 0, 0, preds_test) # 負の値は0にする

# ファイル保存
submission = pd.DataFrame({
    "test_id": test['test_id'],
    "price": preds_test
})
submission.to_csv("submission.csv", index=False)

print("\n【完了】 submission.csv を保存しました！")

