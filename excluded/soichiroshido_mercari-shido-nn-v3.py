# ==========================================
# 【Step 0 & 1】データ読み込みと特徴量エンジニアリング (強化版)
# ==========================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Flatten, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# データの解凍 (Kaggle環境でのみ必要)
!apt-get install p7zip > /dev/null
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z > /dev/null
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip > /dev/null

print("データを読み込んでいます...")
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test_stg2.tsv', sep='\t')

# 全データ結合
train_len = len(train)
y_train = np.log1p(train['price']).astype(np.float32)
mercari_df = pd.concat([train.drop('price', axis=1), test], ignore_index=True)

test_ids = test['test_id'].values

# 欠損値処理
mercari_df['category_name'] = mercari_df['category_name'].fillna('Missing')
mercari_df['brand_name'] = mercari_df['brand_name'].fillna('Missing')
mercari_df['item_description'] = mercari_df['item_description'].fillna('No description yet')
mercari_df['name'] = mercari_df['name'].fillna('Missing')

# --- 【改善点1】商品名と説明文を結合する ---
# 商品名(name)に含まれる "Nike", "iPhone" などの強力な単語をTF-IDFに取り込むため
print("テキストデータを結合中 (Name + Description)...")
mercari_df['all_text'] = (mercari_df['name'].astype(str) + " " + mercari_df['item_description'].astype(str)).astype(str)

# --- カテゴリ特徴量 (Label Encoding) ---
print("カテゴリ特徴量を処理中...")
split_cats = mercari_df['category_name'].str.split('/', expand=True, n=2)
mercari_df['cat_1'] = split_cats[0].fillna('Missing')
mercari_df['cat_2'] = split_cats[1].fillna('Missing')
mercari_df['cat_3'] = split_cats[2].fillna('Missing')

CAT_FEATS = ['brand_name', 'item_condition_id', 'shipping', 'cat_1', 'cat_2', 'cat_3']
NN_INPUT_SHAPES = {}
for col in CAT_FEATS:
    le = LabelEncoder()
    mercari_df[col] = le.fit_transform(mercari_df[col].astype(str))
    vocab_size = mercari_df[col].max() + 1
    embedding_dim = min(50, int(vocab_size / 10) + 1)
    NN_INPUT_SHAPES[col] = {'size': vocab_size, 'dim': embedding_dim}

# --- 数値特徴量 ---
print("数値特徴量を計算中...")
mercari_df['name_len'] = np.log1p(mercari_df['name'].str.len()).astype(np.float32)
mercari_df['desc_len'] = np.log1p(mercari_df['item_description'].str.len()).astype(np.float32)
# 簡易的な特徴量のみに絞る (メモリ節約)
X_num_full = mercari_df[['name_len', 'desc_len']].values.astype(np.float32)
scaler = StandardScaler()
X_num_full = scaler.fit_transform(X_num_full)

# --- TF-IDF (次元数を少し増やす: 5000 -> 10000) ---
# 商品名が入ったので語彙を増やして表現力を上げる
print("TF-IDF (10,000次元) を計算中...")
tfidf = TfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 2), 
    stop_words='english', 
    dtype=np.float32
)
X_tfidf_full = tfidf.fit_transform(mercari_df['all_text'])

# メモリ掃除
del train, test, split_cats, mercari_df['all_text'], mercari_df['item_description'], mercari_df['name']
gc.collect()


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


# ==========================================
# 【Step 3】学習・可視化 (変更なし)
# ==========================================
# データ分割
X_train_df = mercari_df.iloc[:train_len]
X_train_num = X_num_full[:train_len]
X_train_tfidf = X_tfidf_full[:train_len]
X_test_df = mercari_df.iloc[train_len:]
X_test_num = X_num_full[train_len:]
X_test_tfidf = X_tfidf_full[train_len:]

train_idx, val_idx = train_test_split(np.arange(train_len), test_size=0.1, random_state=42)

# Generator定義
class MercariGenerator(Sequence):
    def __init__(self, df, X_num, X_tfidf, y=None, batch_size=1024, cat_cols=None, shuffle=True):
        self.df = df
        self.X_num = X_num
        self.X_tfidf = X_tfidf
        if y is not None:
            self.y = y.values.reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)
        else:
            self.y = None
        self.batch_size = batch_size
        self.cat_cols = cat_cols
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle: np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_cat_inputs = [self.df[col].iloc[inds].values.reshape(-1, 1) for col in self.cat_cols]
        batch_num = self.X_num[inds]
        batch_tfidf = self.X_tfidf[inds].toarray()
        
        inputs = {}
        for i, col in enumerate(self.cat_cols): inputs[f'input_{col}'] = batch_cat_inputs[i]
        inputs['input_numerical'] = batch_num
        inputs['input_tfidf'] = batch_tfidf
        
        if self.y is not None:
            return inputs, self.y[inds]
        return inputs
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

BATCH_SIZE = 1024 * 2 # メモリがきつい場合は 1024 に減らしてください
gen_train = MercariGenerator(X_train_df.iloc[train_idx], X_train_num[train_idx], X_train_tfidf[train_idx], y_train[train_idx], BATCH_SIZE, CAT_FEATS, True)
gen_val = MercariGenerator(X_train_df.iloc[val_idx], X_train_num[val_idx], X_train_tfidf[val_idx], y_train[val_idx], BATCH_SIZE, CAT_FEATS, False)

# モデル構築 & 学習
tfidf_dim = X_train_tfidf.shape[1]
model = build_model(NN_INPUT_SHAPES, X_num_full.shape, tfidf_dim)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

print(f"トレーニング開始 (TF-IDF Dim: {tfidf_dim})...")
history = model.fit(
    gen_train,
    epochs=25,
    validation_data=gen_val,
    verbose=1,
    callbacks=[lr_scheduler, early_stop]
)

# 結果可視化
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Val Loss')
ax[0].set_title('Model Loss (MSE)')
ax[0].legend()
ax[1].plot(history.history['rmsle'], label='Train RMSLE')
ax[1].plot(history.history['val_rmsle'], label='Val RMSLE')
ax[1].set_title('Model RMSLE')
ax[1].legend()
plt.show()

# 予測分布確認
preds_val = model.predict(gen_val, verbose=1).flatten()
y_val_actual = y_train[val_idx].values
val_score = np.sqrt(mean_squared_error(y_val_actual, preds_val))
print(f"最終 Validation RMSLE: {val_score:.5f}")

plt.figure(figsize=(8, 8))
plt.scatter(y_val_actual, preds_val, alpha=0.1, s=2)
plt.plot([0, 10], [0, 10], color='red', linestyle='--')
plt.title(f'Actual vs Predicted / RMSLE: {val_score:.4f}')
plt.show()

# Submission
gen_test = MercariGenerator(X_test_df, X_test_num, X_test_tfidf, None, BATCH_SIZE, CAT_FEATS, False)
preds_test = model.predict(gen_test, verbose=1).flatten()
preds_test = np.expm1(preds_test)
preds_test = np.where(preds_test < 0, 0, preds_test)
submission = pd.DataFrame({"test_id": test_ids, "price": preds_test})
submission.to_csv("submission.csv", index=False)
print("完了")


# # --- 共通ライブラリとデータ読み込み ---
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Flatten
# from tensorflow.keras import backend as K
# import warnings
# import gc
# warnings.filterwarnings('ignore')

# # データの解凍 (Kaggle環境でのみ必要)
# !apt-get install p7zip > /dev/null
# !p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z > /dev/null
# !unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip > /dev/null

# print("データを読み込んでいます...")
# # データ型を float32 に統一し、メモリを節約
# train = pd.read_csv('train.tsv', sep='\t')
# test = pd.read_csv('test_stg2.tsv', sep='\t')

# # 全データ結合
# train_len = len(train)
# y_train = np.log1p(train['price']).astype(np.float32) # ターゲットの対数変換と float32 化
# mercari_df = pd.concat([train.drop('price', axis=1), test], ignore_index=True)

# # 欠損値処理
# mercari_df['category_name'] = mercari_df['category_name'].fillna('Missing')
# mercari_df['brand_name'] = mercari_df['brand_name'].fillna('Missing')
# mercari_df['item_description'] = mercari_df['item_description'].fillna('No description yet')
# mercari_df['name'] = mercari_df['name'].fillna('Missing')


# # --- NN用カテゴリ特徴量の準備 (Label Encoding) ---
# print("NN用カテゴリ特徴量を Label Encoding 中...")

# # 1. カテゴリ分割 (Brand x Cat1 の交差特徴量用にも使う)
# split_cats = mercari_df['category_name'].str.split('/', expand=True, n=2)
# mercari_df['cat_1'] = split_cats[0].fillna('Missing')
# mercari_df['cat_2'] = split_cats[1].fillna('Missing')
# mercari_df['cat_3'] = split_cats[2].fillna('Missing')

# # 2. Label Encoding を適用し、特徴量IDと次元数を取得
# CAT_FEATS = ['brand_name', 'item_condition_id', 'shipping', 'cat_1', 'cat_2', 'cat_3']
# # One-HotではなくIDとしてNNに渡すための辞書
# NN_INPUT_SHAPES = {}

# for col in CAT_FEATS:
#     le = LabelEncoder()
#     # Fit on all data, then transform to get indices (0 to N-1)
#     mercari_df[col] = le.fit_transform(mercari_df[col].astype(str))
    
#     # 辞書サイズ (次元数 = 最大ID + 1)
#     vocab_size = mercari_df[col].max() + 1
    
#     # Embedding層の出力次元 (目安: min(50, vocab_size // 10))
#     embedding_dim = min(50, int(vocab_size / 10) + 1)
    
#     NN_INPUT_SHAPES[col] = {'size': vocab_size, 'dim': embedding_dim}

# print("【Step 0 完了】NN用データ準備完了。")


# # 【Step 1】NN向け数値特徴量と高次元TF-IDF特徴量の作成
# print("=== Step 1: NN向け数値特徴量とTF-IDF特徴量(5000次元)の作成 ===")
# import gc
# from scipy.sparse import csr_matrix, hstack

# # --- A. 軽量特徴量群 (Pattern 1.x) ---
# # ※ここは元のコードと同じロジックです
# print("1. Log特徴量 & Keyword Flags & Texture を計算中...")

# # 1. Log特徴量
# name_len_log = np.log1p(mercari_df['name'].astype(str).apply(len)).astype(np.float32)
# desc_len_log = np.log1p(mercari_df['item_description'].astype(str).apply(len)).astype(np.float32)

# # 2. Keyword Flags
# keywords = ['bundle', 'set', 'new', 'sealed', 'auth', 'junk', 'broken', 'missing', 'damage']
# desc_lower = mercari_df['item_description'].str.lower().astype(str)
# kw_flags = []
# for w in keywords:
#     flag = desc_lower.str.contains(w, regex=False).astype(np.float32).values
#     kw_flags.append(flag)
# X_kw = np.column_stack(kw_flags).astype(np.float32)

# # 3. Texture
# desc_str = mercari_df['item_description'].astype(str)
# digit_log = np.log1p(desc_str.apply(lambda x: sum(c.isdigit() for c in x))).astype(np.float32)
# cap_rate = desc_str.apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x)>0 else 0).astype(np.float32)

# # 結合
# NUM_FEATS = pd.DataFrame({
#     'name_len': name_len_log,
#     'desc_len': desc_len_log,
#     'digit': digit_log,
#     'cap': cap_rate,
# })
# NUM_FEATS = pd.concat([NUM_FEATS, pd.DataFrame(X_kw, columns=[f'kw_{i}' for i in range(X_kw.shape[1])])], axis=1)

# # 標準化
# scaler = StandardScaler()
# X_num = scaler.fit_transform(NUM_FEATS.iloc[:train_len])
# X_num_full = scaler.transform(NUM_FEATS).astype(np.float32)

# # --- B. TF-IDF 特徴量 (SVDなし, 5000次元) ---
# print("2. テキスト特徴量 (TF-IDF, 5000次元) を作成中...")

# # max_features を 5000 に増加。SVDは適用しません。
# tfidf = TfidfVectorizer(
#     max_features=5000,   # SVDを使わないため、ここで次元数を担保
#     ngram_range=(1, 2), 
#     stop_words='english', 
#     dtype=np.float32
# )

# # 疎行列 (Sparse Matrix) のまま保持します (重要: ここで .toarray() するとメモリ死します)
# X_tfidf_full = tfidf.fit_transform(mercari_df['item_description'])

# print(f"TF-IDF Shape: {X_tfidf_full.shape}")

# # 不要メモリ開放
# del desc_lower, desc_str, NUM_FEATS, kw_flags, X_kw
# gc.collect()

# print("【Step 1 完了】特徴量作成完了 (TF-IDFは疎行列として保持)。")


# # ==========================================
# # 修正版 Step 2: モデル定義 (BatchNormalization追加)
# # ==========================================
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, Flatten, BatchNormalization
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# def rmsle(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))

# def build_model(NN_INPUT_SHAPES, X_num_shape, tfidf_dim=5000):
#     inputs_cat = []
#     embeddings = []
    
#     # 1. カテゴリ入力
#     for col, params in NN_INPUT_SHAPES.items():
#         input_layer = Input(shape=(1,), name=f'input_{col}')
#         embedding_layer = Embedding(params['size'], params['dim'], input_length=1)(input_layer)
#         flatten_layer = Flatten()(embedding_layer)
#         inputs_cat.append(input_layer)
#         embeddings.append(flatten_layer)

#     # 2. 数値入力
#     input_num = Input(shape=(X_num_shape[1],), name='input_numerical')
    
#     # 3. TF-IDF入力
#     input_tfidf = Input(shape=(tfidf_dim,), name='input_tfidf')
#     tfidf_feat = Dropout(0.1)(input_tfidf)
#     # TF-IDF用の圧縮層を追加 (5000 -> 256) + BN
#     tfidf_dense = Dense(256, activation='relu')(tfidf_feat)
#     tfidf_dense = BatchNormalization()(tfidf_dense)
    
#     # 結合
#     concat_layers = Concatenate()(embeddings + [input_num, tfidf_dense])
    
#     # Dense層 + BatchNormalization (これが学習安定の鍵です)
#     x = Dense(512, activation='relu', kernel_initializer='he_normal')(concat_layers)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    
#     x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
    
#     x = Dense(64, activation='relu', kernel_initializer='he_normal')(x) # 層を一つ追加
#     x = BatchNormalization()(x)
    
#     output = Dense(1, activation='linear', name='output_price')(x)
    
#     full_model = Model(inputs=inputs_cat + [input_num, input_tfidf], outputs=output)
    
#     full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), # BN入れたのでLR少し上げてOK
#                        loss='mse', 
#                        metrics=[rmsle])
#     return full_model


# # ==========================================
# # 修正版 Step 3: Generator & 学習 & 可視化
# # ==========================================
# from tensorflow.keras.utils import Sequence
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- Generator (Shape不一致修正版) ---
# class MercariGenerator(Sequence):
#     def __init__(self, df, X_num, X_tfidf, y=None, batch_size=1024, cat_cols=None, shuffle=True):
#         self.df = df
#         self.X_num = X_num
#         self.X_tfidf = X_tfidf
        
#         # 【重要修正】y を (N, 1) に reshape してブロードキャスト事故を防ぐ
#         if y is not None:
#             self.y = y.values.reshape(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else y.reshape(-1, 1)
#         else:
#             self.y = None
            
#         self.batch_size = batch_size
#         self.cat_cols = cat_cols
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.df))
#         if self.shuffle: np.random.shuffle(self.indices)
            
#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))
    
#     def __getitem__(self, idx):
#         inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
#         batch_cat_inputs = [self.df[col].iloc[inds].values.reshape(-1, 1) for col in self.cat_cols]
#         batch_num = self.X_num[inds]
#         batch_tfidf = self.X_tfidf[inds].toarray()
        
#         inputs = {}
#         for i, col in enumerate(self.cat_cols): inputs[f'input_{col}'] = batch_cat_inputs[i]
#         inputs['input_numerical'] = batch_num
#         inputs['input_tfidf'] = batch_tfidf
        
#         if self.y is not None:
#             return inputs, self.y[inds] # Shape: (Batch, 1)
#         return inputs
    
#     def on_epoch_end(self):
#         if self.shuffle: np.random.shuffle(self.indices)

# # --- Generator作成 ---
# BATCH_SIZE = 1024 * 2
# gen_train = MercariGenerator(X_train_df.iloc[train_idx], X_train_num[train_idx], X_train_tfidf[train_idx], y_train[train_idx], BATCH_SIZE, CAT_FEATS, True)
# gen_val = MercariGenerator(X_train_df.iloc[val_idx], X_train_num[val_idx], X_train_tfidf[val_idx], y_train[val_idx], BATCH_SIZE, CAT_FEATS, False)

# # --- モデル再構築 ---
# tfidf_dim = X_train_tfidf.shape[1]
# model = build_model(NN_INPUT_SHAPES, X_num_full.shape, tfidf_dim)

# # --- コールバック (学習率調整) ---
# # Lossが下がらない場合、学習率を半分にする
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5)
# early_stop = EarlyStopping(monitor='val_loss', patience=6, verbose=1, restore_best_weights=True)

# print("トレーニング開始...")
# history = model.fit(
#     gen_train,
#     epochs=25, # 多めに設定(EarlyStoppingで止まるので)
#     validation_data=gen_val,
#     verbose=1,
#     callbacks=[lr_scheduler, early_stop]
# )

# # ==========================================
# # 【可視化機能】学習結果のグラフ化
# # ==========================================
# def plot_training_history(history):
#     fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Loss Plot
#     ax[0].plot(history.history['loss'], label='Train Loss')
#     ax[0].plot(history.history['val_loss'], label='Val Loss')
#     ax[0].set_title('Model Loss (MSE)')
#     ax[0].set_xlabel('Epoch')
#     ax[0].set_ylabel('Loss')
#     ax[0].legend()
#     ax[0].grid(True)
    
#     # RMSLE Plot
#     ax[1].plot(history.history['rmsle'], label='Train RMSLE')
#     ax[1].plot(history.history['val_rmsle'], label='Val RMSLE')
#     ax[1].set_title('Model RMSLE')
#     ax[1].set_xlabel('Epoch')
#     ax[1].set_ylabel('RMSLE')
#     ax[1].legend()
#     ax[1].grid(True)
    
#     plt.tight_layout()
#     plt.show()

# plot_training_history(history)

# # ==========================================
# # 【可視化機能】予測精度散布図 (Actual vs Predicted)
# # ==========================================
# print("Validationデータの予測分布を確認中...")
# preds_val = model.predict(gen_val, verbose=1).flatten()
# y_val_actual = y_train[val_idx].values # 元データ

# # スコア計算
# val_score = np.sqrt(mean_squared_error(y_val_actual, preds_val))
# print(f"最終 Validation RMSLE: {val_score:.5f}")

# plt.figure(figsize=(8, 8))
# plt.scatter(y_val_actual, preds_val, alpha=0.1, s=2)
# plt.plot([0, 10], [0, 10], color='red', linestyle='--') # 理想線
# plt.title(f'Actual vs Predicted (Log Price) / RMSLE: {val_score:.4f}')
# plt.xlabel('Actual Price (Log)')
# plt.ylabel('Predicted Price (Log)')
# plt.grid(True)
# plt.show()

# # --- 提出用ファイル作成 ---
# print("Submissionファイル作成中...")
# gen_test = MercariGenerator(X_test_df, X_test_num, X_test_tfidf, None, BATCH_SIZE, CAT_FEATS, False)
# preds_test = model.predict(gen_test, verbose=1).flatten()
# preds_test = np.expm1(preds_test)
# preds_test = np.where(preds_test < 0, 0, preds_test)



# submission = pd.DataFrame({"test_id": test['test_id'], "price": preds_test})
# submission.to_csv("submission.csv", index=False)
# print("完了: submission.csv saved.")

