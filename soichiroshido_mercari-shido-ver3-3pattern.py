# --- 共通ライブラリとデータ読み込み ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
# import tensorflow as tf
# import tensorflow_hub as hub
import warnings
import gc
warnings.filterwarnings('ignore')

# データの解凍
!apt-get install p7zip > /dev/null
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z > /dev/null
!unzip -o /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip > /dev/null

print("データを読み込んでいます...")
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test_stg2.tsv', sep='\t')

# 全データ結合
train_len = len(train)
y_train = np.log1p(train['price']) # ターゲットの対数変換
mercari_df = pd.concat([train.drop('price', axis=1), test], ignore_index=True)

# 欠損値処理
mercari_df['category_name'] = mercari_df['category_name'].fillna('Missing')
mercari_df['brand_name'] = mercari_df['brand_name'].fillna('Missing')
mercari_df['item_description'] = mercari_df['item_description'].fillna('No description yet')

# --- 基本特徴量の作成 (ベースライン) ---
print("基本特徴量を作成中...")

# 1. 商品名 (BoW)
cv = CountVectorizer(min_df=10)
X_name = cv.fit_transform(mercari_df['name'])

# 2. カテゴリ・ブランド・状態 (OneHot)
lb = OneHotEncoder(handle_unknown='ignore')
X_brand = lb.fit_transform(mercari_df[['brand_name']])
X_condition = lb.fit_transform(mercari_df[['item_condition_id']])
X_shipping = lb.fit_transform(mercari_df[['shipping']])
X_cat = lb.fit_transform(mercari_df[['category_name']])

# 基本の特徴量セット（これを毎回使い回します）
base_features = [X_name, X_brand, X_condition, X_shipping, X_cat]

print("【Step 0 完了】準備完了。")


def train_and_submit(additional_features, model_name):
    """
    引数:
      additional_features: 追加する特徴量 (csr_matrix または None)
      model_name: 保存するファイル名の識別子 (例: "ridge_log")
    """
    print(f"\n=== Model: {model_name} の実行を開始します ===")
    
    # 1. 特徴量の結合
    if additional_features is not None:
        # 基本特徴量 + 追加特徴量
        current_features = base_features + [additional_features]
    else:
        current_features = base_features
        
    # メモリ効率のために csr_matrix に変換して結合
    X_full = hstack(current_features).tocsr()
    print(f"全特徴量の形状: {X_full.shape}")

    # 2. データの分割
    X_train = X_full[:train_len]
    X_test = X_full[train_len:]
    
    # ローカル評価用の分割 (8:2)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 3. モデル学習 (ローカル評価)
    print("ローカル検証用モデルを学習中...")
    model = Ridge(solver='lsqr', fit_intercept=True)
    model.fit(X_tr, y_tr)
    
    # 4. 結果確認 (RMSLE)
    preds_val = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, preds_val))
    print(f"---------------------------------------------")
    print(f"【結果確認】 Validation RMSLE: {score:.5f}")
    print(f"---------------------------------------------")
    
    # 5. 全データでの再学習 (提出用)
    print("提出用に全データで再学習中...")
    model.fit(X_train, y_train)
    
    # 6. 提出ファイルの作成
    preds_test = model.predict(X_test)
    preds_test = np.expm1(preds_test) # logをもに戻す
    preds_test = np.where(preds_test < 0, 0, preds_test) # 負の値を0に補正

    filename = f"submission.csv"
    submission = pd.DataFrame({
        "test_id": test['test_id'],
        "price": preds_test
    })
    submission.to_csv(filename, index=False)
    print(f"【完了】 提出用ファイル '{filename}' を保存しました。")


# print("--- Pattern 1特徴量作成: Log (テキスト長) ---")

# # テキスト長の特徴量作成
# name_len_log = np.log1p(mercari_df['name'].astype(str).apply(len))
# desc_len_log = np.log1p(mercari_df['item_description'].astype(str).apply(len))

# # 標準化 & 疎行列化
# scaler = StandardScaler()
# X_len = scaler.fit_transform(pd.concat([name_len_log, desc_len_log], axis=1))
# X_len = csr_matrix(X_len)

# # ★ 共通関数を呼び出すだけ！
# train_and_submit(X_len, "ridge_log")


# print("=== Pattern 1.5: Keyword Flags (重要単語フラグ) ===")

# # 1. 調べたいキーワードのリスト
# keywords = [
#     'bundle', 'set', 'lot',           # まとめ売り系 (高くなる傾向)
#     'new', 'sealed', 'tags',          # 新品系
#     'auth', 'authentic',              # 正規品 (ブランド物で重要)
#     'sign', 'autograph',              # サイン入り
#     'free shipping',                  # 送料込み (出品者負担の場合の補正)
#     'broken', 'junk', 'parts',        # ジャンク系 (安くなる傾向)
#     'missing', 'damage'               # 欠品・破損
# ]

# # 2. フラグ作成 (商品説明に含まれているか？)
# # 処理を高速化するため、全て小文字にしてから検索します
# desc_lower = mercari_df['item_description'].str.lower().astype(str)

# keyword_flags = []
# for word in keywords:
#     # 含まれていれば 1, なければ 0
#     col = desc_lower.str.contains(word, regex=False).astype(int)
#     keyword_flags.append(col.values)

# # 3. 行列に変換
# X_keywords = np.column_stack(keyword_flags)
# X_keywords = csr_matrix(X_keywords)

# # ★ 実行
# train_and_submit(X_keywords, "ridge_pattern1_5_keywords")


# print("--- カテゴリ名の分割処理 (cat_1, cat_2, cat_3 の作成) ---")

# # category_name を '/' で分割して新しい列を作る
# # 例: "Men/Tops/T-shirts" -> cat_1="Men", cat_2="Tops", cat_3="T-shirts"
# split_cats = mercari_df['category_name'].str.split('/', expand=True, n=2)

# # 列名を割り当て
# mercari_df['cat_1'] = split_cats[0]
# mercari_df['cat_2'] = split_cats[1]
# mercari_df['cat_3'] = split_cats[2]

# # 欠損値を "Missing" で埋める
# mercari_df['cat_1'] = mercari_df['cat_1'].fillna('Missing')
# mercari_df['cat_2'] = mercari_df['cat_2'].fillna('Missing')
# mercari_df['cat_3'] = mercari_df['cat_3'].fillna('Missing')

# print("ready OK")

# print("=== Pattern 1.6: 交差特徴量 (Brand x Category) ===")
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.sparse import hstack, csr_matrix

# # 1. ブランドとカテゴリ(大分類)を文字列結合する
# # 例: "Nike" + "_" + "Men" -> "Nike_Men"
# interaction_col = mercari_df['brand_name'].astype(str) + "_" + mercari_df['cat_1'].astype(str)

# # 2. CountVectorizerでベクトル化
# # min_df=10: 出現回数が10回未満のレアな組み合わせは無視（メモリ節約）
# cv_interaction = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=10)
# X_interaction = cv_interaction.fit_transform(interaction_col)

# print(f"交差特徴量の次元数: {X_interaction.shape[1]}")

# # ★ 共通関数で実行 (train_and_submitが定義済みであること)
# train_and_submit(X_interaction, "ridge_pattern1_6_interaction")


# print("=== Pattern 1.7: Text Structure (テキストの質感) ===")

# # データ準備
# desc_str = mercari_df['item_description'].astype(str)

# # 1. 数字の数 (例: "16GB", "2 pcs")
# digit_count = desc_str.apply(lambda x: sum(c.isdigit() for c in x))
# digit_count_log = np.log1p(digit_count) # Log変換

# # 2. 大文字の割合 (叫び度合い)
# def cap_ratio(s):
#     if len(s) == 0: return 0
#     return sum(c.isupper() for c in s) / len(s)

# cap_rate = desc_str.apply(cap_ratio)

# # 3. 記号の数 (! や $)
# # 記号が多い＝アピールが強い、またはスパム的
# non_alnum_count = desc_str.apply(lambda x: sum(not c.isalnum() for c in x))
# non_alnum_log = np.log1p(non_alnum_count)

# # 4. 結合して標準化
# feats = pd.DataFrame({
#     'digit_log': digit_count_log,
#     'cap_rate': cap_rate,
#     'symbol_log': non_alnum_log
# })

# scaler = StandardScaler()
# X_texture = scaler.fit_transform(feats)
# X_texture = csr_matrix(X_texture)

# # ★ 実行
# train_and_submit(X_texture, "ridge_pattern1_7_texture")


# print("=== Pattern 1.x 統合: 軽量チューニング全部盛り (Lightweight Enhanced) ===")

# features_lightweight = []

# # ---------------------------------------------------------
# # 1. 【Pattern 1】Log特徴量 (テキスト長)
# # ---------------------------------------------------------
# print("1. テキスト長 (Log) を計算中...")
# name_len_log = np.log1p(mercari_df['name'].astype(str).apply(len))
# desc_len_log = np.log1p(mercari_df['item_description'].astype(str).apply(len))

# # 標準化してリストに追加
# X_len = StandardScaler().fit_transform(pd.concat([name_len_log, desc_len_log], axis=1))
# features_lightweight.append(csr_matrix(X_len))


# # ---------------------------------------------------------
# # 2. 【Pattern 1.5】Keyword Flags (重要単語)
# # ---------------------------------------------------------
# print("2. キーワードフラグ (New/Bundle/Junk...) を作成中...")
# # 価格に大きく影響する単語リスト
# keywords = [
#     'bundle', 'set', 'lot',           # まとめ売り (高)
#     'new', 'sealed', 'tags',          # 新品 (高)
#     'auth', 'authentic',              # 正規品 (高)
#     'sign', 'autograph',              # サイン入り (高)
#     'broken', 'junk', 'parts',        # ジャンク (安)
#     'missing', 'damage',              # 欠品破損 (安)
#     'free shipping'                   # 送料込み
# ]

# # 高速化のため小文字にして一括検索
# desc_lower = mercari_df['item_description'].str.lower().astype(str)
# kw_flags = []
# for w in keywords:
#     # 含まれていれば 1, なければ 0
#     flag = desc_lower.str.contains(w, regex=False).astype(int).values
#     kw_flags.append(flag)

# X_kw = np.column_stack(kw_flags)
# features_lightweight.append(csr_matrix(X_kw))


# # ---------------------------------------------------------
# # 3. 【Pattern 1.6】交差特徴量 (Brand × Category)
# # ---------------------------------------------------------
# # category_name を '/' で分割して新しい列を作る
# # 例: "Men/Tops/T-shirts" -> cat_1="Men", cat_2="Tops", cat_3="T-shirts"
# split_cats = mercari_df['category_name'].str.split('/', expand=True, n=2)

# # 列名を割り当て
# mercari_df['cat_1'] = split_cats[0]
# mercari_df['cat_2'] = split_cats[1]
# mercari_df['cat_3'] = split_cats[2]

# # 欠損値を "Missing" で埋める
# mercari_df['cat_1'] = mercari_df['cat_1'].fillna('Missing')
# mercari_df['cat_2'] = mercari_df['cat_2'].fillna('Missing')
# mercari_df['cat_3'] = mercari_df['cat_3'].fillna('Missing')

# print("ready OK")

# print("3. 交差特徴量 (Brand_Category) を作成中...")
# # "Nike" + "_" + "Men's Shoes" のような結合カテゴリを作る
# interaction_col = mercari_df['brand_name'].astype(str) + "_" + mercari_df['cat_1'].astype(str)

# # 出現回数が少ない組み合わせは無視 (min_df=10) してベクトル化
# cv_inter = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=10)
# X_inter = cv_inter.fit_transform(interaction_col)

# features_lightweight.append(X_inter)


# # ---------------------------------------------------------
# # 4. 【Pattern 1.7】テキストの質感 (Text Texture)
# # ---------------------------------------------------------
# print("4. テキストの質感 (数字, 大文字率) を計算中...")
# desc_str = mercari_df['item_description'].astype(str)

# # 数字の数 (スペックや個数アピール)
# digit_log = np.log1p(desc_str.apply(lambda x: sum(c.isdigit() for c in x)))

# # 大文字の割合 (強調/叫び)
# def get_cap_rate(s):
#     if len(s) == 0: return 0
#     return sum(c.isupper() for c in s) / len(s)

# cap_rate = desc_str.apply(get_cap_rate)

# # 記号の数 (! や $)
# symbol_log = np.log1p(desc_str.apply(lambda x: sum(not c.isalnum() for c in x)))

# # まとめて標準化
# texture_df = pd.DataFrame({
#     'digit': digit_log,
#     'cap': cap_rate,
#     'symbol': symbol_log
# })
# X_texture = StandardScaler().fit_transform(texture_df)
# features_lightweight.append(csr_matrix(X_texture))


# # ---------------------------------------------------------
# # 5. 結合して実行
# # ---------------------------------------------------------
# print("5. 全て結合して学習を開始します...")

# # 作成した軽量特徴量をすべて横に結合
# X_lightweight_all = hstack(features_lightweight).tocsr()

# # 共通関数を呼び出して、学習・評価・提出ファイル作成
# # 出力ファイル名: submission_ridge_pattern1_enhanced.csv
# train_and_submit(X_lightweight_all, "ridge_pattern1_enhanced")


# print("=== Pattern 2 (CPU版): 次元圧縮 (TF-IDF & SVD) ===")

# # ライブラリのインポート (全て sklearn / CPU用)
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# from scipy.sparse import csr_matrix
# import gc

# # 1. TF-IDF ベクトル化
# print("1. TF-IDF計算中 (CPU)...")
# # CPU処理なので、max_features は欲張らず 50,000 程度にします
# tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english')
# X_tfidf = tfidf.fit_transform(mercari_df['item_description'])

# # 2. 次元圧縮 (SVD)
# print("2. SVDで次元圧縮中 (CPU)...")
# # algorithm='randomized' を指定すると CPU でも比較的計算が速いです
# n_comp = 100
# svd = TruncatedSVD(n_components=n_comp, algorithm='randomized', random_state=42)
# X_svd = svd.fit_transform(X_tfidf)

# # メモリ解放 (TF-IDFの巨大な疎行列はもう不要なので消す)
# del X_tfidf
# gc.collect()

# # 3. 標準化 & 疎行列化
# print("3. 標準化処理中...")
# scaler = StandardScaler()
# X_svd = scaler.fit_transform(X_svd)
# X_svd = csr_matrix(X_svd)

# # ★ 共通関数で学習・提出
# # ファイル名: submission_ridge_svd_cpu.csv
# train_and_submit(X_svd, "ridge_svd_cpu")

# # 上の記述だとこんなエラーが出ます
# # ---------------------------------------------------------------------------
# # NotImplementedError                       Traceback (most recent call last)
# # /tmp/ipykernel_48/2378096999.py in <cell line: 0>()
# #      17 # algorithm='jacobi' はGPUで非常に高速かつ正確です
# #      18 gpu_svd = cuSVD(n_components=n_comp, algorithm='jacobi', random_state=42)
# # ---> 19 X_svd_gpu = gpu_svd.fit_transform(X_tfidf)
# #      20 
# #      21 # 結果はGPUメモリ(cupy配列)にあるので、CPU(numpy)に戻して標準化

# # /usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
# #     191 
# #     192                     if process_return:
# # --> 193                         ret = func(*args, **kwargs)
# #     194                     else:
# #     195                         return func(*args, **kwargs)

# # /usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in dispatch(self, *args, **kwargs)
# #     414         if hasattr(self, "dispatch_func"):
# #     415             func_name = gpu_func.__name__
# # --> 416             return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
# #     417         else:
# #     418             return gpu_func(self, *args, **kwargs)

# # /usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
# #     193                         ret = func(*args, **kwargs)
# #     194                     else:
# # --> 195                         return func(*args, **kwargs)
# #     196 
# #     197                 return cm.process_return(ret)

# # base.pyx in cuml.internals.base.UniversalBase.dispatch_func()

# # base.pyx in cuml.internals.base.UniversalBase._dispatch_selector()

# # NotImplementedError: Estimator does not support sparse inputs currently


# print("=== Pattern 3: USE (GPU・メモリ完全攻略版) ===")

# # ---------------------------------------------------------
# # 1. ライブラリ & 設定
# # ---------------------------------------------------------
# !pip install "protobuf<=3.20" --force-reinstall > /dev/null 2>&1

# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import IncrementalPCA
# import gc
# import math

# # GPU設定
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # ---------------------------------------------------------
# # 2. モデルロード & 準備
# # ---------------------------------------------------------
# print("USEモデルロード中...")
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# embed = hub.load(module_url)

# # 次元圧縮の設定 (512次元 -> 64次元)
# # これによりメモリ使用量を約 1/8 に削減します
# n_components = 64
# ipca = IncrementalPCA(n_components=n_components)

# text_data = mercari_df['item_description'].astype(str).tolist()
# total_len = len(text_data)
# BATCH_SIZE = 2048 # GPUなら大きくても大丈夫ですが、不安なら1024へ

# # ---------------------------------------------------------
# # 3. Step 1: PCAの学習 (データの一部を使う)
# # ---------------------------------------------------------
# print("Step 1: 次元圧縮モデルの学習中 (最初の数バッチのみ使用)...")
# # 全データで学習すると時間がかかるため、最初の数万件でPCAの傾向を学習させます
# fit_limit = min(total_len, 50000) 
# steps_fit = math.ceil(fit_limit / BATCH_SIZE)

# for i in range(steps_fit):
#     batch_texts = text_data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
#     emb = embed(batch_texts).numpy()
#     ipca.partial_fit(emb) # 部分学習

# print("PCA学習完了。")

# # ---------------------------------------------------------
# # 4. Step 2: 全データの変換と蓄積
# # ---------------------------------------------------------
# print("Step 2: 全データのベクトル化 & 圧縮を開始...")

# compressed_embeddings = []
# steps_total = math.ceil(total_len / BATCH_SIZE)

# for i in range(steps_total):
#     start = i * BATCH_SIZE
#     end = start + BATCH_SIZE
    
#     # USEでベクトル化 (512次元)
#     emb_512 = embed(text_data[start:end]).numpy()
    
#     # すぐにPCAで圧縮 (64次元)
#     emb_64 = ipca.transform(emb_512)
    
#     # 圧縮した小さいデータだけをリストに追加
#     compressed_embeddings.append(emb_64)
    
#     if i % 100 == 0:
#         print(f"Processed: {i}/{steps_total} batches")

# # メモリ解放
# del text_data
# gc.collect()

# # ---------------------------------------------------------
# # 5. 結合・標準化・提出
# # ---------------------------------------------------------
# print("データの結合中...")
# # ここでメモリに乗るのは圧縮されたデータだけなので安全です
# X_use = np.vstack(compressed_embeddings)

# print(f"完了形状: {X_use.shape} (約 {X_use.nbytes / 1024**3:.2f} GB)")

# # リスト削除
# del compressed_embeddings
# gc.collect()

# print("標準化 & 疎行列化...")
# scaler = StandardScaler()
# X_use = scaler.fit_transform(X_use)
# X_use = csr_matrix(X_use)

# # ★ 共通関数で学習・提出
# train_and_submit(X_use, "ridge_use_pca64")


# print("=== Pattern 3: USE (軽量検証版: Trainデータの50%のみ使用) ===")

# # ---------------------------------------------------------
# # 1. ライブラリ & 設定
# # ---------------------------------------------------------
# !pip install "protobuf<=3.20" --force-reinstall > /dev/null 2>&1

# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import IncrementalPCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error
# import gc
# import math

# # GPU設定
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # ---------------------------------------------------------
# # 2. データサンプリング (★ここが変更点)
# # ---------------------------------------------------------
# print("検証用にデータをサンプリングします...")

# # 学習データ部分だけを取り出す (テストデータはメモリ圧迫の原因なので今回は無視)
# # train_len は Step 0 で定義されている前提
# train_df_subset = mercari_df.iloc[:train_len].copy()

# # ターゲットも合わせる
# train_y_subset = y_train.copy()

# # ランダムに50%に間引く (frac=0.5)
# # ※メモリが厳しい場合は 0.1 (10%) などに下げてください
# sample_frac = 0.5
# print(f"元データ数: {len(train_df_subset)}")

# # インデックスをランダムに取得
# sampled_indices = train_df_subset.sample(frac=sample_frac, random_state=42).index

# # データとターゲットを抽出
# df_sample = train_df_subset.loc[sampled_indices].reset_index(drop=True)
# y_sample = train_y_subset.loc[sampled_indices].reset_index(drop=True)

# print(f"サンプリング後データ数: {len(df_sample)}")

# # ---------------------------------------------------------
# # 3. モデルロード & 準備
# # ---------------------------------------------------------
# print("USEモデルロード中...")
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# embed = hub.load(module_url)

# # 次元圧縮の設定 (512次元 -> 64次元)
# n_components = 64
# ipca = IncrementalPCA(n_components=n_components)

# # テキストリスト作成
# text_data = df_sample['item_description'].astype(str).tolist()
# total_len_sample = len(text_data)
# BATCH_SIZE = 2048 

# # ---------------------------------------------------------
# # 4. Step 1: PCAの学習
# # ---------------------------------------------------------
# print("Step 1: 次元圧縮モデルの学習中...")
# # サンプルデータの一部を使ってPCAを学習
# fit_limit = min(total_len_sample, 20000) 
# steps_fit = math.ceil(fit_limit / BATCH_SIZE)

# for i in range(steps_fit):
#     batch_texts = text_data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
#     emb = embed(batch_texts).numpy()
#     ipca.partial_fit(emb)

# print("PCA学習完了。")

# # ---------------------------------------------------------
# # 5. Step 2: 変換と蓄積
# # ---------------------------------------------------------
# print("Step 2: ベクトル化 & 圧縮を開始...")

# compressed_embeddings = []
# steps_total = math.ceil(total_len_sample / BATCH_SIZE)

# for i in range(steps_total):
#     start = i * BATCH_SIZE
#     end = start + BATCH_SIZE
    
#     emb_512 = embed(text_data[start:end]).numpy()
#     emb_64 = ipca.transform(emb_512)
#     compressed_embeddings.append(emb_64)
    
#     if i % 10 == 0: # 頻繁にログ出し
#         print(f"Processed: {i}/{steps_total} batches")

# del text_data
# gc.collect()

# # ---------------------------------------------------------
# # 6. 学習と評価 (train_and_submitは使わずここで実行)
# # ---------------------------------------------------------
# print("配列結合中...")
# X_use = np.vstack(compressed_embeddings)

# print("標準化中...")
# scaler = StandardScaler()
# X_use = scaler.fit_transform(X_use)

# print(f"学習用データ形状: {X_use.shape}")

# # Train / Valid に分割 (8:2)
# X_tr, X_val, y_tr, y_val = train_test_split(X_use, y_sample, test_size=0.2, random_state=42)

# print("Ridge回帰で学習中...")
# model = Ridge(solver='lsqr', fit_intercept=True)
# model.fit(X_tr, y_tr)

# # 評価
# preds_val = model.predict(X_val)
# score = np.sqrt(mean_squared_error(y_val, preds_val))

# print(f"---------------------------------------------")
# print(f"【USE (Train 50% only)】 Validation RMSLE: {score:.5f}")
# print(f"---------------------------------------------")
# print("※これはテストデータを含まない検証実験のため、submission.csvは作成されません。")


# print("=== Pattern 3+Base: USEと基本特徴量の合体検証 ===")

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from scipy.sparse import hstack, csr_matrix

# # 1. 基本特徴量の作成 (df_sample に対して作り直します)
# print("基本特徴量 (カテゴリ・ブランド等) を作成中...")

# # 商品名 (BoW)
# cv_name = CountVectorizer(min_df=10)
# X_name_sample = cv_name.fit_transform(df_sample['name'])

# # カテゴリ・ブランド・状態 (OneHot)
# # ※未知のカテゴリ等は無視する設定
# lb_sample = OneHotEncoder(handle_unknown='ignore')
# X_brand_sample = lb_sample.fit_transform(df_sample[['brand_name']])
# X_cond_sample = lb_sample.fit_transform(df_sample[['item_condition_id']])
# X_ship_sample = lb_sample.fit_transform(df_sample[['shipping']])
# X_cat_sample = lb_sample.fit_transform(df_sample[['category_name']])

# # 2. 特徴量の結合 (Base + USE)
# print("特徴量を結合中 (Base + USE)...")
# # X_use はさっき作成したものがメモリに残っています
# X_full_sample = hstack([
#     X_name_sample, 
#     X_brand_sample, 
#     X_cond_sample, 
#     X_ship_sample, 
#     X_cat_sample, 
#     X_use  # ここにUSEを追加！
# ]).tocsr()

# print(f"結合後の形状: {X_full_sample.shape}")

# # 3. 再学習と評価
# print("Ridge回帰で結合モデルを評価中...")
# # さっきと同じ分割で検証
# X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
#     X_full_sample, y_sample, test_size=0.2, random_state=42
# )

# model_full = Ridge(solver='lsqr', fit_intercept=True)
# model_full.fit(X_tr_full, y_tr_full)

# preds_val_full = model_full.predict(X_val_full)
# score_full = np.sqrt(mean_squared_error(y_val_full, preds_val_full))

# print(f"---------------------------------------------")
# print(f"【Base + USE】 Validation RMSLE: {score_full:.5f}")
# print(f"---------------------------------------------")


print("=== Final: 全データ処理 & 提出ファイル作成 (Base + USE) ===")

# ---------------------------------------------------------
# 1. ライブラリ & データ読み込み
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
import gc
import math

# GPU設定
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("データを読み込んでいます...")
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test_stg2.tsv', sep='\t')

train_len = len(train)
y_train = np.log1p(train['price']) # 目的変数

# 全データを結合 (これをしないとTest側の特徴量が作れません)
mercari_df = pd.concat([train.drop('price', axis=1), test], ignore_index=True)

# 欠損値埋め
mercari_df['category_name'] = mercari_df['category_name'].fillna('Missing')
mercari_df['brand_name'] = mercari_df['brand_name'].fillna('Missing')
mercari_df['item_description'] = mercari_df['item_description'].fillna('No description yet')

print(f"全データ数: {len(mercari_df)}")

# ---------------------------------------------------------
# 2. USE (Universal Sentence Encoder) 特徴量作成
# ---------------------------------------------------------
print("\n[USE] モデルロード中...")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

# PCA設定 (64次元)
n_components = 64
ipca = IncrementalPCA(n_components=n_components)

text_data = mercari_df['item_description'].astype(str).tolist()
total_len = len(text_data)
BATCH_SIZE = 2048 

# --- PCAの学習 (一部のデータのみで傾向を掴む) ---
print("[USE] PCA学習中 (Subset)...")
fit_limit = min(total_len, 50000) 
steps_fit = math.ceil(fit_limit / BATCH_SIZE)

for i in range(steps_fit):
    batch_texts = text_data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
    emb = embed(batch_texts).numpy()
    ipca.partial_fit(emb)

# --- 全データの変換 ---
print("[USE] 全データのベクトル化 & 圧縮を開始 (時間がかかります)...")
compressed_embeddings = []
steps_total = math.ceil(total_len / BATCH_SIZE)

for i in range(steps_total):
    start = i * BATCH_SIZE
    end = start + BATCH_SIZE
    
    # ベクトル化 -> 圧縮
    emb_512 = embed(text_data[start:end]).numpy()
    emb_64 = ipca.transform(emb_512)
    compressed_embeddings.append(emb_64)
    
    if i % 100 == 0:
        print(f"Processed: {i}/{steps_total} batches")

# メモリ解放
del text_data
gc.collect()

# 結合 & 標準化
X_use = np.vstack(compressed_embeddings)
scaler = StandardScaler()
X_use = scaler.fit_transform(X_use)
X_use = csr_matrix(X_use) # 疎行列化

print(f"[USE] 完了。形状: {X_use.shape}")

# ---------------------------------------------------------
# 3. 基本特徴量 (Base Features) 作成
# ---------------------------------------------------------
print("\n[Base] 基本特徴量を作成中...")

# 商品名 (BoW)
cv = CountVectorizer(min_df=10)
X_name = cv.fit_transform(mercari_df['name'])

# カテゴリ・ブランド・状態 (OneHot)
lb = OneHotEncoder(handle_unknown='ignore')
X_brand = lb.fit_transform(mercari_df[['brand_name']])
X_condition = lb.fit_transform(mercari_df[['item_condition_id']])
X_shipping = lb.fit_transform(mercari_df[['shipping']])
X_cat = lb.fit_transform(mercari_df[['category_name']])

# ---------------------------------------------------------
# 4. 全結合 & 学習 & 提出
# ---------------------------------------------------------
print("\n[Final] 特徴量結合中...")
# Base特徴量 + USE特徴量
X_full = hstack([X_name, X_brand, X_condition, X_shipping, X_cat, X_use]).tocsr()

print(f"全特徴量の形状: {X_full.shape}")

# Train / Test 分割
X_train_final = X_full[:train_len]
X_test_final = X_full[train_len:]

print("全データでRidgeモデルを学習中...")
model = Ridge(solver='lsqr', fit_intercept=True)
model.fit(X_train_final, y_train)

print("テストデータの予測中...")
preds = model.predict(X_test_final)

# 値の補正 (Log -> 元の価格に戻す)
preds = np.expm1(preds)
preds = np.where(preds < 0, 0, preds) # 負の値は0にする

# ファイル保存
submission = pd.DataFrame({
    "test_id": test['test_id'],
    "price": preds
})
submission.to_csv("submission.csv", index=False)

print("\n【完了】 submission.csv を保存しました！")


# print("=== 修正・メモリ対策版: 軽量特徴量 + SVD (Float32) ===")

# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix, hstack
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# import gc

# features_combined = []

# # ---------------------------------------------------------
# # 0. カテゴリ名の分割 (必須)
# # ---------------------------------------------------------
# print("0. カテゴリ名を分割中...")
# # メモリ節約のため、必要な列だけ処理
# split_cats = mercari_df['category_name'].str.split('/', expand=True, n=2)
# mercari_df['cat_1'] = split_cats[0].fillna('Missing')
# # cat_2, cat_3 は今回使わないのでDataFrameに代入せずメモリ節約
# del split_cats
# gc.collect()

# # ---------------------------------------------------------
# # 1. 【Pattern 1.x】軽量特徴量群 (Float32化)
# # ---------------------------------------------------------
# print("1. 軽量特徴量を作成中...")

# # (A) Log特徴量
# # ★対策: dtype=np.float32 を指定
# name_len_log = np.log1p(mercari_df['name'].astype(str).apply(len)).astype(np.float32)
# desc_len_log = np.log1p(mercari_df['item_description'].astype(str).apply(len)).astype(np.float32)
# X_len = StandardScaler().fit_transform(pd.concat([name_len_log, desc_len_log], axis=1))
# features_combined.append(csr_matrix(X_len, dtype=np.float32))

# # (B) Keyword Flags
# keywords = ['bundle', 'set', 'new', 'sealed', 'auth', 'junk', 'broken', 'missing', 'damage']
# desc_lower = mercari_df['item_description'].str.lower().astype(str)
# kw_flags = []
# for w in keywords:
#     flag = desc_lower.str.contains(w, regex=False).astype(np.float32).values
#     kw_flags.append(flag)
# X_kw = csr_matrix(np.column_stack(kw_flags), dtype=np.float32)
# features_combined.append(X_kw)

# # (C) 交差特徴量
# # ★対策: min_df を 10 -> 30 に上げて列数を削減
# print("   交差特徴量 (min_df=30)...")
# interaction_col = mercari_df['brand_name'].astype(str) + "_" + mercari_df['cat_1'].astype(str)
# cv_inter = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=30, dtype=np.float32)
# X_inter = cv_inter.fit_transform(interaction_col)
# features_combined.append(X_inter)

# # (D) テキストの質感
# desc_str = mercari_df['item_description'].astype(str)
# digit_log = np.log1p(desc_str.apply(lambda x: sum(c.isdigit() for c in x))).astype(np.float32)
# cap_rate = desc_str.apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x)>0 else 0).astype(np.float32)
# X_texture = StandardScaler().fit_transform(pd.DataFrame({'digit': digit_log, 'cap': cap_rate}))
# features_combined.append(csr_matrix(X_texture, dtype=np.float32))

# # メモリお掃除
# del name_len_log, desc_len_log, desc_lower, interaction_col, desc_str, digit_log, cap_rate
# gc.collect()

# # ---------------------------------------------------------
# # 2. 【Pattern 2】TF-IDF & SVD (省メモリ版)
# # ---------------------------------------------------------
# print("2. 次元圧縮 (TF-IDF & SVD) を作成中...")

# # TF-IDF (Float32)
# tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words='english', dtype=np.float32)
# X_tfidf_tmp = tfidf.fit_transform(mercari_df['item_description'])

# # SVD (100->80次元に削減)
# n_comp = 80
# svd = TruncatedSVD(n_components=n_comp, algorithm='randomized', random_state=42)
# X_svd = svd.fit_transform(X_tfidf_tmp)
# X_svd = X_svd.astype(np.float32) # 念のため型変換

# # 標準化
# X_svd = StandardScaler().fit_transform(X_svd)
# features_combined.append(csr_matrix(X_svd, dtype=np.float32))

# del X_tfidf_tmp, tfidf, svd, X_svd
# gc.collect()

# # ---------------------------------------------------------
# # 3. 結合して実行
# # ---------------------------------------------------------
# print("3. 全特徴量を結合して学習開始...")
# # float32 で結合
# X_final_combined = hstack(features_combined, dtype=np.float32).tocsr()

# print(f"最終行列の形状: {X_final_combined.shape}")

# # 共通関数で実行
# train_and_submit(X_final_combined, "ridge_light_mix")

