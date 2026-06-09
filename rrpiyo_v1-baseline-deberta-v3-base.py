# データ分析の三種の神器をインポートするッピ！
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# グラフの見た目をかっこよくするおまじない
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
# 日本語も表示できるようにしておくッピ
# (Kaggle環境では必要に応じてフォントのインストールが必要になることもあるッピ)
# plt.rcParams['font.family'] = 'IPAGothic' 
print("おまじない、完了ッピ！")


# データが保存されている住所（パス）を指定するッピ
BASE_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/'

# 3種類のデータマメ（train, test, sample_submission）を読み込むッピ
train_df = pd.read_csv(BASE_PATH + 'train.csv')
test_df = pd.read_csv(BASE_PATH + 'test.csv')
submission_df = pd.read_csv(BASE_PATH + 'sample_submission.csv')

print('訓練データマメのサイズ:', train_df.shape)
print('テストデータマメのサイズ:', test_df.shape)
print('提出用フォーマットのサイズ:', submission_df.shape)


print("--- 訓練データの中身 ---")
display(train_df.head())

print("\n--- テストデータの中身 ---")
display(test_df.head())


print("--- 訓練データの健康診断 ---")
train_df.info()


# グラフの日本語が豆腐みたいに四角くならないようにするおまじないだッピ！
!pip install japanize-matplotlib -q
import japanize_matplotlib

print("日本語表示の準備OKだッピ！もう一度グラフのセルを実行してみてッピ！")


# --- 弟の発見に基づく、新しいコードだッピ！ ---

# まずはMisconception列の欠けている部分（NaN）を、
# 'NA'という文字で埋めるッピ。これが大事な下準備だッピ。
train_df['Misconception'] = train_df['Misconception'].fillna('NA')

# 次に、'Category'列と'Misconception'列をコロン「:」で連結して、
# 新しい'label'列を作るッピ！これぞ錬金術だッピ！
train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']

print("新しい'label'列を正しく作成できたッピ！ほら、この通り！")
display(train_df.head())


# --- これで、前回のグラフ描画コードが動くはずだッピ！ ---

# 新しく作った'label'列の各値の数を数えるッピ
label_counts = train_df['label'].value_counts()

# 数が多すぎるから、上位20件だけ表示してみるッピ
top_n = 20
plt.figure(figsize=(12, 8))
sns.barplot(y=label_counts.index[:top_n], x=label_counts.values[:top_n], orient='h')
plt.title(f'上位{top_n}件のラベル分布', fontsize=20)
plt.xlabel('件数', fontsize=15)
plt.ylabel('ラベル (Category:Misconception)', fontsize=15)
plt.show()

print(f"全部で {len(label_counts)} 種類のラベルがあるッピ！")
print(label_counts)


# --- 必要な料理道具（ライブラリ）のインポート ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("料理道具の準備OKだッピ！")


# --- 0. 【新処理】超レアなラベルを、一旦除外するッピ！ ---
# まず、各ラベルが何回出現するかを数えるッピ
label_counts = train_df['label'].value_counts()
# 次に、2回以上出現するラベルの名前だけをリストアップするッピ
labels_to_keep = label_counts[label_counts >= 2].index

# 元のデータフレームから、2回以上出現するラベルがついた行だけを抽出するッピ
train_filtered_df = train_df[train_df['label'].isin(labels_to_keep)]

print(f"元のデータ数: {len(train_df)}")
print(f"レアなラベルを除外した後のデータ数: {len(train_filtered_df)}")
print(f"お休みしてもらった超レアなデータマメの数: {len(train_df) - len(train_filtered_df)}")


# --- 1. 料理の準備：今度はフィルタリングしたデータで分けるッピ ---
X_train, X_val, y_train, y_val = train_test_split(
    train_filtered_df['StudentExplanation'], # フィルタリング後のデータを使う！
    train_filtered_df['label'],              # フィルタリング後のデータを使う！
    test_size=0.2,
    random_state=42,
    stratify=train_filtered_df['label']      # これでstratifyが安全に使えるッピ！
)

print(f"練習用データ: {len(X_train)}件, 実力テスト用データ: {len(X_val)}件に分けたッピ！")


# --- 2. テキストを数字に変換する魔法の粉「TF-IDF」 ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("テキストを数字のベクトルに変換完了ッピ！")


# --- 3. シンプルだけど強力なオーブン「ロジスティック回帰」で学習 ---
model = LogisticRegression(max_iter=1000, random_state=42)
print("モデルの学習（クッキング）を開始するッピ...")
model.fit(X_train_tfidf, y_train)
print("モデルの学習が完了したッピ！")


# --- 4. 実力テストの時間 ---
y_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)

print("\n--- ベースラインモデル（たまご焼き）の味見結果 ---")
print(f"Accuracy（正解率）: {accuracy:.4f}")


# --- 新しい料理道具をインポートするッピ ---
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

print("新しい料理道具、LightGBMの準備OKだッピ！")

# --- 0. MAP@3を計算する関数を定義するッピ ---
# これはコンペで勝つための、最重要の計算機だッピ！
def map_at_3(y_true, y_pred_proba):
    # y_true: ['label_A', 'label_B', ...]
    # y_pred_proba: [[0.1, 0.2, ...], [0.3, 0.05, ...], ...]
    
    # ラベルを数字に変換する辞書を作成
    classes = np.unique(y_true)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # 予測確率が高い順に、上位3つのクラスのインデックスを取得
    top3_preds_idx = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]
    
    scores = []
    for i, true_label in enumerate(y_true):
        true_idx = class_to_idx[true_label]
        pred_indices = top3_preds_idx[i]
        
        score = 0.0
        if true_idx in pred_indices:
            rank = np.where(pred_indices == true_idx)[0][0] + 1
            score = 1.0 / rank
        scores.append(score)
        
    return np.mean(scores)

# --- 1. データの準備（これはさっきと同じだッピ） ---
# レアなラベルを除外
label_counts = train_df['label'].value_counts()
labels_to_keep = label_counts[label_counts >= 2].index
train_filtered_df = train_df[train_df['label'].isin(labels_to_keep)]

# データを分割
X_train, X_val, y_train, y_val = train_test_split(
    train_filtered_df['StudentExplanation'],
    train_filtered_df['label'],
    test_size=0.2,
    random_state=42,
    stratify=train_filtered_df['label']
)

# --- 2. TF-IDF（これもさっきと同じ） ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# --- 3. LightGBMで学習！ ---
# LightGBMはラベルを文字列のまま扱えないので、LabelEncoderで数字に変換するッピ
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# モデルを定義
lgbm = lgb.LGBMClassifier(random_state=42)

print("超強力オーブン、LightGBMでの学習を開始するッピ！")
lgbm.fit(X_train_tfidf, y_train_encoded)
print("学習完了ッピ！")

# --- 4. MAP@3で実力テスト！ ---
# 今度は predict_proba() を使って、各ラベルの「確率」を予測させるッピ！
y_pred_proba = lgbm.predict_proba(X_val_tfidf)

# MAP@3スコアを計算！
# y_val_encodedではなく、元の文字列ラベルy_valを渡すことに注意！
map3_score = map_at_3(y_val.to_numpy(), y_pred_proba)

print("\n--- LightGBMモデルの味見結果 ---")
print(f"MAP@3 スコア: {map3_score:.4f}")


# --- 新しい料理道具 ---
import scipy.sparse
import re #正規表現を扱うための道具

# --- 1. 数学マメ特徴量を作る関数を定義するッピ ---
def create_math_features(texts):
    features_list = []
    for text in texts:
        # 数字の数を数える
        num_digits = len(re.findall(r'\d', text))
        # 単語数を数える
        num_words = len(text.split())
        # ユニークな単語数を数える
        num_unique_words = len(set(text.split()))
        # 単語の平均の長さを計算
        avg_word_length = np.mean([len(w) for w in text.split()]) if num_words > 0 else 0
        
        features = {
            'num_digits': num_digits,
            'num_words': num_words,
            'num_unique_words': num_unique_words,
            'word_density': num_unique_words / (num_words + 1e-6),
            'avg_word_length': avg_word_length,
        }
        features_list.append(features)
        
    return pd.DataFrame(features_list)

print("数学マメ特徴量を作る関数の準備OKだッピ！")

# --- 2. 練習用と実力テスト用の両方に、数学マメ特徴量を作るッピ ---
X_train_math_features = create_math_features(X_train)
X_val_math_features = create_math_features(X_val)

print("特徴量を作成しました:", X_train_math_features.shape, X_val_math_features.shape)
display(X_train_math_features.head())

# --- 3. TF-IDF特徴量と、数学マメ特徴量を合体させるッピ！ ---
# これが、Kaggleでよく使われるテクニックだッピ
X_train_combined = scipy.sparse.hstack([X_train_tfidf, scipy.sparse.csr_matrix(X_train_math_features)])
X_val_combined = scipy.sparse.hstack([X_val_tfidf, scipy.sparse.csr_matrix(X_val_math_features)])

print("特徴量の合体完了！新しい訓練データの形:", X_train_combined.shape)


# --- 4. 新しい特徴量を使って、再度LightGBMで学習・評価するッピ！ ---
# モデルの定義やラベルのエンコードはさっきと同じ
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

lgbm = lgb.LGBMClassifier(random_state=42)

print("新しいスパイスを加えたデータで、学習を開始するッピ！")
lgbm.fit(X_train_combined, y_train_encoded)
print("学習完了ッピ！")

y_pred_proba_new = lgbm.predict_proba(X_val_combined)
map3_score_new = map_at_3(y_val.to_numpy(), y_pred_proba_new)

print("\n--- 新しいスパイス追加後の味見結果 ---")
print(f"以前のMAP@3 スコア: 0.2111")
print(f"新しいMAP@3 スコア: {map3_score_new:.4f}")

if map3_score_new > 0.2111:
    print("\nやったッピ！スコアが上がった！このスパイスは効果ありだッピ！")
else:
    print("\nうーん、スコアが変わらないか、下がってしまったッピ…。スパイスの組み合わせが悪かったかもしれないッピ。")


# --- 新しい魔法の道具 Optuna をインポート ---
import optuna

print("自動チューニング道具 Optuna の準備OKだッピ！")

# --- 1. Optunaに最適化させたい「目的（objective）」を教える ---
def objective(trial):
    # ここで、試してほしいパラメータの候補（範囲）を指定するッピ
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    
    # モデルを定義して学習
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_combined, y_train_encoded)
    
    # 評価
    y_pred_proba = model.predict_proba(X_val_combined)
    score = map_at_3(y_val.to_numpy(), y_pred_proba)
    
    return score

# --- 2. 最適化の「研究（study）」を開始するッピ！ ---
# 'maximize'で、スコアが最大になるように指示する
study = optuna.create_study(direction='maximize')

# n_trials=30 で、30通りのパラメータの組み合わせを試してもらうッピ
# （時間をかければかけるほど、良い設定が見つかる可能性があるッピ）
print("ハイパーパラメータチューニングを開始します。少し時間がかかるッピ...")
study.optimize(objective, n_trials=30)
print("チューニング完了ッピ！")


# --- 3. 結果発表！ ---
print("\n--- Optunaが見つけた最高のレシピ ---")
print(f"最高のMAP@3スコア: {study.best_value:.4f}")
print("その時の最高のパラメータ設定:")
print(study.best_params)

