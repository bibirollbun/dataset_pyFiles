# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer,PolynomialFeatures,RobustScaler,StandardScaler,QuantileTransformer
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install pytorch_tabnet


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.info()


train.head()


sns.heatmap(train.select_dtypes(exclude=['object', 'category']).corr(),annot=True)


sns.heatmap(test.select_dtypes(exclude=['object', 'category']).corr(),annot=True)


train['loan_paid_back'].value_counts().plot(kind='bar')


X = train.copy()
tr_id = train['id']
te_id = test['id']
y = X['loan_paid_back']

X.drop(['id','loan_paid_back'],axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)
test_pe = test.copy()


cat_cols_tr = train.select_dtypes(include=['object', 'category']).columns
cat_cols_te = test.select_dtypes(include=['object', 'category']).columns
num_cols_tr = train.select_dtypes(exclude=['object', 'category']).columns
num_cols_te = test.select_dtypes(exclude=['object', 'category']).columns


from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
import category_encoders as ce
# OrdinalEncoder : 複数項目に対する、一括ラベルエンコーディング

# train_oe = train.copy()
cat_e = ce.OneHotEncoder(cols=cat_cols_tr,handle_unknown='impute')
cat_tr = cat_e.fit_transform(train[cat_cols_tr]) # 欠損値不可
print(cat_tr.tail())

cat_te = cat_e.transform(test[cat_cols_te])
print(cat_te.tail())




X=pd.concat([X,cat_tr],axis=1)
X.drop(cat_cols_tr,axis=1,inplace=True)
test=pd.concat([test,cat_te],axis=1)
test.drop(cat_cols_te,axis=1,inplace=True)





scaler = StandardScaler()
feature_en = PolynomialFeatures(interaction_only=True)

X_feat = feature_en.fit_transform(X)
X_robust = scaler.fit_transform(X)

test_feat = feature_en.transform(test)  
test_robust = scaler.transform(test)




# ===== Step 1: ライブラリのインストールと読み込み =====
# !pip install pytorch-tabnet

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# ===== GPU確認 =====
print(f"PyTorchバージョン: {torch.__version__}")
print(f"CUDA利用可能: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA バージョン: {torch.version.cuda}")
    print(f"利用可能なGPU数: {torch.cuda.device_count()}")
    print(f"GPU名: {torch.cuda.get_device_name(0)}")

X_ = X_robust.copy()
# 訓練・検証・テストデータに分割
X_train, X_temp, y_train, y_temp = train_test_split(
    X_, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n訓練データ: {X_train.shape}")
print(f"検証データ: {X_valid.shape}")
print(f"テストデータ: {X_test.shape}")

# ===== Step 3: TabNetClassifierの初期化（GPU設定・修正版） =====
"""
device_name の指定方法（引用: https://pypi.org/project/pytorch-tabnet/）:
- 'cpu': CPU訓練
- 'cuda': GPU訓練（修正箇所）
- 'auto': GPU自動検出

注意: 'gpu'ではなく'cuda'を使用する必要があります
"""

# GPUが利用可能な場合は'cuda'、そうでなければ'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n使用デバイス: {device}")

tabnet_params = {
    "n_d": 8,                      # 決定層の幅
    "n_a": 8,                      # 注意機構の幅
    "n_steps": 10,                  # 決定ステップ数
    "gamma": 1.3,                  # 特徴選択の強度
    "lambda_sparse": 1e-2,         # スパース正則化
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=5e-2),
    "momentum": 0.02,
    "mask_type": "sparsemax",      # マスクタイプ
    "scheduler_params": {"step_size": 10, "gamma": 0.9},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "seed": 42,
    "verbose": 1,
    "device_name": device          # 'cuda'または'cpu'に修正
}

clf = TabNetClassifier(**tabnet_params)

# ===== Step 4: モデルの訓練 =====
print("\nモデル訓練開始...")
clf.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=["valid"],
    eval_metric=["auc", "accuracy"],  # 評価指標
    max_epochs=100,
    patience=20,                      # Early stopping
    batch_size=256,
    virtual_batch_size=128,           # Ghost Batch Normalization用
    num_workers=0,
    drop_last=False
)

# ===== Step 5: predict_probaで確率値の予測 =====
print("\n予測開始...")
# 検証データで予測
y_valid_proba = clf.predict_proba(X_valid)
print(f"\n検証データの予測確率値の形状: {y_valid_proba.shape}")
print(f"サンプル予測確率（最初の5件）:")
print(y_valid_proba[:5])

# テストデータで予測
y_test_proba = clf.predict_proba(X_test)
y_test_pred = clf.predict(X_test)

print(f"\nテストデータの予測確率値の形状: {y_test_proba.shape}")

# ===== Step 6: モデル評価 =====
print("\n" + "="*50)
print("モデル評価結果")
print("="*50)

# AUCスコア（確率値を使用）
auc_score = roc_auc_score(y_test, y_test_proba[:, 1])
print(f"\nテストデータのAUCスコア: {auc_score:.4f}")

# 精度
accuracy = accuracy_score(y_test, y_test_pred)
print(f"テストデータの精度: {accuracy:.4f}")

# 詳細な分類レポート
print("\n分類レポート:")
print(classification_report(y_test, y_test_pred))

# ===== Step 7: 特徴量の重要度 =====
print("\n" + "="*50)
print("特徴量の重要度")
print("="*50)
feature_importances = clf.feature_importances_
for i, importance in enumerate(feature_importances):
    print(f"特徴量 {i:2d}: {importance:.4f}")

# 重要度の高い特徴量トップ5
top_features = np.argsort(feature_importances)[::-1][:5]
print("\n重要度トップ5の特徴量:")
for rank, feat_idx in enumerate(top_features, 1):
    print(f"{rank}位: 特徴量{feat_idx} (重要度: {feature_importances[feat_idx]:.4f})")

# ===== Step 8: モデルの保存と読み込み =====
saved_filepath = clf.save_model("tabnet_binary_classifier")
print(f"\nモデルを保存しました: {saved_filepath}")

# モデルの読み込み例（必要な場合）
# loaded_clf = TabNetClassifier()
# loaded_clf.load_model(saved_filepath)
# y_pred_loaded = loaded_clf.predict_proba(X_test)
# print("読み込んだモデルで予測完了")

print("\n" + "="*50)
print("完了！")
print("="*50)


"""
LightGBM Binary Classification - 新規データ予測機能追加版
引用元:
- LightGBM Documentation: https://lightgbm.readthedocs.io/en/latest/
- pandas: https://pandas.pydata.org/docs/
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import pickle
from pathlib import Path


class LGBMBinaryClassifier:
    """
    LightGBM Binary Classifier with GPU Support and Stratified Sampling
    
    引用元: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    """
    
    def __init__(self, use_gpu=True, random_state=42):
        """
        Parameters:
        -----------
        use_gpu : bool
            GPU使用の有無
        random_state : int
            再現性のための乱数シード
        """
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.label_encoders = {}
        
        # GPU設定
        # 引用元: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
        self.device_type = 'gpu' if use_gpu else 'cpu'
        
    def preprocess_data(self, X, y=None, is_train=True):
        """
        データの前処理（カテゴリカル変数のエンコーディング）
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データ
        y : pd.Series or np.ndarray, optional
            目的変数
        is_train : bool
            学習データかどうか
            
        Returns:
        --------
        X_processed : pd.DataFrame
            前処理済みデータ
        y_processed : np.ndarray or None
            前処理済み目的変数
        """
        X_processed = X.copy()
        
        # カテゴリカル変数の検出とエンコーディング
        categorical_cols = X_processed.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        for col in categorical_cols:
            if is_train:
                # 学習時: LabelEncoderを作成・保存
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(
                    X_processed[col].astype(str)
                )
            else:
                # 予測時: 保存済みのLabelEncoderを使用
                if col in self.label_encoders:
                    # 未知のカテゴリは最頻値で埋める
                    X_processed[col] = X_processed[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ 
                        else self.label_encoders[col].classes_[0]
                    )
                    X_processed[col] = self.label_encoders[col].transform(
                        X_processed[col].astype(str)
                    )
        
        # 欠損値の処理
        X_processed = X_processed.fillna(-999)
        
        if y is not None:
            y_processed = np.array(y)
            return X_processed, y_processed
        
        return X_processed, None
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None, params=None):
        """
        モデルの学習
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            学習データの特徴量
        y_train : pd.Series or np.ndarray
            学習データの目的変数
        X_valid : pd.DataFrame, optional
            検証データの特徴量
        y_valid : pd.Series or np.ndarray, optional
            検証データの目的変数
        params : dict, optional
            LightGBMのハイパーパラメータ
            
        引用元: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        """
        # データの前処理
        X_train_processed, y_train_processed = self.preprocess_data(
            X_train, y_train, is_train=True
        )
        self.feature_names = X_train_processed.columns.tolist()
        
        # デフォルトパラメータの設定
        # 引用元: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        default_params = {
            'objective': 'binary', 
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'max_depth': 12,
            'num_leaves': 50,
            'learning_rate': 0.01,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'subsample_freq': 1,
            'min_child_samples': 20,
            'reg_alpha': 0.05,
            'reg_lambda': 0.1,
            'n_jobs': -1, 
            'verbose': -1,
            'device_type': self.device_type,
            'random_state': self.random_state
        }
        
        # GPU固有の設定
        # 引用元: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
        if self.use_gpu:
            default_params.update({
                'gpu_use_dp': False,  # 単精度浮動小数点を使用
                'max_bin': 63  # GPUでの推奨値
            })
        
        if params:
            default_params.update(params)
        
        # LGBMClassifierの初期化
        self.model = lgb.LGBMClassifier(**default_params)
        
        # 検証データの前処理
        eval_set = None
        if X_valid is not None and y_valid is not None:
            X_valid_processed, y_valid_processed = self.preprocess_data(
                X_valid, y_valid, is_train=False
            )
            eval_set = [(X_valid_processed, y_valid_processed)]
        
        # モデルの学習
        # 引用元: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit
        self.model.fit(
            X_train_processed,
            y_train_processed,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ] if eval_set else None
        )
        
        print(f"✓ モデル学習完了")
        print(f"  - Best iteration: {self.model.best_iteration_}")
        
    def predict_proba(self, X):
        """
        確率予測
        
        Parameters:
        -----------
        X : pd.DataFrame
            予測対象データ
            
        Returns:
        --------
        proba : np.ndarray
            予測確率 (n_samples, 2)
            [:,0]: クラス0の確率
            [:,1]: クラス1の確率
            
        引用元: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.predict_proba
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。先にtrain()を実行してください。")
        
        X_processed, _ = self.preprocess_data(X, is_train=False)
        proba = self.model.predict_proba(X_processed)
        
        return proba
    
    def predict(self, X, threshold=0.5):
        """
        クラス予測
        
        Parameters:
        -----------
        X : pd.DataFrame
            予測対象データ
        threshold : float
            分類閾値（デフォルト: 0.5）
            
        Returns:
        --------
        predictions : np.ndarray
            予測クラス
        """
        proba = self.predict_proba(X)
        predictions = (proba[:, 1] >= threshold).astype(int)
        return predictions
    
    # ========================================
    # 新規データ予測メソッド（追加）
    # ========================================
    
    def predict_new_data(self, X_new, threshold=0.5, return_proba=True):
        """
        新規データに対する予測（確率とクラスの両方を返す）
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            予測対象の新規データ
        threshold : float
            分類閾値（デフォルト: 0.5）
        return_proba : bool
            確率も返すかどうか
            
        Returns:
        --------
        result : pd.DataFrame
            予測結果を含むDataFrame
            - predicted_class: 予測クラス (0 or 1)
            - predicted_proba_class0: クラス0の予測確率
            - predicted_proba_class1: クラス1の予測確率
            
        引用元: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。先にtrain()を実行してください。")
        
        print(f"新規データの予測を開始...")
        print(f"  - データサイズ: {X_new.shape}")
        
        # 予測確率の計算
        proba = self.predict_proba(X_new)
        
        # 予測クラスの計算
        predicted_class = (proba[:, 1] >= threshold).astype(int)
        
        # 結果をDataFrameにまとめる
        result = pd.DataFrame({
            'predicted_class': predicted_class,
            'predicted_proba_class0': proba[:, 0],
            'predicted_proba_class1': proba[:, 1]
        })
        
        # クラス1の確率に基づいて信頼度を追加
        result['confidence'] = np.where(
            result['predicted_class'] == 1,
            result['predicted_proba_class1'],
            result['predicted_proba_class0']
        )
        
        print(f"✓ 予測完了")
        print(f"  - クラス1の予測数: {(predicted_class == 1).sum()}")
        print(f"  - クラス0の予測数: {(predicted_class == 0).sum()}")
        print(f"  - 平均信頼度: {result['confidence'].mean():.4f}")
        
        return result
    
    def predict_from_csv(self, csv_path, threshold=0.5, output_path=None):
        """
        CSVファイルから新規データを読み込んで予測
        
        Parameters:
        -----------
        csv_path : str
            予測対象のCSVファイルパス
        threshold : float
            分類閾値
        output_path : str, optional
            予測結果を保存するCSVファイルパス
            
        Returns:
        --------
        result_df : pd.DataFrame
            元のデータ + 予測結果
            
        引用元: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        
        print(f"CSVファイルから予測を実行: {csv_path}")
        
        # CSVファイルの読み込み
        X_new = pd.read_csv(csv_path)
        print(f"  - 読み込みデータサイズ: {X_new.shape}")
        
        # 予測の実行
        predictions = self.predict_new_data(X_new, threshold=threshold)
        
        # 元のデータと予測結果を結合
        result_df = pd.concat([X_new, predictions], axis=1)
        
        # 結果を保存
        if output_path:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✓ 予測結果を保存: {output_path}")
        
        return result_df
    
    def batch_predict(self, X_new, batch_size=1000, threshold=0.5):
        """
        大量データに対するバッチ予測
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            予測対象の新規データ
        batch_size : int
            バッチサイズ（メモリ効率のため）
        threshold : float
            分類閾値
            
        Returns:
        --------
        result : pd.DataFrame
            予測結果
            
        引用元: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        
        print(f"バッチ予測を開始...")
        print(f"  - 総データ数: {len(X_new)}")
        print(f"  - バッチサイズ: {batch_size}")
        
        n_samples = len(X_new)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        all_predictions = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            X_batch = X_new.iloc[start_idx:end_idx]
            
            # バッチごとに予測
            proba_batch = self.predict_proba(X_batch)
            predicted_class_batch = (proba_batch[:, 1] >= threshold).astype(int)
            
            batch_result = pd.DataFrame({
                'predicted_class': predicted_class_batch,
                'predicted_proba_class0': proba_batch[:, 0],
                'predicted_proba_class1': proba_batch[:, 1]
            })
            
            all_predictions.append(batch_result)
            
            if (i + 1) % 10 == 0:
                print(f"  - 進捗: {i + 1}/{n_batches} バッチ完了")
        
        # 全バッチの結果を結合
        result = pd.concat(all_predictions, ignore_index=True)
        
        print(f"✓ バッチ予測完了")
        
        return result
    
    def predict_with_explanation(self, X_new, threshold=0.5, top_n_features=5):
        """
        予測結果と重要特徴量の説明を返す
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            予測対象データ
        threshold : float
            分類閾値
        top_n_features : int
            表示する重要特徴量の数
            
        Returns:
        --------
        result_df : pd.DataFrame
            予測結果と説明
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        
        print(f"予測と説明を生成中...")
        
        # 予測の実行
        predictions = self.predict_new_data(X_new, threshold=threshold)
        
        # 特徴量重要度の取得
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(top_n_features)['feature'].tolist()
        
        # 予測結果に重要特徴量の値を追加
        result_df = predictions.copy()
        
        X_processed, _ = self.preprocess_data(X_new, is_train=False)
        
        for feature in top_features:
            if feature in X_processed.columns:
                result_df[f'feature_{feature}'] = X_processed[feature].values
        
        print(f"✓ 予測と説明の生成完了")
        print(f"  - 上位{top_n_features}の重要特徴量: {top_features}")
        
        return result_df
    
    # ========================================
    # モデル保存・読み込みメソッド
    # ========================================
    
    def save_model(self, model_path='lgbm_model.pkl'):
        """
        モデルを保存
        
        Parameters:
        -----------
        model_path : str
            保存先のファイルパス
            
        引用元: https://docs.python.org/3/library/pickle.html
        """
        if self.model is None:
            raise ValueError("保存するモデルがありません。")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'use_gpu': self.use_gpu,
            'random_state': self.random_state
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ モデルを保存: {model_path}")
    
    def load_model(self, model_path='lgbm_model.pkl'):
        """
        モデルを読み込み
        
        Parameters:
        -----------
        model_path : str
            読み込むファイルパス
            
        引用元: https://docs.python.org/3/library/pickle.html
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.label_encoders = model_data['label_encoders']
        self.use_gpu = model_data['use_gpu']
        self.random_state = model_data['random_state']
        
        print(f"✓ モデルを読み込み: {model_path}")
    
    def evaluate(self, X, y_true, threshold=0.5):
        """
        モデルの評価
        
        Parameters:
        -----------
        X : pd.DataFrame
            評価データの特徴量
        y_true : pd.Series or np.ndarray
            正解ラベル
        threshold : float
            分類閾値
            
        Returns:
        --------
        metrics : dict
            評価指標の辞書
        """
        proba = self.predict_proba(X)
        y_pred = self.predict(X, threshold=threshold)
        
        metrics = {
            'AUC': roc_auc_score(y_true, proba[:, 1]),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred)
        }
        
        print("\n=== 評価結果 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\n=== 混同行列 ===")
        print(confusion_matrix(y_true, y_pred))
        
        print("\n=== 分類レポート ===")
        print(classification_report(y_true, y_pred))
        
        return metrics
    
    def get_feature_importance(self, importance_type='gain', top_n=20):
        """
        特徴量重要度の取得
        
        Parameters:
        -----------
        importance_type : str
            'gain' or 'split'
        top_n : int
            表示する上位特徴量数
            
        引用元: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.feature_importances_
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== 特徴量重要度 Top {top_n} ===")
        print(feature_importance_df.head(top_n))
        
        return feature_importance_df


# ========================================
# 使用例
# ========================================

def main_with_new_data_prediction():
    """
    新規データ予測機能の使用例
    """
    
    print("=" * 70)
    print("LightGBM Binary Classification - 新規データ予測の例")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    
    # Step 1: 学習データの準備
    print("\nStep 1: 学習データの準備")
    print("-" * 70)
    
    df_train = train.copy()

    y_train = df_train['loan_paid_back']
    X_train = df_train.drop(['id','loan_paid_back'], axis=1)
    
    # 層化抽出で分割
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )
    
    print(f"学習データ: {X_train.shape}")
    print(f"検証データ: {X_valid.shape}")
    
    # Step 2: モデルの学習
    print("\nStep 2: モデルの学習")
    print("-" * 70)
    
    classifier = LGBMBinaryClassifier(use_gpu=True, random_state=42)
    
    classifier.train(
        X_train, y_train,
        X_valid, y_valid,
        params={'n_estimators': 1000, 'learning_rate': 0.01}
    )
    
    # Step 3: 新規データの準備（実際の未知データを想定）
    print("\nStep 3: 新規データの準備（未知データの想定）")
    print("-" * 70)
    
    # テストデータ
    df_new = test_pe.copy()
    print(f"新規データサイズ: {df_new.shape}")
    
    # Step 4: 新規データに対する予測
    print("\nStep 4: 新規データに対する予測")
    print("-" * 70)
    
    # 方法1: predict_new_data（基本）
    predictions = classifier.predict_new_data(df_new, threshold=0.5)
    print(f"\n予測結果（最初の10件）:")
    print(predictions.head(10))
    
    # 方法2: predict_with_explanation（説明付き）
    print("\nStep 5: 説明付き予測")
    print("-" * 70)
    
    predictions_with_explanation = classifier.predict_with_explanation(
        df_new.head(10),
        threshold=0.5,
        top_n_features=5
    )
    print(predictions_with_explanation)
    
    # 方法3: バッチ予測（大量データ用）
    print("\nStep 6: バッチ予測")
    print("-" * 70)
    
    batch_predictions = classifier.batch_predict(
        df_new,
        batch_size=250,
        threshold=0.5
    )
    print(f"バッチ予測結果の統計:")
    print(batch_predictions.describe())
    
    # Step 7: CSVファイルを使った予測
    print("\nStep 7: CSVファイルを使った予測")
    print("-" * 70)
    
    # サンプルCSVを作成
    df_new.to_csv('new_data_sample.csv', index=False)
    print("✓ サンプルCSVを作成: new_data_sample.csv")
    
    # CSVから予測
    result_df = classifier.predict_from_csv(
        'new_data_sample.csv',
        threshold=0.5,
        output_path='predictions_output.csv'
    )
    
    print(f"\n予測結果の概要:")
    print(result_df[['predicted_class', 'predicted_proba_class1', 'confidence']].describe())
    
    # Step 8: モデルの保存と読み込み
    print("\nStep 8: モデルの保存と読み込み")
    print("-" * 70)
    
    # モデルを保存
    classifier.save_model('trained_lgbm_model.pkl')
    
    # 新しいインスタンスでモデルを読み込み
    new_classifier = LGBMBinaryClassifier()
    new_classifier.load_model('trained_lgbm_model.pkl')
    
    # 読み込んだモデルで予測
    loaded_predictions = new_classifier.predict_new_data(df_new.head(5))
    print(f"\n読み込んだモデルでの予測結果:")
    print(loaded_predictions)
    
    # Step 9: 異なる閾値での予測比較
    print("\nStep 9: 異なる閾値での予測比較")
    print("-" * 70)
    
    sample_data = df_new.head(100)
    
    for threshold in [0.3, 0.5, 0.7]:
        preds = classifier.predict_new_data(sample_data, threshold=threshold)
        class1_count = (preds['predicted_class'] == 1).sum()
        avg_conf = preds['confidence'].mean()
        
        print(f"\n閾値 {threshold}:")
        print(f"  - クラス1予測数: {class1_count} / 100")
        print(f"  - 平均信頼度: {avg_conf:.4f}")

    return predictions


if __name__ == "__main__":
    pred_data = main_with_new_data_prediction()


pred_data


sub_ = test_robust.copy()


sub['loan_paid_back'] = (clf.predict_proba(sub_)[:,1]+pred_data['predicted_proba_class1'])/2.


sub.to_csv('submission.csv',index=False)


sub.head()




