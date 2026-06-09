# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install pytorch_tabnet
!pip install autogluon.features


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


print(train.info(),test.info())


train['accident_risk'].value_counts().plot(kind='bar',figsize=(10,5))


sns.heatmap(train.select_dtypes(exclude=['object', 'category']).corr(),annot=True)


cat_cols_tr = train.select_dtypes(include=['object', 'category']).columns
cat_cols_te = test.select_dtypes(include=['object', 'category']).columns
num_cols_tr = train.select_dtypes(exclude=['object', 'category']).columns
num_cols_te = test.select_dtypes(exclude=['object', 'category']).columns


for i in cat_cols_tr:
    train[[i,'accident_risk']].groupby(i).sum().plot(kind='bar')


tr_id = train['id']
te_id = test['id']
target = train['accident_risk']

train.drop(['id','accident_risk'],axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)





from sklearn.preprocessing import OrdinalEncoder
# OrdinalEncoder : 複数項目に対する、一括ラベルエンコーディング

train_oe = train.copy()
oe = OrdinalEncoder()
train[cat_cols_tr] = oe.fit_transform(train[cat_cols_tr]) # 欠損値不可
print(train.tail())

test[cat_cols_tr] = oe.transform(test[cat_cols_tr])
print(test.tail())


from autogluon.features.generators import AutoMLPipelineFeatureGenerator
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
train = auto_ml_pipeline_feature_generator.fit_transform(X=train,y=target)
test = auto_ml_pipeline_feature_generator.transform(X=test)


import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class FeatureCombiner:
    """特徴量の組み合わせに特化したクラス（シンプル版）"""
    
    def __init__(self, df):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            入力データフレーム
        """
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def add_multiply_features(self, columns=None):
        """
        特徴量の積を追加
        
        Parameters:
        -----------
        columns : list, optional
            対象列。Noneの場合は全数値列
        
        Returns:
        --------
        pd.DataFrame
            積の特徴量を追加したデータフレーム
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col1, col2 in combinations(columns, 2):
            self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
        
        print(f"積の特徴量を追加: {len(list(combinations(columns, 2)))} 個")
        return self.df.copy()
    
    def add_addition_features(self, columns=None):
        """
        特徴量の和を追加
        
        Parameters:
        -----------
        columns : list, optional
            対象列。Noneの場合は全数値列
        
        Returns:
        --------
        pd.DataFrame
            和の特徴量を追加したデータフレーム
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col1, col2 in combinations(columns, 2):
            self.df[f'{col1}_plus_{col2}'] = self.df[col1] + self.df[col2]
        
        print(f"和の特徴量を追加: {len(list(combinations(columns, 2)))} 個")
        return self.df.copy()
    
    def add_subtraction_features(self, columns=None):
        """
        特徴量の差を追加
        
        Parameters:
        -----------
        columns : list, optional
            対象列。Noneの場合は全数値列
        
        Returns:
        --------
        pd.DataFrame
            差の特徴量を追加したデータフレーム
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col1, col2 in combinations(columns, 2):
            self.df[f'{col1}_minus_{col2}'] = self.df[col1] - self.df[col2]
        
        print(f"差の特徴量を追加: {len(list(combinations(columns, 2)))} 個")
        return self.df.copy()
    
    def add_division_features(self, columns=None):
        """
        特徴量の比率を追加
        
        Parameters:
        -----------
        columns : list, optional
            対象列。Noneの場合は全数値列
        
        Returns:
        --------
        pd.DataFrame
            比率の特徴量を追加したデータフレーム
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col1, col2 in combinations(columns, 2):
            self.df[f'{col1}_div_{col2}'] = self.df[col1] / (self.df[col2] + 1e-10)
        
        print(f"比率の特徴量を追加: {len(list(combinations(columns, 2)))} 個")
        return self.df.copy()
    
    def add_polynomial_features(self, columns=None, degrees=[2, 3]):
        """
        多項式特徴量を追加
        
        Parameters:
        -----------
        columns : list, optional
            対象列。Noneの場合は全数値列
        degrees : list
            次数のリスト（例: [2, 3] で2乗と3乗）
        
        Returns:
        --------
        pd.DataFrame
            多項式特徴量を追加したデータフレーム
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        count = 0
        for col in columns:
            for degree in degrees:
                self.df[f'{col}_pow{degree}'] = self.df[col] ** degree
                count += 1
        
        print(f"多項式特徴量を追加: {count} 個")
        return self.df.copy()
    
    def add_all_combinations(self, columns=None, operations=['multiply', 'add', 'subtract', 'divide']):
        """
        指定した演算の組み合わせをすべて追加
        
        Parameters:
        -----------
        columns : list, optional
            対象列。Noneの場合は全数値列
        operations : list
            実行する演算のリスト
            ['multiply', 'add', 'subtract', 'divide']
        
        Returns:
        --------
        pd.DataFrame
            すべての組み合わせ特徴量を追加したデータフレーム
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_count = len(self.df.columns)
        
        if 'multiply' in operations:
            for col1, col2 in combinations(columns, 2):
                self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
        
        if 'add' in operations:
            for col1, col2 in combinations(columns, 2):
                self.df[f'{col1}_plus_{col2}'] = self.df[col1] + self.df[col2]
        
        if 'subtract' in operations:
            for col1, col2 in combinations(columns, 2):
                self.df[f'{col1}_minus_{col2}'] = self.df[col1] - self.df[col2]
        
        if 'divide' in operations:
            for col1, col2 in combinations(columns, 2):
                self.df[f'{col1}_div_{col2}'] = self.df[col1] / (self.df[col2] + 1e-10)
        
        added_count = len(self.df.columns) - initial_count
        print(f"組み合わせ特徴量を追加: {added_count} 個")
        return self.df.copy()
    
    def get_dataframe(self):
        """
        現在のデータフレームを取得
        
        Returns:
        --------
        pd.DataFrame
            処理後のデータフレーム
        """
        return self.df.copy()
    
    def reset(self):
        """
        データフレームを初期状態にリセット
        
        Returns:
        --------
        pd.DataFrame
            元のデータフレーム
        """
        self.df = self.df[self.original_columns].copy()
        print("データフレームを初期状態にリセットしました")
        return self.df.copy()
    
    def get_info(self):
        """現在の状態を表示"""
        print("=" * 60)
        print("データフレーム情報")
        print("=" * 60)
        print(f"行数: {len(self.df)}")
        print(f"元の特徴量数: {len(self.original_columns)}")
        print(f"現在の特徴量数: {len(self.df.columns)}")
        print(f"追加された特徴量数: {len(self.df.columns) - len(self.original_columns)}")
        print("=" * 60)


# ========== 使用例 ==========

if __name__ == "__main__":
    # サンプルデータ作成
    # np.random.seed(42)
    
    # sample_df = pd.DataFrame({
    #     'age': np.random.randint(20, 70, 100),
    #     'income': np.random.randint(30000, 150000, 100),
    #     'experience': np.random.randint(0, 30, 100),
    #     'target': np.random.randint(0, 2, 100)
    # })
    
    # print("元のデータ")
    # print("=" * 60)
    # print(sample_df.head())
    # print(f"データ形状: {sample_df.shape}\n")
    
    # クラスのインスタンス作成
    combiner = FeatureCombiner(train)
    combiner_te = FeatureCombiner(test)
    
    # ========== 例1: 積の特徴量のみ追加 ==========
    # print("\n[例1] 積の特徴量のみ追加")
    # print("-" * 60)
    # result1 = combiner.add_multiply_features(columns=['age', 'income', 'experience'])
    # print(f"新しい列: {[col for col in result1.columns if '_x_' in col]}")
    
    # # リセット
    # combiner.reset()
    
    # ========== 例2: 複数のメソッドを順次実行 ==========
    # print("\n[例2] 複数のメソッドを順次実行")
    # print("-" * 60)
    # combiner.add_multiply_features(columns=['age', 'income'])
    # combiner.add_division_features(columns=['income', 'experience'])
    # combiner.add_polynomial_features(columns=['age'], degrees=[2])
    # combiner.get_info()
    
    # # リセット
    # combiner.reset()
    
    # ========== 例3: すべての組み合わせを一度に追加 ==========
    print("\n[例3] すべての組み合わせを一度に追加")
    print("-" * 60)
    result3 = combiner.add_all_combinations(
        columns=train.columns,
        operations=['multiply', 'divide']
    )

    result3 = combiner_te.add_all_combinations(
        columns=test.columns,
        operations=['multiply', 'divide']
    )
    combiner.get_info()
    combiner_te.get_info()
    
    # 最終結果を取得
    final_df = combiner.get_dataframe()
    final_df_te = combiner_te.get_dataframe()
    print(f"\n最終データ形状: {final_df.shape}")
    print(f"\n最終データ形状: {final_df_te.shape}")
    print(f"\n特徴量の例（最初の10列）:")
    print(final_df.columns[:10].tolist())


final_df


test


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
final_df[final_df.columns] = scaler.fit_transform(final_df)
final_df_te[final_df_te.columns] = scaler.transform(final_df_te)

print(final_df,final_df_te)
# train['accident_risk']=target





"""
LightGBM Modeling Module
========================
モデリング部分とパラメータ設定のみを抽出

使い方:
    from lgb_model import LGBMConfig, LGBMModel
    
    config = LGBMConfig.binary_classification()
    model = LGBMModel(config)
    model.train_cross_validation(X, y)
    predictions = model.predict(X_test)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    mean_squared_error, mean_absolute_error, f1_score
)
from typing import Optional, Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Parameter Configuration Class
# =============================================================================
class LGBMConfig:
    """LightGBMのパラメータ設定クラス"""
    
    def __init__(
        self,
        task: str = 'binary',
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 10000,
        early_stopping_rounds: int = 100,
        verbose_eval: int = 100,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        task : str
            'binary', 'multiclass', 'regression'
        params : dict, optional
            LightGBMのパラメータ辞書
        num_boost_round : int
            最大ブースティング回数
        early_stopping_rounds : int
            Early stoppingのラウンド数
        verbose_eval : int
            ログ出力の間隔
        n_splits : int
            交差検証のfold数
        random_state : int
            乱数シード
        """
        self.task = task
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.n_splits = n_splits
        self.random_state = random_state
        
        # パラメータが指定されていなければデフォルトを使用
        if params is None:
            self.params = self._get_default_params()
        else:
            self.params = params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """タスクに応じたデフォルトパラメータを返す"""
        base_params = {
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 5,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if self.task == 'binary':
            base_params.update({
                'objective': 'binary',
                'metric': 'auc',
                'is_unbalance': False
            })
        elif self.task == 'multiclass':
            base_params.update({
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': 3  # 必要に応じて変更
            })
        elif self.task == 'regression':
            base_params.update({
                'objective': 'regression',
                'metric': 'rmse'
            })
        
        return base_params
    
    @classmethod
    def binary_classification(
        cls,
        metric: str = 'auc',
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        **kwargs
    ):
        """二値分類用の設定"""
        params = {
            'objective': 'binary',
            'metric': metric,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'boosting_type': 'gbdt',
            'subsample': 0.8,
            'subsample_freq': 5,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'verbose': -1
        }
        params.update(kwargs)
        return cls(task='binary', params=params)
    
    @classmethod
    def multiclass_classification(
        cls,
        num_class: int,
        metric: str = 'multi_logloss',
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        **kwargs
    ):
        """多クラス分類用の設定"""
        params = {
            'objective': 'multiclass',
            'num_class': num_class,
            'metric': metric,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'boosting_type': 'gbdt',
            'subsample': 0.8,
            'subsample_freq': 5,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'verbose': -1
        }
        params.update(kwargs)
        return cls(task='multiclass', params=params)
    
    @classmethod
    def regression(
        cls,
        metric: str = 'rmse',
        learning_rate: float = 0.05,
        num_leaves: int = 62,
        **kwargs
    ):
        """回帰用の設定"""
        params = {
            'objective': 'regression',
            'metric': metric,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'boosting_type': 'gbdt',
            'subsample': 0.8,
            'subsample_freq': 5,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'n_jobs': -1,
            'verbose': -1
        }
        params.update(kwargs)
        return cls(task='regression', params=params)
    
    @classmethod
    def fast_training(cls, task: str = 'binary', **kwargs):
        """高速訓練用（精度は低い）"""
        params = {
            'learning_rate': 0.1,
            'num_leaves': 15,
            'max_depth': 5,
            'min_child_samples': 50,
        }
        params.update(kwargs)
        return cls(
            task=task,
            params=params,
            num_boost_round=1000,
            early_stopping_rounds=50
        )
    
    @classmethod
    def high_accuracy(cls, task: str = 'binary', **kwargs):
        """高精度訓練用（時間がかかる）"""
        params = {
            'learning_rate': 0.01,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 10,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
        params.update(kwargs)
        return cls(
            task=task,
            params=params,
            num_boost_round=20000,
            early_stopping_rounds=200
        )


# =============================================================================
# Model Class
# =============================================================================
class LGBMModel:
    """LightGBMモデルのラッパークラス"""
    
    def __init__(self, config: LGBMConfig):
        """
        Parameters
        ----------
        config : LGBMConfig
            モデル設定
        """
        self.config = config
        self.models: List[lgb.Booster] = []
        self.oof_predictions: Optional[np.ndarray] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.cv_scores: List[float] = []
        self.best_iterations: List[int] = []
    
    def train_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.ndarray] = None,
        categorical_features: Optional[List[str]] = None
    ) -> lgb.Booster:
        """
        単一モデルの訓練
        
        Parameters
        ----------
        X_train : pd.DataFrame
            訓練データ
        y_train : np.ndarray
            訓練ラベル
        X_valid : pd.DataFrame, optional
            検証データ
        y_valid : np.ndarray, optional
            検証ラベル
        categorical_features : list, optional
            カテゴリカル特徴量のリスト
        
        Returns
        -------
        lgb.Booster
            訓練済みモデル
        """
        # データセット作成
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                categorical_feature=categorical_features,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # モデル訓練
        model = lgb.train(
            self.config.params,
            train_data,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds
                ),
                lgb.log_evaluation(period=self.config.verbose_eval)
            ]
        )
        
        return model
    
    def train_cross_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        categorical_features: Optional[List[str]] = None,
        stratified: bool = True
    ) -> Tuple[np.ndarray, List[lgb.Booster]]:
        """
        交差検証による訓練
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
        y : np.ndarray
            ターゲット
        groups : np.ndarray, optional
            グループ情報（GroupKFold用）
        categorical_features : list, optional
            カテゴリカル特徴量のリスト
        stratified : bool
            層化抽出を行うか
        
        Returns
        -------
        oof_predictions : np.ndarray
            Out-of-Fold予測値
        models : list
            訓練済みモデルのリスト
        """
        print(f"Starting {self.config.n_splits}-Fold Cross Validation...")
        print(f"Task: {self.config.task}")
        print(f"Data shape: {X.shape}")
        
        # 交差検証の分割器
        if stratified and self.config.task in ['binary', 'multiclass']:
            kf = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
            splits = kf.split(X, y)
        else:
            kf = KFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
            splits = kf.split(X)
        
        # OOF予測の初期化
        if self.config.task == 'multiclass':
            num_class = self.config.params.get('num_class', 3)
            oof_predictions = np.zeros((len(X), num_class))
        else:
            oof_predictions = np.zeros(len(X))
        
        models = []
        cv_scores = []
        best_iterations = []
        
        # 各Foldで訓練
        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{self.config.n_splits}")
            print(f"{'='*60}")
            
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            
            # モデル訓練
            model = self.train_single_model(
                X_train, y_train,
                X_valid, y_valid,
                categorical_features
            )
            
            models.append(model)
            best_iterations.append(model.best_iteration)
            
            # 予測
            valid_pred = model.predict(
                X_valid,
                num_iteration=model.best_iteration
            )
            
            if self.config.task == 'multiclass':
                oof_predictions[valid_idx] = valid_pred
                y_pred_class = valid_pred.argmax(axis=1)
                score = self._calculate_metric(y_valid, y_pred_class)
            else:
                oof_predictions[valid_idx] = valid_pred
                score = self._calculate_metric(y_valid, valid_pred)
            
            cv_scores.append(score)
            print(f"Fold {fold + 1} Score: {score:.6f}")
            print(f"Best iteration: {model.best_iteration}")
        
        # 全体スコア
        if self.config.task == 'multiclass':
            overall_pred = oof_predictions.argmax(axis=1)
            overall_score = self._calculate_metric(y, overall_pred)
        else:
            overall_score = self._calculate_metric(y, oof_predictions)
        
        print(f"\n{'='*60}")
        print(f"Overall CV Score: {overall_score:.6f}")
        print(f"CV Scores: {[f'{s:.6f}' for s in cv_scores]}")
        print(f"Mean: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
        print(f"Best iterations: {best_iterations}")
        print(f"{'='*60}\n")
        
        # 結果を保存
        self.models = models
        self.oof_predictions = oof_predictions
        self.cv_scores = cv_scores
        self.best_iterations = best_iterations
        
        return oof_predictions, models
    
    def predict(
        self,
        X: pd.DataFrame,
        use_best_iteration: bool = True
    ) -> np.ndarray:
        """
        予測
        
        Parameters
        ----------
        X : pd.DataFrame
            予測データ
        use_best_iteration : bool
            best_iterationを使用するか
        
        Returns
        -------
        np.ndarray
            予測値
        """
        if not self.models:
            raise ValueError("Model not trained yet. Call train_* method first.")
        
        predictions = []
        
        for i, model in enumerate(self.models):
            num_iteration = model.best_iteration if use_best_iteration else None
            pred = model.predict(X, num_iteration=num_iteration)
            predictions.append(pred)
        
        # 平均を取る
        predictions = np.array(predictions)
        
        if self.config.task == 'multiclass':
            # (n_models, n_samples, n_classes) -> (n_samples, n_classes)
            return predictions.mean(axis=0)
        else:
            # (n_models, n_samples) -> (n_samples,)
            return predictions.mean(axis=0)
    
    def predict_single_model(
        self,
        X: pd.DataFrame,
        model_index: int = 0,
        use_best_iteration: bool = True
    ) -> np.ndarray:
        """
        特定のモデルで予測
        
        Parameters
        ----------
        X : pd.DataFrame
            予測データ
        model_index : int
            使用するモデルのインデックス
        use_best_iteration : bool
            best_iterationを使用するか
        
        Returns
        -------
        np.ndarray
            予測値
        """
        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} out of range")
        
        model = self.models[model_index]
        num_iteration = model.best_iteration if use_best_iteration else None
        
        return model.predict(X, num_iteration=num_iteration)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        特徴量重要度の取得
        
        Parameters
        ----------
        importance_type : str
            'gain' or 'split'
        
        Returns
        -------
        pd.DataFrame
            特徴量重要度
        """
        if not self.models:
            raise ValueError("Model not trained yet.")
        
        importance_df = pd.DataFrame()
        
        for i, model in enumerate(self.models):
            fold_importance = pd.DataFrame({
                'feature': model.feature_name(),
                'importance': model.feature_importance(importance_type=importance_type),
                'fold': i + 1
            })
            importance_df = pd.concat([importance_df, fold_importance], axis=0)
        
        # 平均重要度を計算
        avg_importance = importance_df.groupby('feature')['importance'].agg(['mean', 'std'])
        avg_importance = avg_importance.sort_values('mean', ascending=False)
        
        self.feature_importance = avg_importance
        
        return importance_df
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """評価指標の計算"""
        metric = self.config.params.get('metric', 'auc')
        
        if self.config.task == 'binary':
            if metric == 'auc':
                return roc_auc_score(y_true, y_pred)
            elif metric == 'logloss':
                return log_loss(y_true, y_pred)
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
                return accuracy_score(y_true, y_pred_binary)
        
        elif self.config.task == 'multiclass':
            if metric in ['multi_logloss', 'logloss']:
                return log_loss(y_true, y_pred)
            else:
                return accuracy_score(y_true, y_pred)
        
        else:  # regression
            if metric == 'rmse':
                return np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'mae':
                return mean_absolute_error(y_true, y_pred)
            else:
                return mean_squared_error(y_true, y_pred)
    
    def get_best_params(self) -> Dict[str, Any]:
        """最適なパラメータを取得"""
        return self.config.params.copy()
    
    def save_models(self, filepath: str):
        """モデルの保存"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'config': self.config,
                'oof_predictions': self.oof_predictions,
                'cv_scores': self.cv_scores,
                'best_iterations': self.best_iterations
            }, f)
        print(f"Models saved to {filepath}")
    
    @classmethod
    def load_models(cls, filepath: str):
        """モデルの読み込み"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['config'])
        model.models = data['models']
        model.oof_predictions = data['oof_predictions']
        model.cv_scores = data['cv_scores']
        model.best_iterations = data['best_iterations']
        
        print(f"Models loaded from {filepath}")
        return model



# =============================================================================
# Usage Examples
# =============================================================================
if __name__ == "__main__":
    # 設定作成
    # config = LGBMConfig.binary_classification(
    #     metric='auc',
    #     learning_rate=0.05,
    #     num_leaves=31
    # )
    
    # # モデル訓練
    # model = LGBMModel(config)
    # oof_pred, trained_models = model.train_cross_validation(train, target)
    
    # # 特徴量重要度
    # importance_df = model.get_feature_importance()
    # print("\nTop 10 Important Features:")
    # print(model.feature_importance.head(10))
    
    # # 予測
    # test_pred = model.predict(test)
    # print(f"\nTest predictions (first 10): {test_pred[:10]}")
    
    # === Example 2: Regression ===
    # print("\n" + "="*70)
    # print("[Example 2] Regression")
    # print("-" * 70)
    
    # X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    # X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(20)])
    
    config_reg = LGBMConfig.regression(metric='rmse')
    model_reg = LGBMModel(config_reg)
    oof_pred_reg, _ = model_reg.train_cross_validation(final_df, target)
    pred = model_reg.predict(final_df_te)
    



pred


"""
TabNet Regression Baseline Model for Kaggle Competitions
==========================================================
This script provides a baseline implementation using TabNet for regression tasks.
TabNet is a deep learning model designed for tabular data with built-in interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import warnings
warnings.filterwarnings('ignore')


def get_device():
    """
    Check if CUDA is available and return appropriate device
    
    Returns:
    --------
    device_name : str
        'cuda' if GPU is available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detected: {gpu_name}")
        print(f"✓ GPU memory: {gpu_memory:.2f} GB")
    else:
        device = 'cpu'
        print("✗ GPU not available, using CPU")
        print("  (Install CUDA-enabled PyTorch for GPU acceleration)")
    
    return device


class TabNetRegressionBaseline:
    """
    TabNet Regression Baseline Model for Kaggle Competitions
    
    Parameters:
    -----------
    use_scaler : bool
        Whether to scale the target variable (recommended for large values)
    device : str
        Device to use for training ('auto', 'cuda', or 'cpu')
        'auto' will automatically detect GPU availability
    """
    
    def __init__(self, use_scaler=False, device='auto'):
        self.model = None
        self.label_encoders = {}
        self.cat_indices = []
        self.cat_dims = []
        self.use_scaler = use_scaler
        self.target_scaler = None
        if use_scaler:
            self.target_scaler = StandardScaler()
        
        # Set device (GPU or CPU)
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = device
            print(f"Using device: {device}")
        
    def preprocess_data(self, df, target_col=None, is_train=True):
        """
        Preprocess the data: handle missing values and encode categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of the target column (for training data)
        is_train : bool
            Whether this is training data
        
        Returns:
        --------
        X : np.ndarray
            Features
        y : np.ndarray or None
            Target (None for test data)
        """
        df = df.copy()
        
        # Separate features and target
        if is_train and target_col:
            y = df[target_col].values
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df.copy()
        
        # Identify categorical and numerical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {len(cat_cols)}")
        print(f"Numerical columns: {len(num_cols)}")
        
        # Encode categorical variables
        if is_train:
            self.cat_indices = []
            self.cat_dims = []
            
        for i, col in enumerate(cat_cols):
            if is_train:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                self.cat_indices.append(X.columns.get_loc(col))
                self.cat_dims.append(len(self.label_encoders[col].classes_))
            else:
                # Handle unseen categories in test set
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ 
                    else self.label_encoders[col].classes_[0]
                )
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Handle missing values in numerical columns
        for col in num_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Convert to numpy array
        X = X.values.astype(np.float32)
        
        if y is not None:
            # Scale target if needed
            if self.use_scaler and is_train:
                y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            elif self.use_scaler and not is_train:
                y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
            y = y.astype(np.float32)
        
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, 
              max_epochs=200, patience=50, batch_size=1024, 
              learning_rate=2e-2, verbose=1):
        """
        Train the TabNet regression model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation target
        max_epochs : int
            Maximum number of epochs
        patience : int
            Early stopping patience
        batch_size : int
            Batch size for training
        learning_rate : float
            Initial learning rate
        verbose : int
            Verbosity level
        """
        
        # Initialize model
        self.model = TabNetRegressor(
            n_d=64,  # Width of the decision prediction layer
            n_a=64,  # Width of the attention embedding for each mask
            n_steps=5,  # Number of steps in the architecture
            gamma=1.5,  # Coefficient for feature reusage in the masks
            cat_idxs=self.cat_indices,
            cat_dims=self.cat_dims,
            cat_emb_dim=1,
            lambda_sparse=1e-4,  # Sparsity loss weight
            momentum=0.3,
            clip_value=2.0,
            optimizer_fn=torch.optim.Adam,
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            optimizer_params=dict(lr=learning_rate),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            epsilon=1e-15,
            verbose=verbose,
            seed=42,
            device_name=self.device  # Use GPU if available
        )
        
        # Train the model
        self.model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_val, y_val.reshape(-1, 1))],
            eval_name=['val'],
            eval_metric=['rmse'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        print("\nTraining completed!")
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        predictions = self.model.predict(X).flatten()
        
        # Inverse transform if scaler was used
        if self.use_scaler and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
        
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate the model with multiple regression metrics
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True values
        
        Returns:
        --------
        metrics : dict
            Dictionary containing RMSE, MAE, and R2 scores
        """
        # Get predictions
        predictions = self.predict(X)
        
        # Inverse transform true values if scaler was used
        y_eval = y.copy()
        if self.use_scaler and self.target_scaler is not None:
            y_eval = self.target_scaler.inverse_transform(
                y_eval.reshape(-1, 1)
            ).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_eval, predictions))
        mae = mean_absolute_error(y_eval, predictions)
        r2 = r2_score(y_eval, predictions)
        
        # Calculate RMSLE if all values are positive
        if np.all(y_eval > 0) and np.all(predictions > 0):
            rmsle = np.sqrt(mean_squared_error(
                np.log1p(y_eval), np.log1p(predictions)
            ))
        else:
            rmsle = None
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'RMSLE': rmsle
        }
        
        print("\n=== Evaluation Metrics ===")
        print(f"RMSE:  {rmse:.6f}")
        print(f"MAE:   {mae:.6f}")
        print(f"R²:    {r2:.6f}")
        if rmsle is not None:
            print(f"RMSLE: {rmsle:.6f}")
        print("=" * 26)
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the trained model
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        
        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with feature importance
        """
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def cross_validate(self, X, y, n_splits=5, max_epochs=200, patience=50):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        n_splits : int
            Number of CV folds
        max_epochs : int
            Maximum epochs per fold
        patience : int
            Early stopping patience
        
        Returns:
        --------
        cv_scores : dict
            Dictionary with mean and std of metrics
        oof_predictions : np.ndarray
            Out-of-fold predictions
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        oof_predictions = np.zeros(len(y))
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\n{'='*60}")
            print(f"Training Fold {fold}/{n_splits}")
            print(f"{'='*60}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model for this fold
            self.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                max_epochs=max_epochs,
                patience=patience,
                verbose=0
            )
            
            # Get predictions
            fold_preds = self.predict(X_val_fold)
            oof_predictions[val_idx] = fold_preds
            
            # Evaluate
            y_val_eval = y_val_fold.copy()
            if self.use_scaler and self.target_scaler is not None:
                y_val_eval = self.target_scaler.inverse_transform(
                    y_val_eval.reshape(-1, 1)
                ).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_val_eval, fold_preds))
            mae = mean_absolute_error(y_val_eval, fold_preds)
            r2 = r2_score(y_val_eval, fold_preds)
            
            fold_metrics.append({'RMSE': rmse, 'MAE': mae, 'R2': r2})
            print(f"\nFold {fold} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # Calculate mean and std across folds
        cv_scores = {
            'RMSE_mean': np.mean([m['RMSE'] for m in fold_metrics]),
            'RMSE_std': np.std([m['RMSE'] for m in fold_metrics]),
            'MAE_mean': np.mean([m['MAE'] for m in fold_metrics]),
            'MAE_std': np.std([m['MAE'] for m in fold_metrics]),
            'R2_mean': np.mean([m['R2'] for m in fold_metrics]),
            'R2_std': np.std([m['R2'] for m in fold_metrics])
        }
        
        print(f"\n{'='*60}")
        print("Cross-Validation Results")
        print(f"{'='*60}")
        print(f"RMSE: {cv_scores['RMSE_mean']:.6f} (+/- {cv_scores['RMSE_std']:.6f})")
        print(f"MAE:  {cv_scores['MAE_mean']:.6f} (+/- {cv_scores['MAE_std']:.6f})")
        print(f"R²:   {cv_scores['R2_mean']:.6f} (+/- {cv_scores['R2_std']:.6f})")
        print(f"{'='*60}")
        
        return cv_scores, oof_predictions


def main():
    """
    Main function to run the TabNet regression baseline
    
    Usage:
    ------
    Modify the following variables according to your competition:
    - train_path: path to training data
    - test_path: path to test data
    - target_col: name of the target column
    - id_col: name of the ID column
    - use_cv: whether to use cross-validation
    """
    
    # ===== CONFIGURATION =====
    # Modify these according to your competition
    train_path = '/kaggle/input/playground-series-s5e10/train.csv'  # Path to training data
    test_path = '/kaggle/input/playground-series-s5e10/test.csv'    # Path to test data
    target_col = 'accident_risk'     # Name of the target column
    id_col = 'id'             # Name of the ID column
    use_cv = False            # Use cross-validation (slower but more robust)
    n_splits = 5              # Number of CV folds (if use_cv=True)
    use_scaler = False        # Scale target variable (useful for large values)
    
    # Model hyperparameters
    max_epochs = 200
    patience = 50
    batch_size = 1024
    learning_rate = 2e-2
    # =========================
    
    print("="*60)
    print("TabNet Regression Baseline")
    print("="*60)
    
    # Check GPU availability
    print("\n[Device Information]")
    device = get_device()
    print()
    
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Basic statistics about target
    print(f"\nTarget statistics:")
    print(train_df[target_col].describe())
    
    # Save test IDs for submission
    test_ids = test_df[id_col] if id_col in test_df.columns else None
    
    # Drop ID column
    if id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])
    if id_col in test_df.columns:
        test_df = test_df.drop(columns=[id_col])
    
    # Initialize model
    print(f"\nInitializing TabNet Regression model...")
    print(f"Target scaling: {'Enabled' if use_scaler else 'Disabled'}")
    baseline = TabNetRegressionBaseline(use_scaler=use_scaler, device=device)
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    X_full, y_full = baseline.preprocess_data(train_df, target_col=target_col, is_train=True)
    
    if use_cv:
        # Cross-validation mode
        print(f"\nUsing {n_splits}-fold cross-validation...")
        cv_scores, oof_predictions = baseline.cross_validate(
            X_full, y_full,
            n_splits=n_splits,
            max_epochs=max_epochs,
            patience=patience
        )
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'oof_predictions': oof_predictions,
            'true_values': y_full
        })
        oof_df.to_csv('oof_predictions.csv', index=False)
        print("\nOut-of-fold predictions saved to: oof_predictions.csv")
        
    else:
        # Simple train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        # Train the model
        print("\nTraining TabNet model...")
        baseline.train(
            X_train, y_train, X_val, y_val,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        baseline.evaluate(X_val, y_val)
    
    # Get feature importance
    feature_names = [col for col in train_df.columns if col != target_col]
    importance_df = baseline.get_feature_importance(feature_names)
    print("\nTop 10 Important Features:")
    print(importance_df.head(10))
    importance_df.to_csv('feature_importance.csv', index=False)
    print("\nFeature importance saved to: feature_importance.csv")
    
    # Preprocess test data and make predictions
    print("\nMaking predictions on test data...")
    X_test, _ = baseline.preprocess_data(test_df, is_train=False)
    test_predictions = baseline.predict(X_test)
    
    # Create submission file
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        id_col: test_ids if test_ids is not None else range(len(test_predictions)),
        target_col: test_predictions
    })
    
    
    print("\nSubmission file created: submission.csv")
    print(f"Submission shape: {submission.shape}")
    print("\nFirst few rows of submission:")
    print(submission.head())
    print("\nPrediction statistics:")
    print(submission[target_col].describe())

    return submission


if __name__ == "__main__":
    sub2 = main()


sub2.describe()


sub['accident_risk']=(pred + sub2['accident_risk'])/2.


sub.head()


sub.to_csv('submission.csv',index=False)




