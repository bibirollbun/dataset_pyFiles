# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!/usr/bin/env python3

# =================== Kaggle CPU Baseline (TỐI ƯU HÓA) =====================
# - Giữ lại ý tưởng pipeline và grid search cốt lõi.
# - NÂNG CẤP: Thêm LightGBM (LGBMClassifier), một mô hình boosting mạnh mẽ,
#   vào quá trình đánh giá để tìm ra mô hình tốt nhất.
# - NÂNG CẤP: Thêm bước Feature Engineering dựa trên thời gian và ticker,
#   tạo ra các đặc trưng rolling window và date-based để nắm bắt xu hướng.
# - TỐI ƯU: Tinh chỉnh không gian tìm kiếm và thêm logging chi tiết hơn.
# - GIỮ NGUYÊN: Vẫn sử dụng StratifiedGroupKFold để tránh data leakage giữa các ticker.
# ============================================================================

!pip install pandas numpy scikit-learn imbalanced-learn lightgbm -q

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import hashlib, json, itertools, warnings
from tqdm.auto import tqdm

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# --- Imports từ Scikit-learn, Imbalanced-learn và LightGBM ---
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    Normalizer, MaxAbsScaler, KBinsDiscretizer
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile,
    f_classif, mutual_info_classif, chi2
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb

# ---------------------- DataType utils ----------------------
class DataTypeHandler:
    @staticmethod
    def identify_feature_types(X, threshold=3):
        n_samples, n_features = X.shape
        boolean_mask = np.zeros(n_features, dtype=bool)
        for i in range(n_features):
            col = X[:, i]
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) <= 2:
                boolean_mask[i] = True
        return boolean_mask

    @staticmethod
    def safe_variance_threshold(X, threshold=0.01):
        keep = []
        for i in range(X.shape[1]):
            col = X[:, i].astype(float)
            if np.var(col) > threshold:
                keep.append(i)
        return np.array(keep)

# ---------------------- Pipeline dataclasses (Không thay đổi nhiều) ----------------------
@dataclass
class ProcessingInput:
    X: np.ndarray
    y: np.ndarray
    boolean_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingOutput:
    X: np.ndarray
    y: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    transformation_log: List[str] = field(default_factory=list)

@dataclass
class StepConfig:
    enabled: bool = True
    method: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    config_id: str = ""
    outlier_removal: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
    scaling: StepConfig = field(default_factory=lambda: StepConfig(enabled=True, method="standard"))
    feature_selection: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
    dimensionality_reduction: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
    balancing: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
    def to_dict(self):
        return {
            'outlier_removal': {'enabled': self.outlier_removal.enabled, 'method': self.outlier_removal.method},
            'scaling': {'enabled': self.scaling.enabled, 'method': self.scaling.method},
            'feature_selection': {'enabled': self.feature_selection.enabled, 'method': self.feature_selection.method},
            'dimensionality_reduction': {'enabled': self.dimensionality_reduction.enabled, 'method': self.dimensionality_reduction.method},
            'balancing': {'enabled': self.balancing.enabled, 'method': self.balancing.method}
        }
    def generate_id(self):
        self.config_id = hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
        return self.config_id

# ---------------------- Steps (Không thay đổi) ----------------------
# Giữ nguyên các class Step vì chúng đã được thiết kế module hóa tốt
class ExpandedOutlierRemovalStep:
    def __init__(self, config: StepConfig): self.config = config; self.bounds = {}; self.boolean_mask = None
    def fit(self, data: ProcessingInput):
        if not self.config.enabled: return self
        self.boolean_mask = data.boolean_mask if data.boolean_mask is not None else np.zeros(data.X.shape[1], dtype=bool)
        cont_idx = np.where(~self.boolean_mask)[0]
        if len(cont_idx) == 0: return self
        method = self.config.method
        if method == "iqr":
            mult = self.config.params.get('multiplier', 1.5)
            for i in cont_idx:
                Q1, Q3 = np.percentile(data.X[:, i], 25), np.percentile(data.X[:, i], 75)
                IQR = Q3 - Q1; self.bounds[i] = (Q1 - mult*IQR, Q3 + mult*IQR)
        elif method == "percentile":
            lo, hi = self.config.params.get('lower', 1), self.config.params.get('upper', 99)
            for i in cont_idx: self.bounds[i] = (np.percentile(data.X[:, i], lo), np.percentile(data.X[:, i], hi))
        return self
    def transform(self, data: ProcessingInput):
        if not self.config.enabled: return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata, transformation_log=["outlier: skipped"])
        X = data.X.copy().astype(float)
        for i,(lo,hi) in self.bounds.items(): X[:, i] = np.clip(X[:, i], lo, hi)
        return ProcessingOutput(X=X, y=data.y, metadata=data.metadata, transformation_log=[f"outlier:{self.config.method}"])

class ExpandedScalingStep:
    def __init__(self, config: StepConfig): self.config = config; self.scaler = None; self.boolean_mask = None; self.cont_idx = None
    def fit(self, data: ProcessingInput):
        if not self.config.enabled: return self
        self.boolean_mask = data.boolean_mask if data.boolean_mask is not None else np.zeros(data.X.shape[1], dtype=bool)
        self.cont_idx = np.where(~self.boolean_mask)[0]
        if len(self.cont_idx)==0: return self
        Xc = data.X[:, self.cont_idx].astype(float)
        m = self.config.method
        if m == "standard": self.scaler = StandardScaler()
        elif m == "robust": self.scaler = RobustScaler()
        elif m == "minmax": self.scaler = MinMaxScaler()
        elif m == "quantile_uniform": self.scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=min(1000, len(Xc)), random_state=42)
        else: self.scaler = StandardScaler()
        try: self.scaler.fit(Xc)
        except: self.scaler = StandardScaler().fit(Xc)
        return self
    def transform(self, data: ProcessingInput):
        if not self.config.enabled or self.scaler is None: return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata, transformation_log=["scaling: skipped"])
        X = data.X.copy().astype(float)
        if len(self.cont_idx)>0:
            try: X[:, self.cont_idx] = self.scaler.transform(X[:, self.cont_idx])
            except: pass
        return ProcessingOutput(X=X, y=data.y, metadata=data.metadata, transformation_log=[f"scaling:{self.config.method}"])

class ExpandedFeatureSelectionStep:
    def __init__(self, config: StepConfig): self.config = config; self.selected = None
    def fit(self, data: ProcessingInput):
        if not self.config.enabled: return self
        Xf, y = data.X.astype(float), data.y
        m = self.config.method
        k = min(self.config.params.get('k', 500), Xf.shape[1])
        if m == "kbest_f": self.selected = np.where(SelectKBest(f_classif, k=k).fit(Xf, y).get_support())[0]
        elif m == "kbest_mutual": self.selected = np.where(SelectKBest(mutual_info_classif, k=k).fit(Xf, y).get_support())[0]
        if self.selected is None or len(self.selected) < 10: self.selected = np.arange(min(500, Xf.shape[1]))
        return self
    def transform(self, data: ProcessingInput):
        if not self.config.enabled or self.selected is None: return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata, transformation_log=["feature: skipped"])
        return ProcessingOutput(X=data.X[:, self.selected], y=data.y, metadata=data.metadata, transformation_log=[f"feature:{self.config.method}({len(self.selected)})"])

class ExpandedDimensionalityReductionStep:
    def __init__(self, config: StepConfig): self.config = config; self.reducer = None
    def fit(self, data: ProcessingInput):
        if not self.config.enabled: return self
        n_components = min(self.config.params.get('n_components', 100), min(data.X.shape)-1)
        if n_components < 2: return self
        m = self.config.method
        try:
            if m == "pca": self.reducer = PCA(n_components=n_components, random_state=42).fit(data.X)
            elif m == "svd": self.reducer = TruncatedSVD(n_components=n_components, random_state=42).fit(data.X)
        except: self.reducer = None
        return self
    def transform(self, data: ProcessingInput):
        if not self.config.enabled or self.reducer is None: return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata, transformation_log=["dim: skipped"])
        try: X = self.reducer.transform(data.X.astype(float))
        except: X = data.X
        return ProcessingOutput(X=X, y=data.y, metadata=data.metadata, transformation_log=[f"dim:{self.config.method}"])

class ExpandedBalancingStep:
    def __init__(self, config: StepConfig): self.config = config; self.balancer = None
    def fit(self, data: ProcessingInput):
        if not self.config.enabled: return self
        self.balancer = RandomOverSampler(random_state=42)
        return self
    def transform(self, data: ProcessingInput):
        if not self.config.enabled or self.balancer is None: return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata, transformation_log=["balance: skipped"])
        Xb, yb = self.balancer.fit_resample(data.X.astype(float), data.y)
        return ProcessingOutput(X=Xb, yb=yb, metadata=data.metadata, transformation_log=["balance: oversample"])

# ---------------------- Comprehensive Pipeline (Không thay đổi) ----------------------
class ComprehensivePipeline:
    def __init__(self, config: PipelineConfig, boolean_mask=None):
        self.config = config
        self.boolean_mask = boolean_mask.copy() if boolean_mask is not None else None
        self.steps = [
            ExpandedOutlierRemovalStep(self.config.outlier_removal),
            ExpandedScalingStep(self.config.scaling),
            ExpandedFeatureSelectionStep(self.config.feature_selection),
            ExpandedDimensionalityReductionStep(self.config.dimensionality_reduction),
            ExpandedBalancingStep(self.config.balancing)
        ]
    def fit(self, X: np.ndarray, y: np.ndarray):
        data = ProcessingInput(X=X, y=y, boolean_mask=self.boolean_mask)
        temp_bool_mask = self.boolean_mask
        for step in self.steps:
            step.fit(data)
            out = step.transform(data)
            if isinstance(step, ExpandedFeatureSelectionStep) and temp_bool_mask is not None and hasattr(step, "selected"):
                temp_bool_mask = temp_bool_mask[step.selected] if step.selected is not None else temp_bool_mask
            data = ProcessingInput(X=out.X, y=out.y, boolean_mask=temp_bool_mask)
        return self
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        data = ProcessingInput(X=X, y=y if y is not None else np.zeros(X.shape[0]), boolean_mask=self.boolean_mask)
        temp_bool_mask = self.boolean_mask
        for i, step in enumerate(self.steps):
            if i == len(self.steps) - 1 and (y is None or not self.config.balancing.enabled):
                break
            out = step.transform(data)
            if isinstance(step, ExpandedFeatureSelectionStep) and temp_bool_mask is not None and hasattr(step, "selected"):
                 temp_bool_mask = temp_bool_mask[step.selected] if step.selected is not None else temp_bool_mask
            data = ProcessingInput(X=out.X, y=out.y, boolean_mask=temp_bool_mask)
        return data.X, data.y
    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X, y)

# ---------------------- NÂNG CẤP: Feature Engineering ----------------------
def create_time_series_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Tạo các đặc trưng mới từ cột thời gian và theo từng ticker."""
    print("Bắt đầu tạo đặc trưng chuỗi thời gian...")
    df_out = df.copy()
    
    # Chuyển cột 't' sang dạng datetime
    df_out['t'] = pd.to_datetime(df_out['t'])
    
    # 1. Đặc trưng dựa trên ngày tháng
    df_out['time_month'] = df_out['t'].dt.month
    df_out['time_week'] = df_out['t'].dt.isocalendar().week.astype(int)
    df_out['time_dayofweek'] = df_out['t'].dt.dayofweek
    
    # 2. Đặc trưng rolling window (chỉ chọn 10 cột đầu tiên để làm ví dụ)
    # Trong thực tế, bạn có thể cần phân tích để chọn các cột quan trọng hơn
    rolling_cols = feature_cols[:10] 
    windows = [3, 7, 14] # Các cửa sổ thời gian
    
    # Sắp xếp để đảm bảo rolling hoạt động đúng
    df_out = df_out.sort_values(by=['ticker_id', 't'])
    
    for ticker_id, group in tqdm(df_out.groupby('ticker_id'), desc="Tạo Rolling Features"):
        for col in rolling_cols:
            for w in windows:
                # Rolling mean
                df_out.loc[group.index, f'{col}_roll_mean_{w}'] = group[col].rolling(window=w, min_periods=1).mean()
                # Rolling std
                df_out.loc[group.index, f'{col}_roll_std_{w}'] = group[col].rolling(window=w, min_periods=1).std()

    # Điền các giá trị NaN có thể phát sinh từ rolling
    df_out.fillna(method='bfill', inplace=True)
    
    print(f"Hoàn thành! Đã thêm {len(df_out.columns) - len(df.columns)} đặc trưng mới.")
    return df_out

# ---------------------- NÂNG CẤP: Orchestrator ----------------------
class GridSearchOrchestrator:
    def __init__(self, X_train, y_train, boolean_mask, groups):
        self.X_train = X_train
        self.y_train = y_train
        self.boolean_mask = boolean_mask
        self.groups = groups
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        self.best_model = None
        self.best_pipeline = None

    def generate_all_configurations(self):
        # Tinh chỉnh lại các lựa chọn để tập trung vào những phương pháp hiệu quả
        outlier_options = [StepConfig(enabled=False), StepConfig(enabled=True, method="iqr")]
        scaling_options = [
            StepConfig(enabled=False),
            StepConfig(enabled=True, method="standard"),
            StepConfig(enabled=True, method="robust"),
            StepConfig(enabled=True, method="minmax"),
        ]
        feature_selection_options = [
            StepConfig(enabled=False),
            StepConfig(enabled=True, method="kbest_f", params={'k': 500}),
        ]
        dim_reduction_options = [
            StepConfig(enabled=False),
            StepConfig(enabled=True, method="svd", params={'n_components': 128}),
        ]
        balancing_options = [StepConfig(enabled=False), StepConfig(enabled=True)]

        combos = list(itertools.product(
            outlier_options, scaling_options, feature_selection_options,
            dim_reduction_options, balancing_options
        ))
        return [PipelineConfig(outlier_removal=c[0], scaling=c[1], feature_selection=c[2],
                               dimensionality_reduction=c[3], balancing=c[4]).generate_id() and c for c in
                [PipelineConfig(outlier_removal=c[0], scaling=c[1], feature_selection=c[2],
                                dimensionality_reduction=c[3], balancing=c[4]) for c in combos]]

    def evaluate_configuration(self, config, n_splits=5):
        try:
            pipe = ComprehensivePipeline(config, self.boolean_mask)
            Xp, yp = pipe.fit_transform(self.X_train, self.y_train)

            classes = np.unique(yp)
            cw = compute_class_weight('balanced', classes=classes, y=yp)
            cw_dict = dict(zip(classes, cw))

            # NÂNG CẤP: Thêm LGBM vào danh sách mô hình cần đánh giá
            models = {
                "logreg": LogisticRegression(max_iter=1000, class_weight=cw_dict, random_state=42),
                "ridge": RidgeClassifier(class_weight='balanced', random_state=42),
                "lgbm": lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            }

            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            best_local_score = -np.inf
            best_local_model_name = None
            
            for name, mdl in models.items():
                # Sử dụng cross_val_score để code gọn hơn
                scores = cross_val_score(mdl, Xp, yp, cv=cv, groups=self.groups, scoring='accuracy', n_jobs=-1)
                mean_acc = np.mean(scores)
                
                if mean_acc > best_local_score:
                    best_local_score = mean_acc
                    best_local_model_name = name

            best_model_instance = models[best_local_model_name]
            best_model_instance.fit(Xp, yp)
            
            return {
                'config': config, 'cv_acc': best_local_score,
                'model': best_model_instance, 'model_name': best_local_model_name,
                'pipeline': pipe, 'shape': Xp.shape
            }
        except Exception as e:
            print(f"Lỗi khi đánh giá config: {e}")
            return None

    def run_grid_search(self, max_configs=18, n_splits=5):
        all_cfgs = self.generate_all_configurations()
        total = len(all_cfgs)
        print(f"\nTổng không gian cấu hình: {total} | Thử nghiệm {min(max_configs, total)} cấu hình.")
        
        idx = np.linspace(0, total - 1, num=min(max_configs, total), dtype=int)
        cfgs_to_test = [all_cfgs[i] for i in idx]

        for i, cfg in enumerate(tqdm(cfgs_to_test, desc="Grid Search Progress"), 1):
            res = self.evaluate_configuration(cfg, n_splits)
            if res is None:
                continue
            
            print(f"Config {i}/{len(cfgs_to_test)} (ID={cfg.config_id}) | "
                  f"Model: {res['model_name']} | CV ACC={res['cv_acc']:.4f} | Shape={res['shape']}")

            if res['cv_acc'] > self.best_score:
                self.best_score = res['cv_acc']
                self.best_config = res['config']
                self.best_model = res['model']
                self.best_pipeline = res['pipeline']
            self.results.append(res)
            
        self._print_summary()
        return self.best_config, self.best_model, self.best_pipeline

    def _print_summary(self):
        print("\n" + "="*80)
        print("TÓM TẮT KẾT QUẢ GRID SEARCH (metric: Accuracy)")
        print("="*80)
        if not self.results:
            print("Không có cấu hình nào hợp lệ.")
            return

        top_5 = sorted(self.results, key=lambda r: r['cv_acc'], reverse=True)[:5]
        print(f"Đã thử: {len(self.results)} cấu hình | CV ACC Tốt nhất: {self.best_score:.4f} | ID tốt nhất: {self.best_config.config_id}")
        
        print("\nTop 5 Cấu hình tốt nhất:")
        for i, r in enumerate(top_5, 1):
            cfg = r['config']
            parts = [
                f"outlier:{cfg.outlier_removal.method}" if cfg.outlier_removal.enabled else "no_outlier",
                f"scale:{cfg.scaling.method}" if cfg.scaling.enabled else "no_scale",
                f"feat_sel:{cfg.feature_selection.method}" if cfg.feature_selection.enabled else "no_feat_sel",
                f"dim_red:{cfg.dimensionality_reduction.method}" if cfg.dimensionality_reduction.enabled else "no_dim_red",
                f"balance:on" if cfg.balancing.enabled else "no_balance",
            ]
            print(f"  {i}. ACC={r['cv_acc']:.4f} | Model: {r['model_name']:<7} | Cấu hình: {' | '.join(parts)}")

# ---------------------- Main ----------------------
def main():
    # --- Các hằng số có thể cấu hình ---
    DATA_ROOT = "/kaggle/input/detecting-reversal-points-in-us-equities/competition_data" # Thay đổi thành "/kaggle/input/..." nếu chạy trên Kaggle
    BEST_CONFIGS_TO_TEST = 20 # Số lượng cấu hình để thử nghiệm
    CV_SPLITS = 5 # Số fold cho cross-validation

    print("1. Đang tải dữ liệu...")
    try:
        train_df = pd.read_csv(f"{DATA_ROOT}/train.csv")
        test_df = pd.read_csv(f"{DATA_ROOT}/test.csv")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file train.csv/test.csv. Hãy đảm bảo chúng ở trong thư mục DATA_ROOT.")
        return

    if set(train_df["class_label"].dropna().unique()) - {"H","L","None"}:
        mapping = {"HH":"H","LH":"H","HL":"L","LL":"L"}
        train_df["class_label"] = train_df["class_label"].map(mapping).fillna("None")

    meta_cols = [c for c in ['ticker_id', 't', 'class_label', 'id'] if c in train_df.columns or c in test_df.columns]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    # NÂNG CẤP: Áp dụng Feature Engineering
    train_df = create_time_series_features(train_df, feature_cols)
    test_df = create_time_series_features(test_df, feature_cols)
    
    # Cập nhật lại danh sách các cột đặc trưng
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    
    print("\n2. Chuẩn bị dữ liệu cho mô hình...")
    X_train = train_df[feature_cols].values
    y_train_str = train_df["class_label"].values
    classes = ['H','L','None']
    class_map = {label: i for i, label in enumerate(classes)}
    y_train = np.array([class_map[c] for c in y_train_str])
    X_test = test_df[feature_cols].values

    # Xử lý NaN, inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

    print("Đang xác định loại đặc trưng (boolean/continuous)...")
    boolean_mask = DataTypeHandler.identify_feature_types(X_train)
    print(f"Tìm thấy: {boolean_mask.sum()} boolean, {X_train.shape[1] - boolean_mask.sum()} continuous.")

    groups = train_df["ticker_id"].astype(str).values

    print("\n3. Bắt đầu Grid Search để tìm pipeline và mô hình tốt nhất...")
    orchestrator = GridSearchOrchestrator(X_train, y_train, boolean_mask, groups)
    best_config, best_model, best_pipeline = orchestrator.run_grid_search(max_configs=BEST_CONFIGS_TO_TEST, n_splits=CV_SPLITS)

    if best_model is not None:
        print("\n4. Huấn luyện mô hình tốt nhất trên toàn bộ dữ liệu training...")
        # Không cần fit_transform lại, vì pipeline đã được fit trong quá trình tìm kiếm
        # Chỉ cần transform dữ liệu train và test
        X_train_final, y_train_final = best_pipeline.transform(X_train, y_train)
        X_test_final, _ = best_pipeline.transform(X_test)
        
        print(f"Huấn luyện mô hình {type(best_model).__name__}...")
        best_model.fit(X_train_final, y_train_final)

        print("5. Dự đoán trên tập test và tạo file submission...")
        pred_id = best_model.predict(X_test_final)
        pred_str = [classes[i] for i in pred_id]

        sub = pd.DataFrame({
            "id": test_df["id"],
            "class_label": pred_str
        })
        sub.to_csv("submission.csv", index=False)
        print("\nĐã lưu file submission.csv thành công!")
        print("Phân phối dự đoán:")
        print(sub["class_label"].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

if __name__ == "__main__":
    main()

