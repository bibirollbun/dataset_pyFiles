%%capture
!pip install laspy
!pip install rasterio


import os
import numpy as np
import pandas as pd
import laspy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, ConfusionMatrixDisplay, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Hàm trích xuất đặc trưng nâng cao
def compute_cloud_features(path: str, *, height_threshold: float = 1.0) -> dict:
    las = laspy.read(path)
    X, Y, Z = np.rollaxis(las.xyz, axis=1)
    height_mask = Z > height_threshold
    X, Y, Z = X[height_mask], Y[height_mask], Z[height_mask]

    # Đặc trưng cơ bản
    stats = {
        "z_max": Z.max(),
        "z_mean": Z.mean(),
        "z_std": Z.std(),
        "z_skew": scipy.stats.skew(Z),
        "z_kurt": scipy.stats.kurtosis(Z),
        "pct_z_above_mean": np.mean(Z > Z.mean()),
        "pct_z_above_2": np.mean(Z > 2),
        "point_density": Z.size / (X.ptp() * Y.ptp() * Z.ptp() + 1e-10),
        "distance_std": np.std(np.sqrt(X**2 + Y**2 + Z**2))
    }

    # Phân vị và tỷ lệ tích lũy
    percentiles = np.percentile(Z, range(10, 100, 10))
    counts, _ = np.histogram(Z, bins=np.linspace(Z.min(), Z.max(), 11))
    cum_ratios = np.cumsum(counts / Z.size)[:-1]
    
    for i, (z, ratio) in enumerate(zip(percentiles, cum_ratios), 1):
        stats.update({f"z_decile_{i}": z, f"z_cumrat_{i}": ratio})

    # Đặc trưng hình học 3D
    covariance_matrix = np.cov(np.vstack((X, Y, Z)))
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)[::-1]
    lambda1, lambda2, lambda3 = eigenvalues
    
    shape_features = {
        "linearity": (lambda1 - lambda2) / (lambda1 + 1e-10),
        "planarity": (lambda2 - lambda3) / (lambda1 + 1e-10),
        "scatter": lambda3 / (lambda1 + 1e-10),
        "omnivariance": np.cbrt(lambda1 * lambda2 * lambda3),
        "eigentropy": -sum(e * np.log(e + 1e-10) for e in eigenvalues),
        "sum_of_eigenvalues": eigenvalues.sum(),
        "curvature": lambda3 / (eigenvalues.sum() + 1e-10),
    }
    
    return {**stats, **shape_features}

# Hàm tạo DataFrame với xử lý tên file
def create_feature_dataframe(folder, label_map):
    data = []
    file_paths = []
    
    for class_name, label in label_map.items():
        class_folder = os.path.join(folder, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: {class_folder} không tồn tại.")
            continue
            
        for sub_class in os.listdir(class_folder):
            sub_class_path = os.path.join(class_folder, sub_class)
            if not os.path.isdir(sub_class_path):
                continue
                
            for file in os.listdir(sub_class_path):
                if file.endswith(".las"):
                    file_path = os.path.join(sub_class_path, file)
                    try:
                        features = compute_cloud_features(file_path)
                        features.update({"name": file.replace(".las", ""), "label": label})
                        data.append(features)
                        file_paths.append(file_path)
                    except Exception as e:
                        print(f"Lỗi xử lý {file_path}: {str(e)}")
    
    df = pd.DataFrame(data)
    path_df = pd.DataFrame({"path": file_paths})
    path_df["species"] = path_df["path"].str.split("/").str[-2]
    conifers = {"Spruce", "Fir", "Pine"}
    path_df["type"] = path_df["species"].map(lambda x: "Coniferous" if x in conifers else "Deciduous")
    
    return pd.concat([path_df, df], axis=1).reset_index(drop=True)

# Khởi tạo đường dẫn
TRAIN_FOLDER = "/kaggle/input/hutechaichallenge2024-bc/Train"
TEST_FOLDER = "/kaggle/input/hutechaichallenge2024-bc/Test"
label_map = {"Coniferous": 0, "Deciduous": 1}

# Kiểm tra và tạo DataFrame
if not os.path.exists(TRAIN_FOLDER) or not os.path.exists(TEST_FOLDER):
    raise FileNotFoundError("Thư mục dữ liệu không tồn tại!")

print("Trích xuất đặc trưng từ tập huấn luyện...")
train_df = create_feature_dataframe(TRAIN_FOLDER, label_map)
print("Trích xuất đặc trưng từ tập kiểm tra...")
test_df = create_feature_dataframe(TEST_FOLDER, label_map)

# Chuẩn bị dữ liệu
X = train_df.drop(columns=["label", "name", "path", "species", "type"])
y = train_df["type"].map({"Coniferous": 0, "Deciduous": 1})
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# Phân chia dữ liệu
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df["species"]
)

# Pipeline nâng cao
class_weights = {0: 1, 1: 5}  # Điều chỉnh dựa trên phân phối lớp
pipeline = Pipeline([
    ('smote_tomek', SMOTETomek(random_state=42)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('stacking', StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(scale_pos_weight=5)),
            ('lgbm', LGBMClassifier(class_weight=class_weights)),
            ('catboost', CatBoostClassifier(verbose=0, class_weights=class_weights))
        ],
        final_estimator=RandomForestClassifier(n_estimators=100, class_weight=class_weights),
        n_jobs=-1
    ))
])

# Tối ưu siêu tham số với GridSearchCV
param_grid = {
    'stacking__xgb__n_estimators': [200, 300, 400],
    'stacking__xgb__max_depth': [5, 7, 9],
    'stacking__xgb__learning_rate': [0.01, 0.1, 0.2],
    'stacking__lgbm__n_estimators': [100, 150, 200],
    'stacking__lgbm__max_depth': [5, 7, 9],
    'stacking__catboost__iterations': [200, 300, 400],
    'stacking__catboost__depth': [6, 8, 10],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=3
)

print("Bắt đầu tối ưu siêu tham số...")
grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Đánh giá cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Độ chính xác trung bình (10-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Dự đoán và đánh giá
y_pred = best_pipeline.predict(X_val)
y_proba = best_pipeline.predict_proba(X_val)[:, 1]
print("\nBáo cáo phân loại:")
print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))
print("F1-Score:", f1_score(y_val, y_pred))

# Xuất file submission
test_features = test_df.drop(columns=["name", "path", "species", "type", "label"])
test_features = test_features[X_train.columns]
test_predictions = best_pipeline.predict(test_features)

submission = pd.DataFrame({"name": test_df["name"], "label": test_predictions})
submission = submission.sort_values(by="name")
submission.to_csv("submission.csv", index=False)

print("\n5 dòng đầu file submission:")
print(submission.head())

