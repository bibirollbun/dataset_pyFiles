%%capture
!pip install laspy


import os
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, make_scorer, balanced_accuracy_score
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


def doc_du_lieu_lidar(duong_dan_file):
    with laspy.open(duong_dan_file) as f:
        las = f.read()
        return np.vstack((las.x, las.y, las.z)).transpose()

def trich_xuat_dac_trung(points):
    z = points[:, 2]
    
    
    height = np.max(z) - np.min(z)
    mean_z = np.mean(z)
    std_z = np.std(z)
    cv_z = std_z / mean_z if mean_z != 0 else 0
    percentiles = np.percentile(z, [10, 25, 50, 75, 90])

    skewness = skew(z)
    kurtosis_val = kurtosis(z)
    
   
    hist, _ = np.histogram(z, bins=10)
    hist = hist / len(z)
    
    # Mật độ điểm
    x_range = np.ptp(points[:, 0])
    y_range = np.ptp(points[:, 1])
    density = len(points) / (x_range * y_range) if (x_range * y_range) != 0 else 0
    
    return np.concatenate([
        [density, height, mean_z, std_z, cv_z, skewness, kurtosis_val],
        percentiles,
        hist
    ])

# ================== ĐỌC DỮ LIỆU TỪ THƯ MỤC ==================
def doc_du_lieu_tu_thu_muc(duong_dan, anh_xa_nhan=None):
    X, y, filenames = [], [], []
    
    for class_name in os.listdir(duong_dan):
        class_dir = os.path.join(duong_dan, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.endswith('.las'):
                    points = doc_du_lieu_lidar(os.path.join(class_dir, fname))
                    features = trich_xuat_dac_trung(points)
                    X.append(features)
                    filenames.append(os.path.splitext(fname)[0])
                    if anh_xa_nhan:
                        y.append(anh_xa_nhan[class_name])
    
    return np.array(X), np.array(y), filenames

# ================== CẤU HÌNH ==================
DUONG_DAN_TRAIN = '/kaggle/input/hutechaichallenge2024-mc/Train'
DUONG_DAN_TEST = '/kaggle/input/hutechaichallenge2024-mc/Test'
ANH_XA_NHAN = {
    'Fir': 0,
    'Pine': 1,
    'Spruce': 2,
    'Alder': 3,
    'Aspen': 4,
    'Birch': 5,
    'Tilia': 6
}

# ================== TIỀN XỬ LÝ DỮ LIỆU ==================
X_train, y_train, _ = doc_du_lieu_tu_thu_muc(DUONG_DAN_TRAIN, ANH_XA_NHAN)
X_test, _, test_files = doc_du_lieu_tu_thu_muc(DUONG_DAN_TEST)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Class distribution:", Counter(y_train))


scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}


X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)


pipeline = ImbPipeline([
    ('smote', SMOTE(
        sampling_strategy='auto',  
        k_neighbors=2,
        random_state=42
    )),
    ('xgb', XGBClassifier(
        objective='multi:softprob',
        eval_metric='merror',
        random_state=42,
        use_label_encoder=False,
        early_stopping_rounds=10
    ))
])

param_grid = {
    'xgb__max_depth': [3, 5],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__n_estimators': [100, 200],
    'xgb__subsample': [0.7, 0.8],
    'xgb__reg_alpha': [0.5, 1],
    'xgb__reg_lambda': [0.5, 1]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
             'f1_macro': 'f1_macro'},
    refit='balanced_accuracy',
    n_jobs=-1,
    verbose=1,
    error_score='raise'  
)


fit_params = {'xgb__eval_set': [(X_val, y_val)]}
grid_search.fit(X_train_split, y_train_split, **fit_params)


best_model = grid_search.best_estimator_
cv_results = grid_search.cv_results_

print("Best Parameters:", grid_search.best_params_)
print("Best Balanced Accuracy:", grid_search.best_score_)


cv_scores = cross_val_score(
    best_model,
    X_train_scaled,
    y_train,
    cv=StratifiedKFold(10, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    n_jobs=-1
)
print(f"\nBalanced Accuracy 10-fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


y_pred = best_model.predict(X_train_scaled)
print("\nClassification Report:")
print(classification_report(y_train, y_pred, target_names=ANH_XA_NHAN.keys()))


ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_train_scaled,
    y_train,
    normalize='true',
    display_labels=ANH_XA_NHAN.keys(),
    cmap=plt.cm.Blues
)
plt.xticks(rotation=45)
plt.title("Normalized Confusion Matrix")
plt.show()


y_test_pred = best_model.predict(X_test_scaled)
result = pd.DataFrame({'name': test_files, 'label': y_test_pred})
result.to_csv('submission.csv', index=False)
print("\n=== Đã tạo file submission.csv thành công ===")

