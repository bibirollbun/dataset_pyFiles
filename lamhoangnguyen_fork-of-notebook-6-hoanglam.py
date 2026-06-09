import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
import itertools



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Tạo DataFrame tổng hợp
# ------------------------------

results_data = [

    # Notebook 1 - Baseline
    {'model': 'LogisticRegression (MI)', 'modality':'IMU', 'f1_macro':0.330270, 'accuracy':0.425000, 'train_time':0.014678, 'n_params':0.010},
    {'model': 'LogisticRegression (PCA)', 'modality':'IMU', 'f1_macro':0.309983, 'accuracy':0.416667, 'train_time':0.010892, 'n_params':0.010},

    # Notebook 3 - Transformer
    {'model': 'Vanilla Transformer', 'modality':'IMU', 'f1_macro':0.854641, 'accuracy':0.851608, 'train_time':10784.24, 'n_params':0.352},

    # Notebook 4 - Fusion
    {'model': 'EARLY_ALL', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.370722, 'accuracy':0.4711, 'train_time':40.19, 'n_params':0.405},
    {'model': 'LATE_ALL', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.278066, 'accuracy':0.3533, 'train_time':53.84, 'n_params':0.142},
    {'model': 'MOE_ALL', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.258067, 'accuracy':0.3324, 'train_time':54.67, 'n_params':0.251},
    {'model': 'ATTN_ALL', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.247515, 'accuracy':0.3142, 'train_time':53.99, 'n_params':0.093},
    {'model': 'IMU_ONLY', 'modality':'IMU', 'f1_macro':0.144073, 'accuracy':0.1816, 'train_time':6.859, 'n_params':0.039},
    {'model': 'THERMO_ONLY', 'modality':'Thermo', 'f1_macro':0.131439, 'accuracy':0.2042, 'train_time':6.476, 'n_params':0.038},
    {'model': 'TOF_ONLY', 'modality':'ToF', 'f1_macro':0.119538, 'accuracy':0.1868, 'train_time':52.26, 'n_params':0.075},

    # Notebook 5 - Representation Learning
    {'model': 'SimCLR:LinearProbe', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.3249, 'accuracy':0.6986, 'train_time':401.88, 'n_params':0.298},
    {'model': 'SimCLR:FineTune', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.3693, 'accuracy':0.7547, 'train_time':417.34, 'n_params':0.266},
    {'model': 'FromScratch', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.3616, 'accuracy':0.7628, 'train_time':17.33, 'n_params':0.266},
    {'model': 'MAE:LinearProbe', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.3633, 'accuracy':0.7116, 'train_time':143.34, 'n_params':0.457},
    {'model': 'MAE:FineTune', 'modality':'IMU+ToF+Thermo', 'f1_macro':0.3775, 'accuracy':0.7660, 'train_time':160.72, 'n_params':0.457},

    # Notebook 6 - Graph / Spatial
    {'model': 'GCN', 'modality':'ToF', 'f1_macro':0.1956, 'accuracy':0.6426, 'train_time':16.70, 'n_params':0.108},
    {'model': 'GAT', 'modality':'ToF', 'f1_macro':0.1956, 'accuracy':0.6426, 'train_time':31.47, 'n_params':0.141},
    {'model': 'CNN2D+GRU', 'modality':'ToF', 'f1_macro':0.3496, 'accuracy':0.7019, 'train_time':16.93, 'n_params':0.842},
    {'model': 'CNN3D', 'modality':'ToF', 'f1_macro':0.1956, 'accuracy':0.6426, 'train_time':9.04, 'n_params':0.070},

]

results = pd.DataFrame(results_data)

# Hiển thị bảng
results



# ------------------------------
# Trade-off: Accuracy/F1 vs Training Time vs Params
# ------------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=results,
    x='train_time',
    y='f1_macro',
    size='n_params',
    hue='modality',
    palette='Set2',
    sizes=(50,500)
)
plt.xlabel("Training Time (s)")
plt.ylabel("Macro F1 Score")
plt.title("Trade-off: Accuracy/F1 vs Training Time vs Params")
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()



# ------------------------------
# Heatmap: Macro F1 Score per Model & Modality
# ------------------------------
perf_table = results.pivot_table(index='model', columns='modality', values='f1_macro')
plt.figure(figsize=(12,6))
sns.heatmap(perf_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Macro F1 Score Heatmap")
plt.show()



# ------------------------------
# Ranking by Modality & Model
# ------------------------------
modality_ranking = results.groupby('modality')['f1_macro'].mean().sort_values(ascending=False)
model_ranking = results.groupby('model')['f1_macro'].mean().sort_values(ascending=False)

print("Modality Ranking:\n", modality_ranking)
print("\nModel Ranking:\n", model_ranking)


