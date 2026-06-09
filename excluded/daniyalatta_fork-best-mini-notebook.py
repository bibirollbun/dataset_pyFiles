import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


INPUT_PATH = '/kaggle/input/'
WORKING_PATH = '/kaggle/working/'

test_df = pd.read_csv(INPUT_PATH + "playground-series-s5e8/test.csv")
submissions = {
    'LightGBM': pd.read_csv(INPUT_PATH + "ps-s5e8-lightgb-model-add-original-dataset/submission.csv"),
    'XGB_NN': pd.read_csv(INPUT_PATH + "train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv"),
    'Best_Model': pd.read_csv(INPUT_PATH + "21-august-2025-ps-s5e8/submission 0.977621.csv")
}


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


predictions_df = test_df[['id']].copy()
for name, sub in submissions.items():
    predictions_df[name] = sub['y']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for i, (name, preds) in enumerate(predictions_df.iloc[:, 1:].items()):
    ax = axes[i//2, i%2]
    sns.histplot(preds, bins=50, ax=ax, kde=True)
    ax.set_title(f'{name}')

plt.tight_layout()
plt.savefig(WORKING_PATH + 'predictions_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

def create_ensemble(predictions, weights):
    ensemble = pd.Series(0, index=predictions.index)
    total_weight = 0
    for model, weight in weights.items():
        ensemble += predictions[model] * weight
        total_weight += weight
    return ensemble / total_weight

ensemble_configs = [
    {'name': 'ensemble1', 'weights': {'LightGBM': 0.23, 'XGB_NN': 0.77}},
    {'name': 'ensemble2', 'weights': {'LightGBM': 0.45, 'XGB_NN': 0.55}},
    {'name': 'ensemble3', 'weights': {'XGB_NN': 0.64, 'Best_Model': 0.36}},
    {'name': 'ensemble4', 'weights': {'XGB_NN': 0.3, 'Best_Model': 0.7}},
    {'name': 'ensemble5', 'weights': {'LightGBM': 0.4, 'XGB_NN': 0.4, 'Best_Model': 0.2}}
]


for config in ensemble_configs:
    ensemble_pred = create_ensemble(predictions_df.iloc[:, 1:], config['weights'])
    submission = pd.DataFrame({"id": test_df["id"], "y": ensemble_pred})
    submission.to_csv(WORKING_PATH + f"{config['name']}.csv", index=False)

for config in ensemble_configs:
    ensemble_pred = create_ensemble(predictions_df.iloc[:, 1:], config['weights'])
    predictions_df[config['name']] = ensemble_pred


corr_matrix = predictions_df.iloc[:, 1:].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f')
plt.title('Predictions Correlation')
plt.tight_layout()
plt.savefig(WORKING_PATH + 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 8))
for col in predictions_df.columns[4:]:
    sns.kdeplot(predictions_df[col], label=col, linewidth=2)
plt.title('Ensemble Comparison')
plt.xlabel('Predicted Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(WORKING_PATH + 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


























































