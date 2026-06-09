import numpy as np
import pandas as pd
import optuna
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc


oof_list = pd.DataFrame()
pred_list = pd.DataFrame()


v9_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-9/oof_loss_cv_0.07201366708085293_pr_auc_cv_0.37595195310571133.csv')
v15_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-15/oof_loss_cv_0.23992747661544025_pr_auc_cv_0.4518239684699883.csv')
v16_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-16/oof_loss_cv_0.30748499352813885_pr_auc_cv_0.4226937445583319.csv')
v18_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-18/kaggle/working/oof_loss_cv_0.450903825902202_pr_auc_cv_0.4312267124674702.csv')
v19_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-v19/kaggle/working/oof_loss_cv_0.7479273149886787_pr_auc_cv_0.4353148638093963.csv')
v20_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-20/kaggle/working/oof_loss_cv_0.7522058008967644_pr_auc_cv_0.45172707417401614.csv')
v24_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-24/oof_loss_cv_0.2694772363080218_pr_auc_cv_0.41405244064539537.csv')
v25_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-25/oof_loss_cv_0.7779362849345652_pr_auc_cv_0.45898639958198073.csv')
v26_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-26/oof_loss_cv_0.7455776842330402_pr_auc_cv_0.4424245613177511.csv')
v27_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-27/oof_loss_cv_0.5734032209530995_pr_auc_cv_0.43732585129205626.csv')
v28_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-28/oof_loss_cv_0.7000405918131809_pr_auc_cv_0.39214443998030135.csv')
v29_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-29/oof_loss_cv_0.7018927015519529_pr_auc_cv_0.3941094168379383.csv')
v30_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-30/oof_loss_cv_0.6696156461148548_pr_auc_cv_0.3575307273828824.csv')
v31_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-31/oof_loss_cv_0.917163171475651_pr_auc_cv_0.459696207191758.csv')
v32_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-32/oof_loss_cv_0.8584093696578862_pr_auc_cv_0.4851654588148684.csv')
v33_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-33/oof_loss_cv_0.42738356876964323_pr_auc_cv_0.2204202183640966.csv')

v34_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-34/oof_loss_cv_0.4146278450453783_pr_auc_cv_0.22019817855451138.csv')
v35_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-35/oof_loss_cv_0.8522108434176018_pr_auc_cv_0.48651487972389473.csv')
v36_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-36/oof_loss_cv_0.9052584779083098_pr_auc_cv_0.44989943631473595.csv')
v37_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-37/oof_loss_cv_0.7052658111929482_pr_auc_cv_0.35362688951751053.csv')
v38_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-38/oof_loss_cv_0.7125911049333591_pr_auc_cv_0.39063594356670084.csv')
v39_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-39/oof_loss_cv_0.7012495385202865_pr_auc_cv_0.39756669580548487.csv')
v40_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-40/oof_loss_cv_0.5626315739359252_pr_auc_cv_0.44643279195427243.csv')
v41_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version41/oof_loss_cv_0.7383007896179021_pr_auc_cv_0.4498902629540524 (1).csv')
v43_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-43/oof_loss_cv_0.2642719408370356_pr_auc_cv_0.4100284233166272 (1).csv')
v45_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-45/oof_loss_cv_0.761118515745796_pr_auc_cv_0.43177125199871286.csv')
v53_cv_df = pd.read_csv('/kaggle/input/phems-multi-model-version-53/oof_loss_cv_0.23322887157746774_pr_auc_cv_0.4589158785581527.csv')


v9_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-9/submission.csv')
v15_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-15/submission.csv')
v16_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-16/submission.csv')
v18_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-18/kaggle/working/submission.csv')
v19_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-v19/kaggle/working/submission.csv')
v20_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-20/kaggle/working/submission.csv')
v24_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-24/submission - 2025-01-27T085240.951.csv')
v25_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-25/submission - 2025-01-27T085503.793.csv')
v26_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-26/submission - 2025-01-27T085555.538.csv')
v27_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-27/submission - 2025-01-27T085651.552.csv')
v28_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-28/submission - 2025-01-27T085803.742.csv')
v29_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-29/submission - 2025-01-27T085853.072.csv')
v30_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-30/submission - 2025-01-27T085950.646.csv')
v31_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-31/submission - 2025-01-27T090039.954.csv')
v32_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-32/submission - 2025-01-27T090227.624.csv')
v33_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-33/submission - 2025-01-27T090816.900.csv')

v34_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-34/submission - 2025-01-27T140323.202.csv')
v35_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-35/submission - 2025-01-27T140446.051.csv')
v36_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-36/submission - 2025-01-27T140551.643.csv')
v37_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-37/submission - 2025-01-27T140715.770.csv')
v38_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-38/submission - 2025-01-27T140809.874.csv')
v39_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-39/submission - 2025-01-27T140901.792.csv')
v40_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-40/submission - 2025-01-27T141019.927.csv')

v41_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version41/submission - 2025-01-27T193022.329.csv')
v43_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-43/submission - 2025-01-27T192922.114.csv')
v45_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-45/submission - 2025-01-27T192034.416.csv')
v53_sub_df = pd.read_csv('/kaggle/input/phems-multi-model-version-53/submission.csv')


oof_list['v9'] = (v9_cv_df["0"] + v9_cv_df["1"]) / 2
oof_list['v15'] = (v15_cv_df["0"] + v15_cv_df["1"]) / 2
oof_list['v16'] = (v16_cv_df["0"] + v16_cv_df["1"]) / 2
oof_list['v18'] = (v18_cv_df["0"] + v18_cv_df["1"]) / 2
oof_list['v19'] = (v19_cv_df["0"] + v19_cv_df["1"]) / 2
oof_list['v20'] = (v20_cv_df["0"] + v20_cv_df["1"]) / 2
oof_list['v24'] = (v24_cv_df["0"] + v24_cv_df["1"]) / 2
oof_list['v25'] = (v25_cv_df["0"] + v25_cv_df["1"]) / 2
oof_list['v26'] = (v26_cv_df["0"] + v26_cv_df["1"]) / 2
oof_list['v27'] = (v27_cv_df["0"] + v27_cv_df["1"]) / 2
oof_list['v28'] = (v28_cv_df["0"] + v28_cv_df["1"]) / 2
oof_list['v29'] = (v29_cv_df["0"] + v29_cv_df["1"]) / 2
oof_list['v30'] = (v30_cv_df["0"] + v30_cv_df["1"]) / 2
oof_list['v31'] = (v31_cv_df["0"] + v31_cv_df["1"]) / 2
oof_list['v32'] = (v32_cv_df["0"] + v32_cv_df["1"]) / 2
oof_list['v33'] = (v33_cv_df["0"] + v33_cv_df["1"]) / 2
oof_list['v34'] = (v34_cv_df["0"] + v34_cv_df["1"]) / 2
oof_list['v35'] = (v35_cv_df["0"] + v35_cv_df["1"]) / 2
oof_list['v36'] = (v36_cv_df["0"] + v36_cv_df["1"]) / 2
oof_list['v37'] = (v37_cv_df["0"] + v37_cv_df["1"]) / 2
oof_list['v38'] = (v38_cv_df["0"] + v38_cv_df["1"]) / 2
oof_list['v39'] = (v39_cv_df["0"] + v39_cv_df["1"]) / 2
oof_list['v40'] = (v40_cv_df["0"] + v40_cv_df["1"]) / 2
# oof_list['v41'] = (v41_cv_df["0"] + v41_cv_df["1"]) / 2
oof_list['v43'] = (v43_cv_df["0"] + v43_cv_df["1"]) / 2
oof_list['v45'] = (v45_cv_df["0"] + v45_cv_df["1"]) / 2
oof_list['v53'] = (v53_cv_df["0"] + v53_cv_df["1"]) / 2

pred_list['v9'] = v9_sub_df['SepsisLabel']
pred_list['v15'] = v15_sub_df['SepsisLabel']
pred_list['v16'] = v16_sub_df['SepsisLabel']
pred_list['v18'] = v18_sub_df['SepsisLabel']
pred_list['v19'] = v19_sub_df['SepsisLabel']
pred_list['v20'] = v20_sub_df['SepsisLabel']
pred_list['v24'] = v24_sub_df['SepsisLabel']
pred_list['v25'] = v25_sub_df['SepsisLabel']
pred_list['v26'] = v26_sub_df['SepsisLabel']
pred_list['v27'] = v27_sub_df['SepsisLabel']
pred_list['v28'] = v28_sub_df['SepsisLabel']
pred_list['v29'] = v29_sub_df['SepsisLabel']
pred_list['v30'] = v30_sub_df['SepsisLabel']
pred_list['v31'] = v31_sub_df['SepsisLabel']
pred_list['v32'] = v32_sub_df['SepsisLabel']
pred_list['v33'] = v33_sub_df['SepsisLabel']
pred_list['v34'] = v34_sub_df['SepsisLabel']
pred_list['v35'] = v35_sub_df['SepsisLabel']
pred_list['v36'] = v36_sub_df['SepsisLabel']
pred_list['v37'] = v37_sub_df['SepsisLabel']
pred_list['v38'] = v38_sub_df['SepsisLabel']
pred_list['v39'] = v39_sub_df['SepsisLabel']
pred_list['v40'] = v40_sub_df['SepsisLabel']
# pred_list['v41'] = v41_sub_df['SepsisLabel']
pred_list['v43'] = v43_sub_df['SepsisLabel']
pred_list['v45'] = v45_sub_df['SepsisLabel']
pred_list['v53'] = v53_sub_df['SepsisLabel']

oofs = oof_list.values
preds = pred_list.values


lb_dict = {
    "v19": 0.608,
    "v18": 0.501,
    "v16": 0.604,
    "v15": 0.541,
    "v9": 0.578,
    "v20": 0.578,
    "v32": 0.644,
    "v24": 0.610,
    "v28": 0.634,
    "v29": 0.626,
    "v33": 0.358,
    "v31": 0.629,
    "v30": 0.570,
    "v27": 0.573,
    "v26": 0.603,
    "v25": 0.628,
}


from tqdm import tqdm
def create_mean_median_feature(df):
    preds = []
    for _, row in tqdm(df.iterrows()):
        row_median = row.median()
        row_mean = row[row != row_median].mean()
        row_mean_median = (row_median + row_mean) / 2
        preds.append(row_mean_median)
    return preds
    

pred_list['mean_median'] = create_mean_median_feature(pred_list)
oof_list['mean_median'] = create_mean_median_feature(oof_list)


v20_cv_df


pred_list['max'] = np.max(preds, axis = 1)
oof_list['max'] = np.max(oofs, axis = 1)

pred_list['min'] = np.min(preds, axis = 1)
oof_list['min'] = np.min(oofs, axis = 1)

pred_list['mean'] = np.mean(preds, axis = 1)
oof_list['mean'] = np.mean(oofs, axis = 1)

pred_list['std'] = np.std(preds, axis = 1)
oof_list['std'] = np.std(oofs, axis = 1)


print("Correlation between multi models:")
correlations = pd.DataFrame(np.corrcoef(oof_list.T), columns = oof_list.columns)

display(correlations)


oof_list.columns == pred_list.columns


def accuracy_cv_score(pred, real, w):
    
    y_pred =(pred > 0.5).astype(np.int32)
    conf_matrix = confusion_matrix(real, y_pred)

    neg = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1])
    pos = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    return neg * w + pos * (1 - w)
def cal_score(y_true:np.array,y_pro:np.array):
    # pos_idx=np.where(y_true==1)[0]
    # neg_idx=np.where(y_true==0)[0]
    # pos_pro=sorted(y_pro[pos_idx])
    # neg_pro=sorted(y_pro[neg_idx])
    # total_sample_cnt,greater_sample_cnt=len(pos_pro)*len(neg_pro),0
    # left,right=0,0
    # while left<len(pos_pro):
    #     while right<len(neg_pro) and (pos_pro[left]>neg_pro[right]):
    #         right+=1
    #     if right<len(neg_pro):
    #         greater_sample_cnt+=right
    #         left+=1
    #     else:
    #         greater_sample_cnt+=len(neg_pro)*(len(pos_pro)-left)
    #         left=len(pos_pro)
    # auc_score=greater_sample_cnt/total_sample_cnt
    # return auc_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_pro)
    pr_auc = auc(recall, precision)
    return pr_auc
    
def get_optimize_metric_value(y_true, y_pred, w = 0.5):
    pr_auc_cv = cal_score(y_true, y_pred)
    accuracy_cv = accuracy_cv_score(y_pred, y_true, w)

    
    return -pr_auc_cv
    





y_target = v9_cv_df['target'].values
def multiplier_objective(trial):
    
    multipliers = np.array([trial.suggest_float(f'{label}', -1, 1) for label in list(oof_list)])
    
    oof_ensemble = oof_list.to_numpy() @ (multipliers / multipliers.sum())
    oof_ensemble = np.clip(oof_ensemble, 0, 1)
    # oof_ensemble = (oof_ensemble > 0.5).astype(np.int32)

    val_score = get_optimize_metric_value(y_target, oof_ensemble)
    return val_score


best_weights = pd.read_csv('/kaggle/input/phems-weights/weights.csv')
best_weights


best_weights = best_weights.values



plt.figure(figsize=(16, 12))
a = []
b = 0
for i in tqdm(range(30)):
    
    weights = best_weights[i]
    best_y_oof_ensemble = oof_list.to_numpy() @ weights
    best_y_oof_ensemble = np.clip(best_y_oof_ensemble, 0, 1)
    a.append(best_y_oof_ensemble)
    b += best_y_oof_ensemble
    
    precision, recall, _ = precision_recall_curve(y_target, best_y_oof_ensemble)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', linewidth=2)

a = np.array(a)
a = np.max(a, axis = 0)
b = b / 30

precision, recall, _ = precision_recall_curve(y_target, a)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'MAX 30 Tries (AUC = {pr_auc:.2f})', linewidth=2)

precision, recall, _ = precision_recall_curve(y_target, b)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'MEAN 30 Tries (AUC = {pr_auc:.2f})', linewidth=2)

plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve')
plt.legend(fontsize=6)
plt.grid(alpha=0.4)
plt.show()
    


plt.figure(figsize = (16, 14), dpi = 300)
a = []
b = 0

for i in tqdm(range(30)):
    
    weights = best_weights[i]
    best_y_oof_ensemble = oof_list.to_numpy() @ weights
    best_y_oof_ensemble = np.clip(best_y_oof_ensemble, 0, 1)
    a.append(best_y_oof_ensemble)
    b += best_y_oof_ensemble
    
    sns.kdeplot(best_y_oof_ensemble, fill = True, label = f'{i}')
a = np.array(a)
a = np.max(a, axis = 0)
b = b / 30

sns.kdeplot(a, fill = True, label = 'MAX 30 Tries')
sns.kdeplot(b, fill = True, label = 'MEAN 30 Tries')

plt.title(f"Distribution of SepsisLabel")
plt.legend()
plt.show()

    


df = pd.read_csv('/kaggle/input/phems-try-v3/submission_multi_try.csv')
del df['SepsisLabel']
df


cols = [c for c in df.columns if c not in ['person_id_datetime', 'SepsisLabel']]
df['mean'] = np.mean(df[cols].values, axis = 1)
df['min'] = np.min(df[cols].values, axis = 1)
df['max'] = np.max(df[cols].values, axis = 1)


plt.figure(figsize = (15, 10), dpi = 300)
for c in ['mean', 'max', 'min']:
    sns.kdeplot(df[c].values, fill = True, label = c)
plt.title(f"Distribution of SepsisLabel", weight = 'bold', size = 25)
plt.legend()
plt.show()


for c in ['mean']:
    sub_df = df.copy()
    sub_df = sub_df[['person_id_datetime', c]]
    sub_df['SepsisLabel'] = sub_df[c]
    del sub_df[c]
    sub_df.to_csv(f'submission.csv', index = False)

