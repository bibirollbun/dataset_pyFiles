import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,auc,roc_curve
from sklearn.metrics import classification_report
import warnings
import xgboost as xgb
from sklearn.impute import KNNImputer
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv',index_col = 'id')



train['Stage_fear'] = train['Stage_fear'].map({'No':1,'Yes':2})
test['Stage_fear'] = test['Stage_fear'].map({'No':1,'Yes':2})

train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'No':1, "Yes":2})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'No':1, "Yes":2})



cat_columns = ['Stage_fear','Drained_after_socializing']
num_columns = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']


X_all = train.drop(columns=['Personality'])  
y_all = train['Personality']
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns)

train = pd.concat([X_imputed, y_all.reset_index(drop=True)], axis=1)


X_all = test
test = pd.DataFrame(imputer.transform(X_all), columns=X_all.columns)


train['Personality'] = train['Personality'].map({'Extrovert':0,'Introvert':1})


n_features = len(train.columns)-1
cols = train.drop('Personality',axis=1).columns
fig, axs = plt.subplots(nrows = n_features,ncols=2,figsize = (12, 4 * n_features),dpi = 100)

for i, col in enumerate(cols):
    ## HISTOGRAMS
    sns.histplot(data=train, x=col, hue='Personality', kde=True, ax=axs[i,0], multiple='dodge',palette='seismic')
    axs[i,0].set_title(f'HISTOGRAM_{col}')
    axs[i,0].grid(True, linestyle='--', alpha = 0.5)
    ## VIOLINPLOT
    sns.violinplot(data = train,x='Personality', y = col, ax=axs[i,1],palette='seismic')
    axs[i,1].set_title(f'Personality_vs_{col}')
    axs[i,1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


g = sns.pairplot(data=train, hue='Personality')
g._legend.remove()
plt.tight_layout()
plt.show()


fig = plt.figure(figsize = (8,8),dpi=100)
sns.heatmap(train.corr(),annot=True)
plt.title(f'Correletions')
plt.tight_layout()
plt.show()


X_pca = train.drop('Personality', axis =1 ) 
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_pca)
## EXPLAINED VARIANCE AFTER PCA
pca.explained_variance_ratio_.sum()


df_pca = pd.DataFrame(X_pca,columns= ['PCA1','PCA2','PCA3'],index=train.index) ## Remember to define the index after dropping duplicates
df_pca = pd.concat([df_pca,train['Personality']],axis=1)
df_pca.head()


fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111,projection='3d')

scatter = ax.scatter(
    df_pca['PCA1'],
    df_pca['PCA2'],
    df_pca['PCA3'],
    c = df_pca['Personality'],
    cmap='rainbow',
    marker='o',
    alpha=0.6,
)
ax.set_title(f'3D PLOT')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.colorbar(scatter)
plt.show()


X = train.drop('Personality', axis=1)
y = train['Personality']


def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = 0

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score



oof = np.zeros(len(X))
test_preds = np.zeros(len(test))  

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
thresholds = []


params = {
    'max_depth': 12,
    'colsample_bytree': 0.5590480329278282, 
    'subsample': 0.4572883708952284,
    'learning_rate': 0.009106526547027212,
    'gamma': 0.7861025486849484, 
    'max_delta_step': 3,
    'reg_alpha': 0.36003155961495115,
    'reg_lambda': 0.8184527950570306,
    'random_state': 42,
    'tree_method': 'hist',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'device': 'cuda',
    'enable_categorical':True
}

for i,(train_idx, valid_idx) in enumerate(skf.split(X, y)):
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(**params, n_estimators=6_000,early_stopping_rounds=50)

    model.fit(x_train, y_train,
        eval_set=[(x_valid, y_valid)],verbose = False)

    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(x_valid, y_valid)

    y_valid_proba = calibrated_model.predict_proba(x_valid)[:, 1]
    best_thresh, _ = find_best_threshold(y_valid, y_valid_proba)
    thresholds.append(best_thresh)

  
    oof[valid_idx] = (y_valid_proba >= best_thresh).astype(int)

    acc_score = accuracy_score(oof[valid_idx],y_valid) 

    test_proba = calibrated_model.predict_proba(test)[:, 1]
    test_preds += (test_proba >= best_thresh).astype(int) / skf.n_splits
    print(f"âœ… FOLD {i+1}: ACC Score: {acc_score:.5f}")


final_acc = accuracy_score(oof,y)
print(f"âœ… Final ACC Score: {final_acc:.5f}") 


np.save('oof_1',oof)
np.save('preds_1',test_preds)


def plot_cm(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])
    disp.plot(ax=ax,colorbar=False)

def auc_roc_plot(y_true,y_pred, ax = None):
    fpr, tpr ,_ = roc_curve(y_true,y_pred)
    roc_auc = auc(fpr,tpr)
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')  
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid()


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=100)
plot_cm(oof, y, ax=axs[0])
axs[0].set_title('Confusion matrix')
auc_roc_plot(oof, y, ax=axs[1])
axs[1].set_title('ROC AUC')

plt.tight_layout()
plt.show()


print(classification_report(oof,y))



test_preds  = pd.Series(test_preds).map({0: 'Extrovert', 1: 'Introvert'})
test_preds


submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = test_preds
submission.to_csv('submission.csv',index = False)


submission.head()

