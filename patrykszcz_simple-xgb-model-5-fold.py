import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv",index_col = 'id')


train.head()


train.info()


cat_columns = [i for i in train.columns if train[i].dtype == np.object_]
num_columns = [i for i in train.columns if i not in cat_columns]


train.isnull().sum()


train.duplicated().sum()


for i in cat_columns:
    print(f'Unique values in cat column: {i} - {train[i].nunique()}')
for i in cat_columns[:-1]:
    print(f'Unique values in cat column: {i} - {test[i].nunique()}')


train.describe()


test.describe()


label_enc = LabelEncoder()
for i in cat_columns[:-1]:
    train[i] = label_enc.fit_transform(train[i])
    test[i] = label_enc.transform(test[i])



n_features = len(train.columns)-1
cols = train.drop('Fertilizer Name',axis=1).columns

fig, axs = plt.subplots(nrows=n_features,ncols=2, figsize = (15,6*n_features), dpi = 100)

for i, col in enumerate(cols):
    ## HISTOGRAMS
    sns.histplot(data=train, x = col,hue = 'Fertilizer Name', kde = True,multiple="stack",ax=axs[i,0],palette='seismic')
    axs[i,0].set_title(f'HISTOGRAM_{col}')
    axs[i,0].grid(True, linestyle='--', alpha = 0.5)

    ## VIOLINPLOT
    sns.violinplot(data=train , x = 'Fertilizer Name', y= col,ax =axs[i,1],palette='seismic')
    axs[i,1].set_title(f'Personality_vs_{col}')
    axs[i,1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])


plt.figure(figsize = (8,7))
sns.heatmap(train.corr(numeric_only=True),annot=True)
plt.tight_layout()
plt.title('Coreletion numric features')
plt.show()


X_pca = train.drop('Fertilizer Name', axis =1 )
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_pca)
## EXPLAINED VARIANCE AFTER PCA
pca.explained_variance_ratio_.sum()


df_pca = pd.DataFrame(X_pca,columns= ['PCA1','PCA2','PCA3'],index=train.index) 
df_pca = pd.concat([df_pca,train['Fertilizer Name']],axis=1)
df_pca.head()


fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111,projection='3d')

scatter = ax.scatter(
    df_pca['PCA1'],
    df_pca['PCA2'],
    df_pca['PCA3'],
    c = df_pca['Fertilizer Name'],
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


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


X = train.drop('Fertilizer Name',axis = 1)
y = train["Fertilizer Name"]


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob = np.zeros(shape = (len(test),y.nunique()))


xgb_model = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha= 2.7,
    reg_lambda= 1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state = 13,
    enable_categorical=True,
    device = 'cuda')

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    xgb_model.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 0)
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob +=xgb_model.predict_proba(test)

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")

    



top_3_preds = np.argsort(oof, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y]
map3_score = mapk(actual, top_3_preds)
print(f'âœ… Final MAP@3 Score: {map3_score:.5f} ')


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")

