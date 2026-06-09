import matplotlib.pyplot as plt, missingno as msno, numpy as np, pandas as pd, seaborn as sns
import gc, itertools, json, math, os, pprint, warnings

from pathlib import Path
from phik import phik_matrix
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


DATA_DIR = Path('/kaggle/input/playground-series-s5e8/')
train  = pd.read_csv(DATA_DIR / 'train.csv')
test   = pd.read_csv(DATA_DIR / 'test.csv')
sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
display(train.head())


train.info(memory_usage='deep')


display(train.describe(include=[np.number, 'bool']).T)


cat_cols = train.select_dtypes(include=['object']).columns
for c in cat_cols:
    print(f"\n{c} – unique={train[c].nunique()}")
    print(train[c].value_counts(dropna=False)
            .to_frame('freq')
            .assign(perc=lambda d: d.freq.mul(100).div(len(train)).round(2))
            .head(10))


print('Duplicate rows   :', train.duplicated().sum())
print('Duplicate id(s)  :', train['id'].duplicated().sum())


qc = []
for c in train.columns:
    top_frac = train[c].value_counts(normalize=True, dropna=False).iloc[0]
    if top_frac > 0.999:
        qc.append((c, top_frac))
pd.DataFrame(qc, columns=['column','top_frac']).style.background_gradient(cmap='Reds')

na = train.isna().mean().mul(100).sort_values(ascending=False)
display(na.to_frame('%missing').style.background_gradient(cmap='Blues'))


plt.figure(figsize=(12,4))
msno.matrix(train)
plt.title('Missingness Matrix'); plt.show()


ax = train['y'].value_counts(normalize=True).mul(100).plot.bar(
        rot=0, figsize=(4,3), title='Target Distribution (%)')
for p in ax.patches:
    ax.text(p.get_x()+0.05, p.get_height()+1,
            f"{p.get_height():.2f}%", fontsize=9)
plt.show()


num_cols = train.select_dtypes(include=['int64','float64']).columns.drop(['id','y'])
print(f'Numerical cols ({len(num_cols)}):', list(num_cols))


n_cols = 3
n_rows = math.ceil(len(num_cols)/n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*3.5))
for i, col in enumerate(num_cols):
    r, c = divmod(i, n_cols)
    sns.histplot(train[col], kde=True, bins=50, ax=axs[r][c], label='train')
    if col in test.columns:
        sns.histplot(test[col], kde=False, bins=50, ax=axs[r][c],
                     color='orange', alpha=.3, label='test')
    axs[r][c].set_title(col); axs[r][c].legend()
plt.tight_layout(); plt.show()


plt.figure(figsize=(12,0.4*len(num_cols)))
sns.boxplot(data=train[num_cols], orient='h', showfliers=True)
plt.title('Boxplots – all numeric'); plt.tight_layout(); plt.show()


outlier_report = {}
for col in num_cols:
    q1,q3 = np.percentile(train[col],[25,75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    outlier_report[col] = ((train[col] < lo) | (train[col] > hi)).mean()*100
pd.Series(outlier_report, name='%outliers').sort_values(ascending=False)\
    .to_frame().style.background_gradient(cmap='Reds')


corr = train[num_cols].corr(method='pearson')
plt.figure(figsize=(10,8))
sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Pearson Correlation Matrix'); plt.show()


target_corr = train[num_cols].corrwith(train['y']).sort_values(key=abs, ascending=False)
display(target_corr.to_frame('r(y, x)').style.background_gradient(cmap='BrBG'))


cat_cols = train.select_dtypes(include=['object','category','bool']).columns.tolist()
cat_cols.remove('y') if 'y' in cat_cols else None
print(f'Categorical cols ({len(cat_cols)}):', cat_cols)


def bar_plot(col, top_n=20):
    vc = train[col].value_counts(dropna=False)
    top = vc.head(top_n)
    plt.figure(figsize=(6,3+0.15*len(top)))
    sns.barplot(y=top.index, x=top.values, orient='h')
    plt.title(f'{col} – top {top_n} counts'); plt.show()
    
for col in cat_cols:
    bar_plot(col)


for col in cat_cols:
    rate = train.groupby(col)['y'].mean().sort_values(ascending=False)
    display(rate.to_frame(f'P(y=1 | {col})').head(10))


phik_mat = train[cat_cols].phik_matrix(interval_cols=[]).loc[cat_cols, cat_cols]
plt.figure(figsize=(len(cat_cols), len(cat_cols)))
sns.heatmap(phik_mat, cmap='Purples', square=True, annot=True)
plt.title('Phik association between categoricals'); plt.show()


pd.set_option('display.float_format', '{:.2e}'.format)  # scientific notation

chi_res = {}
for col in cat_cols:
    tbl = pd.crosstab(train[col], train['y'])
    chi2, p, dof, _ = chi2_contingency(tbl)
    chi_res[col] = p
pd.Series(chi_res, name='p_value').sort_values()\
    .to_frame().head(10).style.background_gradient(cmap='Greens_r')


for col in num_cols:
    sns.boxplot(x='y', y=col, data=train.sample(50000, random_state=0))
    plt.title(f'{col} by y'); plt.show()


tt_p = {c : ttest_ind(train.loc[train['y']==0, c],
                      train.loc[train['y']==1, c],
                      equal_var=False).pvalue for c in num_cols}
pd.Series(tt_p, name='t_test_p').sort_values()\
    .to_frame().style.background_gradient(cmap='Greens_r')


X_cat = train[cat_cols].astype(str)
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_enc = enc.fit_transform(X_cat)
X_num = train[num_cols].values
X = np.hstack([X_num, X_cat_enc])
y = train['y'].values

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
gb = GradientBoostingClassifier(random_state=1)
gb.fit(X_tr, y_tr)
print('Sanity ROC-AUC on hold-out:', roc_auc_score(y_val, gb.predict_proba(X_val)[:,1]).round(3))


sample.to_csv('submission.csv', index=False)




