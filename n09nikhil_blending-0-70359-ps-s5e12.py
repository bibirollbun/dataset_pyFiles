import os,re
import numpy as np,pandas as pd
from collections import Counter,defaultdict
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


ROOT_DIR='/kaggle/input/blending-ps-s5e12'
OUT_DIR='/kaggle/working/'
PLOT_DIR='/mnt/data'
NUM_CLUSTERS=5
LAMBDA_GRID=[0.0,0.05,0.1,0.25,0.5]
TARGET_MEAN=0.623
TARGET_STD=0.193

os.makedirs(OUT_DIR,exist_ok=True)
os.makedirs(PLOT_DIR,exist_ok=True)



def detect_pred_col(df):
    for c in ['diagnosed_diabetes','target','prediction','pred','probability']:
        if c in df.columns: return c
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    nonid=[c for c in numeric if c.lower() not in ('id','ids','patient_id')]
    if nonid: return nonid[0]
    if numeric: return numeric[0]
    return None

def extract_score_from_name(name):
    m=re.search(r"([0-9]+\.[0-9]{4,6})",os.path.basename(name))
    if m:
        try: return float(m.group(1))
        except: return None
    return None

def normalize_w(w):
    w=np.array(w,dtype=float)
    s=w.sum()
    if s==0 or np.isnan(s): return np.ones_like(w)/len(w)
    return w/s



files=sorted([f for f in os.listdir(ROOT_DIR) if f.lower().endswith('.csv')])
subs=[]
for fn in files:
    fp=os.path.join(ROOT_DIR,fn)
    try:
        df=pd.read_csv(fp)
    except:
        continue
    pred_col=detect_pred_col(df)
    if pred_col is None:
        continue
    score=extract_score_from_name(fn)
    subs.append({'filename':fn,'score':score,'preds':df[pred_col].astype(float).values,'id':df['id'].values if 'id' in df.columns else None})

lengths=[len(s['preds']) for s in subs]
most_common_len=Counter(lengths).most_common(1)[0][0]
subs=[s for s in subs if len(s['preds'])==most_common_len]

ids=None
for s in subs:
    if s.get('id') is not None:
        ids=s['id']; break
if ids is None:
    ids=np.arange(most_common_len)

filenames=[s['filename'] for s in subs]
scores_all=np.array([0.0 if s['score'] is None else float(s['score']) for s in subs])
preds_mat=np.column_stack([s['preds'] for s in subs])
n_models=preds_mat.shape[1]



pearson=np.corrcoef(preds_mat.T)
avg_pairwise_corr=(pearson.sum()-n_models)/(n_models*(n_models-1))

dist=1-pearson
np.fill_diagonal(dist,0.0)
condensed=squareform(dist,checks=False)
Z=linkage(condensed,method='average')
clusters=fcluster(Z,NUM_CLUSTERS,criterion='maxclust')

cluster_members=defaultdict(list)
for i,c in enumerate(clusters):
    cluster_members[c].append(i)

cluster_info={}
for c,idxs in sorted(cluster_members.items()):
    if len(idxs)==1:
        medoid_idx=idxs[0]; avg_within=1.0
    else:
        submat=pearson[np.ix_(idxs,idxs)]
        avg_corrs=submat.mean(axis=1)
        medoid_local=int(np.argmax(avg_corrs))
        medoid_idx=idxs[medoid_local]
        tri=submat[np.triu_indices_from(submat,k=1)]
        avg_within=float(np.mean(tri)) if len(tri)>0 else 1.0
    cluster_info[c]={'indices':idxs,'size':len(idxs),'medoid':medoid_idx,'medoid_name':filenames[medoid_idx],'avg_within_corr':avg_within}

clusters_sorted=sorted(cluster_info.items(),key=lambda x:x[0])
medoid_indices=[info['medoid'] for _,info in clusters_sorted]
medoid_names=[filenames[i] for i in medoid_indices]
medoid_scores=np.array([scores_all[i] for i in medoid_indices],dtype=float)
medoid_preds=np.column_stack([preds_mat[:,i] for i in medoid_indices])



medoid_corr=np.corrcoef(medoid_preds.T)
corr_off=medoid_corr.copy()
np.fill_diagonal(corr_off,0.0)

def objective_neg(w,lambda_div):
    w=np.array(w)
    s_term=np.dot(w,medoid_scores)
    div_term=w@corr_off@w
    return -(s_term - lambda_div*div_term)

cons=({'type':'eq','fun':lambda w: w.sum()-1.0})
bnds=[(0.0,1.0)]*len(medoid_indices)
x0=np.ones(len(medoid_indices))/float(len(medoid_indices))

best_result=None
best_lambda=None
best_obj=1e9

for lam in LAMBDA_GRID:
    res=minimize(objective_neg,x0,args=(lam,),method='SLSQP',bounds=bnds,constraints=cons,options={'ftol':1e-9,'maxiter':1000})
    if res.fun<best_obj:
        best_obj=res.fun; best_result=res; best_lambda=lam

w_opt=np.maximum(best_result.x,0.0)
w_opt=w_opt/w_opt.sum()

submission_medoid_weighted=medoid_preds.dot(w_opt)



sorted_idx=np.argsort(-scores_all)
K=min(12,n_models)
top12_idx=list(sorted_idx[:K])
pw=np.power(np.maximum([scores_all[i] for i in top12_idx],1e-12),3)
pw=pw/pw.sum() if pw.sum()>0 else np.ones_like(pw)/len(pw)
top12_power=preds_mat[:,top12_idx].dot(pw)

submission_meta_candidate=submission_medoid_weighted*0.6 + top12_power*0.4

best_single_idx=sorted_idx[0]
K10=min(10,n_models)
top10_idx=list(sorted_idx[:K10])
w10=np.array([scores_all[i] for i in top10_idx],dtype=float)
w10=w10/w10.sum() if w10.sum()>0 else np.ones_like(w10)/len(w10)
submission_safe=preds_mat[:,best_single_idx]*0.7 + preds_mat[:,top10_idx].dot(w10)*0.3

fn1=os.path.join(OUT_DIR,'submission_cluster_medoid_weighted.csv')
fn2=os.path.join(OUT_DIR,'submission_cluster_medoid_meta_combo.csv')
fn3=os.path.join(OUT_DIR,'submission_safe_blend_from_opt.csv')

pd.DataFrame({'id':ids,'diagnosed_diabetes':np.clip(submission_medoid_weighted,0,1)}).to_csv(fn1,index=False)
pd.DataFrame({'id':ids,'diagnosed_diabetes':np.clip(submission_meta_candidate,0,1)}).to_csv(fn2,index=False)
pd.DataFrame({'id':ids,'diagnosed_diabetes':np.clip(submission_safe,0,1)}).to_csv(fn3,index=False)

diag=pd.DataFrame({'medoid_index':medoid_indices,'medoid_name':medoid_names,'medoid_score':medoid_scores,'weight':w_opt})
diag_fp=os.path.join(PLOT_DIR,'medoid_weights_diag.csv')
diag.to_csv(diag_fp,index=False)

print("Submission files ready:")
print("  1)", fn1)
print("  2)", fn2)
print("  3)", fn3)
print("Diagnostics saved to", diag_fp)



from IPython.display import Image, display

diag_fp='/mnt/data/medoid_weights_diag.csv'
diag=pd.read_csv(diag_fp)
w=diag['weight'].values
names=list(diag['medoid_name'].values)

plt.figure(figsize=(8,3))
bars=plt.bar(range(len(w)),w,color='skyblue')
plt.xticks(range(len(w)),[os.path.basename(n) for n in names],rotation=45,ha='right',fontsize=7)
plt.title('Medoid Weights',fontsize=12)
plt.ylabel('Weight')
plt.tight_layout()
plt.show()
plt.savefig('/mnt/data/vis_weights.png',dpi=120)
plt.close()

plt.figure(figsize=(5,4))
sns.heatmap(np.corrcoef(medoid_preds.T),annot=False,cmap='coolwarm',cbar=True)
plt.title('Medoid Correlation Heatmap',fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('/mnt/data/vis_medoid_corr.png',dpi=120)
plt.close()

print("Saved submission files:", [f for f in os.listdir(OUT_DIR) if f.startswith('submission_')])

for p in ['/mnt/data/vis_weights.png','/mnt/data/vis_medoid_corr.png']:
    if os.path.exists(p):
        display(Image(filename=p))





