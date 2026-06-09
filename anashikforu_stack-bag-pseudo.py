# -*- coding: utf-8 -*-
"""
Ultraâ€‘Advanced Introvert vs Extrovert Prediction
â€” Preâ€‘tune XGB & LGB rounds â†’ Full 15â€‘model Stacking
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

import xgboost as xgb
# Optional LightGBM
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
# Optional CatBoost
try:
    from catboost import CatBoostClassifier
    has_cat = True
except ImportError:
    has_cat = False

RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# 1) FUNCTIONS TO TUNE BOOSTING ROUNDS VIA CV
# -----------------------------------------------------------------------------
def tune_xgb_rounds(X, y, folds=5, seed=RANDOM_STATE):
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'auc',
        'seed': seed
    }
    cvdf = xgb.cv(
        params, dtrain,
        num_boost_round=1000,
        nfold=folds,
        early_stopping_rounds=50,
        seed=seed,
        metrics='auc',
        as_pandas=True,
        stratified=True
    )
    return len(cvdf)

def tune_lgb_rounds(X, y, folds=5, seed=RANDOM_STATE):
    import lightgbm as _lgb
    dtrain = _lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'metric': 'auc',
        'verbosity': -1,
        'seed': seed
    }
    cv_results = _lgb.cv(
        params, dtrain,
        num_boost_round=1000,
        nfold=folds,
        seed=seed,
        stratified=True
    )
    # cv_results may be dict or DataFrame
    if hasattr(cv_results, 'columns'):
        auc_col = [c for c in cv_results.columns if 'auc' in c.lower() and 'mean' in c.lower()][0]
        return int(cv_results[auc_col].idxmax()) + 1
    else:
        auc_key = [k for k in cv_results if 'auc' in k.lower() and 'mean' in k.lower()][0]
        return int(np.argmax(cv_results[auc_key])) + 1

# -----------------------------------------------------------------------------
# 2) ULTRADATAPROCESSOR 
# -----------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder as _LE
from sklearn.impute import KNNImputer

class UltraDataProcessor:
    def detect_feature_types_advanced(self, df):
        numeric, binary, ordinal, categorical = [], [], [], []
        for c in df.columns:
            nunq = df[c].nunique()
            conv = pd.to_numeric(df[c], errors='coerce')
            ratio = conv.notna().sum()/len(df)
            if ratio>0.95:
                if nunq==2:
                    binary.append(c)
                elif nunq<=15 and all(float(x).is_integer() for x in df[c].dropna().head(100)):
                    ordinal.append(c)
                else:
                    numeric.append(c)
            else:
                categorical.append(c)
        return numeric, binary, ordinal, categorical

    def advanced_categorical_encoding(self, X_tr, X_te, y_tr, col):
        Xc = X_tr[col].astype(str).fillna('MISSING')
        Xt = X_te[col].astype(str).fillna('MISSING')
        feats = {}
        # Label
        le = _LE().fit(pd.concat([Xc,Xt]))
        feats[f'{col}_label'] = (le.transform(Xc), le.transform(Xt))
        # Frequency
        freq = Xc.value_counts().to_dict()
        feats[f'{col}_freq'] = (Xc.map(freq).fillna(0).values, Xt.map(freq).fillna(0).values)
        # Target smoothing
        overall = y_tr.mean()
        tmap = {}
        for v in Xc.unique():
            mask = (Xc==v)
            if mask.sum()>5:
                m = y_tr[mask].mean(); cnt=mask.sum()
                tmap[v] = (m*cnt + overall*10)/(cnt+10)
            else:
                tmap[v] = overall
        feats[f'{col}_target'] = (Xc.map(tmap).fillna(overall).values, Xt.map(tmap).fillna(overall).values)
        # Likelihood
        pos = {}
        for v in Xc.unique():
            mask = (Xc==v)
            pos[v] = (y_tr[mask].sum()/mask.sum()) if mask.sum()>3 else overall
        feats[f'{col}_pos_rate'] = (Xc.map(pos).fillna(overall).values, Xt.map(pos).fillna(overall).values)
        # Rank
        rk = Xc.value_counts().rank(method='dense').to_dict()
        feats[f'{col}_rank'] = (Xc.map(rk).fillna(0).values, Xt.map(rk).fillna(0).values)
        return feats

    def process_numeric_features(self, X_tr, X_te, cols):
        out_tr, out_te = {}, {}
        for c in cols:
            a=pd.to_numeric(X_tr[c],errors='coerce')
            b=pd.to_numeric(X_te[c],errors='coerce')
            out_tr[c],out_te[c]=a.values,b.values
            if a.min()>0:
                out_tr[f'{c}_log']=np.log1p(a).values
                out_te[f'{c}_log']=np.log1p(b).values
            if a.min()>=0:
                out_tr[f'{c}_sqrt']=np.sqrt(a.fillna(0)).values
                out_te[f'{c}_sqrt']=np.sqrt(b.fillna(0)).values
            try:
                from sklearn.preprocessing import QuantileTransformer
                qt=QuantileTransformer(output_distribution='normal',random_state=RANDOM_STATE)
                qa=qt.fit_transform(a.fillna(a.median()).values.reshape(-1,1)).flatten()
                qb=qt.transform(b.fillna(a.median()).values.reshape(-1,1)).flatten()
                out_tr[f'{c}_quantile']=qa
                out_te[f'{c}_quantile']=qb
            except:
                pass
        return out_tr, out_te

    def fit_transform(self, X_tr, X_te, y_tr):
        num,binf,ordl,catf=self.detect_feature_types_advanced(X_tr)
        tr_num,te_num=self.process_numeric_features(X_tr,X_te,num+ordl)
        tr_cat,te_cat={},{}
        for c in catf+binf:
            ef=self.advanced_categorical_encoding(X_tr,X_te,y_tr,c)
            for k,(u,v) in ef.items():
                tr_cat[k],te_cat[k]=u,v
        all_tr={**tr_num,**tr_cat}
        all_te={**te_num,**te_cat}
        names=list(all_tr.keys())
        Xp=np.column_stack([all_tr[n] for n in names])
        Xt=np.column_stack([all_te[n] for n in names])
        imp=KNNImputer(n_neighbors=7,weights='distance')
        return imp.fit_transform(Xp), imp.transform(Xt)

# -----------------------------------------------------------------------------
# 3) ADVANCEDFEATUREENGINEER 
# -----------------------------------------------------------------------------
from sklearn.decomposition import PCA,FastICA,TruncatedSVD
from sklearn.cluster import KMeans

class AdvancedFeatureEngineer:
    def create_personality_features_v2(self,X):
        # â€¦ same as before â€¦
        if X.shape[1]==0: return np.zeros((X.shape[0],0))
        F=[np.std(X,axis=1),np.mean(X,axis=1),np.median(X,axis=1),
           np.max(X,axis=1)-np.min(X,axis=1)]
        q25=np.percentile(X,25,axis=1);q75=np.percentile(X,75,axis=1)
        F+=[np.sum(X<=q25[:,None],axis=1),np.sum(X>=q75[:,None],axis=1)]
        segs,sz=4,X.shape[1]//4
        for i in range(segs):
            st,ed=i*sz,(i+1)*sz if i<segs-1 else X.shape[1]
            F.append(np.mean(X[:,st:ed],axis=1))
        try:
            F.append(stats.skew(X,axis=1));F.append(stats.kurtosis(X,axis=1))
        except:
            F+=[np.zeros(X.shape[0]),np.zeros(X.shape[0])]
        ent=np.array([-np.sum((cnts:=np.unique(X[i],return_counts=True)[1])/len(X[i])*
                     np.log2(cnts/len(X[i])+1e-10)) for i in range(X.shape[0])])
        F.append(ent)
        return np.column_stack(F)

    def create_advanced_interactions(self,X,maxf=100):
        # â€¦ same as before â€¦
        if X.shape[1]<2: return np.zeros((X.shape[0],0))
        var=np.var(X,axis=0);top=np.argsort(var)[-min(15,len(var)):]
        I,cnt=[],0
        for i in range(len(top)):
            for j in range(i+1,len(top)):
                if cnt>=maxf: break
                a,b=top[i],top[j]
                I.append(X[:,a]*X[:,b]);cnt+=1
                if cnt<maxf:
                    I.append(X[:,a]+X[:,b]);cnt+=1
                if cnt<maxf and np.all(X[:,b]!=0):
                    I.append(X[:,a]/(X[:,b]+1e-8));cnt+=1
                if cnt<maxf:
                    I.append(X[:,a]-X[:,b]);cnt+=1
                if cnt>=maxf: break
            if cnt>=maxf: break
        return np.column_stack(I) if I else np.zeros((X.shape[0],0))

    def create_clustering_features(self,X_tr,X_te,clusters=[3,5,8,12]):
        # â€¦ same as before â€¦
        CT,CE=[],[]
        for n in clusters:
            if n<=X_tr.shape[0]:
                km=KMeans(n_clusters=n,random_state=RANDOM_STATE,n_init=10)
                trc=km.fit_predict(X_tr);tec=km.predict(X_te)
                CT.append(trc);CE.append(tec)
                dtr=km.transform(X_tr);dte=km.transform(X_te)
                for fn in [np.min,np.max,np.mean]:
                    CT.append(fn(dtr,axis=1));CE.append(fn(dte,axis=1))
        return (np.column_stack(CT) if CT else np.zeros((X_tr.shape[0],0)),
                np.column_stack(CE) if CE else np.zeros((X_te.shape[0],0)))

    def create_decomposition_features(self,X_tr,X_te):
        # â€¦ same as before â€¦
        DR,DE=[],[]
        try:
            nc=min(12,X_tr.shape[1],X_tr.shape[0])
            if nc>=2:
                pca=PCA(n_components=nc,random_state=RANDOM_STATE)
                DR.append(pca.fit_transform(X_tr));DE.append(pca.transform(X_te))
        except: pass
        try:
            nc=min(8,X_tr.shape[1],X_tr.shape[0])
            if nc>=2:
                ica=FastICA(n_components=nc,random_state=RANDOM_STATE,max_iter=500)
                DR.append(ica.fit_transform(X_tr));DE.append(ica.transform(X_te))
        except: pass
        try:
            nc=min(10,X_tr.shape[1]-1,X_tr.shape[0])
            if nc>=1:
                svd=TruncatedSVD(n_components=nc,random_state=RANDOM_STATE)
                DR.append(svd.fit_transform(X_tr));DE.append(svd.transform(X_te))
        except: pass
        if DR: return np.hstack(DR),np.hstack(DE)
        return np.zeros((X_tr.shape[0],0)),np.zeros((X_te.shape[0],0))

    def fit_transform(self,X_tr,X_te):
        p_tr,p_te=self.create_personality_features_v2(X_tr), self.create_personality_features_v2(X_te)
        i_tr,i_te=self.create_advanced_interactions(X_tr), self.create_advanced_interactions(X_te)
        c_tr,c_te=self.create_clustering_features(X_tr,X_te)
        d_tr,d_te=self.create_decomposition_features(X_tr,X_te)
        mats_tr=[X_tr,p_tr,i_tr,c_tr,d_tr]
        mats_te=[X_te,p_te,i_te,c_te,d_te]
        Xf=np.hstack([m for m in mats_tr if m.size>0])
        Yf=np.hstack([m for m in mats_te if m.size>0])
        return Xf,Yf


# -----------------------------------------------------------------------------
if __name__=='__main__':
    # Load
    train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test =pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    X_df=train.drop(['id','Personality'],axis=1)
    y_ser=train['Personality']
    tgt_enc=LabelEncoder().fit(y_ser)
    y=tgt_enc.transform(y_ser)

    # Encode objects for CV tuning
    X_tune=X_df.copy()
    for c in X_tune.select_dtypes(include=['object']):
        X_tune[c]=LabelEncoder().fit_transform(X_tune[c].astype(str))

    # 1) Preâ€�tune XGB & LGB rounds
    print("ğŸ”� Tuning XGBoost & LightGBM roundsâ€¦")
    xgb_rounds=tune_xgb_rounds(X_tune,y)
    print(f"  â€¢ XGBoost â†’ {xgb_rounds}")
    if lgb:
        lgb_rounds=tune_lgb_rounds(X_tune,y)
        print(f"  â€¢ LightGBM â†’ {lgb_rounds}")
    # fix CatBoost iterations
    cat_rounds=500 if has_cat else None
    if has_cat:
        print(f"  â€¢ CatBoost â†’ {cat_rounds} (fixed)")

    # 2) Ultraâ€�advanced processing
    udp=UltraDataProcessor()
    X_proc,X_test_proc=udp.fit_transform(X_df,test.drop('id',axis=1),y)

    # 3) Advanced feature engineering
    afe=AdvancedFeatureEngineer()
    X_feat,X_test_feat=afe.fit_transform(X_proc,X_test_proc)

    # 4) Impute + scale
    imp=SimpleImputer(strategy='median')
    X_imp=imp.fit_transform(X_feat)
    X_test_imp=imp.transform(X_test_feat)
    scl=StandardScaler()
    X=scl.fit_transform(X_imp)
    X_test=scl.transform(X_test_imp)

    # 5) Base learners (no earlyâ€�stopping in stacking)
    learners=[(
        'xgb', xgb.XGBClassifier(
            n_estimators=xgb_rounds,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=4
        )
    )]
    if lgb:
        learners.append((
            'lgb', lgb.LGBMClassifier(
                n_estimators=lgb_rounds,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=RANDOM_STATE,
                n_jobs=4
            )
        ))
    if has_cat:
        learners.append((
            'cat', CatBoostClassifier(
                iterations=cat_rounds,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                verbose=False,
                random_seed=RANDOM_STATE
            )
        ))
    learners.append((
        'rf', RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE
        )
    ))

    # 6) Feature selector
    selector=SelectKBest(f_classif, k='all')

    # 7) Stacking ensemble
    stack=StackingClassifier(
        estimators=learners,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=True
    )

    # 8) 5â€�fold CV
    print("\nğŸš€ 5â€‘fold CV on stacked ensemble...")
    cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE)
    scores=[]
    for tr,va in cv.split(X,y):
        stack.fit(selector.fit_transform(X[tr],y[tr]), y[tr])
        preds=stack.predict(selector.transform(X[va]))
        scores.append(accuracy_score(y[va],preds))
    print(f"Stacking CV accuracy: {np.mean(scores):.4f} Â± {np.std(scores)*2:.4f}")

    # 9) Final train & submission
    stack.fit(selector.fit_transform(X,y), y)
    final_preds=stack.predict(selector.transform(X_test))
    submission=pd.DataFrame({
        'id': test['id'],
        'Personality': tgt_enc.inverse_transform(final_preds)
    })
    submission.to_csv('submission.csv',index=False)
    print("âœ… submission.csv written")


