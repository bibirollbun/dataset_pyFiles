import pickle


from lightgbm import LGBMClassifier,log_evaluation,early_stopping
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
import gc
import pandas as pd


oof_pred_probs={}
test_pred_probs={}
scores={}


!ls /kaggle/input


with open('/kaggle/input/optimal-fertilizer/result.pkl','rb') as f:
    meta=pickle.load(f)

oof_pred_probs=meta['c1']
test_pred_probs=meta['c2']
scores=meta['c3']


oof_pred_probs


oof_pred_probs['XGBoost'].shape


test_pred_probs['XGBoost'].shape


scores['XGBoost']


X=pd.DataFrame(np.concatenate(list(oof_pred_probs.values()),axis=1))
X_test=pd.DataFrame(np.concatenate(list(test_pred_probs.values()),axis=1))


X_test


train_df=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')


target='Fertilizer Name'
n_folds=5
seed=42


le=LabelEncoder()
train_df[target]=le.fit_transform(train_df[target])



y=train_df[target]



def map3(y_true, y_pred_probs):
    y_true = [[x] for x in y_true]
    y_pred_probs = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1].tolist()
    
    def ap3(y_true, y_pred_probs):
        y_pred_probs = y_pred_probs[:3]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(y_pred_probs):
            if p in y_true and p not in y_pred_probs[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not y_true:
            return 0.0

        return score
    
    return np.mean([ap3(a,p) for a,p in zip(y_true, y_pred_probs)])


class Trainer:
    def __init__(self, model):
        self.model = model
        

    def fit_predict(self, X, y, X_test, X_original=None, y_original=None, fit_args={}):
        print(f"Training {self.model.__class__.__name__}\n")
        
        scores = []        
        oof_pred_probs = np.zeros((X.shape[0], 7))
        test_pred_probs = np.zeros((X_test.shape[0], 7))
        
        skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if X_original is not None and y_original is not None:
                X_train = pd.concat([X_train, X_original])
                y_train = pd.concat([y_train, y_original])
            
            model = clone(self.model)
            
            if fit_args:
                model.fit(X_train, y_train, **fit_args, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
            
            y_pred_probs = model.predict_proba(X_val)
            oof_pred_probs[val_idx] = y_pred_probs
            
            temp_test_pred_probs = model.predict_proba(X_test)
            test_pred_probs += temp_test_pred_probs / n_folds
            
            score = map3(y_val, y_pred_probs)
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
            
            if fit_args:
                print(f"\n--- Fold {fold_idx + 1} - MAP@3: {score:.6f}\n\n")
            else:
                print(f"--- Fold {fold_idx + 1} - MAP@3: {score:.6f}")
                            
        overall_score = map3(y, oof_pred_probs)
            
        print(f"\n------ Overall MAP@3: {overall_score:.6f} | Average MAP@3: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
        
        return oof_pred_probs, test_pred_probs, scores

    def tune(self, X, y):        
        scores = []        
        
        skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            y_pred_probs = model.predict_proba(X_val)            
            score = map3(y_val, y_pred_probs)
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
            
        return np.mean(scores)


lr_model=LogisticRegression(random_state=42,max_iter=10000,solver='liblinear',penalty='l2',
                            C=32.89802104596641,
                            tol=0.0029878837974181643,
                            fit_intercept=True)
lr_trainer=Trainer(lr_model)
_,lr_test_pred_probs,scores['Ensemble']=lr_trainer.fit_predict(X,y,X_test)


final_predictions=[]
for i in np.argsort(lr_test_pred_probs)[:,-3:][:,::-1]:
  prediction=le.inverse_transform(i)
  final_predictions.append(" ".join(prediction))


sub=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
sub[target]=final_predictions


np.mean(scores['Ensemble'])


sub.to_csv('submission.csv',index=False)


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=False)
order = scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()

padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.4))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", color="grey")
axs[0].set_title(f"Fold MAP@3")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color="grey")
axs[1].set_title(f"Mean MAP@3")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = "cyan" if "ensemble" in model.lower() else "grey"
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va="center")

plt.tight_layout()
plt.show()

