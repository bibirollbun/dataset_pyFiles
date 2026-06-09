# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install scikit-learn==1.5.2 koolbox


import warnings 
from IPython.display import display,HTML
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_regression 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns 
from koolbox import Trainer
from catboost import CatBoostClassifier   
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  
from scipy.special import logit    
import joblib 
import optuna
import json
warnings.filterwarnings("ignore")


class geeks:
    train_dir="/kaggle/input/playground-series-s5e7/train.csv"
    test_dir="/kaggle/input/playground-series-s5e7/test.csv" 
    sub_dir="/kaggle/input/playground-series-s5e7/sample_submission.csv"
    orig_dir="/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv"
    
    seed=42

    target="Personality" 

    n_folds=5  

    cv=StratifiedKFold(n_splits=n_folds,random_state=seed,shuffle=True) 

    numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                         'Going_outside', 'Friends_circle_size', 'Post_frequency']
    categorical_features = ['Stage_fear', 'Drained_after_socializing']

    metric=accuracy_score 

    n_optuna_trials=500     


def load_and_explore_data():
    train_df=pd.read_csv(geeks.train_dir,index_col="id")
    test_df=pd.read_csv(geeks.test_dir,index_col="id")

    print(train_df.info())

    missing_df=pd.DataFrame({"Features":train_df.columns,
                             "Missing_Count":train_df.isna().sum().values,
                             "Missing_Percentage":(train_df.isna().sum().values/len(train_df)*100).round(2)
                            }) 
    missing_df=missing_df[missing_df["Missing_Percentage"]>0].sort_values(by="Missing_Percentage",ascending=False) 

    if len(missing_df)>0:  
        display(missing_df.style.background_gradient(subset=["Missing_Percentage"],cmap='YlOrRd'))

    return train_df,test_df
        

    
    


train_df,test_df=load_and_explore_data()


train_df[geeks.target].value_counts()


def visualize(train_df): 

    fig,ax=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle("Personality Type Distribution Analysis",fontsize=16) 

    target_counts = train_df[geeks.target].value_counts()
    colors = ['#FF6B6B', '#4ECDC4'] 

    wedges,texts,autotexts=ax[0].pie(
        target_counts.values,
        labels=target_counts.index, 
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.05,0.05), 
        shadow=True
    )    

    for text in autotexts:
        text.set_fontsize(12)
        text.set_weight('bold') 

    ax[0].set_title("Distribution Breakdown",fontsize=14,pad=20)      


    bars=ax[1].bar(target_counts.index,target_counts.values,color=colors,alpha=0.8,
                  edgecolor='black')  
    ax[1].set_title('Sample breakdown',fontsize=14,pad=20)      
    ax[1].set_xlabel('Personality Type',fontsize=12) 
    ax[1].set_ylabel('Number of Sample',fontsize=12)   

    for bar,count in zip(bars,target_counts.values):
        height=bar.get_height() 
        ax[1].text(bar.get_x()+bar.get_width()/2,height+100,f"{count:,}",ha="center",
                   va="bottom",fontweight='bold',fontsize=11)

    ax[1].grid(axis='y',alpha=0.3) 

    plt.tight_layout() 
    plt.show() 

visualize(train_df)


def create_distribution(train_df):
    
    fig,ax=plt.subplots(3,3,figsize=(18,16)) 
    fig.suptitle(" Feature Distributions by Personality Type",fontsize=18,y=1.02) 
    ax=ax.flatten() 

    for i,feature in enumerate(geeks.numerical_features):
        data_to_plot=train_df.dropna(subset=[feature])  
        sns.violinplot(data_to_plot,x=geeks.target,
                       y=feature,ax=ax[i],palette=['#FF6B6B', '#4ECDC4']) 
        for j,per in enumerate(['Extrovert','Introvert']):
            mean=data_to_plot[data_to_plot[geeks.target]==per][feature].mean() 
            ax[i].axhline(y=mean,xmin=j*0.5+0.1,xmax=j*0.5+0.4,color='black',linestyle="--",
                         linewidth=2)  
            ax[i].text(j,mean,f'{mean:.2f}',ha='center',va='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        ax[i].set_title(f'{feature}', fontsize=14, fontweight='bold')
        ax[i].set_xlabel('')


    
    for i,feature in enumerate(geeks.categorical_features):   
        axes=ax[len(geeks.numerical_features)+i] 

        ct=pd.crosstab(train_df[feature],train_df[geeks.target],normalize='columns')*100 

        ct.plot(kind='bar',ax=axes,alpha=0.8,color=['#FF6B6B', '#4ECDC4'])  

        axes.set_title(f'{feature} Distribution (%)', fontsize=14, fontweight='bold')
        axes.set_xlabel('')
        axes.set_ylabel('Percentage')
        axes.legend(title='Personality', loc='upper right')
        axes.set_xticklabels(axes.get_xticklabels(), rotation=0)  

    for i in  range(len(geeks.numerical_features) + len(geeks.categorical_features), len(ax)): 
        ax[i].axis('off')
    plt.tight_layout() 
    plt.show()

        


create_distribution(train_df)


def create_correlation(train_df):
    corr_data=train_df.copy() 

    corr_data['Stage_fear']=corr_data['Stage_fear'].map({"No":0,"Yes":1}) 
    corr_data['Drained_after_socializing']=corr_data['Drained_after_socializing'].map({'No':0,
                                                                                      "Yes":1})
    corr_data[geeks.target]=corr_data[geeks.target].map({'Extrovert':0,
                                                        'Introvert':1})  

    corr_data=corr_data.corr() 

    mask=np.triu(np.ones_like(corr_data,dtype=bool)) 

    plt.figure(figsize=(12,10))
    
    sns.heatmap(corr_data,mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'shrink':.8,'label':"Correlation Coefficient"},
                annot_kws={'fontsize':10}
               )    
    plt.title('Feature Correlation Coefficient',fontsize=16,fontweight='bold')  

    plt.xticks(rotation=45,ha='right') 

    plt.yticks(rotation=0)    

    target_corr = corr_data[geeks.target].drop(geeks.target).sort_values(ascending=False) 

    for feature, corr_value in target_corr.items():
        direction = "↑" if corr_value > 0 else "↓"
        print(f"   {feature}: {corr_value:+.4f} {direction}")
    
    return target_corr

corr_data=create_correlation(train_df)


def calculate_feature_importance(train_df): 

    X_temp=train_df.drop([geeks.target],axis=1).copy() 
    y_temp=train_df[geeks.target].map({'Extrovert':0,'Introvert':1}) 

    X_temp['Stage_fear'] = X_temp['Stage_fear'].map({'No': 0, 'Yes': 1})
    X_temp['Drained_after_socializing'] = X_temp['Drained_after_socializing'].map(
        {'No': 0, 'Yes': 1})    
    X_temp=X_temp.fillna(0) 
    mi_score=mutual_info_regression(X_temp,y_temp,random_state=geeks.seed)    

    mi_df=pd.DataFrame({'Feature':X_temp.columns,
                        'Mutual_Information':mi_score}).sort_values('Mutual_Information',ascending=False)   

    display(mi_df.style.background_gradient(subset=['Mutual_Information'],cmap='YlOrRd'))   

calculate_feature_importance(train_df)    


train=pd.read_csv(geeks.train_dir,index_col='id') 
test=pd.read_csv(geeks.test_dir,index_col='id')  

original=pd.read_csv(geeks.orig_dir)   
original=original.rename(columns={'Personality':'match_p'})
original=original.drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency'])
train=train.merge(original,how='left')
test=test.merge(original,how='left')

cat_cols = ["Stage_fear", "Drained_after_socializing"]

train[cat_cols]=train[cat_cols].fillna('missing').astype('category')   
test[cat_cols]=test[cat_cols].fillna('missing').astype('category')       

train[geeks.target]=train[geeks.target].map({'Extrovert':0,'Introvert':1}) 
train["match_p"] = train["match_p"].map({"Extrovert": 0, "Introvert": 1})
test["match_p"] = test["match_p"].map({"Extrovert": 0, "Introvert": 1}) 

X_train=train.drop(geeks.target,axis=1)                    
y_train=train[geeks.target] 

X_test=test
_X_test=test.copy()


def save_submission(name,X_test,test_pred_probs,score,threshold=0.5): 
    sub=pd.read_csv(geeks.sub_dir)      
    sub[geeks.target]=(test_pred_probs>threshold).astype(int) 
    sub.loc[X_test['match_p']==0,geeks.target]=1
    sub.loc[X_test['match_p']==1,geeks.target]=0   
    sub[geeks.target]=sub[geeks.target].map({0:'Extrovert',1:'Introvert'})  
    sub.to_csv(f'submission.csv',index=False)
    return sub.head()


scores={} 
oof_pred_probs={} 
test_pred_probs={}


cb_params={
    "border_count":39, 
    "colsample_bylevel":0.19459088572914465, 
    "depth":5,
    "iterations":1467,
    "l2_leaf_reg":31.236169478676036,
    "learning_rate": 0.06852669420904771,
    "min_child_samples":160,   
    "random_state":42, 
    "random_strength":0.8517786189616939, 
    "scale_pos_weight":1.1691394390533685, 
    "subsample":0.3192330024411618, 
    "verbose":False, 
    "cat_features":cat_cols
} 

cb_trainer=Trainer(CatBoostClassifier(**cb_params),
                  cv=geeks.cv,
                  metric=geeks.metric,
                  use_early_stopping=False,
                  task='binary',
                  metric_precision=6)   
cb_trainer.fit(X_train,y_train) 
scores['CatBoost']=cb_trainer.fold_scores 
oof_pred_probs['CatBoost']=cb_trainer.oof_preds 
test_pred_probs['CatBoost']=cb_trainer.predict(X_test)


xgb_params={
    "colsample_bylevel":0.8168489864941239,
    "colsample_bytree": 0.8850485490950061,
    "colsample_bynode":0.8379339940113913, 
    "gamma":2.3977359439809276, 
    "learning_rate":0.0616974880921061,  
    "max_depth":8,
    "max_leaves":10,
    "min_child_weight":10, 
    "n_estimators":696, 
    "n_jobs":-1,
    "random_state":42, 
    "reg_alpha":1.849084818346014, 
    "reg_lambda":29.680324563362227, 
    "subsample":0.5902901569391961,
    "verbosity":0,
    "enable_categorical":True
}    

xgb_trainer=Trainer(XGBClassifier(**xgb_params),
                   cv=geeks.cv,
                   metric=geeks.metric,
                   task='binary', 
                   metric_precision=6)
xgb_trainer.fit(X_train,y_train) 
scores['XGBoost']=xgb_trainer.fold_scores
oof_pred_probs['XGBoost']=xgb_trainer.oof_preds 
test_pred_probs['XGBoost']=xgb_trainer.predict(X_test)


lgbm_params={
    "boosting_type":"gbdt",
    "colsample_bytree":0.6467443250209886, 
    "learning_rate":0.06547186748153115, 
    "min_child_samples":34,
    "min_child_weight":0.24399244943904663, 
    "n_estimators":498, 
    "n_jobs":-1, 
    "num_leaves":158, 
    "random_state":42,   
    "reg_alpha":6.568921253574134,
    "reg_lambda":62.66165355751099,
    "subsample":0.0011019938618584968,
    "verbose":-1
}
lgbm_goss_params={
    "boosting_type":"goss", 
    "colsample_bytree":0.8384834064170148, 
    "learning_rate": 0.07006829797238343, 
    "min_child_samples": 46,
    "min_child_weight": 0.7625394962666617,
    "n_estimators": 1887,
    "n_jobs": -1,
    "num_leaves": 341,
    "random_state": 42,
    "reg_alpha": 10.53082019937197,
    "reg_lambda": 67.44600065144685,
    "subsample": 0.4925008305336127,
    "verbose": -1
}

lgbm_dart_params={
    "boosting_type": "dart",
    "colsample_bytree": 0.7592971191793424,
    "learning_rate": 0.046141766106846074,
    "min_child_samples": 18,
    "min_child_weight": 0.4740109054323218,
    "n_estimators": 4035,
    "n_jobs": -1,
    "num_leaves": 393,
    "random_state": 42,
    "reg_alpha": 48.016799341666605,
    "reg_lambda": 89.12860300833658,
    "subsample": 0.016333358901112538,
    "verbose": -1
}



lgbm_gbdt_trainer=Trainer(
    LGBMClassifier(**lgbm_params), 
    cv=geeks.cv,
    metric=geeks.metric, 
    use_early_stopping=False,
    task='binary', 
    metric_precision=6
) 
lgbm_gbdt_trainer.fit(X_train,y_train) 

scores['Lightgbm_gbdt']=lgbm_gbdt_trainer.fold_scores 
oof_pred_probs['Lightgbm_gbdt']=lgbm_gbdt_trainer.oof_preds 
test_pred_probs['Lightgbm_gbdt']=lgbm_gbdt_trainer.predict(X_test)


lgbm_goss_trainer=Trainer(LGBMClassifier(**lgbm_goss_params),
                         cv=geeks.cv,
                         metric=geeks.metric,
                         use_early_stopping=False,
                         task='binary',
                         metric_precision=6) 

lgbm_goss_trainer.fit(X_train,y_train) 

scores['Lightgbm_goss']=lgbm_goss_trainer.fold_scores 
oof_pred_probs['Lightgbm_goss']=lgbm_goss_trainer.oof_preds
test_pred_probs['Lightgbm_goss']=lgbm_goss_trainer.predict(X_test)


lgbm_dart_trainer=Trainer(LGBMClassifier(**lgbm_dart_params),
                         cv=geeks.cv,
                         metric=geeks.metric, 
                         task='binary',
                         use_early_stopping=False,
                         metric_precision=6) 

lgbm_dart_trainer.fit(X_train,y_train) 
scores['Lightgbm_dart']=lgbm_dart_trainer.fold_scores 
oof_pred_probs['Lightgbm_dart']=lgbm_dart_trainer.oof_preds 
test_pred_probs['Lightgbm_dart']=lgbm_dart_trainer.predict(X_test)


def plot_weights(weight,title):
    sorted_indices=np.argsort(weight[0])[::-1] 
    sorted_coeffs=np.array(weight[0])[sorted_indices] 
    sorted_model_names=np.array(list(oof_pred_probs.keys()))[sorted_indices] 

    plt.figure(figsize=(10,weight.shape[1]*0.4)) 
    ax=sns.barplot(x=sorted_coeffs,y=sorted_model_names,palette='RdYlGn_r') 

    for i,(value,name) in enumerate(zip(sorted_coeffs,sorted_model_names)):
        if value>=0:
            ax.text(value,i,f'{value:.3f}',va='center',ha='left',color='black') 
        else :
            ax.text(value,i,f'{value:.3f}',va='center',ha='right',color='black')   


    xlim=ax.get_xlim() 
    ax.set_xlim(xlim[0]-0.1*abs(xlim[0]),xlim[1]+0.1*abs(xlim[1]))    

    plt.title(title) 
    plt.xlabel('') 
    plt.ylabel('') 
    plt.tight_layout() 
    plt.show() 


X_train=logit(pd.DataFrame(oof_pred_probs).clip(1e-15,1-1e-15)) 
X_test=logit(pd.DataFrame(test_pred_probs).clip(1e-15,1-1e-15)) 

joblib.dump(oof_pred_probs,'oof_pred_probs.pkl') 
joblib.dump(test_pred_probs,'test_pred_probs.pkl')


def objective(trial):
    solver_penalty_option=[
        ('liblinear','l1'), 
        ('liblinear','l2'), 
        ('lbfgs','l2'), 
        ('lbfgs',None), 
        ('newton-cg','l2'),
        ('newton-cg',None), 
        ('newton-cholesky','l2'), 
        ('newton-cholesky',None)
    ]     
    solver,penalty=trial.suggest_categorical('solver_penalty',solver_penalty_option)  

    params={
        'random_state':geeks.seed,
        'max_iter':1000,
        'C':trial.suggest_float('C',0,1), 
        'tol':trial.suggest_float('tol',1e-6,1e-2),
        'fit_intercept':trial.suggest_categorical('fit_intercept',[True,False]),
        'class_weight':trial.suggest_categorical('class_weight',['balanced',None]), 
        'solver':solver,
        'penalty':penalty
    }   

    threshold=trial.suggest_float('threshold',0,1) 

    trainer=Trainer(LogisticRegression(**params),
                   cv=geeks.cv,
                   metric=geeks.metric,
                   metric_precision=6,
                   metric_threshold=threshold,
                   use_early_stopping=False,
                   verbose=False,
                   task='binary')   
    trainer.fit(X_train,y_train) 

    return np.mean(trainer.fold_scores)    


sampler=optuna.samplers.TPESampler(seed=geeks.seed,multivariate=True,n_startup_trials=geeks.n_optuna_trials//10)  
study=optuna.create_study(direction='maximize',sampler=sampler) 
study.optimize(objective,n_trials=geeks.n_optuna_trials,n_jobs=-1) 
best_params=study.best_params


solver,penalty=best_params['solver_penalty'] 
lr_params={
    'random_state':geeks.seed,
    'max_iter':1000, 
    'C':best_params['C'], 
    'tol':best_params['tol'], 
    'fit_intercept':best_params['fit_intercept'],
    'class_weight':best_params['class_weight'], 
    'solver':solver, 
    'penalty':penalty
}


print(json.dumps(lr_params,indent=2))


best_threshold = study.best_params['threshold']
print(f'Best threshold: {best_threshold:.3f}')


lr_trainer = Trainer(
    LogisticRegression(**lr_params),
    cv=geeks.cv,
    metric=geeks.metric,
    metric_threshold=best_threshold,
    metric_precision=6,
    use_early_stopping=False,
    task="binary",
)

lr_trainer.fit(X_train, y_train)

scores["LogisticRegression"] = lr_trainer.fold_scores
lr_test_pred_probs = lr_trainer.predict(X_test)


save_submission('logistic_regression',_X_test,lr_test_pred_probs,np.mean(scores['LogisticRegression']),best_threshold)


lr_coeffs=np.zeros((1,len(X_train.columns)))   
for estimator in lr_trainer.estimators:
    lr_coeffs +=estimator.coef_/geeks.n_folds


plot_weights(lr_coeffs,'LR Coefficients')


def objective(trial):
    weights=np.array([trial.suggest_float(m,-1,1) for m in oof_pred_probs.keys()]) 
    weights/=np.sum(weights)  

    preds=np.zeros(len(y_train))
    for m,weight in zip(oof_pred_probs.keys(),weights):
        preds+=oof_pred_probs[m]*weight    

    threshold=trial.suggest_float('threshold',0,1) 

    return accuracy_score(y_train,(preds>threshold).astype(int))        

sampler=optuna.samplers.TPESampler(seed=geeks.seed,multivariate=True,n_startup_trials=geeks.n_optuna_trials//10)  
study=optuna.create_study(direction='maximize',sampler=sampler) 
study.optimize(objective,n_trials=geeks.n_optuna_trials,n_jobs=-1) 
best_params=study.best_params


scores['WeightedAvearge']=[study.best_value]*geeks.n_folds


best_weights = np.array([study.best_params[m] for m in oof_pred_probs.keys()])
best_weights /= np.sum(best_weights)

best_weights={
    model:weight for model,weight in sorted(
        zip(oof_pred_probs.keys(),best_weights),
        key=lambda x:x[1],
        reverse=True)
}
print(json.dumps(best_weights,indent=2))


best_threshold = study.best_params['threshold']
print(f'Best threshold: {best_threshold:.3f}')


weighted_test_preds = np.zeros(len(test_pred_probs["CatBoost"]))
for m, weight in best_weights.items():
    weighted_test_preds += test_pred_probs[m] * weight


save_submission('weighted-ensemble', _X_test, weighted_test_preds, np.mean(scores['WeightedAvearge']), best_threshold)


scores=pd.DataFrame(scores) 
mean_scores=scores.mean().sort_values(ascending=False) 
order=scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()  
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding        

fig,axs=plt.subplots(1,2,figsize=(15,scores.shape[1]*0.4))   
boxplot=sns.boxplot(data=scores,order=order,ax=axs[0],orient='h',color='grey') 
axs[0].set_title('Fold Accuracy')  
axs[0].set_xlabel('')
axs[0].set_ylabel('')   

barplot=sns.barplot(x=mean_scores.values,y=mean_scores.index,ax=axs[1],color='grey') 
axs[1].set_title('Average Accuracy') 
axs[1].set_xlabel('') 
axs[1].set_ylabel('') 
axs[1].set_xlim(left=lower_limit,right=upper_limit)   

for i,(score,model) in enumerate(zip(mean_scores.values,mean_scores.index)):
    color='cyan' if 'logistic' in model.lower() or 'weighted' in model.lower() else 'gray' 
    boxplot.patches[i].set_facecolor(color) 
    barplot.patches[i].set_facecolor(color) 
    barplot.text(score,i,round(score,6),va='center')  

plt.tight_layout() 
plt.show()

