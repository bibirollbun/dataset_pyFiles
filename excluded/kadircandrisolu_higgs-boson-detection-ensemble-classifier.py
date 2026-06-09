import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
pio.renderers.default = 'iframe'
pd.options.display.max_columns = None
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("train.csv")
print("Train shape:",train.shape)
train.head()


print(train.describe())
print(train.info())

class_distribution = train['label'].value_counts(normalize=True)
print(class_distribution)


plt.figure(figsize=(12, 10))
correlation = train.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


n_features = len(train.columns) - 1  # Exclude target
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

for i, feature in enumerate(train.columns[:-1]):
    if i % 9 == 0: 
        plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, (i % 9) + 1)
    sns.histplot(data=train, x=feature, hue='label', kde=True, bins=30, legend=False if (i % 9) != 0 else True)
    plt.title(f'Distribution of {feature}', fontsize=10)
    plt.tight_layout()
    
    if (i % 9) == 8 or i == n_features - 1:  
        plt.tight_layout()
        plt.show()


class CFG:
    train_path = Path('train.csv')
    test_path = Path('test.csv')
    subm_path = Path('sample_submission.csv')
    
    colorscale = 'Redor'
    color = '#A2574F'

    n_splits = 5

    weights = [0.50, 0.50]

    xgb_params = {
            "objective": "binary:logistic",  
            "eval_metric": "auc",
            "max_depth": 8,
            "eta": 0.01,                     
            "subsample": 0.8,                  
            "n_estimators": 2000,
            "random_state": 42,
            "tree_method": "hist"            
        }

    lgb_params = {
        'objective': 'binary',           
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 37,                
        'learning_rate': 0.01,  
        'bagging_fraction': 0.8,                   
        'n_estimators': 2000,
        'verbose': 0,
        'random_state': 42
    }


class MD:

    def __init__(self, data, n_splits):
        self.data = data
        self.n_splits = n_splits
        
    def _prepare_cv(self):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(self.data))
        
        return cv, oof_preds
    
    def validate_model(self, preds, title):
    
        y_true = self.data['label']
        auc_score = roc_auc_score(y_true, preds)
        print(f'Overall AUC Score for {title}: {auc_score:.5f}')
    
    def train_model(self, params, title):
        X = self.data.drop(['label'], axis=1)
        y = self.data['label']
        models, fold_scores = [], []
        
        cv, oof_preds = self._prepare_cv()
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
            print(f"Fold {fold + 1}")
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
            if title.startswith('LightGBM'):
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.log_evaluation(70)]
                )
                
            elif title.startswith('XGBoost'):
                model = XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=0
                )
                
            models.append(model)
            
            if title.startswith('LightGBM') or title.startswith('XGBoost'):
                oof_preds[valid_index] = model.predict_proba(X_valid)[:, 1]
                    
            fold_auc = roc_auc_score(y_valid, oof_preds[valid_index])
            fold_scores.append(fold_auc)
            print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
            
        overall_auc = roc_auc_score(y, oof_preds)
        print(f"\n{title} Overall AUC: {overall_auc:.4f}")
        print(f"Fold AUCs: {[f'{score:.4f}' for score in fold_scores]}")
        print(f"Average Fold AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        self.plot_roc_curve(y, oof_preds, title)
        
        return models, oof_preds
    
    def infer_model(self, data, models):
        return np.mean([model.predict(data) for model in models], axis=0)

    def plot_roc_curve(self, y_true, y_pred, title):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title} - Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_feature_importance(self, models, title):
        importance_df = pd.DataFrame()
        feature_names = self.data.drop(['label'], axis=1).columns
        
        for i, model in enumerate(models):
            if title.startswith('LightGBM'):
                importances = pd.DataFrame({
                    'feature': feature_names,
                    f'importance_fold_{i}': model.feature_importances_
                })
            elif title.startswith('XGBoost'):
                importances = pd.DataFrame({
                    'feature': feature_names,
                    f'importance_fold_{i}': model.feature_importances_
                })
                
            if importance_df.empty:
                importance_df = importances
            else:
                importance_df = importance_df.merge(importances, on='feature')
        
        importance_df['mean_importance'] = importance_df.filter(like='importance_fold').mean(axis=1)
        importance_df['std_importance'] = importance_df.filter(like='importance_fold').std(axis=1)
        importance_df = importance_df.sort_values('mean_importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(14, max(12, len(importance_df) * 0.25)))
        ax.barh(range(len(importance_df)), importance_df['mean_importance'], 
                xerr=importance_df['std_importance'], 
                align='center', 
                alpha=0.8,
                capsize=5)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{title} Feature Importance with Standard Deviation')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        


md = MD(train, CFG.n_splits)


xgb, xgb_oof_preds = md.train_model(CFG.xgb_params, title='XGBoost')


md.plot_feature_importance(xgb, 'XGBoost')


xgb_preds = md.infer_model(test, xgb)


lgb, lgb_oof_preds = md.train_model(CFG.lgb_params, title='LightGBM')


md.plot_feature_importance(xgb, 'LightGBM')


lgb_preds = md.infer_model(test, lgb)


oof_preds = [xgb_oof_preds,lgb_oof_preds]
preds = [xgb_preds,lgb_preds]


ensemble_oof_preds = np.dot(CFG.weights, oof_preds)


md.validate_model(ensemble_oof_preds, 'Ensemble Model')


ensemble_preds = np.dot(CFG.weights, preds)


sample_submission = pd.read_csv(CFG.subm_path)


submission = sample_submission.copy()  
submission['Predicted'] = ensemble_preds

print(submission.head())


submission.to_csv('LGB+XGB_baseline.csv', index=False, float_format='%.18e')




