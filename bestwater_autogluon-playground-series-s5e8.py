"Good Luck"


%%time 
# !pip install autogluon.tabular --no-cache-dir -q  >/dev/null 2>&1   
!pip install autogluon.tabular -q  >/dev/null 2>&1  
!pip install --upgrade xgboost  >/dev/null 2>&1   


import os
import numpy as np
import pandas as pd    

from autogluon.tabular import TabularDataset, TabularPredictor
import autogluon.tabular
print(autogluon.tabular.__version__) 


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv") 
train.head()


# å‰”é™¤æ— æ•ˆç‰¹å¾�
drop_features =  [
    'id', 
] 
train.drop(columns=drop_features, inplace=True)
test.drop(columns=drop_features, inplace=True)
train


test.head()


import os
import sys

def is_interactive():
    """åˆ¤æ–­æ˜¯å�¦åœ¨äº¤äº’ç�¯å¢ƒï¼ˆJupyter Notebook / Kaggle å‰�å�°ï¼‰è¿�è¡Œ"""
    try:
        from IPython import get_ipython
        ipy = get_ipython()

        # æ£€æŸ¥æ˜¯å�¦åœ¨ Jupyter / IPython äº¤äº’æ¨¡å¼�ä¸‹è¿�è¡Œ
        if ipy is None:
            return False  # é�� Jupyter ç�¯å¢ƒ
        
        shell = ipy.__class__.__name__
        is_jupyter = shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]

        # Kaggle å��å�°è¿�è¡Œæ—¶ï¼ŒKAGGLE_KERNEL_RUN_TYPE = "batch"
        is_kaggle_batch = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"

        return is_jupyter and not is_kaggle_batch
    except:
        return False  # ä»»ä½•å¼‚å¸¸æƒ…å†µéƒ½è®¤ä¸ºæ˜¯é��äº¤äº’ç�¯å¢ƒ

# æ£€æµ‹ç�¯å¢ƒ
print('Interactive?', is_interactive())

if is_interactive():
    time_limit = 200 
    feature_time_limit = 200
    train = train[:1000] 
    print("ğŸ”¹ äº¤äº’æ¨¡å¼�ï¼šä½¿ç”¨ 1000 æ�¡æ•°æ�®å¿«é€Ÿæµ‹è¯•", time_limit, feature_time_limit)
else:
    time_limit = 1000 
    feature_time_limit = 1000
    print("âœ… Kaggle Notebook å��å�°è¿�è¡Œ", time_limit, feature_time_limit) 


%%time
from autogluon.tabular import TabularDataset, TabularPredictor 

predictor = TabularPredictor(label = 'y',
                                 eval_metric = 'roc_auc',  
                                # groups="fold",
                            ).fit(train, 
                time_limit = time_limit,     
                # tuning_data = tuning_data ,
                presets='best_quality',  
                # auto_stack=True,         
                # presets= 'best_quality', 
                # use_bag_holdout=True ,  # âœ… åŠ ä¸Šè¿™ä¸ª

                # auto_stack=True,           
                # excluded_model_types=['KNN'],  
                # hyperparameters = 'very_light', 
                # excluded_model_types=['KNN'],  
                # ag_args_fit={"stopping_metric": "log_loss"},
                # num_cpus=4, 
                num_gpus=2, 
             )


extra_metrics  = ["accuracy",    "balanced_accuracy",    "mcc",    "log_loss",    "pac",     "quadratic_kappa",
    "roc_auc",    "average_precision",    "precision",    "precision_macro",    "precision_micro",    "precision_weighted",    "recall",
    "recall_macro",    "recall_micro",    "recall_weighted",    "f1",    "f1_macro",    "f1_micro",    "f1_weighted"]
lb = predictor.leaderboard(train, extra_metrics=extra_metrics, silent=True)
lb.style.background_gradient(subset=extra_metrics, cmap="RdYlGn")


os.makedirs('AutogluonFiles', exist_ok=True)  


%%time 
probs = predictor.predict_proba(test)  # è¿”å›�æ¯�ä¸ªæ ·æœ¬å¯¹æ¯�ç±»è‚¥æ–™çš„é¢„æµ‹æ¦‚ç�‡ï¼ˆDataFrame æ ¼å¼�ï¼‰ 


# ä¿�å­˜ä¸º .npy æ–‡ä»¶
np.save('AutogluonFiles/auto_oof.npy', probs.to_numpy())
probs


# è�·å�– OOF é¢„æµ‹æ•°æ�®
oof_predictions = predictor.predict_proba_oof() # è¿”å›�ä¸€ä¸ª Pandas Series

# å°† OOF é¢„æµ‹æ•°æ�®è½¬ä¸º DataFrame å¹¶æ·»åŠ åˆ—å��
# oof_predictions_df = oof_predictions.to_frame(name='oof_prediction')

# ä¿�å­˜ OOF æ•°æ�®åˆ° CSV æ–‡ä»¶
oof_predictions.to_csv('AutogluonFiles/oof_predictions.csv', index=False)

# ä¿�å­˜ä¸º .npy æ–‡ä»¶
np.save('AutogluonFiles/auto_oof.npy', oof_predictions.to_numpy())

# æŸ¥çœ‹æ•°æ�®
oof_predictions


submission = pd.DataFrame({'id': sample_submission['id'], 'Personality': probs[1]})
submission.to_csv('AutogluonFiles/submission.csv', index=False)   

submission


%%time 
feature_importance = predictor.feature_importance(train,time_limit=feature_time_limit)  
feature_importance.to_csv("feature_importance.csv") 


pd.set_option('display.max_rows', 500)  
pd.set_option('display.max_columns', 500)
feature_importance

