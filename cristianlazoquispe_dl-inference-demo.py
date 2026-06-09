!pip uninstall -y scikit-learn

!pip uninstall -y category-encoders

!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_lightning-2.4.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/torchmetrics-1.5.2-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabnet-4.1.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/einops-0.7.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabular-1.1.1-py2.py3-none-any.whl


import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

%load_ext autoreload
%autoreload 2


n_splits = 10
target_name = "target_5"
loss_name = ""
act_function = "sigmoid"
mse_pair_joinmodel = False
target_log_scale = False

use_masked_pair = False
mse_pair_joinloss = False

datos = [
#['', '', True, True, 0.6687345530407058],
#['', '', True, False, 0.6619418896818766],
['', '', False, True, 0.6794605661929045],
#['', '', False, False, 0.6724179105422328],
#['', 'sigmoid', True, False, 0.6701672954667133],
['', 'sigmoid', False, True, 0.6751429046114953],
#['', 'sigmoid', False, False, 0.6755276943598966],
#['margin_based_listnet_loss', '', True, True, 0.6649212775260886],
#['margin_based_listnet_loss', '', True, False, 0.6620994657911118],
#['margin_based_listnet_loss', 'sigmoid', True, False, 0.6603552472303216],
#['margin_based_listnet_loss', 'sigmoid', False, True, 0.6662931069080722],
]


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


sys.path.append("/kaggle/input/scriptscibmtr/CIBMTR-2025/")
from src.utils.analysis import my_df_describe
from src.cibmtr.metric import score
from src.cibmtr import targets
from src.pipelines import DL


#PATH_DATA = "../../../data/equity-post-HCT-survival-predictions/"
PATH_DATA = "/kaggle/input/equity-post-HCT-survival-predictions/"
test, train = DL.fe.load_data(PATH_DATA = PATH_DATA)

train["target_3"] = targets.create_target_NelsonAalenFitter(train,nelson_aalen_smoothing=0,by_race=True)
train["target_6"] = targets.create_target_KaplanMeierFitter(train,by_race=True)    
train["target_1"] = targets.create_target_NelsonAalenFitter(train,nelson_aalen_smoothing=0,n_splits = 5, use_folds = True)
train["target_5"] = targets.create_target_KaplanMeierFitter(train,by_race=False)    
train[["target_3Nrace_6Krace_sum","target_3Nrace_6Krace_mul"]] = targets.get_combinations(train,"target_3","target_6")

for target_name_aux in ["target_1","target_3","target_3Nrace_6Krace_sum"]:
    targets.compute_score_target(train,train[target_name_aux], name = target_name)



    
    
#!rm -r /kaggle/working/CIBMTR-2025



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



results = []
version_number = 1
#version_name = "cibmtrv1_analsisi1"
version_name = "folds10"
for loss_name,act_function,mse_pair_joinmodel,target_log_scale,_ in datos:
    name_version = f"target_5_{loss_name}_{act_function}_{mse_pair_joinmodel}_{target_log_scale}"

    #/kaggle/input/cibmtrv1/pytorch/cibmtrv1_analsisi1/2/CIBMTR-2025/code/cristian/baselines/results/models/target_5___False_False
    #/kaggle/input/cibmtrv1/pytorch/folds10/1/CIBMTR-2025/code/cristian/baselines/results/models/target_5__sigmoid_False_True
        
    model_path = f"/kaggle/input/cibmtrv1/pytorch/{version_name}/{version_number}/CIBMTR-2025/code/cristian/baselines/results/models/{name_version}"
    
    #target_5__sigmoid_False_False
    model_names = []
    path_folds = []
    
    hparams = {
        "embedding_dim": 32,#,16,
        "projection_dim": 224,#112,
        "hidden_dim": 56,#56,#56,
        "lr": 0.06464861983337984,
        "dropout": 0.05463240181423116,
        "cox_weight": 0,#0.05, #0.05,
        "race_weight":0.1, # 0.1 es mejor
        "pair_weight":1,
        "aux_weight": 0.26545778308743806,
        "margin": 0.2588153271003354,
        "weight_decay": 0.0002773544957610778,
        "patience": 20, #earlystopping
        "use_masked_pair":use_masked_pair,
        "mse_pair_joinloss":mse_pair_joinloss,
    
        "loss_name": loss_name,
        "act_function": act_function,
        "mse_pair_joinmodel":mse_pair_joinmodel,
        "target_log_scale":target_log_scale
    
    }
    
    for i in range(n_splits):
        model_names.append("best_model.ckpt")
        path_folds.append(f"{model_path}/{target_name}/fold_{i}/")
            
    DL.pipeline.global_seed(42)
    training_path,submission_path,cv_score,folds_scores =  DL.pipeline.inference(PATH_DATA,"./Results",test, train,n_splits=n_splits,target_name= "target_5",
                                                           model_name = name_version, 
                                                          path_folds = path_folds,
                                                             model_names=model_names,
                                                          hparams=hparams)

    results.append((training_path,submission_path,cv_score,folds_scores))

    df_train_pred = pd.read_csv(training_path)
    sns.histplot(df_train_pred["prediction"], kde=True, bins=50, color='skyblue')
    plt.title(f'DistribuciÃ³n de Predicciones Train')
    plt.xlabel('Probabilidad')
    plt.ylabel('Frecuencia')
    plt.show()



from src.utils.analysis import my_df_describe

numeric_desc, categorical_desc = my_df_describe(train,name = 'train dataset',show = False,path='',save=False)

numeric_columns = list(set([i for i in numeric_desc.index if "target_" not in i])-set(["efs","efs_time"]))
cat_columns = list(categorical_desc.index)


import numpy as np
from scipy.stats import rankdata

target_name = "target_5"

#weights = [2, 0.1, 10, 2, 0.1, 3, 1]#, 0.01, 0.01, 0.01, 0.01]
#weights = [1, 1, 5, 1, 1, 3, 1, 1, 1, 1, 1]

weights = [0.5, 0.5]

train_pred = np.array([0]*len(train),dtype=float)
sub_pred   = np.array([0]*len(test),dtype=float)
for i,(training_path,submission_path,cv_score,folds_scores) in enumerate(results):
    print("cv_score:",cv_score)
    #train_pred+= weights[i]*np.array(rankdata(pd.read_csv(training_path)["prediction"], method='average'))
    #sub_pred  += weights[i]*np.array(rankdata(pd.read_csv(submission_path)["prediction"], method='average'))
    train_pred+= weights[i]*np.array(pd.read_csv(training_path)["prediction"])
    sub_pred  += weights[i]*np.array(pd.read_csv(submission_path)["prediction"])


#train_pred = rankdata(train_pred/len(results), method='average')
#sub_pred   = rankdata(sub_pred/len(results), method='average')

df_train_pred = pd.DataFrame({"ID": train["ID"], "prediction": train_pred})
cv_score = score(train[['ID', 'efs', 'efs_time', 'race_group']].copy(),df_train_pred.copy(), "ID")
print("\nğŸš€ CV Ensemble Train Final: {:.5f}".format(cv_score))

test_df =  pd.DataFrame({"ID": test["ID"], "prediction": sub_pred})
test_df.to_csv("submission.csv", index=False)

sns.histplot(df_train_pred["prediction"], kde=True, bins=100, color='skyblue')
plt.title(f'DistribuciÃ³n de Predicciones Train')
plt.xlabel('Probabilidad')
plt.ylabel('Frecuencia')
plt.show()

test_df


import numpy as np
import pandas as pd
from scipy.stats import rankdata

train_pred_list = []
sub_pred_list   = []

for i,(training_path,submission_path,cv_score,folds_scores) in enumerate(results):
    
    train_pred_list.append(rankdata(np.array(pd.read_csv(training_path)["prediction"]), method='average'))
    sub_pred_list.append(rankdata(np.array(pd.read_csv(submission_path)["prediction"]), method='average'))

train_targets = pd.DataFrame(np.array(train_pred_list).T, columns = [f"target_{i}" for i in range(len(results))])
test_targets = pd.DataFrame(np.array(sub_pred_list).T, columns = [f"target_{i}" for i in range(len(results))])
train_targets


#train.columns




























