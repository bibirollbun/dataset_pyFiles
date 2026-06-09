import os
import numpy as np 
import kagglehub
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import argparse
import random
from autogluon.features.generators import PipelineFeatureGenerator, IdentityFeatureGenerator, BulkFeatureGenerator, AsTypeFeatureGenerator, DropUniqueFeatureGenerator, DropDuplicatesFeatureGenerator, FillNaFeatureGenerator 
from sklearn.metrics import roc_auc_score
from autogluon.core.metrics import make_scorer


# If run locally
input_autogluon_immrep25_path = kagglehub.dataset_download('samuelchagas/input-autogluon-immrep25')

# Kaggle notebook
for dirname, _, filenames in os.walk('/kaggle/input/input-autogluon-immrep25'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


read_data_rep = pd.read_csv('/kaggle/input/input-autogluon-immrep25/train_set_factors.csv', header='infer')
#Get the length of the representations (number of columns minus the class column)
n_cols = read_data_rep.shape[1]-1
#Name the dataframe columns 
col_names = [f'V{i}' for i in range(n_cols)] + ['Target']
read_data_rep.columns = col_names

#Split by peptide - Select random peptides to be used as unseen peptides in the test set 
read_data_raw = pd.read_csv('/kaggle/input/input-autogluon-immrep25/train_onlyPeps.csv',
                            names=['peptide'], header=None) #need to have a file with the peptide sequence
pep_seqs = sorted(list(set(read_data_raw.peptide)))

# filter out the peptides to exclude - peptide with >1000 occurences
exclude_peps = ["KLGGALQAK", "GILGFVFTL", "RAKFKQLL"]
pep_seqs = [pep for pep in pep_seqs if pep not in exclude_peps]

#select 100 random peptides - fixed order with random.seed() and sorted()
random.seed(42)
sample_pep_seqs = random.sample(pep_seqs, 100)
save_pep_seqs = pd.DataFrame({"peptides": sample_pep_seqs })

# Filter out the peptides to exclude

train_data_tmp = read_data_rep[~read_data_raw['peptide'].isin(sample_pep_seqs)]
train_data = train_data_tmp.sample(frac=1, random_state=25)
test_data = read_data_rep[read_data_raw['peptide'].isin(sample_pep_seqs)]

#pipeline for feature generation in AutoGluon
mypipeline = BulkFeatureGenerator(
    generators = [
        #[AsTypeFeatureGenerator()],
        [FillNaFeatureGenerator()],
        [IdentityFeatureGenerator()],
    ], verbosity=4
)
X_train_data = train_data.drop(columns=['Target']) #only when using the autogluon feature generator -> labels can not be there otherwise will be converted to features
Y_train_data = train_data['Target']
X_train_transformed = mypipeline.fit_transform(X=X_train_data) #transform data when using feature generator
X_test = test_data.drop(columns=['Target'])
y_test = test_data['Target']
print(X_train_transformed)

#setup and run autogluon predictions
label = 'Target'

#this is in case we need to define the models manually
hyperparameters = {
    'FASTAI': [
        {'ag_args_fit': {'num_gpus': 1}}
    ],
    'RF': [
        {'ag_args_fit': {'num_gpus': 1}}, {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}
    ],
    'NN_TORCH': [
        {'ag_args_fit': {'num_gpus': 1}}
    ],
    'KNN': [
        {'ag_args_fit': {'num_gpus': 1}}, {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}
    ],
    ##'CAT': [ #take too long
    ##    {}
    ##],
    'XT': [
        {'ag_args_fit': {'num_gpus': 1}}, {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}
    ],
    'XGB': [
        {'ag_args_fit': {'num_gpus': 0}} #without GPU
    ]
}

#defaut auc roc 0.1 
def roc_auc01(y_true, y_pred_prob):
    auc_scores = roc_auc_score(y_true, y_pred_prob, max_fpr=0.1)
    return auc_scores
roc_auc01_scorer = make_scorer(name='custom_roc_auc', score_func=roc_auc01, optimum=1,greater_is_better=True) #create scorer for autogluon

#train the predictor
print('Making predictions...')
predictor = TabularPredictor(label=label, path="/kaggle/working/", eval_metric=roc_auc01_scorer).fit(train_data, hyperparameters=hyperparameters,refit_full=False, excluded_model_types=['GBM'], feature_generator=mypipeline) 

leaderboard_df = predictor.leaderboard(test_data, extra_metrics=["roc_auc", "average_precision", roc_auc01_scorer], silent=False)

print(leaderboard_df)
# Save the models metrics to CSV
#leaderboard_df = leaderboard_df.round(4)
#leaderboard_df.to_csv(f"autogluon_leaderboard.csv", index=False)

#print("Leaderboard saved to autogluon_leaderboard.csv")

#test the predictor in the test set
y_test = test_data[label] #classes of the test set
y_pred = predictor.predict(test_data.drop(columns=[label])) #make predictions on the test set
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True, detailed_report=True) #check the performance
print(perf)


def roc_auc01(y_true, y_pred_prob):
    auc_scores = roc_auc_score(y_true, y_pred_prob, max_fpr=0.1)
    return auc_scores
roc_auc01_scorer = make_scorer(name='custom_roc_auc', score_func=roc_auc01, optimum=1,greater_is_better=True) #create scorer for autogluon

label = 'Target'
#read data for prediction
read_data_rep = pd.read_csv(f'/kaggle/input/input-autogluon-immrep25/test_set_factors.csv')
#Get the length of the representations
n_cols = read_data_rep.shape[1]
#Name the dataframe columns 
col_names = [f'V{i}' for i in range(n_cols)]
read_data_rep.columns = col_names

#set the path that stores the trained model
#predictor = TabularPredictor.load(f"/kaggle/working/model_autogluon", verbosity=4)

y_pred_prob = predictor.predict_proba(read_data_rep) #make predictions on the test set
#y_pred = predictor.predict(read_data_rep.drop(columns=[label])) #make predictions on the test set
#print(y_pred)
df_out_prob = y_pred_prob.iloc[:,1]

# data from kaggle
kaggle_sample_submission = pd.read_csv('/kaggle/input/input-autogluon-immrep25/sample_submission.csv')
kaggle_sample_submission.iloc[:, -1] = df_out_prob
kaggle_sample_submission.to_csv('/kaggle/working/submission_autogluon03.csv', index=False)




