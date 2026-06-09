import sys
sys.path.append("/kaggle/input/your-biopython-dataset")

!pip install /kaggle/input/biopython/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import h2o
h2o.init()
from h2o.estimators import H2OXGBoostEstimator
import numpy as np 
import pandas as pd 
import xgboost as xgb
from scipy.stats import entropy
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error
from collections import Counter
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import molecular_weight, MeltingTemp as mt


train_sequences =  pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv') 
train_labels =  pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv') 
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv') 
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


def seq_entropy(seq):
    freqs = np.array([seq.count(base)/len(seq) for base in "ACGU"])
    return entropy(freqs)


def get_kmers(sequence, k=3):
    return Counter([sequence[i:i+k] for i in range(len(sequence)-k+1)])


def has_palindrome(seq, k=4):
    for i in range(len(seq) - k + 1):
        sub = seq[i:i+k]
        if sub == sub[::-1].translate(str.maketrans("AUCG", "UAGC")):
            return 1
    return 0


def feat_eng(df):
    df['seq_length'] = df['sequence'].str.len()
    df['a_cnt'] = df['sequence'].str.count("A")
    df['c_cnt'] = df['sequence'].str.count("C")
    df['u_cnt'] = df['sequence'].str.count("U")
    df['g_cnt'] = df['sequence'].str.count("G")
    df['ac_cnt'] = df['sequence'].str.count("AC")
    df['au_cnt'] = df['sequence'].str.count("AU")
    df['ag_cnt'] = df['sequence'].str.count("AG")
    df['ca_cnt'] = df['sequence'].str.count("CA")
    df['cu_cnt'] = df['sequence'].str.count("CU")
    df['cg_cnt'] = df['sequence'].str.count("CG")
    df['ua_cnt'] = df['sequence'].str.count("UA")
    df['uc_cnt'] = df['sequence'].str.count("UC")
    df['ug_cnt'] = df['sequence'].str.count("UG")
    df['ga_cnt'] = df['sequence'].str.count("GA")
    df['gc_cnt'] = df['sequence'].str.count("GC")
    df['gu_cnt'] = df['sequence'].str.count("GU")
    df['aa_cnt'] = df['sequence'].str.count("AA")
    df['cc_cnt'] = df['sequence'].str.count("CC")
    df['uu_cnt'] = df['sequence'].str.count("UU")
    df['gg_cnt'] = df['sequence'].str.count("GG")
    
    df['begin_sequence'] = df['sequence'].str[0]
    df['end_sequence'] = df['sequence'].str[-1]

    df['gc_ratio'] = (df['g_cnt'] + df['c_cnt']) / df['seq_length']
    df['au_ratio'] = (df['a_cnt'] + df['u_cnt']) / df['seq_length']

    df['tm'] = df['sequence'].apply(lambda x: mt.Tm_NN(x, nn_table=mt.RNA_NN1))
    
    df['entropy'] = df['sequence'].apply(seq_entropy)

    df['gc_fraction'] = df['sequence'].apply(gc_fraction)

    df['3mer_freqs'] = df['sequence'].apply(lambda x: get_kmers(x, 3))
    
    df['palindrome'] = df['sequence'].apply(lambda x: has_palindrome(x, k=6))
    
    df = df.drop(['sequence', 'temporal_cutoff', 'description', 'all_sequences'],axis=1)

    return df


train_sequences_fe = feat_eng(train_sequences)
test_sequences_fe = feat_eng(test_sequences)


print(train_sequences_fe.columns)


train_labels['target_id'] = train_labels['ID'].str.split('_',expand=True)[0] + '_' + train_labels['ID'].str.split('_',expand=True)[1]
train_combined = train_labels.merge(train_sequences_fe,how='left',on='target_id')

print(train_labels['target_id'].head())
print(train_sequences_fe['target_id'].head())

train_combined['x_1'] = train_combined.groupby(['target_id','resname'])['x_1'].transform(lambda x: x.fillna(x.mean()))
train_combined['y_1'] = train_combined.groupby(['target_id','resname'])['y_1'].transform(lambda x: x.fillna(x.mean()))
train_combined['z_1'] = train_combined.groupby(['target_id','resname'])['z_1'].transform(lambda x: x.fillna(x.mean()))

print("Before dropna:", train_combined.shape)
print(train_combined.isna().sum())

train_combined = train_combined.dropna()
train_combined = train_combined.drop('target_id',axis=1)


def generate_test_label(tmp_id, tmp_seq):

    tmp_seq_len = len(tmp_seq)
    tmp_df = pd.DataFrame()
    
    tmp_df['resname'] = [x for x in tmp_seq]
    tmp_df.insert(0, 'ID', tmp_id) 
    tmp_df['target_id'] = tmp_df['ID']
    tmp_df['resid'] = list(range(1,tmp_seq_len+1))
    tmp_df['ID'] = [x+"_" + str(y) for x,y in zip(tmp_df['ID'],tmp_df['resid'])]
    return tmp_df
    
test_labels = pd.DataFrame()
for x,y in zip(test_sequences['target_id'],test_sequences['sequence']):
    test_labels =  pd.concat([test_labels, generate_test_label(x,y)])

test_combined = test_labels.merge(test_sequences_fe,how='left',on='target_id')
test_combined = test_combined.drop('target_id',axis=1)


features = train_combined.drop(['ID','x_1','y_1','z_1'],axis=1).columns.to_list()

# ----------- X_1 -----------
train_hframe = h2o.H2OFrame(train_combined.drop(['ID', 'y_1', 'z_1'], axis=1))
train, valid = train_hframe.split_frame(ratios=[.8], seed=1)

xgb_x = H2OXGBoostEstimator(
    booster='gbtree',
    ntrees=500,
    max_depth=20,
    learn_rate=0.01,
    sample_rate=0.8,
    col_sample_rate=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    nfolds=5,
    keep_cross_validation_predictions=True,
    seed=1,
    score_tree_interval=10,
    stopping_rounds=20,
    stopping_metric="RMSE",
    stopping_tolerance=1e-4
)
xgb_x.train(x=features, y='x_1', training_frame=train, validation_frame=valid)

# ----------- Y_1 -----------
train_hframe = h2o.H2OFrame(train_combined.drop(['ID', 'x_1', 'z_1'], axis=1))
train, valid = train_hframe.split_frame(ratios=[.8], seed=1)

xgb_y = H2OXGBoostEstimator(
    booster='gbtree',
    ntrees=500,
    max_depth=20,
    learn_rate=0.01,
    sample_rate=0.8,
    col_sample_rate=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    nfolds=5,
    keep_cross_validation_predictions=True,
    seed=1,
    score_tree_interval=10,
    stopping_rounds=20,
    stopping_metric="RMSE",
    stopping_tolerance=1e-4
)
xgb_y.train(x=features, y='y_1', training_frame=train, validation_frame=valid)

# ----------- Z_1 -----------
train_hframe = h2o.H2OFrame(train_combined.drop(['ID', 'x_1', 'y_1'], axis=1))
train, valid = train_hframe.split_frame(ratios=[.8], seed=1)

xgb_z = H2OXGBoostEstimator(
    booster='gbtree',
    ntrees=500,
    max_depth=20,
    learn_rate=0.01,
    sample_rate=0.8,
    col_sample_rate=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    nfolds=5,
    keep_cross_validation_predictions=True,
    seed=1,
    score_tree_interval=10,
    stopping_rounds=20,
    stopping_metric="RMSE",
    stopping_tolerance=1e-4
)
xgb_z.train(x=features, y='z_1', training_frame=train, validation_frame=valid)


print("x_1 Metrics:")
print(f"  MSE: {xgb_x.mse(valid=True)}")
print(f"  RMSE: {xgb_x.rmse(valid=True)}")
print(f"  MAE: {xgb_x.mae(valid=True)}")
print(f"  R2: {xgb_x.r2(valid=True)}")

print("y_1 Metrics:")
print(f"  MSE: {xgb_y.mse(valid=True)}")
print(f"  RMSE: {xgb_y.rmse(valid=True)}")
print(f"  MAE: {xgb_y.mae(valid=True)}")
print(f"  R2: {xgb_y.r2(valid=True)}")

print("z_1 Metrics:")
print(f"  MSE: {xgb_z.mse(valid=True)}")
print(f"  RMSE: {xgb_z.rmse(valid=True)}")
print(f"  MAE: {xgb_z.mae(valid=True)}")
print(f"  R2: {xgb_z.r2(valid=True)}")


test_hframe = h2o.H2OFrame(test_combined.drop(['ID'],axis=1)) 
preds_x = xgb_x.predict(test_hframe)
preds_y = xgb_y.predict(test_hframe)
preds_z = xgb_z.predict(test_hframe)


submission = test_labels.drop('target_id',axis=1)

submission['x_1'] =  preds_x.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['y_1'] =preds_y.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['z_1'] = preds_z.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)

submission['x_2'] = preds_x.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['y_2'] =preds_y.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['z_2'] = preds_z.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)

submission['x_3'] = preds_x.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['y_3'] =preds_y.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['z_3'] = preds_z.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)

submission['x_4'] = preds_x.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['y_4'] =preds_y.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['z_4'] = preds_z.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)

submission['x_5'] = preds_x.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['y_5'] =preds_y.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)
submission['z_5'] = preds_z.as_data_frame(use_pandas=True, header=True,use_multi_thread=True)


submission.to_csv('submission.csv',index=False)
submission.head()

