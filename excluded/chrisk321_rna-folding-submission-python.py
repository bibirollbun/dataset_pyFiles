import h2o
h2o.init()
from h2o.estimators import H2OXGBoostEstimator
import numpy as np 
import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error


# Reading in the competition material
train_sequences =  pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv') 
train_labels =  pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv') 
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv') 
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


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
    df = df.drop(['sequence', 'temporal_cutoff', 'description', 'all_sequences'],axis=1)

    return df

train_sequences_fe = feat_eng(train_sequences)
test_sequences_fe = feat_eng(test_sequences)


# Let's check the shape of our df
print(train_sequences_fe.shape)
print(test_sequences_fe.shape)


# create target_id to join with train_sequences
train_labels['target_id'] = train_labels['ID'].str.split('_',expand=True)[0] + '_' + train_labels['ID'].str.split('_',expand=True)[1]

# Combined features with train_labels
train_combined = train_labels.merge(train_sequences_fe,how='left',on='target_id')

# Handling missing values
print(f"There are {train_combined.isna().sum().sum()} missing values before imputation.")

# Impute missing values with the target_id, resname group average
train_combined['x_1'] = train_combined.groupby(['target_id','resname'])['x_1'].transform(lambda x: x.fillna(x.mean()))
train_combined['y_1'] = train_combined.groupby(['target_id','resname'])['y_1'].transform(lambda x: x.fillna(x.mean()))
train_combined['z_1'] = train_combined.groupby(['target_id','resname'])['z_1'].transform(lambda x: x.fillna(x.mean()))

# Handling missing values
print(f"There are {train_combined.isna().sum().sum()} missing values after imputation.")

# Some targets only have NA for x,y,z coordinates...we're going to remove those
train_combined = train_combined.dropna()

print(f"The final NA count is {train_combined.isna().sum().sum()}.")

# Drop target_id column...no longer needed
train_combined = train_combined.drop('target_id',axis=1)


def generate_test_label(tmp_id, tmp_seq):

    # Take a target_id & sequence, for each sequence, generate a ID resname resid format, append to df for all sequences
 
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

# Combined test features with test_labels
test_combined = test_labels.merge(test_sequences_fe,how='left',on='target_id')
# Drop target_id column...no longer needed
test_combined = test_combined.drop('target_id',axis=1)


features = train_combined.drop(['ID','x_1','y_1','z_1'],axis=1).columns.to_list()


### Predict X ###
train_hframe = h2o.H2OFrame(train_combined.drop(['ID','y_1','z_1'],axis=1)) # will need to separate out x_1,y_1,z_1 & ID for each iteration

# Split the dataset into a train and valid set:
train, valid = train_hframe.split_frame(ratios=[.8], seed=1)

xgb_x = H2OXGBoostEstimator(booster='gbtree', ntrees=10,max_depth=13,learn_rate=.3,tree_method='exact',grow_policy='depthwise',nfolds=5,keep_cross_validation_predictions = True,seed=1)
xgb_x.train(x = features, y = 'x_1', training_frame = train, validation_frame = valid)

# Model Performance
print(xgb_x.mse(valid=True))


### Predict Y ###
train_hframe = h2o.H2OFrame(train_combined.drop(['ID','x_1','z_1'],axis=1)) # will need to separate out x_1,y_1,z_1 & ID for each iteration

# Split the dataset into a train and valid set:
train, valid = train_hframe.split_frame(ratios=[.8], seed=1)

xgb_y = H2OXGBoostEstimator(booster='gbtree', ntrees=10,max_depth=13,learn_rate=.3,tree_method='exact',grow_policy='depthwise',nfolds=5,keep_cross_validation_predictions = True,seed=1)
xgb_y.train(x = features, y = 'y_1', training_frame = train, validation_frame = valid)

# Model Performance
print(xgb_y.mse(valid=True))


### Predict Z ###
train_hframe = h2o.H2OFrame(train_combined.drop(['ID','y_1','x_1'],axis=1)) # will need to separate out x_1,y_1,z_1 & ID for each iteration

# Split the dataset into a train and valid set:
train, valid = train_hframe.split_frame(ratios=[.8], seed=1)

xgb_z = H2OXGBoostEstimator(booster='gbtree', ntrees=10,max_depth=13,learn_rate=.3,tree_method='exact',grow_policy='depthwise',nfolds=5,keep_cross_validation_predictions = True,seed=1)
xgb_z.train(x = features, y = 'z_1', training_frame = train, validation_frame = valid)

# Model Performance
print(xgb_z.mse(valid=True))



test_hframe = h2o.H2OFrame(test_combined.drop(['ID'],axis=1)) 
preds_x = xgb_x.predict(test_hframe)
preds_y = xgb_y.predict(test_hframe)
preds_z = xgb_z.predict(test_hframe)


# Drop the target_id from test_labels as it is not in the submission format
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


# This is what our submission looks like
submission.tail(3)


submission.index


submission.to_csv('submission.csv',index=False)

