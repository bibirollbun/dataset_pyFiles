import numpy as np
import pandas as pd
from sklearn import *
from collections import Counter
from IPython.display import display, HTML
import xgboost as xgb
from xgboost import XGBRegressor
import warnings, pickle, tqdm
warnings.filterwarnings("ignore")
#import hdbscan

p = '/kaggle/input/stanford-rna-3d-folding/'

trains = pd.read_csv(p+'train_sequences.csv')
trainl = pd.read_csv(p+'train_labels.csv')

vals = pd.read_csv(p+'validation_sequences.csv')
vall = pd.read_csv(p+'validation_labels.csv')

tests = pd.read_csv(p+'test_sequences.csv')
subs = pd.read_csv(p+'sample_submission.csv')


print('Train Sequence Before: ', len(trains))
trains = trains[trains['temporal_cutoff']<'2022-05-27']
print('Train Sequence After: ', len(trains))

#Remove Nulls in XYZ, whole sequence
remove_ids = trainl[trainl.isna().any(axis=1)]['ID'].values
remove_ids = list(set(['_'.join(i.split('_')[:-1]) for i in remove_ids]))
print('Removed IDs: ', len(remove_ids))

trains = trains[~(trains['target_id'].isin(remove_ids))]
print('Train Sequence After: ', len(trains))

trains = trains.reset_index(drop=True)

cutoff_train_ids = []
for i in range(len(trains)):
    target = trains['target_id'][i]
    seq = [s for s in trains['sequence'][i]]
    for j, s in enumerate(seq):
        res = str(target) + '_' + str(j+1)
        cutoff_train_ids.append(res)

print('Train Labels Before: ', len(trainl))
trainl = trainl[trainl['ID'].isin(cutoff_train_ids)]
print('Train Labels After: ', len(trainl))


RNA_SEQ_TRANS = {}
from itertools import product
li = ['A', 'C', 'G', 'U']

for i in range(1,12):
    comb = product(li, repeat=i)
    RNA_SEQ_TRANS[i] = {''.join(l):i for i,l in enumerate(comb)}

for i in RNA_SEQ_TRANS:  #4**i
    print(i, len(RNA_SEQ_TRANS[i]))


#####| default_exp core #HAVE TO RESOLVE PATH ISSUES


#####| export #HAVE TO RESOLVE PATH ISSUES

import pandas.api.types
import pandas as pd
import os, re, shutil
import kagglehub

temp_path = '/kaggle/tempfile/'
if not os.path.exists(temp_path):
    os.makedirs(temp_path)
usalign_path = kagglehub.dataset_download('metric/usalign')
if not os.path.exists(temp_path + '/USalign'):
    shutil.copy(usalign_path + '/USalign', temp_path + 'USalign')

!chmod +rwx {temp_path}/USalign

class RNA3DFMetric:
    def __init__(self):
        pass
    
    def parse_tmscore_output(self, output):
        tm_score_match = re.findall(r'TM-score=\s+([\d.]+)', output)[1]
        if not tm_score_match:
            raise ValueError('No TM score found')
        return float(tm_score_match)

    def write_target_line(self, asr, an, rn, rnum, xc, yc, zc, atom_type='P') -> str:
        return f'ATOM{asr:>7d}  {an:<6s}{rn:<4s}{rnum:>3d}{xc:>12.3f}{yc:>8.3f}{zc:>8.3f}  1.00  0.00{atom_type:>12s}\n'
    
    def write2pdb(self, df: pd.DataFrame, xyz_id: str, target_path: str) -> int:
        resolved_cnt = 0
        with open(target_path, 'w') as target_file:
            for _, row in df.iterrows():
                x_coord = row[f'x_{xyz_id}']
                y_coord = row[f'y_{xyz_id}']
                z_coord = row[f'z_{xyz_id}']
                if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17:
                    resolved_cnt += 1
                    target_line = self.write_target_line(int(row['resid']), "C1'", row['resname'], int(row['resid']),x_coord,y_coord,z_coord,'C')
                    target_file.write(target_line)
        return resolved_cnt

    def score(self, solution: pd.DataFrame, submission: pd.DataFrame, verbose=False) -> float:
        solution['target_id'] = solution['ID'].apply(lambda x: x.split('_')[0])
        submission['target_id'] = submission['ID'].apply(lambda x: x.split('_')[0])
        results = []
        for target_id, group_native in solution.groupby('target_id'):
            group_predicted = submission[submission['target_id'] == target_id]
            native_pdb = 'native.pdb'
            predicted_pdb = 'predicted.pdb'
            target_id_scores = []
            for pred_cnt in range(1, 6):
                prediction_scores = []
                for native_cnt in range(1, 41):
                    resolved_cnt = self.write2pdb(group_native, native_cnt, native_pdb)
                    _ = self.write2pdb(group_predicted, pred_cnt, predicted_pdb)
                    if resolved_cnt > 0:
                        command = f'{temp_path}/USalign {predicted_pdb} {native_pdb} -atom " C1\'"'
                        usalign_output = os.popen(command).read()
                        if verbose: print(usalign_output)
                        prediction_scores.append(self.parse_tmscore_output(usalign_output))
                        print(prediction_scores[-1])
                target_id_scores.append(max(prediction_scores))
            results.append(max(target_id_scores))
        return float(sum(results) / len(results))


bases = {'A':{'id': 0, 'x':0, 'y':0, 'z':0},
         'C':{'id': 1, 'x':0, 'y':0, 'z':0},
         'G':{'id': 2, 'x':0, 'y':0, 'z':0}, 
         'U':{'id': 3, 'x':0, 'y':0, 'z':0}}

for b in ['A','C','G','U']:
    bases[b]['x'] =  trainl[((trainl['resid']==1) & (trainl['resname']==b))]['x_1'].mean()
    bases[b]['y'] =  trainl[((trainl['resid']==1) & (trainl['resname']==b))]['y_1'].mean()
    bases[b]['z'] =  trainl[((trainl['resid']==1) & (trainl['resname']==b))]['z_1'].mean()
bases


#ADDITIONAL FEATURE IDEAS:
#  Normalize starting points for training and validation data (Min/Max scale to 0 by IDs)
#  Test averaging the previous points from the different model outputs
#  Try blending outputs with the NN models
#  Add +/- Sequence Bases

def ngrams(seq, n):
    ngrams = zip(*[seq[i:] for i in range(n)])
    return {''.join(k):v for k,v in Counter(ngrams).items()}
    
def getBasePositionFeatures(df, dflabels, labels=True, models=[]):
    if labels:
        dflabels = {l: [x,y,z] for l, x, y, z in trainl[['ID', 'x_1', 'y_1', 'z_1']].values}
    df['seq_len'] = df['sequence'].map(len)
    ngramc = ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU', 'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU', 'AAA', 'AAC', 'AAG', 'AAU', 'ACA', 'ACC', 'ACG', 'ACU', 'AGA', 'AGC', 'AGG', 'AGU', 'AUA', 'AUC', 'AUG', 'AUU', 'CAA', 'CAC', 'CAG', 'CAU', 'CCA', 'CCC', 'CCG', 'CCU', 'CGA', 'CGC', 'CGG', 'CGU', 'CUA', 'CUC', 'CUG', 'CUU', 'GAA', 'GAC', 'GAG', 'GAU', 'GCA', 'GCC', 'GCG', 'GCU', 'GGA', 'GGC', 'GGG', 'GGU', 'GUA', 'GUC', 'GUG', 'GUU', 'UAA', 'UAC', 'UAG', 'UAU', 'UCA', 'UCC', 'UCG', 'UCU', 'UGA', 'UGC', 'UGG', 'UGU', 'UUA', 'UUC', 'UUG', 'UUU']
    cols = ['ID', 'resname', 'base', 'resid', 'seq_len', 'prior_a_count', 'prior_c_count', 'prior_g_count', 'prior_u_count', 'A', 'C', 'G', 'U'] + ngramc + ['prev_x', 'prev_y', 'prev_z', 'x_1', 'y_1', 'z_1']
    tcols = [c for c in cols if c not in ['ID', 'resname', 'x_1', 'y_1', 'z_1']]

    sub = []
    for i in range(len(df)):
        lstart = [0] * len(cols)
        lstart[cols.index('ID')] = str(df['target_id'][i])
        lstart[cols.index('seq_len')] = df['seq_len'][i]
        total_bases = dict(Counter(df['sequence'][i]))
        for k in total_bases:
            lstart[cols.index(k)] = total_bases[k]
        ngrams2 = ngrams(df['sequence'][i], 2)
        for k in ngrams2:
            lstart[cols.index(k)] = ngrams2[k]
        ngrams3 = ngrams(df['sequence'][i], 3)
        for k in ngrams3:
            lstart[cols.index(k)] = ngrams3[k]
        seq = [s for s in df['sequence'][i]]
        for j, s in enumerate(seq):
            l_item = lstart[:]
            l_item[cols.index('ID')] += '_' + str(j+1)
            ID_ = l_item[cols.index('ID')]
            l_item[cols.index('resname')] = s
            l_item[cols.index('base')] = bases[s]['id']
            l_item[cols.index('resid')] = j+1
            
            c = dict(Counter(df['sequence'][i][:j+1]))
            for k in c:
                l_item[cols.index('prior_' + str(k).lower() + '_count')] = c[k]

            #Last Labels
            if j+1 > 1:
                l_item[cols.index('prev_x')] = sub[-1][cols.index('x_1')]
                l_item[cols.index('prev_y')] = sub[-1][cols.index('y_1')]
                l_item[cols.index('prev_z')] = sub[-1][cols.index('z_1')]
                
            if labels: #Training
                l_item[cols.index('x_1')] = dflabels[ID_][0]
                l_item[cols.index('y_1')] = dflabels[ID_][1]
                l_item[cols.index('z_1')] = dflabels[ID_][2]
            else: #Prediction
                l_item = l_item[:-3] #remove x_1, y_1, z_1
                resp = [l_item[cols.index(k)] for k in tcols]
                for mid, model in enumerate(models):
                    dfp = pd.DataFrame([resp], columns=tcols)
                    if j+1 > 1:
                        dfp['prev_x'] = sub[-1][cols.index('x_1') + (mid*3)]
                        dfp['prev_y'] = sub[-1][cols.index('y_1') + (mid*3)]
                        dfp['prev_z'] = sub[-1][cols.index('z_1') + (mid*3)]
                    l_item +=  list(model.predict(dfp)[0])
            sub.append(l_item)
    if labels == False:
        cols += ['x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']
    sub = pd.DataFrame(sub, columns=cols)
    return sub


%%time
train = getBasePositionFeatures(trains, trainl)  #Extend to 5 sets for use with 5 models
train.head()


%%time

tcols = ['base', 'resid', 'seq_len', 'prior_a_count', 'prior_c_count', 'prior_g_count', 'prior_u_count', 'A', 'C', 'G', 'U', 'AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU', 'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU', 'AAA', 'AAC', 'AAG', 'AAU', 'ACA', 'ACC', 'ACG', 'ACU', 'AGA', 'AGC', 'AGG', 'AGU', 'AUA', 'AUC', 'AUG', 'AUU', 'CAA', 'CAC', 'CAG', 'CAU', 'CCA', 'CCC', 'CCG', 'CCU', 'CGA', 'CGC', 'CGG', 'CGU', 'CUA', 'CUC', 'CUG', 'CUU', 'GAA', 'GAC', 'GAG', 'GAU', 'GCA', 'GCC', 'GCG', 'GCU', 'GGA', 'GGC', 'GGG', 'GGU', 'GUA', 'GUC', 'GUG', 'GUU', 'UAA', 'UAC', 'UAG', 'UAU', 'UCA', 'UCC', 'UCG', 'UCU', 'UGA', 'UGC', 'UGG', 'UGU', 'UUA', 'UUC', 'UUG', 'UUU', 'prev_x', 'prev_y', 'prev_z']
pcols = ['x_1', 'y_1', 'z_1']

models = []
for i in tqdm.tqdm(range(5)): 
    model = XGBRegressor(n_estimators=7000, max_depth=6+i, learning_rate=0.2, tree_method='hist', n_jobs=-1, random_state=27)
    model.fit(train[tcols], train[pcols])
    models.append(model)
    with open('model'+str(i)+'.pkl', "wb") as f:
        pickle.dump(model, f)
#models = [models[0]] * 5


#models = []
#for i in range(5):
    #with open('model'+str(i)+'.pkl', "rb") as f:
    #    model = pickle.load(f)
    #medels.append(model)


%%time
val = getBasePositionFeatures(vals, vall, False, models)
val.head()


scols = subs.columns
sub = val[:]
sub = sub[scols]


%%time
RNA3DFM = RNA3DFMetric()
RNA3DFM.score(vall, sub, False)


%%time
test = getBasePositionFeatures(tests, tests, False, models)
test.head()


scols = subs.columns
sub = test[:]
sub = sub[scols]
sub.to_csv('submission.csv', index=False)


#https://zhanggroup.org/US-align/help/
#Reference Visualizations

