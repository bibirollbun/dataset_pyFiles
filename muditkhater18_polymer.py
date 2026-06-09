!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedGroupKFold,GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
os.environ["TOKENIZERS_PARALLELISM"] = "false"



traini = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
testi = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
targetsi = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


traini


testi


#remove null
traini.isnull().sum()


%%time

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

desc_names = [desc[0] for desc in Descriptors.descList]
descriptors = [compute_all_descriptors(smi) for smi in traini['SMILES'].to_list()]
descriptors = pd.DataFrame(descriptors, columns=desc_names)

traini = pd.concat([traini,descriptors],axis=1)


descriptors = [compute_all_descriptors(smi) for smi in testi['SMILES'].to_list()]
descriptors = pd.DataFrame(descriptors, columns=desc_names)
testi = pd.concat([testi,descriptors],axis=1)


traini


testi


traini.columns.values
testi.columns.values


%%time
def lgb_kfold(traini_df, testi_df, target, feats, folds):    
    params = {    
         'objective' : 'mae',#'binary', 
         'metric' : 'mae', 
         'num_leaves': 31,
         'min_data_in_leaf': 30,#30,
         'learning_rate': 0.01,
         'max_depth': -1,
         'max_bin': 256,
         'boosting': 'gbdt',
         'feature_fraction': 0.7,
         'bagging_freq': 1,
         'bagging_fraction': 0.7,
         'bagging_seed': 42,
         "lambda_l1":1,
         "lambda_l2":1,
         'verbosity': -1,        
         'num_boost_round' : 20000,
         'device_type' : 'cpu'        
    }      
    
    oof_preds = np.zeros(traini_df.shape[0])
    sub_preds = np.zeros(testi_df.shape[0])
    cv_list = []
    df_importances = pd.DataFrame()
    for n_fold, (traini_idx, valid_idx) in enumerate(folds.split(traini_df, traini_df[target])):     
        print ('n_fold:',n_fold)
        
        traini_x = traini_df[feats].iloc[traini_idx].values
        traini_y = traini_df[target].iloc[traini_idx].values
        
        valid_x = traini_df[feats].iloc[valid_idx].values
        valid_y = traini_df[target].iloc[valid_idx].values

        testi_x = testi_df[feats]
        
        print ('traini_x',traini_x.shape)
        print ('valid_x',valid_x.shape)    
        print ('testi_x',testi_x.shape)  
        
        dtraini = lgb.Dataset(traini_x, label=traini_y, )
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtraini, ) 
        callbacks = [
        lgb.log_evaluation(period=100,),
        lgb.early_stopping(200)    
        ]
        bst = lgb.train(params, dtraini,valid_sets=[dval,dtraini],callbacks=callbacks)
        feature_importances = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)#[:100]
        for f in feature_importances[:30]:
            print (f)       
            
        new_feats = []
        importances = []
        for f in feature_importances:
            new_feats.append(f[0])
            importances.append(f[1])
        df_importance = pd.DataFrame()
        df_importance['feature'] = new_feats
        df_importance['importance'] = importances
        df_importance['fold'] = n_fold
        
        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        
        sub_preds += bst.predict(testi_x, num_iteration=bst.best_iteration) / n_splits
        
        df_importances = pd.concat([df_importances,df_importance])    

        cv = mae(traini_df[target],  oof_preds)
      
        
    
    
    return oof_preds,sub_preds

n_splits = 10
seed = 817
folds = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
feats = ['MaxAbsEStateIndex', 'MaxEStateIndex',
       'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt',
       'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
       'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
       'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI',
       'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
       'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc',
       'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
       'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
       'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
       'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
       'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
       'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
       'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
       'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
       'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
       'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
       'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA',
       'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
       'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
       'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
       'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
       'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
       'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
       'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles',
       'NumAromaticHeterocycles', 'NumAromaticRings',
       'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors',
       'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles',
       'NumRotatableBonds', 'NumSaturatedCarbocycles',
       'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms',
       'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP',
       'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
       'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
       'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN',
       'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
       'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
       'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
       'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
       'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
       'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
       'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
       'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
       'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
       'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
       'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',
       'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
       'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
       'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
       'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
       'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
       'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
       'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']
for t in targetsi:
    print (t)
    if len(testi)<10:
        testi[t] = 0
    else:    
        traini_df = traini[traini[t].notnull()]
        oof_preds,sub_preds = lgb_kfold(traini_df, testi, t, feats, folds)
        testi[t] = sub_preds


testi


testi[['id','Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_csv('submission.csv',index=False)

