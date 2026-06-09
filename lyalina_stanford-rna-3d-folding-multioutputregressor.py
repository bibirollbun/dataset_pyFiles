# packages

# standard
import numpy as np
import pandas as pd
import time

import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import plotly.graph_objects as go

# warning handling
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  
warnings.filterwarnings('ignore', category=RuntimeWarning)

# configs
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 150)

default_color_1 = 'darkblue'
set_plot = False



from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
clr = Style.RESET_ALL

# load data
df_train = pd.read_csv('../input/stanford-rna-3d-folding/train_labels.csv')
df_valid = pd.read_csv('../input/stanford-rna-3d-folding/validation_labels.csv')
df_train_seq = pd.read_csv('../input/stanford-rna-3d-folding/train_sequences.csv')
df_valid_seq = pd.read_csv('../input/stanford-rna-3d-folding/validation_sequences.csv')
df_test = pd.read_csv('../input/stanford-rna-3d-folding/test_sequences.csv')


# ------------------
# Train data
# ------------------

#df_train_seq["all_sequences_str"] = df_train_seq["all_sequences"].astype(str)
df_train_seq['sequence_id'] = df_train_seq["target_id"]
df_train_seq["sequence_num"] = df_train_seq["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[0:1])[1:])
df_train_seq["sequence_group"] = df_train_seq["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("_")[0:1])[1:])
df_train_seq["sequence_group_num"] = df_train_seq["sequence_num"].apply(lambda x: "_".join(x.split("_")[1:2]))

# for nan in all_sequences handle  sequence_group    df_train_seq[df_train_seq["all_sequences"].isna()]

df_train_seq.loc[df_train_seq.target_id == '2ZJQ_Y', 'sequence_num'] = '2ZJQ_1'
df_train_seq.loc[df_train_seq.target_id == '2ZJQ_X', 'sequence_num'] = '2ZJQ_1'
df_train_seq.loc[df_train_seq.target_id == '4V65_A1', 'sequence_num'] = '4V65_1'
df_train_seq.loc[df_train_seq.target_id == '4V65_BB', 'sequence_num'] = '4V65_1'
df_train_seq.loc[df_train_seq.target_id == '4V5F_CA', 'sequence_num'] = '4V5F_1'

df_train_seq.loc[df_train_seq.target_id == '2ZJQ_Y', 'sequence_group'] = '2ZJQ_Y'
df_train_seq.loc[df_train_seq.target_id == '2ZJQ_X', 'sequence_group'] = '2ZJQ_X'
df_train_seq.loc[df_train_seq.target_id == '4V65_A1', 'sequence_group'] = '4V65_A1'
df_train_seq.loc[df_train_seq.target_id == '4V65_BB', 'sequence_group'] = '4V65_BB'
df_train_seq.loc[df_train_seq.target_id == '4V5F_CA', 'sequence_group'] = '4V5F_CA'

df_train_seq.loc[df_train_seq.target_id == '2ZJQ_Y', 'sequence_group_num'] = 1
df_train_seq.loc[df_train_seq.target_id == '2ZJQ_X', 'sequence_group_num'] = 1
df_train_seq.loc[df_train_seq.target_id == '4V65_A1', 'sequence_group_num'] = 1
df_train_seq.loc[df_train_seq.target_id == '4V65_BB', 'sequence_group_num'] = 1
df_train_seq.loc[df_train_seq.target_id == '4V5F_CA', 'sequence_group_num'] = 1

df_train = df_train.fillna(0)

df_train["sequence_id"] = df_train["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
df_train["sequence_target"] = df_train["ID"].apply(lambda x: "".join(x.split("_")[0]))
df_train["sequence_class"] = df_train["ID"].apply(lambda x: "".join(x.split("_")[1])) #"_".join(    #df_train["ID"][1].split("_")[0:2]


train = pd.merge(df_train, df_train_seq[['sequence','temporal_cutoff',	'description',	'all_sequences',	'sequence_id',	'sequence_num',	'sequence_group',	'sequence_group_num']], on="sequence_id")

# ------------------
# Valid data
# ------------------
# replace extreme values by NaN
df_valid.replace(to_replace=-1E18, value=np.nan, inplace=True);

df_valid_seq['sequence_id'] = df_valid_seq["target_id"]
df_valid_seq["sequence_num"] = df_valid_seq["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[0:1])[1:])
df_valid_seq["sequence_group"] = df_valid_seq["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("_")[0:1])[1:])
df_valid_seq["sequence_group_num"] = df_valid_seq["sequence_num"].apply(lambda x: "_".join(x.split("_")[1:2]))

df_valid = df_valid.fillna(0)

df_valid["sequence_id"] = df_valid["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
df_valid["sequence_target"] = df_valid["ID"].apply(lambda x: "".join(x.split("_")[0]))
df_valid["sequence_class"] = df_valid["ID"].apply(lambda x: "".join(x.split("_")[1])) #"_".join(    #df_train["ID"][1].split("_")[0:2]

valid = pd.merge(df_valid[['ID',	'resname',	'resid',	'x_1',	'y_1',	'z_1', 'sequence_id',	'sequence_target',	'sequence_class']], \
                 df_valid_seq[['sequence','temporal_cutoff',	'description',	'all_sequences',	'sequence_id',	'sequence_num',	'sequence_group',	'sequence_group_num']], on="sequence_id")

# ------------------
# Test data
# ------------------

df_test['sequence_id'] = df_test["target_id"]
df_test["sequence_num"] = df_test["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[0:1])[1:])
df_test["sequence_group"] = df_test["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("_")[0:1])[1:])
df_test["sequence_group_num"] = df_test["sequence_num"].apply(lambda x: "".join(x.split("_")[1:2]))


df_test_seq = df_test.copy()

df_test_seq['resname'] = df_test_seq['sequence'].apply(list)  # Преобразуем каждую строку в список букв
df_test_seq['target_id'+'sequence'] = df_test_seq['target_id']+df_test_seq['sequence']

# create new dataset for every resid by row
test = df_test_seq.explode('resname')  
test['resid'] = test.groupby('target_idsequence').cumcount() + 1
test['x_1'] = 0
test['y_1'] = 0
test['z_1'] = 0

test["sequence_target"] = test["target_id"].apply(lambda x: "".join(x.split("_")[0]))
test["sequence_class"] = 1
test["ID"] = test["target_id"]+"_"+test["resid"].astype(str)

test = test[['ID','resname'	,'resid',	'x_1',	'y_1',	'z_1',	'sequence_id',	'sequence_target',	'sequence_class',	'sequence',	'temporal_cutoff',	'description'	,'all_sequences', 'sequence_num'	,'sequence_group'	,'sequence_group_num']]

del df_train ,df_valid ,df_train_seq ,df_valid_seq ,df_test 

target_col = ['x_1', 'y_1', 'z_1']
train[target_col] = train[target_col].fillna(train[target_col].mean())

display(train[1:2])
display(valid[1:2])
display(test[1:2])


from sklearn.base import BaseEstimator, TransformerMixin
class AggFeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, group_col, agg_col, agg_func):
        self.group_col = group_col
        self.group_col_name = ''
        for col in group_col:
            self.group_col_name += col
        self.agg_col = agg_col
        self.agg_func = agg_func
        self.agg_df = None
        self.medians = None
        
    def fit(self, X, y=None):
        group_col = self.group_col
        agg_col = self.agg_col
        agg_func = self.agg_func
        
        self.agg_df = X.groupby(group_col)[agg_col].agg(agg_func)
        self.agg_df.columns = [f'{self.group_col_name}_{agg}_{_agg_col}' for _agg_col in agg_col for agg in agg_func]
        self.medians = X[agg_col].median()
        
        return self
    
    def transform(self, X):
        group_col = self.group_col
        agg_col = self.agg_col
        agg_func = self.agg_func
        agg_df = self.agg_df
        medians = self.medians
        
        X_merged = pd.merge(X, agg_df, left_on=group_col, right_index=True, how='left')
        X_merged.fillna(medians, inplace=True)
        X_agg = X_merged.loc[:, [f'{self.group_col_name}_{agg}_{_agg_col}' for _agg_col in agg_col for agg in agg_func]]
        
        return X_agg
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_agg = self.transform(X)
        return X_agg


def find_atoms_in_region(x, y, z, x_target, y_target, z_target, tolerance=100):
    """
    Filter all RNA atoms that are within a given tolerance range around a target coordinate.
    """
    filtered_atoms =  (min(x_target - tolerance, x_target + tolerance) < x < max(x_target - tolerance, x_target + tolerance)) \
        & (min(y_target - tolerance, y_target + tolerance) < y < max(y_target - tolerance, y_target + tolerance)) \
        & (min(z_target - tolerance, z_target + tolerance) < z < max(z_target - tolerance, z_target + tolerance))

    return filtered_atoms
    
def euclidean_distance(x1,y1,z1,x2,y2,z2):
    return np.sqrt((x2 - x1)**2 +
                   (y2 - y1)**2 +
                   (z2 - z1)**2)

def count_nucleotides_map(sequence):
    return {
        'A': sequence.count('A'),
        'C': sequence.count('C'),
        'G': sequence.count('G'),
        'U': sequence.count('U')
    }
def count_nucleotides(sequence):
      return    f"A{sequence.count('A')}C{sequence.count('C')}G{sequence.count('G')}U{sequence.count('U')}"

def get_nucleotide_combinations(seq, start_pos, length):
        if start_pos + length <= len(seq):
            result = seq[start_pos:start_pos + length]
        else:
            result = '-'  
        return result
    
def FE (df):
    df['temporal_cutoff'] = pd.to_datetime(df['temporal_cutoff']).astype('int64') // 10**9
    df["res"] = df["resname"] + df["resid"].astype(str)
                                    
    df["length"] = df["sequence"].str.len()
    
    df['A_cnt'] = df['sequence'].astype(str).str.count("A")
    df['C_cnt'] = df['sequence'].astype(str).str.count("C")
    df['U_cnt'] = df['sequence'].astype(str).str.count("U")
    df['G_cnt'] = df['sequence'].astype(str).str.count("G")
    df['AC_cnt'] = df['sequence'].astype(str).str.count("AC")
    df['AU_cnt'] = df['sequence'].astype(str).str.count("AU")
    df['AG_cnt'] = df['sequence'].astype(str).str.count("AG")
    df['CA_cnt'] = df['sequence'].astype(str).str.count("CA")
    df['CU_cnt'] = df['sequence'].astype(str).str.count("CU")
    df['CG_cnt'] = df['sequence'].astype(str).str.count("CG")
    df['UA_cnt'] = df['sequence'].astype(str).str.count("UA")
    df['UC_cnt'] = df['sequence'].astype(str).str.count("UC")
    df['UG_cnt'] = df['sequence'].astype(str).str.count("UG")
    df['GA_cnt'] = df['sequence'].astype(str).str.count("GA")
    df['GC_cnt'] = df['sequence'].astype(str).str.count("GC")
    df['GU_cnt'] = df['sequence'].astype(str).str.count("GU")
    df['AA_cnt'] = df['sequence'].astype(str).str.count("AA")
    df['CC_cnt'] = df['sequence'].astype(str).str.count("CC")
    df['UU_cnt'] = df['sequence'].astype(str).str.count("UU")
    df['GG_cnt'] = df['sequence'].astype(str).str.count("GG")
    
    df['begin_seq'] = df['sequence'].astype(str).str[0]
    df['end_seq'] = df['sequence'].astype(str).str[-1]

    df['nucleotide_ngram2'] = df.apply(lambda row: get_nucleotide_combinations( row['sequence'], row['resid']-1, 2),   axis=1)
    df['nucleotide_ngram3'] = df.apply(lambda row: get_nucleotide_combinations( row['sequence'], row['resid']-1, 3),   axis=1)
    df['nucleotide_ngram4'] = df.apply(lambda row: get_nucleotide_combinations( row['sequence'], row['resid']-1, 4),   axis=1)
    df['nucleotide_ngram2_prev'] = df['nucleotide_ngram2'].shift(1)
    df['nucleotide_ngram2_prev'].replace(to_replace=np.nan, value="<", inplace=True)
  
    #df['nucleotide_counts'] = df['sequence'].apply(count_nucleotides)
    df['counts_a'] = df['sequence'].apply(lambda x: x.count('A'))
    df['counts_c'] = df['sequence'].apply(lambda x: x.count('C'))
    df['counts_g'] = df['sequence'].apply(lambda x: x.count('G'))
    df['counts_u'] = df['sequence'].apply(lambda x: x.count('U'))

    df["GC_content"] = df["sequence"].apply(
        lambda seq: (seq.count("G") + seq.count("C")) / len(seq) )
    df["GA_content"] = df["sequence"].apply(
        lambda seq: (seq.count("G") + seq.count("A")) / len(seq) )
    df["GU_content"] = df["sequence"].apply(
        lambda seq: (seq.count("G") + seq.count("U")) / len(seq) )
    df["CA_content"] = df["sequence"].apply(
        lambda seq: (seq.count("C") + seq.count("A")) / len(seq) )
    df["CU_content"] = df["sequence"].apply(
        lambda seq: (seq.count("C") + seq.count("U")) / len(seq) )
    df["AU_content"] = df["sequence"].apply(
        lambda seq: (seq.count("A") + seq.count("U")) / len(seq) )

    df['resname_prev'] = df['resname'].shift(1)
    df['resname_next'] = df['resname'].shift(-1)
    df['resname_prev'].replace(to_replace=np.nan, value="<", inplace=True)
    df['resname_next'].replace(to_replace=np.nan, value=">", inplace=True)

    df['prev_x'] = df['x_1'].shift(1)
    df['prev_y'] = df['y_1'].shift(1)
    df['prev_z'] = df['z_1'].shift(1)
    df['next_x'] = df['x_1'].shift(-1)
    df['next_y'] = df['y_1'].shift(-1)
    df['next_z'] = df['z_1'].shift(-1)

    df["distance_from_origin"] = np.sqrt(df["x_1"]**2 + df["y_1"]**2 + df["z_1"]**2 )

    df['distance_prev'] = df.apply(lambda row: euclidean_distance( row["x_1"], row["y_1"], row["x_1"],row['prev_x'], row['prev_y'], row['prev_z']), axis=1)
    df['distance_next'] = df.apply(lambda row: euclidean_distance( row["x_1"], row["y_1"], row["x_1"],row['next_x'], row['next_y'], row['next_z']), axis=1)

    df["chain"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[1:2])[6:].replace('auth', '').replace(' ', '').strip())
    df["RNA"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[2:3])[0:])  
    df["RNA_chain"] = df["RNA"].astype(str).apply(lambda x: "_".join(x.split("(5'-R(")[1:])[0:].replace(")-3')","").replace(") -3')",""))  
    df["add_chain"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[3:][4:5] )[6:].replace('auth', '').replace(' ', '').strip())
    df["add_RNA"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[3:][5:6] ))
    df["type_RNA"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[3:][6:7] ))  
    df["class"] = df["type_RNA"].astype(str).apply(lambda x: "_".join(x.split("\n")[0:1] ))
    df["class_chain"] = df["type_RNA"].astype(str).apply(lambda x: "_".join(x.split("\n")[1:] ))
    
    df = df.fillna(0)


# ------------------
# Prepare dataset for model 
# ------------------
print(f"Prepare datasets for model ")

train['set'] = 'train'
valid['set'] = 'valid'
test['set'] = 'test'

X_combine = pd.concat([train,valid])
X_combine = pd.concat([X_combine,test])

# ------------------
# Create Features
# ------------------

FE(X_combine)


FEATURES = list(set(X_combine.columns.tolist()) - set(train.columns.tolist()))
print(f"create :{len(X_combine.columns.tolist()) - len(train.columns.tolist())} features,\n ", f"for columns :{FEATURES} \n ")

print(f"train + valid + test shape :{blu}{X_combine.shape}{clr}, ")
print(f'{"-" * 100}')
# ------------------
# Aggregate Features
# ------------------
#group_cols = [
#        ['sequence_group'],['resname'],['res'],['nucleotide_ngram2'],	['nucleotide_ngram3'],	['nucleotide_ngram4'], ["chain"]	  #['res','nres']
#    ]
#agg_col = [
#        'x_1',
#        'y_1', 
#        'z_1'
#    ]
group_cols = [   #resname_prev	resname_next
        ['sequence_group'],['resname'],['res'],['nucleotide_ngram2'],['nucleotide_ngram2_prev'],	["chain",'res']
    ]
agg_col = [
        'x_1','y_1', 'z_1',
        #'prev_x','prev_y','prev_z','next_x','next_y','next_z','distance_prev','distance_next',
        'distance_from_origin',
    ]


def agg_train_df(df, group_cols,agg_col ):
    
    agg_train = []
    print(f"agregate feature \n group by :{group_cols},\n ", f"for columns :{agg_col} \n ")
    for group_col in group_cols:
            agg_extractor = AggFeatureExtractor(group_col=group_col, agg_col=agg_col, agg_func=['mean',  'min', 'max', 'std'])
            agg_extractor.fit(df)
            agg_train.append(agg_extractor.transform(X_combine))
    df = pd.concat([df] + agg_train, axis=1).fillna(0)
    return df

X_combine = agg_train_df(X_combine,group_cols,agg_col)

group_cols = [   #resname_prev	resname_next
    	['nucleotide_ngram3'],	['nucleotide_ngram4'], 
        ['resname_prev','res',	'resname_next']	  ,['sequence_group','res'],
    ]
agg_col = [
        'x_1','y_1', 'z_1',
        #'prev_x','prev_y','prev_z','next_x','next_y','next_z',
        'distance_prev','distance_next','distance_from_origin',
    ]

X_combine = agg_train_df(X_combine,group_cols,agg_col)

X_combine['x_window'] = X_combine.apply(lambda row: (row['sequence_group_max_x_1'] - row['sequence_group_min_x_1']),   axis=1)
X_combine['y_window'] = X_combine.apply(lambda row: (row['sequence_group_max_y_1'] - row['sequence_group_min_y_1']),   axis=1)
X_combine['z_window'] = X_combine.apply(lambda row: (row['sequence_group_max_z_1'] - row['sequence_group_min_z_1']),   axis=1)

tolerance = 50
X_combine[f'atom_round{tolerance}'] = X_combine.apply(lambda row: find_atoms_in_region( row["x_1"], row["y_1"],row["z_1"],row['sequence_group_max_x_1'] - row['sequence_group_min_x_1'],row['sequence_group_max_y_1'] - row['sequence_group_min_y_1'],row['sequence_group_max_z_1'] - row['sequence_group_min_z_1'], tolerance),   axis=1)
tolerance = 100
X_combine[f'atom_round{tolerance}'] = X_combine.apply(lambda row: find_atoms_in_region( row["x_1"], row["y_1"],row["z_1"],row['sequence_group_max_x_1'] - row['sequence_group_min_x_1'],row['sequence_group_max_y_1'] - row['sequence_group_min_y_1'],row['sequence_group_max_z_1'] - row['sequence_group_min_z_1'], tolerance),   axis=1)
tolerance = 150
X_combine[f'atom_round{tolerance}'] = X_combine.apply(lambda row: find_atoms_in_region( row["x_1"], row["y_1"],row["z_1"],row['sequence_group_max_x_1'] - row['sequence_group_min_x_1'],row['sequence_group_max_y_1'] - row['sequence_group_min_y_1'],row['sequence_group_max_z_1'] - row['sequence_group_min_z_1'], tolerance),   axis=1)
tolerance = 300
X_combine[f'atom_round{tolerance}'] = X_combine.apply(lambda row: find_atoms_in_region( row["x_1"], row["y_1"],row["z_1"],row['sequence_group_max_x_1'] - row['sequence_group_min_x_1'],row['sequence_group_max_y_1'] - row['sequence_group_min_y_1'],row['sequence_group_max_z_1'] - row['sequence_group_min_z_1'], tolerance),   axis=1)


one_hot_encoded = pd.get_dummies(X_combine['resname'], prefix='nucleotide')

X_combine = pd.concat([X_combine, one_hot_encoded], axis=1)

print(f"train + valid + test shape :{blu}{X_combine.shape}{clr}, ")
print(f'{"-" * 100}')

X_combine.head(3)




### ------------------
# Create Features with bag word for Descripction and RNA columns
# ------------------
count_vectorizer = CountVectorizer(
    analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=None)    
bag_of_words_combine = count_vectorizer.fit_transform(X_combine['description'])
bag_of_words_combine1 = count_vectorizer.fit_transform(X_combine['RNA'])

svd = TruncatedSVD(n_components=20, n_iter=30, random_state=12)
truncated_bag_of_words_combine = svd.fit_transform(bag_of_words_combine)
svd1 = TruncatedSVD(n_components=5, n_iter=25, random_state=12)
truncated_bag_of_words_combine1 = svd1.fit_transform(bag_of_words_combine1)

add_col = [f'desc_{i}' for i in svd.get_feature_names_out()]
add_col1 = [f'rna_{i}' for i in svd1.get_feature_names_out()]

bag_df = pd.DataFrame(truncated_bag_of_words_combine, columns = add_col )
bag_df1 = pd.DataFrame(truncated_bag_of_words_combine1, columns = add_col1)

X_combine = X_combine.reset_index(drop=True)
X_combine = pd.concat([X_combine, bag_df1], axis = 1)
print(f"train + valid + test shape :{blu}{X_combine.shape}{clr}, ")

X_combine = X_combine.reset_index(drop=True)
X_combine = pd.concat([X_combine, bag_df], axis = 1)
print(f"train + valid + test shape :{blu}{X_combine.shape}{clr}, ")

del bag_df,bag_df1
#X_test[add_col] = pd.DataFrame(truncated_bag_of_words_test, columns = add_col )
X_combine.head(5)


# ------------------
# Label encode features 
# ------------------
RMV = ['x_1',	'y_1',	'z_1','prev_x',	'prev_y',	'prev_z',	'next_x',	'next_y',	'next_z',
       'distance_from_origin',	'distance_prev',	'distance_next', 'sequence_group_num'	]    
# from the final dataset we remove all columns calculated based on coordinates and leave those calculated by groups

FEATURES = [c for c in X_combine.columns if not c in RMV]

for_delete = ['ID','all_sequences', 'set', 'description',"type_RNA",'sequence_id',	'sequence_target'] #todo  sequence_id	sequence_target

cat_features = X_combine.drop(RMV, axis=1).select_dtypes(include=['object']).columns.tolist()
num_features = X_combine.drop(RMV, axis=1).select_dtypes(exclude=['object']).columns.tolist()

for_label_encode = [c for c in cat_features if not c in for_delete] 

print(f"Category features will be encoded: ",for_label_encode)
print(f"\nNumerical: ",num_features)

for i,c in enumerate(for_label_encode):
    combine = X_combine[c]
    combine,_ = pd.factorize(combine)
    X_combine[c] = combine.astype("float32")  # int32
  


# ------------------
# Result datasets for model 
# ------------------
X_train = X_combine[X_combine['set'] == 'train']
X_valid = X_combine[X_combine['set'] == 'valid']
X_test = X_combine[X_combine['set'] == 'test']

X_combine = X_combine.drop(for_delete, axis =1, errors='ignore')
X_train = X_train.drop(for_delete, axis =1, errors='ignore')
X_valid = X_valid.drop(for_delete, axis =1, errors='ignore')
X_test = X_test.drop(for_delete, axis =1, errors='ignore')

FEATURES = [c for c in X_train.columns if not c in RMV]
target_col = ['x_1', 'y_1', 'z_1']

print(f"train shape :{blu}{X_train.shape}{clr}, ", f"valid shape :{blu}{X_valid.shape}{clr}, ", f"test shape :{blu}{X_test.shape}{clr}")

print(f"X_train ->  isnull :{X_train.isnull().values.sum()}")
print(f"X_valid ->  isnull :{X_valid.isnull().values.sum()}")
print(f"X_test -> isnull :{X_test.isnull().values.sum()}")

X_train.head(5)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



reduce_mem_usage(X_combine)
reduce_mem_usage(X_train)
reduce_mem_usage(X_valid)
reduce_mem_usage(X_test)


FEATURES = [
    "atom_round150",
    "atom_round50",
    "CG_cnt",
    "chain",
    "chainres_max_x_1",
    "chainres_max_y_1",
    "chainres_max_z_1",
    "CC_cnt",
    "AU_cnt",
    #"chainres_min_x_1",
    "desc_truncatedsvd1",
    "desc_truncatedsvd11",
    #"desc_truncatedsvd12",
    "desc_truncatedsvd16",
    #"desc_truncatedsvd8",
    #"desc_truncatedsvd9",
    "nucleotide_ngram3_min_x_1",
    "nucleotide_ngram3_min_y_1",
    "nucleotide_ngram3_min_z_1",
    "nucleotide_ngram4_mean_distance_from_origin",
    "res_max_distance_from_origin",
    "res_std_distance_from_origin",
    "resname_mean_x_1",
    "resname_mean_y_1",
    "resname_mean_z_1",
    "resname_prevresresname_next_max_x_1",
    "resname_prevresresname_next_max_y_1",
    "resname_prevresresname_next_max_z_1",
    "nucleotide_ngram3_mean_distance_from_origin",
    "resname_prevresresname_next_mean_x_1",
    "resname_prevresresname_next_mean_y_1",
    "resname_prevresresname_next_mean_z_1",
    "RNA",
    "rna_truncatedsvd0",
    "resname_prevresresname_next_min_x_1",
    "resname_prevresresname_next_min_y_1",
    "resname_prevresresname_next_min_z_1",
    "sequence_groupres_mean_x_1",
    "sequence_groupres_mean_y_1",
    "sequence_groupres_mean_z_1",
    "sequence_group_mean_x_1",
    "sequence_group_mean_y_1",
    "sequence_group_mean_z_1",
    "sequence_group_min_x_1",
    "sequence_group_min_y_1",
    "sequence_group_min_z_1",
    "chainres_min_x_1",
    "chainres_min_y_1",
    "chainres_min_z_1",
    'x_window','y_window','z_window'
]


if 0:
    corr_matrix = X_train[target_col+FEATURES].corr().abs()

    threshold = 0.3
    
    filtered_corr_df = corr_matrix[(corr_matrix >= threshold) & (corr_matrix != 1.000)] 
    
    plt.figure(figsize=(30,30))
    sns.heatmap(filtered_corr_df, annot=True, cmap="Reds")
    plt.show()


#FEATURES = ['resname','resid'] + list_f + add_col + add_col1
FEATURES = [c for c in X_combine.columns if not c in RMV]
print(len(FEATURES))
#FEATURES


def calculate_tm_score_exact(pred_coords, true_coords):
    """
    Implementation closer to the official method used by US-align.
    """
    # Remove padding
    mask = ~np.all(true_coords == 0)
    pred = pred_coords[mask]
    true = true_coords[mask]
    
    Lref = len(true_coords)
    
    # Define d0 exactly as in the evaluation formula
    if Lref >= 30:
        d0 = 0.6 * np.sqrt(Lref - 0.5) - 2.5
    elif Lref >= 24:
        d0 = 0.7
    elif Lref >= 20:
        d0 = 0.6
    elif Lref >= 16:
        d0 = 0.5
    elif Lref >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    
    # Normalize structures
    pred_centered = pred - np.mean(pred, axis=0)
    true_centered = true - np.mean(true, axis=0)
    
    # Covariance matrix for optimal rotation
    covariance = np.dot(pred_centered.T, true_centered)
    U, S, Vt = np.linalg.svd(covariance)
    rotation = np.dot(U, Vt)
    
    # Apply rotation
    pred_aligned = np.dot(pred_centered, rotation)
    
    # Calculate distances
    distances = np.sqrt(np.sum((pred_aligned - true_centered) ** 2, axis=1))
    
    # Calculate TM-score terms
    tm_terms = 1.0 / (1.0 + (distances / d0) ** 2)
    tm_score = np.sum(tm_terms) / Lref
    
    return float(tm_score)
    
def calculate_tm_score(pred_coords, true_coords, d0_scale=1.24):
    """
    Calculates a robust approximation of the TM-score between predicted and true coordinates.
    Adds protection against division by zero and NaN.
    """
    # Remove padding (rows with zeros) from true structures
    mask = ~np.all(true_coords == 0)
    pred = pred_coords[mask]
    true = true_coords[mask]
    
    L = len(true_coords)

    if L < 3:
        return 0.0
    
    # Define d0 based on L (values adapted for RNA)
    if L >= 30:
        d0 = 0.6 * np.sqrt(L - 0.5) - 2.5
        d0 = max(0.1, d0)
    elif L >= 24:
        d0 = 0.7
    elif L >= 20:
        d0 = 0.6
    elif L >= 16:
        d0 = 0.5
    elif L >= 12:
        d0 = 0.4
    else:
        d0 = 0.3
    
    distances = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    tm_terms = 1.0 / (1.0 + (distances / (d0 + 1e-8)) ** 2)
    tm_score = np.sum(tm_terms) / L
    return float(tm_score)

def plot_projection(pred):
# Plot the results
    plt.figure(figsize=(15, 5))
    s = 100
    a = 0.4
    
    
    plt.subplot(1, 6, 1)  
    plt.scatter(valid['x_1'],valid['y_1'], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
    plt.title('X Y projection valid')
    
    plt.subplot(1, 6, 2)  
    plt.scatter(valid['y_1'],valid['z_1'], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
    plt.title('Y Z projection valid')
    
    plt.subplot(1, 6, 3) 
    plt.scatter(valid['x_1'], valid['z_1'], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
    plt.title('X Z projection valid')
    
    plt.subplot(1, 6, 4)  
    plt.scatter(pred[:, 0],pred[:, 1], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
    plt.title('X Y projection')
    
    plt.subplot(1, 6, 5)  
    plt.scatter(pred[:, 1],pred[:, 2], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
    plt.title('Y Z projection')
    
    plt.subplot(1, 6, 6) 
    plt.scatter(pred[:, 0],pred[:, 1], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
    plt.title('X Z projection')
    
    plt.tight_layout()  
    
    plt.legend()
    plt.show()


from sklearn.model_selection import KFold,cross_val_score,RepeatedKFold,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report,  auc, roc_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.tree import DecisionTreeRegressor


reg = linear_model.MultiTaskElasticNetCV(cv=3, l1_ratio=0.55, max_iter=2000, tol=0.001, random_state=0, selection='cyclic')
reg = linear_model.MultiTaskElasticNet( l1_ratio=0.95, max_iter=500, tol=0.001, random_state=0, selection='cyclic')
#reg = linear_model.MultiTaskElasticNet(alpha=0.91, l1_ratio=0.95, max_iter=1000, tol=0.001, random_state=0, selection='cyclic')
#reg = RadiusNeighborsRegressor(radius=1000.0)
reg.fit(X_train[FEATURES], X_train[target_col],)

print("Model score: ",reg.score(X_valid[FEATURES], X_valid[target_col]))
oof = reg.predict(X_valid[FEATURES])
print("Valid shape: ", oof.shape)
pred = reg.predict(X_test[FEATURES])
print("Predictions shape: ",pred.shape)
display(pred)
plot_projection(pred) 
y_valid = X_valid[target_col].copy()
y_valid = y_valid.reset_index(drop = True)

y_pred = pd.DataFrame(pred[:,0:3], columns = target_col)

y_valid = np.nan_to_num(X_valid[target_col], nan=0.0)
y_pred = np.nan_to_num(y_pred, nan=0.0)
tm_scores = []
tm_scores1 = []

for i in range(len(X_valid[target_col])):
       
        tm = calculate_tm_score(y_pred[i], y_valid[i])
        tm1 = calculate_tm_score_exact(y_pred[i], y_valid[i])
        tm_scores.append(tm)
        tm_scores1.append(tm1)
avg_tm_score = np.mean(tm_scores)
print(f"Approximate average TM-score: {avg_tm_score:.8f}")
avg_tm_score1 = np.mean(tm_scores1)
print(f"Approximate average TM-score exact: {avg_tm_score1:.8f}")


from sklearn.feature_selection import SequentialFeatureSelector

if 0:
    selector = SequentialFeatureSelector(reg ,n_features_to_select=30, direction='forward', scoring="neg_root_mean_squared_error", cv=2)
    
    selector.fit_transform(X_train[FEATURES], X_train[target_col])
    mask = selector.get_support() #list of booleans
    new_features = [] # The list of your K best features
    feature_names = FEATURES
    for bool_val, feature in zip(mask, feature_names):
        if bool_val:
            new_features.append(feature)
    FEATURES = new_features
    #dataframe = pd.DataFrame(X_train, columns=new_features)
    #dataframe


#dataframe[list(set(new_features)-set(target_col))#


from sklearn import linear_model
# Fit estimators
ESTIMATORS = {
    #+"KNeighborsRegressor": KNeighborsRegressor(n_neighbors=15, weights='distance', algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski'),
    #! "RadiusNeighborsRegressor": RadiusNeighborsRegressor(radius=100.0,weights='distance' , algorithm='kd_tree', leaf_size=30, p=1,),
    # "LinearRegression": LinearRegression(),
    # "RidgeCV": RidgeCV(alphas=(0.01, 5.0, 10.0), cv=3),
    # "GaussianProcessRegressor": GaussianProcessRegressor(),?
    #"KernelRidge": KernelRidge(),?
    #"Lars": linear_model.Lars(),
    #"Lasso": linear_model.Lasso(),
    #"LassoLars": linear_model.LassoLars(),
    #"MultiTaskLassoCV": linear_model.MultiTaskLassoCV(), cv
    #+"OrthogonalMatchingPursuit": linear_model.OrthogonalMatchingPursuit(),
    #"RANSACRegressor1": linear_model.RANSACRegressor(estimator=KNeighborsRegressor(n_neighbors=15, weights='distance', algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski'),min_samples=500,random_state=0),
# test ortogonal first    "RANSACRegressor2": linear_model.RANSACRegressor(estimator=linear_model.MultiTaskElasticNet( l1_ratio=0.95, max_iter=500, tol=0.001, random_state=0, selection='cyclic'),min_samples=500,random_state=0),
    "RANSACRegressor3": linear_model.RANSACRegressor(estimator=linear_model.OrthogonalMatchingPursuit(tol = 0.00001), max_trials=1000,  loss='squared_error',   min_samples=2000,random_state=0),   # add clf
    #"PLSCanonical": PLSCanonical(n_components=3,scale=True, algorithm='svd', max_iter=2000,), #svd
    #"PLSRegression": PLSRegression(n_components=3,scale=False,  max_iter=300,), 
    #"DecisionTreeRegressor":DecisionTreeRegressor(max_depth=8),
    #"RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_features=10, random_state=0, criterion='absolute_error'),
    #"ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, max_features=10, random_state=0, criterion='absolute_error'), #squared_error”, “absolute_error”, “friedman_mse”, “poisson”
    #"MultiTaskLassoCV": linear_model.MultiTaskLassoCV(cv=3),
}

y_test_predict = dict()
oof_test = dict()
for name, estimator in ESTIMATORS.items():
    
    print(f"Model {name}")
    estimator.fit(X_train[FEATURES], X_train[target_col])

    print(f".... score: ",estimator.score(X_valid[FEATURES], X_valid[target_col]))
    oof_test[name] = estimator.predict(X_valid[FEATURES])
    y_test_predict[name] = estimator.predict(X_test[FEATURES])
    #print("Valid shape: ", oof.shape)
    #pred = estimator.predict(X_test[FEATURES])
    #print("Predictions shape: ",y_test_predict[name].shape)
    #display(pred)

    #for j, est in enumerate(sorted(ESTIMATORS)):
    
    plot_projection(y_test_predict[name])    

    y_valid = X_valid[target_col].copy()
    y_valid = y_valid.reset_index(drop = True)
    
    y_pred = pd.DataFrame(y_test_predict[name][:,0:3], columns = target_col)
    
    y_valid = np.nan_to_num(y_valid, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    tm_scores = []
    tm_scores1 = []
    
    for i in range(len(X_valid[target_col])):
           
            tm = calculate_tm_score(y_pred[i], y_valid[i])
            tm1 = calculate_tm_score_exact(y_pred[i], y_valid[i])
            tm_scores.append(tm)
            tm_scores1.append(tm1)
        
    avg_tm_score = np.mean(tm_scores)
    print(f"Approximate average TM-score: {avg_tm_score:.8f}")
    avg_tm_score1 = np.mean(tm_scores1)
    print(f"Approximate average TM-score exact: {avg_tm_score1:.8f}")
    print(f'{"-" * 100}')


pred=y_pred


print(y_pred[:, 0].shape)
y_pred[:, 0]


print(pred[:, 0].shape)
pred[:, 0]



sub = pd.read_csv('../input/stanford-rna-3d-folding/sample_submission.csv')
col_name = ['x_1',	'y_1',	'z_1',	'x_2',	'y_2',	'z_2',	'x_3',	'y_3',	'z_3',	'x_4',	'y_4',	'z_4',	'x_5',	'y_5',	'z_5']
sub = sub.drop(col_name, axis =1)
sub['x_1'] = pred[:, 0]
sub['y_1'] = pred[:, 1]
sub['z_1'] = pred[:, 2]

sub['x_2'] = 0.0
sub['y_2'] = 0.0
sub['z_2'] = 0.0

sub['x_3'] = 0.0
sub['y_3'] = 0.0
sub['z_3'] = 0.0

sub['x_4'] = 0.0
sub['y_4'] = 0.0
sub['z_4'] = 0.0

sub['x_5'] = 0.0
sub['y_5'] = 0.0
sub['z_5'] = 0.0

#for i in range(1,6):
 #   columns+=[f"x_{i}"]
#    columns+=[f"y_{i}"]
 #   columns+=[f"z_{i}"]

VER = 1
sub.to_csv('submission.csv', index=False)
sub


def plot_structure(df: pd.DataFrame, sequence_id: str) -> None:
    sequence_df = df[df["sequence_id"] == sequence_id]
    sequence_points = sequence_df[["x_1", "y_1", "z_1", "resname"]]
    seq_lst = sequence_df['resname'].to_list()
    seq_str = ''.join(seq_lst)
    print(seq_str)
    #print(sequence_points)
    
    colors = {"A": "red", "G": "blue", "C": "green", "U": "orange"}
    fig = go.Figure()
    
    for resname, color in colors.items():
        subset = sequence_df[sequence_df["resname"] == resname]
        fig.add_trace(go.Scatter3d(
            x=subset["x_1"], y=subset["y_1"], z=subset["z_1"],
            mode='markers',
            marker=dict(size=5, color=color),
            name=resname,
            opacity=0.8
        ))
    
    fig.add_trace(go.Scatter3d(
        x=sequence_df["x_1"], y=sequence_df["y_1"], z=sequence_df["z_1"],
        mode='lines',
        line=dict(color='gray', width=2),
        name='RNA Backbone'
    ))
    
    fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            title=f'3D RNA Structure of sequence {sequence_id}',
        )
            
    fig.show(renderer="iframe")


sequence_id = "8UYS"
sequence_df = valid[valid["sequence_group"] == sequence_id].copy()

sequence_points = sequence_df[["x_1", "y_1", "z_1", "resname"]]
seq_lst = sequence_df['resname'].to_list()
seq_str = ''.join(seq_lst)
sequence_df[0:1]
plot_structure(sequence_df, sequence_df["sequence_id"].max())


sub_id = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1])).unique()

def print_plot_str(c):
    print("----------------------------------------------------------------")
    print(c)
    sequence_df = sub[sub["sequence_group"] == c].copy()
    #display(sequence_df.head(3))
    plot_structure(sequence_df, c)

sub["sequence_group"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
sub["sequence_id"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
print_plot_str('R1149')



print_plot_str('R1108')


print_plot_str('R1156')


print_plot_str('R1136')


print_plot_str('R1126')


print_plot_str('R1116')


print_plot_str('R1138')


print_plot_str('R1117v2')


print_plot_str('R1128')


print_plot_str('R1190')


print_plot_str('R1107')


print_plot_str('R1189')


sequence_df = sub[sub["sequence_group"] == 'R1138'].copy()
X = sequence_df['x_1']
Y = sequence_df['y_1']
Z = sequence_df['z_1']


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)  
sns.scatterplot(data=sequence_df, x=X, y=Y, s=Z*Z.max()/X.max(),hue = 'resname')
plt.title('X Y projection')

plt.subplot(1, 3, 2)  
sns.scatterplot(data=sequence_df, x=Y, y=Z, s=X*X.max()/Y.max(),hue = 'resname')
plt.title('Y Z projection')

plt.subplot(1, 3, 3) 
sns.scatterplot(data=sequence_df, x=X, y=Z, s=Y*Y.max()/X.max(),hue = 'resname')
plt.title('X Z projection')

plt.tight_layout()  
plt.show()


g = sns.pairplot(data=sequence_df[['x_1', 'y_1', 'z_1', 'resname']],
             hue = 'resname', height=2.5,diag_kind="kde",
             diag_kws = {'color' : default_color_1},
             plot_kws = {'s' : 15, 
                         'alpha' : 0.5,
                         'color' : default_color_1})
g.map_lower(sns.kdeplot, levels=1, color=".2")

