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
df_train_l = pd.read_csv('../input/stanford-rna-3d-folding/train_labels.csv')

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

#df_train = df_train.fillna(0)

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

#df_valid = df_valid.fillna(0)

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

#del df_train ,df_valid ,df_train_seq ,df_valid_seq ,df_test 

print(f"train ->  isnull :{train.isnull().values.sum()}")
print(f"valid ->  isnull :{valid.isnull().values.sum()}")
print(f"test -> isnull :{test.isnull().values.sum()}")

display(train[1:2])
display(valid[1:2])
display(test[1:2])


df_train[['ID'	,'resname',	'resid',	'x_1',	'y_1',	'z_1']].describe()


train[['resname',	'resid',	'x_1',	'y_1',	'z_1']].describe()


target_col = ['x_1', 'y_1', 'z_1']

train[target_col] = train[target_col].fillna(train[target_col].mean())
valid[target_col] = valid[target_col].fillna(valid[target_col].mean())


#find group from test in train dataset
print("sequence_group find in train test:")
for i in train['sequence_group'].unique():
    for j in test['sequence_group'].unique():
        if i == j:
            print(i)
print("sequence_group find in train valid:")
for i in train['sequence_group'].unique():
    for j in valid['sequence_group'].unique():
        if i == j:
            print(i)



valid[valid['sequence_group'] == '8UYS'][1:4]


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
  
    df['nucleotide_counts'] = df['sequence'].apply(count_nucleotides)
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

    #df['prev_x'] = df['x_1'].shift(1)
    #df['prev_y'] = df['y_1'].shift(1)
    #df['prev_z'] = df['z_1'].shift(1)
    #df['next_x'] = df['x_1'].shift(-1)
    #df['next_y'] = df['y_1'].shift(-1)
    #df['next_z'] = df['z_1'].shift(-1)

    #df["distance_from_origin"] = np.sqrt(df["x_1"]**2 + df["y_1"]**2 + df["z_1"]**2 )

   # df['distance_prev'] = df.apply(lambda row: euclidean_distance( row["x_1"], row["y_1"], row["x_1"],row['prev_x'], row['prev_y'], row['prev_z']), axis=1)
    #df['distance_next'] = df.apply(lambda row: euclidean_distance( row["x_1"], row["y_1"], row["x_1"],row['next_x'], row['next_y'], row['next_z']), axis=1)

    df["chain"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[1:2])[6:].replace('auth', '').replace(' ', '').strip())
    df["RNA"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[2:3])[0:])  
    df["RNA_chain"] = df["RNA"].astype(str).apply(lambda x: "_".join(x.split("(5'-R(")[1:])[0:].replace(")-3')","").replace(") -3')",""))  
    df["add_chain"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[3:][4:5] )[6:].replace('auth', '').replace(' ', '').strip())
    df["add_RNA"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[3:][5:6] ))
    df["type_RNA"] = df["all_sequences"].astype(str).apply(lambda x: "_".join(x.split("|")[3:][6:7] ))  
    df["class"] = df["type_RNA"].astype(str).apply(lambda x: "_".join(x.split("\n")[0:1] ))
    df["class_chain"] = df["type_RNA"].astype(str).apply(lambda x: "_".join(x.split("\n")[1:] ))
    
    #df = df.fillna(0)


# ------------------
# Prepare dataset for model 
# ------------------
print(f"Prepare datasets for model ")

#train['set'] = 'train'
#valid['set'] = 'valid'
#test['set'] = 'test'


# ------------------
# Create Features
# ------------------

FE(train)
FE(valid)
FE(test)


FEATURES = train.columns.tolist()
print(f"train shape :{blu}{train.shape}{clr}, ", f"valid shape :{blu}{valid.shape}{clr}, ", f"test shape :{blu}{test.shape}{clr}")

print(f"features shape :{blu}{len(FEATURES)}{clr} ")
print(f'{"-" * 100}')

one_hot_encoded = pd.get_dummies(train['resname'], prefix='nucleotide')
train = pd.concat([train, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(valid['resname'], prefix='nucleotide')
valid = pd.concat([valid, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(test['resname'], prefix='nucleotide')
test = pd.concat([test, one_hot_encoded], axis=1)


print(f"train shape :{blu}{train.shape}{clr}, ", f"valid shape :{blu}{valid.shape}{clr}, ", f"test shape :{blu}{test.shape}{clr}")

print(f'{"-" * 100}')



train.head(3)




# ------------------
# Create Features with bag word for Descripction and RNA columns
# ------------------
def bag_of_word(df):
    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=None)    
    bag_of_words_combine = count_vectorizer.fit_transform(df['description'])
    bag_of_words_combine1 = count_vectorizer.fit_transform(df['RNA'])
    
    svd = TruncatedSVD(n_components=50, n_iter=30, random_state=12)
    truncated_bag_of_words_combine = svd.fit_transform(bag_of_words_combine)
    svd1 = TruncatedSVD(n_components=15, n_iter=25, random_state=12)
    truncated_bag_of_words_combine1 = svd1.fit_transform(bag_of_words_combine1)
    
    add_col = [f'desc_{i}' for i in svd.get_feature_names_out()]
    add_col1 = [f'rna_{i}' for i in svd1.get_feature_names_out()]
    
    bag_df = pd.DataFrame(truncated_bag_of_words_combine, columns = add_col )
    bag_df1 = pd.DataFrame(truncated_bag_of_words_combine1, columns = add_col1)
    
    df = df.reset_index(drop=True)
    df = pd.concat([df, bag_df1], axis = 1)
    
    df = df.reset_index(drop=True)
    df = pd.concat([df, bag_df], axis = 1)
    
    del bag_df,bag_df1
    
    return df

train = bag_of_word(train)
valid = bag_of_word(valid)
test = bag_of_word(test)

train.head(5)


cc = "sequence_group"
plt.figure(figsize=(10, 5))
sns.histplot(valid[cc], kde=True)
plt.title(f" {cc} Distribution")
plt.xlabel(cc)
plt.ylabel("Frequency")
plt.show()

print(valid[cc].describe())
#correlation = valid[["x_1","y_1","z_1", cc]].corr()
#print("Correlation between :\n", correlation)


# ------------------
# Label encode features 
# ------------------
RMV = ['x_1',	'y_1',	'z_1']    
# from the final dataset we remove all columns calculated based on coordinates and leave those calculated by groups

FEATURES = [c for c in train.columns if not c in RMV]

for_delete = ['ID','all_sequences',  'description',"type_RNA",'sequence_id',	'sequence_target'] 

cat_features = train.drop(RMV, axis=1).select_dtypes(include=['object']).columns.tolist()
num_features = train.drop(RMV, axis=1).select_dtypes(exclude=['object']).columns.tolist()

for_label_encode = [c for c in cat_features if not c in for_delete] 

print(f"Category features will be encoded: ",for_label_encode)
print(f"\nNumerical: ",num_features)



from sklearn.preprocessing import LabelEncoder

# Create dictionary to store label encoders
label_encoders = {}

for col in for_label_encode:
    # Handle potential NaN values before encoding
    
    train[col] = train[col].astype(str).fillna('missing')
    valid[col] = train[col].astype(str).fillna('missing')
    test[col] = test[col].astype(str).fillna('missing')
    
    # Create and fit label encoder
    le = LabelEncoder()
    le.fit(pd.concat([train[col],pd.concat([valid[col],test[col]],axis=0)], axis =0))
    train[col] = le.transform(train[col])
    valid[col] = le.transform(valid[col])
    test[col] = le.transform(test[col])
    
    label_encoders[col] = le


print(f"X_train ->  isnull :{train.isnull().values.sum()}")
print(f"X_valid ->  isnull :{valid.isnull().values.sum()}")
print(f"X_test -> isnull :{test.isnull().values.sum()}")


train.isnull().sum()



df_train


df_train.isna().sum()


label_encoders


train = train.dropna()
valid = valid.dropna()

#for i,c in enumerate(for_label_encode):
#    combine = X_combine[c]
#    combine,_ = pd.factorize(combine)
#    X_combine[c] = combine.astype("float32")



# ------------------
# Result datasets for model 
# ------------------

train = train.drop(for_delete, axis =1, errors='ignore')
valid = valid.drop(for_delete, axis =1, errors='ignore')
test = test.drop(for_delete, axis =1, errors='ignore')


X_train = train.copy()
X_valid = valid.copy()
X_test = test.copy()

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


reduce_mem_usage(X_train)
reduce_mem_usage(X_valid)
reduce_mem_usage(X_test)


X_train = X_train.drop('nucleotide_-', axis =1)
X_train = X_train.drop('nucleotide_X', axis =1)
FEATURES = [c for c in X_train.columns if not c in RMV]
X_train


X_train.info()


#FEATURES = ['resname','resid'] + list_f + add_col + add_col1
print(len(FEATURES))

FEATURES


from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from xgboost import plot_importance
from xgboost import XGBRegressor
import xgboost as xgb
print(f"XGBoost version",xgb.__version__)


clf = 'xgb'
#clf = 'lnr'


clf = 'xgb'
#clf = 'lnr'
xgb_params = { 'learning_rate': 0.01,
                        #'min_child_weight': 0.3,
                        'early_stopping_rounds': 300,
                        #multi_strategy="multi_output_tree",
                        #'num_target':  3,
                        'booster': 'gbtree',
                        'tree_method': "hist",
                        'n_estimators': 3000,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'n_jobs': 10,
                        'max_depth': 15,
                        'eval_metric':  'rmse' }

FOLDS = 2
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(X_train[FEATURES]),3))  # x,y,z
pred = np.zeros((len(X_test[FEATURES]),3))
X1_test = X_test[FEATURES]



X_train = X_train.reset_index(drop=True).copy()


%%time

#  K FOLD x
for i, (train_index, valid_index) in enumerate(kf.split(X_train[FEATURES])):
    print(f"### X Fold {i+1} ###")
    
    X1_train = X_train.loc[train_index,FEATURES].reset_index(drop=True).copy()

    X1_valid = X_train.loc[valid_index,FEATURES].reset_index(drop=True).copy()
    y1_train = X_train.loc[train_index,['x_1']]
    y1_valid = X_train.loc[valid_index,['x_1']]
    X1_test = X1_test[FEATURES].reset_index(drop=True).copy()
    print(f"X train shape :{blu}{X1_train.shape}{clr}, ", f"X valid shape :{blu}{X1_valid.shape}{clr}, ", f"X test shape :{blu}{X1_test.shape}{clr}")
    print(f"x train shape :{blu}{y1_train.shape}{clr}, ", f"x valid shape :{blu}{y1_valid.shape}{clr}, ")

    
    xgb_x = XGBRegressor(**xgb_params)
    xgb_x.fit(X1_train, y1_train, eval_set=[(X1_valid, y1_valid)],  verbose=300 )
    
    oof[valid_index,0] = xgb_x.predict(X1_valid)
    pred[:,0] += xgb_x.predict(X1_test)
    print('MAE x valid: %s' % mean_absolute_error(oof[valid_index,0], y1_valid))
    print("Model score: ",xgb_x.score(X1_train, y1_train))
pred[:,0] /= FOLDS
print("~result:")
display(pred)

X1_test['x_1'] = pred[:,0]



#  K FOLD y
for i, (train_index, valid_index) in enumerate(kf.split(X_train[FEATURES])):
    print(f"### Y Fold {i+1} ###")
    
    X2_train = X_train.loc[train_index,FEATURES + ['x_1']].reset_index(drop=True).copy()
    
    X2_valid = X_train.loc[valid_index,FEATURES + ['x_1']].reset_index(drop=True).copy()
    y2_train = X_train.loc[train_index,['y_1']]
    y2_valid = X_train.loc[valid_index,['y_1']]
    X2_test = X1_test[FEATURES + ['x_1']].reset_index(drop=True).copy()
    print(f"X2 train shape :{blu}{X2_train.shape}{clr}, ", f"X2 valid shape :{blu}{X2_valid.shape}{clr}, ", f"X2 test shape :{blu}{X2_test.shape}{clr}")
    print(f"y2 train shape :{blu}{y2_train.shape}{clr}, ", f"y2 valid shape :{blu}{y2_valid.shape}{clr}, ")

    xgb_y = XGBRegressor(**xgb_params)
    xgb_y.fit(X2_train, y2_train, eval_set=[(X2_valid, y2_valid)],  verbose=300 )
    oof[valid_index,1] = xgb_y.predict(X2_valid)
    pred[:,1] += xgb_y.predict(X2_test)
    print('MAE y valid: %s' % mean_absolute_error(oof[valid_index,1], y2_valid))
    print("Model score: ",xgb_y.score(X2_train, y2_train))
    
pred[:,1] /= FOLDS
print("~result:")
display(pred)

X1_test['y_1'] = pred[:,1]


#  K FOLD z
for i, (train_index, valid_index) in enumerate(kf.split(X_train[FEATURES])):
    print(f"### Z Fold {i+1} ###")
    
    X3_train = X_train.loc[train_index,FEATURES + ['x_1','y_1']].reset_index(drop=True).copy()
    X3_valid = X_train.loc[valid_index,FEATURES + ['x_1','y_1']].reset_index(drop=True).copy()
    y3_train = X_train.loc[train_index,['z_1']]
    y3_valid = X_train.loc[valid_index,['z_1']]
    X3_test = X1_test[FEATURES + ['x_1','y_1']].reset_index(drop=True).copy()
    print(f"X3 train shape :{blu}{X3_train.shape}{clr}, ", f"X3 valid shape :{blu}{X3_valid.shape}{clr}, ", f"X3 test shape :{blu}{X3_test.shape}{clr}")
    print(f"y3 train shape :{blu}{y3_train.shape}{clr}, ", f"y3 valid shape :{blu}{y3_valid.shape}{clr}, ")
    xgb_z = XGBRegressor(**xgb_params)
    xgb_z.fit(X3_train, y3_train, eval_set=[(X3_valid, y3_valid)],  verbose=300 )
    oof[valid_index,2] = xgb_z.predict(X3_valid)
    pred[:,2] += xgb_z.predict(X3_test)
    print('MAE z valid: %s' % mean_absolute_error(oof[valid_index,2], y3_valid))
    print("Model score: ",xgb_z.score(X3_train, y3_train))
    
    
pred[:,2] /= FOLDS
print("~result:")
display(pred)

X1_test['z_1'] = pred[:,2]


    
    # CLEAR MEMORY
#del X1_train, X1_valid, X2_train, X2_valid,X3_train, X3_valid
#del y1_train, y1_valid, y2_train, y2_valid, y3_train, y3_valid
#if i != FOLDS-1: del xgb_x,xgb_y,xgb_z
     



sorted_idx = np.argsort(xgb_x.feature_importances_)[::-1]
for index in sorted_idx:
    print([X1_test.columns[index], xgb_x.feature_importances_[index]]) 
fig, ax = plt.subplots(figsize=(16, 20))
 
# Horizontal Bar Plot
ax.barh(X1_test.columns[sorted_idx][:30], xgb_x.feature_importances_[sorted_idx][:30])

ax = plt.gca() #Getting the current axis
ax.spines['bottom'].set_visible(False) 
ax.spines['top'].set_visible(False) 
ax.spines['left'].set_visible(False) 
ax.spines['right'].set_visible(False) 

plt.setp(ax.spines.values(), visible=False) 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)
 
# Add x, y gridlines
ax.grid( color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width(), i.get_y(),
             str(round((i.get_width()), 3)),
             fontsize=8, 
             color='grey')
 
# Add Plot Title
ax.set_title('Permutation Importance',
             loc='left', )
 
# Add Text watermark
fig.text(0.9, 0.15, 'xgb_z', fontsize=12,
         color='grey', ha='right', va='top',
         alpha=0.7)
 
# Show Plot
plt.show()
#plot_importance(xgb_x, max_num_features = 15)
#plt.show()


feature_important = xgb_x.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features


y_valid = X_valid[target_col].copy()
y_valid = y_valid.reset_index(drop = True)
y_pred = pd.DataFrame(pred[:,0:3], columns = target_col)

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
print(f"Approximate average TM-score: {avg_tm_score1:.8f}")


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

#submission[['x_1','y_1','z_1','x_2','y_2','z_2','x_3','y_3','z_3','x_4','y_4','z_4','x_5','y_5','z_5']] = test_predictions[['x_1','y_1','z_1','x_2','y_2','z_2','x_3','y_3','z_3','x_4','y_4','z_4','x_5','y_5','z_5']] 

#for i in range(1,6):
 #   columns+=[f"x_{i}"]
#    columns+=[f"y_{i}"]
 #   columns+=[f"z_{i}"]

VER = 1
sub.to_csv('submission.csv', index=False)
sub


# Plot the results
plt.figure(figsize=(15, 5))
s = 100
a = 0.4


plt.subplot(1, 6, 1)  
plt.scatter(valid['x_1'],valid['y_1'], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
plt.title('X Y projection valid')

plt.subplot(1, 6, 2)  
plt.scatter(valid['y_1'], valid['z_1'], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
plt.title('Y Z projection valid')

plt.subplot(1, 6, 3) 
plt.scatter(valid['x_1'],valid['z_1'], edgecolor="k",c="cornflowerblue", s=s,alpha=a)
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


sub_id = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1])).unique()
#sub["sequence_group"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
#sub["sequence_id"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))

#sequence_id = "R1149"
#sequence_df = sub[sub["sequence_group"] == sequence_id].copy()

def print_plot_str(c):
    #for c in sub["sequence_group"].unique():
    print("----------------------------------------------------------------")
    print(c)
    sequence_df = sub[sub["sequence_group"] == c].copy()
    #display(sequence_df.head(3))
    plot_structure(sequence_df, c)

sub["sequence_group"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
sub["sequence_id"] = sub["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
print_plot_str('R1107')



print_plot_str('R1108')


print_plot_str('R1156')


print_plot_str('R1136')


print_plot_str('R1126')


print_plot_str('R1116')


print_plot_str('R1138')


print_plot_str('R1117v2')


print_plot_str('R1128')


print_plot_str('R1190')


print_plot_str('R1149')


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

