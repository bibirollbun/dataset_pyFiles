%%capture
!pip install rdkit
!pip install atomInSmiles


import pandas as pd
from rdkit import Chem
import atomInSmiles # specialized tokenizer
from tqdm.notebook import tqdm
from pprint import pprint
tqdm.pandas(desc="Processing")


MAIN_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'


# Drug design dataset
dd_df = pd.read_csv('/kaggle/input/chembl22/chembl_22_clean_1576904_sorted_std_final.smi', 
                 sep='\t',
                 # nrows=100000,
                 header=None,
                 names=['SMILES'],
                 usecols=[0])

dd_df.sample(5)


# Toxicity dataset 
tx_df_1 = pd.read_csv('/kaggle/input/smiles-toxicity/data/NR-ER-test/names_smiles.csv', 
                   header=None,
                   names=['SMILES'],
                   usecols=[1])

tx_df_2 = pd.read_csv('/kaggle/input/smiles-toxicity/data/NR-ER-train/names_smiles.csv', 
                   header=None,
                   names=['SMILES'],
                   usecols=[1])

tx_df = pd.concat([tx_df_1, tx_df_2], ignore_index=True)
tx_df.sample(5)


# competition dataset
train_df = pd.read_csv(MAIN_PATH + 'train.csv')
test_df = pd.read_csv(MAIN_PATH + 'test.csv')
supp_ds1_df = pd.read_csv(MAIN_PATH + 'train_supplement/dataset1.csv')
supp_ds2_df = pd.read_csv(MAIN_PATH + 'train_supplement/dataset2.csv')
supp_ds3_df = pd.read_csv(MAIN_PATH + 'train_supplement/dataset3.csv')
supp_ds4_df = pd.read_csv(MAIN_PATH + 'train_supplement/dataset4.csv')


dfs_dict = {'train_df': ('Competition train dataset', train_df),
            'test_df': ('Competition test dataset', test_df),
            'supp_ds1_df': ('Competition sumlement dataset1', supp_ds1_df),
            'supp_ds2_df': ('Competition sumlement dataset2', supp_ds2_df),
            'supp_ds3_df': ('Competition sumlement dataset3', supp_ds3_df),
            'supp_ds4_df': ('Competition sumlement dataset4', supp_ds4_df),
            'dd_df': ('Drug design', dd_df),
            'tx_df': ('Toxicity dataset', tx_df),
           }


def df_styler(styler, 
              subset, 
              is_numeric=False,
              totals=False):
    
    if is_numeric:
        styler.format('{:,d}', subset=subset)
    else:
        styler.map(lambda x: 'text-align: left;', subset=subset)
    
    if totals:
        styler.map(lambda x: 'font-weight:bold;', subset=subset)
        styler.relabel_index(['Total'])
    
    return styler

df = pd.DataFrame([(data[0], data[1].size) for data in dfs_dict.values()], 
                  columns=['Dataset', 'Length']
                 )
summary = df.agg(['sum'])
summary['Dataset'] = ''

df.style.pipe(df_styler, subset=['Dataset'])\
        .pipe(df_styler, subset=['Length'], is_numeric=True)\
    .concat(summary.style.pipe(df_styler, subset=['Length'], is_numeric=True, totals=True))\
    .set_caption("Number of molecules")\
    .set_table_styles([{'selector': 'caption', 'props': 'font-size:18pt; color:gray;'}], 
                      overwrite=False)


# some tables have non-unique values
df = pd.DataFrame([(data[0], data[1].shape[0] - data[1].nunique().iloc[0]) for data in dfs_dict.values()], 
                  columns=['Dataset', 'Non-unique values']
                 )

summary_excl_loc = df.agg(['sum'])
summary_excl_loc['Dataset'] = ''

df.style.pipe(df_styler, subset=['Dataset'])\
        .pipe(df_styler, subset=['Non-unique values'], is_numeric=True)\
    .concat(summary_excl_loc.style.pipe(df_styler, subset=['Non-unique values'], is_numeric=True, totals=True))\
    .set_caption("Non-unique values")\
    .set_table_styles([{'selector': 'caption', 'props': 'font-size:18pt; color:gray;'}], 
                      overwrite=False)


# let's exclude it
union_df = pd.DataFrame(pd.concat([df[1] for df in dfs_dict.values()]).SMILES.unique(), columns=['SMILES'])

fin_len = union_df.size
orig_len = summary['Length'].item()
excl_loc = summary_excl_loc['Non-unique values'].item()

print(f'Result length {fin_len:,d}')
print(f'Excluded non-unique: \n\tlocal  - {excl_loc:,d}\n\tglobal - {orig_len - fin_len - excl_loc:,d}')


tokenizer = atomInSmiles.encode

# additional datasets have SMILES that rdkit can't handle
def tokenize_w_err(data):
    
    try:
        tokens = tokenizer(data)
    except:
        tokens = 'error'
    return tokens


def take_sample(smiles: str):
    """
    Just shows different representations of the molecule: 
    SMILES, tokens, structural formula
    -----------
    Parameters:
    smiles - string in SMILES notation
    """
    
    print('{0}Sample{0} \n'.format('='*35))
    print(f'SMILES: {smiles} \n')
    
    print('Tokens:')
    pprint(tokenize_w_err(smiles))
    
    print('\nMolecule:')
    display(Chem.MolFromSmiles(smiles))

# take random sample
take_sample(union_df.sample(1).SMILES.item())


# methane
take_sample('C')


take_sample('CC') # ethane is just a line in skeletal form


take_sample('CCC')


take_sample(union_df.iloc[0].item())


tokenized_df = union_df.SMILES.progress_map(tokenize_w_err)


tokenized_df.info()


# concat SMILES and tokens
res_df = pd.concat([union_df, tokenized_df], axis=1)
res_df.columns = ['SMILES', 'tokens']
res_df.head()


# let's handle an errors
err_df = res_df[res_df['tokens']=='error']
print(f'Number of erros - {err_df.shape[0]} ({err_df.shape[0]/res_df.shape[0]:.2%})\n')
take_sample(err_df.iloc[0].SMILES)


# O=[N](=O)=O ==> O[N](=O)=O   - nitrogen becomes pentavalent
take_sample('O[N](=O)=O')


# delete an erroros
print(f'size befor - {res_df.shape[0]:,d}')
res_df = res_df[res_df['tokens']!='error']
print(f'size after - {res_df.shape[0]:,d}')


res_df.to_parquet('tokenized_smiles.parquet')

