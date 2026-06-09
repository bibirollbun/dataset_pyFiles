import pandas as pd
# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# è¯»å�–è®­ç»ƒåº�åˆ—æ•°æ�®
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
# è¯»å�–è®­ç»ƒæ ‡ç­¾æ•°æ�®
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
# è¯»å�–éªŒè¯�åº�åˆ—æ•°æ�®
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
# è¯»å�–éªŒè¯�æ ‡ç­¾æ•°æ�®
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')


train_sequences.shape


# æ‰“å�°è®­ç»ƒåº�åˆ—çš„è¡Œæ•°å’Œåˆ—æ•°
print(f'train_sequences has {train_sequences.shape[0]} rows and {train_sequences.shape[1]} columns')
# æ‰“å�°è®­ç»ƒåº�åˆ—ä¸­ç¼ºå¤±å€¼çš„æ€»æ•°
print(f'train_sequences has {train_sequences.isna().sum().sum()} NAs.')
# å¿«é€ŸæŸ¥çœ‹æ•°æ�®çš„å‰�3è¡Œ
train_sequences.head(12)


# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))
# è®¡ç®—åº�åˆ—é•¿åº¦å¹¶åˆ›å»ºæ–°çš„ DataFrame
train_sequences['length'] = train_sequences['sequence'].str.len()
# ç»˜åˆ¶ç®±çº¿å›¾
sns.boxplot(x='length', data=train_sequences)
plt.xlabel("Sequence length")
plt.ylabel("Sequence")
plt.title("Boxplot of sequence length values (with outliers)")
plt.grid(True)
plt.show()


# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))

# è®¡ç®—åº�åˆ—é•¿åº¦å¹¶åˆ›å»ºæ–°çš„ DataFrame
train_sequences['length'] = train_sequences['sequence'].str.len()

# ç»˜åˆ¶ç®±çº¿å›¾ï¼Œä¸�æ˜¾ç¤ºå¼‚å¸¸å€¼
sns.boxplot(x='length', data=train_sequences, showfliers=False)
plt.xlabel("Sequence length")
plt.ylabel("Sequence")
plt.title("Boxplot of sequence length values (without outliers)")
plt.grid(True)
plt.show()


# åˆ›å»ºæ–°çš„åˆ— 'cutoff_year' å’Œ 'sequence_length'
train_sequences['cutoff_year'] = pd.to_datetime(train_sequences['temporal_cutoff']).dt.year
train_sequences['sequence_length'] = train_sequences['sequence'].str.len()

# è®¡ç®—æ¯�å¹´å¹³å�‡åº�åˆ—é•¿åº¦
avg_sequence_length = (
    train_sequences
    .groupby('cutoff_year')
    .agg(avg_sequence_length=('sequence_length', 'mean'))
    .reset_index()
)

# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))

# ç»˜åˆ¶å®�é™…åº�åˆ—é•¿åº¦å’Œå¤šé¡¹å¼�å¹³æ»‘çº¿
sns.regplot(x='cutoff_year', y='avg_sequence_length', data=avg_sequence_length, 
            order=3, ci=None, label='Smoothed Line', color='blue')
plt.plot(avg_sequence_length['cutoff_year'], avg_sequence_length['avg_sequence_length'], 
         color='black', label='Actual Sequence Length')

# æ·»åŠ æ ‡ç­¾å’Œæ ‡é¢˜
plt.xlabel("Temporal Cutoff Year")
plt.ylabel("Average Sequence Length discovered")
plt.title("Average length of published sequence over time")
plt.legend()
plt.grid(True)
plt.show()


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# å�‡è®¾ train_sequences æ•°æ�®æ¡†ä¸­åŒ…å�« 'target_id' åˆ—
# æ��å�– pdb_id å’Œ chain_id
train_sequences[['pdb_id', 'chain_id']] = train_sequences['target_id'].str.split('_', expand=True)

# è®¡ç®—å”¯ä¸€çš„ pdb_id æ•°é‡�
pdb_id_count = train_sequences['pdb_id'].nunique()

# è®¡ç®—å”¯ä¸€çš„ chain_id æ•°é‡�
chain_id_count = train_sequences['chain_id'].nunique()

# æ‰“å�°ç»“æ�œ
print(f"There are {pdb_id_count} unique pdb_id's in the dataset.")
print(f"There are {chain_id_count} unique chain_id's in the dataset.")


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# å�‡è®¾ train_sequences æ•°æ�®æ¡†ä¸­åŒ…å�« 'target_id' åˆ—
# æ��å�– pdb_id å’Œ chain_id
train_sequences[['pdb_id', 'chain_id']] = train_sequences['target_id'].str.split('_', expand=True)

# è®¡ç®—æ¯�ä¸ª pdb_id çš„æ€»æ•°ï¼Œå¹¶ç­›é€‰å‡ºé‡�å¤�çš„ pdb_id
repeated_pdb_ids = (
    train_sequences
    .groupby('pdb_id')
    .size()
    .reset_index(name='total')
    .query('total > 1')
    .sort_values(by='total', ascending=False)
)

# è�·å�–å‰� 10 ä¸ªé‡�å¤�çš„ pdb_id
top_repeated_pdb_ids = repeated_pdb_ids.head(10)

# æ‰“å�°ç»“æ�œ
print("In total, there are", repeated_pdb_ids.shape[0], "pdb_id's that are repeated in train_sequences.")
print("Top 10 most repeated pdb_id's:")
top_repeated_pdb_ids# In total, there are 60 pdb_id's that are repeated in train_sequences, with the top 10 most repeated shown below


train_sequences[train_sequences['pdb_id']=='4V5Z']


# å�‡è®¾ train_labels æ˜¯ä¸€ä¸ª Pandas DataFrame
# æ‰“å�°è¡Œæ•°å’Œåˆ—æ•°
print(f'train_labels has {train_labels.shape[0]} rows and {train_labels.shape[1]} columns.')
# æ‰“å�°ç¼ºå¤±å€¼çš„æ€»æ•°
print(f'train_labels has {train_labels.isna().sum().sum()} NAs.')
# å¿«é€ŸæŸ¥çœ‹æ•°æ�®
train_labels.head(3)


train_labels.shape


# æŸ¥æ‰¾åŒ…å�«ç¼ºå¤±å€¼çš„è¡Œå¹¶æ˜¾ç¤ºå‰� 3 è¡Œ
na_rows = train_labels[train_labels.isna().any(axis=1)]
# æ‰“å�°å‰� 3 è¡Œ
na_rows.head(3)


na_rows.shape


# é€‰æ‹© x_1, y_1, z_1 åˆ—å¹¶è½¬æ�¢ä¸ºé•¿æ ¼å¼�
train_labels_long = train_labels[['x_1', 'y_1', 'z_1']].melt(var_name='coordinate', value_name='value')
# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))
# ç»˜åˆ¶ç®±çº¿å›¾
sns.boxplot(x='value', y='coordinate', data=train_labels_long, hue='coordinate', palette='YlGnBu', dodge=False)
# æ·»åŠ æ ‡ç­¾å’Œæ ‡é¢˜
plt.xlabel("Coordinate Values")
plt.ylabel("Coordinates")
plt.title("Boxplot of coordinate values in train_labels")
plt.legend(title='Coordinate', loc='upper right')
plt.grid(True)
plt.show()


# è®¡ç®—æ¯�ç§� resname çš„å‡ºç�°æ¬¡æ•°
resname_counts = train_labels.groupby('resname').size().reset_index(name='total')
# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))
# ç»˜åˆ¶æŸ±çŠ¶å›¾
sns.barplot(x='resname', y='total', data=resname_counts, palette='Blues')
# æ·»åŠ æŸ±å­�ä¸Šçš„è®¡æ•°æ ‡ç­¾
for index, row in resname_counts.iterrows():
    plt.text(index, row['total'], row['total'], ha='center', va='bottom')
# è®¾ç½®æ ‡ç­¾å’Œæ ‡é¢˜
plt.xlabel("Resname")
plt.ylabel("Count")
plt.title("Barplot of resname occurrences in train_labels")
# éš�è—�å›¾ä¾‹
plt.legend().set_visible(False)
# ä½¿ç”¨ç®€çº¦ä¸»é¢˜
sns.set_theme(style="whitegrid")
# æ˜¾ç¤ºå›¾å½¢
plt.show()


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# å�‡è®¾ validation_sequences æ˜¯ä¸€ä¸ª Pandas DataFrame
# æ‰“å�°è¡Œæ•°å’Œåˆ—æ•°
print(f'validation_sequences has {validation_sequences.shape[0]} rows and {validation_sequences.shape[1]} columns.')

# æ‰“å�°ç¼ºå¤±å€¼çš„æ€»æ•°
print(f'validation_sequences has {validation_sequences.isna().sum().sum()} NAs.')

# å¿«é€ŸæŸ¥çœ‹æ•°æ�®
validation_sequences.head(3)


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# å�‡è®¾ validation_sequences æ˜¯ä¸€ä¸ª Pandas DataFrame
# è®¡ç®—åº�åˆ—é•¿åº¦å¹¶åˆ›å»ºæ–°åˆ—
validation_sequences['length'] = validation_sequences['sequence'].str.len()

# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))

# ç»˜åˆ¶ç®±çº¿å›¾
sns.boxplot(x='length', data=validation_sequences, showfliers=False)

# è®¾ç½®æ ‡ç­¾å’Œæ ‡é¢˜
plt.xlabel("Sequence length")
plt.ylabel("Sequence")
plt.title("Boxplot of sequence length values (without outliers)")

# ä½¿ç”¨ç®€çº¦ä¸»é¢˜
sns.set_theme(style="whitegrid")

# æ˜¾ç¤ºå›¾å½¢
plt.show()


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# å�‡è®¾ train_sequences å’Œ validation_sequences æ˜¯ Pandas DataFrame
# è®¡ç®— train_sequences çš„ä¸­ä½�æ•°åº�åˆ—é•¿åº¦
train_sequence_med = train_sequences['sequence'].str.len().median()

# è®¡ç®— validation_sequences çš„ä¸­ä½�æ•°åº�åˆ—é•¿åº¦
valid_sequence_med = validation_sequences['sequence'].str.len().median()

# æ‰“å�°ç»“æ�œ
print(f'The train_sequences median sequence length is {train_sequence_med}.')
print(f'The validation_sequences median sequence length is {valid_sequence_med}.')


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# å�‡è®¾ validation_labels æ˜¯ä¸€ä¸ª Pandas DataFrame
# æ‰“å�°è¡Œæ•°å’Œåˆ—æ•°
print(f'validation_labels has {validation_labels.shape[0]} rows and {validation_labels.shape[1]} columns.')

# æ‰“å�°ç¼ºå¤±å€¼çš„æ€»æ•°
print(f'validation_labels has {validation_labels.isna().sum().sum()} NAs.')

print("Note validation_labels has 123 columns; train_labels has 6.")
# å¿«é€ŸæŸ¥çœ‹æ•°æ�®
validation_labels.head(3)


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# å�‡è®¾ validation_labels æ˜¯ä¸€ä¸ª Pandas DataFrame
# è®¡ç®—æ¯�ç§� resname çš„å‡ºç�°æ¬¡æ•°
resname_counts = validation_labels.groupby('resname').size().reset_index(name='total')

# è®¾ç½®ç”»å¸ƒçš„å¤§å°�
plt.figure(figsize=(15, 8))

# ç»˜åˆ¶æŸ±çŠ¶å›¾
sns.barplot(x='resname', y='total', data=resname_counts, palette='Blues', order=resname_counts.sort_values('total')['resname'])

# æ·»åŠ æŸ±å­�ä¸Šçš„è®¡æ•°æ ‡ç­¾
for index, row in resname_counts.iterrows():
    plt.text(index, row['total'], row['total'], ha='center', va='bottom')

# è®¾ç½®æ ‡ç­¾å’Œæ ‡é¢˜
plt.xlabel("Resname")
plt.ylabel("Count")
plt.title("Barplot of resname occurrences in validation_labels")

# éš�è—�å›¾ä¾‹
plt.legend().set_visible(False)

# ä½¿ç”¨ç®€çº¦ä¸»é¢˜
sns.set_theme(style="whitegrid")

# æ˜¾ç¤ºå›¾å½¢
plt.show()

# è¯´æ˜�
print("Note there are no missing or 'X' resnames.")
print("Also of note is in this dataset, 'U' occurs more frequently than 'A'. 'C' and 'G' maintain their ordering.")


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# è¯»å�–æ ·æœ¬æ��äº¤æ–‡ä»¶
sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')

# æ‰“å�°è¡Œæ•°å’Œåˆ—æ•°
print(f'There are {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns.')

# å¿«é€ŸæŸ¥çœ‹å‰� 6 è¡Œæ•°æ�®
print("Note: It is required to submit 5 sets of coordinate plots (x,y,z), per ID, resname & resid from the target_id, which is 18 columns.")
print("For rows in the submission, we need one row for each resname/resid for each target.")
print("From the sample submission, we need 2515 rows.")
sample_submission.head(6)


test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
# æ‰“å�°è¡Œæ•°å’Œåˆ—æ•°
print(f'There are {test_sequences.shape[0]} rows and {test_sequences.shape[1]} columns.')
# è¯´æ˜�
print("Reading in the test sequences, we can see we have 12 rows and the same 5 columns found in train & validation sequences.")
test_sequences.head()


test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
# è®¡ç®—æ‰€æœ‰åº�åˆ—é•¿åº¦çš„æ€»å’Œ
total_length = test_sequences['sequence'].str.len().sum()
print(f'Total length of sequences in test_sequences is {total_length}.')
# åˆ›å»ºä¸€ä¸ª DataFrame æ�¥å­˜å‚¨æ��äº¤æ•°æ�®
submission_df = pd.DataFrame(columns=['target_id', 'resname', 'resid', 'x', 'y', 'z'])  # æ ¹æ�®éœ€è¦�å®šä¹‰åˆ—
# è¯´æ˜�
print("Now, we need to create the df to house our submission.")


# è¯»å�–æµ‹è¯•åº�åˆ—æ–‡ä»¶
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
# å®šä¹‰è§£æ��å‡½æ•°
def parse_target(tmp_id, tmp_sequence):
    seq_length = len(tmp_sequence)
    tmp_df = pd.DataFrame({
        'ID': [tmp_id] * seq_length,
        'resname': list(tmp_sequence),
        'resid': range(1, seq_length + 1)
    })
    return tmp_df
# åˆ›å»ºç”¨äº�å­˜å‚¨ç»“æ�œçš„ DataFrame
test_clean = pd.DataFrame(columns=['ID', 'resname', 'resid'])
# å¯¹æ¯�ä¸ª target_id / sequence åº”ç”¨è§£æ��å‡½æ•°ï¼Œå¹¶å°†ç»“æ�œå�ˆå¹¶
for index, row in test_sequences.iterrows():
    tmp_df = parse_target(row['target_id'], row['sequence'])
    test_clean = pd.concat([test_clean, tmp_df], ignore_index=True)
# æ‰“å�°è¡Œæ•°å’Œå‰�å‡ è¡Œæ•°æ�®
print(f'There are {len(test_clean)} rows in the test_clean df.')
# è¯´æ˜�
print("That works... now let's get back to the training data, add some features, & train a model.")
test_clean.head()


# å¯¼å…¥å¿…è¦�çš„åº“
import pandas as pd

# å®šä¹‰ç‰¹å¾�å·¥ç¨‹å‡½æ•°
def feat_eng(df):
    # è®¡ç®—åº�åˆ—é•¿åº¦å�Šå�„ä¸ªæ ¸è‹·é…¸çš„è®¡æ•°
    df['seq_length'] = df['sequence'].str.len()  # åº�åˆ—é•¿åº¦
    df['A_cnt'] = df['sequence'].str.count("A")  # A çš„è®¡æ•°
    df['C_cnt'] = df['sequence'].str.count("C")  # C çš„è®¡æ•°
    df['U_cnt'] = df['sequence'].str.count("U")  # U çš„è®¡æ•°
    df['G_cnt'] = df['sequence'].str.count("G")  # G çš„è®¡æ•°
    
    # è®¡ç®—äºŒè�šä½“çš„è®¡æ•°
    df['AC_cnt'] = df['sequence'].str.count("AC")  # AC çš„è®¡æ•°
    df['AU_cnt'] = df['sequence'].str.count("AU")  # AU çš„è®¡æ•°
    df['AG_cnt'] = df['sequence'].str.count("AG")  # AG çš„è®¡æ•°
    df['CA_cnt'] = df['sequence'].str.count("CA")  # CA çš„è®¡æ•°
    df['CU_cnt'] = df['sequence'].str.count("CU")  # CU çš„è®¡æ•°
    df['CG_cnt'] = df['sequence'].str.count("CG")  # CG çš„è®¡æ•°
    df['UA_cnt'] = df['sequence'].str.count("UA")  # UA çš„è®¡æ•°
    df['UC_cnt'] = df['sequence'].str.count("UC")  # UC çš„è®¡æ•°
    df['UG_cnt'] = df['sequence'].str.count("UG")  # UG çš„è®¡æ•°
    df['GA_cnt'] = df['sequence'].str.count("GA")  # GA çš„è®¡æ•°
    df['GC_cnt'] = df['sequence'].str.count("GC")  # GC çš„è®¡æ•°
    df['GU_cnt'] = df['sequence'].str.count("GU")  # GU çš„è®¡æ•°
    df['AA_cnt'] = df['sequence'].str.count("AA")  # AA çš„è®¡æ•°
    df['CC_cnt'] = df['sequence'].str.count("CC")  # CC çš„è®¡æ•°
    df['UU_cnt'] = df['sequence'].str.count("UU")  # UU çš„è®¡æ•°
    df['GG_cnt'] = df['sequence'].str.count("GG")  # GG çš„è®¡æ•°
    
    # è�·å�–åº�åˆ—çš„èµ·å§‹å’Œç»“æ�Ÿæ ¸è‹·é…¸
    df['begin_seq'] = df['sequence'].str[0]  # åº�åˆ—çš„ç¬¬ä¸€ä¸ªæ ¸è‹·é…¸
    df['end_seq'] = df['sequence'].str[-1]  # åº�åˆ—çš„æœ€å��ä¸€ä¸ªæ ¸è‹·é…¸

    # åˆ é™¤ä¸�éœ€è¦�çš„åˆ—
    df = df.drop(columns=['sequence', 'temporal_cutoff', 'description', 'all_sequences'])
    return df

# åº”ç”¨ç‰¹å¾�å·¥ç¨‹å‡½æ•°
train_sequences = feat_eng(train_sequences)

# ä»� ID åˆ›å»º target_id ä»¥æ–¹ä¾¿ä¸� meta_train æ•°æ�®å�ˆå¹¶
train_labels['target_id'] = train_labels['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))

# å�ˆå¹¶è®­ç»ƒæ ‡ç­¾å’Œè®­ç»ƒåº�åˆ—ä»¥åˆ›å»ºæœ€ç»ˆçš„è®­ç»ƒæ•°æ�®
train_data_clean = pd.merge(train_labels, train_sequences, on="target_id", how="left")

# ç”¨æ¯�ä¸ª target_id å’Œ resname çš„ç»„å¹³å�‡å€¼å¡«è¡¥ç¼ºå¤±çš„ x_1, y_1, z_1 å€¼
train_data_clean[['x_1', 'y_1', 'z_1']] = train_data_clean.groupby(['target_id', 'resname'])[['x_1', 'y_1', 'z_1']].transform(
    lambda x: x.fillna(x.mean())
)

# åˆ é™¤ä»»ä½•ä»�ç„¶å­˜åœ¨ç¼ºå¤±å€¼çš„è¡Œ (~ 1979 è¡Œ)
train_data_clean = train_data_clean.dropna()

# æ‰“å�°æ•°æ�®é›†ä¸­ç¼ºå¤±å€¼çš„æ€»æ•°
print(f'There are {train_data_clean.isna().sum().sum()} NA\'s in the dataset!')


# ä»�è®­ç»ƒæ•°æ�®é›†ä¸­é€‰æ‹©ç‰¹å¾�åˆ—ï¼Œæ�’é™¤ x_1, y_1, z_1 å’Œ ID åˆ—
features = train_data_clean.drop(columns=['x_1', 'y_1', 'z_1', 'ID']).columns.tolist()

# æ‰“å�°ç‰¹å¾�åˆ—å��
print(features)


import h2o
from h2o.estimators import H2OXGBoostEstimator

# Initialize H2O
h2o.init()

# Convert train_data_clean to H2OFrame
train_hframe = h2o.H2OFrame(train_data_clean)
train_hframe['resname'] = train_hframe['resname'].asfactor()
train_hframe['begin_seq'] = train_hframe['begin_seq'].asfactor()
train_hframe['end_seq'] = train_hframe['end_seq'].asfactor()

# Train model for X
train_hframe_x = train_hframe.drop(['ID', 'y_1', 'z_1'])
splits_x = train_hframe_x.split_frame(ratios=[0.8], seed=1)
train_x = splits_x[0]
valid_x = splits_x[1]

xgb_x = H2OXGBoostEstimator(
    nfolds=5,
    seed=1,
    booster='gbtree',
    ntrees=200,
    max_depth=15,
    keep_cross_validation_predictions=True
)

xgb_x.train(x=features, y='x_1', training_frame=train_x, validation_frame=valid_x)
print(xgb_x.mse(train=True, valid=True))

# Train model for Y
train_hframe_y = train_hframe.drop(['ID', 'x_1', 'z_1'])
splits_y = train_hframe_y.split_frame(ratios=[0.8], seed=1)
train_y = splits_y[0]
valid_y = splits_y[1]

xgb_y = H2OXGBoostEstimator(
    nfolds=5,
    seed=1,
    booster='gbtree',
    ntrees=200,
    max_depth=15,
    keep_cross_validation_predictions=True
)

xgb_y.train(x=features, y='y_1', training_frame=train_y, validation_frame=valid_y)
print(xgb_y.mse(train=True, valid=True))

# Train model for Z
train_hframe_z = train_hframe.drop(['ID', 'y_1', 'x_1'])
splits_z = train_hframe_z.split_frame(ratios=[0.8], seed=1)
train_z = splits_z[0]
valid_z = splits_z[1]

xgb_z = H2OXGBoostEstimator(
    nfolds=5,
    seed=1,
    booster='gbtree',
    ntrees=200,
    max_depth=15,
    keep_cross_validation_predictions=True
)

xgb_z.train(x=features, y='z_1', training_frame=train_z, validation_frame=valid_z)
print(xgb_z.mse(train=True, valid=True))

# Shutdown H2O
h2o.shutdown()


# åˆ›å»ºæµ‹è¯•åº�åˆ—çš„ç‰¹å¾�
test_sequences = feat_eng(test_sequences)

# å°†ä¹‹å‰�åˆ›å»ºçš„ test_clean ä¸� test_sequences å�ˆå¹¶
test_data_clean = test_clean.merge(test_sequences, left_on="ID", right_on="target_id", how="left")

# å°† resnameã€�begin_seq å’Œ end_seq åˆ—è½¬æ�¢ä¸ºå› å­�
test_data_clean['resname'] = test_data_clean['resname'].astype('category')  # è½¬æ�¢ä¸ºç±»åˆ«
test_data_clean['begin_seq'] = test_data_clean['begin_seq'].astype('category')  # è½¬æ�¢ä¸ºç±»åˆ«
test_data_clean['end_seq'] = test_data_clean['end_seq'].astype('category')  # è½¬æ�¢ä¸ºç±»åˆ«



# å¯¹ X è¿›è¡Œé¢„æµ‹
preds_x = xgb_x.predict(test_hframe.drop(['ID']))  # åˆ é™¤ ID åˆ—è¿›è¡Œé¢„æµ‹
print(preds_x)  # æ‰“å�°é¢„æµ‹ç»“æ�œ

# å¯¹ Y è¿›è¡Œé¢„æµ‹
preds_y = xgb_y.predict(test_hframe.drop(['ID']))  # åˆ é™¤ ID åˆ—è¿›è¡Œé¢„æµ‹
print(preds_y)  # æ‰“å�°é¢„æµ‹ç»“æ�œ

# å¯¹ Z è¿›è¡Œé¢„æµ‹
preds_z = xgb_z.predict(test_hframe.drop(['ID']))  # åˆ é™¤ ID åˆ—è¿›è¡Œé¢„æµ‹
print(preds_z)  # æ‰“å�°é¢„æµ‹ç»“æ�œ

# å�¯é€‰ï¼šå½“å®Œæˆ�æ‰€æœ‰æ“�ä½œå��å…³é—­ H2Oï¼ˆå¦‚æ�œéœ€è¦�ï¼‰
# h2o.shutdown()  # é‡Šæ”¾èµ„æº�


import pandas as pd

# Create the submission DataFrame
submission = pd.DataFrame({
    'ID': test_data_clean['ID'],
    'resname': test_data_clean['resname'],
    'resid': test_data_clean['resid']
})

# Add predictions to the submission DataFrame
submission = pd.concat([
    submission,
    preds_x.as_data_frame(),
    preds_y.as_data_frame(),
    preds_z.as_data_frame(),
    preds_x.as_data_frame(),
    preds_y.as_data_frame(),
    preds_z.as_data_frame(),
    preds_x.as_data_frame(),
    preds_y.as_data_frame(),
    preds_z.as_data_frame(),
    preds_x.as_data_frame(),
    preds_y.as_data_frame(),
    preds_z.as_data_frame()
], axis=1)

# Assuming sample_submission has the correct column names
submission.columns = sample_submission.columns

# Optionally, save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)


# Update ID to match submission ID column
submission['ID'] = submission['ID'].astype(str) + '_' + submission['resid'].astype(str)

# Create a sort_order column within the sample_submission
sample_submission['sort_order'] = range(1, len(sample_submission) + 1)

# Merge submission with sample_submission to include sort_order
submission = submission.merge(
    sample_submission[['ID', 'sort_order']],
    on='ID',
    how='left'
)

# Sort the submission by sort_order and drop the sort_order column
submission = submission.sort_values(by='sort_order').drop(columns=['sort_order']).reset_index(drop=True)

# Optionally, save the updated submission DataFrame to a CSV file
submission.to_csv('updated_submission.csv', index=False)


submission.head(3)


submission.to_csv('submission.csv',index = False)

