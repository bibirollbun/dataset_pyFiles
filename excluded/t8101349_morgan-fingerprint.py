# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install rdkit


import duckdb
import pandas as pd
from tqdm import tqdm
import numpy as np # linear algebra
from rdkit import Chem
from rdkit.Chem import AllChem



import pyarrow.parquet as pq
import pandas as pd
import gc  # å¼•å…¥å�ƒåœ¾å›�æ”¶æ¨¡çµ„

filename = "/kaggle/input/leash-BELKA/train.parquet"
columns_to_read = ["molecule_smiles", "protein_name", "binds"]

batch_size = 60000    # æ¯�æ¬¡è®€å�– 60,000 ç­†
target_rows = 1200000  # æ¯�è¼ªå„²å­˜ 1,200,000 ç­†
total_batches = 15    # ç¸½å…±åŸ·è¡Œ 15 æ¬¡
total_rows = 0        # è¨˜éŒ„ç•¶å‰�ç´¯ç©�ç­†æ•¸

parquet_file = pq.ParquetFile(filename)

# ç�²å�–ç¸½è¡Œçµ„æ•¸é‡�
num_row_groups = parquet_file.num_row_groups
print(f"Total row groups in file: {num_row_groups}")

# è¨ˆç®—æ¯�æ¬¡è®€å�–å¤šå°‘è¡Œçµ„
row_groups_per_batch = target_rows // batch_size


# é–‹å§‹é€²è¡Œæ‰¹æ¬¡è™•ç�†
for i in range(total_batches):
    chunks = []
    current_rows = 0  # æ¯�è¼ªçš„è¨ˆæ•¸å™¨
    batch_start_row_group = i * row_groups_per_batch  # ç›´æ�¥ä¾�åº�å�–
    batch_end_row_group = min(batch_start_row_group + row_groups_per_batch, num_row_groups)

    if batch_end_row_group >= num_row_groups:
        batch_end_row_group = num_row_groups  # é�¿å…�è¶…å‡ºè¡Œçµ„ç¯„åœ�

    print(f"âœ… ç¬¬ {i+1} æ¬¡è™•ç�†ï¼šå¾�è¡Œçµ„ {batch_start_row_group} åˆ°è¡Œçµ„ {batch_end_row_group}")

    # ä½¿ç”¨ pyarrow çš„ ParquetFile ç›´æ�¥è®€å�–æŒ‡å®šç¯„åœ�çš„è¡Œçµ„
    for row_group_idx in range(batch_start_row_group, batch_end_row_group):
        try:
            batch = parquet_file.read_row_groups([row_group_idx], columns=columns_to_read)
            chunk = batch.to_pandas()

            # æª¢æŸ¥æ˜¯å�¦æœ‰è³‡æ–™
            if not chunk.empty:
                chunks.append(chunk)
                current_rows += len(chunk)
                total_rows += len(chunk)

            # å¦‚æ�œè®€å�–åˆ°æŒ‡å®šç¯„åœ�çš„è³‡æ–™ï¼Œå°±å�œæ­¢
            if total_rows >= target_rows * (i + 1):
                break  # å¦‚æ�œå·²ç¶“è®€åˆ°è©²æ‰¹æ¬¡çš„çµ�å°¾å°±å�œæ­¢

        except Exception as e:
            print(f"âš ï¸� è®€å�–è¡Œçµ„ {row_group_idx} æ™‚ç™¼ç”ŸéŒ¯èª¤: {e}")

    if chunks:  # ç¢ºä¿�æœ‰è³‡æ–™æ‰�é€²è¡Œå�ˆä½µ
        # å�ˆä½µ DataFrame
        batch_df = pd.concat(chunks, ignore_index=True)

        # å­˜æˆ� parquetï¼Œæ¯�æ¬¡éƒ½å­˜ä¸�å�Œçš„æª”æ¡ˆ
        output_filename = f"/kaggle/working/train_part{i+1}.parquet"
        final_df.to_parquet(output_filename, index=False)

        print(f"âœ… ç¬¬ {i+1} æ¬¡å­˜æª”ï¼š{len(final_df)} ç­†ï¼Œå·²ç´¯ç©� {total_rows} ç­†")

        # æ¸…ç�†ç„¡ç”¨çš„è®Šæ•¸ï¼Œé‡‹æ”¾è¨˜æ†¶é«”
        del batch_df, batch_pivot, smiles_df, final_df
        gc.collect()  # åŸ·è¡Œå�ƒåœ¾å›�æ”¶
    else:
        print(f"âš ï¸� ç¬¬ {i+1} æ¬¡è™•ç�†æœªè®€å�–åˆ°ä»»ä½•è³‡æ–™ï¼Œè·³é��è©²æ‰¹æ¬¡ã€‚")


# 15 å€‹ Parquet æª”æ¡ˆ
parquet_files = [f"/kaggle/working/train_part{i+1}.parquet" for i in range(0, 15)]

# åˆ�å§‹åŒ– DuckDB é€£ç·š
con = duckdb.connect()

# å­˜æ”¾æ‰€æœ‰æ‰¹æ¬¡çš„ DataFrame
all_samples = []

# é€�å€‹è™•ç�† 15 å€‹æª”æ¡ˆ
for i, file in enumerate(parquet_files):
    print(f"ğŸ“‚ æ­£åœ¨è™•ç�†æª”æ¡ˆ: {file}")

    df = con.query(f"""(SELECT * FROM parquet_scan('{file}')
                            WHERE bind = 0
                            ORDER BY random()
                            LIMIT 15000)
                            UNION ALL
                            (SELECT * FROM parquet_scan('{file}')
                            WHERE bind = 1 
                            ORDER BY random()
                            LIMIT 5000)""").df()

# å„²å­˜è©²æ‰¹æ¬¡çµ�æ�œ
    output_filename = f"/kaggle/working/sampled_test_part{i+1}.parquet"
    df.to_parquet(output_filename, index=False)
    print(f"âœ… å·²å„²å­˜æŠ½æ¨£çµ�æ�œ: {output_filename}ï¼ˆå…± {len(df)} ç­†ï¼‰")

    all_samples.append(df)

# å�ˆä½µæ‰€æœ‰çµ�æ�œ
final_test_df = pd.concat(all_samples, ignore_index=True)

# å„²å­˜ç¸½å�ˆä½µçš„ Parquet
final_test_output = "/kaggle/working/sampled_train_all.parquet"
final_test_df.to_parquet(final_test_output, index=False)
print(f"ğŸ�¯ å…¨éƒ¨ 15 å€‹æª”æ¡ˆå·²è™•ç�†å®Œç•¢ï¼Œæœ€çµ‚å�ˆä½µæª”æ¡ˆ: {final_test_output}ï¼ˆå…± {len(final_test_df)} ç­†ï¼‰")

# é—œé–‰ DuckDB
con.close()


# è¨­å®šæª”æ¡ˆè·¯å¾‘
train_path = '/kaggle/working/sampled_train_all.parquet'

# å»ºç«‹ DuckDB é€£ç·š
con = duckdb.connect()

# ä½¿ç”¨é€²åº¦æ¢�ä¾†é¡¯ç¤ºé€²åº¦
with tqdm(total=2, desc="Processing Data") as pbar:
    # æŸ¥è©¢ç¬¬ä¸€éƒ¨åˆ†æ•¸æ“š
    df_part1 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 0
                              ORDER BY random()
                              LIMIT 150000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

    # æŸ¥è©¢ç¬¬äºŒéƒ¨åˆ†æ•¸æ“š
    df_part2 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 1
                              ORDER BY random()
                              LIMIT 50000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

# å�ˆä½µå…©éƒ¨åˆ†æ•¸æ“š
df = pd.concat([df_part1, df_part2], ignore_index=True)

# éš¨æ©Ÿæ´—ç‰Œæ•¸æ“šï¼ˆfrac=1 è¡¨ç¤ºä¿�æŒ�å�Ÿå§‹å¤§å°�ï¼Œshuffle æ•´å€‹ DataFrameï¼‰
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# é—œé–‰é€£ç·š
con.close()


# # ç¢ºèª�æ•¸æ“š
print(df.head())


def smiles_to_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)

# å°� "molecule_smiles" æ¬„ä½�é€²è¡Œè½‰æ�›ä¸¦é¡¯ç¤ºé€²åº¦æ¢�
tqdm.pandas(desc="Transforming molecule_smiles")
df["molecule_smiles"] = df["molecule_smiles"].progress_apply(lambda x: smiles_to_morgan_fingerprint(x))

# å°� protein æ¬„ä½�é€²è¡Œ One-Hot Encoding
protein_one_hot = pd.get_dummies(df["protein_name"], prefix="protein").astype(int)

# å�ˆä½µ One-Hot çµ�æ�œ
df_one_hot = pd.concat([df, protein_one_hot], axis=1)

# å�ˆä½µéœ€è¦�çš„æ¬„ä½�ï¼šmolecule_smiles, binds, å’Œç¶“é�� One-Hot Encoding çš„ protein
df_one_hot = pd.concat([df_one_hot[["id", "molecule_smiles", "binds"]], protein_one_hot], axis=1)


df_one_hot


# # å�¯é�¸ï¼šç§»é™¤å�Ÿå§‹ protein æ¬„ä½�
# df1_one_hot.drop("protein_name", axis=1, inplace=True)

# # æª¢è¦–è™•ç�†å¾Œæ•¸æ“š
# print(df_one_hot.head())

# åƒ…ä¿�ç•™éœ€è¦�çš„æ¬„ä½�
columns_to_keep = ["id", "molecule_smiles", "binds"] + protein_one_hot.columns.tolist()
df_filtered = df_one_hot[columns_to_keep]

# å„²å­˜è™•ç�†å¾Œçš„æ•¸æ“š
output_filename = "train_transformed_morgan(150k,50k).parquet"
df_filtered.to_parquet(output_filename, index=False)

# output_filename = f"test_transformed_morgan(10k,10k).parquet"
# df.to_parquet(output_filename, index=False)


morganfile = '/kaggle/working/train_transformed_morgan(150k,50k).parquet'
morgan = pd.read_parquet(morganfile)
morgan


topolfile = '/kaggle/input/dataset1/Data_Transformed__TopologicalFingerprint(100k100k)/train_transformed__topological(100k,100k).parquet'
topol = pd.read_parquet(topolfile)
topol


# è¨­å®šæª”æ¡ˆè·¯å¾‘
train_path = '/kaggle/input/leash-BELKA/train.parquet'

# å»ºç«‹ DuckDB é€£ç·š
con = duckdb.connect()

# ä½¿ç”¨é€²åº¦æ¢�ä¾†é¡¯ç¤ºé€²åº¦
with tqdm(total=2, desc="Processing Data") as pbar:
    # æŸ¥è©¢ç¬¬ä¸€éƒ¨åˆ†æ•¸æ“š
    df_part1 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 0
                              ORDER BY random()
                              LIMIT 100000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

    # æŸ¥è©¢ç¬¬äºŒéƒ¨åˆ†æ•¸æ“š
    df_part2 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 1
                              ORDER BY random()
                              LIMIT 100000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

# å�ˆä½µå…©éƒ¨åˆ†æ•¸æ“š
df = pd.concat([df_part1, df_part2], ignore_index=True)

# éš¨æ©Ÿæ´—ç‰Œæ•¸æ“šï¼ˆfrac=1 è¡¨ç¤ºä¿�æŒ�å�Ÿå§‹å¤§å°�ï¼Œshuffle æ•´å€‹ DataFrameï¼‰
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# é—œé–‰é€£ç·š
con.close()


print(df.head())


def smiles_to_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)

# å°� "molecule_smiles" æ¬„ä½�é€²è¡Œè½‰æ�›ä¸¦é¡¯ç¤ºé€²åº¦æ¢�
tqdm.pandas(desc="Transforming molecule_smiles")
df["molecule_smiles"] = df["molecule_smiles"].progress_apply(lambda x: smiles_to_morgan_fingerprint(x))

# å°� protein æ¬„ä½�é€²è¡Œ One-Hot Encoding
protein_one_hot = pd.get_dummies(df["protein_name"], prefix="protein").astype(int)

# å�ˆä½µ One-Hot çµ�æ�œ
df_one_hot = pd.concat([df, protein_one_hot], axis=1)

# å�ˆä½µéœ€è¦�çš„æ¬„ä½�ï¼šmolecule_smiles, binds, å’Œç¶“é�� One-Hot Encoding çš„ protein
df_one_hot = pd.concat([df_one_hot[["id", "molecule_smiles", "binds"]], protein_one_hot], axis=1)


df_one_hot


# # å�¯é�¸ï¼šç§»é™¤å�Ÿå§‹ protein æ¬„ä½�
# df1_one_hot.drop("protein_name", axis=1, inplace=True)

# # æª¢è¦–è™•ç�†å¾Œæ•¸æ“š
# print(df_one_hot.head())

# åƒ…ä¿�ç•™éœ€è¦�çš„æ¬„ä½�
columns_to_keep = ["id", "molecule_smiles", "binds"] + protein_one_hot.columns.tolist()
df_filtered = df_one_hot[columns_to_keep]

# å„²å­˜è™•ç�†å¾Œçš„æ•¸æ“š

output_filename = "train_transformed_morgan(100k,100k).parquet"
df_filtered.to_parquet(output_filename, index=False)


morganfile = '/kaggle/input/train1/train_transformed_morgan(100k100k).parquet'
morgan = pd.read_parquet(morganfile)
morgan


morganfile = '/kaggle/input/testset/test_transformed__morgan(180k20k).parquet'
morgan = pd.read_parquet(morganfile)
morgan

