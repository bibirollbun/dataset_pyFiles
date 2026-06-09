!pip install fastparquet -q
import pandas as pd
df_fc = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv', engine='pyarrow')
df_fc.to_parquet(f"/kaggle/working/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.parquet", compression=None, engine="fastparquet")
df_fc = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv', engine='pyarrow')
df_fc.to_parquet(f"/kaggle/working/TEST_FUNCTIONAL_CONNECTOME_MATRICES.parquet", compression=None, engine="fastparquet")

