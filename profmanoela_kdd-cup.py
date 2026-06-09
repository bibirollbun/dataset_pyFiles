import pandas as pd
import os

dir_path = '/kaggle/input/bi-master-24-2-deteccao-de-intrusao-de-rede/'

data = pd.read_csv(os.path.join(dir_path, 'treino.csv'), index_col=0)
data.head()


data.out.value_counts()




