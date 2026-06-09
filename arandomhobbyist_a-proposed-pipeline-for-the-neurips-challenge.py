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


import os
print(os.listdir('/kaggle/working'))



!python --version



import numpy, scipy, sklearn
print(numpy.__version__, scipy.__version__, sklearn.__version__)


import os
wheel_dir = '/kaggle/input/rdkit-wheels'
print(os.listdir(wheel_dir))



!pip install --no-index $wheel_dir/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl



!python --version
!uname -a
!pip install torch torchvision torchaudio --quiet



!python --version
!pip show torch



!pip install torch-scatter --no-binary=torch-scatter
!pip install torch-sparse --no-binary=torch-sparse
!pip install torch-cluster --no-binary=torch-cluster
!pip install torch-spline-conv --no-binary=torch-spline-conv
!pip install torch-geometric




import sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)



!mkdir -p /kaggle/working/wheels
!pip wheel --no-binary=torch-scatter torch-scatter -w /kaggle/working/wheels
!pip wheel --no-binary=torch-sparse torch-sparse -w /kaggle/working/wheels
!pip wheel --no-binary=torch-cluster torch-cluster -w /kaggle/working/wheels
!pip wheel --no-binary=torch-spline-conv torch-spline-conv -w /kaggle/working/wheels
!pip wheel --no-binary=torch-geometric torch-geometric -w /kaggle/working/wheels
!ls -l /kaggle/working/wheels
!pip install --no-index --find-links=/kaggle/working/wheels torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric



!mkdir -p /kaggle/working/wheels
!cp /root/.cache/pip/wheels/b8/d4/0e/a80af2465354ea7355a2c153b11af2da739cfcf08b6c0b28e2/*.whl /kaggle/working/wheels/
!cp /root/.cache/pip/wheels/75/e2/1e/299c596063839303657c211f587f05591891cc6cf126d94d21/*.whl /kaggle/working/wheels/



!pip install /kaggle/working/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl



!find / -name '*.whl' 2>/dev/null



import numpy, PIL, rdkit, torch_geometric
print("Numpy:", numpy.__version__)
print("Pillow:", PIL.__version__)
print("RDKit:", rdkit.__version__)
print("Torch Geometric:", torch_geometric.__version__)
print("Torch Molecule:", torch_molecule.__version__)




import sys, platform
print("Python:", sys.version)
print("Platform:", platform.system(), platform.machine())


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

def assemble_dataframes():
    csv_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    train_df = pd.read_csv(csv_path)

    train_df['is_valid'] = train_df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    train_df = train_df[train_df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)


# 1. split off 20% for dev_test
    temp_df, dev_test = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,  # for reproducibility
        shuffle=True
    )

# 2. split the remaining 80% into 75% train / 25% valid → 0.6 / 0.2 overall
    dev_train, dev_val = train_test_split(
        temp_df,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of the original
        random_state=42,
        shuffle=True
    )

    return dev_train, dev_val, train_df, temp_df, dev_test

dev_train, dev_val, train_df, temp_df, dev_test = assemble_dataframes()

print(f"Total rows:   {len(train_df)}")
print(f"Dev train:    {len(dev_train)} ({len(dev_train)/len(train_df):.2%})")
print(f"Dev valid:    {len(dev_val)} ({len(dev_val)/len(train_df):.2%})")
print(f"Dev test:     {len(dev_test)} ({len(dev_test)/len(train_df):.2%})")
print(f"Polymer example:{dev_train['SMILES'].to_list()[:3]}")
print(f"Columns:{dev_train.columns}")



import os
print(os.listdir('/kaggle/input'))
import sys
sys.path.append('/kaggle/input/dimenetmolecularpredictor/dimenet_molecular_predictor.py')
from dimenet_molecular_predictor import DimeNetMolecularPredictor




import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def dimenet_training(dev_train, dev_val):
    X_train = dev_train['SMILES'].to_list()
    y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
    X_val = dev_val['SMILES'].to_list()
    y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

    gnn = DimeNetMolecularPredictor(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    avg_val_loss = gnn.fit()

    print('Average Validation Loss:')
    print(avg_val_loss)

    return gnn


def create_submission(gnn):
    import os
    if gnn is not None:
        sample_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
        print(sample_sub.head())

        test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
        X_test = test_df['SMILES'].to_list()

        preds = gnn.predict(X_test)

        submission_df = sample_sub.copy()
        submission_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = preds

        print('submission_df', submission_df.head())

        # Save CSV file
        submission_path = '/kaggle/working/submission.csv'
        submission_df.to_csv(submission_path, index=False)

        # Check if file saved successfully
        if os.path.isfile(submission_path):
            print(f"✅ Submission file saved at {submission_path}")
        else:
            print(f"❌ Submission file NOT found at {submission_path}")

if __name__ == '__main__':
    import os
    try:
        gnn = dimenet_training(dev_train=dev_train, dev_val=dev_val)
        print(f'GNN Type: {type(gnn)}')
        create_submission(gnn=gnn)

        print("Files in /kaggle/working:", os.listdir('/kaggle/working'))

    except Exception as e:
        print("❌ Failed to create submission file:", e)

    
    

