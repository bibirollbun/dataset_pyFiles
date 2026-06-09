import os
import subprocess
from pathlib import Path

req = Path('/kaggle/input/mixture9/other/default/1/requirements.txt')
print('requirements.txt exists:', req.exists())
if req.exists():
    subprocess.run(['python', '-m', 'pip', 'install', '-q', '-r', str(req)], check=True)
    print('Installed requirements.txt')
else:
    print('Using Kaggle preinstalled packages')


import os
import glob
import shutil
from pathlib import Path
import pandas as pd

print('Kaggle:', bool(os.environ.get('KAGGLE_URL_BASE')))
print('Working directory:', os.getcwd())
print('Input roots:', glob.glob('/kaggle/input/*')[:10])

if not Path('kaggle_diabetes2.py').exists():
    candidates = glob.glob('/kaggle/input/mixture9/other/default/1/kaggle_diabetes2.py', recursive=True)
    if candidates:
        shutil.copy(candidates[0], 'kaggle_diabetes2.py')
        print('Copied script from:', candidates[0])
    else:
        raise FileNotFoundError('kaggle_diabetes2.py not found. Add it to the notebook or attach it as a Kaggle Dataset.')

print('Script path:', Path('kaggle_diabetes2.py').resolve())


TRAIN_PATH = '/kaggle/input/playground-series-s5e12/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
META_PATH = '/kaggle/input/meta-dataset/diabetes_dataset.csv'

print('TRAIN exists:', Path(TRAIN_PATH).exists(), TRAIN_PATH)
print('TEST exists:', Path(TEST_PATH).exists(), TEST_PATH)
print('META exists:', Path(META_PATH).exists(), META_PATH)

if not Path(TRAIN_PATH).exists() or not Path(TEST_PATH).exists():
    train_candidates = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    test_candidates = glob.glob('/kaggle/input/**/test.csv', recursive=True)
    if train_candidates and test_candidates:
        TRAIN_PATH = train_candidates[0]
        TEST_PATH = test_candidates[0]
        print('Auto-selected train:', TRAIN_PATH)
        print('Auto-selected test:', TEST_PATH)
    else:
        raise FileNotFoundError('Could not find train.csv/test.csv under /kaggle/input')

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
display(train_df.head())
display(test_df.head())
print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)


import subprocess
from pathlib import Path

DEVICE = 'auto'
N_SPLITS = 5
N_REPEATS = 1
HOLDOUT_SIZE = 0.0
META_EXISTS = Path(META_PATH).exists()
USE_META = False
TEST_CHUNK_SIZE = 50000

cmd = [
    'python', 'kaggle_diabetes2.py',
    '--train-path', TRAIN_PATH,
    '--test-path', TEST_PATH,
    '--meta-path', META_PATH,
    '--n-splits', str(N_SPLITS),
    '--n-repeats', str(N_REPEATS),
    '--device', DEVICE,
    '--holdout-size', str(HOLDOUT_SIZE),
    '--test-chunk-size', str(TEST_CHUNK_SIZE),
    '--output-dir', '/kaggle/working',
    '--submission-path', 'submission.csv',
    '--oof-path', 'oof_predictions.csv'
]

if (not USE_META) or (not META_EXISTS):
    cmd.append('--no-meta')

print(' '.join(cmd))
subprocess.run(cmd, check=True)


sub_path = Path('/kaggle/working/submission.csv')
print('Submission exists:', sub_path.exists(), str(sub_path))
sub = pd.read_csv(sub_path)
display(sub.head())
print('Submission shape:', sub.shape)
print('Columns:', list(sub.columns))

