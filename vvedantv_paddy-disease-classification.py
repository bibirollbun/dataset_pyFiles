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


# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *


comp = 'paddy-disease-classification'

path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')


path


from fastai.vision.all import *
set_seed(42)

path.ls()


trn_path = path/'train_images'
files = get_image_files(trn_path)


img = PILImage.create(files[0])
print(img.size)
img.to_thumb(128)
img.show()


from fastcore.parallel import *

def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
                                   item_tfms=Resize(480, method='squish'),
                                   batch_tfms=aug_transforms(size=128, min_scale=0.75))
dls.show_batch(max_n=6)


learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path=',').to_fp16()


learn.lr_find(suggest_funcs=(valley, slide))


learn.fine_tune(3, 0.01)


ss = pd.read_csv(path/'sample_submission.csv')
ss


tst_files = get_image_files(path/'test_images').sorted() 
tst_dl = dls.test_dl(tst_files)


probs,_,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs


dls.vocab


preds_labels = [dls.vocab[i] for i in idxs]



ss['label'] = preds_labels
ss.to_csv('submission.csv', index=False)
!head submission.csv




