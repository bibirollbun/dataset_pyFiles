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


#install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *


#Getting set up

comp = 'paddy-disease-classification'

path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')


path


from fastai.vision.all import *


path.ls()


#looking at the data
trn_path = path/'train_images'
files = get_image_files(trn_path)


img = PILImage.create(files[0])
print(img)
img.to_thumb(128)


#looks like the images might be 480*640. Let's check all their sizes. This is faster if we do it in parallel

from fastcore.parallel import *

def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=8)

pd.Series(sizes).value_counts()


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
                                  item_tfms=Resize(480, method="squish"),
                                  batch_tfms=aug_transforms(size=128, min_scale=0.75))


dls.show_batch(max_n=6)


#Our First Model:
#The model resent26d which is the fastest resolution-independent model which gets into the top-15 lists there

learn = vision_learner(dls, 'resnet26d', metrics = error_rate, path=",").to_fp16()


#let us see the learnigng rate finder shows:

learn.lr_find(suggest_funcs=(valley, slide))


learn.fine_tune(3, 0.01)


dls


#We are now ready to build our first submission. Let us take a look at the sample Kaggle provided to see what it needs to look like
#Submitting to Kaggle

ss = pd.read_csv(path/'sample_submission.csv')
ss


#So, we need a CSV containing all the test images, in alphabetical order and the predicted lable for each one. We can create the needed test set using fastai like so

tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)


probs, _, idxs = learn.get_preds(dl = tst_dl, with_decoded=True)


probs


idxs


dls.vocab


mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results


#kaggle expects the submission as a CSV file so let's save it and check the first few lines

ss["label"] = results
ss.to_csv("submission.csv", index=False)
!head submission.csv


if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'intital rn26d 128px', comp)




