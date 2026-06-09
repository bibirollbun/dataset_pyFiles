from pathlib import Path
input_dir = Path("/kaggle/input")
file_paths = [f for f in input_dir.rglob("*") if f.is_file()]




import torch
import numpy as np
from pathlib import Path
from fastai.vision.all import *

# Set seed for reproducibility
set_seed(42)

# Define the dataset path
path = Path("/kaggle/input/paddy-disease-classification")

# Use fastai's convenience method
print(path.ls())   # lists files/folders inside /kaggle/input




!ls '/kaggle/input/paddy-disease-classification/train_images'
trn_path = path/'train_images'



trn_path = path/'train_images'

# Recursively get all images inside class subfolders
files = get_image_files(trn_path)

print(len(files))   # should print 10407
print(files[:5])    # peek at first 5



img = PILImage.create(files[0])
print(img.size)
img.to_thumb(400)


from fastcore.parallel import *

def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)
learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()


learn.lr_find(suggest_funcs=(valley, slide))


learn.fine_tune(3, 0.01)


ss = pd.read_csv(path/'sample_submission.csv')
ss


tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)


probs,_,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs


dls.vocab


mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results


ss['label'] = results
ss.to_csv('submission.csv', index=False)
!head subm.csv

