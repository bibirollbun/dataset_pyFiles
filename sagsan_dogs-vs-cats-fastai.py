import os
from pathlib import Path
from fastai.vision.all import untar_data, URLs, get_image_files
import zipfile

# boilerplate to setup path to images on either a local (fastai download) or Kaggle run

LOCAL_RUN = os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost') == 'Localhost'

path = None
if LOCAL_RUN:
    path = untar_data(URLs.PETS)/'images'
else:
    input = Path("/kaggle/input/dogs-vs-cats")
    os.mkdir("images")
    with zipfile.ZipFile(input/"train.zip", 'r') as zf:
        zf.extractall("images")
    with zipfile.ZipFile(input/"test1.zip", 'r') as zf:
        zf.extractall("images")
    path = Path("./images")

# make sure path is setup and your files exist.
assert len(get_image_files(path/'train')) > 1000
assert len(get_image_files(path/'test1')) > 1000


## this is more boilerplate code just used for development so I don't have to work 
# with the entire set

import random

# get a subset of train file so it runs faster, by default return everything
def get_train_filenames(subset_size = -1): 
    fnames = get_image_files(path/'train')

    if subset_size < 0 or len(fnames) < subset_size:
        return fnames
    else:
        random.seed(42)
        return random.sample(fnames, subset_size)


from fastai.vision.all import ImageDataLoaders, Resize

def get_label(filename):
    if "dog" in filename:
        return 'dog'
    elif "cat" in filename:
        return 'cat'
    else:
        raise ValueError("Neither dog or cat specified in filename")

dls = ImageDataLoaders.from_name_func(   #from_name creates dataset based on filename
                      path=".", # work dir
                      fnames=get_train_filenames(), # get training file names
                      label_func=get_label, #labelling function based onf ilename
                      valid_pct=0.2, #hold for validation
                      seed=42,
                      item_tfms=Resize(192)) # resize basedon fastAI recommendation

dls.show_batch()


from fastai.vision.all import vision_learner, resnet18, error_rate
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3, base_lr=1e-4) # use 3 epochs, lr was calculated after running learn.lr_find()



learn.show_results()


#Another thing that is useful is an interpretation object, it can show us where the model made the worse predictions:
from fastai.vision.all import Interpretation
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))


# Export the model for re-use (aka upload to HuggingFace)
learn.export('model.pkl')


test_dl = dls.test_dl(get_image_files(path/"test1"))
preds, _ = learn.get_preds(dl=test_dl)


import pandas as pd

pred_probs, pred_idxs = preds.max(dim=1)

# We loop through the length of the predictions, 
#and drop everything in a df with the correct column names
df = pd.DataFrame(columns=["id", "label"])
for i in range(len(test_dl.items)):
    file_path = test_dl.items[i]

    file_name = file_path.stem #just filename without 

    # The prediction data is also at index 'i'
    confidence = pred_probs[i].item()
    df.loc[len(df)] = [file_name, pred_idxs[i].item()]
    



df_sorted = df.sort_values(by='id') # sort so its easier to manually spot check
df.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

