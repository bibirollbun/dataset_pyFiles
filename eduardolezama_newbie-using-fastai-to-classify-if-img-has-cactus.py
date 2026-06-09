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


from fastai.vision.all import *
from pathlib import Path

path = Path("/kaggle/input/aerial-cactus-identification")
# Show the files we've got in our working dir
print(" | ".join([str(file.name) for file in path.ls()]))


import zipfile
#Direct to the train.zip file
zip_files = ["train.zip","test.zip"]

#unzip in the same directory
for folder in zip_files:
    zip_path = path / folder
    outp_folder = Path("/kaggle/working")
    #Unzip the files
    with zipfile.ZipFile(zip_path,"r") as zip_ref:
        zip_ref.extractall(outp_folder)

print("Files unzipped")
#Since we won't be using input dir as our main dir, let's use working dir
path = Path("/kaggle/working/")


train_path = path/"train"
test_path = path/"test"

#let's get a list of all images in the train folder using get_image_files
files = get_image_files(train_path)
files_tst = get_image_files(test_path)





img = PILImage.create(files[0])
print(img.size)
img.to_thumb(32) #32 because all the images are 32x32, but we will check it


from fastcore.parallel import *
import pandas as pd

def creator(o): return PILImage.create(o).size
sizes = parallel(creator, files, n_workers=8)

#Create a series to see the values of all the diferent sizes using pandas
#This will take less than a minute but is good to check
pd.Series(sizes).value_counts()


sizes_test = parallel(creator,files_tst, n_workers=8)
pd.Series(sizes_test).value_counts()


#The classes are not in the file directoy parent folder, they are in a csv file
labels_path = Path("/kaggle/input/aerial-cactus-identification/train.csv")
#Load the csv file as pandas df
df = pd.read_csv(labels_path)
df.head()


dls = ImageDataLoaders.from_df(
    df,                      # df with labels
    path=train_path,         # path of images
    valid_pct=0.2,           # % for validation
    seed=69,                 # seed of randomness
    label_col='has_cactus',  # label col title
    fn_col='id',             # file names column title
    item_tfms=Resize(32, method='squish'),            # resizing of imgs
    batch_tfms=aug_transforms(size=32, min_scale=1))  # data augmentation



#show a batch
dls.show_batch(max_n=6)



learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path=".").to_fp16()


learn.lr_find(suggest_funcs=(valley, slide))


#Train the model
learn.fine_tune(4, 0.01)


from sklearn.metrics import roc_auc_score
def roc_auc(preds, targs):
    try:
        # Fastai work with tensors and scikitlearn with numpy array.
        # we have to move the tensors to array.
        preds = preds[:, 1].detach().cpu().numpy()  # This is the probability of 1
        targs = targs.cpu().numpy()                # True label
        return roc_auc_score(targs, preds)
    except ValueError:  # Error in case there is only 1 class.
        return None




learn = vision_learner(dls, 'resnet26d', metrics=[error_rate, roc_auc])
learn.fine_tune(4, 0.01)



#How is the sample submission csv?
sample = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv")
sample.head(10)


tst_dl = dls.test_dl(files_tst)


probs, _,= learn.get_preds(dl=tst_dl)


#Extract positive probabilities, which is for each item the second value.
positive_probs = probs[:,1]



filenames = tst_dl.items



# Create the submission DataFrame
submission = pd.DataFrame({
    'id': [f.name for f in filenames],  # Extract only the file name
    'has_cactus': positive_probs
})

submission.head(5)


submission.to_csv('submission.csv', index=False)

