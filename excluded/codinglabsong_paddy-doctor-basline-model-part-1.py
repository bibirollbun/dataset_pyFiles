# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *

# install the data and needed packages
comp = 'paddy-disease-classification'
path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')

from fastai.vision.all import *
set_seed(42) # for reproducibility


path.ls()


# get all the images
trn_path = path/'train_images'
files = get_image_files(trn_path)

# lets take a look at one example image
img = PILImage.create(files[1])
print(img.size)
img.to_thumb(128)


from fastcore.parallel import *

# create a function that returns the image size for a file
def f(o): return PILImage.create(o).size

# use parallel() method to create a list of sizes for each image.
# this also has the advantage of doing the job in parallel on CPU (we use 8 workers here)
sizes = parallel(f, files, n_workers=8)

# lets see the first fize image sizes
print(sizes[:5]) 


pd.Series(sizes).value_counts()


# our DataLoaders object 
dls = ImageDataLoaders.from_folder(
    trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))


# Shows iamges AFTER both item_tfms and batch_tfms
dls.show_batch(max_n=6)


# Define Learner
learn = vision_learner(dls, 'resnet26', metrics=error_rate, path='.').to_fp16()


import torch

print(f"Available GPUs: {torch.cuda.device_count()}")


# enable multi-GPU for this model by wrapping it in a DataParallel object provided by pytorch
learn.model = torch.nn.DataParallel(learn.model)


# find the best learning rate
learn.lr_find(suggest_funcs=(valley, slide))


learn.fine_tune(3, 0.01)


ss = pd.read_csv(path/'sample_submission.csv')
ss


# use get_image_files to find and list all image files in alphabetical order
tst_files = get_image_files(path/'test_images').sorted()

# this test DataLoader applies the same transformations and settings as used for training/validation, allowing you to run predictions on your test set.
tst_dl = dls.test_dl(tst_files)


probs,_,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
probs


idxs


dls.vocab


vocab = np.array(dls.vocab)
results = pd.Series(vocab[idxs], name="idxs")
results


ss['label'] = results
ss.to_csv('submission.csv', index=False)
!head submission.csv


# submit if not on kaggle website
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('submission.csv', 'convnext small 256x192 12 epochs tta', comp)




