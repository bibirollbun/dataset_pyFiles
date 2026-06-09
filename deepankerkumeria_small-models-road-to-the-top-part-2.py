# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install fastai==2.7.0 fastcore==1.5.
    !pip install -Uq fastkaggle

from fastkaggle import *


comp = 'paddy-disease-classification'
path = setup_comp(comp, install='"fastai==2.7.0" "timm<0.4.12"')
from fastai.vision.all import *
set_seed(42)


import timm
timm --version


trn_path = Path('sml')


resize_images(path/'train_images', dest=trn_path, max_size=256, recurse=True)


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize((256,192)))

dls.show_batch(max_n=3)


def train(arch, item, batch, epochs=5):
    dls = ImageDataLoaders.from_folder(trn_path, seed=42, valid_pct=0.2, item_tfms=item, batch_tfms=batch)
    learn = vision_learner(dls, arch, metrics=error_rate).to_fp16()
    learn.fine_tune(epochs, 0.01)
    return learn


import timm
learn = train('resnet26d', item=Resize(192),
              batch=aug_transforms(size=128, min_scale=0.75))


!pip install timm==0.6.12
import timm
# !pip show timm
# List available ConvNeXt models
models = timm.list_models("")
print(models)


# arch = 'convnext_small_in22k'



# learn = train(arch, item=Resize(224, method='padding'),
#               batch=aug_transforms(size=224, min_scale=0.75))
# This wasn't the original code from copied notebook but I was trying to make it work with various model versions available in my version of timm library
arch= 'swin_small_patch4_window7_224' 
learn = train(arch, item=Resize(192),
              batch=aug_transforms(size=224, min_scale=0.75))


learn = train(arch, item=Resize(192),
              batch=aug_transforms(size=224, min_scale=0.75))


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(224, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros))
dls.show_batch(max_n=3)


learn = train(arch, item=Resize((224,224), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
      batch=aug_transforms(size=(224,224), min_scale=0.75))


valid = learn.dls.valid
preds,targs = learn.get_preds(dl=valid)


error_rate(preds, targs)


learn.dls.train.show_batch(max_n=6, unique=True)


tta_preds,_ = learn.tta(dl=valid)


error_rate(tta_preds, targs)


trn_path = path/'train_images'


learn = train(arch, epochs=12,
              item=Resize((480, 360), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
              batch=aug_transforms(size=(224,224), min_scale=0.75))


tta_preds,targs = learn.tta(dl=learn.dls.valid)
error_rate(tta_preds, targs)


tst_files = get_image_files(path/'test_images').sorted()
tst_dl = learn.dls.test_dl(tst_files)


preds,_ = learn.tta(dl=tst_dl)


idxs = preds.argmax(dim=1)


vocab = np.array(learn.dls.vocab)
results = pd.Series(vocab[idxs], name="idxs")


ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = results
ss.to_csv('subm.csv', index=False)
!head subm.csv


if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'convnext small 256x192 12 epochs tta', comp)


# This is what I use to push my notebook from my home PC to Kaggle

if not iskaggle:
    push_notebook('jhoward', 'small-models-road-to-the-top-part-2',
                  title='Small models: Road to the Top, Part 2',
                  file='small-models-road-to-the-top-part-2.ipynb',
                  competition=comp, private=True, gpu=True)

