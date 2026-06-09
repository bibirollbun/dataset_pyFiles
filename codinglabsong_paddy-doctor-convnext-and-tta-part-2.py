# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -q fastkaggle

from fastkaggle import *

comp = 'paddy-disease-classification'
path = setup_comp(comp, install='"fastcore>=1.4.5" "fastai>=2.7.1" "timm>=0.6.2.dev0"')

from fastai.vision.all import *
set_seed(42)


# where we would store our resized images
trn_path = Path('small')


path.ls()


# resizing to 192x256px images
resize_images(path/"train_images", dest=trn_path, max_size=256, recurse=True)


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
                                  item_tfms=Resize((256,192)))
dls.show_batch(max_n=6)


import torch

def train(arch, item, batch, epochs=5, lr=0.01, bs=64):
    # define model
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42, 
                                       item_tfms=item, batch_tfms=batch, 
                                       bs=bs)
    learn = vision_learner(dls, arch, metrics=error_rate, path='.').to_fp16()

    # check the number of GPUs available and print it
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    # if more than one GPU is available, wrap the model using DataParallel
    if num_gpus > 1:
        learn.model = torch.nn.DataParallel(learn.model)

    # train the model
    learn.fine_tune(epochs, lr)
    return learn


def find_lr(arch, item, batch, bs=64):
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42, 
                                       item_tfms=item, batch_tfms=batch, 
                                       bs=bs)
    learn = vision_learner(dls, arch, metrics=error_rate, path='.').to_fp16()

    learn.lr_find(suggest_funcs=(valley, slide))


arch = 'resnet26'
find_lr(arch, item=Resize(192, method='squish'), batch=aug_transforms(size=128, min_scale=0.75), 
        bs=256)


learn = train(arch, item=Resize(192, method='squish'), 
              batch=aug_transforms(size=128, min_scale=0.75), 
              bs=256)


arch = "convnext_base_in22k"


find_lr(arch, item=Resize(192, method='squish'), batch=aug_transforms(size=128, min_scale=0.75), bs=256)


learn = train(arch, item=Resize(192, method='squish'),
             batch=aug_transforms(size=128, min_scale=0.75), 
             bs=256)


learn = train(arch, item=Resize(192), 
              batch=aug_transforms(size=128, min_scale=0.75),
              bs=256)


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
                                  item_tfms=Resize(192, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros))
dls.show_batch(max_n=3)


learn = train(arch, item=Resize((256,192), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
            batch=aug_transforms(size=(171,128), min_scale=0.75),
            bs=256)


valid = learn.dls.valid
preds,targs = learn.get_preds(dl=valid)


error_rate(preds, targs)


learn.dls.train.show_batch(max_n=6, unique=True)


tta_preds,_ = learn.tta(dl=valid)


error_rate(tta_preds, targs)


trn_path = path/'train_images'


find_lr(arch, item=Resize((480,360), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros), 
        batch=aug_transforms(size=(256,192), min_scale=0.75), 
        bs=128)


learn = train(arch, 
             item=Resize((480,360), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
             batch=aug_transforms(size=(256,192), min_scale=0.75),
             epochs=12, lr=0.08, bs=128)


tta_preds,targs = learn.tta(dl=learn.dls.valid)
error_rate(tta_preds,targs)


tst_files = get_image_files(path/"test_images").sorted()
tst_dl = learn.dls.test_dl(tst_files)

# Then do TTA
preds,_ = learn.tta(dl=tst_dl)


idxs = preds.argmax(dim=1)


vocab = np.array(learn.dls.vocab)
results = pd.Series(vocab[idxs], name="idxs")


ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = results
ss.to_csv('submission.csv', index=False)
!head submission.csv


# submit if not on kaggle website
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('submission.csv', 'convnext_base_in22k 256x192 12 epochs tta', comp)

