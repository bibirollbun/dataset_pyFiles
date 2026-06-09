!pip install -q fastkaggle fastai timm==0.9.16



from fastkaggle import setup_comp
import gc, torch



comp = 'paddy-disease-classification'
path = setup_comp(comp, install='"fastcore>=1.4.5" "fastai>=2.7.1" "timm>=0.6.2.dev0"')
from fastai.vision.all import *
from fastai.losses import LabelSmoothingCrossEntropy



from fastai.vision.all import *
set_seed(42)



train_path = path/'train_images'
print('class folders:', train_path.ls()[:5])
print('total images :', len(get_image_files(train_path)))


dls = ImageDataLoaders.from_folder(
    train_path, valid_pct=0.2, seed=42,
    item_tfms = Resize(256,  method='pad', pad_mode=PadMode.Zeros),
    batch_tfms= [*aug_transforms(size=128, min_scale=0.75),
                 Normalize.from_stats(*imagenet_stats)],
    bs = 96
)

learn = vision_learner(
          dls, 'convnext_small_in22k',
          metrics=error_rate,
          model_dir=Path('/kaggle/working/models'),
        loss_func = LabelSmoothingCrossEntropy(0.1)

       ).to_fp16()
learn.lr_find(suggest_funcs=(valley, slide))
learn.fine_tune(1, 1e-3)
learn.save('cns_128')
gc.collect(); torch.cuda.empty_cache()


dls = ImageDataLoaders.from_folder(
    train_path, valid_pct=0.2, seed=42,
    item_tfms = Resize(448, method='pad', pad_mode=PadMode.Zeros),
    batch_tfms= [*aug_transforms(size=224, min_scale=0.75),
                 Normalize.from_stats(*imagenet_stats)],
    bs = 64
)

learn.dls = dls
learn.load('cns_128')
learn.freeze()
learn.lr_find(suggest_funcs=(valley, slide))
learn.fine_tune(4, 1.5e-3)
learn.save('cns_224')
gc.collect(); torch.cuda.empty_cache()


dls = ImageDataLoaders.from_folder(
    train_path, valid_pct=0.2, seed=42,
    item_tfms = Resize((480, 360),
                       method='pad', pad_mode=PadMode.Zeros),
    batch_tfms= [*aug_transforms(size=(320,240), min_scale=0.75),
                 Normalize.from_stats(*imagenet_stats)],
    bs = 40
)

learn.dls = dls
learn.load('cns_224')
learn.unfreeze()
learn.lr_find(suggest_funcs=(valley, slide))
learn.fit_one_cycle(3, 5e-5)
learn.save('cns_320')
gc.collect(); torch.cuda.empty_cache()


dls = ImageDataLoaders.from_folder(
    train_path, valid_pct=0.2, seed=42,
    item_tfms = Resize((480,360), method='pad', pad_mode=PadMode.Zeros),
    batch_tfms= [*aug_transforms(size=(480,360), min_scale=0.75),
                 Normalize.from_stats(*imagenet_stats)],
    bs = 24
)

learn.dls = dls
learn.load('cns_320')
learn.unfreeze()
learn.lr_find(suggest_funcs=(valley, slide))
learn.fit_one_cycle(10, 1e-5)
learn.save('cns_480')


tta_preds,targs = learn.tta(dl=learn.dls.valid)
error_rate(tta_preds, targs)


from fastai.vision.all import get_image_files
import pandas as pd, numpy as np

tst_files = get_image_files(path/'test_images').sorted()
tst_dl    = learn.dls.test_dl(tst_files)

tta_preds, _ = learn.tta(dl=tst_dl)

idxs = tta_preds.argmax(dim=1).cpu().numpy()

vocab  = np.array(learn.dls.vocab)
labels = vocab[idxs]

ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = labels

ss.to_csv('submission.csv', index=False)

print(ss.head())

