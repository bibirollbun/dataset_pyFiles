# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -q fastkaggle

from fastkaggle import *


comp = 'paddy-disease-classification'
path = setup_comp(comp, install='"fastcore>=1.4.5" "fastai>=2.7.1" "timm>=0.6.2.dev0"')
from fastai.vision.all import *
set_seed(42)


trn_path = Path('small')
resize_images(path/'train_images', dest=trn_path, recurse=True, max_size=256)


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
                      item_tfms = Resize((256, 192)))


dls.show_batch(max_n=3)


def train(arch, item, batch, epochs=5, seed=None):
    dls = ImageDataLoaders.from_folder(trn_path, item_tfms=item,
                                     batch_tfms=batch, epochs=epochs, seed=seed, valid_pct=0.2)
    learn = vision_learner(dls, arch, metrics=error_rate).to_fp16()
    learn.fine_tune(epochs, 0.01)
    return learn


learn = train('resnet26d', item=Resize(192), batch=aug_transforms(size=128, min_scale=0.75))


arch = 'convnext_small_in22k'


learn_big = train(arch, item=Resize(192), batch=aug_transforms(size=128, min_scale=0.75))


learn_big = train(arch, item=Resize(192, method='squish'),
                  batch=aug_transforms(size=128, min_scale=0.75))


# Time to try padding
learn_big = train(arch, item=Resize((256, 192), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                  batch= aug_transforms(size=128, min_scale=0.75), seed=42)


valid = learn_big.dls.valid
preds,targs = learn_big.get_preds(dl=valid)


error_rate(preds, targs)


learn_big.dls.train.show_batch(max_n=6, unique=True)
# seems like the parameter unique is used to show images that come from the same original image


tta_preds,_ = learn_big.tta(dl=valid)


error_rate(tta_preds, targs)


trn_path = path/'train_images'
learner = train(arch, epochs=12, seed=42,
                item=Resize((480, 360), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                batch=aug_transforms(size=(256,192), min_scale=0.75))


tst_files = get_image_files(path/'test_images').sorted()
test_dls = learner.dls.test_dl(tst_files)


preds,_ = learner.tta(dl=test_dls)


idxs = preds.argmax(dim=1)


vocab = np.array(learn.dls.vocab)
labels = pd.Series(vocab[idxs], name='idxs')


ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = labels
ss.to_csv('submission.csv', index=False)
!head submission.csv




