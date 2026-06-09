# Apparently this notebook requires cargo ... copied from another notebook
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y


# install fastkaggle and other deps if not available
!(. "$HOME/.cargo/env"; pip install -Uq fastkaggle fastai huggingface_hub "timm>=0.6.2.dev0")

from fastkaggle import *


comp = 'paddy-disease-classification'

path = setup_comp(comp) # ,install='fastai "timm>=0.6.2.dev0"' )


path


from fastai.vision.all import *
set_seed(42)

path.ls()


trn_path = path/'train_images'
files = get_image_files(trn_path)


img = PILImage.create(files[0])
print(img.size)
img.to_thumb(128)


from fastcore.parallel import *

def f(o): 
    return PILImage.create(o).size

sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()


dls = ImageDataLoaders.from_folder(
    trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)


learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()


learn.lr_find(suggest_funcs=(valley, slide))


learn.fine_tune(3, 10**-2)


ss = pd.read_csv(path/'sample_submission.csv')
ss


tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)


probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs


dls.vocab


mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results


ss['label'] = results
ss.to_csv('subm.csv', index=False)
!head subm.csv


if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'initial rn26d 128px', comp)


if not iskaggle:
    push_notebook('jhoward', 'first-steps-road-to-the-top-part-1',
                  title='First Steps: Road to the Top, Part 1',
                  file='first-steps-road-to-the-top-part-1.ipynb',
                  competition=comp, private=False, gpu=True)




