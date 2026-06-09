# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *


comp = 'paddy-disease-classification'
from pathlib import Path
# path = setup_comp(comp, install='"timm>=0.6.2.dev0"')
path = Path("/kaggle/input/paddy-disease-classification")
from fastai.vision.all import *
set_seed(42)
path.ls()
# tst_files = get_image_files(path/'test_images').sorted()


df = pd.read_csv(path/'train.csv')
df.label.value_counts()


trn_path = path/'train_images'/'bacterial_panicle_blight'


def train(arch, size, item=Resize(480, method='squish'), accum=1, finetune=True, epochs=12):
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, item_tfms=item,
        batch_tfms=aug_transforms(size=size, min_scale=0.75), bs=64//accum)
    cbs = GradientAccumulation(64) if accum else []
    learn = vision_learner(dls, arch, metrics=error_rate, cbs=cbs).to_fp16()
    if finetune:
        learn.fine_tune(epochs, 0.01)
        return learn.tta(dl=dls.test_dl(tst_files))
    else:
        learn.unfreeze()
        learn.fit_one_cycle(epochs, 0.01)


train('convnext_small.fb_in22k', 128, epochs=1, accum=1, finetune=False)


import gc
def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()


report_gpu()


train('convnext_small.fb_in22k', 128, epochs=1, accum=2, finetune=False)
report_gpu()


train('convnext_small_in22k', 128, epochs=1, accum=4, finetune=False)
report_gpu()


train('convnext_large.fb_in22k', 224, epochs=1, accum=2, finetune=False)
report_gpu()


train('convnext_large_in22k', (320,240), epochs=1, accum=2, finetune=False)
report_gpu()


train('vit_large_patch16_224', 224, epochs=1, accum=2, finetune=False)
report_gpu()


# train('swinv2_large_window12_192.ms_in22k', 192, epochs=1, accum=2, finetune=False)
# report_gpu()


# train('swin_large_patch4_window7_224', 224, epochs=1, accum=2, finetune=False)
# report_gpu()





res = 640,480
models = {
    'convnext_large_in22k': {
        (Resize(res), (320,224)),
    }, 'vit_large_patch16_224': {
        (Resize(480, method='squish'), 224),
        (Resize(res), 224),
    # }, 'swinv2_large_window12_192_22k': {
    #     (Resize(480, method='squish'), 192),
    #     (Resize(res), 192),
    # }, 'swin_large_patch4_window7_224': {
    #     (Resize(res), 224),
    }
}


trn_path = path/'train_images'


tta_res = []

for arch,details in models.items():
    for item,size in details:
        print('---',arch)
        print(size)
        print(item.name)
        tta_res.append(train(arch, size, item=item, accum=2)) #, epochs=1))
        gc.collect()
        torch.cuda.empty_cache()


save_pickle('tta_res.pkl', tta_res)
tta_prs = first(zip(*tta_res))
tta_prs += tta_prs[1:3]
print(tta_prs.shape)
avg_pr = torch.stack(tta_prs).mean(0)
print(avg_pr.shape)

