!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!(. "$HOME/.cargo/env"; pip install -Uqq fastkaggle fastai fastcore huggingface_hub "timm==0.6.13")



from fastkaggle import *

comp = 'paddy-disease-classification'
path = setup_comp(comp)  # , install='fastai "timm>=0.6.2.dev0"')
from fastai.vision.all import *
set_seed(42)

tst_files = get_image_files(path/'test_images').sorted()


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


train('convnext_small_in22k', 128, epochs=1, accum=1, finetune=False)


import gc
def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()


report_gpu()


train('convnext_small_in22k', 128, epochs=1, accum=2, finetune=False)
report_gpu()


train('convnext_small_in22k', 128, epochs=1, accum=4, finetune=False)
report_gpu()


train('convnext_large_in22k', 224, epochs=1, accum=2, finetune=False)
report_gpu()


train('convnext_large_in22k', (320,240), epochs=1, accum=2, finetune=False)
report_gpu()


train('vit_large_patch16_224', 224, epochs=1, accum=2, finetune=False)
report_gpu()


train('swinv2_large_window12_192_22k', 192, epochs=1, accum=2, finetune=False)
report_gpu()


train('swin_large_patch4_window7_224', 224, epochs=1, accum=2, finetune=False)
report_gpu()


res = 640,480


models = {
    'convnext_large_in22k': {
        (Resize(res), (320,224)),
    }, 'vit_large_patch16_224': {
        (Resize(480, method='squish'), 224),
        (Resize(res), 224),
    }, 'swinv2_large_window12_192_22k': {
        (Resize(480, method='squish'), 192),
        (Resize(res), 192),
    }, 'swin_large_patch4_window7_224': {
        (Resize(res), 224),
    }
}


trn_path = path/'train_images'


import datetime

tta_res = []
report_gpu()
for i, (arch, details) in enumerate(models.items()):
    print(f"arch {i} of {len(models)} of size {len(details)}")
    for item, size in details:
        print('---',arch)
        print(size)
        print(item.name)
        # Epochs was 12, I do not have the patience!
        tta_res.append(train(arch, size, item=item, accum=4, epochs=1))
        report_gpu()
        print(datetime.datetime.now().isoformat())
        print()
        #gc.collect()
        #torch.cuda.empty_cache()


save_pickle('tta_res.pkl', tta_res)


tta_prs = first(zip(*tta_res))


tta_prs += tta_prs[1:3]


avg_pr = torch.stack(tta_prs).mean(0)
avg_pr.shape


dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=224, min_scale=0.75))


idxs = avg_pr.argmax(dim=1)
vocab = np.array(dls.vocab)
ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = vocab[idxs]
ss.to_csv('submission.csv', index=False)


if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'part 3 v2', comp)


# This is what I use to push my notebook from my home PC to Kaggle

if not iskaggle:
    push_notebook('jhoward', 'scaling-up-road-to-the-top-part-3',
                  title='Scaling Up: Road to the Top, Part 3',
                  file='10-scaling-up-road-to-the-top-part-3.ipynb',
                  competition=comp, private=False, gpu=True)

