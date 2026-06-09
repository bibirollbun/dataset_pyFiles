import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib as plt
from fastai.vision.all import *
from sklearn.model_selection import KFold
import torch 
from PIL import Image
import re 
import gc 

NOTEBOOK_NAME = 'kaggle'


train_path = Path('/kaggle/input/paddy-disease-classification/train_images')
test_path = Path('/kaggle/input/paddy-disease-classification/test_images')


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def train(arch, item, batch, accum = 2, epochs = 5, lr = 0.01):
    dls = ImageDataLoaders.from_folder(train_path, valid_pct=0.2, item_tfms=item, batch_tfms=batch,  bs=64//accum, num_workers=4)
    cbs = GradientAccumulation(64) if accum else []
    learn = vision_learner(dls, arch, metrics=error_rate, cbs = cbs).to_fp16()
    learn.fine_tune(epochs, base_lr=lr)
    learn.arch = arch
    _, targs = learn.get_preds(dl=learn.dls.valid)
    tta_preds,_ = learn.tta(dl=learn.dls.valid)
    print(f'TTA error rate:{error_rate(tta_preds, targs)}')
    return learn


def save_logits(learn, description):
	classes = learn.dls.vocab
	dls_test = learn.dls.test_dl(sorted(get_image_files(test_path)))
	logits, _  = learn.tta(dl=dls_test)
	# #preds
	sample_sub = pd.read_csv('/kaggle/input/paddy-disease-classification/sample_submission.csv', index_col=0)
	# sample_sub['label'] = classes[preds]
	# sample_sub.to_csv(f'subs/{NOTEBOOK_NAME}_{learn.arch}_{description}.csv')
    #logits
	logits = pd.DataFrame(logits)
	logits.index, logits.columns = sample_sub.index, classes
	logits.to_csv(f'/kaggle/working/{NOTEBOOK_NAME}_{learn.arch}_{description}.csv')


def preds_from_logits(filename):
	sample_sub = pd.read_csv('sample_submission.csv', index_col=0)
	sample_sub['label'] = pd.read_csv(f'/kaggle/working/{filename}',index_col=0).idxmax(axis=1)
	sample_sub.to_csv(f'/kaggle/working/{filename}')


augment_transform = aug_transforms( size=224,  min_scale=0.75,  max_rotate=15,  max_lighting=0.2,  max_zoom=1.1, max_warp=0.1, p_affine=0.75, p_lighting=0.75)


# learn = train('beitv2_base_patch16_224.in1k_ft_in22k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()


clear_memory()


# learn = train('coatnet_0_rw_224.sw_in1k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()


clear_memory()


# learn = train('coatnet_1_rw_224.sw_in1k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()



clear_memory()


# learn = train('convnext_base.fb_in22k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()



clear_memory()


# learn = train('eva02_small_patch14_224.mim_in22k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()



clear_memory()


# learn = train('tf_efficientnetv2_m.in21k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()



clear_memory()


# learn = train('vit_base_r50_s16_224.orig_in21k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()



clear_memory()


# learn = train('deit_base_patch16_224.fb_in1k', item=Resize((320,240)), batch=augment_transform, epochs=15, lr =0.003)
# save_logits(learn, 'default_augment')
# del learn
# clear_memory()



clear_memory()


weights = [3,2,1,2,1,1,1,1]
logits_path = sorted(Path('/kaggle/input/paddy-doctor').ls())
logits = weights[0] * pd.read_csv(logits_path[0], index_col = 0)
for path, w in zip(logits_path[1:], weights[1:]):
    logits += w * pd.read_csv(path, index_col = 0)
logits /=sum(weights)


sample_submition = pd.read_csv('/kaggle/input/paddy-disease-classification/sample_submission.csv', index_col=0)
sample_submition['label'] = logits.idxmax(axis ='columns')


sample_submition.head()


sample_submition.to_csv('/kaggle/working/submission.csv')

