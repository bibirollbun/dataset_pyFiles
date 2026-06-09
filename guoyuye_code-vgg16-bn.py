import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
from os.path import expanduser, join, dirname, basename


from fastai.vision.all import *


class cfg(dict):
    # dot.notation access to dictionary attributes
    # Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    seed = 1
    
    # you have 4 sizes: 28, 64, 128, 224
    data_root = '/kaggle/input/cp-3501-retinamnist-v-202501/medmnist_kaggle_v240608b/medmnist_kaggle_v240608b'
    data_path = join(data_root, 'retinamnist_224')
    train_images_dir = join(data_path, 'train_images')
    test_images_dir = join(data_path, 'test_images')
    train_img_size = 224   # 32, 64, 128, 224, 256, or try matching to pretrained model's native image size.
    batch_size = 32 
    
    item_tfms = RandomResizedCrop(train_img_size, min_scale=0.5)
    batch_tfms = aug_transforms(mult=2, flip_vert=True, max_rotate=360.)
    
    sample_submission_path = '/kaggle/input/cp-3501-retinamnist-v-202501/sample_submission.csv'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # this may crash in some models!
#     torch.backends.cudnn.deterministic = False  # change to False if crashing
    torch.backends.cudnn.benchmark = False
    
set_seed(cfg.seed) 


# Create a DataBlock
# data_block0 is our data block without any agmentations
data_block0 = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=cfg.seed),
    get_y=parent_label,
#     item_tfms=Resize(224),
    item_tfms=Resize(cfg.train_img_size),
#     batch_tfms=aug_transforms()  # let's connect augmentations later
)

# Create DataLoaders
# dls = data_block.dataloaders(data_path, bs=64)
dls0 = data_block0.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)

# Let's view a raining batch
dls0.train.show_batch(max_n=8, nrows=2)


data_block = data_block0.new(item_tfms=cfg.item_tfms, batch_tfms=cfg.batch_tfms)  

dls = data_block.dataloaders(cfg.train_images_dir, bs=cfg.batch_size)
dls.train.show_batch(max_n=4, nrows=1, unique=True)  # True= will show the same image


# from collections import Counter
# labels = [parent_label(o) for o in get_image_files(cfg.train_images_dir)]
# print(Counter(labels))


# Create a learner
learn = vision_learner(dls, vgg16_bn, metrics=accuracy)

# Find an optimal learning rate
# learn.lr_find()
learn.lr_find(suggest_funcs=(valley, slide))


# print(dls.vocab)


# from collections import Counter
# import torch
# from torch.nn import CrossEntropyLoss
# label_counts = Counter([parent_label(o) for o in get_image_files(cfg.train_images_dir)])

# # Make sure the order is consistent with dls.vocab
# counts = torch.tensor([label_counts[v] for v in dls.vocab], dtype=torch.float)
# weights = 1.0 / counts
# class_weights = (weights / weights.sum()).to(dls.device)

# loss_func = CrossEntropyLoss(weight=class_weights)
# learn.loss_func = loss_func


?learn.lr_find


lr = 0.006


?learn.fine_tune


learn.fine_tune(10, base_lr=lr)


# print(len(learn.dls.valid_ds))  # See if the validation set has data


# preds, targs = learn.get_preds()
# print(len(preds), len(targs))


# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# preds, targs = learn.get_preds()
# pred_labels = preds.argmax(dim=1)

# cm = confusion_matrix(targs, pred_labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap="Blues", xticks_rotation=45)
# plt.title("Confusion Matrix")
# plt.show()


# interp = ClassificationInterpretation.from_learner(learn)

# # Use the interpret() method to get preds, targs
# preds, targs = learn.get_preds(dl=learn.dls.valid)
# decoded = preds.argmax(dim=1)

# print("preds:", preds.shape)
# print("targs:", targs.shape)
# print("decoded:", decoded.shape)


# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Get validation set predictions manually from learner
# preds, targs = learn.get_preds(dl=learn.dls.valid)
# decoded = preds.argmax(dim=1)

# # Drawing the confusion matrix
# cm = confusion_matrix(targs, decoded)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=learn.dls.vocab)
# disp.plot(cmap="Blues", xticks_rotation=45)
# plt.title("Confusion Matrix")
# plt.show()



interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


import glob
test_files = glob.glob(join(cfg.test_images_dir, '*.png'))
print(test_files[:10])


# test_dl = learn.dls.test_dl(test_files)

# preds, _ = learn.get_preds(dl=test_dl)
# decoded = preds.argmax(dim=1)

# df = pd.DataFrame({'image_id': [basename(f) for f in test_files], 'label': decoded})
# df


from tqdm import tqdm
from os.path import basename
# from tqdm.notebook import tqdm
# Iterate over the list of image paths and make predictions
preds = []
pred_ids = []
for img_path in tqdm(test_files):  # tqdm not working?  learn was breaking it https://github.com/fastai/fastai/issues/3366
# for img_path in test_files:
    img_id = basename(img_path)
    pred_ids += [img_id]
    # https://github.com/fastai/fastai/issues/3366
    with learn.no_bar(), learn.no_logging():
        res = learn.predict(img_path)
    pred_label, val, probs = res
    preds += [pred_label]
#     print(res)
    
df = pd.DataFrame({'image_id':pred_ids})
df['label'] = preds
df


# let's sort it as per given submission sample
sub = pd.read_csv(cfg.sample_submission_path)
sub


# Failure to overwrite the submission with the results predicted by the new model
df_sorted = df.set_index("image_id").loc[sub["image_id"]].reset_index()
df_sorted.to_csv("submission.csv", index=False)


learn.summary()




