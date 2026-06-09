!pip uninstall -y fastai
!pip install fastai==1.0.61 
import fastai
print(fastai.__version__)
import warnings, torch
warnings.filterwarnings("ignore", category=FutureWarning)
import fastai
print("Fastai version:", fastai.__version__)
print("Torch version:", torch.__version__) 


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

import os
base_dir = '../input/histopathologic-cancer-detection/'
print(os.listdir(base_dir))


# Data paths
train_path = '/kaggle/input/histopathologic-cancer-detection/train/'
test_path = '/kaggle/input/histopathologic-cancer-detection/test/'
# Read CSV file
train = pd.read_csv("/kaggle/input/histopathologic-cancer-train-test-index/train_split.csv")
test = pd.read_csv("/kaggle/input/histopathologic-cancer-train-test-index/test_split.csv")


import random
ORIGINAL_SIZE = 96      # original size of the images - do not change

# AUGMENTATION VARIABLES
CROP_SIZE = 90          # final size after crop
RANDOM_ROTATION = 3    # range (0-180), 180 allows all rotation variations, 0=no change
RANDOM_SHIFT = 2        # center crop shift in x and y axes, 0=no change. This cannot be more than (ORIGINAL_SIZE - CROP_SIZE)//2 
RANDOM_BRIGHTNESS = 7  # range (0-100), 0=no change
RANDOM_CONTRAST = 5    # range (0-100), 0=no change
RANDOM_90_DEG_TURN = 1  # 0 or 1= random turn to left or right

def readCroppedImage(path, augmentations = True):
    # augmentations parameter is included for counting statistics from images, where we don't want augmentations
    
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    
    if(not augmentations):
        return rgb_img / 255
    
    #random rotation
    rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)
    if(RANDOM_90_DEG_TURN == 1):
        rotation += random.randint(-1,1) * 90
    M = cv2.getRotationMatrix2D((48,48),rotation,1)   # the center point is the rotation anchor
    rgb_img = cv2.warpAffine(rgb_img,M,(96,96))
    
    #random x,y-shift
    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    
    # crop to center and normalize to 0-1 range
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255
    
    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if(flip_hor):
        rgb_img = rgb_img[:, ::-1]
    if(flip_ver):
        rgb_img = rgb_img[::-1, :]
        
    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    rgb_img = rgb_img + br
    
    # Random contrast
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    rgb_img = rgb_img * cr
    
    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)
    
    return rgb_img


from sklearn.model_selection import train_test_split

# we read the csv file earlier to pandas dataframe, now we set index to id so we can perform
train_df = train.set_index('id')

train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

# split, this function returns more than we need as we only need the validation indexes for fastai
tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)


# fastai 1.0
from fastai import *
from fastai.vision import *
from torchvision.models import *    # import *=all the models from torchvision  

arch = densenet169                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment
BATCH_SIZE = 128                    # specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory
sz = CROP_SIZE                      # input size is the crop size
MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name 


# create dataframe for the fastai loader
train_dict = {'name': train_path + train_names, 'label': train_labels}
df = pd.DataFrame(data=train_dict)
# create test dataframe
test_names = []
for f in os.listdir(test_path):
    test_names.append(test_path + f)
df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])
# Subclass ImageList to use our own image opening function
class MyImageItemList(ImageList):
    def open(self, fn:PathOrStr)->Image:
        img = readCroppedImage(fn.replace('/./','').replace('//','/'))
        # This ndarray image has to be converted to tensor before passing on as fastai Image, we can use pil2tensor
        return vision.Image(px=pil2tensor(img, np.float32))
    
# Create ImageDataBunch using fastai data block API
imgDataBunch = (MyImageItemList.from_df(path='/', df=df, suffix='.tif')
        #Where to find the data?
        .split_by_idx(val_idx)
        #How to split in train/valid?
        .label_from_df(cols='label')
        #Where are the labels?
        .add_test(MyImageItemList.from_df(path='/', df=df_test))
        #dataframe pointing to the test set?
        .transform(tfms=[[],[]], size=sz)
        # We have our custom transformations implemented in the image loader but we could apply transformations also here
        # Even though we don't apply transformations here, we set two empty lists to tfms. Train and Validation augmentations
        .databunch(bs=BATCH_SIZE)
        # convert to databunch
        .normalize([tensor([0.702447, 0.546243, 0.696453]), tensor([0.238893, 0.282094, 0.216251])])
        # Normalize with training set stats. These are means and std's of each three channel and we calculated these previously in the stats step.
       )


# Next, we create a convnet learner object
# ps = dropout percentage (0-1) in the final layer
def getLearner():
    return create_cnn(imgDataBunch, arch, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

learner = getLearner()


# We can use lr_find with different weight decays and record all losses so that we can plot them on the same graph
# Number of iterations is by default 100, but at this low number of itrations, there might be too much variance
# from random sampling that makes it difficult to compare WD's. I recommend using an iteration count of at least 300 for more consistent results.
lrs = []
losses = []
wds = []
iter_count = 600

# WEIGHT DECAY = 1e-6
learner.lr_find(wd=1e-6, num_it=iter_count)
lrs.append(learner.recorder.lrs)
losses.append(learner.recorder.losses)
wds.append('1e-6')
learner = getLearner() #reset learner - this gets more consistent starting conditions

# WEIGHT DECAY = 1e-4
learner.lr_find(wd=1e-4, num_it=iter_count)
lrs.append(learner.recorder.lrs)
losses.append(learner.recorder.losses)
wds.append('1e-4')
learner = getLearner() #reset learner - this gets more consistent starting conditions

# WEIGHT DECAY = 1e-2
learner.lr_find(wd=1e-2, num_it=iter_count)
lrs.append(learner.recorder.lrs)
losses.append(learner.recorder.losses)
wds.append('1e-2')
learner = getLearner() #reset learner


# Plot weight decays
_, ax = plt.subplots(1,1)
min_y = 0.5
max_y = 0.55
for i in range(len(losses)):
    ax.plot(lrs[i], losses[i])
    min_y = min(np.asarray(losses[i]).min(), min_y)
ax.set_ylabel("Loss")
ax.set_xlabel("Learning Rate")
ax.set_xscale('log')
#ax ranges may need some tuning with different model architectures 
ax.set_xlim((1e-3,3e-1))
ax.set_ylim((min_y - 0.02,max_y))
ax.legend(wds)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))


max_lr = 2e-2
wd = 1e-4
# 1cycle policy
learner.fit_one_cycle(cyc_len=8, max_lr=max_lr, wd=wd)


# plot learning rate of the one cycle
learner.recorder.plot_lr()


# predict the validation set with our model
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(title='Confusion matrix')


# before we continue, lets save the model at this stage
learner.save(MODEL_PATH + '_stage1')


import os

print("Current working directory:", os.getcwd())
print("Model will be saved to:", MODEL_PATH + '_stage1.pth')



import torch

state = torch.load(learner.path / 'models' / f'{MODEL_PATH}_stage1.pth', weights_only=False)
learner.model.load_state_dict(state['model'])

learner.unfreeze()
learner.lr_find(wd=wd)
learner.recorder.plot()



# Now, smaller learning rates. This time we define the min and max lr of the cycle
learner.fit_one_cycle(cyc_len=12, max_lr=slice(4e-5,4e-4))


import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

def plot_loss_vs_epoch(learner, smooth_window=5, polyorder=2):
    recorder = learner.recorder

    # Get data 
    train_loss = [float(l) for l in recorder.losses]       # theo batch
    val_loss = [float(l) for l in recorder.val_losses]     # theo epoch
    n_epoch = len(val_loss)
    

    # Compute train loss 
    batch_per_epoch = len(train_loss) // n_epoch
    train_loss_epoch = [
        sum(train_loss[i*batch_per_epoch:(i+1)*batch_per_epoch]) / batch_per_epoch
        for i in range(n_epoch)
    ]

    # (smooth)
    if n_epoch >= smooth_window:  # chỉ làm mượt khi dữ liệu đủ dài
        train_smooth = savgol_filter(train_loss_epoch, smooth_window, polyorder)
        val_smooth = savgol_filter(val_loss, smooth_window, polyorder)
    else:
        train_smooth, val_smooth = train_loss_epoch, val_loss

    # DataFrame
    df = pd.DataFrame({
        'epoch': range(n_epoch),
        'train_loss': train_smooth,
        'valid_loss': val_smooth
    })


    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='steelblue', linewidth=2.5)
    plt.plot(df['epoch'], df['valid_loss'], label='Validation Loss', color='darkorange', linewidth=2.5)
    plt.title('Train vs Validation Loss per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
plot_loss_vs_epoch(learner)


learner.recorder.plot_losses()


# lets take a second look at the confusion matrix. See if how much we improved.
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(title='Confusion matrix')


# Save the finetuned model
learner.save(MODEL_PATH + '_stage2')

# if the model was better before finetuning, uncomment this to load the previous stage
#learner.load(MODEL_PATH + '_stage1')


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# Get predictions from learner 
preds, y_true, loss = learner.get_preds(with_loss=True)
probs = preds[:, 1].numpy()  # Probability of class 1
y_true = y_true.numpy()


def evaluate_model_fastai(y_true, probs, min_tpr=0.95, tune_threshold=True):
    """
    Evaluate the model based on the output of learner.get_preds().
    Can automatically tune the threshold prioritizing Recall (TPR).
    """
    # Tune threshold 
    threshold = 0.5
    if tune_threshold:
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        valid_idx = np.where(tpr >= min_tpr)[0]

        if len(valid_idx) == 0:
            print(f"No threshold achieves TPR >= {min_tpr}. The threshold with the highest TPR will be chosen.")
            best_idx = np.argmax(tpr)
        else:
            best_idx = valid_idx[np.argmin(fpr[valid_idx])]

        threshold = thresholds[best_idx]
        print(f"\nTuning threshold (TPR priority): {threshold:.4f}")
        print(f"TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")

    # Apply threshold
    y_pred = (probs >= threshold).astype(int)

    # Metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, probs)[:2])

    print("\n=== Evaluation Report ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")

    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Threshold={threshold:.3f})")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc,
        "threshold": threshold,
        "confusion_matrix": cm
    }

results = evaluate_model_fastai(y_true, probs, tune_threshold=True, min_tpr=0.95)

