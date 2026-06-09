


!unzip -q ../input/timm-with-dependencies/timm_all -d timm-with-dependencies
!pip install --no-index --find-links timm-with-dependencies timm
!pip install /kaggle/input/dicomsdl-offline-installer/dicomsdl-0.109.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

from fastai.vision.learner import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import ActivationType

from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd
import numpy as np
from pdb import set_trace


NUM_EPOCHS = 4
NUM_SPLITS = 4

RESIZE_TO = (1024, 1024)

DATA_PATH = '/kaggle/input/rsna-breast-cancer-detection'
TRAIN_IMAGE_DIR = '/kaggle/input/rsna-mammography-images-as-pngs/images_as_pngs_cv2_512'
TEST_DICOM_DIR = '/kaggle/input/rsna-breast-cancer-detection/test_images'
MODEL_PATH = '/kaggle/input/rsna-trained-model-weights/tf_effv2_s_208_402/tf_effv2_s_208_402'

label_smoothing_weights = torch.tensor([1,10]).float()
if torch.cuda.is_available():
    label_smoothing_weights = label_smoothing_weights.cuda()


train_csv = pd.read_csv(f'{DATA_PATH}/train.csv')
patient_id_any_cancer = train_csv.groupby('patient_id').cancer.max().reset_index()
skf = StratifiedKFold(NUM_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(patient_id_any_cancer.patient_id, patient_id_any_cancer.cancer))


#https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369267  
def pfbeta_torch(preds, labels, beta=1):
    if preds.dim() != 2 or (preds.dim() == 2 and preds.shape[1] !=2): raise ValueError('Houston, we got a problem')
    preds = preds[:, 1]
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0

# https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369886    
def pfbeta_torch_thresh(preds, labels):
    optimized_preds = optimize_preds(preds, labels)
    return pfbeta_torch(optimized_preds, labels)

def optimize_preds(preds, labels=None, thresh=None, return_thresh=False, print_results=False):
    preds = preds.clone()
    if labels is not None: without_thresh = pfbeta_torch(preds, labels)
    
    if not thresh and labels is not None:
        threshs = np.linspace(0, 1, 101)
        f1s = [pfbeta_torch((preds > thr).float(), labels) for thr in threshs]
        idx = np.argmax(f1s)
        thresh, best_pfbeta = threshs[idx], f1s[idx]

    preds = (preds > thresh).float()

    if print_results:
        print(f'without optimization: {without_thresh}')
        pfbeta = pfbeta_torch(preds, labels)
        print(f'with optimization: {pfbeta}')
        print(f'best_thresh = {thresh}')
    if return_thresh:
        return thresh
    return preds

fn2label = {fn: cancer_or_not for fn, cancer_or_not in zip(train_csv['image_id'].astype('str'), train_csv['cancer'])}

def splitting_func(paths):
    train = []
    valid = []
    for idx, path in enumerate(paths):
        if int(path.parent.name) in patient_id_any_cancer.iloc[splits[SPLIT][0]].patient_id.values:
            train.append(idx)
        else:
            valid.append(idx)
    return train, valid

def label_func(path):
    return fn2label[path.stem]

def get_items(image_dir_path):
    items = []
    for p in get_image_files(image_dir_path):
        items.append(p)
        if p.stem in fn2label and int(p.parent.name) in patient_id_any_cancer.iloc[splits[SPLIT][0]].patient_id.values:
            if label_func(p) == 1:
                for _ in range(5):
                    items.append(p)
    return items


from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from torch.nn import Flatten

def get_dataloaders():
    train_image_path = TRAIN_IMAGE_DIR
    batch_tfms = [
    *aug_transforms(
        flip_vert=False,
        max_rotate=10,
        max_zoom=1.1,
        max_lighting=0.2
    )
    ]
    dblock = DataBlock(
        blocks    = (ImageBlock, CategoryBlock),
        get_items = get_items,
        get_y = label_func,
        splitter  = splitting_func,
        batch_tfms=batch_tfms
    )
    dsets = dblock.datasets(train_image_path)
    return dblock.dataloaders(train_image_path, batch_size=32)

def get_learner(arch):
    if arch in ['tf_efficientnet_b4_ns', 'tf_efficientnetv2_s']:
        n_feats = 1792
    else:
        n_feats = 512

    learner = vision_learner(
        get_dataloaders(),
        arch,
        custom_head=nn.Sequential(
            SelectAdaptivePool2d(pool_type='avg', flatten=Flatten()),
            nn.Linear(n_feats, 2)
        ),
        metrics=[
            error_rate,
            AccumMetric(pfbeta_torch, activation=ActivationType.Softmax, flatten=False),
            AccumMetric(pfbeta_torch_thresh, activation=ActivationType.Softmax, flatten=False)
        ],
       loss_func=CrossEntropyLossFlat(weight=torch.tensor([1,20]).float()),
        pretrained=True,
        normalize=True
    ).to_fp16()

    return learner



# This is a dependency that is needed for reading DICOM images

try:
    import pylibjpeg
except:
    !rm -rf /root/.cache/torch/hub/checkpoints/
    !mkdir -p /root/.cache/torch/hub/checkpoints/
    !pip install /kaggle/input/rsna-2022-whl/{pydicom-2.3.0-py3-none-any.whl,pylibjpeg-1.4.0-py3-none-any.whl,python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}
    !pip install /kaggle/input/rsna-2022-whl/{torch-1.12.1-cp37-cp37m-manylinux1_x86_64.whl,torchvision-0.13.1-cp37-cp37m-manylinux1_x86_64.whl}

# copying the pretrained weights

if not os.path.exists('/root/.cache/torch/hub/checkpoints/'):
        os.makedirs('/root/.cache/torch/hub/checkpoints/')
!cp '/kaggle/input/pretrained-model-weights-for-fastai/resnet18-f37072fd.pth' '/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
!cp '/kaggle/input/pretrained-model-weights-for-fastai/tf_efficientnetv2_s-eb54923e.pth' '/root/.cache/torch/hub/checkpoints/tf_efficientnetv2_s-eb54923e.pth'


%%time

preds, labels = [], []

SPLIT = 0 # our learner needs this to construct its dataloaders...
learn = get_learner('tf_efficientnet_b4_ns')
learn.freeze()
learn.fit_one_cycle(3, 3e-4)

learn.unfreeze()
learn.fit_one_cycle(8, 1e-4)



# instead of training, to conserve pipeline time, I am uploading models trained locally
# uncomment the lines below for training
  
# for SPLIT in range(NUM_SPLITS):
#     learn = get_learner()
#     learn.unfreeze()
#     learn.fit_one_cycle(NUM_EPOCHS, 1e-4, pct_start=0.1)
#     learn.save(f'{MODEL_PATH}/{SPLIT}')
        
#     output = learn.get_preds()
#     preds.append(output[0])
#     labels.append(output[1])





# threshold = optimize_preds(torch.cat(preds), torch.cat(labels), return_thresh=True, print_results=True)
threshold = 0.402


# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut
import dicomsdl
    
from pathlib import Path
import multiprocessing as mp
import cv2

!rm -rf test_resized_{RESIZE_TO[0]}

def dicom_file_to_ary(path):
    dcm_file = dicomsdl.open(str(path))
    data = dcm_file.pixelData()

    data = (data - data.min()) / (data.max() - data.min())

    if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = 1 - data

    data = cv2.resize(data, RESIZE_TO)
    data = (data * 255).astype(np.uint8)
    return data

directories = list(Path(TEST_DICOM_DIR).iterdir())

def process_directory(directory_path):
    parent_directory = str(directory_path).split('/')[-1]
    !mkdir -p test_resized_{RESIZE_TO[0]}/{parent_directory}
    for image_path in directory_path.iterdir():
        processed_ary = dicom_file_to_ary(image_path)
        cv2.imwrite(
            f'test_resized_{RESIZE_TO[0]}/{parent_directory}/{image_path.stem}.png',
            processed_ary
        )

with mp.Pool(mp.cpu_count()) as p:
    p.map(process_directory, directories)


%%time

preds_all = []

test_dl = learn.dls.test_dl(get_image_files(f'test_resized_{RESIZE_TO[0]}'))
for SPLIT in range(NUM_SPLITS):
    learn.load(f'{MODEL_PATH}/{SPLIT}')
    preds, _ = learn.get_preds(dl=test_dl)
    preds_all.append(preds)


preds = torch.zeros_like(preds_all[0])
for pred in preds_all:
    preds += pred

preds /= NUM_SPLITS


preds = optimize_preds(preds, thresh=threshold)
image_ids = [path.stem for path in test_dl.items]

image_id2pred = defaultdict(lambda: 0)
for image_id, pred in zip(image_ids, preds[:, 1]):
    image_id2pred[int(image_id)] = pred.item()


test_csv = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')

prediction_ids = []
preds = []

for _, row in test_csv.iterrows():
    prediction_ids.append(row.prediction_id)
    preds.append(image_id2pred[row.image_id])

submission = pd.DataFrame(data={'prediction_id': prediction_ids, 'cancer': preds}).groupby('prediction_id').max().reset_index()
submission.head()


submission.to_csv('submission.csv', index=False)

