# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, sys, shutil, copy
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
import glob
import pydoc

sys.path.insert(0, '/kaggle/input/timm-1-0-20/timm-1.0.20/')

import pydicom

import torch
import timm

import kaggle_evaluation.rsna_inference_server


print(f'torch version: {torch.__version__}')
print(f'timm version: {timm.__version__}')


ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

NUM_CLASSES = len(LABEL_COLS)


INPUT_VOLUME_CHANNELS = 3
IN_CHANNELS = 3

CROP_BACKBONE_MODEL_NAME = 'resnet18'
CROP_DEPTH_SIZE = 48
CROP_BACKBONE_CFG = DictConfig({
    'target_type': 'timm.create_model',
    'kwargs': {
        'model_name': 'vit_small_plus_patch16_dinov3.lvd1689m',
        'pretrained': False,
        'in_chans': 3,
        'global_pool': '',
        'num_classes': 0,
    },
})
CROP_INPUT_SIZE = [128, 128]
CROP_MODEL_CHECKPOINT_FILE = "/kaggle/input/vesselroibboxextractor-loss-00-20260129-153709/pytorch/default/1/best_loss_checkpoint_vesselroibboxextractor_00_20260129_153709.pth"

CLASSIFIER_BACKBONE_MODEL_NAME = 'resnet18'
CLASSIFIER_INPUT_SIZE = [352, 352]
CLASSIFIER_MODEL_CHECKPOINT_FILES = [
    # best acc
    "/kaggle/input/models/wissunpower/diseasedetector-00-acc-20260224-172817/pytorch/default/1/best_acc_checkpoint_diseasedetector_00_20260224_172817.pth",
    "/kaggle/input/models/wissunpower/diseasedetector-01-acc-20260226-150114/pytorch/default/1/best_acc_checkpoint_diseasedetector_01_20260226_150114.pth",
    "/kaggle/input/models/wissunpower/diseasedetector-02-acc-20260228-035434/pytorch/default/1/best_acc_checkpoint_diseasedetector_02_20260228_035434.pth",
    "/kaggle/input/models/wissunpower/diseasedetector-03-acc-20260301-225028/pytorch/default/1/best_acc_checkpoint_diseasedetector_03_20260301_225028.pth",
    "/kaggle/input/models/wissunpower/diseasedetector-04-acc-20260303-172040/pytorch/default/1/best_acc_checkpoint_diseasedetector_04_20260303_172040.pth",

    # best loss
    # "/kaggle/input/models/wissunpower/diseasedetector-00-loss-20260224-172817/pytorch/default/1/best_loss_checkpoint_diseasedetector_00_20260224_172817.pth",
    # "/kaggle/input/models/wissunpower/diseasedetector-01-loss-20260226-150114/pytorch/default/1/best_loss_checkpoint_diseasedetector_01_20260226_150114.pth",
    # "/kaggle/input/models/wissunpower/diseasedetector-02-loss-20260228-035434/pytorch/default/1/best_loss_checkpoint_diseasedetector_02_20260228_035434.pth",
    # "/kaggle/input/models/wissunpower/diseasedetector-03-loss-20260301-225028/pytorch/default/1/best_loss_checkpoint_diseasedetector_03_20260301_225028.pth",
    # "/kaggle/input/models/wissunpower/diseasedetector-04-loss-20260303-172040/pytorch/default/1/best_loss_checkpoint_diseasedetector_04_20260303_172040.pth",
]

PREPROCESS_CROP_OPT = DictConfig({
    'enable': False,
    'info_file_path': '',
})
PREPROCESS_RESIZE_OPT = DictConfig({
    'enable': False,
    'target_shape': [352, 352],
    'mode': 'bilinear',
})

BATCH_SIZE = 16


class DICOMToNPYPreprocessor:
    def __init__(self
                 , crop: DictConfig
                 , resize: DictConfig
                 ):
        self.crop_opt = crop
        self.resize_opt = resize

        if self.crop_opt.enable:
            self.crop_info = np.load(crop.info_file_path, allow_pickle=True).item()

    def __call__(self, dcm_set_path: str) -> tuple[np.ndarray, list]:
        files = np.array(glob.glob(os.path.join(dcm_set_path, '*.dcm')))

        dcms = [pydicom.dcmread(file) for file in files]

        shapes = [(d.Rows, d.Columns) for d in dcms]
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        
        valid_data = []
        for d, file in zip(dcms, files):
            if (d.Rows, d.Columns) == most_common_shape:
                valid_data.append([d, file])
        
        if len(valid_data) > 1:
            valid_data.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))
        
        #volume = np.stack([dcm.pixel_array for dcm in valid_dcms])
        
        #t = time.time()
        volume = np.stack([dcm[0].pixel_array for dcm in valid_data])
        
        if volume.shape[0] == 1:
            volume = volume[0]
            valid_data = [valid_data[0]] * volume.shape[0]
        
        if self.crop_opt.enable:
            series_uid = os.path.basename(dcm_set_path)
            height, width = volume.shape[-2:]
            x1, x2, y1, y2 = self.crop_info[series_uid]
            volume = volume[:
                      , int(y1 * height * 0.9):int(y2 * height * 1.1)
                      , int(x1 * width * 0.9):int(x2 * width * 1.1)]

        if self.resize_opt.enable:
            volume = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0)
            volume = torch.nn.functional.interpolate(volume, (self.resize_opt.target_shape[0], self.resize_opt.target_shape[1])
                                                 , mode=self.resize_opt.mode, align_corners=False)
            volume = volume.squeeze(0).numpy()
        
        return volume, valid_data


class VesselROIBBoxExtractor(torch.nn.Module):
    def __init__(self
                 , depth_size: int
                 , backbone: DictConfig):
        super(VesselROIBBoxExtractor, self).__init__()

        backbone_type = pydoc.locate(backbone.target_type)
        self.backbone = backbone_type(**backbone.kwargs)

        num_features = self.backbone.num_features

        self.head = torch.nn.Linear(num_features * depth_size // backbone.kwargs.in_chans, 4)
        
        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(self.head.weight, gain)
        self.head.bias.data.fill_(0)
    
    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        if 4 == len(volumes.shape):
            volumes.unsqueeze(0)
        
        batch_size, num_groups, channels, height, width = volumes.shape

        volumes = volumes.reshape(batch_size * num_groups, channels, height, width)
        features = self.backbone(volumes)

        features = features.mean(1)

        features = features.reshape(batch_size, num_groups, -1)
        features = features.flatten(1, 2)

        logits = self.head(features)

        logits = logits.sigmoid()

        return logits
    
    @torch.no_grad()
    def inference(self, volumes: torch.Tensor) -> torch.Tensor:
        if 4 == len(volumes.shape):
            volumes = volumes.unsqueeze(0)
        
        predict = self.forward(volumes)
        
        return predict


class DiseaseDetector(torch.nn.Module):
    def __init__(self
                 , num_classes: int
                 , backbone
                 ):
        super(DiseaseDetector, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone

        num_features = self.backbone.num_features

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(num_features, self.num_classes)
        
        gain = torch.nn.init.calculate_gain('linear')
        torch.nn.init.xavier_uniform_(self.head.weight, gain)
        self.head.bias.data.fill_(0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)

        if len(features.shape) == 3:
            features = features.mean(1)
        elif len(features.shape) > 3:
            features = self.avg_pool(features).flatten(1, 3)

        logits = self.head(features)

        return logits
    
    @torch.no_grad()
    def inference(self, image: torch.Tensor) -> torch.Tensor:
        if 3 == len(image.shape):
            image = image.unsqueeze(0)
        
        predict = self.forward(image)
        
        return predict.sigmoid()


device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
print(f'device: {device}')


crop_model = VesselROIBBoxExtractor(depth_size=CROP_DEPTH_SIZE, backbone=CROP_BACKBONE_CFG)
crop_model.load_state_dict(torch.load(CROP_MODEL_CHECKPOINT_FILE, map_location='cpu'))
crop_model = crop_model.to(device)
crop_model


classifier_models = []

for model_checkpoint_file in CLASSIFIER_MODEL_CHECKPOINT_FILES:
    classifier_backbone_model = timm.create_model(CLASSIFIER_BACKBONE_MODEL_NAME, pretrained=False, in_chans=IN_CHANNELS, global_pool='', num_classes=0)
    # classifier_backbone_model
    
    classifier_model = DiseaseDetector(num_classes=NUM_CLASSES, backbone=classifier_backbone_model)
    classifier_model.load_state_dict(torch.load(model_checkpoint_file, map_location='cpu'))
    classifier_model = classifier_model.to(device)
    # classifier_model

    classifier_models.append(copy.deepcopy(classifier_model))


preprocessor = DICOMToNPYPreprocessor(crop=PREPROCESS_CROP_OPT, resize=PREPROCESS_RESIZE_OPT)
preprocessor


def crop_preprocessing(volume):
    
    volume = volume[np.linspace(0, len(volume)-1, CROP_DEPTH_SIZE).astype(np.int16)]
    
    volume = volume.reshape(CROP_DEPTH_SIZE//IN_CHANNELS, IN_CHANNELS, volume.shape[1], volume.shape[2])
        
    volume = torch.as_tensor((volume - volume.min()) / (volume.max() - volume.min())).float()
    
    volume = torch.nn.functional.interpolate(volume, CROP_INPUT_SIZE, mode='bilinear')
    
    return volume


def crop_predict(volume):
    
    volume = crop_preprocessing(volume)
    volume = volume.unsqueeze(0)
    
    #config.volume = volume

    crop_model.eval()

    with torch.no_grad():

        volume = volume.to(device)
        
        with torch.autocast(device_type=str(device)):
            logits = crop_model(volume)
        outputs = logits.float().detach().cpu().numpy()
    
    return outputs[0]


def bin_preprocessing(volume: np.ndarray) -> torch.Tensor:
    volume = torch.as_tensor(volume).to(torch.float32)

    D = volume.shape[0]

    volume_last = torch.stack([volume[i-2] if i-2>-1 else volume[i] for i in range(D)])
    volume_next = torch.stack([volume[i+2] if i+2<D else volume[i] for i in range(D)])
    #volume_next = volume_last #Bug in training, to be corrected further in training
    
    volume = torch.stack([volume_last, volume, volume_next], 1)

    d = volume.shape[0]
    vmin = volume.view(d, -1).min(dim=1).values.view(d, 1, 1, 1)
    vmax = volume.view(d, -1).max(dim=1).values.view(d, 1, 1, 1)
    volume = ((volume - vmin) / (vmax - vmin + 1e-8)).float()
    
    volume = torch.nn.functional.interpolate(volume, CLASSIFIER_INPUT_SIZE, mode='bilinear')

    return volume


def bin_predict(volume: np.ndarray) -> np.ndarray:
    
    volume = bin_preprocessing(volume)
    
    with torch.no_grad():
        total_outputs = []

        for classifier_model in classifier_models:
            classifier_model.eval()
            
            outputs = []
            
            for i in range(0, volume.shape[0], BATCH_SIZE):
                start_idx = i
                end_idx = min(i + BATCH_SIZE, volume.shape[0])
                batch_images = volume[start_idx:end_idx]
                
                batch_images = batch_images.to(device).float()
                
                with torch.autocast(device_type=str(device)):
                    logits = classifier_model(batch_images)
                
                outs = logits.float().sigmoid().detach().cpu().numpy()
                
                outputs.extend(outs)
            
            outputs = np.stack(outputs)

            total_outputs.append(outputs)
        
        total_outputs = np.stack(total_outputs).mean(0)

    return total_outputs


def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:

    series_uid = os.path.basename(series_path)

    try:
        volume, valid_data = preprocessor(series_path)

        x1, x2, y1, y2 = crop_predict(volume)
        
        height, width = volume.shape[-2:]
        volume = volume[:, int(y1*height*0.9):int(y2*height*1.1), int(x1*width*0.9):int(x2*width*1.1)]

        predictions = bin_predict(volume=volume)

        final_pred = predictions.max(0)

        result_df = pl.DataFrame(
            data=[[series_uid] + final_pred.tolist()],
            schema=[ID_COL, *LABEL_COLS],
            orient='row'
        )
    except Exception as e:
        # Return a fallback dataframe with the correct schema
        result_df = pl.DataFrame(
            data=[[series_uid] + [0.1] * len(LABEL_COLS)],
            schema=[ID_COL, *LABEL_COLS],
            orient='row'
        )
    finally:
        # This code is required to prevent "out of disk space" and "directory not empty" errors.
        # It deletes the shared folder and then immediately recreates it, ensuring it's
        # empty and ready for the next prediction.

        # shared_dir = '/kaggle/shared'
        shared_dir = './shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
    
    return result_df.drop(ID_COL)


results = predict('/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381')
results.head()


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))


for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
import subprocess
from IPython.display import FileLink, display

def download_file(path, download_file_name):
    os.chdir('/kaggle/working/')
    zip_name = f"/kaggle/working/{download_file_name}.zip"
    command = f"zip {zip_name} {path} -r"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Unable to run zip command!")
        print(result.stderr)
        return
    display(FileLink(f'{download_file_name}.zip'))


download_file('./submission.parquet', 'submission')




