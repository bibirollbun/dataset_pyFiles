
!pip install /kaggle/input/rsna-monai-pydicom-nibabel-wheels/monai-1.3.1-py3-none-any.whl
!pip install /kaggle/input/rsna-monai-pydicom-nibabel-wheels/pydicom-2.4.4-py3-none-any.whl
!pip install /kaggle/input/rsna-monai-pydicom-nibabel-wheels/nibabel-5.2.1-py3-none-any.whl


!pip install /kaggle/input/rsna-monai-pydicom-nibabel-wheels/typed_argument_parser-1.10.1-py3-none-any.whl


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")

import tqdm
import monai
from monai.transforms import (
    CastToTyped,
    CenterScaleCropd,
    Compose,
    CropForeground,
    CropForegroundd,
    EnsureChannelFirstd,
    InvertibleTransform,
    LoadImaged,
    MapTransform,
    RandCropByPosNegLabel,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGridDistortiond,
    Resize,
    Resized,
    Spacing,
    Spacingd,
    SpatialCrop,
    SpatialPad,
    SpatialPadd,
    ToTensord,
    TraceableTransform,
)
import itertools
from collections.abc import Sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final
from sklearn.model_selection import train_test_split

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from monai.networks.nets.swin_unetr import SwinTransformer, MERGING_MODE

from monai.networks.nets import SEResNet50, SEResNet101, SEResNext101
from monai.networks.blocks.squeeze_and_excitation import SEBottleneck, SEResNetBottleneck


#đã sửa model path
MODEL_PATH = "/kaggle/input/checkpoint-data"



from tap import Tap

class SimpleArgumentParser(Tap):
    workdir = "./workdir"
    task_name = "SEResNext101_custom_[no_resample]_[augs1]_32x128x256_[v2]"
    project_name = "kaggle_rsna2024"
    data_dir = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
    eval_before_training = False

    # model config
    in_channels=1
    spatial_dims=3
    layers=(3, 4, 23, 3)
    dropout_prob=0.2
    inplanes=64
    model_name = "SEResNext101_custom"
    checkpoint = "/mnt/sda/RSNA_2024/workdir/SEResNext101_custom_[no_resample]_[augs1]_32x128x256/model_best.pth"

    # data config
    modality = "Sagittal T2/STIR"
    test_size = 0.2
    random_state = 42
    batch_size = 3
    num_workers = 0
    image_size = (32, 128, 256) # (384, 384)
    resample_z_slices = None
    cache_dir = "cache_dir_[no_resample]_32x128x256"

    # train config
    epochs = 50
    accumulation_steps = 2  # Number of batches to accumulate gradients
    label_smoothing_epsilon = 0.01
    lr = 0.001
    class_weights = [1.0, 2.0, 4.0]

args = SimpleArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CropBySpineMRI(monai.transforms.MapTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def crop_volume_by_black_border(self, vol, threshold=30):
        """
        Crops the input volume along the h axis based on the given threshold.
        
        Parameters:
        vol (np.ndarray): Input volume with shape (d, h, w)
        threshold (float): Threshold value for cropping (default is 50)
        
        Returns:
        np.ndarray: Cropped volume
        """

        def crop_by_h(vol, thr):
            # Compute the mean along the h axis
            means = np.mean(vol, axis=(0, 2))
            
            # Apply the threshold
            above_threshold = means > thr
            
            # Find the first and last indices where the condition is True
            start = np.argmax(above_threshold)
            end = len(above_threshold) - np.argmax(above_threshold[::-1])
            
            # Crop the volume
            cropped_vol = vol[:, start:end, :]
            
            return cropped_vol

        def crop_by_w(vol, thr):
            # Compute the mean along the h axis
            means = np.mean(vol, axis=(0, 1))
            
            # Apply the threshold
            above_threshold = means > thr
            
            # Find the first and last indices where the condition is True
            start = np.argmax(above_threshold)
            end = len(above_threshold) - np.argmax(above_threshold[::-1])
            
            # Crop the volume
            cropped_vol = vol[:, :, start:end]
            
            return cropped_vol

        vol = crop_by_h(vol, threshold)
        vol = crop_by_w(vol, threshold)
        return vol

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            dim = len(img.shape)

            if dim == 4:
                img = np.squeeze(img, axis=0)

            # Assuming img has shape (D, H, W)
            new_img = self.crop_volume_by_black_border(img)
            _, h, w = new_img.shape

            if key == "Sagittal T1":
                new_img = new_img[:, int(h * 0.3):int(h * 0.7), :]
            elif key == "Sagittal T2/STIR":
                new_img = new_img[:, int(h * 0.3):int(h * 0.7), :]
            elif key == "Axial T2":
                pass
            else:
                raise ValueError(f"{key} not modality")

            if dim == 4:
                new_img = np.expand_dims(new_img, axis=0)

            d[key] = new_img

        return d


class ExpandChannelFirstd(monai.transforms.MapTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key]
            dim = len(img.shape)
            # Assuming img has shape (D, H, W)

            if dim == 3 and args.spatial_dims == 3:
                img = img.unsqueeze(0)

            d[key] = img

        return d
    

class ResampleZ(monai.transforms.MapTransform):
    def __init__(self, new_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_depth = new_depth

    def __call__(self, data):
        d = dict(data)

        if self.new_depth is None:
            return d

        for key in self.keys:
            img = d[key]
            dim = len(img.shape)
            # Assuming img has shape (D, H, W)
            if dim == 3:
                _, h, w = img.shape
                # Add channel dimension (C, D, H, W) where C=1
                img = img.unsqueeze(0)
            else:
                _, _, h, w = img.shape

            # Interpolate
            img = F.interpolate(img.unsqueeze(0), size=(self.new_depth, h, w), mode='trilinear', align_corners=False)

            # Remove added dimensions
            if dim == 3:
                img = img.squeeze(0).squeeze(0)

            d[key] = img

        return d


class RSNADataset(monai.data.Dataset):
    def __init__(self, df=None, train_series_descriptions=None, is_train=True, cache_dir=None, hash_func=None):
        self.is_train = is_train
        self.df = df
        self.cache_dir = cache_dir
        self.hash_func = hash_func
        
        if is_train:
            self.study_ids = list(set(df['study_id'].values.tolist()))
        else:
            self.study_ids = list(set(train_series_descriptions['study_id'].values.tolist()))

        self.train_series_descriptions = train_series_descriptions
        self.labels = [
            'spinal_canal_stenosis_l1_l2',
            'spinal_canal_stenosis_l2_l3',
            'spinal_canal_stenosis_l3_l4',
            'spinal_canal_stenosis_l4_l5',
            'spinal_canal_stenosis_l5_s1',
            'left_neural_foraminal_narrowing_l1_l2',
            'left_neural_foraminal_narrowing_l2_l3',
            'left_neural_foraminal_narrowing_l3_l4',
            'left_neural_foraminal_narrowing_l4_l5',
            'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l1_l2',
            'right_neural_foraminal_narrowing_l2_l3',
            'right_neural_foraminal_narrowing_l3_l4',
            'right_neural_foraminal_narrowing_l4_l5',
            'right_neural_foraminal_narrowing_l5_s1',
            'left_subarticular_stenosis_l1_l2', 
            'left_subarticular_stenosis_l2_l3',
            'left_subarticular_stenosis_l3_l4', 
            'left_subarticular_stenosis_l4_l5',
            'left_subarticular_stenosis_l5_s1', 
            'right_subarticular_stenosis_l1_l2',
            'right_subarticular_stenosis_l2_l3',
            'right_subarticular_stenosis_l3_l4',
            'right_subarticular_stenosis_l4_l5',
            'right_subarticular_stenosis_l5_s1'
        ]
        self.target_mapping = {
            "Normal/Mild": [1, 0, 0],
            "Moderate":[0, 1, 0],
            "Severe": [0, 0, 1]
        }
        self.ss_mapping = {
            0: "normal_mild", 1: "moderate", 2: "severe"
        }
        self.data_dir = args.data_dir
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        series_descriptions = self.train_series_descriptions.loc[self.train_series_descriptions.study_id == study_id]  # series_id, series_description

        data = {}
        data["study_id"] = str(study_id)

        if self.cache_dir:
            fpath_hash = self.hash_func(data).decode("utf-8")
            hashfile = os.path.join(self.cache_dir, f"{fpath_hash}.pt")
            if os.path.isfile(hashfile):
                return data

        for series_id, series_description in series_descriptions[["series_id", "series_description"]].values:
            if series_description == args.modality:

                if self.is_train:
                    path_to_dicom_dir = f"{self.data_dir}/train_images/{study_id}/{series_id}"
                else:
                    path_to_dicom_dir = f"{self.data_dir}/test_images/{study_id}/{series_id}"
                # dicom = self.apply_transform(path_to_dicom_dir)
                # data[series_description] = dicom

                data[series_description] = path_to_dicom_dir
        
        if self.is_train:
            data["target"] = self.target_processing(index)

        return data
    
    def target_processing(self, index):
        row = self.df.iloc[[index]]
        
        data = []
        for label in self.labels:
            severity = row[[label]].values[0][0]
            target = self.target_mapping[severity]
            data.append(target)
            
        return np.asarray(data)

    def apply_transform(self, path_to_dicom_dir):
        return self.transform(path_to_dicom_dir)

    def get_transform(self):
        return monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=[args.modality]),
            monai.transforms.EnsureChannelFirstd(keys=[args.modality], channel_dim=-1),
            ResampleZ(keys=[args.modality], new_depth=args.resample_z_slices),
            CropBySpineMRI(keys=[args.modality]),
            ExpandChannelFirstd(keys=[args.modality]),
            monai.transforms.Resized(keys=[args.modality], spatial_size=args.image_size, mode='trilinear'),
            monai.transforms.NormalizeIntensityd(keys=[args.modality]),
            monai.transforms.ToTensord(keys=[args.modality])
        ])

def get_dataset(df, train_series_descriptions, is_train=True):
    ds = RSNADataset(
        df=df, 
        train_series_descriptions=train_series_descriptions, 
        is_train=is_train, 
        cache_dir=args.cache_dir,
        hash_func=f_name_hash
    )

    if args.cache_dir is not None:
        return monai.data.PersistentDataset(
            data=ds,
            transform=ds.get_transform(),
            cache_dir=args.cache_dir,
            hash_func=f_name_hash,
        )
    else:
        return monai.data.Dataset(data=ds, transform=ds.get_transform())


class SEResNext101_custom(SEResNext101):
    def __init__(
        self,
        in_channels=args.in_channels,
        spatial_dims=args.spatial_dims,
        dropout_prob=args.dropout_prob,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            spatial_dims=spatial_dims,
            dropout_prob=dropout_prob,
            num_classes=75,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logits(x)
        x = x.view(x.size(0), 25, 3)
        return x


def load_net():
    if args.model_name == "SEResNext101_custom":
        net = SEResNext101_custom(
            in_channels=args.in_channels,
            spatial_dims=args.spatial_dims,
        )
    return net


# net = SEResNet50Custom(in_channels=1, spatial_dims=2, layers=(3, 4, 6, 3), dropout_prob=0.2, inplanes=64)
net = load_net()

# net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model_best.pth"), map_location="cpu")["state_dict"])
net = net.to(device)

if device == 'cuda':
    net = torch.compile(net)
    net(torch.randn(1, 224, 224).to(device))


def logloss(target, predict):
    target = torch.mean(target.float(), dim=0).argmax(dim=-1).cpu().detach().numpy()
    predict = torch.median(predict, dim=0).values.cpu().detach().numpy()
    labels = [0,1,2]
    mapping = {0: 1, 1: 2, 2: 4}
    sample_weight = [mapping[y] for y in target]

    return sklearn.metrics.log_loss(
        y_true=target,
        y_pred=predict,
        labels=labels,
        sample_weight=sample_weight
    )

def format_val_to_submission(val, val_predictions, labels):
    submission = []

    for idx, row in val.iterrows():
        study_id = row['study_id']
        predictions = val_predictions[idx % len(val_predictions)]

        for i, label in enumerate(labels):
            row_id = f"{study_id}_{label}"
            normal_mild_value = float(predictions[i][0])
            moderate_value = float(predictions[i][1])
            severe_value = float(predictions[i][2])
            submission.append([row_id, normal_mild_value, moderate_value, severe_value])
    
    submission_df = pd.DataFrame(submission, columns=['row_id', 'normal_mild', 'moderate', 'severe'])
    return submission_df

def prepare_tensor(tensor):
    # Check if tensor has shape (25, 3), expand dims to (1, 25, 3)
    if tensor.shape == torch.Size([25, 3]):
        tensor = tensor.unsqueeze(0)  # Adds a batch dimension

    return tensor

def validate_model_with_submission_format(net, val_dataloader, val_df, labels):
    net.eval()  # Set the model to evaluation mode
    val_predictions = []
    val_targets = []

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm.tqdm(val_dataloader, desc="Val", total=len(val_dataloader)):
#             in_tensor = batch['Sagittal T2/STIR'].view(batch['Sagittal T2/STIR'].shape[1], 1, *batch['Sagittal T2/STIR'].shape[2:])
            in_tensor = batch[args.modality]
            target = batch['target']

            in_tensor = in_tensor.to(device)
            target = target.to(device).squeeze(0)  # Adjust target shape to match predict
            
            predict = net(in_tensor).softmax(dim=-1)
            # predict = torch.median(predict, dim=0).values  # Apply median after softmax

            predict = prepare_tensor(predict)
            target = prepare_tensor(target)

            val_predictions.append(predict)
            val_targets.append(target)

    # Concatenate all tensors in the lists along the batch dimension
    val_predictions = torch.cat(val_predictions, dim=0)  # Shape (N, 25, 3)
    val_targets = torch.cat(val_targets, dim=0)    # Shape (N, 25, 3)

    val_predictions = val_predictions.cpu().numpy()
    val_targets = val_targets.cpu().numpy()

    # Format the validation data to submission format
    formatted_submission = format_val_to_submission(val_df.reset_index(drop=True), val_predictions, labels)

    # Calculate log loss
    target_labels = []
    predict_labels = []
    sample_weights = []

    for i, row in formatted_submission.iterrows():
        row_id_parts = row['row_id'].split('_')
        study_id = row_id_parts[0]
        label = '_'.join(row_id_parts[1:])
        
        true_values = val_df[val_df['study_id'] == int(study_id)][label].values[0]
        true_class = ['Normal/Mild', 'Moderate', 'Severe'].index(true_values)
        target_labels.append(true_class)
        
        predict_values = row[['normal_mild', 'moderate', 'severe']].values
        predict_labels.append(predict_values)
        
        mapping = {0: 1, 1: 2, 2: 4}
        sample_weights.append(mapping[true_class])

    target_labels = np.array(target_labels)
    predict_labels = np.array(predict_labels)

    log_loss_value = sklearn.metrics.log_loss(
        y_true=target_labels,
        y_pred=predict_labels,
        labels=[0, 1, 2],
        sample_weight=sample_weights
    )

    return log_loss_value







if len(os.listdir("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images")) == 1:
    
    train = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
    train_label_coordinates = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv")
    train_series_descriptions = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv")
    train = train.dropna()

    train, val = train_test_split(train, test_size=0.2, random_state=42)

    val_dataset = RSNADataset(df=val, train_series_descriptions=train_series_descriptions)
    val_dataset = monai.data.Dataset(data=val_dataset, transform=val_dataset.get_transform())
    val_dataloader = monai.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_loss = validate_model_with_submission_format(net, val_dataloader, val, val_dataset.data.labels)

    print('LOGLOSS:', val_loss,)


test_series_descriptions = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_series_descriptions.csv")
ss = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/sample_submission.csv")

test_dataset = RSNADataset(df=None, train_series_descriptions=test_series_descriptions, is_train=False)
test_dataset = monai.data.Dataset(data=test_dataset, transform=test_dataset.get_transform())
test_dataloader = monai.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)



# Validation function
def inference_model(net, test_dataloader, ss, labels):
    
    new_ss = pd.DataFrame(columns=ss.columns)

    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for batch in test_dataloader:
            in_tensor = batch[args.modality]
            in_tensor = in_tensor.to(device)

            study_id = int(batch['study_id'][0])

            predict = net(in_tensor).softmax(dim=-1).cpu().numpy()[0]

            for i in range(25):
                label = labels[i]
                row_id = f"{study_id}_{label}"
                normal_mild_value = float(predict[i][0])
                moderate_value = float(predict[i][1])
                severe_value = float(predict[i][2])
                
                data = [[row_id, normal_mild_value, moderate_value, severe_value]]
                data_df = pd.DataFrame(data, columns=ss.columns)
                
                # Concatenate the new DataFrame to new_ss
                new_ss = pd.concat([new_ss, data_df], axis=0, ignore_index=True)


    return new_ss



new_ss = inference_model(net, test_dataloader, ss, test_dataset.data.labels)


new_ss


new_ss.to_csv("submission.csv", index=False)

