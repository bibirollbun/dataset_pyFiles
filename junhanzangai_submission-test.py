deps_path = '/kaggle/input/czii-cryoet-dependencies'


! cp -r /kaggle/input/czii-cryoet-dependencies/asciitree-0.3.3/ asciitree-0.3.3/


! pip wheel asciitree-0.3.3/asciitree-0.3.3/


!pip install asciitree-0.3.3-py3-none-any.whl


! pip install -q --no-index --find-links {deps_path} --requirement {deps_path}/requirements.txt


from typing import List, Tuple, Union
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandCropByLabelClassesd,
    NormalizeIntensityd,
    RandZoomd,
    AsDiscrete,
)
from monai.losses import (
    DiceLoss,
    DiceFocalLoss,
    DiceCELoss,
    TverskyLoss,
    GeneralizedDiceLoss,
    FocalLoss,
)


import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # JAX를 CPU 모드로 설정


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.
    
    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension
        
    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
    if dimension_size <= patch_size:
        return [0]
        
    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size)
    
    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
    
    return positions

def extract_3d_patches_minimal_overlap(arrays: List[np.ndarray], patch_size: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.
    
    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (m, n, l)
    patch_size : int
        Size of cubic patches (a x a x a)
        
    Returns:
    --------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (x, y, z) for each patch
    """
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")
    
    # Verify all arrays have the same shape
    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    
    if patch_size > min(shape):
        raise ValueError(f"patch_size ({patch_size}) must be smaller than smallest dimension {min(shape)}")
    
    m, n, l = shape
    patches = []
    coordinates = []
    
    # Calculate starting positions for each dimension
    x_starts = calculate_patch_starts(m, patch_size)
    y_starts = calculate_patch_starts(n, patch_size)
    z_starts = calculate_patch_starts(l, patch_size)
    
    # Extract patches from each array
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                        x:x + patch_size,
                        y:y + patch_size,
                        z:z + patch_size
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))
    
    return patches, coordinates

# Note: I should probably averge the overlapping areas, 
# but here they are just overwritten by the most recent one. 

def reconstruct_array(patches: List[np.ndarray], 
                     coordinates: List[Tuple[int, int, int]], 
                     original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Reconstruct array from patches.
    
    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patches to reconstruct from
    coordinates : List[Tuple[int, int, int]]
        Starting coordinates for each patch
    original_shape : Tuple[int, int, int]
        Shape of the original array
        
    Returns:
    --------
    np.ndarray
        Reconstructed array
    """
    reconstructed = np.zeros(original_shape, dtype=np.int64)  # To track overlapping regions
    
    patch_size = patches[0].shape[0]
    
    for patch, (x, y, z) in zip(patches, coordinates):
        reconstructed[
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ] = patch
        
    
    return reconstructed


import pandas as pd

def dict_to_df(coord_dict, experiment_name):
    """
    Convert dictionary of coordinates to pandas DataFrame.
    
    Parameters:
    -----------
    coord_dict : dict
        Dictionary where keys are labels and values are Nx3 coordinate arrays
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['x', 'y', 'z', 'label']
    """
    # Create lists to store data
    all_coords = []
    all_labels = []
    
    # Process each label and its coordinates
    for label, coords in coord_dict.items():
        all_coords.append(coords)
        all_labels.extend([label] * len(coords))
    
    # Concatenate all coordinates
    all_coords = np.vstack(all_coords)
    
    df = pd.DataFrame({
        'experiment': experiment_name,
        'particle_type': all_labels,
        'x': all_coords[:, 0],
        'y': all_coords[:, 1],
        'z': all_coords[:, 2]
    })

    
    return df


TRAIN_DATA_DIR = "/kaggle/input/czii-numpy-dataset-20250107"
TEST_DATA_DIR = "/kaggle/input/czii-cryo-et-object-identification"


import pytorch_lightning as pl

from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau
)

class Model(pl.LightningModule):
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=7,
                 channels=(48, 64, 80, 80), strides=(2, 2, 1),
                 num_res_units=1, lr=1e-3,
                 scheduler_type='one_cycle'):
            super().__init__()
            self.save_hyperparameters()

            # Model
            self.model = UNet(
              spatial_dims=self.hparams.spatial_dims,
              in_channels=self.hparams.in_channels,
              out_channels=self.hparams.out_channels,
              channels=self.hparams.channels,
              strides=self.hparams.strides,
              num_res_units=self.hparams.num_res_units,
              norm='batch',  # BatchNorm3d
              dropout=0.2,
          )

            # Loss function
            self.loss_fn = TverskyLoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                alpha=0.5,
                beta=0.95
            )

            # Metric
            self.metric_fn = DiceMetric(
                include_background=False,
                reduction="mean",
                get_not_nans=False
            )

            # Learning rate와 scheduler 설정
            self.lr = lr
            self.scheduler_type = scheduler_type

            # 결과 저장용 리스트
            self.training_step_outputs = []
            self.validation_step_outputs = []

            # Class weights 정의
            self.class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 0.0])

            # Storage for validation outputs
            self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)

        metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i)
                             for i in decollate_batch(y_hat)]
        metric_val_labels = [AsDiscrete(to_onehot=self.hparams.out_channels)(i)
                            for i in decollate_batch(y)]

        self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
        metrics = self.metric_fn.aggregate(reduction="mean_batch")

        # 클래스별 가중치를 device로 이동
        class_weights = self.class_weights.to(metrics.device)

        # 가중치가 적용된 전체 메트릭
        weighted_metric = (metrics * class_weights).sum() / class_weights.sum()

        # 로깅
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_metric', weighted_metric, on_step=False, on_epoch=True)

        output = {
            'val_loss': val_loss.detach(),
            'val_metric': weighted_metric.detach(),
            'class_metrics': metrics.detach()
        }

        self.validation_outputs.append(output)

        # 출력
        print(f"\nEpoch {self.current_epoch}, Validation batch {batch_idx}")
        print(f"Loss: {val_loss:.4f}, Metric: {weighted_metric:.4f}")

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # 메트릭 로깅 추가
        self.log("train_loss", loss, on_step=False, on_epoch=True)  # 여기를 추가

        print(f"Epoch {self.current_epoch}, Training batch {batch_idx}, Loss: {loss.item():.4f}")

        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        print(f"\n{'='*40}")
        print(f"Epoch {self.current_epoch} Training completed")
        print(f"Average training loss: {epoch_mean:.4f}")
        print(f"{'='*40}\n")
        self.training_step_outputs.clear()

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            print("No validation outputs found!")
            return

        try:
            # 평균 계산
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
            avg_metric = torch.stack([x['val_metric'] for x in self.validation_outputs]).mean()
            class_metrics = torch.stack([x['class_metrics'] for x in self.validation_outputs]).mean(dim=0)

            print(f"\n{'='*70}")
            print(f"Validation Epoch {self.current_epoch} Summary")
            print(f"{'='*70}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Weighted Metric: {avg_metric:.4f}")
            print("\nClass-wise Performance:")
            class_names = ['Ribosome', 'Virus-like', 'Apo-ferritin',
                          'Thyroglobulin (Hard)', 'β-galactosidase (Hard)', 'Beta-amylase (Not evaluated)']

            for i, (name, metric) in enumerate(zip(class_names, class_metrics)):
                print(f"  {name:<20} {metric:.4f}")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"Error in validation epoch end: {str(e)}")
        finally:
            # 메트릭 리셋
            self.metric_fn.reset()
            self.validation_outputs = []

    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

      if self.scheduler_type == 'one_cycle':
          # dataloader가 설정되기 전에는 cosine scheduler를 사용
          if not hasattr(self.trainer, 'train_dataloader') or self.trainer.train_dataloader is None:
              print("Warning: train_dataloader not set, switching to cosine scheduler")
              scheduler = CosineAnnealingWarmRestarts(
                  optimizer,
                  T_0=10,
                  T_mult=2,
                  eta_min=1e-6
              )
              scheduler_config = {
                  "scheduler": scheduler,
                  "interval": "epoch",
                  "frequency": 1
              }
          else:
              steps_per_epoch = len(self.trainer.train_dataloader())
              total_steps = steps_per_epoch * self.trainer.max_epochs

              scheduler = OneCycleLR(
                  optimizer,
                  max_lr=self.lr,
                  total_steps=total_steps,
                  pct_start=0.3,
                  div_factor=25.0,
                  final_div_factor=1e4
              )
              scheduler_config = {
                  "scheduler": scheduler,
                  "interval": "step",
                  "frequency": 1
              }

      elif self.scheduler_type == 'plateau':
          scheduler = ReduceLROnPlateau(
              optimizer,
              mode='max',
              factor=0.5,
              patience=100,
              min_lr=1e-6,
              verbose=True
          )
          scheduler_config = {
              "scheduler": scheduler,
              "interval": "epoch",
              "monitor": 'val_metric',
              "frequency": 1
          }

      else:  # cosine as default
          scheduler = CosineAnnealingWarmRestarts(
              optimizer,
              T_0=10,
              T_mult=2,
              eta_min=1e-6
          )
          scheduler_config = {
              "scheduler": scheduler,
              "interval": "epoch",
              "frequency": 1
          }

      return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


channels = (64, 128, 256, 256)
strides_pattern = (2, 2, 1)
num_res_units = 1
learning_rate = 1e-3
num_epochs = 1000

model = Model(channels=channels, strides=strides_pattern, num_res_units=num_res_units, lr=learning_rate)


# # 체크포인트 로드
# (2) 체크포인트 로드
ckpt_paths = [
    "/kaggle/input/20250122-v34-7fold/fold0.ckpt",
    "/kaggle/input/20250122-v34-7fold/fold1.ckpt",
    # "/kaggle/input/20250122-v34-7fold/fold2.ckpt",
]

# 여러 모델을 불러와 리스트에 저장
models_ensemble = []
for cp in ckpt_paths:
    m = Model.load_from_checkpoint(cp)
    m.eval()
    m.to("cuda")
    models_ensemble.append(m)


# # 학습 시작 전에 print 문 추가
# print("Starting training...")
# trainer.fit(model, train_loader, valid_loader)
# print("Training completed!")
# torch.save(model.state_dict(), 'final_model.pth')

# # pth
# model.load_state_dict(torch.load('/kaggle/input/20250115-cz/final_model.pth'))

# # 모델을 평가 모드로 설정 (테스트 또는 추론 시)
# model.eval()


import json
copick_config_path = TRAIN_DATA_DIR + "/copick.config"

with open(copick_config_path) as f:
    copick_config = json.load(f)

copick_config['static_root'] = '/kaggle/input/czii-cryo-et-object-identification/test/static'

copick_test_config_path = 'copick_test.config'

with open(copick_test_config_path, 'w') as outfile:
    json.dump(copick_config, outfile)


import copick

root = copick.from_file(copick_test_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"


# Non-random transforms to be cached
inference_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS")
])


import cc3d

id_to_name = {1: "apo-ferritin", 
              2: "beta-amylase",
              3: "beta-galactosidase", 
              4: "ribosome", 
              5: "thyroglobulin", 
              6: "virus-like-particle"}


import numpy as np
import torch
import cc3d
import time  # 시간 측정용

from monai.inferers import sliding_window_inference
from monai.data import CacheDataset

# 간단한 3D flip 함수들
def flip_x_3d(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=[2])  # x축 뒤집기

def flip_y_3d(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=[3])  # y축 뒤집기

def flip_z_3d(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=[4])  # z축 뒤집기

def identity_3d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor

# (forward_transform, inverse_transform) 쌍 목록
tta_ops = [
    (identity_3d, identity_3d),
    (flip_y_3d, flip_y_3d),
    (flip_z_3d, flip_z_3d),
]

def ensemble_tta_predictor(sub_volume: torch.Tensor) -> torch.Tensor:
    """
    sub_volume : (B=1, C=1, D, H, W) 형태의 3D sub-volume Tensor
    - 여기서 여러 모델(models_ensemble)에 대해, 
      TTA(fwd_op -> 모델추론 -> inv_op)를 모두 수행 후 평균내어 반환한다.
    
    반환 : (B=1, out_channels=7, D, H, W)
    """
    all_logits = []

    with torch.no_grad():
        # (1) 앙상블 대상 모델들을 순회
        for model in models_ensemble:
            # (2) TTA 변환들 순회
            for fwd_op, inv_op in tta_ops:
                # 1) 변환
                vol_t = fwd_op(sub_volume)          # (1,1,D,H,W)
                # 2) 모델 추론(로짓)
                logits = model(vol_t)               # (1,7,D,H,W)
                # 3) 원래 좌표로 역변환
                logits_inv = inv_op(logits)         # (1,7,D,H,W)

                # 리스트에 저장
                all_logits.append(logits_inv)

    # 여러 모델 × 여러 TTA 로짓을 평균 -> (1,7,D,H,W)
    final_logits = torch.mean(torch.stack(all_logits, dim=0), dim=0)
    return final_logits

# -----------------------------------------------------------
# 슬라이딩 윈도우 + 앙상블 + TTA 인퍼런스 (후처리 없음) 예시
# -----------------------------------------------------------
BLOB_THRESHOLD = 250
classes = [1, 2, 3, 4, 5, 6]
total_start = time.time()

with torch.no_grad():
    location_df = []

    for run in root.runs:
        run_start = time.time()  # 한 번의 run 시작 시간

        print(run)

        # 1) 볼륨(10Å voxel) 로드
        load_start = time.time()
        tomo = run.get_voxel_spacing(10)
        tomo_arr = tomo.get_tomogram(tomo_type).numpy()  # shape: (X, Y, Z)
        load_end = time.time()
        print(f"[Timer] Volume load time: {load_end - load_start:.3f} sec")

        # 2) Dataset 로드(전처리)
        prep_start = time.time()
        data_dict = [{"image": tomo_arr}]
        tomo_ds = CacheDataset(data=data_dict, transform=inference_transforms, cache_rate=1.0)
        volume_tensor = tomo_ds[0]["image"].unsqueeze(0).to("cuda")  # (1,1,X,Y,Z)
        prep_end = time.time()
        print(f"[Timer] Dataset prep time: {prep_end - prep_start:.3f} sec")

        # 3) Sliding Window Inference
        #    -> predictor=ensemble_tta_predictor 로 교체
        infer_start = time.time()
        out_logits = sliding_window_inference(
            inputs=volume_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=6,
            predictor=ensemble_tta_predictor,  # <-- 여기서 앙상블+TTA 진행
            overlap=0.25,
            mode="gaussian"
        )
        infer_end = time.time()
        print(f"[Timer] SW Inference(Ensemble+TTA) time: {infer_end - infer_start:.3f} sec")

        # 4) Softmax 후 argmax
        post_start = time.time()
        out_probs = torch.softmax(out_logits, dim=1)  # (1,7,X,Y,Z)
        out_probs_np = out_probs[0].cpu().numpy()     # (7, X, Y, Z)
        reconstructed_mask = np.argmax(out_probs_np, axis=0)  # (X, Y, Z)
        post_end = time.time()
        print(f"[Timer] Postprocess(softmax+argmax) time: {post_end - post_start:.3f} sec")

        # 클래스별 aspect ratio 임계값 설정 (실제 모양 정보를 반영)
        aspect_ratio_thresholds = {
            "apo-ferritin": 0.8,
            "beta-galactosidase": 0.6,
            "ribosome": 0.65,
            "thyroglobulin": 0.6,
            "virus-like-particle": 0.8
        }
        
        cc_start = time.time()
        location = {}
        
        # classes는 [1, 2, 3, 4, 5, 6]와 같이 번호가 할당되어 있다고 가정하고,
        # id_to_name을 통해 실제 클래스 이름으로 매핑합니다.
        for c in classes:
            # reconstructed_mask == c 인 영역에 대해 연결 요소 분석 실행
            cc = cc3d.connected_components(reconstructed_mask == c)
            stats = cc3d.statistics(cc)
            
            # 배경 label(0)은 제외하고 실제 객체는 인덱스 1부터 사용
            centroids = np.array(stats["centroids"][1:])  # 보통 [z, y, x] 순서로 반환됨
            bbox_list = stats["bounding_boxes"][1:]         # 각 객체의 bounding box, 형식: (slice_z, slice_y, slice_x)
            voxel_counts = np.array(stats["voxel_counts"][1:])
            
            valid_indices = []
            # 해당 클래스의 이름과 임계값을 설정 (없으면 기본값 0.7 사용)
            class_name = id_to_name[c]
            ar_threshold = aspect_ratio_thresholds.get(class_name, 0.7)
            
            for i, bbox in enumerate(bbox_list):
                # 각 bounding box는 (slice_z, slice_y, slice_x) 형태이며,
                # 각 slice 객체에서 start와 stop 값을 사용하여 길이를 계산합니다.
                slice_z, slice_y, slice_x = bbox  # 각각의 축에 대한 slice 객체
                depth  = slice_z.stop - slice_z.start
                height = slice_y.stop - slice_y.start
                width  = slice_x.stop - slice_x.start
                
                # Aspect ratio: 최소 길이를 최대 길이로 나눔 (1에 가까울수록 구형)
                aspect_ratio = min(width, height, depth) / max(width, height, depth)
                
                # 조건: voxel 수가 BLOB_THRESHOLD 이상이며, aspect ratio가 클래스에 맞는 임계값 이상일 경우
                if voxel_counts[i] > BLOB_THRESHOLD and aspect_ratio >= ar_threshold:
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                # valid한 객체들의 중심 좌표를 선택합니다.
                valid_centroids = centroids[valid_indices]
                # voxel 크기 보정을 위해 (예: 10.012444)를 곱합니다.
                valid_centroids = valid_centroids * 10.012444  
                # cc3d 결과의 좌표 순서가 [z, y, x]이면, 이를 [x, y, z]로 변경합니다.
                valid_xyz = np.ascontiguousarray(valid_centroids[:, ::-1])
                location[class_name] = valid_xyz
        
        cc_end = time.time()
        print(f"[Timer] Connected components + centroids + shape filtering time: {cc_end - cc_start:.3f} sec")
        
        # 이후 dict_to_df 함수를 이용해 DataFrame으로 변환하여 location_df에 저장합니다.
        df = dict_to_df(location, run.name)
        location_df.append(df)

        run_end = time.time()
        print(f"[Timer] Single run total time: {run_end - run_start:.3f} sec\n")

    # 모든 run 결과 결합
    location_df = pd.concat(location_df)

# 전체 파이프라인 시간 측정 종료
total_end = time.time()
print(f"전체 파이프라인 수행 시간: {total_end - total_start:.3f} 초")
print(f'estimated predict time is {(total_end - total_start)/3*500:.4f} seconds')


location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))
location_df.to_csv("submission.csv", index=False)


!ls


!cp -r /kaggle/input/hengck-czii-cryo-et-01/* .


from czii_helper import *
from dataset import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    MODE = 'submit'
else:
    MODE = 'local'







valid_dir ='/kaggle/input/czii-cryo-et-object-identification/train'
valid_id = ['TS_5_4', 'TS_69_2', 'TS_6_4']

def do_one_eval(truth, predict, threshold):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric


def compute_lb(submit_df, overlay_dir):
    valid_id = list(submit_df['experiment'].unique())
    print(valid_id)

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(id, overlay_dir) #=f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df['experiment'] == id]
        for p in PARTICLE:
            p = dotdict(p)
            print('\r', id, p.name, end='', flush=True)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius* 0.5)
            eval_df.append(dotdict(
                id=id, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            ))
    print('')
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
    gb.loc[:, 'precision'] = gb['hit'] / gb['P']
    gb.loc[:, 'precision'] = gb['precision'].fillna(0)
    gb.loc[:, 'recall'] = gb['hit'] / gb['T']
    gb.loc[:, 'recall'] = gb['recall'].fillna(0)
    gb.loc[:, 'f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall'])
    gb.loc[:, 'f-beta4'] = gb['f-beta4'].fillna(0)

    gb = gb.sort_values('particle_type').reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, 'weight'] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()
    return gb, lb_score


#debug
if 1:
    if MODE=='local':
    #if 1:
        submit_df=pd.read_csv(
           'submission.csv'
            # '/kaggle/input/hengck-czii-cryo-et-weights-01/submission.csv'
        )
        gb, lb_score = compute_lb(submit_df, f'{valid_dir}/overlay/ExperimentRuns')
        print(gb)
        print('lb_score:',lb_score)
        print('')


        #show one ----------------------------------
        fig = plt.figure(figsize=(18, 8))

        # debug 시각화: valid_id에 있는 모든 experiment에 대해 시각화하기
        for exp_id in valid_id:
            # 해당 experiment의 정답 데이터를 읽어옵니다.
            truth = read_one_truth(exp_id, overlay_dir=f'{valid_dir}/overlay/ExperimentRuns')
            # submission 데이터에서 해당 experiment에 해당하는 부분만 선택합니다.
            id_df = submit_df[submit_df['experiment'] == exp_id]
            
            # experiment 별 figure 생성 (여기서는 2행 3열의 subplot 구성)
            fig = plt.figure(figsize=(18, 8))
            for p in PARTICLE:
                p = dotdict(p)
                # 해당 particle type에 대한 정답과 예측 데이터를 추출합니다.
                xyz_truth = truth[p.name]
                xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
                
                # 평가 함수 호출
                hit, fp, miss, _ = do_one_eval(xyz_truth, xyz_predict, p.radius)
                print(exp_id, p.name)
                print('\t num truth   :', len(xyz_truth))
                print('\t num predict :', len(xyz_predict))
                print('\t num hit     :', len(hit[0]))
                print('\t num fp      :', len(fp))
                print('\t num miss    :', len(miss))
            
                # subplot에 3D scatter plot 그리기
                ax = fig.add_subplot(2, 3, p.label, projection='3d')
                if hit[0]:
                    # 예측 중 맞춘 점들 (빨간색 점)와 대응하는 정답 (빨간 테두리 원)
                    pt_pred = xyz_predict[hit[0]]
                    ax.scatter(pt_pred[:, 0], pt_pred[:, 1], pt_pred[:, 2], alpha=0.5, color='r')
                    pt_truth = xyz_truth[hit[1]]
                    ax.scatter(pt_truth[:, 0], pt_truth[:, 1], pt_truth[:, 2], s=80, facecolors='none', edgecolors='r')
                if fp:
                    # false positive (검은색 점)
                    pt_fp = xyz_predict[fp]
                    ax.scatter(pt_fp[:, 0], pt_fp[:, 1], pt_fp[:, 2], alpha=1, color='k')
                if miss:
                    # miss (검은 테두리 원)
                    pt_miss = xyz_truth[miss]
                    ax.scatter(pt_miss[:, 0], pt_miss[:, 1], pt_miss[:, 2], s=160, alpha=1, facecolors='none', edgecolors='k')
            
                ax.set_title(f'{p.name} ({p.difficulty})')
            
            plt.suptitle(f'Experiment: {exp_id}', fontsize=16)
            plt.tight_layout()
            plt.show()







