import os
import random
from pathlib import Path

import torch
import numpy as np


class Config:
    # 환경 설정
    IS_KAGGLE = os.path.exists('/kaggle/input')
    RANDOM_SEED = 42
    VERBOSE = 1
    
    # 데이터 경로
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025')
        MODEL_DIR = Path('/kaggle/input/byu-hrnet/pytorch/w32_384x288/1')
    else:
        DATA_DIR = Path('byu-locating-bacterial-flagellar-motors-2025')
        MODEL_DIR = Path('model')
    
    
    # 데이터 설정
    DATA = {
        'label_file': 'train_labels.csv',
        'heatmap_dir': 'heatmaps',  # 히트맵 저장 디렉토리
        'num_slices': 0,  # Motor 위치 기준으로 앞뒤로 볼 이미지 수
        'val_ratio': 0.2,  # validation set 비율
        'input_size': 640,  # 입력 이미지 크기
        'heatmap_size': 160,  # 히트맵 크기 (input_size / 4)
        'heatmap_threshold': 0.9,  # motor 예측을 위한 히트맵 임계값
        'sigma': 5.0,  # 가우시안 커널의 표준편차
        'heatmap_augment': {
            'sigma': (4.5, 5.5),
            'p': 0.0
        },
        'add_multiple_motor': {
            'enabled': True,
            'min_separate_distance': 10
        },
        'add_negative_motor': {
            'enabled': True,
            'more_samples': 0
        },
        'add_hard_negative': {
            'enabled': False,
            'min_separate_distance': 50,
            'more_samples': 3
        }
    }
    
    # 학습 설정
    TRAIN = {
        'n_fold': 5,
        'fold': 1,
        'batch_size': 8, # 8 for (640, 160), 16 for (512, 128)
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',  # 모델 저장 디렉토리
        'save_freq': 5,  # 모델 저장 주기 (epoch)
        'val_freq': 5,  # validation 실행 주기 (epoch)
        'early_stopping_patience': 30,  # validation metric이 개선되지 않는 최대 epoch 수
        'distance_threshold': 5
    }

    # 모델 설정
    MODEL = {
        'name': 'HRNet',
        'yaml': 'w32_384x288_adam_lr1e-3.yaml',
        'pretrained': 'hrnet_w32-36af842e.pth' if TRAIN['device'] == 'cuda' else ''
    }
    
    # Optimizer 설정
    OPTIMIZER = {
        'name': 'AdamW',
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }

    # Scheduler 설정
    SCHEDULER = {
        'name': 'CosineAnnealingLR_warmup',
        'T_max': 50,  # 최대 epoch 수
        'eta_min': 1e-6,  # 최소 learning rate
        'warmup_epochs': 10,  # 예열 기간
    }

    # Loss 설정
    LOSS = {
        'name': 'AdaptiveWingLoss',
        'omega': 14,
        'theta': 0.5,
        'epsilon': 1,
        'alpha': 2.1,
    }


def set_seed(seed: int = 42):
    """
    모든 random seed를 고정합니다.
    
    Args:
        seed: 고정할 seed 값
    """
    # Python random seed
    random.seed(seed)
    
    # Numpy random seed
    np.random.seed(seed)
    
    # PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # True로 설정하면 성능은 향상되지만 재현성은 떨어짐


# Config에서 seed 가져오기
config = Config()
set_seed(config.RANDOM_SEED)

print(config.IS_KAGGLE)


import os

from typing import Tuple
from datetime import datetime, timedelta


import pandas as pd
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

import sys
sys.path.insert(1, "/kaggle/input/byu-script")
from dataset import MotorDataset, AdditionalMotorDataset, split_train_val_tomo_ids
from model import get_pose_net
from util import AdaptiveWingLoss, calculate_distance, get_cosine_schedule_with_warmup


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    config: Config
) -> Tuple[float, float]:
    """한 epoch 동안의 학습을 수행합니다."""
    model.train()
    total_loss = 0
    total_samples = 0
    total_f_beta = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.TRAIN["num_epochs"]}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        heatmaps = batch['heatmap'].to(device)
        has_motor = batch['has_motor'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Loss 계산
        loss = criterion(outputs, heatmaps)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Fβ-score 계산
        f_beta_score, batch_samples, metrics = calculate_distance(
            outputs, heatmaps, has_motor, config.DATA['heatmap_threshold'],
            distance_threshold = config.TRAIN['distance_threshold']
        )
        total_f_beta += f_beta_score * batch_samples
        
        # 통계 업데이트
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        TP += metrics['true_positive']
        FP += metrics['false_positive']
        FN += metrics['false_negative']
        TN += metrics['true_negative']
        
        # 진행 상황 업데이트
        pbar.set_postfix({
            'loss': f'{total_loss/total_samples:.4f}',
            'f_beta': f'{total_f_beta/total_samples:.4f}',
            'TP': f"{TP}",
            'FP': f"{FP}",
            'FN': f"{FN}",
            'TN': f"{TN}",
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / total_samples, total_f_beta / total_samples

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    total_samples = 0
    total_f_beta = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            images = batch['image'].to(device)
            heatmaps = batch['heatmap'].to(device)
            has_motor = batch['has_motor'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Loss 계산
            loss = criterion(outputs, heatmaps)
            
            # Fβ-score 계산
            f_beta_score, batch_samples, metrics = calculate_distance(
                outputs, heatmaps, has_motor, config.DATA['heatmap_threshold'],
                distance_threshold = config.TRAIN['distance_threshold']
            )
            total_f_beta += f_beta_score * batch_samples
            
            # 통계 업데이트
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            TP += metrics['true_positive']
            FP += metrics['false_positive']
            FN += metrics['false_negative']
            TN += metrics['true_negative']
        
            # 진행 상황 업데이트
            pbar.set_postfix({
                'val_loss': f'{total_loss/total_samples:.4f}',
                'val_f_beta': f'{total_f_beta/total_samples:.4f}',
                'TP': f"{TP}",
                'FP': f"{FP}",
                'FN': f"{FN}",
                'TN': f"{TN}",
            })
    
    return total_loss / total_samples, total_f_beta / total_samples



def plot_training_metrics(train_losses, val_losses, train_f_betas, val_f_betas, learning_rates, save_dir=Path("")):
    """학습 과정 시각화"""
    epochs = range(1, len(train_losses) + 1)
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training/Validation Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training/Validation Fβ-score
    axes[0, 1].plot(epochs, train_f_betas, 'b-', label='Training Fβ-score')
    axes[0, 1].plot(epochs, val_f_betas, 'r-', label='Validation Fβ-score')
    axes[0, 1].set_title('Training and Validation Fβ-score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Fβ-score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate (로그 스케일)
    axes[1, 0].semilogy(epochs, learning_rates, 'g-')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate (log scale)')
    axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png')
    plt.close()


def run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    save_dir: Path
) -> None:
    """모델을 학습합니다."""
    # 디바이스 설정
    device = torch.device(config.TRAIN['device'])
    model = model.to(device)
    
    # Loss 함수 설정
    criterion = AdaptiveWingLoss(
        omega=config.LOSS['omega'],
        theta=config.LOSS['theta'],
        epsilon=config.LOSS['epsilon'],
        alpha=config.LOSS['alpha']
    )
    
    # Optimizer 설정
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAIN['learning_rate'],
        betas=config.OPTIMIZER['betas'],
        eps=config.OPTIMIZER['eps'],
        weight_decay=config.TRAIN['weight_decay']
    )
    
    # Scheduler 설정
    if config.SCHEDULER['name'] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.SCHEDULER['T_max'],
            eta_min=config.SCHEDULER['eta_min']
        )
    elif config.SCHEDULER['name'] == "CosineAnnealingLR_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_epochs=config.SCHEDULER['warmup_epochs'],
            num_training_epochs=config.SCHEDULER['T_max'],
            min_lr=config.SCHEDULER['eta_min']
        )
    
    # 모델 저장 디렉토리 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 학습 시작 시간
    start_time = datetime.now()
    print(f'Training started at {start_time}')

    # metrics 저장을 위한 리스트 초기화
    train_losses = []
    val_losses = []
    train_f_betas = []
    val_f_betas = []
    lrs = []

    # 최고 성능 기록
    best_val_f_beta = 0.0  # Fβ-score는 높을수록 좋음
    patience_counter = 0
    
    # 학습 루프
    for epoch in range(config.TRAIN['num_epochs']):
        # 학습
        train_loss, train_f_beta = train(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validation
        if (epoch + 1) % config.TRAIN['val_freq'] == 0:
            val_loss, val_f_beta = validate(model, val_loader, criterion, device, config)
            
            # metrics 저장
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f_betas.append(train_f_beta)
            val_f_betas.append(val_f_beta)
            lrs.append(optimizer.param_groups[0]["lr"])

            # 최고 성능 모델 저장
            if val_f_beta > best_val_f_beta:
                best_val_f_beta = val_f_beta
                torch.save(
                    model.state_dict(),
                    save_dir / f'best_model.pth'
                )
                print(f'New best model saved with val_f_beta: {val_f_beta:.4f}')
        
        # Early stopping 체크
        if (epoch + 1) % config.TRAIN['val_freq'] == 0:
            if val_f_beta > best_val_f_beta:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.TRAIN['early_stopping_patience']:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

        # 모델 저장
        if (epoch + 1) % config.TRAIN['save_freq'] == 0:
            torch.save(
                model.state_dict(),
                save_dir / f'model_epoch_{epoch+1}.pth'
            )
        
        # Learning rate 업데이트
        scheduler.step()
        
        # 에폭별 결과 출력
        print(f'Epoch {epoch+1}/{config.TRAIN["num_epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Fβ-score: {train_f_beta:.4f}')
        if (epoch + 1) % config.TRAIN['val_freq'] == 0:
            print(f'Val Loss: {val_loss:.4f}, Val Fβ-score: {val_f_beta:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print('-' * 50)

        current_time = datetime.now()
        if current_time - start_time > timedelta(hours=11, minutes=30):
            break
    
    # 학습 종료 시간
    end_time = datetime.now()
    print(f'Training finished at {end_time}')
    print(f'Total training time: {end_time - start_time}')
    print(f'Best validation Fβ-score: {best_val_f_beta:.4f}')

    # 학습 완료 후
    plot_training_metrics(train_losses, val_losses, train_f_betas, val_f_betas, lrs, save_dir=save_dir)


# 데이터셋 생성
folds = split_train_val_tomo_ids(
    df=pd.read_csv(os.path.join(config.DATA_DIR, config.DATA["label_file"])),
    val_ratio=config.DATA['val_ratio'],
    include_zero_motor=config.DATA['add_negative_motor']['enabled'],
    include_multiple_motor=config.DATA['add_multiple_motor']['enabled'],
    n_fold=config.TRAIN['n_fold'],
    fold=config.TRAIN['fold']
)

for k, (train_tomo_ids, val_tomo_ids) in enumerate(folds):
    # 데이터 증강 설정
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], is_check_shapes=False)

    val_transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], is_check_shapes=False)
    
    # 데이터셋 생성
    train_dataset = MotorDataset(
        data_dir=os.path.join(str(config.DATA_DIR), "train"),
        label_file=os.path.join(str(config.DATA_DIR), config.DATA['label_file']),
        heatmap_dir=config.DATA['heatmap_dir'],
        tomo_ids=train_tomo_ids,
        split="train",
        input_size=(config.DATA['input_size'], config.DATA['input_size']),
        heatmap_size=(config.DATA['heatmap_size'], config.DATA['heatmap_size']),
        num_slices=config.DATA['num_slices'],
        sigma=config.DATA['sigma'],
        transform=transform,
        heatmap_augment=config.DATA['heatmap_augment'],
        add_multiple_motor=config.DATA['add_multiple_motor'],
        add_negative_motor=config.DATA['add_negative_motor'],
        add_hard_negative=config.DATA['add_hard_negative']
    )
    train_dataset.describe()

    if config.IS_KAGGLE:
        df = pd.read_csv("/kaggle/input/cryoet-flagellar-motors-dataset/labels.csv") # TODO: refactor
        if not config.DATA['add_multiple_motor']['enabled']:
            tomo_id_counts = df["tomo_id"].value_counts()
            train_tomo_ids = tomo_id_counts[tomo_id_counts == 1].index
        else:
            train_tomo_ids = df["tomo_id"].unique()
            
        additional_train_dataset = AdditionalMotorDataset(
            data_dir=str("/kaggle/input/cryoet-flagellar-motors-dataset/jpgs"),
            label_file="/kaggle/input/cryoet-flagellar-motors-dataset/labels.csv",
            heatmap_dir=config.DATA['heatmap_dir'],
            input_size=(config.DATA['input_size'], config.DATA['input_size']),
            heatmap_size=(config.DATA['heatmap_size'], config.DATA['heatmap_size']),
            tomo_ids=train_tomo_ids,
            split="train",
            num_slices=config.DATA['num_slices'],
            sigma=config.DATA['sigma'],
            transform=transform,
            heatmap_augment=config.DATA['heatmap_augment'],
            add_multiple_motor=config.DATA['add_multiple_motor'],
            add_negative_motor=config.DATA['add_negative_motor'],
            add_hard_negative=config.DATA['add_hard_negative']
        )
        additional_train_dataset.describe()

        train_dataset = ConcatDataset([train_dataset, additional_train_dataset])
        
    print(f"Total number of samples: {len(train_dataset)}")

    val_dataset = MotorDataset(
        data_dir=os.path.join(str(config.DATA_DIR), "train"),
        label_file=os.path.join(str(config.DATA_DIR), config.DATA['label_file']),
        heatmap_dir=config.DATA['heatmap_dir'],
        tomo_ids=val_tomo_ids,
        split="val",
        input_size=(config.DATA['input_size'], config.DATA['input_size']),
        heatmap_size=(config.DATA['heatmap_size'], config.DATA['heatmap_size']),
        num_slices=config.DATA['num_slices'],
        sigma=config.DATA['sigma'],
        transform=val_transform,
        heatmap_augment=config.DATA['heatmap_augment'],
        add_multiple_motor=config.DATA['add_multiple_motor'],
        add_negative_motor=config.DATA['add_negative_motor'],
        add_hard_negative=config.DATA['add_hard_negative']
    )
    val_dataset.describe()
    
    print(f"Total number of samples: {len(val_dataset)}")
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=True,
        num_workers=config.TRAIN['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=False,
        num_workers=config.TRAIN['num_workers'],
        pin_memory=True
    )

    # 모델 생성
    model = get_pose_net(
        cfg_path=os.path.join(config.MODEL_DIR, config.MODEL["yaml"]),
        input_size=config.DATA['input_size'],
        heatmap_size=config.DATA['heatmap_size'],
        pretrained=config.MODEL["pretrained"],
        model_dir=config.MODEL_DIR
    )

    # 학습 실행
    run(model, train_loader, val_loader, config, config.TRAIN['save_dir'] + f"_fold_{k}")


def visualize_results(model, dataset, num_samples=3):
    # 모델을 CPU로 이동
    model = model.to('cpu')
    model.eval()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        # 이미지와 히트맵을 CPU로 이동
        image = sample['image'].to('cpu')
        target_heatmap = sample['heatmap'].to('cpu')
        tomo_id = sample["tomo_id"]
        slice_z = sample["motor_z"]
        
        with torch.no_grad():
            pred_heatmap = model(image.unsqueeze(0))
        
        # 원본 이미지
        if image.shape[0] == 3:  # RGB 이미지인 경우
            image_gray = image.mean(dim=0)  # (H, W)
        else:
            image_gray = image.squeeze()  # (H, W)
        axes[i, 0].imshow(image_gray.numpy(), cmap='gray')
        axes[i, 0].set_title(f'Original Image\nTomo ID: {tomo_id}\nSlice: {slice_z}')
        axes[i, 0].axis('off')
        
        # 예측 히트맵
        max_val = pred_heatmap.max().item()
        axes[i, 1].imshow(pred_heatmap.squeeze().numpy(), cmap='hot')
        axes[i, 1].set_title(f'Predicted Heatmap\nMax Value: {max_val:.4f}')
        axes[i, 1].axis('off')
        
        # 정답 히트맵
        max_val = target_heatmap.max().item()
        axes[i, 2].imshow(target_heatmap.squeeze().numpy(), cmap='hot')
        axes[i, 2].set_title(f'Target Heatmap\nMax Value: {max_val:.4f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# 학습 완료 후 시각화
visualize_results(model, train_dataset)
visualize_results(model, val_dataset)







