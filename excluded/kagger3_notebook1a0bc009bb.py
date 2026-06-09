import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio
import os
import pandas as pd
from tqdm import tqdm
import torch
from pathlib import Path
import gc
import psutil

class CFG:
    NUM_CLASSES = 206
    SAMPLE_RATE = 32000
    WINDOW_SIZE = 5  # 秒
    WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SIZE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHTS_PATH = "/kaggle/input/pthfewe/best_model_fold0_auc0.9103_epoch14.pth"  # TODO: 替换为你的权重路径
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"  # 路径变量与主流notebook一致
    sample_submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'  # 官方样例
    submission_csv = '/kaggle/working/submission.csv'  # 你要生成的提交文件
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/efficnetpth/efficientnet_v2_s-dd5fe13b.pth'  # 主模型权重目录
    effnet_path = '/kaggle/input/efficnetpth/efficientnet_v2_s-dd5fe13b.pth'  # EfficientNet权重文件路径
    FS = 32000
    N_FFT = 1024
    HOP_LENGTH = 320
    N_MELS = 24
    FMIN = 50
    FMAX = 10000
    TARGET_SHAPE = (256, 256)
    model_name = 'tf_efficientnetv2_s.in21k'
    in_channels = 3
    batch_size = 1
    use_tta = False
    tta_count = 1
    threshold = 0.5
    use_specific_folds = False
    folds = [0]
    debug = False
    debug_count = 3

cfg = CFG()

def print_mem():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"[内存占用] {mem:.2f} MB")

# ===== 1. Cnn14实现 =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Cnn14(nn.Module):
    def __init__(self, num_classes=527, in_channels=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc = nn.Linear(512, 2048)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = torch.mean(x, dim=(2, 3))
        embedding = self.fc(x)
        return {'embedding': embedding}

# ===== 2. 音频预处理模块 =====
class ComplexAbs(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class AudioProcessor(nn.Module):
    def __init__(self, sample_rate=32000):
        super().__init__()
        self.sample_rate = sample_rate
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            hop_length=320,
            power=None
        )
        self.to_complex_abs = ComplexAbs()
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=24,
            sample_rate=sample_rate,
            f_min=50,
            f_max=10000,
            n_stft=513
        )
        self.log_compress = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        # waveform: (batch, 160000)  # 5秒音频，采样率32000
        x = self.stft(waveform)
        x = self.to_complex_abs(x)
        x = self.mel_scale(x)
        x = self.log_compress(x)
        x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)
        return x  # (batch, 1, 64, time)

# ===== 3. 音频特征提取模块 =====
class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AudioProcessor()
        self.cnn14 = Cnn14()

    def forward(self, waveform):
        x = self.processor(waveform)  # (batch, 1, 64, time)
        output = self.cnn14(x)
        return output['embedding']  # (batch, 2048)

# ===== 4. 深度可分离投影 =====
class DepthwiseProjection(nn.Module):
    def __init__(self, in_dim=2048, out_dim=1280):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.GELU(),
            nn.Conv1d(in_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.proj(x.unsqueeze(-1)).squeeze(-1)

# ===== 5. FPN视觉特征提取 =====
class FeaturePyramid(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.stage_indices = [2, 3, 4]
        self.channels = sum([backbone.feature_info[i]['num_chs'] for i in self.stage_indices])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(self.channels, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.GELU()
        )

    def forward(self, x):
        feats = self.backbone(x)
        selected = [feats[i] for i in self.stage_indices]
        target_size = selected[-1].shape[2:]
        resized = [
            F.adaptive_avg_pool2d(f, target_size)
            for f in selected[:-1]
        ] + [selected[-1]]
        fused = self.fuse_conv(torch.cat(resized, dim=1))
        return fused.mean(dim=[2, 3])

# ===== 6. 双向融合模块 =====
class BidirectionalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vis_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.aud_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, vis, aud):
        vis_gate = self.vis_gate(torch.cat([vis, aud], dim=1))
        vis_out = vis * vis_gate + aud * (1 - vis_gate)
        aud_gate = self.aud_gate(torch.cat([aud, vis], dim=1))
        aud_out = aud * aud_gate + vis * (1 - aud_gate)
        return torch.cat([vis_out, aud_out], dim=1)

# ===== 7. 随机模态丢弃 =====
class ModalityDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, vis, aud):
        if self.training:
            vis = torch.where(torch.rand_like(vis) < self.p, 0., vis)
            aud = torch.where(torch.rand_like(aud) < self.p, 0., aud)
        return vis, aud

# ===== 8. 完整模型结构 =====
class EnhancedBirdNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # pretrained=False，加载本地权重
        self.vis_backbone = timm.create_model('tf_efficientnetv2_s.in21k', features_only=True, pretrained=False)
        
        # 加载本地 EfficientNet 权重
        effnet_path = cfg.effnet_path
        try:
            # 方法1: 使用weights_only=False
            print("尝试使用 weights_only=False 加载EfficientNet权重...")
            effnet_weights = torch.load(effnet_path, map_location='cpu', weights_only=False)
        except Exception as e1:
            print(f"EfficientNet加载方法1失败: {str(e1)}")
            try:
                # 方法2: 使用safe_globals上下文管理器
                print("尝试使用 safe_globals 上下文管理器加载EfficientNet权重...")
                import torch.serialization
                with torch.serialization.safe_globals(['numpy', 'numpy._core.multiarray.scalar']):
                    effnet_weights = torch.load(effnet_path, map_location='cpu')
            except Exception as e2:
                print(f"EfficientNet加载方法2失败: {str(e2)}")
                # 方法3: 使用pickle直接加载
                print("尝试使用 pickle 直接加载EfficientNet权重...")
                import pickle
                with open(effnet_path, 'rb') as f:
                    effnet_weights = pickle.load(f)
        
        if 'model' in effnet_weights:
            effnet_state = effnet_weights['model']
        elif 'state_dict' in effnet_weights:
            effnet_state = effnet_weights['state_dict']
        else:
            effnet_state = effnet_weights
        load_result = self.vis_backbone.load_state_dict(effnet_state, strict=False)
        print('EfficientNet 权重加载结果:')
        print('  missing_keys:', load_result.missing_keys)
        print('  unexpected_keys:', load_result.unexpected_keys)
        self.pyramid = FeaturePyramid(self.vis_backbone)
        self.audio_extractor = AudioFeatureExtractor()
        self.audio_proj = DepthwiseProjection()
        self.modality_drop = ModalityDropout(p=0.2)
        self.bi_fusion = BidirectionalFusion(1280)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, img, audio_waveform, use_modality_dropout=True):
        # img: (batch, 3, 64, 256) 或 (batch, 3, 256, 256)  # 与训练时一致
        # audio_waveform: (batch, 160000)
        vis = self.pyramid(img)
        aud = self.audio_proj(self.audio_extractor(audio_waveform))
        if use_modality_dropout:
            vis, aud = self.modality_drop(vis, aud)
        fused = self.bi_fusion(vis, aud)
        return self.classifier(fused)

def main():
    import torch
    import os
    print(f"使用设备: {cfg.DEVICE}")
    print(f"模型权重目录: {cfg.model_path}")
    print(f"测试音频目录: {cfg.test_soundscapes}")
    print_mem()
    # 读取 sample_submission.csv 获取列名，只读取一次
    sample = pd.read_csv(cfg.sample_submission_csv)
    class_names = list(sample.columns)[1:]  # 除去 row_id
    print(f"类别数量: {len(class_names)}")
    print(f"实际查找的测试集路径: {cfg.test_soundscapes}")
    if os.path.exists(cfg.test_soundscapes):
        files_in_dir = os.listdir(cfg.test_soundscapes)
        print(f"该目录下文件: {files_in_dir}")
    else:
        print("测试集路径不存在！")
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    print(f"找到 {len(test_files)} 个测试音频文件: {[str(f) for f in test_files]}")
    if len(test_files) == 0:
        print(f"[错误] 没有找到测试音频文件! 请检查测试集路径: {cfg.test_soundscapes}")
        print(f"当前目录下文件: {os.listdir(cfg.test_soundscapes) if os.path.exists(cfg.test_soundscapes) else '目录不存在'}")
        # raise FileNotFoundError(f"未找到任何.ogg测试音频文件, 请检查路径和Kaggle数据集挂载!")
    # 加载模型
    try:
        model = EnhancedBirdNet(num_classes=cfg.NUM_CLASSES)
        try:
            print("尝试使用 weights_only=False 加载模型...")
            state = torch.load(cfg.WEIGHTS_PATH, map_location=cfg.DEVICE, weights_only=False)
        except Exception as e1:
            print(f"方法1失败: {str(e1)}")
            try:
                print("尝试使用 safe_globals 上下文管理器加载模型...")
                import torch.serialization
                with torch.serialization.safe_globals(['numpy', 'numpy._core.multiarray.scalar']):
                    state = torch.load(cfg.WEIGHTS_PATH, map_location=cfg.DEVICE)
            except Exception as e2:
                print(f"方法2失败: {str(e2)}")
                print("尝试使用 pickle 直接加载模型...")
                import pickle
                with open(cfg.WEIGHTS_PATH, 'rb') as f:
                    state = pickle.load(f)
        if 'model_state_dict' in state:
            load_result = model.load_state_dict(state['model_state_dict'], strict=False)
            print('主模型权重加载结果:')
            print('  missing_keys:', load_result.missing_keys)
            print('  unexpected_keys:', load_result.unexpected_keys)
        elif 'state_dict' in state:
            load_result = model.load_state_dict(state['state_dict'], strict=False)
            print('主模型权重加载结果:')
            print('  missing_keys:', load_result.missing_keys)
            print('  unexpected_keys:', load_result.unexpected_keys)
        else:
            load_result = model.load_state_dict(state, strict=False)
            print('主模型权重加载结果:')
            print('  missing_keys:', load_result.missing_keys)
            print('  unexpected_keys:', load_result.unexpected_keys)
        model.to(cfg.DEVICE)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        sample.to_csv(cfg.submission_csv, index=False)
        print(f"已创建空的提交文件: {cfg.submission_csv}")
        return
    results = []
    for file_path in tqdm(test_files, desc='Processing files'):
        try:
            fname = os.path.basename(file_path)
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.mean(0)
            if sr != cfg.SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, cfg.SAMPLE_RATE)
            total_samples = waveform.shape[0]
            num_windows = (total_samples - 1) // cfg.WINDOW_SAMPLES + 1
            for i in range(num_windows):
                start = i * cfg.WINDOW_SAMPLES
                end = start + cfg.WINDOW_SAMPLES
                window = waveform[start:end]
                if window.shape[0] < cfg.WINDOW_SAMPLES:
                    window = torch.nn.functional.pad(window, (0, cfg.WINDOW_SAMPLES - window.shape[0]))
                window = window.unsqueeze(0).to(cfg.DEVICE)
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=cfg.SAMPLE_RATE,
                    n_fft=1024,
                    hop_length=320,
                    n_mels=24,
                    f_min=50,
                    f_max=10000
                ).to(cfg.DEVICE)
                db_transform = torchaudio.transforms.AmplitudeToDB().to(cfg.DEVICE)
                mel = mel_transform(window)
                mel_db = db_transform(mel)
                mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                spec = torch.stack([mel_db] * 3, dim=1)
                spec = torch.nn.functional.interpolate(
                    spec, size=(256, 256), mode='bilinear', align_corners=False
                )
                with torch.no_grad():
                    logits = model(spec, window, use_modality_dropout=False)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                end_time = (i + 1) * 5
                row_id = f'{fname}_{end_time}'
                row = {'row_id': row_id}
                for idx, class_name in enumerate(class_names):
                    row[class_name] = probs[idx]
                results.append(row)
                del window, mel, mel_db, spec, logits, probs
                torch.cuda.empty_cache()
            del waveform
            gc.collect()
            print_mem()
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            gc.collect()
            print_mem()
    submission = pd.DataFrame(results)
    submission = submission.reindex(columns=['row_id'] + class_names, fill_value=0.0)
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print(f"已保存提交文件: /kaggle/working/submission.csv")
    print_mem()
    import os
    if not os.path.exists('/kaggle/working/submission.csv'):
        print("未检测到 submission.csv，兜底生成一个空的提交文件。")
        sample = pd.read_csv(cfg.sample_submission_csv)
        sample.to_csv('/kaggle/working/submission.csv', index=False)
        print(f"兜底生成空的提交文件: /kaggle/working/submission.csv")
    else:
        print("submission.csv 已正常生成。")

# 直接调用 main()
main()

        





