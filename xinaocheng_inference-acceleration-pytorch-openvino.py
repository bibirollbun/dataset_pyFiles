%%capture
!pip install -U openvino-dev
!pip install -U openvino-telemetry  --no-index --find-links /kaggle/input/openvino
!pip install -U openvino  --no-index --find-links /kaggle/input/openvino


!pip list | grep openvino


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd


# Here we use the model from BirdCLEF2025 for example

# Define the model architecture
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(cfg['model_name'], pretrained=False, in_chans=cfg['in_channels'],drop_rate=0.2,drop_path_rate=0.2)

        if 'efficientnet' in cfg['model_name']:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone_out, cfg['num_classes'])

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
    
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits



taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
taxonomy_df = pd.read_csv(taxonomy_csv)
num_classes = len(taxonomy_df)
print(num_classes)


cfg = {
    'model_name': 'tf_efficientnet_b4_ns',
    'in_channels': 1,
    'num_classes': 206  
}


class CFG:
    """
    Configuration class holding all paths and parameters required for the inference pipeline.
    """
    test_soundscapes = '/kaggle/input/birdclef-2025/train_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/efficient-net-b4-ns-normal/pytorch/default/2'
    
    # Audio parameters
    FS = 32000  
    WINDOW_SIZE = 5  
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512#64
    N_MELS = 148
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (380, 380)
    
    model_name = 'tf_efficientnet_b4_ns'
    in_channels = 1
    device = "cpu" 
    
    # Inference parameters
    batch_size = 16
    use_tta = False  
    tta_count = 3   
    threshold = 0.5
    
    use_specific_folds = False  # If False, use all found models
    folds = [0, 1]  # Used only if use_specific_folds is True
    
    debug = False
    debug_count = 3


fold_paths = [
    '/kaggle/input/efficient-net-b4-ns-normal/pytorch/default/2/model_fold0.pth',
    '/kaggle/input/efficient-net-b4-ns-normal/pytorch/default/2/model_fold1.pth',
    '/kaggle/input/efficient-net-b4-ns-normal/pytorch/default/2/model_fold2.pth',
    '/kaggle/input/efficient-net-b4-ns-normal/pytorch/default/2/model_fold3.pth',
    '/kaggle/input/efficient-net-b4-ns-normal/pytorch/default/2/model_fold4.pth',
]

for i, path in enumerate(fold_paths):
    model = BirdCLEFModel(cfg, num_classes=cfg['num_classes'])
    checkpoint = torch.load(path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, 1, 380, 380)
    onnx_path = f'effnet_b4_fold{i}.onnx'
    
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11, do_constant_folding=True)
    
    print(f"[✓] Exported ONNX for fold {i}: {onnx_path}")


!python -m openvino.tools.mo --input_model effnet_b4_fold0.onnx --compress_to_fp16=False  --output_dir openvino_ir/fold0
!python -m openvino.tools.mo --input_model effnet_b4_fold1.onnx --compress_to_fp16=False  --output_dir openvino_ir/fold1
!python -m openvino.tools.mo --input_model effnet_b4_fold2.onnx --compress_to_fp16=False  --output_dir openvino_ir/fold2
!python -m openvino.tools.mo --input_model effnet_b4_fold3.onnx --compress_to_fp16=False  --output_dir openvino_ir/fold3
!python -m openvino.tools.mo --input_model effnet_b4_fold4.onnx --compress_to_fp16=False  --output_dir openvino_ir/fold4

