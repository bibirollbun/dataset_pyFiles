!pip install -q segmentation-models-pytorch


import torch
import torch.nn as nn
import numpy as np
import pydicom
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101, densenet121
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= ğŸ”¹ Step 1: Load All Models Once
def load_ensemble_models():
    # Transforms
    transform_chexnet = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])

    # Segmentation model
    seg_model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
    seg_model.load_state_dict(torch.load("/kaggle/input/x-ray-segmention-model/xray_Segmention_model.pth", map_location=DEVICE))
    seg_model.eval().to(DEVICE)

    # Classification models
    def get_resnet_model():
        model = resnet101(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
        return model.to(DEVICE)

    def get_chexnet_model():
        model = densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
        return model.to(DEVICE)

    model_seg = get_resnet_model()
    model_seg.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/best_fc_only_resnet101_epoch10_Segmentation.pth", map_location=DEVICE))
    model_seg.eval()

    model_chex_all = get_chexnet_model()
    model_chex_all.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch7.pth", map_location=DEVICE))
    model_chex_all.eval()

    model_chex_2 = get_chexnet_model()
    model_chex_2.load_state_dict(torch.load("/kaggle/input/models-normal-vs-abnormal/chexnet_fc_only_epoch9_2classes.pth", map_location=DEVICE))
    model_chex_2.eval()

    return {
        "transform_chexnet": transform_chexnet,
        "transform_seg": transform_seg,
        "seg_model": seg_model,
        "model_seg": model_seg,
        "model_chex_all": model_chex_all,
        "model_chex_2": model_chex_2
    }

# ========= ğŸ”¹ Step 2: Predict Single DICOM Image
def predict_image_class(dicom_path, models_dict, threshold=0.71):
    # Load image
    dicom = pydicom.dcmread(dicom_path)
    image_array = dicom.pixel_array.astype(np.float32)
    image_array -= image_array.min()
    image_array /= image_array.max()
    image_array *= 255
    image_array = image_array.astype(np.uint8)

    if len(image_array.shape) == 2:
        image_pil = Image.fromarray(image_array).convert("RGB")
    else:
        image_pil = Image.fromarray(image_array)

    # Transforms
    transform_chexnet = models_dict["transform_chexnet"]
    transform_seg = models_dict["transform_seg"]

    img_chexnet = transform_chexnet(image_pil).unsqueeze(0).to(DEVICE)
    img_seg_input = transform_seg(image_pil).unsqueeze(0).to(DEVICE)

    # Apply segmentation mask
    with torch.no_grad():
        mask = torch.sigmoid(models_dict["seg_model"](img_seg_input)).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.float32)

    image_resized = image_pil.resize((512, 512))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    masked_image = image_np * np.expand_dims(mask, axis=-1)
    masked_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
    img_segmented = transform_chexnet(masked_pil).unsqueeze(0).to(DEVICE)

    # Get probabilities from each model
    def get_prob(model, img_tensor):
        with torch.no_grad():
            out = model(img_tensor)
            return torch.softmax(out, dim=1)[0, 1].item()

    p1 = get_prob(models_dict["model_seg"], img_segmented)
    p2 = get_prob(models_dict["model_chex_all"], img_chexnet)
    p3 = get_prob(models_dict["model_chex_2"], img_chexnet)

    probs = [p1, p2, p3]
    final_prob = np.mean(probs)
    pred_class = int(final_prob >= threshold)
    label = "Abnormal" if pred_class else "Normal"

    # print(f"\nğŸ“� DICOM File: {dicom_path}")
    # print(f"ğŸ“Š Ensemble Probability: {final_prob:.4f}")
    # print(f"ğŸ§  Prediction: {label}")

    return label
#once
models_dict = load_ensemble_models()



predict_image_class("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/000434271f63a053c4128a0ba6352c7f.dicom", models_dict)

