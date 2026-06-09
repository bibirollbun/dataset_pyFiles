import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2, nibabel as nib
from torchvision.models import resnet18

# match training architecture
def get_model():
    model = resnet18(weights=None)   # don't download pretrained
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

# load your trained weights
model.load_state_dict(torch.load("/kaggle/input/notebook23cee6a84c/model.pth", map_location=device))
model.eval()
print("✅ Model loaded for inference")



def load_scan_as_mip(uid, scan_dir="/kaggle/input/rsna-intracranial-aneurysm-detection/test"):
    scan_path = f"{scan_dir}/{uid}.nii"
    scan = nib.load(scan_path).get_fdata()
    scan = np.clip(scan, -1000, 2000)
    scan = (scan - scan.min()) / (scan.max() - scan.min())
    mip = np.max(scan, axis=2)
    mip_resized = cv2.resize(mip, (224,224))
    return mip_resized.astype(np.float32)

def predict_scan(uid, model, device):
    img = load_scan_as_mip(uid)
    tensor = torch.tensor(img[None,None,:,:], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return output[1]



import kaggle_evaluation.rsna_inference_server
import shutil, os
import polars as pl

ID_COL = "SeriesInstanceUID"
LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]

# This is what Kaggle will call
def predict(series_path: str) -> pl.DataFrame:
    """Given a DICOM series folder, return predictions for all 15 labels."""
    series_id = os.path.basename(series_path)

    # --- load scan, make mip ---
    try:
        img = load_scan_as_mip(series_id, scan_dir=series_path)   # adjust your load function to use dicoms
        tensor = torch.tensor(img[None,None,:,:], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        aneurysm_prob = float(output[1])
    except Exception as e:
        print(f"Error with {series_id}: {e}")
        aneurysm_prob = 0.5

    # replicate across all required columns
    preds = [aneurysm_prob] * len(LABEL_COLS)

    df = pl.DataFrame([[series_id] + preds], schema=[ID_COL, *LABEL_COLS])

    # required cleanup step
    shutil.rmtree("/kaggle/shared", ignore_errors=True)

    return df.drop(ID_COL)

# launch inference server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()   # production (Kaggle test set)
else:
    inference_server.run_local_gateway()   # local debug


