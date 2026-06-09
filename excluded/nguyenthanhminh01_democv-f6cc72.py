!nvidia-smi


import pandas as pd

# Load your CSVs
# train_df = pd.read_csv('/kaggle/working/train.csv')
# test_df = pd.read_csv('/kaggle/working/test.csv')
# val_df = pd.read_csv('/kaggle/working/val.csv')
train_df = pd.read_csv('/kaggle/input/landmark/train.csv')
test_df = pd.read_csv('/kaggle/input/landmark/test.csv')
val_df = pd.read_csv('/kaggle/input/landmark/val.csv')
# Build mapping
landmark_id_to_idx_4_models = {lid: idx for idx, lid in enumerate(sorted(train_df['landmark_id'].unique()))}
 

# Map class_idx
train_df['class_idx'] = train_df['landmark_id'].map(landmark_id_to_idx_4_models)
test_df['class_idx'] = test_df['landmark_id'].map(landmark_id_to_idx_4_models)
val_df['class_idx'] = val_df['landmark_id'].map(landmark_id_to_idx_4_models)

NUM_CLASSES = len(landmark_id_to_idx_4_models)


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

IMAGE_ROOT = "/kaggle/input/landmark-recognition-2021/train"
CSV_PATH = "/kaggle/input/landmark-labels/train_with_landmark_names_fixed.csv"
IMAGE_SIZE = 224

# === Load CSV ===
df = pd.read_csv(CSV_PATH,encoding = "Latin1")

# Drop missing or malformed rows
df = df.dropna(subset=['id', 'landmark_id'])

# Ensure landmark_id is int
df['landmark_id'] = df['landmark_id'].astype(int)

# === Map landmark_id to class indices ===
landmark_id_to_idx_dolg = {lid: idx for idx, lid in enumerate(sorted(df['landmark_id'].unique()))}
df['class_idx'] = df['landmark_id'].map(landmark_id_to_idx_dolg)


pip install efficientnet_pytorch


import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

NUM_CLASSES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- RESNET VERSION 1 --------
resnet_v1 = models.resnet50(pretrained=False)
in_features = resnet_v1.fc.in_features
resnet_v1.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
resnet_v1.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_resnet-5.pth", map_location=device))
resnet_v1 = resnet_v1.to(device)

# -------- RESNET VERSION 2 --------
resnet_v2 = models.resnet50(pretrained=False)
in_features = resnet_v2.fc.in_features
resnet_v2.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
resnet_v2.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_resnet_ver_2-7.pth", map_location=device))
resnet_v2 = resnet_v2.to(device)


# -------- EFFICIENTNET VERSION 1 --------
efficientnet_v1 = EfficientNet.from_name('efficientnet-b2')
in_features = efficientnet_v1._fc.in_features
efficientnet_v1._fc = nn.Linear(in_features, NUM_CLASSES)
efficientnet_v1.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_efficientnet-2.pth", map_location=device))
efficientnet_v1 = efficientnet_v1.to(device)

# -------- EFFICIENTNET VERSION 2 --------
efficientnet_v2 = EfficientNet.from_name('efficientnet-b2')
in_features = efficientnet_v2._fc.in_features
efficientnet_v2._fc = nn.Linear(in_features, NUM_CLASSES)
efficientnet_v2.load_state_dict(torch.load("/kaggle/input/model-effiecient/best_model_efficientnet_ver_2-2.pth", map_location=device))
efficientnet_v2 = efficientnet_v2.to(device)


pip install gradio


import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import math
import torch.nn.functional as F
from torchvision import transforms
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- ArcFace + DOLG Definitions ----
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        b, c, h, w = local_feat.shape
        global_feat_expanded = global_feat.view(b, c, 1, 1).expand(-1, -1, h, w)
        attn = self.attn(local_feat + global_feat_expanded)
        fused = local_feat * attn + global_feat_expanded * (1 - attn)
        return fused

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, input, labels):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits

class DOLG_ArcFace(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = tv_models.resnet50(pretrained=True)
        self.backbone_common = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1,
            resnet.layer2, resnet.layer3
        )
        self.backbone_global = resnet.layer4
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(2048, embedding_dim)
        self.local_conv = nn.Conv2d(1024, embedding_dim, kernel_size=1)
        self.fusion = AttentionFusion(embedding_dim)
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        shared_feat = self.backbone_common(x)
        global_feat_map = self.backbone_global(shared_feat)
        global_feat = self.global_pool(global_feat_map).view(x.size(0), -1)
        global_feat = self.global_fc(global_feat)
        local_feat = self.local_conv(shared_feat)
        fused_feat = self.fusion(local_feat, global_feat)
        emb = self.head(fused_feat)
        return emb


import collections
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub
# Load train embeddings
with open("/kaggle/input/delf-embeddings/folder (1)/kaggle/working/train_embeddings_final.pkl", "rb") as f:
    train_embeddings = pickle.load(f)

# Fix path
correct_base_path = "/kaggle/input/train-sub-delf-1/train_sub"
train_embeddings["images_paths"] = [
    os.path.join(correct_base_path, *os.path.normpath(p).split(os.sep)[-2:])
    for p in train_embeddings["images_paths"]
]

# TensorFlow DELF
delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

# EfficientNet as feature extractor
from efficientnet_pytorch import EfficientNet
NUM_CLASSES = 100 
efficientnet_v2_delf = EfficientNet.from_name('efficientnet-b2')
in_features = efficientnet_v2_delf._fc.in_features
efficientnet_v2_delf._fc = nn.Linear(in_features, NUM_CLASSES)

# Load weights
checkpoint_path = "/kaggle/input/model-effiecient/best_model_efficientnet_ver_2-2.pth"
efficientnet_v2_delf.load_state_dict(torch.load(checkpoint_path, map_location=device))
efficientnet_v2_delf._fc = nn.Identity()
embedding_model = efficientnet_v2_delf.to(device)
embedding_model.eval()

# Helper: convert to tensor
transform_emb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embeddings(image: Image.Image):
    image_tensor = transform_emb(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = efficientnet_v2_delf(image_tensor).cpu().numpy()
    return emb

def run_delf(image_np):
    if isinstance(image_np, np.ndarray):
        image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    float_img = tf.image.convert_image_dtype(image_tf, tf.float32)
    input_dict = {
        'image': float_img,
        'image_scales': tf.constant([1.0], dtype=tf.float32),
        'max_feature_num': tf.constant(1000, dtype=tf.int32),
        'score_threshold': tf.constant(100.0, dtype=tf.float32)
    }
    with tf.device('/CPU:0'):  # trÃ¡nh lá»—i cuDNN
        return delf(**input_dict)

def delf_rerank(query_img, top_df):
    query_np = np.array(query_img.resize((224, 224)))
    delf_q = run_delf(query_np)

    inliers_list = []
    for path in top_df['image_paths']:
        try:
            img_np = np.array(Image.open(path).resize((224, 224)))
            delf_k = run_delf(img_np)

            d1_tree = cKDTree(delf_q['descriptors'])
            _, indices = d1_tree.query(delf_k['descriptors'], distance_upper_bound=0.8)

            locations_k = np.array([
                delf_k['locations'][i]
                for i in range(len(indices)) if indices[i] != len(delf_q['descriptors'])
            ])
            locations_q = np.array([
                delf_q['locations'][indices[i]]
                for i in range(len(indices)) if indices[i] != len(delf_q['descriptors'])
            ])

            _, inliers = ransac((locations_q, locations_k), AffineTransform,
                                min_samples=3, residual_threshold=20, max_trials=1000)
            total_inliers = sum(inliers)
        except:
            total_inliers = 1
        inliers_list.append(total_inliers)

    top_df['inliers'] = inliers_list
    top_df['reranked_conf'] = np.sqrt(top_df['inliers']) * top_df['cos_similarity']
    top_df = top_df.sort_values("reranked_conf", ascending=False)
    return top_df




# ---- Load Pretrained Models ----
# Binary classification model
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark.pth", map_location=device))
mobilenet_binary.to(device).eval()


dolg_model = DOLG_ArcFace().to(device).eval()
arcface_head = ArcFace(in_features=512, out_features=102).to(device).eval()
ckpt = torch.load("/kaggle/input/resnet-dolg/ResNetDOLG.pth", map_location=device)
dolg_model.load_state_dict(ckpt['model_state_dict'])
arcface_head.load_state_dict(ckpt['arcface_state_dict'])

# Models dictionary
models = {
    "resnet_v1": resnet_v1,
    "resnet_v2": resnet_v2,
    "efficientnet_v1": efficientnet_v1,
    "efficientnet_v2": efficientnet_v2
}

# ---- Transforms ----
transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transforms_dict = {
    "resnet_v1": A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()]),
    "resnet_v2": A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()]),
    "efficientnet_v1": A.Compose([A.Resize(260, 260), A.Normalize(), ToTensorV2()]),
    "efficientnet_v2": A.Compose([A.Resize(260, 260), A.Normalize(), ToTensorV2()]),
    "dolg": A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224), A.Normalize([0.5]*3, [0.5]*3), ToTensorV2()])
}
import pandas as pd

csv_path = "/kaggle/input/landmark-labels/train_with_landmark_names_fixed.csv"
df = pd.read_csv(csv_path, encoding="latin1")

landmark_id_to_name = (
    df.drop_duplicates("landmark_id")[["landmark_id", "landmark_name"]]
    .set_index("landmark_id")["landmark_name"]
    .to_dict()
)


from PIL import Image
import os

def predict(img: Image.Image):
    img = img.convert("RGB")
    img_np = np.array(img)
    final_output = []

    mobilenet_tensor = transform_mobilenet(image=img_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(mobilenet_binary(mobilenet_tensor)).item()
    final_output.append(f"[Binary Classifier] Landmark probability: {prob:.2%}")
    
    if prob < 0.1:
        return "\n".join(final_output) + "\n\nThis image is NOT a landmark.", []

    final_output.append("This image IS a landmark. Running all models...\n")

    top_images = [] 

    model_list = list(models.keys()) + ["dolg", "retrieval"]
    for model_name in model_list:
        debug_log = [f"--- Result from {model_name.upper()} ---"]
        try:
            if model_name == "retrieval":
                query_emb = get_embeddings(img)
                sim_matrix = cosine_similarity(query_emb, train_embeddings["embedded_images"])
                top5_idx = np.argsort(sim_matrix[0])[::-1][:5]
                top_df = pd.DataFrame({
                    "image_paths": [train_embeddings["images_paths"][i] for i in top5_idx],
                    "cos_similarity": [sim_matrix[0][i] for i in top5_idx],
                    "prediction": [train_embeddings["labels"][i] for i in top5_idx],
                })
                idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx_4_models.items()}
                top_df["landmark_id"] = top_df["prediction"].map(idx_to_landmark_id)
                top_df = delf_rerank(img, top_df)
                vote = collections.Counter(top_df["landmark_id"])
                final_landmark_id = vote.most_common(1)[0][0]
                name = landmark_id_to_name.get(final_landmark_id, f"Class {final_landmark_id}")
                debug_log.append(f"Prediction: {name} (ID: {final_landmark_id})")

                # Load top 5 images as PIL
                for path in top_df["image_paths"]:
                    if os.path.exists(path):
                        top_images.append(Image.open(path).convert("RGB"))
            else:
                transform = transforms_dict[model_name]
                img_tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
                model = dolg_model if model_name == "dolg" else models[model_name]
                model.eval()
                with torch.no_grad():
                    if model_name == "dolg":
                        emb = model(img_tensor)
                        logits = arcface_head(emb, torch.tensor([0], device=device))
                        idx_map = landmark_id_to_idx_dolg
                    else:
                        logits = model(img_tensor)
                        idx_map = landmark_id_to_idx_4_models
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
                idx_to_landmark_id = {v: k for k, v in idx_map.items()}
                landmark_id = idx_to_landmark_id.get(pred_idx, pred_idx)
                name = landmark_id_to_name.get(landmark_id, f"Class {landmark_id}")
                debug_log.append(f"Prediction: {name} (ID: {landmark_id}) - Confidence: {confidence:.2%}")
        except Exception as e:
            debug_log.append(f"[ERROR] Failed with {model_name}: {str(e)}")

        final_output.extend(debug_log)
        final_output.append("")

    return "\n".join(final_output), top_images
# --- Launch Gradio ---
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Model Predictions"),
        gr.Gallery(label="Top 5 Retrieved Images", columns=5, height="auto")
    ],
    title="Landmark Recognition System with Binary model Classification",
    description="Upload an image. If it's a landmark, the system predicts using 6 models and shows top 5 similar images from the database."
)
interface.launch()




# ---- Landmark Non landmark ----
# === Replace Mobilenet Binary Classifier with Feature Similarity Check ===
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as tv_models

# ---- Load the precomputed features ----
landmark_features = np.load('/kaggle/input/feature-vector/landmark_features.npy')
non_landmark_features = np.load('/kaggle/input/feature-vector/non_landmark_features.npy')

# ---- Define feature extractor ----
feature_model = tv_models.resnet50(pretrained=True)
feature_model.fc = torch.nn.Identity()
feature_model = feature_model.eval().to(device)

feature_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_feature_tensor(image_pil):
    img_tensor = feature_transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_model(img_tensor).squeeze().cpu().numpy()
        return feature / np.linalg.norm(feature)

def classify_landmark_similarity(image_pil, threshold=0.7):
    new_feat = extract_feature_tensor(image_pil)
    landmark_sims = np.dot(landmark_features, new_feat)
    non_landmark_sims = np.dot(non_landmark_features, new_feat)
    max_landmark_sim = landmark_sims.max() if landmark_sims.size > 0 else 0
    max_non_landmark_sim = non_landmark_sims.max() if non_landmark_sims.size > 0 else 0

    if max_landmark_sim > threshold and max_landmark_sim > max_non_landmark_sim:
        return "Landmark", max_landmark_sim
    elif max_non_landmark_sim > threshold:
        return "Non-Landmark", max_non_landmark_sim
    else:
        return "Unknown", max(max_landmark_sim, max_non_landmark_sim)


dolg_model = DOLG_ArcFace().to(device).eval()
arcface_head = ArcFace(in_features=512, out_features=102).to(device).eval()
ckpt = torch.load("/kaggle/input/resnet-dolg/ResNetDOLG.pth", map_location=device)
dolg_model.load_state_dict(ckpt['model_state_dict'])
arcface_head.load_state_dict(ckpt['arcface_state_dict'])

# Models dictionary
models = {
    "resnet_v1": resnet_v1,
    "resnet_v2": resnet_v2,
    "efficientnet_v1": efficientnet_v1,
    "efficientnet_v2": efficientnet_v2
}

# ---- Transforms ----
transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transforms_dict = {
    "resnet_v1": A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()]),
    "resnet_v2": A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()]),
    "efficientnet_v1": A.Compose([A.Resize(260, 260), A.Normalize(), ToTensorV2()]),
    "efficientnet_v2": A.Compose([A.Resize(260, 260), A.Normalize(), ToTensorV2()]),
    "dolg": A.Compose([A.Resize(256, 256), A.CenterCrop(224, 224), A.Normalize([0.5]*3, [0.5]*3), ToTensorV2()])
}
import pandas as pd

csv_path = "/kaggle/input/landmark-labels/train_with_landmark_names_fixed.csv"
df = pd.read_csv(csv_path, encoding="latin1")

landmark_id_to_name = (
    df.drop_duplicates("landmark_id")[["landmark_id", "landmark_name"]]
    .set_index("landmark_id")["landmark_name"]
    .to_dict()
)

import collections
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub



def predict(img: Image.Image):
    img = img.convert("RGB")
    img_np = np.array(img)

    final_output = []

    # Step 1: Feature Similarity Check (Landmark vs Non-Landmark)
    status, sim_score = classify_landmark_similarity(img)
    final_output.append(f"[Similarity Check] Status: {status} (Sim score: {sim_score:.2%})")

    if status != "Landmark":
        return "\n".join(final_output) + "\n\nThis image is NOT a landmark.", []

    final_output.append("\nThis image IS a landmark. Running all recognition models...\n")
    model_list = list(models.keys()) + ["dolg", "retrieval"]
    retrieved_images = []

    for model_name in model_list:
        debug_log = [f"--- Result from {model_name.upper()} ---"]

        try:
            if model_name == "retrieval":
                query_emb = get_embeddings(img)
                sim_matrix = cosine_similarity(query_emb, train_embeddings["embedded_images"])
                top5_idx = np.argsort(sim_matrix[0])[::-1][:5]
                top_df = pd.DataFrame({
                    "image_paths": [train_embeddings["images_paths"][i] for i in top5_idx],
                    "cos_similarity": [sim_matrix[0][i] for i in top5_idx],
                    "prediction": [train_embeddings["labels"][i] for i in top5_idx],
                })
                idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx_4_models.items()}
                top_df["landmark_id"] = top_df["prediction"].map(idx_to_landmark_id)
                top_df = delf_rerank(img, top_df)
                vote = collections.Counter(top_df["landmark_id"])
                final_landmark_id = vote.most_common(1)[0][0]
                name = landmark_id_to_name.get(final_landmark_id, f"Class {final_landmark_id}")
                debug_log.append(f"Prediction: {name} (ID: {final_landmark_id})")

                for path in top_df["image_paths"]:
                    try:
                        pil_img = Image.open(path).convert("RGB")
                        retrieved_images.append(pil_img)
                    except Exception as e:
                        debug_log.append(f"[WARNING] Failed to load image {path}: {e}")
            else:
                transform = transforms_dict[model_name]
                img_tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
                model = dolg_model if model_name == "dolg" else models[model_name]
                model.eval()
                with torch.no_grad():
                    if model_name == "dolg":
                        emb = model(img_tensor)
                        logits = arcface_head(emb, torch.tensor([0], device=device))
                        idx_map = landmark_id_to_idx_dolg
                    else:
                        logits = model(img_tensor)
                        idx_map = landmark_id_to_idx_4_models
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
                idx_to_landmark_id = {v: k for k, v in idx_map.items()}
                landmark_id = idx_to_landmark_id.get(pred_idx, pred_idx)
                name = landmark_id_to_name.get(landmark_id, f"Class {landmark_id}")
                debug_log.append(f"Prediction: {name} (ID: {landmark_id}) - Confidence: {confidence:.2%}")
        except Exception as e:
            debug_log.append(f"[ERROR] Failed with {model_name}: {str(e)}")

        final_output.extend(debug_log)
        final_output.append("")

    return "\n".join(final_output), retrieved_images
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Recognition Results"),
        gr.Gallery(label="Top 5 Retrieved Images", columns=5, height="auto")
    ],
    title="Landmark Recognition System (Feature Similarity + All Models)",
    description="Upload an image. If it's a landmark, the system uses 6 models to predict and shows top 5 visually similar images."
)
interface.launch()


import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import collections
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub

# ---- ODIN ----
def odin_score(image_tensor, model, temperature=10, epsilon=0.0014):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad = True
    model.eval()
    logits = model(image_tensor)
    logits = logits / temperature
    pred_class = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, pred_class)
    model.zero_grad()
    loss.backward()
    gradient = torch.sign(image_tensor.grad.data)
    perturbed = image_tensor - epsilon * gradient
    perturbed = torch.clamp(perturbed, 0, 1)
    with torch.no_grad():
        logits_perturbed = model(perturbed) / temperature
        softmax_scores = F.softmax(logits_perturbed, dim=1)
        score = torch.max(softmax_scores).item()
    return score

# ---- Main predict function ----
# def predict(img: Image.Image, model_name: str):
#     debug_log = []
#     img = img.convert("RGB")
#     img_np = np.array(img)
#     debug_log.append(f"[DEBUG] Image shape: {img_np.shape}, dtype: {img_np.dtype}")

#     if model_name == "retrieval":
#         query_emb = get_embeddings(img)
#         sim_matrix = cosine_similarity(query_emb, train_embeddings["embedded_images"])
#         top5_idx = np.argsort(sim_matrix[0])[::-1][:5]
#         top_df = pd.DataFrame({
#             "image_paths": [train_embeddings["images_paths"][i] for i in top5_idx],
#             "cos_similarity": [sim_matrix[0][i] for i in top5_idx],
#             "prediction": [train_embeddings["labels"][i] for i in top5_idx],
#         })
#         idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx_4_models.items()}
#         top_df["landmark_id"] = top_df["prediction"].map(idx_to_landmark_id)
#         top_df = delf_rerank(img, top_df)
#         vote = collections.Counter(top_df["landmark_id"])
#         final_landmark_id = vote.most_common(1)[0][0]
#         name = landmark_id_to_name.get(final_landmark_id, f"Class {final_landmark_id}")
#         debug_log.append(f"[DEBUG] Retrieval top prediction: {final_landmark_id} - {name}")
#         return f"Prediction using retrieval:\n{name} (ID: {final_landmark_id})\n\n" 

#     # Classification
#     transform = transforms_dict[model_name]
#     img_tensor = transform(image=img_np)["image"].to(device)
#     debug_log.append(f"[DEBUG] Input tensor shape: {img_tensor.shape}")

#     model = dolg_model if model_name == "dolg" else models[model_name]
#     model.eval()

#     # ODIN score
#     odin_confidence = odin_score(img_tensor, model, temperature=10, epsilon=0.0014)
#     debug_log.append(f"[DEBUG] ODIN score: {odin_confidence:.4f}")

#     with torch.no_grad():
#         img_tensor = img_tensor.unsqueeze(0)
#         if model_name == "dolg":
#             emb = model(img_tensor)
#             logits = arcface_head(emb, torch.tensor([0], device=device))
#         else:
#             logits = model(img_tensor)

#         probs = F.softmax(logits, dim=1)
#         pred_idx = probs.argmax(dim=1).item()
#         softmax_confidence = probs[0, pred_idx].item()

#     # Fusion
#     fusion_score = softmax_confidence * odin_confidence
#     debug_log.append(f"[DEBUG] Softmax confidence: {softmax_confidence:.4f}")
#     debug_log.append(f"[DEBUG] Fusion score (Softmax * ODIN): {fusion_score:.4f}")

#     idx_to_landmark_id = (
#         {v: k for k, v in landmark_id_to_idx_dolg.items()}
#         if model_name == "dolg" else
#         {v: k for k, v in landmark_id_to_idx_4_models.items()}
#     )
#     landmark_id = idx_to_landmark_id.get(pred_idx, pred_idx)
#     name = landmark_id_to_name.get(landmark_id, f"Class {landmark_id}")
#     debug_log.append(f"[DEBUG] Final prediction: {landmark_id} - {name}")

#     return (
#         f"Prediction using {model_name}:\n"
#         f"{name} (ID: {landmark_id})\n"  + "\n".join(debug_log)
    # )

from PIL import Image
import os

def predict(img: Image.Image):
    img = img.convert("RGB")
    img_np = np.array(img)
    final_output = [f"[INFO] Image shape: {img_np.shape}"]
    gallery_images = []

    model_list = list(models.keys()) + ["dolg", "retrieval"]

    for model_name in model_list:
        debug_log = [f"--- Result from {model_name.upper()} ---"]

        try:
            if model_name == "retrieval":
                query_emb = get_embeddings(img)
                sim_matrix = cosine_similarity(query_emb, train_embeddings["embedded_images"])
                top5_idx = np.argsort(sim_matrix[0])[::-1][:5]
                top_df = pd.DataFrame({
                    "image_paths": [train_embeddings["images_paths"][i] for i in top5_idx],
                    "cos_similarity": [sim_matrix[0][i] for i in top5_idx],
                    "prediction": [train_embeddings["labels"][i] for i in top5_idx],
                })
                idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx_4_models.items()}
                top_df["landmark_id"] = top_df["prediction"].map(idx_to_landmark_id)
                top_df = delf_rerank(img, top_df)
                vote = collections.Counter(top_df["landmark_id"])
                final_landmark_id = vote.most_common(1)[0][0]
                name = landmark_id_to_name.get(final_landmark_id, f"Class {final_landmark_id}")
                debug_log.append(f"Prediction: {name} (ID: {final_landmark_id})")

                # Load top 5 retrieved images with caption
                for i, row in top_df.iterrows():
                    try:
                        image_path = row["image_paths"]
                        pil_img = Image.open(image_path).convert("RGB")
                        caption = f"ID: {row['landmark_id']} | Sim: {row['cos_similarity']:.3f}"
                        gallery_images.append((pil_img, caption))
                    except Exception as e:
                        debug_log.append(f"[WARNING] Failed to load image {image_path}: {e}")
            else:
                transform = transforms_dict[model_name]
                img_tensor = transform(image=img_np)["image"].to(device)

                model = dolg_model if model_name == "dolg" else models[model_name]
                model.eval()

                # ODIN score
                odin_confidence = odin_score(img_tensor, model, temperature=10, epsilon=0.0014)
                debug_log.append(f"ODIN confidence: {odin_confidence:.4f}")

                with torch.no_grad():
                    img_tensor = img_tensor.unsqueeze(0)
                    if model_name == "dolg":
                        emb = model(img_tensor)
                        logits = arcface_head(emb, torch.tensor([0], device=device))
                        idx_map = landmark_id_to_idx_dolg
                    else:
                        logits = model(img_tensor)
                        idx_map = landmark_id_to_idx_4_models

                    probs = F.softmax(logits, dim=1)
                    pred_idx = probs.argmax(dim=1).item()
                    softmax_confidence = probs[0, pred_idx].item()
                    fusion_score = softmax_confidence * odin_confidence

                    idx_to_landmark_id = {v: k for k, v in idx_map.items()}
                    landmark_id = idx_to_landmark_id.get(pred_idx, pred_idx)
                    name = landmark_id_to_name.get(landmark_id, f"Class {landmark_id}")

                    debug_log.append(f"Softmax confidence: {softmax_confidence:.4f}")
                    debug_log.append(f"Fusion score: {fusion_score:.4f}")
                    debug_log.append(f"Prediction: {name} (ID: {landmark_id})")
        except Exception as e:
            debug_log.append(f"[ERROR] Failed on model {model_name}: {str(e)}")

        final_output.extend(debug_log)
        final_output.append("")

    return "\n".join(final_output), gallery_images
# Launch
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Recognition Results"),
        gr.Gallery(label="Top 5 Retrieved Images", columns=5, height="auto")
    ],
    title="Landmark Recognition with ODIN and Retrieval Gallery",
    description="Upload an image. The system predicts using 6 models. ODIN is used for OOD detection. The retrieval model shows the top 5 most similar images."
)
interface.launch()


class DOLG_ArcFace(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        # Shared layers
        self.backbone_common = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1,
            resnet.layer2, resnet.layer3
        )

        # ResNet layer4 expects input with 1024 channels (not from local_conv!)
        self.backbone_global = resnet.layer4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(2048, embedding_dim)

        self.local_conv = nn.Conv2d(1024, embedding_dim, kernel_size=1)

        self.fusion = AttentionFusion(embedding_dim)

        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        shared_feat = self.backbone_common(x)  # Output: [B, 1024, H, W]

        # Global branch
        global_feat_map = self.backbone_global(shared_feat)  # Output: [B, 2048, H/2, W/2]
        global_feat = self.global_pool(global_feat_map).view(x.size(0), -1)  # [B, 2048]
        global_feat = self.global_fc(global_feat)  # [B, 512]

        # Local branch
        local_feat = self.local_conv(shared_feat)  # [B, 512, H, W]

        # Fuse
        fused_feat = self.fusion(local_feat, global_feat)  # [B, 512, H, W]
        emb = self.head(fused_feat)  # [B, 512]
        return emb


import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def predict_image(image_path, model, arcface_head, transform, device, idx_to_landmark_id=None, id_to_name=None):
    model.eval()
    arcface_head.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    label_tensor = torch.tensor([0], device=device)

    # Forward pass
    with torch.no_grad():
        embedding = model(input_tensor)
        logits = arcface_head(embedding, label_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_idx = pred.item()
    confidence = conf.item()

    if idx_to_landmark_id:
        pred_landmark_id = idx_to_landmark_id.get(pred_idx, "Unknown ID")
    else:
        pred_landmark_id = pred_idx

    if id_to_name:
        pred_name = id_to_name.get(pred_landmark_id, "Unknown Landmark")
    else:
        pred_name = f"Class {pred_idx}"

    print(f"Predicted: {pred_name} (ID: {pred_landmark_id}) | Confidence: {confidence:.2f}")
    return pred_idx, confidence


# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
NUM_CLASSES = 102
model = DOLG_ArcFace(embedding_dim=512).to(device)
arcface_head = ArcFace(in_features=512, out_features=NUM_CLASSES).to(device)  # Replace NUM_CLASSES

# Load checkpoint
checkpoint = torch.load("/kaggle/input/resnet-dolg/ResNetDOLG.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
arcface_head.load_state_dict(checkpoint['arcface_state_dict'])

# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Optional mappings
idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx.items()}
landmark_id_to_name = {
    27: "Isa Khan Niyazi's tomb",
    # ... fill from your metadata
}

# ---- Predict ----
image_path = "/kaggle/input/test-landmark-images/golden_gate_foggy.jpg"
predict_image(
    image_path,
    model,
    arcface_head,
    transform,
    device,
    idx_to_landmark_id=idx_to_landmark_id,
    id_to_name=landmark_id_to_name
)



import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
#from model import DOLG_ArcFace, ArcFace  # Make sure this file exists

# ============ Device ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Pretrained Classification Models ============
model_dict = {
    "resnet18": tv_models.resnet18(pretrained=True).to(device),
    "resnet50": tv_models.resnet50(pretrained=True).to(device),
    "efficientnet_b0": tv_models.efficientnet_b0(pretrained=True).to(device),
    "efficientnet_b3": tv_models.efficientnet_b3(pretrained=True).to(device)
}

# ============ Add DOLG Model ============
embedding_dim = 512
num_classes = 102  # set this according to your dataset

dolg_model = DOLG_ArcFace(embedding_dim=embedding_dim).to(device)
arcface_head = ArcFace(in_features=embedding_dim, out_features=num_classes).to(device)

# Load weights
checkpoint = torch.load("/kaggle/input/resnet-dolg/kaggle/working/best_model.pth", map_location=device)
dolg_model.load_state_dict(checkpoint["model_state_dict"])
arcface_head.load_state_dict(checkpoint["arcface_state_dict"])
dolg_model.eval()
arcface_head.eval()

model_dict["resnet_dolg"] = (dolg_model, arcface_head)  # special handling for this one

# ============ Load Binary MobileNet ============
mobilenet_binary = tv_models.mobilenet_v2(pretrained=False)
mobilenet_binary.classifier[1] = nn.Linear(mobilenet_binary.last_channel, 1)
mobilenet_binary.load_state_dict(torch.load("/kaggle/input/mobilenet-landmark-binary-classification/mobilenet_landmark.pth", map_location=device))
mobilenet_binary.to(device)
mobilenet_binary.eval()

# ============ Transforms ============
transform_efficientnet = A.Compose([
    A.Resize(260, 260),
    A.Normalize(),
    ToTensorV2()
])

transform_resnet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

transform_mobilenet = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ============ Inference Function ============
def predict(img: Image.Image, model_name: str):
    img_np = np.array(img)

    # Step 1: Run binary classifier
    binary_transformed = transform_mobilenet(image=img_np)
    binary_tensor = binary_transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        binary_output = mobilenet_binary(binary_tensor)
        prob = torch.sigmoid(binary_output).item()
        is_landmark = prob >= 0.5

    if not is_landmark:
        return f"The image does not contain a landmark. (Confidence: {prob:.2f})"

    # Step 2: Classify landmark
    transform = transform_efficientnet if model_name.startswith("efficientnet") else transform_resnet
    transformed = transform(image=img_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Handle ResNet + DOLG differently
    if model_name == "resnet_dolg":
        dolg_model, arcface_head = model_dict["resnet_dolg"]
        with torch.no_grad():
            embedding = dolg_model(img_tensor)
            logits = arcface_head(embedding, torch.zeros(1, dtype=torch.long).to(device))  # dummy label
            prob_class = F.softmax(logits, dim=1)
            conf, pred = torch.max(prob_class, dim=1)
        return f"Landmark detected (Confidence: {prob:.2f})\nPredicted class (ResNet+DOLG): {pred.item()} (Conf: {conf.item():.2f})"
    
    # Standard models
    model = model_dict[model_name]
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
        idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx.items()}
        pred_landmark = idx_to_landmark_id[pred]

    return f"Landmark detected (Confidence: {prob:.2f})\nPredicted class ({model_name}): {pred_landmark}"

# ============ Gradio Interface ============
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=list(model_dict.keys()), label="Choose Model")
    ],
    outputs="text",
    title="Landmark Classification Demo (ResNet + DOLG + ArcFace Integrated)"
)

interface.launch()


def get_image(path, resize = False, reshape = False, target_size = None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, dsize = (target_size, target_size))
    if reshape:
        img = tf.reshape(img, [1, target_size, target_size, 3])
    return img


def get_query_image(img, resize=False, reshape=False, target_size=None):
    # Náº¿u lÃ  PIL Image, chuyá»ƒn sang numpy
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Ä�áº£m báº£o áº£nh á»Ÿ dáº¡ng RGB
    if img.shape[-1] == 4:  # RGBA
        img = img[:, :, :3]
    
    if resize:
        img = cv2.resize(img, dsize=(target_size, target_size))
    
    if reshape:
        img = tf.reshape(img, [1, target_size, target_size, 3])
    
    return img


def get_embeddings(model, image_paths, input_size, as_df = True):
    embeddings = {}
    embeddings['images_paths'] = []
    embeddings['embedded_images'] = []
    
    target_dir = os.path.split(os.path.split(image_paths[0])[0])[0]
    
    print(f"Retrieving embeddings for {target_dir} with {model.name}...")
    for image_path in tqdm(image_paths):
        embeddings['images_paths'].append(image_path)
        embedded_image = model.predict(get_image(image_path,
                                                 resize = True,
                                                 reshape = True,
                                                 target_size = input_size))
        embeddings['embedded_images'].append(embedded_image)
    
    if as_df:
        embeddings = pd.DataFrame(embeddings)
    
    return embeddings


def get_query_embedding(model, images, input_size):
    embeddings = []
    for img in images:
        processed_img = get_query_image(img, resize=True, reshape=True, target_size=input_size)
        embedded = model.predict(processed_img)
        embeddings.append(embedded)
    return embeddings


# Get similarities between query key pair
def get_similarities(query, key):
    '''
    Get cosine similarity matrix between query and key pairs
    Arguments:
    query, key: embedded images
    '''
    query_array = np.stack(query.tolist()).reshape(query.shape[0],
                                                   query[0].shape[1])
    key_array = np.stack(key.tolist()).reshape(key.shape[0],
                                               key[0].shape[1])
    
    # Initializing similarity matrix
    similarity = np.zeros((query_array.shape[0], key_array.shape[0]))
    
    # Getting pairwise similarities
    print(f"Getting pairwise {query_array.shape[0]} query: {key_array.shape[0]} key similarities...")
    for query_index in tqdm(range(query_array.shape[0])):
        similarity[query_index] = 1 - spatial.distance.cdist(query_array[np.newaxis, query_index, :],
                                                             key_array,
                                                             'cosine')[0]
    return similarity


import os
from tqdm.notebook import tqdm
train_img_paths = []
train_df = pd.read_csv("/kaggle/input/landmark/train.csv", header=0)


for i, (idx, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df))):
    image_id = row['id']
    # Extract subfolders based on first 3 characters
    subfolder = f"{image_id[0]}/{image_id[1]}/{image_id[2]}"
    image_path = f"/kaggle/input/landmark-recognition-2021/train/{subfolder}/{image_id}.jpg"

    # Chá»‰ thÃªm náº¿u file tá»“n táº¡i
    if os.path.exists(image_path):
        train_img_paths.append(image_path)
    else:
        print(f"â�Œ File not found: {image_path}")


!pip install -Uq tensorflow
import os; os._exit(00)  # khá»Ÿi Ä‘á»™ng láº¡i kernel


from tensorflow.keras.models import load_model
embedding_model = load_model("/kaggle/input/delf-model/efficientnet_embedding_model-2.h5", compile=False)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 100  # <-- báº¡n cáº§n Ä‘iá»�n Ä‘Ãºng sá»‘ class khi training

# Rebuild model
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', pooling='avg')
x = tf.keras.layers.Dense(512, activation='relu', name='embedding_512')(base_model.output)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Embedding model
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer('embedding_512').output,
    name="EfficientNetB0_embed512"
)


import cv2
IMG_SIZE = 224
train_embeddings = get_embeddings(model = embedding_model,
                                 image_paths = train_img_paths,
                                 input_size = IMG_SIZE)



# Calculating confidence score per submission
def confidence_top(query = None, key = None, similarity = None, query_image_index = None, top = 5):
    '''
    Arguments:
    query_image_index = index of query image on similarity matrix query axis
    Return confidence scores for top N predictions
    '''
    query_paths = query['images_paths']
    key_paths = key['images_paths']
    
    similar_n = np.argsort(similarity[query_image_index])[::-1][:top]
    
    confidence_df = {}    
    confidence_df['top_similar'] = []
    for similar in similar_n:
        confidence_df['top_similar'].append(similar)

    confidence_df['image_paths'] = []
    for similar in similar_n:
        similar_image_path = key_paths[similar]
        confidence_df['image_paths'].append(similar_image_path)    
        
    confidence_df['prediction'] = []
    for similar in similar_n:
        similar_image_path = key_paths[similar]
        y = int(os.path.split(os.path.split(similar_image_path)[0])[1])
        idx_to_landmark_id = {v: k for k, v in landmark_id_to_idx.items()}
        pred_landmark = idx_to_landmark_id[y]
        confidence_df['prediction'].append(pred_landmark)  
    
    confidence_df['cos_similarity'] = []
    for similar in similar_n:
        confidence_df['cos_similarity'].append(similarity[query_image_index][similar]) 
    
    return pd.DataFrame(confidence_df)


from absl import logging
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.util import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import tensorflow_hub as hub
from six.moves.urllib.request import urlopen
delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']


# DELF module
def run_delf(image):
    '''
    Apply DELF module to the input image
    Arguments:
    image: np.array resized image
    '''
    float_image = tf.image.convert_image_dtype(image, tf.float32)

    return delf(
      image = float_image,
      score_threshold = tf.constant(100.0),
      image_scales = tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num = tf.constant(1000))


DELF_IMG_SIZE = 600
def delf_rerank(img,query = None, key = None, query_image_index = None, confidence_df = None, re_sort = True):
    distance_threshold = 0.8
    query_paths = query['images_paths']
    key_paths = key['images_paths']
    
    query_image = get_query_image(img,
                            resize = True,
                            target_size = DELF_IMG_SIZE)
    
    delf_result_query = run_delf(query_image)
    
    # Read query features
    num_features_query = delf_result_query['locations'].shape[0]
    
    inliers_list = []
    print(f"Retrieving local features for top {len(confidence_df['image_paths'])} key images...")
    for image_path in tqdm(confidence_df['image_paths']):
        key_image = get_image(image_path,
                          resize = True,
                          target_size = DELF_IMG_SIZE)
        
        delf_result_key = run_delf(key_image)
    
        # Read key features
        num_features_key = delf_result_key['locations'].shape[0]

        # Find nearest-neighbor matches using a KD tree.
        d1_tree = cKDTree(delf_result_query['descriptors'])
        _, indices = d1_tree.query(
          delf_result_key['descriptors'],
          distance_upper_bound=distance_threshold)

        # Select feature locations for putative matches.
        locations_k_to_use = np.array([
          delf_result_key['locations'][i,]
          for i in range(num_features_key)
          if indices[i] != num_features_query
        ])
        locations_q_to_use = np.array([
          delf_result_query['locations'][indices[i],]
          for i in range(num_features_key)
          if indices[i] != num_features_query
        ])

        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac(
              (locations_q_to_use, locations_k_to_use),
              AffineTransform,
              min_samples=3,
              residual_threshold=20,
              max_trials=1000)
        except:
            inliers = [0]
        
        # Handling 0 inliers
        try:
            total_inliers = sum(inliers)
            inliers_list.append(total_inliers)
        except:
            inliers_list.append(1) # Appending inlier = 1 to avoid null confidence
    
    confidence_df['inliers'] = inliers_list
    
    original_confidence = confidence_df['inliers']
    reranked_confidence = np.sqrt(original_confidence) * confidence_df['cos_similarity']
    confidence_df['reranked_conf'] = reranked_confidence
    
    if re_sort:
        confidence_df.sort_values('reranked_conf', ascending = False, inplace = True)
    
    return confidence_df


import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image as Image

def recognize_and_visualize_gradio(img, 
                                   train_embeddings, 
                                   model, 
                                   top_n=5, 
                                   use_rerank=True):
    # Step 1: Get query embedding
    query_embedding = get_query_embeddings(model=model,
                                           img=img,
                                           input_size=IMG_SIZE)

    # Step 2: Cosine similarity
    sim_matrix = get_similarities(query_embedding['embedded_images'],
                                   train_embeddings['embedded_images'])

    # Step 3: Top-N predictions
    confidence_df = confidence_top(query=query_embedding,
                                   key=train_embeddings,
                                   similarity=sim_matrix,
                                   query_image_index=0,
                                   top=top_n)

    # Step 4: Reranking (náº¿u cáº§n)
    if use_rerank:
        confidence_df = delf_rerank(query=query_embedding,
                                    key=train_embeddings,
                                    query_image_index=0,
                                    confidence_df=confidence_df,
                                    re_sort=True)

    # Step 5: Váº½ áº£nh truy váº¥n
    query_image = get_query_image(img, resize=True, target_size=DELF_IMG_SIZE)
    fig1 = plt.figure(figsize=(4, 4))
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis("off")
    
    # LÆ°u ra buffer
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches='tight')
    plt.close(fig1)
    buf1.seek(0)
    query_image_pil = Image.open(buf1)

    # Step 6: Váº½ áº£nh tÆ°Æ¡ng tá»±
    fig2, axs = plt.subplots(1, top_n, figsize=(15, 5))
    for i in range(top_n):
        img_path = confidence_df['image_paths'].iloc[i]
        similar_img = get_image(img_path, resize=True, target_size=DELF_IMG_SIZE)
        axs[i].imshow(similar_img)
        axs[i].set_title(f"ID: {confidence_df['prediction'].iloc[i]}\nSim: {confidence_df['cos_similarity'].iloc[i]:.2f}")
        axs[i].axis("off")

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    similar_images_pil = Image.open(buf2)

    # Step 7: Tráº£ káº¿t quáº£
    predicted_landmark = confidence_df['prediction'].iloc[0]
    return predicted_landmark, query_image_pil, similar_images_pil


interface = gr.Interface(
    fn=lambda img: recognize_and_visualize_gradio(img, train_embeddings, model, top_n=5),
    inputs=gr.Image(type="pil", label="Upload Query Image"),
    outputs=[
        gr.Label(label="Predicted Landmark ID"),
        gr.Image(label="Query Image"),
        gr.Image(label="Top Similar Images")
    ],
    title="ğŸ”� Landmark Recognition",
    description="Upload an image to find its most similar landmarks"
)

interface.launch()

