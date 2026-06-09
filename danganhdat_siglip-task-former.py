# !pip -q install torch


# import torch
# import os
# from glob import glob
# from tqdm import tqdm
# from PIL import Image
# from transformers import AutoProcessor, AutoModel
# import pickle
# from pathlib import Path


# # Load model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "google/siglip2-base-patch16-512" 
# # model_name = "google/siglip2-large-patch16-512"
# # model_name = "google/siglip2-so400m-patch16-512"
# model = AutoModel.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="sdpa"
# )
# model = model.eval()
# next(model.parameters()).dtype
# processor = AutoProcessor.from_pretrained(model_name)

# # Function to encode + normalize an image
# def encode_image(image_path: str):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         embedding = model.get_image_features(**inputs)
#         embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize
#     return embedding.squeeze().cpu().tolist()


# def count_params(module):
#     total = sum(p.numel() for p in module.parameters())
#     trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
#     return total, trainable

# # SigLiP models usually have `vision_model` and `text_model`
# image_total, image_trainable = count_params(model.vision_model)
# text_total, text_trainable = count_params(model.text_model)

# print(f"Image encoder parameters: {image_total:,} (trainable: {image_trainable:,})")
# print(f"Text encoder parameters: {text_total:,} (trainable: {text_trainable:,})")
# print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

# # Image encoder parameters: 93,520,128 (trainable: 93,520,128)
# # Text encoder parameters: 282,303,744 (trainable: 282,303,744)
# # Total model parameters: 375,823,874


# from transformers.image_utils import load_image
# import time

# image = load_image("/kaggle/input/aic25-batch2-keyframes-a/Keyframes_K01/keyframes/K01_V001/001.jpg")
# inputs = processor(images=[image], return_tensors="pt").to(model.device)
# t = time.time()
# with torch.no_grad():
#     image_embeddings = model.get_image_features(**inputs)
# print(time.time() - t)
# print(image_embeddings.shape) # torch.Size([1, 1024])


# # Dataset roots
# base_paths = [
#     "/kaggle/input/aic25-batch2-keyframes-a",
#     "/kaggle/input/aic25-batch2-keyframes-b"
#     "/kaggle/input/aic25-batch2-keyframes-c"
# ]

# # Collect all final folders (leaf dirs containing images)
# final_folders = set()
# for base_path in base_paths:
#     for root, dirs, files in os.walk(base_path):
#         if any(f.lower().endswith(('.jpg', '.jpeg')) for f in files):
#             final_folders.add(root)

# final_folders = sorted(final_folders)

# print(f"Total final folders found: {len(final_folders)}")


# save_dir = Path("/kaggle/working/embeddings")
# save_dir.mkdir(parents=True, exist_ok=True)

# total_images = 0
# total_embeddings = 0

# for folder in tqdm(final_folders, desc="Processing folders", unit="folder"):
#     t = time.time()
#     # Collect images (.jpg + .jpeg)
#     images = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.jpeg"))
#     if not images:
#         continue

#     total_images += len(images)
#     embedding_data = []

#     for image_path in images:
#         try:
#             image_embedding = encode_image(image_path)
#             rel_path = os.path.relpath(image_path, folder)  # relative inside its folder
#             embedding_data.append({"vector": image_embedding, "path": rel_path})
#         except Exception as e:
#             print(f"Failed on {image_path}: {e}")

#     # Save one pickle per folder (checkpoint safe)
#     if embedding_data:
#         folder_name = os.path.basename(folder)
#         out_file = save_dir / f"{folder_name}_siglip_embeddings.pkl"
#         with open(out_file, "wb") as f:
#             pickle.dump(embedding_data, f)
#         total_embeddings += len(embedding_data)
#         print(f"Saved {len(embedding_data)} embeddings → {out_file}")
#     print(time.time() - t)

# print(f"Done.")
# print(f"Total images processed: {total_images}")
# print(f"Total embeddings saved: {total_embeddings}")
# print(f"Pickle files in: {save_dir}")


# Save embeddings as pickle
# embedding_pickle = "/kaggle/working/aic25_batch1_siglip_embedding_normalized.pkl"
# with open(embedding_pickle, "wb") as f:
#     pickle.dump(embedding_data, f)
    
# object_pickle = "/kaggle/working/aic25_batch1_yolo11_object.pkl"
# with open(object_pickle, "wb") as f:
#     pickle.dump(object_data, f)

# print(f"Saved {len(embedding_data)} normalized embeddings to {embedding_pickle }")
# print(f"Saved {len(object_data)} normalized embeddings to {object_pickle}")


# from ultralytics import YOLO

# # Load model
# yolo = YOLO("yolo11l.pt")


# image_dir = "/kaggle/input/aic25-batch1-keyframes/keyframes"
# embedding_data = []
# object_data = []

# folders = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
# total_images = 0

# for folder in tqdm(folders, desc="Processing folders", unit="folder"):
#     images = glob(os.path.join(folder, "*.jpg"))
#     total_images += len(images)
#     for image_path in images:
#         image_embedding = encode_image(image_path)
#         # object_detection = yolo(image_path)
        
#         rel_path = os.path.relpath(image_path, image_dir)
        
#         embedding_data.append({"vector": image_embedding, "path": rel_path})
#         # object_data.append({"object": object_detection,"path": rel_path})


# # Save embeddings as pickle
# embedding_pickle = "/kaggle/working/aic25_batch1_siglip_embedding_normalized.pkl"
# with open(embedding_pickle, "wb") as f:
#     pickle.dump(embedding_data, f)
    
# # object_pickle = "/kaggle/working/aic25_batch1_yolo11_object.pkl"
# # with open(object_pickle, "wb") as f:
# #     pickle.dump(object_data, f)

# print(f"Saved {len(embedding_data)} normalized embeddings to {embedding_pickle }")
# # print(f"Saved {len(object_data)} normalized embeddings to {object_pickle}")


# # Run inference
# results = yolo.predict(source="/kaggle/input/aic25-batch1-keyframes/keyframes/L26_V030/058.jpg", conf=0.25)

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen


!pip install -q ftfy


!git clone https://github.com/janesjanes/tsbir.git


from pathlib import Path
CODE_PATH = Path('/kaggle/working/tsbir/code/')
MODEL_PATH = Path('/kaggle/working/tsbir/model/')
DATA_PATH = Path('/kaggle/working/tsbir/data/')
IMAGE_PATH = Path('/kaggle/working/tsbir/images/')
SKETCH_PATH = Path('/kaggle/working/tsbir/sketches/')


!wget -N https://patsorn.me/projects/tsbir/data/tsbir_model_final.pt -P {MODEL_PATH}


import os
import numpy as np
import json
import torch
import sys
CODE_PATH =  Path("/kaggle/working/tsbir/code")
sys.path.append(CODE_PATH)


max_retries = 10
for attempt in range(1, max_retries + 1):
    try:
        from clip.model import convert_weights, CLIP
        print(f"Import succeeded on attempt {attempt}")
        break
    except ModuleNotFoundError as e:
        print(f"Attempt {attempt}: {e}")
        if attempt < max_retries:
            time.sleep(1)
        else:
            raise RuntimeError(f"Failed to import CLIP after {max_retries} attempts") from e


model_config_file = CODE_PATH / 'training/model_configs/ViT-B-16.json'
model_file = MODEL_PATH / 'tsbir_model_final.pt'


gpu = 0
torch.cuda.set_device(gpu)

with open(model_config_file, 'r') as f:
    model_info = json.load(f)
        
model = CLIP(**model_info)

loc = "cuda:{}".format(gpu)
checkpoint = torch.load(model_file, map_location=loc, weights_only=False)

sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}

model.load_state_dict(sd, strict=False)

model.eval()

model = model.cuda()


import os
import glob
from pathlib import Path
from tqdm import tqdm
import json
import random
import ftfy
import tqdm
import torch
from torch.utils.data import DataLoader
import time
from PIL import Image


def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out
import os

from clip.clip import _transform, load
convert_weights(model)
preprocess_train = _transform(model.visual.input_resolution, is_train=True)
preprocess_val = _transform(model.visual.input_resolution, is_train=False)
preprocess_fn = (preprocess_train, preprocess_val)

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    
class SimpleImageFolder(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
       
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, image_path
       
        
    
    def __len__(self):
        return len(self.image_paths)


# root_dir = Path("/kaggle/input/aic25-batch1-keyframes/keyframes")
# folders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
# print(f"Found {len(folders)} folders")


import os

# Dataset roots
base_paths = [
    # "/kaggle/input/aic25-batch2-keyframes-a",
    # "/kaggle/input/aic25-batch2-keyframes-b",
    # "/kaggle/input/aic25-batch2-keyframes-c",
    # "/kaggle/input/aic25-batch2-keyframes-d",
    # "/kaggle/input/aic25-batch2-keyframes-e",
    
    # "/kaggle/input/aic25-batch2-keyframes-a/Keyframes_K01/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-a/Keyframes_K02/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-a/Keyframes_K03/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-a/Keyframes_K04/keyframes",
    
    "/kaggle/input/aic25-batch2-keyframes-b/Keyframes_K05/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-b/Keyframes_K06/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-b/Keyframes_K07/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-b/Keyframes_K08/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-c/Keyframes_K09/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-c/Keyframes_K10/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-c/Keyframes_K11/keyframes",
    "/kaggle/input/aic25-batch2-keyframes-c/Keyframes_K12/keyframes",
    
    # "/kaggle/input/aic25-batch2-keyframes-d/Keyframes_K13/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-d/Keyframes_K14/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-d/Keyframes_K15/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-d/Keyframes_K16/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-e/Keyframes_K17/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-e/Keyframes_K18/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-e/Keyframes_K19/keyframes",
    # "/kaggle/input/aic25-batch2-keyframes-e/Keyframes_K20/keyframes",
]

# Collect unique folder names containing images
folders = set()
for base_path in base_paths:
    for root, dirs, files in os.walk(base_path):
        if any(f.lower().endswith(('.jpg', '.jpeg')) for f in files): folders.add(root)

folders = sorted(folders)

print(f"Total final folders found: {len(folders)}")
print(folders[:20])  # show first 20 as preview


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


cumulative_loss = 0.0
num_elements = 0.0
all_image_path = []
all_image_features = []
batch_num = 0
total_processed = 0   # <-- new counter
model = model.cuda().eval()

with torch.no_grad():
    for folder in folders:
        images = sorted(glob.glob(str(folder / "*.jpg")))
        if not images:
            continue
        
        print(f"\n{folder.name}: {len(images)} images")

        dataset = SimpleImageFolder(images, transform=preprocess_val)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=None,
            drop_last=False,
        )
        dataloader.num_samples = len(dataset)
        dataloader.num_batches = len(dataloader)
        
        data = DataInfo(dataloader, None)
        batch_num = 0
        for batch in dataloader:
            images_tensor, image_paths = batch
            images_tensor = images_tensor.cuda(gpu, non_blocking=True)

            image_features = model.encode_image(images_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for feat in image_features:
                all_image_features.append(feat.cpu().numpy())
            for path in image_paths:
                all_image_path.append(path)

            total_processed += len(image_paths)   # <-- update counter

            print(f"Batch {batch_num} -- Done | Total processed: {total_processed}")
            batch_num += 1

print(f"\nFinished processing {total_processed} images in total.")


import pickle

# --- Save embeddings ---
embedding_data = [
    {
        "vector": vec.tolist(),  # convert numpy -> list
        "path": path.split("keyframes/")[-1]  # keep only after 'keyframes/'
    }
    for vec, path in zip(all_image_features, all_image_path)
]

embedding_pickle = "/kaggle/working/aic25_taskformer_embeddings_normalized.pkl"
with open(embedding_pickle, "wb") as f:
    pickle.dump(embedding_data, f)

print(f"Saved {len(embedding_data)} embeddings to {embedding_pickle}")
print("Sample entry:", embedding_data[0])


# import torch
# import numpy as np
# from PIL import Image

# # Pick one embedding to test
# sample = embedding_data[-100]  
# test_path = sample["path"]
# test_vector = np.array(sample["vector"], dtype=np.float32)

# # Reconstruct full path (add prefix back)
# full_path = f"/kaggle/input/aic25-batch1-keyframes/keyframes/{test_path}"

# # Load and preprocess
# image = Image.open(full_path).convert("RGB")
# image_tensor = preprocess_val(image).unsqueeze(0).cuda()

# with torch.no_grad():
#     new_feat = model.encode_image(image_tensor)
#     new_feat = new_feat / new_feat.norm(dim=-1, keepdim=True)

# new_vector = new_feat.cpu().numpy().squeeze()

# # Compare cosine similarity
# cosine_sim = np.dot(test_vector, new_vector)

# print("Testing path:", test_path)
# print("Cosine similarity with stored embedding:", cosine_sim)


# !wget -N https://patsorn.me/projects/tsbir/data/tsbir_model_final.pt -P {MODEL_PATH} # TẢI WEIGHT BẰNG TAY RỒI ĐẶT VÔ MODEL_PATH


# class TaskFormerEncoder:

#     def __init__(self, model_file = '../tsbir/model/tsbir_model_final.pt', device=None):
#         gpu = 0
#         model_config_file = '../tsbir/code/training/model_configs/ViT-B-16.json'
        
#         self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
#         with open(model_config_file, 'r') as f:
#             model_info = json.load(f)
#         self.model = CLIP(**model_info)
#         checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
#         sd = checkpoint["state_dict"]
#         if next(iter(sd.items()))[0].startswith('module'):
#             sd = {k[len('module.'):]: v for k, v in sd.items()}
#         self.model.load_state_dict(sd, strict=False)
#         self.model = self.model.to(self.device).eval()
#         self.transformer = _transform(self.model.visual.input_resolution, is_train=False)

#     def get_feature(self, query_sketch, query_text):
#         sketch = Image.open(query_sketch).convert("RGB")
#         img1 = self.transformer(sketch).unsqueeze(0).to(self.device)
#         txt = tokenize([str(query_text)]).to(self.device)
#         with torch.no_grad():
#             sketch_feature = self.model.encode_sketch(img1)
#             text_feature = self.model.encode_text(txt)
#             sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
#             text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
#         return self.model.feature_fuse(sketch_feature, text_feature).squeeze().tolist()


