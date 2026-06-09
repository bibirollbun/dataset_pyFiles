!pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps
!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/clip/keras/clip-vit-base-patch32/6/model.weights.h5 /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth
!pip install -U /kaggle/input/faiss-gpu-173-python310/faiss_gpu-1.7.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# !pip install umap-learn


import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
import h5py
import dataclasses
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import cv2 as cv
from tqdm import tqdm
from time import time, sleep
# import umap
import gc
import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt

import pycolmap
import sys
sys.path.append('/kaggle/input/imc25-utils')
from database import *
from h5_to_db import *
import metric


device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')


data_dir = "/kaggle/input/image-matching-challenge-2025"
train_labels = pd.read_csv(f"{data_dir}/train_labels.csv")
train_labels


def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=10)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I
    
def get_image_embeddings(image_paths, clip_model, processor, device):
    batch_size = 2048
    num_images = len(image_paths)
    features = []
    for i in range(num_images // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > num_images:
            end = num_images
        images_batch = []
        for image_path in image_paths[start:end]:
            images_batch.append(Image.open(image_path))  

        with torch.no_grad():
            inputs = processor(images=images_batch, return_tensors="pt", padding=True).to(device)
            feature = clip_model.get_image_features(**inputs)
            features.append(feature)

        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{num_images}]")
            
    features = torch.cat(features)
    return features

def construct_text_counterparts(samples, device, is_train=True):
    # Load Pretrained CLIP
    # model_name = "openai/clip-vit-base-patch32"
    model_name = "/kaggle/input/clip-vit/pytorch/b-32-laion2b-s34b-b79k/1"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    # Get image paths from all datasets
    image_paths = []

    
    for dataset, predictions in samples.items():
        images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
        if not is_train and not os.path.isdir(images_dir):
            continue
        paths = [os.path.join(images_dir, p.filename) for p in predictions]
        image_paths.extend(paths)

    def get_text_embeddings():
        nouns = pd.read_csv(f"/kaggle/input/wordnetnouns/WordNetNouns.csv").values
        nouns_num = nouns.shape[0]
        batch_size = 2048
        features = []
        for i in range(nouns_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > nouns_num:
                end = nouns_num
            nouns_batch = nouns[start:end]
            with torch.no_grad():
                prompt = [f"a photo of a {word}" for word in nouns_batch[:, 0]]
                text = processor(text=prompt, return_tensors="pt", padding=True, truncation=True).to(device)
                feature = clip_model.get_text_features(**text)
                features.append(feature)
            if i % 50 == 0:
                print(f"[Completed {i * batch_size}/{nouns_num}]")
        features = torch.cat(features)
        return features, nouns

    # Get text embeddings using CLIP
    text_embeddings, nouns = get_text_embeddings()
    
    # Get image embeddings using CLIP
    image_embeddings = get_image_embeddings(image_paths, clip_model, processor, device)

    # Normalize embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

    text_embeddings = text_embeddings.half()
    image_embeddings = image_embeddings.half()

    n_nouns = text_embeddings.shape[0]
    n_images = image_embeddings.shape[0]
    
    # Find cluster_num semantic clusters in image embeddings based on text embeddings
    cluster_num = n_images//40
    preds = kmeans(image_embeddings.cpu().numpy(), cluster_num)

    image_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()
    for k in range(cluster_num):
        image_centers[k] = image_embeddings[preds == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    # Match nouns to image centers
    similarity = torch.matmul(image_centers, text_embeddings.T)
    softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    # Identify highly distinguishable nouns by choosing topK most confident nouns for each image cluster
    topK = 5
    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        softmax_class = softmax_nouns[:, class_index]
        confidence = softmax_class.max(dim=0)[0]
        rank = torch.argsort(confidence, descending=True)
        selected_idx[class_index[rank[:topK]]] = True
    selected_idx = selected_idx.cpu().numpy()

    print(selected_idx.sum(), "nouns selected.")
    text_embeddings_selected = text_embeddings[selected_idx]

    # Use selected nouns for zero-shot classification
    tau = 0.005
    retrieval_embeddings = []
    batch_size = 8192
    for i in range(n_images // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > n_images:
            end = n_images
            images_batch = image_embeddings[start:end]
        similarity = torch.matmul(image_embeddings[start:end], text_embeddings_selected.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ text_embeddings_selected).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{n_images}]")
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1)
    concat_embeddings = torch.cat([image_embeddings, retrieval_embedding], axis=1)

    # Cleanup
    del clip_model
    torch.cuda.empty_cache()
    gc.collect()
    selected_nouns = [noun for noun, is_selected in zip(nouns[:, 0], selected_idx) if is_selected]
    return concat_embeddings, text_embeddings_selected, selected_nouns

def get_dataset_embeddings(samples, dataset, concat_embeddings):
    # Util to select the right embeddings from concat embeddings by dataset name
    start = 0
    count = 0
    for ds, predictions in samples.items():
        if ds != dataset:
            start += len(predictions)
        else:
            count = len(predictions)
            break
    end = start + count
    embeddings = concat_embeddings[start:end].cpu().numpy()
    return embeddings

def cluster_images(embeddings, n_neighbors, min_dist, metric='cosine'):
    # Dimensionality reduction
    umap_model = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        n_components = 2,
        metric = metric
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings


# Code provided by Octavi Grau https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue
def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def get_global_desc(fnames, device = torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2
    
def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs


def get_image_pairs_shortlist(fnames,
                              descs,
                              sim_th = 0.6, # should be strict
                              min_pairs = 30,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    
    dm = torch.cdist(descs, descs, p=2).numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list

def detect_aliked(img_fnames,
                  feature_dir = '.featureout',
                  num_features = 4096,
                  resize_to = 1024,
                  device=torch.device('cpu')):
    dtype = torch.float32 # ALIKED has issues with float16
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, resize=resize_to).eval().to(device, dtype)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return

def match_with_lightglue(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=25,verbose=True):
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                                "depth_confidence": -1,
                                                 "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(desc1,
                                         desc2,
                                         KF.laf_from_center_scale_ori(kp1[None]),
                                         KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs)  == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print (f'{key1}-{key2}: {n_matches} matches')
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return

def import_into_colmap(img_dir, feature_dir ='.featureout', database_path = 'colmap.db'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
    return

@dataclasses.dataclass
class Prediction:
    image_id: str | None  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None


is_train = False # Set to False if submitting to contest
workdir = '/kaggle/working/result/'
os.makedirs(workdir, exist_ok=True)

if is_train:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')

samples = {}
competition_data = pd.read_csv(sample_submission_csv)
for _, row in competition_data.iterrows():
    # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(
        Prediction(
            image_id=None if is_train else row.image_id,
            dataset=row.dataset,
            filename=row.image
        )
    )

for dataset in samples:
    print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')


# Set to 0 to use DINOv2 embeddings
# Set to 1 to construct text counterparts; 
# Set to 2 to only get image embeddings using CLIP
clip_option = 1

if clip_option == 1:
    # Construct text counterparts on all images (zero-shot classification)
    concat_embeddings, text_embeddings_selected, nouns = construct_text_counterparts(samples, device, is_train)


def visualize_text_counterparts(example_dataset, nouns, samples, concat_embeddings):
    example_embeddings = get_dataset_embeddings(samples, example_dataset, concat_embeddings)
    embedding_dim = 512 #Default embedding dim of CLIP
    image_embeddings = example_embeddings[:, :embedding_dim]
    best_indices = torch.argmax(torch.tensor(image_embeddings).to(device) @ text_embeddings_selected.T, axis=1).cpu().numpy()
    best_nouns = []
    for idx in best_indices:
        best_nouns.append(nouns[idx])
        
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    
    # Display 5 sample images with their best-matching noun
    for i in range(5):
        index = i
        prediction = samples[example_dataset][index]
        image_path = f"{data_dir}/train/{example_dataset}/{prediction.filename}"
        image = Image.open(image_path)
        image = image.resize((300, 300))
    
        axs[0][i].imshow(image)
        gt_label = train_labels[train_labels["image"] == prediction.filename]["scene"].item()
        axs[0][i].set_title(f"{best_nouns[index]}\n({gt_label})", fontsize=12)
        axs[0][i].axis('off')

        prediction2 = samples[example_dataset][-index-1]
        image_path2 = f"{data_dir}/train/{example_dataset}/{prediction2.filename}"
        image2 = Image.open(image_path2)
        image2 = image2.resize((300, 300))
        axs[1][i].imshow(image2)
        gt_label2 = train_labels[train_labels["image"] == prediction2.filename]["scene"].item()
        axs[1][i].set_title(f"{best_nouns[-index-1]}\n({gt_label2})", fontsize=12, fontweight=15)
        axs[1][i].axis('off')
    
    plt.tight_layout()
    plt.show()


if is_train:
    visualize_text_counterparts("imc2023_heritage", nouns, samples, concat_embeddings)


if is_train:
    visualize_text_counterparts("pt_stpeters_stpauls", nouns, samples, concat_embeddings)


if is_train:
    visualize_text_counterparts("stairs", nouns, samples, concat_embeddings)


# Collect embeddings for each method
if is_train:
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    ndatasets = len(train_labels["dataset"].unique())

    # Collect embeddings for all datasets
    umap_embeddings = [[] for i in range(ndatasets)]
    for i, (dataset, group) in enumerate(train_labels.groupby('dataset', sort=False)):
        predictions = samples[dataset] 
        images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
        image_paths = [os.path.join(images_dir, p.filename) for p in predictions]
        
        # Get DINOv2 embeddings
        embeddings1 = get_global_desc(image_paths, device)

        # Get CLIP vision encoder embeddings
        embeddings2 = get_image_embeddings(image_paths, clip_model, processor, device).cpu().numpy()
    
        # Get CLIP simularity matrix embeddings with text counterparts
        embeddings3 = get_dataset_embeddings(samples, dataset, concat_embeddings)

        n_neighbors = 15
        min_dist = 0.5
    
        umap_embeddings1 = cluster_images(embeddings1, n_neighbors, min_dist)
        umap_embeddings2 = cluster_images(embeddings2, n_neighbors, min_dist)
        umap_embeddings3 = cluster_images(embeddings3, n_neighbors, min_dist)
        umap_embeddings[i].append(umap_embeddings1)
        umap_embeddings[i].append(umap_embeddings2)
        umap_embeddings[i].append(umap_embeddings3)


# Plot UMAP visualizations of embeddings 
datasets_to_plot = [
    "ETs",
    "stairs",
    "pt_stpeters_stpauls",
    "pt_sacrecoeur_trevi_tajmahal",
    "pt_brandenburg_british_buckingham",
    "imc2023_heritage"
]
if is_train:
    n = len(datasets_to_plot)
    fig, axs = plt.subplots(n, 4, figsize=(15, 3*n), sharex=True, sharey=True, constrained_layout=True)
    axs[0][0].set_title("DINOv2 Embeddings")
    axs[0][1].set_title("CLIP Vision Embeddings")
    axs[0][2].set_title("CLIP Cross-Modal Embeddings")
    plt_idx = 0
    for i, (dataset, group) in enumerate(train_labels.groupby('dataset', sort=False)):
        if dataset not in datasets_to_plot:
            continue
            
        # Get the scenes for this dataset, in order of appearance
        scene_order = group['scene'].drop_duplicates().tolist()
        
        # Count number of images per scene
        scene_counts = group['scene'].value_counts().loc[scene_order].tolist()

        labels = [j for j, count in enumerate(scene_counts) for _ in range(count)]
        
        axs[plt_idx][0].scatter(
            umap_embeddings[i][0][:,0], 
            umap_embeddings[i][0][:,1],
            c=labels,
            cmap='viridis'
        )
        axs[plt_idx][1].scatter(
            umap_embeddings[i][1][:,0], 
            umap_embeddings[i][1][:,1],
            c=labels,
            cmap='viridis'
        )
        axs[plt_idx][2].scatter(
            umap_embeddings[i][2][:,0], 
            umap_embeddings[i][2][:,1],
            c=labels,
            cmap='viridis'
        )

        axs[plt_idx][0].set_ylabel(dataset, fontsize=10)
        
        for j in range(3):
            if j > 0:
                axs[plt_idx][j].set_ylabel('')
                axs[plt_idx][j].set_yticklabels([])
            if i < ndatasets - 1:
                axs[plt_idx][j].set_xlabel('')
                axs[plt_idx][j].set_xticklabels([])

        # Show legend in 4th column
        handles = []
        nscenes = len(scene_order)
        for t, label in enumerate(scene_order):
            color_range = max(1, nscenes-1)
            handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(t/color_range), markersize=10, label=label)
            handles.append(handle)

        axs[plt_idx][3].axis('off')
        axs[plt_idx][3].legend(handles=handles, title="Scenes", loc='center left', bbox_to_anchor=(0, 0.5))
        plt_idx += 1
        
    plt.suptitle("UMAP Visualization of Image Clusters", x=0.38, fontsize=16)
    plt.show()


# Code adapted from Octavi Grau https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue
datasets_to_process = None #Run on all test datasets
if is_train:
    # Note: When running on the training dataset, the notebook will hit the time limit and die. Use this filter to run on a few specific datasets.
    datasets_to_process = [
    	# New data.
    	# 'amy_gardens',
    	'ETs',
    	# 'fbk_vineyard',
    	# 'stairs',
    	# Data from IMC 2023 and 2024.
        # 'imc2024_dioscuri_baalshamin',
    	# 'imc2023_theather_imc2024_church',
    	# 'imc2023_heritage',
    	# 'imc2023_haiper',
    	# 'imc2024_lizard_pond',
    	# Crowdsourced PhotoTourism data.
    	# 'pt_stpeters_stpauls',
    	# 'pt_brandenburg_british_buckingham',
    	# 'pt_piazzasanmarco_grandplace',
    	# 'pt_sacrecoeur_trevi_tajmahal',
    ]
    
timings = {
    "shortlisting":[],
    "feature_detection": [],
    "feature_matching":[],
    "RANSAC": [],
    "Reconstruction": [],
}
mapping_result_strs = []

start = 0
print (f"Extracting on device {device}")
for dataset, predictions in samples.items():
    if datasets_to_process and dataset not in datasets_to_process:
        print(f'Skipping "{dataset}"')
        continue
    
    images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
    images = [os.path.join(images_dir, p.filename) for p in predictions]

    print(f'\nProcessing dataset "{dataset}": {len(images)} images')

    filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}

    feature_dir = os.path.join(workdir, 'featureout', dataset)
    os.makedirs(feature_dir, exist_ok=True)

    # Select the correct embeddings based on clip_option
    if clip_option == 0:
        embeddings = get_global_desc(images, device)
    if clip_option == 1:
        embeddings = torch.tensor(get_dataset_embeddings(samples, dataset, concat_embeddings)).float()
    elif clip_option == 2:
        model_name = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        clip_model.eval()
        embeddings = get_image_embeddings(images, clip_model, processor, device).cpu().float()

    # Wrap algos in try-except blocks so we can populate a submission even if one scene crashes.
    try:
        t = time()
        index_pairs = get_image_pairs_shortlist(
            images,
            embeddings,
            sim_th = 0.3, # should be strict
            min_pairs = 20, # we should select at least min_pairs PER IMAGE with biggest similarity
            exhaustive_if_less = 20,
            device=device
        )
        timings['shortlisting'].append(time() - t)
        print (f'Shortlisting. Number of pairs to match: {len(index_pairs)}. Done in {time() - t:.4f} sec')
        gc.collect()
    
        t = time()

        detect_aliked(images, feature_dir, 4096, device=device)
        gc.collect()
        timings['feature_detection'].append(time() - t)
        print(f'Features detected in {time() - t:.4f} sec')
        
        t = time()
        match_with_lightglue(images, index_pairs, feature_dir=feature_dir, device=device, verbose=False)
        timings['feature_matching'].append(time() - t)
        print(f'Features matched in {time() - t:.4f} sec')

        database_path = os.path.join(feature_dir, 'colmap.db')
        if os.path.isfile(database_path):
            os.remove(database_path)
        gc.collect()
        sleep(1)
        import_into_colmap(images_dir, feature_dir=feature_dir, database_path=database_path)
        output_path = f'{feature_dir}/colmap_rec_aliked'
        
        t = time()
        pycolmap.match_exhaustive(database_path)
        timings['RANSAC'].append(time() - t)
        print(f'Ran RANSAC in {time() - t:.4f} sec')
        
        # By default colmap does not generate a reconstruction if less than 10 images are registered.
        # Lower it to 3.
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_model_size = 3
        mapper_options.max_num_models = 25
        os.makedirs(output_path, exist_ok=True)
        t = time()
        maps = pycolmap.incremental_mapping(
            database_path=database_path, 
            image_path=images_dir,
            output_path=output_path,
            options=mapper_options)
        sleep(1)
        timings['Reconstruction'].append(time() - t)
        print(f'Reconstruction done in  {time() - t:.4f} sec')
        print(maps)

        clear_output(wait=False)
    
        registered = 0
        for map_index, cur_map in maps.items():
            for index, image in cur_map.images.items():
                prediction_index = filename_to_index[image.name]
                predictions[prediction_index].cluster_index = map_index
                predictions[prediction_index].rotation = deepcopy(image.cam_from_world.rotation.matrix())
                predictions[prediction_index].translation = deepcopy(image.cam_from_world.translation)
                registered += 1
        mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(images)} images with {len(maps)} clusters'
        mapping_result_strs.append(mapping_result_str)
        print(mapping_result_str)
        gc.collect()
    except Exception as e:
        print(e)
        # raise e
        mapping_result_str = f'Dataset "{dataset}" -> Failed!'
        mapping_result_strs.append(mapping_result_str)
        print(mapping_result_str)

print('\nResults')
for s in mapping_result_strs:
    print(s)

print('\nTimings')
for k, v in timings.items():
    print(f'{k} -> total={sum(v):.02f} sec.')


# Must Create a submission file.

array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

submission_file = '/kaggle/working/submission.csv'
with open(submission_file, 'w') as f:
    if is_train:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.image_id},{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')

!head {submission_file}


# Optional: Load a complete submission.csv for testing
# Here, we use a custom dataset called old-submission to upload a submission file
# submission_file = "/kaggle/input/old-submission/submission.csv"


if is_train:
    t = time()
    final_score, dataset_scores = metric.score(
        gt_csv='/kaggle/input/image-matching-challenge-2025/train_labels.csv',
        user_csv=submission_file,
        thresholds_csv='/kaggle/input/image-matching-challenge-2025/train_thresholds.csv',
        mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    print(f'Computed metric in: {time() - t:.02f} sec.')

