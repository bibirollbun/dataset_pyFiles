import cv2
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import daisy
from tqdm import tqdm
import copy
from skimage import segmentation, color, filters
from skimage.segmentation import slic,  mark_boundaries
from skimage.measure import regionprops
from skimage import img_as_float
import pydot
import networkx as nx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
#import hdbscan


class CFG:
    img_size = (640, 640)
    label_thr = 30
    
    #freak parameter
    freak_thr = 40
    max_kps = 1000
    grid_size = 2
    
    #daisy parameter
    patch_size = 64
    step = 8
    radius = 16


def calibrate_target(im, loc, size=(640, 640)):
    H, W, C = im.shape
    loc[1] = (loc[1]/H) * size[0]
    loc[2] = (loc[2]/H) * size[1]
    return loc


def select_dense_keypoints(keypoints, max_keypoints=500, grid_size=8):
    """
    Selects a dense set of keypoints while preventing excessive clustering.
    Uses a soft clustering approach with grid-based binning.
    """
    # Sort by response (strongest first)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    
    if len(keypoints) <= max_keypoints:
        return keypoints  # If already within limit, return as is

    # Image size (assume square)
    img_size = 640  # Adjust if needed
    cell_size = img_size // grid_size  # Grid divisions

    # Organize keypoints into a grid
    keypoint_grid = {}
    
    for kp in keypoints:
        cell_x = int(kp.pt[0] // cell_size)
        cell_y = int(kp.pt[1] // cell_size)
        cell = (cell_x, cell_y)

        if cell not in keypoint_grid:
            keypoint_grid[cell] = []
        keypoint_grid[cell].append(kp)

    # Select a dense but controlled number of keypoints from each grid cell
    selected_keypoints = []
    per_cell_limit = max_keypoints // (grid_size * grid_size)  # Even distribution

    for cell, kps in keypoint_grid.items():
        # Keep the strongest ones from each grid cell
        selected_keypoints.extend(kps[:per_cell_limit])

    return selected_keypoints[:max_keypoints]  # Ensure total does not exceed max limit


def extract_daisy_at_keypoints(image, keypoints, patch_size=64, step=4, radius=16):
    '''
    patch_size: expand a window through a point 
    step: Computes DAISY descriptors at every 4 pixels
    '''
    descriptors = []
    keypoints_valid = []

    for (y, x) in tqdm(keypoints):
        y = int(y)
        x = int(x)
        half_size = patch_size // 2

        # Ensure keypoint is within image boundaries
        if x - half_size < 0 or y - half_size < 0 or x + half_size >= image.shape[1] or y + half_size >= image.shape[0]:
            continue  # Skip keypoints too close to the edge

        # Extract patch around keypoint
        patch = image[y - half_size:y + half_size, x - half_size:x + half_size]

        # Compute DAISY descriptor for the patch
        daisy_desc = daisy(
            patch[..., 0], step=step, radius=radius, rings=2, histograms=6, orientations=8)

        # Flatten descriptor (single feature vector per keypoint)
        descriptors.append(daisy_desc)
        keypoints_valid.append((y, x))

    descriptors = np.stack(descriptors, axis=0)
    keypoints_valid = np.array(keypoints_valid)
    return descriptors, keypoints_valid


def compute_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1-p2)

def determine_label(p1, p2, thr = 10):
    #1000 A˚=100nm=10 pixels
    dist = compute_dist(p1, p2)
    return (dist<thr).astype('int32')


def labeling(loc, kp, thr=10, use_pt=True):
    loc = loc[1:]
    labels = []
    for kp_ in kp:
        if use_pt:
            label = determine_label(kp_.pt, loc, thr)
        else:
            label = determine_label(kp_, loc, thr)
        labels.append(label)
    labels = np.array(labels)
    return labels


# Create a graph from superpixel regions for graph cuts refinement
def graph_cut_refinement(image, segments):
    # Number of unique segments (superpixels)
    num_segments = np.max(segments) + 1
    
    # Create a graph (networkx)
    G = nx.Graph()
    
    # Add nodes: each superpixel is a node
    for i in range(num_segments):
        region_pixels = np.where(segments == i)
        avg_color = np.mean(image[region_pixels], axis=0)
        G.add_node(i, color=avg_color)  # Add average color as a feature

    # Add edges: between neighboring superpixels
    for i in range(num_segments):
        region_pixels = np.where(segments == i)
        neighbors = get_neighbors(segments, i)
        
        for neighbor in neighbors:
            # Add edge between superpixel i and its neighbor
            G.add_edge(i, neighbor, weight=calculate_edge_weight(image, i, neighbor))
    
    # Apply graph cuts or CRF refinement (simplified here, requires advanced technique)
    refined_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(num_segments):
        region_pixels = np.where(segments == i)
        refined_image[region_pixels] = G.nodes[i]['color']

    return refined_image

# Function to get neighbors of a superpixel
def get_neighbors(segments, superpixel):
    neighbors = set()
    # Search 4-connected neighbors
    # Here, a more advanced method would be needed to find real neighbors
    return neighbors

# Calculate edge weight between superpixels based on color difference
def calculate_edge_weight(image, superpixel1, superpixel2):
    color1 = np.mean(image[np.where(segments == superpixel1)], axis=0)
    color2 = np.mean(image[np.where(segments == superpixel2)], axis=0)
    return np.linalg.norm(color1 - color2)  # Euclidean distance in color space


def increase_contrast(image):
    image = image.astype('float32')
    #image[image>200] *=10
    image[image<64] *=0
    return image.astype('uint8')

def GraphCutsCRF(im, n_segments=100, compactness=10, sigma=1):
    im = increase_contrast(im)
    im = cv2.GaussianBlur(im[...,0], (7, 7), 5)[..., None]
    image = np.concatenate([im]*3, axis=-1)
    segments = slic(image, n_segments=100, compactness=compactness, sigma=sigma)
    plt.imshow(mark_boundaries(im.repeat(3, axis=-1), segments))
    plt.show()
    refined_image_graph_cut = graph_cut_refinement(image, segments)[..., :1]
    plt.imshow(refined_image_graph_cut)
    plt.show()
    output = []
    for i, region_idx in enumerate((np.unique(refined_image_graph_cut))):
        output.append((refined_image_graph_cut==region_idx).astype('float32'))
    output = np.concatenate(output, axis=-1)
    return output.argmax(axis=-1, keepdims=True)


def extract_patch(image, keypoints, patch_size=128):
    H, W = image.shape[:2]  # Get image dimensions
    half_size = patch_size // 2
    patches = []
    image_with_boxes = image.copy()
    image = image.astype('float32')/255.
    for x, y in keypoints:
        x, y = int(x), int(y)
        # Adjust x to keep patch inside image
        x_start = max(0, x - half_size)
        x_end = min(W, x + half_size)
        if x_end - x_start < patch_size:
            x_start = max(0, x_end - patch_size)
            x_end = x_start + patch_size

        # Adjust y to keep patch inside image
        y_start = max(0, y - half_size)
        y_end = min(H, y + half_size)
        if y_end - y_start < patch_size:
            y_start = max(0, y_end - patch_size)
            y_end = y_start + patch_size
        
        # Extract the patch
        patch = image[y_start:y_end, x_start:x_end, 0]
        #patch = cv2.resize(patch, (half_size, half_size))
        patches.append(patch)
        cv2.rectangle(image_with_boxes, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
    patches = np.stack(patches, axis=0)
    plt.imshow(image_with_boxes)
    plt.show()
    return patches


def edge_keypoint(image):
    edges = cv2.Canny(image, 300, 300)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image, None)
    edge_points = np.array([kp.pt for kp in keypoints if edges[int(kp.pt[1]), int(kp.pt[0])] > 0])
    return edge_points


def point_feature_extractor(path, loc=None):
    im = cv2.imread(path)
    im = np.transpose(im, (1, 0, 2))

    if loc is not None:
        #calibrate location
        loc =  calibrate_target(im, loc, CFG.img_size)
        
    #resize
    im = cv2.resize(im, CFG.img_size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]

    #mask generaton
    mask =  GraphCutsCRF(im, 5, 10, 1)

    #keypoint detection
    fast = cv2.FastFeatureDetector_create(threshold=CFG.freak_thr, nonmaxSuppression=True)
    kp = fast.detect(im, None)
    kp = select_dense_keypoints(kp, max_keypoints=CFG.max_kps, grid_size=CFG.grid_size)
    freak = cv2.xfeatures2d.FREAK_create()
    kp, descriptors = freak.compute(im, kp)
    kp = np.array([kp_.pt for kp_ in kp])
    kp_filtered = filter_keypoint(kp, mask)


    #extract patches
    patches = extract_patch(im, kp_filtered, CFG.patch_size)

    #labeling
    if loc is not None:
        #labeling
        labels = labeling(loc, kp_filtered, CFG.label_thr, False)
        return patches, kp_filtered, labels
    return patches, kp_filtered


def determine_round(kp):
    pipeline = Pipeline([
        ('normalize', StandardScaler()),
        ('dbscan', hdbscan.HDBSCAN(min_cluster_size=70, gen_min_span_tree=True))
    ])
    pipeline.fit(kp)
    labels = pipeline[1].labels_
    return 1 in np.unique(labels).tolist(), labels


def filter_keypoint(kp, mask):
    plt.imshow(mask)
    if np.unique(mask).shape[0]==1:
        return kp
    valid_kp = []
    # round_flag, cluster_labels = determine_round(kp)
    # n_rounds = 1 if round_flag else 2
    # print(f'rounds: {n_rounds}')
    for i in range(80):
        mask_i = mask==i
        for j, kp_ in enumerate(kp):
            #is_outlier = cluster_labels[j]
            x, y = int(kp_[0]), int(kp_[1])
            is_valid = mask_i[y, x, 0]
            if is_valid:
                valid_kp.append(kp_)
    return np.array(valid_kp)


#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_01a877_z0148_y0638_x0286.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_00e463_z0218_y0379_x0144.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_08446f_z0239_y0740_x0054.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_0fab19_z0163_y0516_x0681.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_0eb994_z0157_y0835_x0630.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_0da370_z0031_y0356_x0636.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_0de3ee_z0201_y0642_x0801.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_0f9df0_z0097_y0174_x0599.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_0fe63f_z0201_y0362_x0265.jpg'
#path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/train/tomo_1b82d1_z0204_y0697_x0680.jpg'
path = '/kaggle/input/yolo-dataset-byu/yolo_dataset/images/val/tomo_05f919_z0125_y0471_x0845.jpg'


im = cv2.imread(path)
im = np.transpose(im, (1, 0, 2))

#calibrate
loc = path.split('_')[3:]
loc[-1] = int(loc[-1][1:-4])
loc[0] = int(loc[0][1:])
loc[1] = int(loc[1][1:])


#resize
im = cv2.resize(im, (640, 640))
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]


patches, kp, labels = point_feature_extractor(path, loc)


plt.scatter(kp[labels==0][:, 0], kp[labels==0][:, 1], s=5, alpha=1, c = 'blue', label = 'no motor')
plt.scatter(kp[labels==1][:, 0], kp[labels==1][:, 1], s=5, alpha=1, c = 'red', label = 'motor')
plt.imshow(im, cmap='gray')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


patches.shape


plt.imshow(patches[labels==1][0], cmap='gray')


kp.shape


plt.imshow(im, cmap='gray')


!pip install /kaggle/input/pip-install-pyg/torch_spline_conv-1.2.2+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_sparse-0.6.18+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/pyg_lib-0.4.0+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_cluster-1.6.3+pt25cu124-cp310-cp310-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg/torch_geometric-2.6.1-py3-none-any.whl


from torch_geometric.nn import radius_graph
from torch_geometric.utils import degree
import torch


def normalize_kp(kp):
    return kp/np.array(CFG.img_size)


kp_normalized = normalize_kp(kp)


edge_index = radius_graph(torch.from_numpy(kp_normalized), r=0.05, loop=False)


node_degrees = degree(edge_index[0], num_nodes=kp_normalized.shape[0])

# Get max and average degree
max_degree = round(node_degrees.max().item(), 3)
avg_degree = node_degrees.mean().item()
print("Max Degree:", max_degree)
print("Avg Degree:", avg_degree)


G = nx.Graph()
G.add_edges_from(edge_index.t().tolist())

# Extract node positions
pos = {i: kp[i].tolist() for i in range(kp_normalized.shape[0])}

color_map = ['blue' if labels[i] == 0 else 'red' for i in range(kp_normalized.shape[0])]


plt.figure(figsize = (9, 9))
#plt.imshow(im, cmap='gray')
nx.draw(G, pos, with_labels=False, node_size=40, node_color="blue", edge_color="black", font_size=12, alpha=0.3)
plt.title(f"Radius Graph -- (Max_degr, Avg_degr) = {(max_degree, avg_degree)}")
plt.scatter(kp[labels==1][:, 0], kp[labels==1][:, 1], s=40, alpha=1, c = 'red', label = 'motor')




