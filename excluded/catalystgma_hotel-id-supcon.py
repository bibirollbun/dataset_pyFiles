import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision
import itertools
import random
import psutil
import torch
# import faiss # !pip install faiss-cpu faiss-gpu-cu12
import numba
import copy
import json
import time
import tqdm
import sys
import os


# SUPCON_RUNID, SUPCON_EPOCH_CNT = 1747037150, 300
# SUPCON_RUNID, SUPCON_EPOCH_CNT = 1747991087, 50
# SUPCON_RUNID, SUPCON_EPOCH_CNT = 1747991087, 150
# SUPCON_RUNID, SUPCON_EPOCH_CNT = 1748116222, 150
# SUPCON_RUNID, SUPCON_EPOCH_CNT = 1748172677, 150
SUPCON_RUNID, SUPCON_EPOCH_CNT = 1748346687, 50

BASE_SERVER_FOLDER = "/kaggle/input"

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

COCO_CLASS_NAMES = ["other", "chair", "couch", "bed", "dining table", "toilet", "tv"]
DSET_TYPES = ["train", "val"]

UNET_IM_LEN = 256
UNET_RESHAPER = torchvision.transforms.Resize(UNET_IM_LEN, antialias = None)
UNET_NUM_CROP_TRIES = 3

SUPCON_EMBEDDING_SIZE = 512

INFERENCE_BATCH_SIZE = 128
INFERENCE_MEAN_OVER_TOP = 4 # 10 # pentru un hotel_id, fac media peste cele mai bune ?? citiri.
INFERENCE_NEED_TOP = 5 # cate hoteluri tin minte pentru un test image.

def print_used_memory():
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"Host memory used: {round(psutil.Process().memory_info().rss / (2 ** 30), 3)} GB.\nDevice memory used: {round((total - free) / (2 ** 30), 3)} GB.", flush = True)

def get_px(fpath: str, device = DEVICE):
    im = torchvision.io.decode_image(fpath) / 255
    im = UNET_RESHAPER(im)
    if im.shape[0] == 1: # posibil ca imaginea primita sa fie uni-canal.
        im = im.broadcast_to([3, *im.shape[1:]])
    return im.to(device)

# rdup_arr([0, 1, 3, 2, 5], 3) = [0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2, 5, 5, 5]
def rdup_arr(arr: list, k: int):
    return [y for x in arr for y in [x] * k]

def get_hotel_image_crop_offset(hotel_im: torch.tensor, crop_ind: int):
    h, w = hotel_im.shape[1:]
    diff = int(np.linspace(0, max(h - UNET_IM_LEN, w - UNET_IM_LEN), UNET_NUM_CROP_TRIES)[crop_ind])
    dy, dx = (0, diff) if h == UNET_IM_LEN else (diff, 0)
    return dy, dx

@numba.jit(nopython=True)
def load_mmi_neighs(h: np.int32, w: np.int32, i: np.int32, j: np.int32, type_neighs: np.int32):
    # return [(i+di, j+dj) for di, dj in dys_xs if 0 <= i + di < h and 0 <= j + dj < w]
    sol = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if (type_neighs == 0 and abs(di) != abs(dj)) or (type_neighs == -1 and di != 0 and dj == 0) or (type_neighs == 1 and di == 0 and dj != 0):
                if 0 <= i + di < h and 0 <= j + dj < w:
                    sol.append((i+di, j+dj))
    return sol

@numba.jit(nopython=True)
def load_mmi_bfs(h: np.int32, w: np.int32, inf: np.int32, im: np.ndarray, mask: np.ndarray, base: np.ndarray, qus):
    for type_neighs in [-1, 1]:
        qus[0].clear()
        qus[1].clear()

        for i in range(h):
            for j in range(w):
                neighs = load_mmi_neighs(h, w, i, j, 0)
                if len(neighs) > 0:
                    has_mask_neighbor = False
                    for y, x in neighs:
                        if mask[y, x]:
                            has_mask_neighbor = True
                            break
                    if has_mask_neighbor:
                        qus[0].append((np.int32(i), np.int32(j)))

        pin = 0
        d = inf * mask

        while len(qus[pin]):
            for i, j in qus[pin]:
                neighbors = load_mmi_neighs(h, w, i, j, type_neighs)
                for y, x in neighbors:
                    op_y, op_x = base[i, j, 0] - (y - base[i, j, 0]), base[i, j, 1] - (x - base[i, j, 1])
                    if 0 <= op_y < h and 0 <= op_x < w and d[y, x] > 1 + d[i, j]:
                        d[y, x] = 1 + d[i, j]
                        im[:, y, x] += 0.5 * im[:, op_y, op_x]
                        base[y, x] = base[i, j]
                        qus[pin^1].append((np.int32(y), np.int32(x)))
            qus[pin].clear()
            pin ^= 1

def load_mirror_masked_image(fpath: str, device = DEVICE):
    t_start = time.time()

    im = get_px(fpath, device = torch.device("cpu"))

    mask = torch.logical_and(
        torch.all(im.permute(1, 2, 0) >= torch.tensor([0.95, 0, 0]), dim = 2),
        torch.all(im.permute(1, 2, 0) <= torch.tensor([1, 0.1, 0.1]), dim = 2),
    )

    # dilatam imaginea. e posibil sa ramana dungi rosii pe margine daca aplicam doar masca normala.
    mask = F.conv2d(
        mask.unsqueeze(dim = 0).unsqueeze(dim = 0).float(),
        torch.ones(1, 1, 9, 9),
        padding = "same"
    )[0, 0] > 0
    
    # im.shape = [3, ?, ??], cu min(?, ??) = 256.
    # mask.shape = [?, ??]. mask[i, j] = True <=> trebuie inlocuit pixelul respectiv.
    h, w = mask.shape
    
    base = np.array([[(i, j) for j in range(w)] for i in range(h)], dtype = np.int32)
    inf = np.int32(mask.shape[0] * mask.shape[1] + 1)
    im, mask = im.numpy(), mask.numpy()
    im[:, mask] = 0

    # print(f"starting bfs related after {time.time() - t_start} s.", flush = True)

    list_type = numba.types.UniTuple(numba.types.int32, 2) # numba.types.Tuple((numba.types.int32, numba.types.int32))
    qus = [numba.typed.List.empty_list(list_type), numba.typed.List.empty_list(list_type)]
    
    load_mmi_bfs(h, w, inf, im, mask, base, qus)
    im = np.minimum(np.ones_like(im), im)

    # print(f"finished bfs after {time.time() - t_start} s.", flush = True)
    
    return torch.tensor(im, device = DEVICE)


# !ls /kaggle/input/hotel-id-supcon-embeddings/supcon_embeddings | wc -l
# !ls /kaggle/input/hotel-id-to-combat-human-trafficking-2022-fgvc9/test_images
# !ls -l /kaggle/input/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_masks | wc -l


# load supcon resnet.
t_start = time.time()

net = torchvision.models.resnet18() # weights = "DEFAULT"

# reteaua are un singur strat FC, il inlocuiesc.
net.fc = torch.nn.Linear(net.fc.in_features, SUPCON_EMBEDDING_SIZE)

net = net.to(DEVICE)

net.load_state_dict(torch.load(
    f"{BASE_SERVER_FOLDER}/hotel-id-supcon-embeddings/net_{SUPCON_RUNID}_{SUPCON_EPOCH_CNT}.pt",
    weights_only = True, map_location = DEVICE)
)
net.eval()

print(f"Loaded pretrained supcon net {SUPCON_RUNID = }, {SUPCON_EPOCH_CNT = }, {round(time.time() - t_start, 3)} s passed.", flush = True)
print_used_memory()


# load supcon embeddings.
embeddings_all = []
emb_hotel_ids = []

# f"{BASE_SERVER_FOLDER}/hotel-id-supcon-embeddings/supcon_embeddings_{SUPCON_RUNID}/"

for root, dirs, files in os.walk(f"{BASE_SERVER_FOLDER}/hotel-id-supcon-embeddings/supcon_embeddings_{SUPCON_RUNID}_{SUPCON_EPOCH_CNT}/"):
    for fname in tqdm.tqdm(files):
        hotel_id = int(fname.split('_')[-1].split('.')[0])

        try:
            embeddings = torch.load(os.path.join(root, fname), weights_only = True, map_location = DEVICE)
            embeddings = embeddings / torch.norm(embeddings, dim = 1).unsqueeze(dim = 1)        

            embeddings_all.append(embeddings)
            emb_hotel_ids.extend([hotel_id] * len(embeddings))
        except:
            print(f"Failed to load {hotel_id = }")

embeddings_all = torch.cat(embeddings_all)

print(f"{embeddings_all.shape = }")
print_used_memory()


mirrored_test_images = []
test_ims_embeddings = []
test_fnames = []

with torch.no_grad():
    for root, dirs, files in os.walk(f"{BASE_SERVER_FOLDER}/hotel-id-to-combat-human-trafficking-2022-fgvc9/test_images"):
        for fname in tqdm.tqdm(files):
            test_fnames.append(fname)
            mirrored_im = load_mirror_masked_image(os.path.join(root, fname))
    
            for crop_ind in range(UNET_NUM_CROP_TRIES):
                dy, dx = get_hotel_image_crop_offset(mirrored_im, crop_ind)        
                mirrored_test_images.append(mirrored_im[:, dy: dy + UNET_IM_LEN, dx: dx + UNET_IM_LEN])

            if len(mirrored_test_images) >= INFERENCE_BATCH_SIZE:
                test_ims_embeddings.append(net(torch.stack(mirrored_test_images)))
                del mirrored_test_images
                mirrored_test_images = []

    if len(mirrored_test_images) > 0:
        test_ims_embeddings.append(net(torch.stack(mirrored_test_images)))
        del mirrored_test_images
    
    test_ims_embeddings = torch.cat(test_ims_embeddings)
    test_ims_embeddings = test_ims_embeddings / torch.norm(test_ims_embeddings, dim = 1).unsqueeze(dim = 1)
    
    print(f"{test_ims_embeddings.shape = }")
    print_used_memory()


test_sims = test_ims_embeddings @ embeddings_all.T

# pun toate sim scores de la cropurile din aceeasi imagine pe acelasi rand.
test_sims = test_sims.view(-1, test_sims.shape[1] * UNET_NUM_CROP_TRIES).cpu().numpy()

best_hotel_id_matches = []
for i in tqdm.tqdm(range(len(test_sims))):
    mean_hotel_id_scores = {}
    for hotel_id, j in zip(emb_hotel_ids, itertools.count()):
        if hotel_id not in mean_hotel_id_scores:
            mean_hotel_id_scores[hotel_id] = []

        mean_hotel_id_scores[hotel_id].append(test_sims[i, j])

    best_scores_hotel_ids = sorted(
        [(np.mean(sorted(mean_hotel_id_scores[hotel_id], reverse = True)[:INFERENCE_MEAN_OVER_TOP]), hotel_id) for hotel_id in mean_hotel_id_scores],
        reverse = True
    )[:INFERENCE_NEED_TOP]
    
    best_hotel_id_matches.append([hotel_id for mean, hotel_id in best_scores_hotel_ids])        


with open("submission.csv", 'w') as fout:
    fout.write("image_id,hotel_id\n")
    for fname, hotel_matches in zip(test_fnames, best_hotel_id_matches):
        fout.write(f"{fname},{' '.join([str(hotel_id) for hotel_id in hotel_matches])}\n")


# !cat submission.csv




