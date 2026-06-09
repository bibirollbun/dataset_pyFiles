import torchvision
print(torchvision.__version__)



import pandas as pd
import numpy as np
import cv2
import os, re

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt 
plt.rcParams['figure.figsize'] = (10.0, 10.0)


DATA_DIR = "/kaggle/input/costum"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ON_CPU = DEVICE == torch.device('cpu')
TRAIN_BATCH_SIZE = 4 if ON_CPU else 6
VALID_BATCH_SIZE = 2 if ON_CPU else 2
NUM_EPOCHS = 50
NEW_COLUMNS = ['x', 'y', 'w', 'h']


train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))

UNIQ_TRAIN_IMAGE_IDS = train_df["image_id"].unique()

train_df.shape, test_df.shape


# expand the bbox coordinates into x, y, w, h
def expand_bbox(x):
    # also convert everything to np.float
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x), dtype=np.float)
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

# initialize new columns with -1
for new_column in NEW_COLUMNS:
    train_df[new_column] = -1

train_df[NEW_COLUMNS] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df.head()


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image is not None, f"IMAGE NOT FOUND AT {image_path}"
    return image


def draw_bboxes(boxes, image, color=(255,0,0)):
    for box in boxes:
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 3
        )
    return image


def plot_random_train_sample():
    image_id = np.random.choice(UNIQ_TRAIN_IMAGE_IDS)
    plt.title(image_id)
    image = load_image(os.path.join(DATA_DIR, "train", f"{image_id}.jpg"))
    bboxes = (train_df[train_df["image_id"] == image_id][NEW_COLUMNS]).to_numpy()
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    plt.imshow(draw_bboxes(bboxes, image))


plot_random_train_sample()


class WheatDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        super().__init__()

        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = df['image_id'].unique()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = load_image(os.path.join(self.image_dir, f"{image_id}.jpg")).astype(np.float32)
        image /= 255.0
        # change the shape from [h,w,c] to [c,h,w]  
        image = torch.from_numpy(image).permute(2,0,1)

        records = self.df[self.df['image_id'] == image_id]

        boxes = records[NEW_COLUMNS].values
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # change the co-ordinates into expected [x, y, x+w, y+h] format
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # since all the boxes are wheat, it's all 1s
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        # consider iscrowd false for all the boxes
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            pass

        return image, target


def get_model():
    """
    https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn
    """
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=7)
    return model


# 10% into validation
n_validation = int(0.1* len(UNIQ_TRAIN_IMAGE_IDS))
valid_ids = UNIQ_TRAIN_IMAGE_IDS[-n_validation:]
train_ids = UNIQ_TRAIN_IMAGE_IDS[:-n_validation]

df_in_valid = train_df[train_df['image_id'].isin(valid_ids)]
df_in_train = train_df[train_df['image_id'].isin(train_ids)]

print("%i training samples\n%i validation samples" % (len(df_in_train["image_id"].unique()), len(df_in_valid["image_id"].unique())))


train_dataset = WheatDataset(df_in_train, os.path.join(DATA_DIR, "train"))
valid_dataset = WheatDataset(df_in_valid, os.path.join(DATA_DIR, "train"))

# since our single getitem returns image, targets. [shape of targets is different depending on the number of bounding boxes in the image] ?
def collate_fn(batch):
    return tuple(zip(*batch))

train_data_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn
)


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


model = get_model()
model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


train_losses = Averager()
val_losses = Averager()
lowest_val_loss = float('inf')

# has to be in train mode for both train and valid coz the outputs are different in two cases
model.train()
for epoch in range(NUM_EPOCHS):
    train_losses.reset()
    val_losses.reset()

    for batch_index, (images, targets) in enumerate(train_data_loader):
        # move the images and targets to device
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images,targets)
        loss = sum(loss for loss in loss_dict.values())

        # track the loss
        train_losses.send(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 50 == 0:
            print(f"Epoch: {epoch} Batch Index: {batch_index} Loss: {loss.item()}")

    # evaluate
    with torch.no_grad():
        for _, (images, targets) in enumerate(valid_data_loader):
            # move the images and targets to device
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            val_loss_dict = model(images, targets)
            val_loss = sum(loss for loss in val_loss_dict.values())

            # track the loss
            val_losses.send(val_loss.item())

    if val_losses.value < lowest_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
    else:
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    # print stats
    print(f"Epoch #{epoch} TRAIN LOSS: {train_losses.value} VALIDATION LOSS: {val_losses.value}\n")



torch.save(model.state_dict(), 'fasterRCNN_50.pth')





import matplotlib.pyplot as plt
import torch
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# --- CONFIGURATION ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TEST_CSV_PATH = "/kaggle/input/costum/test.csv"
TEST_IMAGES_DIR = "/kaggle/input/costum/test"
IOU_THRESHOLD = 0.5
IMAGES_PER_ROW = 4  # Nombre d'images affichées par ligne

# --- CHARGER LES DONNÉES ---
test_df = pd.read_csv(TEST_CSV_PATH)

# --- CHARGER LE MODÈLE ---
def get_model(num_classes=2):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

model = get_model(num_classes=7)
model.load_state_dict(torch.load("fasterRCNN_50.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- FONCTIONS UTILES ---
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image, image_tensor.unsqueeze(0)

def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def get_real_boxes(image_id):
    records = test_df[test_df["image_id"] == image_id]
    boxes = records['bbox'].apply(lambda x: np.array(eval(x))).values
    if len(boxes) > 0:
        boxes = np.stack(boxes)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
    return boxes if len(boxes) > 0 else np.array([])

def compute_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- CALCUL MATRICE DE CONFUSION ---
TP, FP, FN, TN = 0, 0, 0, 0

y_true, y_pred = [], []

for image_name in os.listdir(TEST_IMAGES_DIR):
    image_id = image_name.split(".")[0]
    image_path = os.path.join(TEST_IMAGES_DIR, image_name)
    _, image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        prediction = model(image_tensor.to(DEVICE))

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_boxes = pred_boxes[pred_scores > 0.5]

    real_boxes = get_real_boxes(image_id)

    matched_preds = set()
    matched_reals = set()
    
    for i, rb in enumerate(real_boxes):
        detected = False
        for j, pb in enumerate(pred_boxes):
            if compute_iou(pb, rb) > IOU_THRESHOLD:
                detected = True
                matched_preds.add(j)
                matched_reals.add(i)
                break
        if detected:
            TP += 1
        else:
            FN += 1

    FP += len(pred_boxes) - len(matched_preds)
    TN += 0  # Pas de vrai négatif significatif dans ce cas

y_true = [1] * TP + [0] * FN
y_pred = [1] * TP + [1] * FP

# --- CALCULER LES MÉTRIQUES ---
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

print(f"Précision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")

# --- AFFICHAGE MATRICE DE CONFUSION ---
conf_matrix = [[TN, FP], [FN, TP]]
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fond (0)", "Objet (1)"],
            yticklabels=["Fond (0)", "Objet (1)"])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de Confusion (IoU > 0.5)")
plt.show()

# --- AFFICHER LES IMAGES PAR LIGNES DE 4 ---
test_images = os.listdir(TEST_IMAGES_DIR)
num_images = len(test_images)
rows = (num_images + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW  # Nombre de lignes

fig, axes = plt.subplots(rows, IMAGES_PER_ROW, figsize=(20, 5 * rows))
axes = axes.flatten()

for idx, image_name in enumerate(test_images):
    image_id = image_name.split(".")[0]
    image_path = os.path.join(TEST_IMAGES_DIR, image_name)
    image, image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        prediction = model(image_tensor.to(DEVICE))

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_boxes = pred_boxes[pred_scores > 0.5]

    real_boxes = get_real_boxes(image_id)

    image_with_boxes = draw_boxes(image.copy(), real_boxes, (0, 0, 255), "GT")
    image_with_boxes = draw_boxes(image_with_boxes, pred_boxes, (0, 255, 0), "Pred")

    axes[idx].imshow(image_with_boxes)
    axes[idx].axis("off")
    axes[idx].set_title(image_name)

for i in range(idx + 1, len(axes)):
    axes[i].axis("off")

plt.show()



import matplotlib.pyplot as plt
import torch
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# --- CONFIGURATION ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TEST_CSV_PATH = "/kaggle/input/costum/test.csv"
TEST_IMAGES_DIR = "/kaggle/input/costum/test"
IOU_THRESHOLD = 0.5
BATCH_SIZE = 10  # Nombre d'images affichées par lot

# --- CHARGER LES DONNÉES ---
test_df = pd.read_csv(TEST_CSV_PATH)

# --- CHARGER LE MODÈLE ---
def get_model(num_classes=2):
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

model = get_model(num_classes=7)
model.load_state_dict(torch.load("fasterRCNN_101.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- FONCTIONS UTILES ---
def preprocess_image(image_path):
    """Charge et prétraite une image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image, image_tensor.unsqueeze(0)  # Image originale + Tensor

def draw_boxes(image, boxes, color, label):
    """Dessine des boîtes sur une image."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def get_real_boxes(image_id):
    """Récupère les boîtes réelles depuis test_df."""
    records = test_df[test_df["image_id"] == image_id]
    boxes = records['bbox'].apply(lambda x: np.array(eval(x))).values
    if len(boxes) > 0:
        boxes = np.stack(boxes)
        boxes[:, 2] += boxes[:, 0]  # Convertir largeur en x2
        boxes[:, 3] += boxes[:, 1]  # Convertir hauteur en y2
    return boxes if len(boxes) > 0 else np.array([])

def compute_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes."""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0



# --- AFFICHER LES IMAGES PAR LOTS ---
test_images = os.listdir(TEST_IMAGES_DIR)
for i in range(0, len(test_images), BATCH_SIZE):
    batch_images = test_images[i:i + BATCH_SIZE]
    fig, axes = plt.subplots(len(batch_images), 1, figsize=(40, 20 * len(batch_images)))
    if len(batch_images) == 1:
        axes = [axes]
    
    for j, image_name in enumerate(batch_images):
        image_id = image_name.split(".")[0]
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image, image_tensor = preprocess_image(image_path)

        with torch.no_grad():
            prediction = model(image_tensor.to(DEVICE))

        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        pred_boxes = pred_boxes[pred_scores > 0.5]  # Filtrer par seuil

        real_boxes = get_real_boxes(image_id)

        image_with_boxes = draw_boxes(image.copy(), real_boxes, (0, 255, 0), "GT")  # Vert
        image_with_boxes = draw_boxes(image_with_boxes, pred_boxes, (255, 0, 0), "Pred")  # Bleu

        axes[j].imshow(image_with_boxes)
        axes[j].axis("off")
        axes[j].set_title(image_name)
    
    plt.show()


