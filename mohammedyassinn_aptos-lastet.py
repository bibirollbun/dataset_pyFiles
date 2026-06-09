# 1. Uninstall existing packages to ensure a clean slate
!pip uninstall scikit-learn imbalanced-learn -y

# 2. Install the known stable versions
!pip install scikit-learn==1.1.3
!pip install imbalanced-learn==0.10.1


import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw


import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
img_dir = os.path.join(data_dir, 'train_images')
processed_dir = '/kaggle/working/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Step 1: Visualize Initial Data
def visualize_initial_data(num_samples_per_class=2):
    train_csv = pd.read_csv(train_csv_path)
    fig, axs = plt.subplots(5, num_samples_per_class, figsize=(10, 20))
    for class_label in range(5):
        class_samples = train_csv[train_csv['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            img = Image.open(img_path)
            axs[class_label, i].imshow(img)
            axs[class_label, i].set_title(f"Class {class_label}: {row['id_code']}")
            axs[class_label, i].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_initial_data()






train_csv = pd.read_csv(train_csv_path)
print(train_csv.shape[0])
print(train_csv['diagnosis'].value_counts())


sizes = train_csv['diagnosis'].value_counts().values
labels = ['No Dr','Moderate DR','Mild Dr','Proliferative DR', 'Sever DR']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%');


import cv2

def smart_resize(img, size=(1024,1024)):
    h, w = img.shape[:2]
    if h > size[0] and w > size[1]:
        interp = cv2.INTER_AREA
    elif h < size[0] and w < size[1]:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation = interp)


def crop_circular(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mask = gray > 30
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis = 0)
    y1, x1 = coords.max(axis = 0)
    cropped = img[y0:y1, x0:x1]
    return cropped


def ben_graham_preprocess(img, sigmaX = 10):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    return img


img_path = os.path.join(img_dir, f"{train_csv.iloc[0,0]}.png")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img.shape


img_resized = smart_resize(img)
img_resized.shape


img_cropped = crop_circular(img_resized)
img_cropped.shape


img_braham = ben_graham_preprocess(img_cropped)
img_braham.shape


plt.imshow(img_braham);


plt.imshow(img);


def preprocess_image(img_path, target_size=512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Order: Crop -> Resize -> Ben Graham (per Kaggle load_ben_color)
    img = smart_resize(img,(1024, 1024))
    img = crop_circular(img)
    img = ben_graham_preprocess(img, sigmaX=10)
    img = smart_resize(img,(target_size, target_size))
    return img


import pandas as pd
# Step 3: Visualize Processed Data
def visualize_processed_data(num_samples_per_class=2):
    train_csv = pd.read_csv(train_csv_path)
    fig, axs = plt.subplots(5, 2 * num_samples_per_class, figsize=(20, 20))
    for class_label in range(5):
        class_samples = train_csv[train_csv['diagnosis'] == class_label].sample(num_samples_per_class, random_state=42)
        for i, (_, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            original = cv2.imread(img_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            processed = preprocess_image(img_path)
            
            axs[class_label, 2*i].imshow(original)
            axs[class_label, 2*i].set_title(f"Original Class {class_label}: {row['id_code']}")
            axs[class_label, 2*i].axis('off')
            
            axs[class_label, 2*i + 1].imshow(processed)
            axs[class_label, 2*i + 1].set_title(f"Processed Class {class_label}: {row['id_code']}")
            axs[class_label, 2*i + 1].axis('off')
    plt.tight_layout()
    plt.show()

# Call
visualize_processed_data()


# Step 4: Save Processed Images
def save_processed_images():
    train_csv = pd.read_csv(train_csv_path)
    for _, row in train_csv.iterrows():
        img_path = os.path.join(img_dir, f"{row['id_code']}.png")
        processed_img = preprocess_image(img_path)
        save_path = os.path.join(processed_dir, f"{row['id_code']}.png")
        cv2.imwrite(save_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved {len(train_csv)} images to {processed_dir}")

# Call
save_processed_images()


import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
data_dir = '/kaggle/input/aptos2019-blindness-detection'
train_csv_path = os.path.join(data_dir, 'train.csv')
processed_dir = '/kaggle/working/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(train_csv_path)

# --- Step 1: Train+Val vs Test (80:20) ---
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['diagnosis'],
    random_state=42
)

# --- Step 2: Train vs Val (80:20 of train_val -> 64:16 overall) ---
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,  # 20% of 80% = 16% of total
    stratify=train_val_df['diagnosis'],
    random_state=42
)

# --- Optional: Print class distribution to verify ---
print("Train class distribution:\n", train_df['diagnosis'].value_counts(normalize=True))
print("Val class distribution:\n", val_df['diagnosis'].value_counts(normalize=True))
print("Test class distribution:\n", test_df['diagnosis'].value_counts(normalize=True))

# --- Save CSVs ---
train_df.to_csv(os.path.join(processed_dir, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(processed_dir, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(processed_dir, 'test_split.csv'), index=False)



import seaborn as sns
import matplotlib.pyplot as plt

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Train distribution
sns.countplot(x='diagnosis', data=train_df, ax=axes[0], palette='viridis')
axes[0].set_title('Train Class Distribution')
axes[0].set_xlabel('Diagnosis')
axes[0].set_ylabel('Count')

# Validation distribution
sns.countplot(x='diagnosis', data=val_df, ax=axes[1], palette='magma')
axes[1].set_title('Validation Class Distribution')
axes[1].set_xlabel('Diagnosis')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()



import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2



class APTOSDataset(Dataset):
    def __init__(self, df, processed_dir, transform=None):
        self.data = df
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(self.processed_dir, img_name)
        
        # --- Read image using PIL ---
        image = Image.open(img_path).convert('RGB')  # ensure 3 channels
        image = np.array(image)  # convert to numpy for Albumentations
        
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label



import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),                  # intermediate resize
    A.CenterCrop(224, 224),              # clean center patch
    A.HorizontalFlip(p=0.5),             # light augmentation
    A.RandomBrightnessContrast(0.1,0.1, p=0.3),
    A.Rotate(limit=5, border_mode=0, p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_test_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])



# Training dataset
train_dataset = APTOSDataset(df=train_df,
                             processed_dir=processed_dir,
                             transform=train_transform)

# Validation dataset
val_dataset = APTOSDataset(df=val_df,
                           processed_dir=processed_dir,
                           transform=val_test_transform)

# Test dataset
test_dataset = APTOSDataset(df=test_df,
                            processed_dir=processed_dir,
                            transform=val_test_transform)



import matplotlib.pyplot as plt
import numpy as np

def visualize_augmentation(dataset, num_samples=5):
    """
    Visualize original and augmented images from a dataset.

    Args:
        dataset (Dataset): Instance of APTOSDataset.
        num_samples (int): Number of images to display.
    """
    plt.figure(figsize=(12, num_samples * 3))
    
    for i in range(num_samples):
        # Get raw image path
        row = dataset.data.iloc[i]
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(dataset.processed_dir, img_name)
        
        # Load original image with PIL
        orig_image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply augmentation / transform
        if dataset.transform:
            aug_image = dataset.transform(image=orig_image)['image']
            # Convert tensor to numpy for visualization
            if isinstance(aug_image, torch.Tensor):
                aug_image = aug_image.permute(1, 2, 0).numpy()
                # Undo normalization for visualization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                aug_image = np.clip((aug_image * std + mean), 0, 1)
        else:
            aug_image = orig_image / 255.0  # normalize for display
        
        # Plot original
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(orig_image)
        plt.title(f"Original - Label {row['diagnosis']}")
        plt.axis('off')
        
        # Plot augmented
        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(aug_image)
        plt.title(f"Augmented - Label {row['diagnosis']}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



# Visualize training dataset
visualize_augmentation(train_dataset, num_samples=5)

# Visualize validation dataset
visualize_augmentation(val_dataset, num_samples=5)



num_workers = min(6, os.cpu_count() or multiprocessing.cpu_count())
print(f"ğŸ§  Using {num_workers} workers")


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True,)


import torch
import torch.nn as nn
from torchvision import models
import os
import numpy as np
from tqdm.notebook import tqdm

def extract_efficientnet_b3_features(
    train_loader,
    val_loader,
    test_loader,
    save_dir,
    device="cuda"
):
    """
    Extract features from EfficientNet-B3 (pretrained) using Global Average Pooling
    and save as .npy files.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---- Load pretrained EfficientNet-B3 ----
    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )
    model.eval()
    model.to(device)

    # ---- Feature extractor (no classifier) ----
    feature_extractor = nn.Sequential(
        model.features,                    # convolutional backbone
        nn.AdaptiveAvgPool2d((1, 1))        # global average pooling
    ).to(device)

    feature_extractor.eval()

    def extract_features(loader, split_name):
        features_list = []
        labels_list = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Extracting {split_name} features"):
                imgs = imgs.to(device)

                feats = feature_extractor(imgs)      # [B, 1536, 1, 1]
                feats = feats.flatten(1)             # [B, 1536]

                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.numpy())

        features_array = np.vstack(features_list)
        labels_array = np.hstack(labels_list)

        np.save(os.path.join(save_dir, f"{split_name}_features.npy"), features_array)
        np.save(os.path.join(save_dir, f"{split_name}_labels.npy"), labels_array)

        print(
            f"{split_name} features saved: {features_array.shape}, "
            f"labels saved: {labels_array.shape}"
        )

        return features_array, labels_array

    train_feats, train_labels = extract_features(train_loader, "train")
    val_feats, val_labels = extract_features(val_loader, "val")
    test_feats, test_labels = extract_features(test_loader, "test")

    return (
        (train_feats, train_labels),
        (val_feats, val_labels),
        (test_feats, test_labels),
    )



save_folder = '/kaggle/working/resnet50_features'

(train_feats, train_labels), (val_feats, val_labels), (test_feats, test_labels) = extract_efficientnet_b3_features(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    save_dir=save_folder,
    device='cpu'
)



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def preprocess_features_with_smote_verbose(train_features, train_labels,
                                           val_features, val_labels,
                                           test_features, test_labels,
                                           pca_components=0.95,
                                           random_state=42):
    """
    Apply scaling -> PCA -> SMOTE on training features and print shapes after each step.
    Apply same scaling & PCA to val/test (no SMOTE).

    Args:
        train_features, val_features, test_features: numpy arrays [N_samples, N_features]
        train_labels, val_labels, test_labels: numpy arrays [N_samples]
        pca_components: number of components or float for variance ratio
        random_state: for reproducibility

    Returns:
        X_train_res, y_train_res, X_val_pca_scaled, y_val, X_test_pca_scaled, y_test
        scaler1, pca, scaler2: fitted objects for reference or reuse
    """
    print("Original shapes:")
    print(f"  Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")

    # 1. Standard scale on training features
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(train_features)
    X_val_scaled = scaler1.transform(val_features)
    X_test_scaled = scaler1.transform(test_features)
    print("After first StandardScaler:")
    print(f"  Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # 2. PCA on scaled features
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"After PCA (n_components={pca.n_components_}):")
    print(f"  Train: {X_train_pca.shape}, Val: {X_val_pca.shape}, Test: {X_test_pca.shape}")

    # 3. Optional: second scaling after PCA
    scaler2 = StandardScaler()
    X_train_pca_scaled = scaler2.fit_transform(X_train_pca)
    X_val_pca_scaled = scaler2.transform(X_val_pca)
    X_test_pca_scaled = scaler2.transform(X_test_pca)
    print("After second StandardScaler (post-PCA):")
    print(f"  Train: {X_train_pca_scaled.shape}, Val: {X_val_pca_scaled.shape}, Test: {X_test_pca_scaled.shape}")

    # 4. Apply SMOTE only on training set
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_pca_scaled, train_labels)
    print("After SMOTE on training set:")
    print(f"  Train: {X_train_res.shape}, Labels: {y_train_res.shape}")

    return X_train_res, y_train_res, X_val_pca_scaled, val_labels, X_test_pca_scaled, test_labels, scaler1, pca, scaler2



# Load saved features
train_features = np.load('resnet50_features/train_features.npy')
train_labels   = np.load('resnet50_features/train_labels.npy')
val_features   = np.load('resnet50_features/val_features.npy')
val_labels     = np.load('resnet50_features/val_labels.npy')
test_features  = np.load('resnet50_features/test_features.npy')
test_labels    = np.load('resnet50_features/test_labels.npy')



X_train_res, y_train_res, X_val_proc, y_val, X_test_proc, y_test, scaler1, pca, scaler2 = \
    preprocess_features_with_smote_verbose(train_features, train_labels,
                                           val_features, val_labels,
                                           test_features, test_labels,
                                           pca_components=0.95)



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier



# -----------------------------
# 1ï¸�âƒ£ Base Classifier Training
# -----------------------------
from sklearn.model_selection import GridSearchCV
# Define all 13 classifiers
base_classifiers = {
    'SVM': SVC(probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    'RidgeClassifier': RidgeClassifier(),
    'SGDClassifier': SGDClassifier(max_iter=1000, random_state=42),
    'NearestCentroid': NearestCentroid(),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

class_names = [0, 1, 2, 3, 4]

# -----------------------------
# Define hyperparameter grids for each classifier
# -----------------------------
param_grids = {
    'SVM': {'C':[0.001,0.01,0.1,1,10,100], 'kernel':['linear','rbf','poly'], 'gamma':['scale','auto']},
    'LogisticRegression': {'C':[0.01,0.1,1,10,100], 'penalty':['l2'], 'solver':['lbfgs']},
    'RandomForest': {'n_estimators':[100,200,300,400], 'max_depth':[None,10,20,30,40], 'min_samples_split':[2,5,10]},
    'KNN': {'n_neighbors':[1,3,5,7,9,11], 'weights':['uniform','distance'], 'p':[1,2], 'algorithm':['auto','ball_tree','kd_tree']},
    'GaussianNB': {},  # no tuning
    'ExtraTrees': {'n_estimators':[100,200,300,400], 'max_depth':[None,10,20,30,40], 'min_samples_split':[2,5,10], 'max_features':['auto','sqrt','log2']},
    'MLP': {'hidden_layer_sizes':[(100,),(100,50),(50,50,25),(64,32)], 'alpha':[0.0001,0.001,0.01], 'activation':['relu','tanh'], 'learning_rate':['constant','adaptive']},
    'PassiveAggressive': {'C':[0.01,0.1,1,10,100]},
    'RidgeClassifier': {'alpha':[0.01,0.1,1,10,100]},
    'SGDClassifier': {'alpha':[0.0001,0.001,0.01,0.1], 'loss':['hinge','log','perceptron'], 'penalty':['l2','l1','elasticnet']},
    'NearestCentroid': {},  # no tuning
    'DecisionTree': {'max_depth':[None,10,20,30,40], 'min_samples_split':[2,5,10,20], 'criterion':['gini','entropy']}
}





def train_tune_select_classifiers(X_train, y_train, X_val, y_val, classifiers_dict, param_grids,
                                  class_names, scoring='f1_weighted', qwk_thresh=0.5, roc_folder='prediction/ML_base_ROC'):
    """
    Train all classifiers with expanded hyperparameter search.
    Only retain classifiers that exceed qwk_thresh on validation for stacking.
    
    Returns:
        tuned_models_selected: dict of tuned classifiers that pass QWK threshold
        tuning_results: dict of best parameters
        qwk_scores: dict of QWK scores for each classifier
    """
    import os
    from sklearn.metrics import cohen_kappa_score, f1_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import clone
    import matplotlib.pyplot as plt
    
    os.makedirs(roc_folder, exist_ok=True)
    n_classes = len(class_names)
    y_val_bin = label_binarize(y_val, classes=list(range(n_classes)))
    
    tuned_models_selected = {}
    tuning_results = {}
    qwk_scores = {}
    
    for name, model in classifiers_dict.items():
        print(f"\n=== {name} ===")
        
        # Hyperparameter tuning if grid exists
        grid = param_grids.get(name, {})
        if grid:
            gscv = GridSearchCV(clone(model), grid, scoring=scoring, cv=3, n_jobs=-1)
            gscv.fit(X_train, y_train)
            best_model = gscv.best_estimator_
            tuning_results[name] = gscv.best_params_
        else:
            best_model = clone(model)
            best_model.fit(X_train, y_train)
            tuning_results[name] = None
        
        # Evaluate on validation
        y_pred = best_model.predict(X_val)
        qk = cohen_kappa_score(y_val, y_pred, weights='quadratic')
        f1 = f1_score(y_val, y_pred, average='weighted')
        qwk_scores[name] = qk
        print(f"[Tuned] QWK: {qk:.4f}, Weighted F1: {f1:.4f}")
        
        # Save ROC
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_val)
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_val)
            if y_score.ndim == 1:
                y_score = np.vstack([1-y_score, y_score]).T
        else:
            y_score = None
        
        if y_score is not None:
            plt.figure(figsize=(7,6))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_val_bin[:,i], y_score[:,i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {class_names[i]} (AUC={roc_auc:.2f})")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} ROC Curve (Tuned)")
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(roc_folder, f"{name}_tuned_ROC.png"))
            plt.close()
        
        # Only keep classifiers above QWK threshold
        if qk >= qwk_thresh:
            tuned_models_selected[name] = best_model
        else:
            print(f"Classifier {name} below QWK threshold ({qwk_thresh}), excluded from stacking.")
    
    return tuned_models_selected, tuning_results, qwk_scores



tuned_models_selected,tuning_results, qwk_scores = train_tune_select_classifiers(
    X_train_res, y_train_res,
    X_val=X_val_proc, y_val=y_val,
    classifiers_dict=base_classifiers,
    param_grids=param_grids,
    class_names=class_names,
    roc_folder='prediction/ML_base_ROC'
)



from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# 1ï¸�âƒ£ Generate OOF Probabilities for Stacking
# -----------------------------
def generate_oof_predictions(fitted_models, X_train, y_train, n_splits=5):
    n_samples = X_train.shape[0]
    n_classifiers = len(fitted_models)
    n_classes = len(np.unique(y_train))
    oof_probs = np.zeros((n_samples, n_classifiers * n_classes))
    oof_labels = np.zeros(n_samples, dtype=y_train.dtype)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (name, model) in enumerate(fitted_models.items()):
        fold_probs = np.zeros((n_samples, n_classes))
        for train_idx, val_idx in skf.split(X_train, y_train):
            clf = clone(model)
            clf.fit(X_train[train_idx], y_train[train_idx])
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_train[val_idx])
            elif hasattr(clf, "decision_function"):
                probs = clf.decision_function(X_train[val_idx])
                if probs.ndim == 1:
                    probs = np.vstack([1-probs, probs]).T
            else:
                probs = np.zeros((len(val_idx), n_classes))
            fold_probs[val_idx] = probs
            oof_labels[val_idx] = y_train[val_idx]
        oof_probs[:, i*n_classes:(i+1)*n_classes] = fold_probs

    return oof_probs, oof_labels

# -----------------------------
# 2ï¸�âƒ£ Generate Validation/Test Probabilities
# -----------------------------
def generate_base_predictions(fitted_models, X_data):
    n_samples = X_data.shape[0]
    n_classifiers = len(fitted_models)
    n_classes = len(fitted_models[list(fitted_models.keys())[0]].classes_)
    probs = np.zeros((n_samples, n_classifiers * n_classes))

    for i, (name, model) in enumerate(fitted_models.items()):
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_data)
        elif hasattr(model, "decision_function"):
            p = model.decision_function(X_data)
            if p.ndim==1:
                p = np.vstack([1-p, p]).T
        else:
            p = np.zeros((n_samples, n_classes))
        probs[:, i*n_classes:(i+1)*n_classes] = p
    return probs

# -----------------------------
# 3ï¸�âƒ£ Stacked Model Training and Evaluation
# -----------------------------
def train_stacked_mlp(X_stack_train, y_stack_train, X_stack_val, y_val, class_names, save_folder='prediction/stacked'):
    os.makedirs(save_folder, exist_ok=True)

    # Scale stacked features
    scaler_stack = StandardScaler()
    X_stack_train_scaled = scaler_stack.fit_transform(X_stack_train)
    X_stack_val_scaled = scaler_stack.transform(X_stack_val)

    # Train MLP meta-learner
    stack_model = MLPClassifier(hidden_layer_sizes=(64,32),
                                activation='relu', solver='adam',
                                max_iter=1000, random_state=42,
                                early_stopping=True, alpha=0.0001)
    stack_model.fit(X_stack_train_scaled, y_stack_train)

    # Predictions
    y_stack_pred = stack_model.predict(X_stack_val_scaled)
    y_stack_probs = stack_model.predict_proba(X_stack_val_scaled)

    # QWK and weighted F1
    qk_stack = cohen_kappa_score(y_val, y_stack_pred, weights='quadratic')
    f1_stack = f1_score(y_val, y_stack_pred, average='weighted')
    print(f"\nStacked Model QWK: {qk_stack:.4f}, Weighted F1: {f1_stack:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_stack_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Stacked Model Confusion Matrix")
    plt.savefig(os.path.join(save_folder, "stacked_confusion.png"))
    plt.show()

    # ROC Curves
    n_classes = len(class_names)
    y_bin = label_binarize(y_val, classes=list(range(n_classes)))

    plt.figure(figsize=(7,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_stack_probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stacked Model ROC Curve")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "stacked_ROC.png"))
    plt.show()

    return stack_model, scaler_stack



tuned_models_selected


# 1ï¸�âƒ£ Generate OOF predictions for stacking
X_stack_train, y_stack_train = generate_oof_predictions(tuned_models_selected, X_train_res, y_train_res)

# 2ï¸�âƒ£ Generate validation probabilities
X_stack_val = generate_base_predictions(tuned_models_selected, X_val_proc)





# 3ï¸�âƒ£ Train stacked MLP and evaluate
stack_model, scaler_stack = train_stacked_mlp(X_stack_train, y_stack_train,
                                               X_stack_val, y_val,
                                               class_names,
                                               save_folder='prediction/stacked')




