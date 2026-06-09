import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from IPython.display import clear_output
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoFeatureExtractor, RegNetModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import BeitImageProcessor, BeitForImageClassification


# Set seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Path to the data
base_dir = "/kaggle/input/cidaut-ai-fake-scene-classification-2024"
base_dir_output = "/kaggle/working"
labels_path = os.path.join(base_dir, "train.csv")
train_path = os.path.join(base_dir, "Train")
test_path = os.path.join(base_dir, "Test")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


labels_df = pd.read_csv(labels_path)

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
sns.countplot(x="label", data=labels_df, ax=ax)
ax.set_title("Label Distribution")
ax.set_xlabel("Label")
ax.set_ylabel("Counts")
plt.tight_layout()
plt.show()

counts = labels_df["label"].value_counts()
normalized = labels_df["label"].value_counts(normalize=True)
result_df = pd.DataFrame({
    "Count": counts,
    "Percentage": normalized
})
result_df["Percentage"] = result_df["Percentage"].apply(lambda x: f"{x:.2%}")
print(result_df)


real_df = labels_df[labels_df['label'] == 'real'].copy()
fake_df = labels_df[labels_df['label'] == 'editada'].copy()

num_real = len(real_df)
num_fake = len(fake_df)
num_train_real = int(0.8 * num_real)
num_train_fake = int(0.8 * num_fake)

# Shuffle the DataFrames
real_df = real_df.sample(frac=1, random_state=random_seed)
fake_df = fake_df.sample(frac=1, random_state=random_seed)

# Split into train and validation
real_train = real_df.iloc[:num_train_real]
real_val = real_df.iloc[num_train_real:]
fake_train = fake_df.iloc[:num_train_fake]
fake_val = fake_df.iloc[num_train_fake:]

# Combine and sort by image name
train_df = pd.concat([real_train, fake_train])
val_df = pd.concat([real_val, fake_val])
train_df = train_df.sort_values('image').reset_index(drop=True)
val_df = val_df.sort_values('image').reset_index(drop=True)

train_df.to_csv(os.path.join(base_dir_output, "test_split.csv"), index=False)
val_df.to_csv(os.path.join(base_dir_output, "val_split.csv"), index=False)


train_df.head()


def load_images_paths(df, path):
    """
    Load all images from a dataframe and return a list of paths to the images.
    """
    if path == "train":
        file_path = train_path
    elif path == "test":
        file_path = test_path
    else:
        raise ValueError(f"Invalid path: {path}")
    
    image_paths = [os.path.join(file_path, filename) for filename in df["image"]]
    return image_paths

all_train_images_paths = load_images_paths(labels_df, "train")
all_test_images_paths = load_images_paths(pd.read_csv(os.path.join(base_dir, "sample_submission.csv")), "test")


os.makedirs(os.path.join(base_dir_output, "embeddings"), exist_ok=True)

def compute_embeddings_batched(model_type, model, processor, image_paths, save_path, batch_size=2):
    """
    Extract embeddings from a list of image paths.
    """
    all_embeddings = []
    all_img_names = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        img_paths = image_paths[i:i+batch_size]
        img_names = [os.path.basename(img_path) for img_path in img_paths]
        all_img_names.extend(img_names)
        images = [Image.open(image_path).convert("RGB") for image_path in img_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if model_type == "clip":
                embeddings = model.get_image_features(**inputs)
            elif model_type == "regnety":
                outputs = model(**inputs)
                embeddings = outputs.pooler_output
                embeddings = embeddings[:, :, 0, 0]
            elif model_type == "dino":
                outputs = model(**inputs)
                embeddings = outputs.pooler_output
            elif model_type == "swin" or model_type == "beit":
                outputs = model(**inputs)
                embeddings = outputs.logits
            else:
                raise ValueError(f"Model type {model_type} not supported.")
        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    df = pd.DataFrame(all_embeddings)
    df["image"] = all_img_names
    df.to_csv(save_path, index=False)


# Load the CLIP model and processor from HuggingFace
model_clip_vit_large = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor_clip_vit_large = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Extract features from the images and save them
compute_embeddings_batched("clip", model_clip_vit_large, processor_clip_vit_large, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_train.csv"), 8)
compute_embeddings_batched("clip", model_clip_vit_large, processor_clip_vit_large, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_test.csv"), 8)

# Clear memory
del model_clip_vit_large, processor_clip_vit_large
torch.cuda.empty_cache()


model_clip_vit_large_336 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
processor_clip_vit_large_336 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

compute_embeddings_batched("clip", model_clip_vit_large_336, processor_clip_vit_large_336, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_336_train.csv"), 8)
compute_embeddings_batched("clip", model_clip_vit_large_336, processor_clip_vit_large_336, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_336_test.csv"), 8)

del model_clip_vit_large_336, processor_clip_vit_large_336
torch.cuda.empty_cache()


model_clip_vit_large_laion = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K").to(device)
processor_clip_vit_large_laion = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

compute_embeddings_batched("clip", model_clip_vit_large_laion, processor_clip_vit_large_laion, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_laion_train.csv"), 8)
compute_embeddings_batched("clip", model_clip_vit_large_laion, processor_clip_vit_large_laion, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_laion_test.csv"), 8)

del model_clip_vit_large_laion, processor_clip_vit_large_laion
torch.cuda.empty_cache()


model_regnety_1280 = RegNetModel.from_pretrained("facebook/regnet-y-1280-seer").to(device)
processor_regnety_1280 = AutoFeatureExtractor.from_pretrained("facebook/regnet-y-1280-seer")

compute_embeddings_batched("regnety", model_regnety_1280, processor_regnety_1280, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_regnety_1280_train.csv"), 8)
compute_embeddings_batched("regnety", model_regnety_1280, processor_regnety_1280, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_regnety_1280_test.csv"), 8)

del model_regnety_1280, processor_regnety_1280
torch.cuda.empty_cache()


model_swinv2_192 = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft").to(device)
processor_swinv2_192 = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")

compute_embeddings_batched("swin", model_swinv2_192, processor_swinv2_192, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_192_train.csv"), 8)
compute_embeddings_batched("swin", model_swinv2_192, processor_swinv2_192, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_192_test.csv"), 8)

del model_swinv2_192, processor_swinv2_192
torch.cuda.empty_cache()


model_swinv2_384 = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft").to(device)
processor_swinv2_384 = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")

compute_embeddings_batched("swin", model_swinv2_384, processor_swinv2_384, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_384_train.csv"), 8)
compute_embeddings_batched("swin", model_swinv2_384, processor_swinv2_384, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_384_test.csv"), 8)

del model_swinv2_384, processor_swinv2_384
torch.cuda.empty_cache()


model_beit_large_224 = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224").to(device)
processor_beit_large_224 = BeitImageProcessor.from_pretrained("microsoft/beit-large-patch16-224")

compute_embeddings_batched("beit", model_beit_large_224, processor_beit_large_224, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_224_train.csv"), 8)
compute_embeddings_batched("beit", model_beit_large_224, processor_beit_large_224, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_224_test.csv"), 8)

del model_beit_large_224, processor_beit_large_224
torch.cuda.empty_cache()


model_beit_large_512 = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-512").to(device)
processor_beit_large_512 = BeitImageProcessor.from_pretrained("microsoft/beit-large-patch16-512")

compute_embeddings_batched("beit", model_beit_large_512, processor_beit_large_512, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_512_train.csv"), 8)
compute_embeddings_batched("beit", model_beit_large_512, processor_beit_large_512, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_512_test.csv"), 8)

del model_beit_large_512, processor_beit_large_512
torch.cuda.empty_cache()


model_dino_giant = AutoModel.from_pretrained('facebook/dinov2-giant').to(device)
processor_dino_giant = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')

compute_embeddings_batched("dino", model_dino_giant, processor_dino_giant, all_train_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_dino_giant_train.csv"), 8)
compute_embeddings_batched("dino", model_dino_giant, processor_dino_giant, all_test_images_paths, os.path.join(base_dir_output, "embeddings", "embeddings_dino_giant_test.csv"), 8)

del model_dino_giant, processor_dino_giant
torch.cuda.empty_cache()


def create_train_val_embeddings_df(embeddings_path, train_df, val_df):
    """
    Create a DataFrame with the embeddings for the training and validation images.
    """
    all_embeddings = pd.read_csv(embeddings_path)
    train_embeddings = all_embeddings[all_embeddings["image"].isin(train_df["image"])]
    val_embeddings = all_embeddings[all_embeddings["image"].isin(val_df["image"])]
    return train_embeddings, val_embeddings

def get_gt_labels_for_img_names(img_names):
    """
    Get the ground truth labels for a list of image names.
    img_names: List of image names.
    """
    labels = []
    for img_name in img_names:
        label = labels_df[labels_df["image"] == img_name]["label"].values[0]
        labels.append(label)
    # convert "real" to 1 and "editada" to 0
    labels = [1 if label == "real" else 0 for label in labels]
    return labels


# Load the embeddings in case the notebook is restarted
train_embeddings_clip_vit_large, val_embeddings_clip_vit_large = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_train.csv"), train_df, val_df)
train_embeddings_clip_vit_large_336, val_embeddings_clip_vit_large_336 = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_336_train.csv"), train_df, val_df)
train_embeddings_clip_vit_large_laion, val_embeddings_clip_vit_large_laion = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_laion_train.csv"), train_df, val_df)
train_embeddings_regnety_1280, val_embeddings_regnety_1280 = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_regnety_1280_train.csv"), train_df, val_df)
train_embeddings_swinv2_192, val_embeddings_swinv2_192 = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_192_train.csv"), train_df, val_df)
train_embeddings_swinv2_384, val_embeddings_swinv2_384 = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_384_train.csv"), train_df, val_df)
train_embeddings_beit_large_224, val_embeddings_beit_large_224 = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_224_train.csv"), train_df, val_df)
train_embeddings_beit_large_512, val_embeddings_beit_large_512 = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_512_train.csv"), train_df, val_df)
train_embeddings_dino_giant, val_embeddings_dino_giant = create_train_val_embeddings_df(os.path.join(base_dir_output, "embeddings", "embeddings_dino_giant_train.csv"), train_df, val_df)


print(f"Train embeddings CLIP ViT Large shape: {train_embeddings_clip_vit_large.shape}")
print(f"Validation embeddings CLIP ViT Large shape: {val_embeddings_clip_vit_large.shape}")
print(f"Train embeddings CLIP ViT Large 336 shape: {train_embeddings_clip_vit_large_336.shape}")
print(f"Validation embeddings CLIP ViT Large 336 shape: {val_embeddings_clip_vit_large_336.shape}")
print(f"Train embeddings CLIP ViT Large Laion shape: {train_embeddings_clip_vit_large_laion.shape}")
print(f"Validation embeddings CLIP ViT Large Laion shape: {val_embeddings_clip_vit_large_laion.shape}")
print(f"Train embeddings RegNetY 1280 shape: {train_embeddings_regnety_1280.shape}")
print(f"Validation embeddings RegNetY 1280 shape: {val_embeddings_regnety_1280.shape}")
print(f"Train embeddings SwinV2 192 shape: {train_embeddings_swinv2_192.shape}")
print(f"Validation embeddings SwinV2 192 shape: {val_embeddings_swinv2_192.shape}")
print(f"Train embeddings SwinV2 384 shape: {train_embeddings_swinv2_384.shape}")
print(f"Validation embeddings SwinV2 384 shape: {val_embeddings_swinv2_384.shape}")
print(f"Train embeddings BEiT Large 224 shape: {train_embeddings_beit_large_224.shape}")
print(f"Validation embeddings BEiT Large 224 shape: {val_embeddings_beit_large_224.shape}")
print(f"Train embeddings BEiT Large 512 shape: {train_embeddings_beit_large_512.shape}")
print(f"Validation embeddings BEiT Large 512 shape: {val_embeddings_beit_large_512.shape}")
print(f"Train embeddings DINOv2 Giant shape: {train_embeddings_dino_giant.shape}")
print(f"Validation embeddings DINOv2 Giant shape: {val_embeddings_dino_giant.shape}")


def plot_pca_on_ax(ax, df, title, n_components=2):
    embeddings = df.drop(columns=["image"]).values
    img_names = df["image"].values
    labels = get_gt_labels_for_img_names(img_names)

    # Perform PCA on feature data
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(embeddings)
    
    ax.scatter(pca_results[:, 0], pca_results[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1, color="gray", zorder=-1)
    ax.set_title(title)


fig, axs = plt.subplots(3, 3, figsize=(8, 8))
plot_pca_on_ax(axs[0, 0], pd.concat([train_embeddings_clip_vit_large, val_embeddings_clip_vit_large], ignore_index=True), "CLIP Large")
plot_pca_on_ax(axs[0, 1], pd.concat([train_embeddings_clip_vit_large_336, val_embeddings_clip_vit_large_336], ignore_index=True), "CLIP Large 336")
plot_pca_on_ax(axs[0, 2], pd.concat([train_embeddings_clip_vit_large_laion, val_embeddings_clip_vit_large_laion], ignore_index=True), "CLIP Large Laion")
plot_pca_on_ax(axs[1, 0], pd.concat([train_embeddings_regnety_1280, val_embeddings_regnety_1280], ignore_index=True), "RegNetY 1280")
plot_pca_on_ax(axs[1, 1], pd.concat([train_embeddings_swinv2_192, val_embeddings_swinv2_192], ignore_index=True), "SwinV2 192")
plot_pca_on_ax(axs[1, 2], pd.concat([train_embeddings_swinv2_384, val_embeddings_swinv2_384], ignore_index=True), "SwinV2 384")
plot_pca_on_ax(axs[2, 0], pd.concat([train_embeddings_beit_large_224, val_embeddings_beit_large_224], ignore_index=True), "BEiT Large 224")
plot_pca_on_ax(axs[2, 1], pd.concat([train_embeddings_beit_large_512, val_embeddings_beit_large_512], ignore_index=True), "BEiT Large 512")
plot_pca_on_ax(axs[2, 2], pd.concat([train_embeddings_dino_giant, val_embeddings_dino_giant], ignore_index=True), "DINOv2 Giant")

plt.suptitle("PCA (n=2) on Image Embeddings", fontsize=16)
plt.tight_layout()
plt.show()


def plot_tsne_on_ax(ax, df, title, n_components=2, perplexity=30, learning_rate="auto", max_iter=1_000, random_state=42):
    embeddings = df.drop(columns=["image"]).values
    img_names = df["image"].values
    labels = get_gt_labels_for_img_names(img_names)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                learning_rate=learning_rate, n_iter=max_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(embeddings)

    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                         c=labels, cmap="coolwarm", alpha=0.7)

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=1, color="gray", zorder=-1)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(3, 3, figsize=(8, 8))
plot_tsne_on_ax(axs[0, 0], pd.concat([train_embeddings_clip_vit_large, val_embeddings_clip_vit_large], ignore_index=True), "CLIP Large")
plot_tsne_on_ax(axs[0, 1], pd.concat([train_embeddings_clip_vit_large_336, val_embeddings_clip_vit_large_336], ignore_index=True), "CLIP Large 336")
plot_tsne_on_ax(axs[0, 2], pd.concat([train_embeddings_clip_vit_large_laion, val_embeddings_clip_vit_large_laion], ignore_index=True), "CLIP Large Laion")
plot_tsne_on_ax(axs[1, 0], pd.concat([train_embeddings_regnety_1280, val_embeddings_regnety_1280], ignore_index=True), "RegNetY 1280")
plot_tsne_on_ax(axs[1, 1], pd.concat([train_embeddings_swinv2_192, val_embeddings_swinv2_192], ignore_index=True), "SwinV2 192")
plot_tsne_on_ax(axs[1, 2], pd.concat([train_embeddings_swinv2_384, val_embeddings_swinv2_384], ignore_index=True), "SwinV2 384")
plot_tsne_on_ax(axs[2, 0], pd.concat([train_embeddings_beit_large_224, val_embeddings_beit_large_224], ignore_index=True), "BEiT Large 224")
plot_tsne_on_ax(axs[2, 1], pd.concat([train_embeddings_beit_large_512, val_embeddings_beit_large_512], ignore_index=True), "BEiT Large 512")
plot_tsne_on_ax(axs[2, 2], pd.concat([train_embeddings_dino_giant, val_embeddings_dino_giant], ignore_index=True), "DINOv2 Giant")

plt.suptitle("t-SNE (n=2) on Image Embeddings", fontsize=16)
plt.tight_layout()
plt.show()


class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


def train_linear_classifier(train_df, val_df=None, epochs=5000, lr=0.01):
    embeddings_train = train_df.drop(columns=["image"]).values
    embeddings_train = torch.tensor(embeddings_train).float().to(device)
    img_names_train = train_df["image"].values
    labels_train = get_gt_labels_for_img_names(img_names_train)
    labels_train = torch.tensor(labels_train).float().to(device)

    if val_df is not None:
        embeddings_val = val_df.drop(columns=["image"]).values
        embeddings_val = torch.tensor(embeddings_val).float().to(device)
        img_names_val = val_df["image"].values
        labels_val = get_gt_labels_for_img_names(img_names_val)

    # Define the linear probe model
    model = LinearClassifier(embeddings_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    # Training loop
    for epoch in range(1, epochs+1):
        # Shuffle the data
        perm = torch.randperm(embeddings_train.size(0))  # Generate a random 
        features_shuffled = embeddings_train[perm]
        labels_shuffled = labels_train[perm]

        model.train()
        optimizer.zero_grad()

        outputs = model(features_shuffled)
        loss = criterion(outputs.squeeze(), labels_shuffled.float())
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        auc = roc_auc_score(labels_shuffled.cpu().numpy(), torch.sigmoid(outputs).cpu().detach().numpy().flatten())
        train_aucs.append(auc)

        if val_df is not None:
            model.eval()
            with torch.no_grad():
                outputs_val = model(embeddings_val)
                predictions_val = torch.sigmoid(outputs_val).cpu().numpy().flatten()
                auc = roc_auc_score(labels_val, predictions_val)
                val_aucs.append(auc)
                val_loss = criterion(outputs_val.squeeze(), torch.tensor(labels_val).float().to(device))
                val_losses.append(val_loss.item())

        if epoch % 100 == 0:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax[0].set_title("Losses") 
            ax[0].plot(train_losses, label="Train Loss")
            if val_df is not None:
                ax[0].plot(val_losses, label="Validation Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].set_yscale("log")
            ax[0].grid(True, linestyle="--")
            ax[0].legend()

            ax[1].set_title("Area Under ROC Curve")
            ax[1].plot(train_aucs, label="Train AUC")
            if val_df is not None:
                ax[1].plot(val_aucs, label="Validation AUC")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("AUC")
            ax[1].grid(True, linestyle="--")
            ax[1].legend()

            fig.suptitle(f"Training Progress Epoch: {epoch}/{epochs}")
            fig.tight_layout()
            plt.show()

    return model


trained_linear_classifier = train_linear_classifier(train_embeddings_clip_vit_large, val_embeddings_clip_vit_large)


trained_linear_classifier = train_linear_classifier(train_embeddings_regnety_1280, val_embeddings_regnety_1280)


trained_linear_classifier = train_linear_classifier(train_embeddings_swinv2_192, val_embeddings_swinv2_192)


trained_linear_classifier = train_linear_classifier(train_embeddings_beit_large_224, val_embeddings_beit_large_224)


trained_linear_classifier = train_linear_classifier(train_embeddings_dino_giant, val_embeddings_dino_giant)


def create_submission_file(linear_model, embeddings_path, save_path):
    all_embeddings = pd.read_csv(embeddings_path)
    embeddings = all_embeddings.drop(columns=["image"]).values
    img_names = all_embeddings["image"].values

    linear_model.eval()
    with torch.no_grad():
        outputs = linear_model(torch.tensor(embeddings).float().to(device))
        predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
    
    submission_df = pd.DataFrame({
        "image": img_names,
        "label": predictions
    })
    submission_df.to_csv(save_path, index=False)


linear_classifier_clip_vit_large = train_linear_classifier(pd.concat([train_embeddings_clip_vit_large, val_embeddings_clip_vit_large], ignore_index=True))
linear_classifier_clip_vit_large_336 = train_linear_classifier(pd.concat([train_embeddings_clip_vit_large_336, val_embeddings_clip_vit_large_336], ignore_index=True))
linear_classifier_clip_vit_large_laion = train_linear_classifier(pd.concat([train_embeddings_clip_vit_large_laion, val_embeddings_clip_vit_large_laion], ignore_index=True))
linear_classifier_regnety_1280 = train_linear_classifier(pd.concat([train_embeddings_regnety_1280, val_embeddings_regnety_1280], ignore_index=True))
linear_classifier_swinv2_192 = train_linear_classifier(pd.concat([train_embeddings_swinv2_192, val_embeddings_swinv2_192], ignore_index=True))
linear_classifier_swinv2_384 = train_linear_classifier(pd.concat([train_embeddings_swinv2_384, val_embeddings_swinv2_384], ignore_index=True))
linear_classifier_beit_large_224 = train_linear_classifier(pd.concat([train_embeddings_beit_large_224, val_embeddings_beit_large_224], ignore_index=True))
linear_classifier_beit_large_512 = train_linear_classifier(pd.concat([train_embeddings_beit_large_512, val_embeddings_beit_large_512], ignore_index=True))
linear_classifier_dino_giant = train_linear_classifier(pd.concat([train_embeddings_dino_giant, val_embeddings_dino_giant], ignore_index=True))


os.makedirs(os.path.join(base_dir_output, "submissions"), exist_ok=True)
create_submission_file(linear_classifier_clip_vit_large, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_test.csv"), os.path.join(base_dir_output, "submissions", "submission_clip_vit_large.csv"))
create_submission_file(linear_classifier_clip_vit_large_336, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_336_test.csv"), os.path.join(base_dir_output, "submissions", "submission_clip_vit_large_336.csv"))
create_submission_file(linear_classifier_clip_vit_large_laion, os.path.join(base_dir_output, "embeddings", "embeddings_clip_vit_large_laion_test.csv"), os.path.join(base_dir_output, "submissions", "submission_clip_vit_large_laion.csv"))
create_submission_file(linear_classifier_regnety_1280, os.path.join(base_dir_output, "embeddings", "embeddings_regnety_1280_test.csv"), os.path.join(base_dir_output, "submissions", "submission_regnety_1280.csv"))
create_submission_file(linear_classifier_swinv2_192, os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_192_test.csv"), os.path.join(base_dir_output, "submissions", "submission_swinv2_192.csv"))
create_submission_file(linear_classifier_swinv2_384, os.path.join(base_dir_output, "embeddings", "embeddings_swinv2_384_test.csv"), os.path.join(base_dir_output, "submissions", "submission_swinv2_384.csv"))
create_submission_file(linear_classifier_beit_large_224, os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_224_test.csv"), os.path.join(base_dir_output, "submissions", "submission_beit_large_224.csv"))
create_submission_file(linear_classifier_beit_large_512, os.path.join(base_dir_output, "embeddings", "embeddings_beit_large_512_test.csv"), os.path.join(base_dir_output, "submissions", "submission_beit_large_512.csv"))
create_submission_file(linear_classifier_dino_giant, os.path.join(base_dir_output, "embeddings", "embeddings_dino_giant_test.csv"), os.path.join(base_dir_output, "submissions", "submission_dino_giant.csv"))

