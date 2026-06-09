# ======================================================
# Training Loop with Early Stopping (final training)
# ======================================================
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=15, device=None,
                patience=3, delta=1e-4, monitor="loss", save_path="best_resnet50_earlystop.pth", verbose=True):
    """
    monitor: 'loss' or 'acc' -> chooses whether to monitor validation loss (min) or validation accuracy (max)
    save_path: path to save best model weights when improvement occurs (None to disable)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "min" if monitor == "loss" else "max"
    early_stopper = EarlyStopping(patience=patience, delta=delta, mode=mode, save_path=save_path)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss_avg = train_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] Summary | "
                  f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # choose metric value to monitor
        metric = avg_val_loss if monitor == "loss" else val_acc

        # pass model so best weights are saved when improvement happens
        if early_stopper(metric, model=model):
            if save_path is not None:
                print(f"Early stopping triggered! Best model saved to: {save_path}")
            else:
                print("Early stopping triggered!")
            break

# ======================================================
# Optuna Objective Function
# ======================================================
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Dataset + Loader
    train_dataset = OSICDataset(train_df, data_dir, transform=transform)
    val_dataset = OSICDataset(val_df, data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    accuracy = train_and_eval(model, criterion, optimizer, train_loader, val_loader, device, epochs=3)
    return accuracy



# ======================================================
# Run Optuna Study
# ======================================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_params = study.best_trial.params
print("Best hyperparameters:", best_params)
print("Best accuracy:", study.best_value)

# ======================================================
# Retrain Best Model (final)
# ======================================================
best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

train_dataset = OSICDataset(train_df, data_dir, transform=transform)
val_dataset = OSICDataset(val_df, data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = nn.CrossEntropyLoss()
if best_params["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9)

# --- final training: monitor='loss' (default). If you prefer to stop on val accuracy, set monitor='acc' ---
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=20,
            patience=3, delta=1e-4, monitor="loss", save_path="best_resnet50_earlystop.pth", verbose=True)

# If you want the final saved file to be named exactly as before, copy it:
try:
    torch.save(model.state_dict(), "best_resnet50.pth")
    print("Final model saved as best_resnet50.pth (latest weights).")
except Exception:
    # saving model might have already been done by early stopper; ignore if not available
    pass

print("Done.")


# ======================================================
# Imports
# ======================================================
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt

# ======================================================
# 1ï¸�âƒ£ DICOM Loader (Fast)
# ======================================================
def load_dicom_as_tensor(dcm_path, device):
    dcm = pydicom.dcmread(dcm_path, force=True)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize and resize
    img = cv2.resize(img, (224, 224))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.repeat(img[..., None], 3, axis=-1)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return img_rgb, tensor

# ======================================================
# 2ï¸�âƒ£ Simple Grad-CAM Class (Optimized)
# ======================================================
class FastGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Attach forward and backward hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

# ======================================================
# 3ï¸�âƒ£ Load Model (Example: ResNet50)
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device).eval()

# ======================================================
# 4ï¸�âƒ£ Select DICOM & Generate Heatmap
# ======================================================
dcm_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/12.dcm"

orig_img, input_tensor = load_dicom_as_tensor(dcm_path, device)

gradcam = FastGradCAM(model, model.layer4[-1])
heatmap = gradcam(input_tensor)

# ======================================================
# 5ï¸�âƒ£ Overlay Heatmap (Fast)
# ======================================================
def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, (img * 255).astype(np.uint8), 1 - alpha, 0)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ğŸ”¥ Fast Grad-CAM")
    plt.show()

overlay_heatmap(orig_img, heatmap)



##recent test 
# FULL GRAD-CAM + EXPLAINABILITY PIPELINE (AUTO SUMMARY)
import os
import time
import json
import numpy as np
import cv2
import pydicom
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt


# 1.DICOM Loader

def load_dicom_as_tensor(dcm_path, device):
    """Load DICOM file, normalize and convert to RGB tensor."""
    dcm = pydicom.dcmread(dcm_path, force=True)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize and resize
    img = cv2.resize(img, (224, 224))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.repeat(img[..., None], 3, axis=-1)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return img_rgb, tensor



# 2.Fast Grad-CAM Class

class FastGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Attach hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()



# 3.Load Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device).eval()


#4.Load DICOM & Generate Heatmap

dcm_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/12.dcm"
orig_img, input_tensor = load_dicom_as_tensor(dcm_path, device)

gradcam = FastGradCAM(model, model.layer4[-1])
heatmap = gradcam(input_tensor)


 
# 5.Overlay Heatmap
 
def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, (img * 255).astype(np.uint8), 1 - alpha, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ğŸ”¥ Fast Grad-CAM")
    plt.show()
    return overlay

overlay_img = overlay_heatmap(orig_img, heatmap)



# 6.Extract Top Hot Regions

def top_regions_from_heatmap(heatmap, topk=3, min_area_frac=0.01):
    """Extract top-K high-activation regions from a Grad-CAM heatmap."""
    H, W = heatmap.shape
    regions = []
    thresholds = [0.85, 0.6, 0.4]
    idc = 0

    for t in thresholds:
        _, bw = cv2.threshold((heatmap * 255).astype('uint8'), int(t * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area_frac = (w * h) / (W * H)
            if area_frac < min_area_frac:
                continue
            x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
            regions.append({
                "id": f"region_{idc}",
                "bbox_normalized": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                "avg_heat": float(np.mean(heatmap[y:y+h, x:x+w])),
                "max_heat": float(np.max(heatmap[y:y+h, x:x+w]))
            })
            idc += 1

    regions = sorted(regions, key=lambda r: r["max_heat"], reverse=True)
    return regions[:topk]



#7.Generate Automatic Heatmap Summary

def generate_heatmap_summary(heatmap):
    """Automatically generate a human-readable summary of Grad-CAM focus areas."""
    H, W = heatmap.shape
    grid_y = np.array_split(np.arange(H), 3)
    grid_x = np.array_split(np.arange(W), 3)

    y_labels = ["upper", "middle", "lower"]
    x_labels = ["left", "central", "right"]

    avg_intensity = np.zeros((3, 3))
    for i, ys in enumerate(grid_y):
        for j, xs in enumerate(grid_x):
            avg_intensity[i, j] = np.mean(heatmap[np.ix_(ys, xs)])

    # Identify top 2 activated zones
    flat_idx = np.argsort(avg_intensity.flatten())[::-1][:2]
    summaries = []
    for idx in flat_idx:
        i, j = divmod(idx, 3)
        summaries.append(f"{y_labels[i]}-{x_labels[j]}")

    return "Activation concentrated in " + " and ".join(summaries) + " regions."



# 8.Save Explainability JSON

regions = top_regions_from_heatmap(heatmap)
heatmap_summary = generate_heatmap_summary(heatmap)

result = {
    "image": dcm_path,
    "explainability": {"method": "Grad-CAM"},
    "model_info": {"name": "ResNet50", "weights": "ImageNet pretrained"},
    "heatmap_summary": heatmap_summary,
    "top_regions": regions,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}

print("Result",result)

with open("gradcam_report.json", "w") as f:
    json.dump(result, f, indent=2)

print("âœ… Grad-CAM report saved as gradcam_report.json")
print("ğŸ“‹ Auto summary:", heatmap_summary)



#9.Draw Top Regions
 
def draw_regions(overlay_img, regions):
    """Visualize bounding boxes for top Grad-CAM regions."""
    H, W, _ = overlay_img.shape
    img_draw = overlay_img.copy()
    for reg in regions:
        x1, y1, x2, y2 = reg["bbox_normalized"]
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, reg["id"], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ğŸ“� Top Activated Regions")
    plt.show()

draw_regions(overlay_img, regions)





def deletion_insertion_curve(model, img, heatmap, steps=50):
    device = next(model.parameters()).device
    img_tensor = torch.tensor(img.transpose(2,0,1)).unsqueeze(0).to(device).float()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    flat_idx = np.argsort(heatmap.flatten())[::-1]
    
    masks = np.ones_like(heatmap).flatten()
    scores = []

    with torch.no_grad():
        for s in range(0, len(flat_idx), len(flat_idx)//steps):
            masks[flat_idx[:s]] = 0
            masked_img = img * masks.reshape(heatmap.shape)[..., None]
            inp = transforms.ToTensor()(masked_img).unsqueeze(0).to(device)
            score = torch.softmax(model(inp), dim=1).max().item()
            scores.append(score)
    return scores

# Evaluate faithfulness
del_curve = deletion_insertion_curve(model, orig_img, heatmap)
auc = np.trapz(del_curve) / len(del_curve)

# Compute interpretability stats
entropy = -np.sum(heatmap * np.log(heatmap + 1e-8))
sparsity = np.mean(heatmap > 0.7)

evaluation_metrics = {
    "AUC_deletion": round(auc, 4),
    "Entropy": round(entropy, 4),
    "Sparsity": round(sparsity, 4)
}

print(json.dumps(evaluation_metrics, indent=2, default=lambda o: float(o)))







# ======================================================
# Imports
# ======================================================
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt

# ======================================================
# 1ï¸�âƒ£ DICOM Loader (Fast)
# ======================================================
def load_dicom_as_tensor(dcm_path, device):
    dcm = pydicom.dcmread(dcm_path, force=True)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize and resize
    img = cv2.resize(img, (224, 224))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.repeat(img[..., None], 3, axis=-1)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return img_rgb, tensor

# ======================================================
# 2ï¸�âƒ£ Simple Grad-CAM Class (Optimized)
# ======================================================
class FastGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Attach forward and backward hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

# ======================================================
# 3ï¸�âƒ£ Load Model (Example: ResNet50)
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device).eval()

# ======================================================
# 4ï¸�âƒ£ Select DICOM & Generate Heatmap
# ======================================================
dcm_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/12.dcm"

orig_img, input_tensor = load_dicom_as_tensor(dcm_path, device)

gradcam = FastGradCAM(model, model.layer4[-1])
heatmap = gradcam(input_tensor)

# ======================================================
# 5ï¸�âƒ£ Overlay Heatmap (Fast)
# ======================================================
def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, (img * 255).astype(np.uint8), 1 - alpha, 0)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ğŸ”¥ Fast Grad-CAM")
    plt.show()

overlay_heatmap(orig_img, heatmap)



# ======================================================
# ğŸ§  FULL GRAD-CAM + EXPLAINABILITY PIPELINE (AUTO SUMMARY)
# ======================================================
import os
import time
import json
import numpy as np
import cv2
import pydicom
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt


# ======================================================
# 1ï¸�âƒ£ DICOM Loader
# ======================================================
def load_dicom_as_tensor(dcm_path, device):
    """Load DICOM file, normalize and convert to RGB tensor."""
    dcm = pydicom.dcmread(dcm_path, force=True)
    img = dcm.pixel_array.astype(np.float32)

    # Normalize and resize
    img = cv2.resize(img, (224, 224))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.repeat(img[..., None], 3, axis=-1)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return img_rgb, tensor


# ======================================================
# 2ï¸�âƒ£ Fast Grad-CAM Class
# ======================================================
class FastGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Attach hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# ======================================================
# 3ï¸�âƒ£ Load Model
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device).eval()


# ======================================================
# 4ï¸�âƒ£ Load DICOM & Generate Heatmap
# ======================================================
dcm_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/13.dcm"
orig_img, input_tensor = load_dicom_as_tensor(dcm_path, device)

gradcam = FastGradCAM(model, model.layer4[-1])
heatmap = gradcam(input_tensor)


# ======================================================
# 5ï¸�âƒ£ Overlay Heatmap
# ======================================================
def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, (img * 255).astype(np.uint8), 1 - alpha, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ğŸ”¥ Fast Grad-CAM")
    plt.show()
    return overlay

overlay_img = overlay_heatmap(orig_img, heatmap)


# ======================================================
# 6ï¸�âƒ£ Extract Top Hot Regions
# ======================================================
def top_regions_from_heatmap(heatmap, topk=3, min_area_frac=0.01):
    """Extract top-K high-activation regions from a Grad-CAM heatmap."""
    H, W = heatmap.shape
    regions = []
    thresholds = [0.85, 0.6, 0.4]
    idc = 0

    for t in thresholds:
        _, bw = cv2.threshold((heatmap * 255).astype('uint8'), int(t * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area_frac = (w * h) / (W * H)
            if area_frac < min_area_frac:
                continue
            x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
            regions.append({
                "id": f"region_{idc}",
                "bbox_normalized": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                "avg_heat": float(np.mean(heatmap[y:y+h, x:x+w])),
                "max_heat": float(np.max(heatmap[y:y+h, x:x+w]))
            })
            idc += 1

    regions = sorted(regions, key=lambda r: r["max_heat"], reverse=True)
    return regions[:topk]


# ======================================================
# 7ï¸�âƒ£ Generate Automatic Heatmap Summary
# ======================================================
def generate_heatmap_summary(heatmap):
    """Automatically generate a human-readable summary of Grad-CAM focus areas."""
    H, W = heatmap.shape
    grid_y = np.array_split(np.arange(H), 3)
    grid_x = np.array_split(np.arange(W), 3)

    y_labels = ["upper", "middle", "lower"]
    x_labels = ["left", "central", "right"]

    avg_intensity = np.zeros((3, 3))
    for i, ys in enumerate(grid_y):
        for j, xs in enumerate(grid_x):
            avg_intensity[i, j] = np.mean(heatmap[np.ix_(ys, xs)])

    # Identify top 2 activated zones
    flat_idx = np.argsort(avg_intensity.flatten())[::-1][:2]
    summaries = []
    for idx in flat_idx:
        i, j = divmod(idx, 3)
        summaries.append(f"{y_labels[i]}-{x_labels[j]}")

    return "Activation concentrated in " + " and ".join(summaries) + " regions."


# ======================================================
# 8ï¸�âƒ£ Save Explainability JSON
# ======================================================
regions = top_regions_from_heatmap(heatmap)
heatmap_summary = generate_heatmap_summary(heatmap)

result = {
    "image": dcm_path,
    "explainability": {"method": "Grad-CAM"},
    "model_info": {"name": "ResNet50", "weights": "ImageNet pretrained"},
    "heatmap_summary": heatmap_summary,
    "top_regions": regions,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}

with open("gradcam_report.json", "w") as f:
    json.dump(result, f, indent=2)

print("âœ… Grad-CAM report saved as gradcam_report.json")
print("ğŸ“‹ Auto summary:", heatmap_summary)


# ======================================================
# 9ï¸�âƒ£ Optional: Draw Top Regions
# ======================================================
def draw_regions(overlay_img, regions):
    """Visualize bounding boxes for top Grad-CAM regions."""
    H, W, _ = overlay_img.shape
    img_draw = overlay_img.copy()
    for reg in regions:
        x1, y1, x2, y2 = reg["bbox_normalized"]
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, reg["id"], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ğŸ“� Top Activated Regions")
    plt.show()

draw_regions(overlay_img, regions)

# ======================================================
# âš ï¸� Disclaimer
# ======================================================
print("âš ï¸� This visualization shows model attention â€” not a medical diagnosis. Always confirm with a radiologist.")


