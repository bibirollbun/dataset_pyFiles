# Keep in mind that I'm using CIFAR - 10 dataset, for model I'm using TinyViT + Random Search
import os
import time
import glob
import math
import json
import random
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
print("Setup was done")


# For my own convenience I'm using reproducability to not see everytime different results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2470, 0.2435, 0.2616)


def get_transforms(use_randaugment=True, cutout=False):
    train_ops = [
        T.RandomResizedCrop(32, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
    if use_randaugment and hasattr(T, "RandAugment"):
        train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
    train_ops.extend([T.ToTensor(), T.Normalize(cifar10_mean, cifar10_std)])
    if cutout:
        
        class Cutout:
            def __init__(self, n_holes=1, length=8):
                self.n_holes = n_holes
                self.length = length
            def __call__(self, img):
                c,h,w = img.shape
                mask = np.ones((h,w), np.float32)
                for _ in range(self.n_holes):
                    y = np.random.randint(h)
                    x = np.random.randint(w)
                    y1 = np.clip(y-self.length//2,0,h)
                    y2 = np.clip(y+self.length//2,0,h)
                    x1 = np.clip(x-self.length//2,0,w)
                    x2 = np.clip(x+self.length//2,0,w)
                    mask[y1:y2, x1:x2] = 0.
                mask = torch.from_numpy(mask).type_as(img[0])
                mask = mask.unsqueeze(0).repeat(c,1,1)
                return img*mask
        train_ops.append(Cutout())
    train_transform = T.Compose(train_ops)
    test_transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(cifar10_mean, cifar10_std)
    ])
    return train_transform, test_transform


def get_cifar10_loaders_with_val(batch_size=128, data_dir="./data",
                                 use_randaugment=True, cutout=False,
                                 val_fraction=0.1, seed=42, num_workers=4):
    train_transform, test_transform = get_transforms(use_randaugment, cutout)
    full_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    if val_fraction > 0.0:
        n = len(full_train)
        val_n = int(n * val_fraction)
        train_n = n - val_n
        train_set, val_set = random_split(full_train, [train_n, val_n], generator=torch.Generator().manual_seed(seed))
        val_indices = val_set.indices if hasattr(val_set, "indices") else None
        if val_indices is not None:
            val_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=test_transform)
            from torch.utils.data import Subset
            val_set = Subset(val_set, val_indices)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = None

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, emb_dim=128, mlp_dim=256, heads=4, dropout=0.1, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size)**2
        self.patch_dim = in_ch * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,self.num_patches+1,emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, dim_feedforward=mlp_dim,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def _patchify(self, x):
        B,C,H,W = x.shape
        p = self.patch_size
        x = x.view(B,C,H//p,p,W//p,p)
        x = x.permute(0,2,4,1,3,5).contiguous()
        x = x.view(B, -1, C*p*p)
        return x

    def forward(self, x):
        B = x.size(0)
        x = self._patchify(x)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:,0])
        x = self.head(x)
        return x


def mixup_data(x,y,alpha=0.2):
    if alpha <= 0: return x,y,None,1.0
    lam = np.random.beta(alpha,alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam*x + (1-lam)*x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    model.train()
    running_loss, correct, total = 0.0,0,0
    t0 = time.time()
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        if use_mixup and mixup_alpha>0:
            xb, y_a, y_b, lam = mixup_data(xb,yb,mixup_alpha)
            out = model(xb)
            loss = lam*criterion(out,y_a) + (1-lam)*criterion(out,y_b)
        else:
            out = model(xb)
            loss = criterion(out,yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*xb.size(0)
        preds = out.argmax(dim=1)
        if use_mixup and mixup_alpha>0:
            correct += (preds==y_a).sum().item()
        else:
            correct += (preds==yb).sum().item()
        total += xb.size(0)
    return running_loss/total, correct/total, time.time()-t0


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0,0,0
    t0=time.time()
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out,yb)
            running_loss += loss.item()*xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
    return running_loss/total, correct/total, time.time()-t0


baseline_config = {
    "patch_size": 4,
    "emb_dim": 64,
    "mlp_mult": 2,
    "heads": 4,
    "dropout": 0.1,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "batch_size": 128,
    "mixup_alpha": 0.0,
    "label_smoothing": 0.0,
    "randaugment": True,
    "cutout": False,
    "max_epochs": 20 # from time to time I'm correcting it
}


train_loader, val_loader, test_loader = get_cifar10_loaders_with_val(
    batch_size=baseline_config["batch_size"],
    use_randaugment=baseline_config["randaugment"],
    cutout=baseline_config["cutout"],
    val_fraction=0.1
)


model = TinyViT(img_size=32,
                patch_size=baseline_config["patch_size"],
                emb_dim=baseline_config["emb_dim"],
                mlp_dim=baseline_config["emb_dim"]*baseline_config["mlp_mult"],
                heads=baseline_config["heads"],
                dropout=baseline_config["dropout"]).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=baseline_config["label_smoothing"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=baseline_config["lr"], weight_decay=baseline_config["weight_decay"])

history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
total_train_time = 0.0
best_val_acc = 0.0
best_epoch = 0

for epoch in range(baseline_config["max_epochs"]):
    t_loss, t_acc, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device,
                                            use_mixup=(baseline_config["mixup_alpha"]>0.0),
                                            mixup_alpha=baseline_config["mixup_alpha"])
    v_loss, v_acc, _ = validate(model, val_loader, criterion, device)
    total_train_time += epoch_time
    history["epoch"].append(epoch+1)
    history["train_loss"].append(t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)
    history["epoch_time"].append(epoch_time)
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        best_epoch = epoch+1
        # Here I'm saving the best validation checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": best_epoch
        }, "baseline_best_checkpoint.pt")
    print(f"Epoch {epoch+1}/{baseline_config['max_epochs']} | train_acc={t_acc:.4f} | val_acc={v_acc:.4f}")


os.makedirs("baseline_results", exist_ok=True)
torch.save(model.state_dict(), "baseline_results/tinyvit_baseline_final.pth")
with open("baseline_results/history.json", "w") as f: json.dump(history, f, indent=2)

baseline_summary = {
    "best_val_acc": float(best_val_acc),
    "best_epoch": int(best_epoch),
    "total_train_time_s": float(total_train_time),
    "config": baseline_config
}
with open("baseline_results/summary.json", "w") as f: json.dump(baseline_summary, f, indent=2)


plt.figure(figsize=(8,4))
plt.plot(history["epoch"], history["train_acc"], label="train_acc")
plt.plot(history["epoch"], history["val_acc"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline TinyViT (1 layer) Accuracy - 100 Epochs")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("baseline_results/baseline_acc_plot.png")
plt.show()


def sample_random_config():
    """Sample a random configuration from predefined ranges."""
    config = {}
    config["patch_size"]   = random.choice([2, 4])
    config["emb_dim"]      = random.choice([64, 96, 128])
    config["mlp_mult"]     = random.choice([2, 3, 4])
    config["heads"]        = random.choice([2, 4, 8])
    config["dropout"]      = random.choice([0.0, 0.1, 0.2, 0.3])

    lr_log_min, lr_log_max = math.log10(1e-4), math.log10(3e-3)
    wd_log_min, wd_log_max = math.log10(1e-4), math.log10(1e-1)

    config["lr"] = 10 ** random.uniform(lr_log_min, lr_log_max)
    config["weight_decay"] = 10 ** random.uniform(wd_log_min, wd_log_max)

    config["batch_size"]   = random.choice([64, 128, 256])
    config["mixup_alpha"]  = random.choice([0.0, 0.1, 0.2, 0.3])
    config["label_smoothing"] = random.choice([0.0, 0.05, 0.1])
    config["randaugment"]  = random.choice([True, False])
    config["cutout"]       = random.choice([True, False])
    config["max_epochs"]   = 30  # from time to time I changing it, to see whether I can afford myself that time

    return config


def build_loaders_for_config(config):
    train_loader, val_loader, test_loader = get_cifar10_loaders_with_val(
        batch_size    = config["batch_size"],
        use_randaugment = config["randaugment"],
        cutout        = config["cutout"],
        val_fraction  = 0.1,
        data_dir      = "./data",
        num_workers   = 4
    )
    return train_loader, val_loader, test_loader


def build_model_for_config(config):
    mlp_dim = config["emb_dim"] * config["mlp_mult"]
    model = TinyViT(
        img_size   = 32,
        patch_size = config["patch_size"],
        in_ch      = 3,
        emb_dim    = config["emb_dim"],
        mlp_dim    = mlp_dim,
        heads      = config["heads"],
        dropout    = config["dropout"],
        num_classes = 10
    ).to(device)
    return model


def run_one_trial(trial_id, config):
    print(f"\n========== Trial {trial_id} ==========")
    print("Config:")
    print(json.dumps(config, indent=2))

    set_seed(42 + trial_id)

    train_loader, val_loader, test_loader = build_loaders_for_config(config)

    model = build_model_for_config(config)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config["lr"],
        weight_decay = config["weight_decay"]
    )

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "epoch_time": []
    }

    best_val_acc = 0.0
    best_epoch   = 0

    train_start_time = time.time()
    for epoch in range(config["max_epochs"]):
        t_loss, t_acc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup  = (config["mixup_alpha"] > 0.0),
            mixup_alpha = config["mixup_alpha"]
        )
        v_loss, v_acc, _ = validate(model, val_loader, criterion, device)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["epoch_time"].append(epoch_time)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch + 1

        print(f"  Epoch {epoch+1}/{config['max_epochs']}: "
              f"train_acc={t_acc:.4f}, val_acc={v_acc:.4f}")

    total_train_time = time.time() - train_start_time

    val_loss, val_acc, _   = validate(model, val_loader, criterion, device)
    test_loss, test_acc, _ = validate(model, test_loader, criterion, device)

    model.eval()
    inf_start = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            _ = model(xb)
    inference_time = time.time() - inf_start

    result = {
        "trial_id": trial_id,
        "config": config,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "total_train_time_s": float(total_train_time),
        "inference_time_s": float(inference_time),
        "history": history
    }

    f_tuple = (
        config["patch_size"],
        1,
        config["emb_dim"],
        config["emb_dim"] * config["mlp_mult"],
        config["heads"]
    )

    print("\n  Trial summary:")
    print(f"    f = (Patch={f_tuple[0]}, Layers={f_tuple[1]}, "
          f"D={f_tuple[2]}, MLP={f_tuple[3]}, Heads={f_tuple[4]})")
    print(f"    Training HPs: lr={config['lr']:.5f}, "
          f"weight_decay={config['weight_decay']:.5f}, "
          f"batch_size={config['batch_size']}, "
          f"dropout={config['dropout']}, "
          f"mixup_alpha={config['mixup_alpha']}, "
          f"label_smoothing={config['label_smoothing']}, "
          f"randaugment={config['randaugment']}, cutout={config['cutout']}")
    print(f"    Total training time: {total_train_time:.2f} s")
    print(f"    Inference time (full test set): {inference_time:.2f} s")
    print(f"    Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"    Final val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")

    return result


num_trials = 8 # I should change this one to 32 combinations
results = []
best_result = None
best_val_acc_overall = -1.0

os.makedirs("random_search_results", exist_ok=True)

for trial_id in range(1, num_trials + 1):
    cfg = sample_random_config()
    res = run_one_trial(trial_id, cfg)
    results.append(res)

    if res["best_val_acc"] > best_val_acc_overall:
        best_val_acc_overall = res["best_val_acc"]
        best_result = res

        torch.save({
            "model_state": build_model_for_config(cfg).state_dict()
        }, "random_search_results/best_tinyvit_randomsearch.pth")


records = []
for r in results:
    cfg = r["config"]
    records.append({
        "trial_id": r["trial_id"],
        "best_val_acc": r["best_val_acc"],
        "test_acc": r["test_acc"],
        "train_time_s": r["total_train_time_s"],
        "inference_time_s": r["inference_time_s"],
        "patch_size": cfg["patch_size"],
        "emb_dim": cfg["emb_dim"],
        "mlp_mult": cfg["mlp_mult"],
        "heads": cfg["heads"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "mixup_alpha": cfg["mixup_alpha"],
        "label_smoothing": cfg["label_smoothing"],
        "randaugment": cfg["randaugment"],
        "cutout": cfg["cutout"],
    })

df = pd.DataFrame.from_records(records)
df.to_csv("random_search_results/random_search_summary.csv", index=False)

print("\nRandom search summary (per trial):")
print(df[[
    "trial_id", "best_val_acc", "test_acc", "train_time_s",
    "patch_size", "emb_dim", "mlp_mult", "heads", "lr", "batch_size"
]])


plt.figure(figsize=(8, 4))
plt.bar(df["trial_id"].astype(str), df["best_val_acc"])
plt.xlabel("Trial ID")
plt.ylabel("Best Validation Accuracy")
plt.title("Random Search: Best Val Accuracy per Trial")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("random_search_results/best_val_acc_per_trial.png")
plt.show()


if best_result is not None:
    hist = best_result["history"]
    plt.figure(figsize=(7, 4))
    plt.plot(hist["epoch"], hist["train_acc"], marker="o", label="Train acc")
    plt.plot(hist["epoch"], hist["val_acc"], marker="s", label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Best Trial #{best_result['trial_id']} Learning Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("random_search_results/best_trial_learning_curves.png")
    plt.show()


with open("random_search_results/all_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n===================================")
print("Short summary of first 5 trials:")
print("===================================")
for res in results[:5]:
    cfg = res["config"]
    f_tuple = (
        cfg["patch_size"],
        1,
        cfg["emb_dim"],
        cfg["emb_dim"] * cfg["mlp_mult"],
        cfg["heads"]
    )
    print(f"\nTrial {res['trial_id']}:")
    print(f"  f = (Patch={f_tuple[0]}, Layers={f_tuple[1]}, "
          f"D={f_tuple[2]}, MLP={f_tuple[3]}, Heads={f_tuple[4]})")
    print(f"  Training HPs: lr={cfg['lr']:.5f}, "
          f"weight_decay={cfg['weight_decay']:.5f}, "
          f"batch_size={cfg['batch_size']}, "
          f"dropout={cfg['dropout']}, "
          f"mixup_alpha={cfg['mixup_alpha']}, "
          f"label_smoothing={cfg['label_smoothing']}, "
          f"randaugment={cfg['randaugment']}, cutout={cfg['cutout']}")
    print(f"  Best val acc: {res['best_val_acc']:.4f} (epoch {res['best_epoch']})")
    print(f"  Final val acc: {res['final_val_acc']:.4f}, "
          f"Test acc: {res['test_acc']:.4f}")
    print(f"  Total training time: {res['total_train_time_s']:.2f} s")
    print(f"  Inference time (test set): {res['inference_time_s']:.2f} s")


print("\n===================================")
print("Best configuration over all trials")
print("===================================")
if best_result is not None:
    cfg = best_result["config"]
    f_tuple = (
        cfg["patch_size"],
        1,
        cfg["emb_dim"],
        cfg["emb_dim"] * cfg["mlp_mult"],
        cfg["heads"]
    )
    print(f"Best validation accuracy: {best_result['best_val_acc']:.4f} "
          f"(epoch {best_result['best_epoch']})")
    print(f"Corresponding test accuracy: {best_result['test_acc']:.4f}")
    print(f"f = (Patch={f_tuple[0]}, Layers={f_tuple[1]}, "
          f"D={f_tuple[2]}, MLP={f_tuple[3]}, Heads={f_tuple[4]})")
    print("Training HPs:")
    print(f"  lr={cfg['lr']:.5f}, weight_decay={cfg['weight_decay']:.5f}")
    print(f"  batch_size={cfg['batch_size']}, dropout={cfg['dropout']}")
    print(f"  mixup_alpha={cfg['mixup_alpha']}, "
          f"label_smoothing={cfg['label_smoothing']}")
    print(f"  randaugment={cfg['randaugment']}, cutout={cfg['cutout']}")
    print(f"  Total training time: {best_result['total_train_time_s']:.2f} s")
    print(f"  Inference time (test set): {best_result['inference_time_s']:.2f} s")
else:
    print("No trials were run.")


records = []
for r in results:
    cfg = r["config"]
    records.append({
        "trial_id": r["trial_id"],
        "best_val_acc": r["best_val_acc"],
        "test_acc": r["test_acc"],
        "total_train_time_s": r["total_train_time_s"],
        "inference_time_s": r["inference_time_s"],
        "patch_size": cfg["patch_size"],
        "emb_dim": cfg["emb_dim"],
        "mlp_mult": cfg["mlp_mult"],
        "heads": cfg["heads"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "mixup_alpha": cfg["mixup_alpha"],
        "label_smoothing": cfg["label_smoothing"],
        "randaugment": cfg["randaugment"],
        "cutout": cfg["cutout"],
    })

df = pd.DataFrame.from_records(records)

df_sorted = df.sort_values("best_val_acc", ascending=False)

df_top5 = df_sorted.head(5).reset_index(drop=True)

print("\n===================================")
print("Top-5 search spaces (by validation accuracy)")
print("===================================")

for idx, row in df_top5.iterrows():
    f_tuple = (
        row["patch_size"],
        1,  # Layers = 1 (fixed in TinyViT)
        row["emb_dim"],
        row["emb_dim"] * row["mlp_mult"],
        row["heads"]
    )
    print(f"\nRank {idx+1} (Trial {row['trial_id']}):")
    print(f"  f = (Patch={f_tuple[0]}, Layers={f_tuple[1]}, "
          f"D={f_tuple[2]}, MLP={f_tuple[3]}, Heads={f_tuple[4]})")
    print(f"  Training HPs:")
    print(f"    lr={row['lr']:.5f}, weight_decay={row['weight_decay']:.5f}")
    print(f"    batch_size={row['batch_size']}, dropout={row['dropout']}")
    print(f"    mixup_alpha={row['mixup_alpha']}, "
          f"label_smoothing={row['label_smoothing']}")
    print(f"    randaugment={row['randaugment']}, cutout={row['cutout']}")
    print(f"  Performance:")
    print(f"    Best val acc = {row['best_val_acc']:.4f}")
    print(f"    Test acc     = {row['test_acc']:.4f}")
    print(f"    Total training time = {row['total_train_time_s']:.2f} s")
    print(f"    Inference time (test set) = {row['inference_time_s']:.2f} s")

best_among_top5 = df_top5.sort_values("test_acc", ascending=False).iloc[0]

print("\n===================================")
print("Best TEST score among Top-5 search spaces")
print("===================================")

f_best = (
    best_among_top5["patch_size"],
    1,
    best_among_top5["emb_dim"],
    best_among_top5["emb_dim"] * best_among_top5["mlp_mult"],
    best_among_top5["heads"]
)

print(f"Best test accuracy (among Top-5) = {best_among_top5['test_acc']:.4f}")
print(f"Achieved by Trial {int(best_among_top5['trial_id'])}")
print(f"f = (Patch={f_best[0]}, Layers={f_best[1]}, "
      f"D={f_best[2]}, MLP={f_best[3]}, Heads={f_best[4]})")
print("Training HPs:")
print(f"  lr={best_among_top5['lr']:.5f}, "
      f"weight_decay={best_among_top5['weight_decay']:.5f}")
print(f"  batch_size={best_among_top5['batch_size']}, "
      f"dropout={best_among_top5['dropout']}")
print(f"  mixup_alpha={best_among_top5['mixup_alpha']}, "
      f"label_smoothing={best_among_top5['label_smoothing']}")
print(f"  randaugment={best_among_top5['randaugment']}, "
      f"cutout={best_among_top5['cutout']}")
print(f"  Total training time: {best_among_top5['total_train_time_s']:.2f} s")
print(f"  Inference time (test set): {best_among_top5['inference_time_s']:.2f} s")

