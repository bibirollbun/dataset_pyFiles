import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm

DATA_DIR    = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'
CSV_PATH    = os.path.join(DATA_DIR, 'stage_2_train.csv')
DCM_DIR     = os.path.join(DATA_DIR, 'stage_2_train')
PNG_ROOT    = '/kaggle/working/png_data'
OUTPUT_DIR  = '/kaggle/working/results'

for d in [PNG_ROOT, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

BATCH_SIZE    = 16  
NUM_EPOCHS    = 20  
LEARNING_RATE = 5e-5 
PATIENCE      = 5 
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES   = ['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']
BALANCE_N     = 1000

def convert_dcm_to_png(dcm_path, png_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array
    def apply_window(image, center, width):
        img = (image - (center - width / 2)) / width
        return np.clip(img, 0, 1)
    b, s, bo = apply_window(img, 40, 80), apply_window(img, 80, 200), apply_window(img, 600, 2800)
    combined = (np.stack([b, s, bo], axis=-1) * 255).astype(np.uint8)
    Image.fromarray(combined).save(png_path)

df = pd.read_csv(CSV_PATH).drop_duplicates()
df[['image','subtype']] = df['ID'].str.rsplit('_', n=1, expand=True)
df = df[df['subtype'] != 'any'].pivot(index='image', columns='subtype', values='Label').reset_index()
df[CLASS_NAMES] = df[CLASS_NAMES].fillna(0).astype(int)

pos_imgs = []
for cls in CLASS_NAMES:
    cls_pos = df[df[cls] == 1]['image'].unique()
    pos_imgs.extend(cls_pos[:min(BALANCE_N, len(cls_pos))])
pos_imgs = list(set(pos_imgs))
neg_imgs = np.random.choice(df[df[CLASS_NAMES].sum(axis=1) == 0]['image'].unique(), len(pos_imgs), replace=False)
selected = list(pos_imgs) + list(neg_imgs)

for img_id in tqdm(selected, desc="Converting"):
    dcm_p, png_p = os.path.join(DCM_DIR, img_id + '.dcm'), os.path.join(PNG_ROOT, img_id + '.png')
    if os.path.exists(dcm_p) and not os.path.exists(png_p): convert_dcm_to_png(dcm_p, png_p)

balanced_df = df[df['image'].isin(selected)]
train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

pos_counts = torch.tensor(train_df[CLASS_NAMES].sum().values, dtype=torch.float32)
neg_counts = len(train_df) - pos_counts
class_weights = (neg_counts / (pos_counts + 1e-6)).to(DEVICE)

class MultiHemoDataset(Dataset):
    def __init__(self, df, root, tf):
        self.df, self.root, self.tf = df.reset_index(drop=True), root, tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, r['image'] + '.png')).convert('RGB')
        img = self.tf(img)
        lbl = torch.from_numpy(r[CLASS_NAMES].astype(np.float32).to_numpy())
        return img, lbl

tf_train = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tf_val = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(MultiHemoDataset(train_df, PNG_ROOT, tf_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MultiHemoDataset(val_df, PNG_ROOT, tf_val), batch_size=BATCH_SIZE, shuffle=False)

def get_model(arch, nc):
    if arch == 'vgg16':
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, nc)
    elif arch == 'vgg19':
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, nc)
    elif arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, nc)
    elif arch == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, nc)
    return m.to(DEVICE)

def run_experiment(arch):
    print(f"\nSTARTING: {arch}")
    model = get_model(arch, len(CLASS_NAMES))
    crit = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    val_crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_l, count, h = np.inf, 0, {'t': [], 'v': []}

    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = crit(model(x), y)
            l.backward()
            opt.step()
            tl += l.item() * x.size(0)
        
        model.eval()
        vl = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vl += val_crit(model(x), y).item() * x.size(0)
        
        atl, avl = tl/len(train_df), vl/len(val_df)
        h['t'].append(atl); h['v'].append(avl)
        sched.step(avl)
        print(f"Ep {ep} - Train Loss: {atl:.4f}, Val Loss: {avl:.4f} | LR: {opt.param_groups[0]['lr']:.7f}")
        
        if avl < best_l:
            best_l, count = avl, 0
            torch.save(model.state_dict(), f'best_{arch}.pth')
        else:
            count += 1
            if count >= PATIENCE: 
                print("Early stopping triggered.")
                break

    plt.figure()
    plt.plot(h['t'], label='Train Loss')
    plt.plot(h['v'], label='Val Loss')
    plt.title(f'{arch} Training History')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{arch}_loss_plot.png'))
    plt.show()

    model.load_state_dict(torch.load(f'best_{arch}.pth', weights_only=True))
    model.eval()
    probs, true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs.extend(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
            true.extend(y.numpy())
    
    preds = (np.array(probs) >= 0.5).astype(int)
    print(f"\nClassification Report for {arch}:")
    print(classification_report(true, preds, target_names=CLASS_NAMES, zero_division=0))
    for i, c in enumerate(CLASS_NAMES):
        print(f"Confusion Matrix for {c}:")
        print(confusion_matrix(np.array(true)[:, i], preds[:, i]))

for a in ['vgg16', 'vgg19', 'resnet50', 'densenet121']:
    run_experiment(a)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class_names = ['epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular']

all_cms = {
    'vgg16': {
        'epidural': [[1229, 152], [39, 166]], 'subdural': [[1033, 272], [102, 179]],
        'subarachnoid': [[947, 389], [72, 178]], 'intraparenchymal': [[1107, 219], [68, 192]],
        'intraventricular': [[1247, 131], [38, 170]]
    },
    'vgg19': {
        'epidural': [[1273, 108], [53, 152]], 'subdural': [[948, 357], [74, 207]],
        'subarachnoid': [[1012, 324], [94, 156]], 'intraparenchymal': [[1099, 227], [53, 207]],
        'intraventricular': [[1267, 111], [47, 161]]
    },
    'resnet50': {
        'epidural': [[1074, 307], [42, 163]], 'subdural': [[1143, 162], [158, 123]],
        'subarachnoid': [[1065, 271], [118, 132]], 'intraparenchymal': [[1206, 120], [96, 164]],
        'intraventricular': [[1298, 80], [61, 147]]
    },
    'densenet121': {
        'epidural': [[1278, 103], [41, 164]], 'subdural': [[1142, 163], [140, 141]],
        'subarachnoid': [[1123, 213], [104, 146]], 'intraparenchymal': [[1185, 141], [64, 196]],
        'intraventricular': [[1307, 71], [39, 169]]
    }
}

f1_scores = {'vgg16': 0.55, 'vgg19': 0.56, 'resnet50': 0.51, 'densenet121': 0.60}

def calculate_model_metrics(model_name, model_data):
    class_accuracies = []
    for cls in class_names:
        tn, fp = model_data[cls][0]
        fn, tp = model_data[cls][1]
        # Accuracy = (Correct Predictions) / (Total Predictions)
        acc = (tp + tn) / (tp + tn + fp + fn)
        class_accuracies.append(acc)
    
    avg_acc = np.mean(class_accuracies)
    return avg_acc

results = []
for model_name in all_cms.keys():
    acc = calculate_model_metrics(model_name, all_cms[model_name])
    results.append({
        'Model': model_name,
        'Hamming Accuracy': acc,
        'Weighted F1': f1_scores[model_name]
    })

df_results = pd.DataFrame(results)
print("--- FULL MODEL PERFORMANCE TABLE ---")
print(df_results)

plt.figure(figsize=(12, 7))
x = np.arange(len(df_results['Model']))
width = 0.35

plt.bar(x - width/2, df_results['Hamming Accuracy'], width, label='Hamming Accuracy', color='#3498db')
plt.bar(x + width/2, df_results['Weighted F1'], width, label='Weighted F1-Score', color='#e74c3c')

plt.xlabel('Model Architecture', fontsize=12, fontweight='bold')
plt.ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
plt.title('Hemorrhage Detection: Model Comparison (Accuracy vs F1)', fontsize=14)
plt.xticks(x, df_results['Model'], fontweight='bold')
plt.legend()

for i, val in enumerate(df_results['Hamming Accuracy']):
    plt.text(i - width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
for i, val in enumerate(df_results['Weighted F1']):
    plt.text(i + width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('all_models_comparison.png')
plt.show()

