import numpy as np
import pandas as pd
import torch
import random
import os 

seed=42

np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import timm
import transformers
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoProcessor, AutoModel, CLIPModel, CLIPProcessor
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm import tqdm 
from catboost import CatBoostClassifier
from torchvision import transforms as v2
from PIL import Image
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


TEST_MODE=True


EPOCHS=5


train=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')
test=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv')
sample=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/sample_submission.csv')

breed_labels=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/breed_labels.csv')
color_labels=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/color_labels.csv')
state_labels=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/StateLabels.csv')


train_img_dir='/kaggle/input/petfinder-adoption-prediction/train_images'
test_img_dir='/kaggle/input/petfinder-adoption-prediction/test_images'


# === 1) Подготовка справочников ===
breed_map = breed_labels.set_index('BreedID')['BreedName'].to_dict()
breed_type_map = breed_labels.set_index('BreedID')['Type'].to_dict()  # иногда полезно
color_map = color_labels.set_index('ColorID')['ColorName'].to_dict()
state_map = state_labels.set_index('StateID')['StateName'].to_dict()

# карта для Type (по описанию PetFinder: 1=Dog, 2=Cat)
type_map = {1: 'Dog', 2: 'Cat'}

def enrich_petfinder_df(df):
    df = df.copy()

    # --- Породы ---
    df['Breed1Name'] = df['Breed1'].map(breed_map)
    df['Breed2Name'] = df['Breed2'].map(breed_map)
    # полезно знать, совпадает ли тип породы с заявленным Type
    df['Breed1Type'] = df['Breed1'].map(breed_type_map)
    df['Breed2Type'] = df['Breed2'].map(breed_type_map)
    # индикаторы наличия второй породы и «смешанная порода»
    df['HasBreed2'] = (df['Breed2'] > 0).astype('int8')
    df['IsMixedBreed'] = (df['HasBreed2'] == 1).astype('int8')

    # агрегированное поле «порода»
    def _agg_breed(row):
        b1 = row['Breed1Name'] if pd.notna(row['Breed1Name']) else ''
        b2 = row['Breed2Name'] if pd.notna(row['Breed2Name']) else ''
        if b2 and b2 != b1:
            return f"{b1} + {b2}"
        return b1
    df['BreedAgg'] = df.apply(_agg_breed, axis=1)

    # --- Цвета ---
    df['Color1Name'] = df['Color1'].map(color_map)
    df['Color2Name'] = df['Color2'].map(color_map)
    df['Color3Name'] = df['Color3'].map(color_map)

    # агрегированный список цветов и их количество
    def _agg_colors(row):
        cols = [row['Color1Name'], row['Color2Name'], row['Color3Name']]
        cols = [c for c in cols if isinstance(c, str) and len(c)]
        # уберём дубли, сохранив порядок
        seen, out = set(), []
        for c in cols:
            if c not in seen:
                seen.add(c); out.append(c)
        return out
    df['ColorsList'] = df.apply(_agg_colors, axis=1)
    df['NumColors'] = df['ColorsList'].apply(len).astype('int8')
    # удобная строка цветов
    df['ColorsAgg'] = df['ColorsList'].apply(lambda xs: ' / '.join(xs) if xs else '')

    # --- Штаты/регионы ---
    df['StateName'] = df['State'].map(state_map)

    # --- Тип питомца строкой ---
    df['TypeName'] = df['Type'].map(type_map)

    return df

# Применяем к train/test
train_en = enrich_petfinder_df(train)
test_en  = enrich_petfinder_df(test)

# (необязательно) покажем, что всё сработало
cols_to_peek = [
    'PetID','Type','TypeName','Breed1','Breed1Name','Breed2','Breed2Name',
    'IsMixedBreed','Color1','Color1Name','Color2','Color2Name','Color3','Color3Name',
    'ColorsAgg','NumColors','State','StateName'
]
display(train_en[cols_to_peek].head())



if TEST_MODE:
    train_en=train_en[:40000]
else:
    train_en=train_en


tr_df, va_df = train_en.copy(), None
tr_df, va_df = train_test_split(
    train_en, test_size=0.2, random_state=seed, stratify=train_en["AdoptionSpeed"]
)


#TEXT_BACKBONE = "microsoft/deberta-v3-base" 


# --- add this class once (e.g. right after imports) ---
import torch
from torch import nn
from transformers import AutoModel

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class DebertaMeanPoolMSD(nn.Module):
    def __init__(self, backbone: str, num_labels: int = 5, n_drop: int = 5, p_drop: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone, num_labels=num_labels)
        self.num_labels = num_labels

        self.backbone = AutoModel.from_pretrained(backbone)  # без add_pooling_layer
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        hidden = self.backbone.config.hidden_size
        self.dropouts = nn.ModuleList([nn.Dropout(p_drop) for _ in range(n_drop)])
        self.classifier = nn.Linear(hidden, num_labels)

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    # Проксируем gradient checkpointing для Trainer
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)
        else:
            if hasattr(self.backbone, "config"):
                self.backbone.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        else:
            if hasattr(self.backbone, "config"):
                self.backbone.config.gradient_checkpointing = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # ВАЖНО: Trainer может прислать num_items_in_batch — удаляем
        kwargs.pop("num_items_in_batch", None)

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        logits = torch.stack([self.classifier(dp(pooled)) for dp in self.dropouts], dim=0).mean(0)

        # Не возвращаем loss — пусть Trainer сам посчитает с учетом label smoothing
        return {"logits": logits}


TEXT_BACKBONE = "microsoft/deberta-v3-base"
model_for_text = DebertaMeanPoolMSD(backbone=TEXT_BACKBONE, num_labels=5, n_drop=5, p_drop=0.2).to(device)



#model_for_text=AutoModelForSequenceClassification.from_pretrained(TEXT_BACKBONE, num_labels=5)
tokenizer=AutoTokenizer.from_pretrained(TEXT_BACKBONE)


PREFS = [
    "vit_base_patch16_224.augreg_in21k_ft1k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.dino",
    "vit_base_patch16_224"  # последний фолбэк — точно есть
]

def safe_create_vit(candidates, num_classes=5, drop_path_rate=0.1, device=device):
    last_err = None
    for name in candidates:
        try:
            print(f"Trying backbone: {name}")
            m = timm.create_model(name, pretrained=True,
                                  num_classes=num_classes, drop_path_rate=drop_path_rate).to(device)
            print("✓ Using image backbone:", name)
            return m, name
        except RuntimeError as e:
            msg = str(e)
            # Если "Invalid pretrained tag", попробуем ту же модель без тега (всё после первой точки)
            if "Invalid pretrained tag" in msg and "." in name:
                base = name.split(".", 1)[0]
                try:
                    print(f"Tag not supported, fallback to: {base}")
                    m = timm.create_model(base, pretrained=True,
                                          num_classes=num_classes, drop_path_rate=drop_path_rate).to(device)
                    print("✓ Using image backbone:", base)
                    return m, base
                except Exception as e2:
                    last_err = e2
                    print("Fallback failed:", e2)
            else:
                last_err = e
                print("Failed:", e)
    raise RuntimeError(f"Could not create any ViT backbone. Last error: {last_err}")

model_for_image, IMG_BACKBONE = safe_create_vit(PREFS)


#model_for_image=timm.create_model(IMG_BACKBONE, pretrained=True, num_classes=5,).to(device)









"""
class PetFinderDataset1(Dataset):
    def __init__(self, df, img_dir, transform, train, tabular, tokenizer, is_text,  ):
        self.df=df
        self.img_dir=img_dir
        self.transform=transform
        self.train=train
        self.tabular=tabular
        self.tokenizer=tokenizer
        self.is_text=is_text
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        image_root=row['PetID']
        image_all=os.path.join(self.img_dir, f'{image_root}-1.jpg')
        image=Image.open(image_all).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        else:
            image=image
        if self.is_text:
            text=row['Description']
            tokenized=self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=256, 
                return_tensors="pt"
                
                
            )
        if self.tabular is not None:
            tab=torch.tensor(row[self.tabular].astype('float32').values, dtype=torch.float32)
        else:
            tab=None

        

        
        if self.train:
            label=torch.tensor(int(row['AdoptionSpeed']), dtype=torch.long)
            return({
                'image': image,
                'label': label,
                'input_ids' : tokenized["input_ids"].squeeze(0),
                'attention_mask': tokenized["attention_mask"].squeeze(0),
                'tabular': tab
            })
        else:
            return({
                'image': image, 
                '':
                'input_ids' : tokenized["input_ids"].squeeze(0),
                'attention_mask': tokenized["attention_mask"].squeeze(0),
                'tabular': tab
            })

"""





class PetFinderDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, tabular_cols=None,
                 tokenizer=None, is_train=True, max_len=256):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.tabular_cols = tabular_cols or []
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = max_len

    def _load_image(self, pet_id):
        path1 = os.path.join(self.img_dir, f"{pet_id}-1.jpg")
        if os.path.exists(path1):
            img = Image.open(path1).convert("RGB")
        else:
            alts = sorted(glob(os.path.join(self.img_dir, f"{pet_id}-*.jpg")))
            img = Image.open(alts[0]).convert("RGB") if alts else Image.new("RGB",(256,256),(0,0,0))
        return self.transform(img) if self.transform else v2.ToTensor()(img)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_image(row["PetID"])

        text = row["Description"] if isinstance(row["Description"], str) else ""
        toks = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        input_ids      = toks["input_ids"].squeeze(0)
        attention_mask = toks["attention_mask"].squeeze(0)

        tab = torch.tensor(row[self.tabular_cols].astype("float32").values, dtype=torch.float32) \
              if self.tabular_cols else torch.empty(0, dtype=torch.float32)

        item = {"image": image, "input_ids": input_ids, "attention_mask": attention_mask, "tabular": tab}
        if self.is_train:
            item["labels"] = torch.tensor(int(row["AdoptionSpeed"]), dtype=torch.long)
        return item





class TextOnlyDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True, max_len=256):
        self.df=df.reset_index(drop=True); self.tok=tokenizer
        self.is_train=is_train; self.max_len=max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row=self.df.iloc[i]
        txt = row["Description"] if isinstance(row["Description"], str) else ""
        toks=self.tok(txt, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in toks.items()}
        if self.is_train:
            item["labels"] = torch.tensor(int(row["AdoptionSpeed"]), dtype=torch.long)
        return item

tr_text = TextOnlyDataset(tr_df, tokenizer, True)
va_text = TextOnlyDataset(va_df, tokenizer, True)

from sklearn.metrics import cohen_kappa_score, f1_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
    }





class CLIPDataset(Dataset):
    def __init__(self, df, img_dir, text_col="Description"):
        self.df=df
        self.img_dir=img_dir
        self.text_col=text_col
    def __len__(self):
        return len(self.df)
    def _pick_image(self, pet_id):
        # 1-я фотка, если нет — любая, если пусто — заглушка
        p1 = os.path.join(self.img_dir, f"{pet_id}-1.jpg")
        if os.path.exists(p1):
            return Image.open(p1).convert("RGB")
        alts = sorted(glob(os.path.join(self.img_dir, f"{pet_id}-*.jpg")))
        if alts:
            return Image.open(alts[0]).convert("RGB")
        # заглушка (редко, но лучше не падать)
        return Image.new("RGB", (256, 256), (0, 0, 0))
        
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        img=self._pick_image(row["PetID"])
        txt = row.get(self.text_col, "")
        if pd.isna(txt): txt = ""
        return {"image_pil": img, "text": txt}





TAB_COLS = [
    'Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
    'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health',
    'Quantity','Fee','VideoAmt','PhotoAmt','State'
]



categorical_feats=train[TAB_COLS].select_dtypes(include='object').columns.tolist()


IMG_SIZE=224


IMG_SIZE=224
train_tfms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tfms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


train_ds = PetFinderDataset(tr_df,  train_img_dir, train_tfms, TAB_COLS, tokenizer, is_train=True)
val_ds   = PetFinderDataset(va_df,  train_img_dir, val_tfms,   TAB_COLS, tokenizer, is_train=True)
test_ds  = PetFinderDataset(test_en,test_img_dir, val_tfms,   TAB_COLS, tokenizer, is_train=False)

# loaders






train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


# now InfoNCE then cross entropy 
model_clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32', ).to(device)
clip_processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')


def clip_collate_fn(batch):
    images = [b["image_pil"] for b in batch]
    texts  = [b["text"]     for b in batch]
    proc = clip_processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")
    # вернём CPU тензоры; перевод на GPU сделаем в train/eval лупах
    return proc


clip_train_ds = CLIPDataset(tr_df, img_dir=train_img_dir, text_col="Description")
clip_val_ds   = CLIPDataset(va_df, img_dir=train_img_dir, text_col="Description")

clip_train_loader = DataLoader(clip_train_ds, batch_size=64, shuffle=True,
                               num_workers=2, pin_memory=True, collate_fn=clip_collate_fn)
clip_val_loader   = DataLoader(clip_val_ds,   batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True, collate_fn=clip_collate_fn)


'''
from torch.optim import AdamW
from tqdm import tqdm

for p in model_clip.parameters(): 
    p.requires_grad = True  # при желании можно частично разморозить

opt = AdamW(model_clip.parameters(), lr=1e-5, weight_decay=0.02)

def train_clip_one_epoch(model, loader, optimizer):
    model.train()
    running = 0.0; n = 0
    pbar = tqdm(loader, desc="CLIP train", leave=False)
    for batch in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch, return_loss=True)   # считает 0.5*(i2t + t2i)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = batch["input_ids"].size(0)
        running += loss.item() * bs; n += bs
        pbar.set_postfix(loss=running / max(n,1))
    return running / max(n,1)

for epoch in range(1, EPOCHS+1):
    tr_loss = train_clip_one_epoch(model_clip, clip_train_loader, opt, )
    print(f"[epoch {epoch}] train_loss={tr_loss:.4f}")
'''


from transformers import CLIPModel, CLIPProcessor

#model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for p in model_clip.parameters():
    p.requires_grad = False
model_clip.eval()  # важно



import torch, numpy as np
import torch.nn.functional as F

@torch.no_grad()
def extract_clip_features(model, loader):
    model.eval()
    all_img, all_txt = [], []
    for batch in tqdm(loader, desc="CLIP feats (val)", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        img_f = model.get_image_features(pixel_values=batch["pixel_values"])   # (B,D)
        txt_f = model.get_text_features(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"])
        img_f = F.normalize(img_f, dim=-1)
        txt_f = F.normalize(txt_f, dim=-1)
        all_img.append(img_f.cpu()); all_txt.append(txt_f.cpu())
    all_img = torch.cat(all_img, dim=0).numpy()
    all_txt = torch.cat(all_txt, dim=0).numpy()
    return all_img, all_txt  # (N,D), (N,D)

def recall_at_k(sim, k):
    # sim: (N,N), строки — img, столбцы — txt; правильная пара всегда по диагонали
    idx_topk = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]  # индексы топ-k по строкам
    hits = np.any(idx_topk == np.arange(sim.shape[0])[:, None], axis=1)
    return float(hits.mean())

@torch.no_grad()
def evaluate_clip_retrieval(model, loader):
    img, txt = extract_clip_features(model, loader)
    sim = img @ txt.T
    r1  = recall_at_k(sim, 1)
    r5  = recall_at_k(sim, 5)
    r10 = recall_at_k(sim, 10)
    # симметрично для t2i
    sim_t = sim.T
    r1_t  = recall_at_k(sim_t, 1)
    r5_t  = recall_at_k(sim_t, 5)
    r10_t = recall_at_k(sim_t, 10)
    return {
        "i2t_R@1": r1, "i2t_R@5": r5, "i2t_R@10": r10,
        "t2i_R@1": r1_t, "t2i_R@5": r5_t, "t2i_R@10": r10_t
    }

metrics = evaluate_clip_retrieval(model_clip, clip_val_loader)
print(metrics)






use_fp16 = (device.type == "cuda")

args = TrainingArguments(
    output_dir="only_text",
    # батчи и аккумулирование — чтобы держать эффективный BS≈32 даже если GPU слабее
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,

    num_train_epochs=1, #5
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                  # ~10% разогрева почти всегда помогает

    # более стабильное обучение
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,        # слегка сглаживаем таргеты (полезно при дисбалансе)
    gradient_checkpointing=True,       # экономим память (немного медленнее)

    # валидация/сохранения/лучшая модель
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,

    # ускорение
    fp16=use_fp16,
    bf16=False,                        # на T4/Каггле bf16 обычно нет

    # логирование и детерминизм
    logging_steps=50,
    report_to="none",
    seed=42,

    # загрузчики
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

# ——— МЕТРИКИ: accuracy, macro/weighted F1 и Quadratic Weighted Kappa ———
def compute_metrics(eval_pred):
    """
    eval_pred: (predictions, label_ids) от Trainer
    predictions — либо logits, либо tuple(logits, ...).
    """
    preds, labels = eval_pred
    if isinstance(preds, (list, tuple)):
        logits = preds[0]
    else:
        logits = preds
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    y_true = labels
    y_pred = logits.argmax(axis=-1)

    acc  = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    qwk  = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return {
        "accuracy": acc,
        "macro_f1": f1_m,
        "weighted_f1": f1_w,
        "qwk": qwk,
    }



trainer=Trainer(
    args=args, 
    model=model_for_text,
    train_dataset=tr_text,
    eval_dataset=va_text,
    compute_metrics=compute_metrics,
)


#trainer.train()


# --- replace your model_for_text creation & TrainingArguments with this ---
TEXT_BACKBONE = "microsoft/deberta-v3-base"  # при наличии GPU можно "deberta-v3-large"
model_for_text = DebertaMeanPoolMSD(backbone=TEXT_BACKBONE, num_labels=5, n_drop=5, p_drop=0.2).to(device)

use_fp16 = (device.type == "cuda")
args = TrainingArguments(
    output_dir="only_text_msd",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=3 if not TEST_MODE else 1,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,
    fp16=use_fp16,
    bf16=False,
    logging_steps=50,
    report_to="none",
    seed=42,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)
trainer = Trainer(
    args=args,
    model=model_for_text,
    train_dataset=tr_text,
    eval_dataset=va_text,
    compute_metrics=compute_metrics,  # твоя функция ок
)
trainer.train()




criterion=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model_for_image.parameters(), lr=3e-4, weight_decay=0.05)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


'''
for epoch in range(1, EPOCHS+1):
    model_for_image.train()
    pbar=tqdm(train_loader, desc='training', leave=False)
    running_loss, num_correct, n=0.0,0,0

    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        X=batch['image'].to(device)
        y=batch['labels'].to(device)

        logits=model_for_image(X)
        loss=criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*X.size(0)
        pbar.set_postfix(loss=running_loss/((step+1)*X.size(0)))
    scheduler.step()
    print(f'train: loss={running_loss/n:.4f}')
    model_for_image.eval()
    
with torch.no_grad():
    pbar=tqdm(val_loader, desc='evaluation', leave=False)
    loss_sum, correct, n=0.0,0,0

    all_probs, all_targets=[],[]
    for batch in  pbar:
        X=batch['image'].to(device)
        y=batch['labels'].to(device)

        logits=model_for_image(X)

        loss=criterion(logits, y)
        loss_sum+=loss.item()*X.size(0)

        n+=X.size(0)
        probs=torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.detach().cpu().numpy())
    all_probs=np.concatenate(all_probs, axis=0)
    all_targets=np.concatenate(all_targets, axis=0)
    print(f"valid: loss={val_loss:.4f}")
    
    '''


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# Критерий с label smoothing — часто помогает на адопшн-рангах
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

use_amp = (device.type == "cuda")
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    all_probs, all_targets = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="validation", leave=False)
        for batch in pbar:
            X = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(X)
                loss   = criterion(logits, y)

            bs = X.size(0)
            loss_sum += loss.item() * bs
            n        += bs

            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.detach().cpu().numpy())

    val_loss   = loss_sum / max(n, 1)
    probs      = np.concatenate(all_probs, axis=0)
    targets    = np.concatenate(all_targets, axis=0)
    preds      = probs.argmax(axis=1)

    acc        = accuracy_score(targets, preds)
    macro_f1   = f1_score(targets, preds, average="macro")
    qwk        = cohen_kappa_score(targets, preds, weights="quadratic")

    metrics = {"acc": acc, "macro_f1": macro_f1, "qwk": qwk}
    return val_loss, probs, targets, metrics

# --------- ТРЕНИРОВКА С ВАЛИДАЦИЕЙ, КЛИППИНГОМ И EARLY STOP ---------
best_qwk   = -1.0
patience   = 3   # остановимся, если QWK не растёт 'patience' эпох подряд
wait       = 0

for epoch in range(1, EPOCHS + 1):
    model_for_image.train()
    running_loss, n = 0.0, 0
    pbar = tqdm(train_loader, desc=f"training epoch {epoch}", leave=False)

    for step, batch in enumerate(pbar):
        X = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model_for_image(X)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model_for_image.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = X.size(0)
        running_loss += loss.item() * bs
        n            += bs
        pbar.set_postfix(train_loss=running_loss / max(n, 1))

    train_loss = running_loss / max(n, 1)

    # — валидация —
    val_loss, val_probs, val_targets, val_metrics = evaluate(model_for_image, val_loader, criterion)

    # планировщик после эпохи
    try:
        scheduler.step()
    except Exception:
        pass

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"acc={val_metrics['acc']:.4f} | "
        f"macro_f1={val_metrics['macro_f1']:.4f} | "
        f"qwk={val_metrics['qwk']:.4f}"
    )

    # — сохраняем лучшую по QWK —
    if val_metrics["qwk"] > best_qwk:
        best_qwk = val_metrics["qwk"]
        wait = 0
        torch.save(
            {
                "model_state": model_for_image.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            "best_vit_qwk.pt",
        )
        print(f"✓ new best QWK={best_qwk:.4f} — checkpoint saved to best_vit_qwk.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping: no QWK improvement.")
            break



from transformers import AutoModel

# вытаскиваем encoder из обученного классификатора
from transformers import AutoModel

# берём базовый encoder той же архитектуры
text_encoder = AutoModel.from_pretrained(TEXT_BACKBONE).to(device).eval()

# подгружаем веса base из твоего дообученного классификатора
if hasattr(model_for_text, "deberta"):
    base_state = model_for_text.deberta.state_dict()
elif hasattr(model_for_text, "bert"):          # на всякий случай для совместимости
    base_state = model_for_text.bert.state_dict()
else:
    # универсально: многие HF-классы имеют .base_model
    base_state = model_for_text.base_model.state_dict()

text_encoder.load_state_dict(base_state)

@torch.no_grad()
def embed_texts(texts, tokenizer, model, batch_size=256, max_len=256):
    out_all = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        toks = tokenizer(list(chunk), padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks)                       # BaseModel output
        cls = out.last_hidden_state[:, 0, :]      # [CLS]/первый токен
        out_all.append(cls.float().cpu().numpy())
    return np.vstack(out_all).astype(np.float32)

tr_text_emb   = embed_texts(tr_df.Description.fillna("").tolist(), tokenizer, text_encoder)
va_text_emb   = embed_texts(va_df.Description.fillna("").tolist(), tokenizer, text_encoder)
test_text_emb = embed_texts(test_en.Description.fillna("").tolist(), tokenizer, text_encoder)


# test CLIP dataloader
clip_test_ds = CLIPDataset(test_en, img_dir=test_img_dir, text_col="Description")
clip_test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=clip_collate_fn)



import torch.nn.functional as F

@torch.no_grad()
def clip_image_embeddings(model, loader):
    model.eval()
    feats = []
    for batch in tqdm(loader, desc="CLIP img emb", leave=False):
        pixel_values = batch["pixel_values"].to(
            device, dtype=next(model.parameters()).dtype, non_blocking=True
        )
        f = model.get_image_features(pixel_values=pixel_values)
        f = F.normalize(f, dim=-1)
        feats.append(f.float().cpu().numpy())
    return np.vstack(feats).astype(np.float32)

@torch.no_grad()
def clip_text_embeddings(model, loader):
    model.eval()
    feats = []
    for batch in tqdm(loader, desc="CLIP txt emb", leave=False):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        f = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        f = F.normalize(f, dim=-1)
        feats.append(f.float().cpu().numpy())
    return np.vstack(feats).astype(np.float32)

# извлекаем
clip_tr_ld = DataLoader(clip_train_ds, batch_size=64, shuffle=False, num_workers=2,
                        pin_memory=True, collate_fn=clip_collate_fn)
clip_va_ld = clip_val_loader
clip_te_ld = clip_test_loader

tr_clip_img = clip_image_embeddings(model_clip, clip_tr_ld)   # (Ntr, 512)
va_clip_img = clip_image_embeddings(model_clip, clip_va_ld)   # (Nva, 512)
te_clip_img = clip_image_embeddings(model_clip, clip_te_ld)   # (Nte, 512)

# опционально: текстовые эмбеддинги CLIP
tr_clip_txt = clip_text_embeddings(model_clip, clip_tr_ld)    # (Ntr, 512)
va_clip_txt = clip_text_embeddings(model_clip, clip_va_ld)    # (Nva, 512)
te_clip_txt = clip_text_embeddings(model_clip, clip_te_ld)    # (Nte, 512)






@torch.no_grad()
@torch.no_grad()
def timm_image_embeddings(model, loader, use_pre_logits=True):
    model.eval()
    feats = []
    for batch in tqdm(loader, desc="timm img emb", leave=False):
        x = batch["image"].to(device, non_blocking=True)

        f = model.forward_features(x)
        # Предпочитаемый путь для timm (работает и для ViT, и для CNN):
        try:
            f = model.forward_head(f, pre_logits=use_pre_logits)  # -> (B, D)
        except Exception:
            # Фоллбек на случай нестандартной головы/модели
            if isinstance(f, dict):
                f = f.get("x", f.get("pooled", next(iter(f.values()))))
            if isinstance(f, (list, tuple)):
                f = f[-1]
            if f.ndim == 4:          # (B, C, H, W)
                f = f.mean(dim=(2, 3))
            elif f.ndim == 3:        # (B, N, D) — ViT-токены
                # берём CLS-токен, если он первый; иначе усредним по токенам
                f = f[:, 0, :] if f.size(1) >= 1 else f.mean(dim=1)
            elif f.ndim == 2:        # (B, D)
                pass
            else:                    # на всякий
                f = f.view(f.size(0), -1)

        feats.append(f.float().cpu().numpy())

    return np.vstack(feats).astype(np.float32)


# лоадер для test (если ещё не сделали)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

tr_img_emb = timm_image_embeddings(model_for_image, train_loader)  # (Ntr, D_vit≈768)
va_img_emb = timm_image_embeddings(model_for_image, val_loader)    # (Nva, D_vit≈768)
te_img_emb = timm_image_embeddings(model_for_image, test_loader)   # (Nte, D_vit≈768)



TAB_COLS = [
    'Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
    'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health',
    'Quantity','Fee','VideoAmt','PhotoAmt','State'
]

Xtr_tab = tr_df[TAB_COLS].astype(np.float32).values
Xva_tab = va_df[TAB_COLS].astype(np.float32).values
Xte_tab = test_en[TAB_COLS].astype(np.float32).values

y_tr = tr_df["AdoptionSpeed"].astype(int).values
y_va = va_df["AdoptionSpeed"].astype(int).values



from catboost import CatBoostClassifier

cbc_tab = CatBoostClassifier(
    iterations=1000 if not TEST_MODE else 200,
    learning_rate=0.03,
    depth=6,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    verbose=100
)
cbc_tab.fit(Xtr_tab, y_tr, eval_set=(Xva_tab, y_va))
probs_va_tab = cbc_tab.predict_proba(Xva_tab)



#data for catboost



#model_catboost.fit()



#model_catboost.predict_proba()


# ТОЛЬКО текстовые эмбеддинги (BERT CLS)
cbc_text = CatBoostClassifier(
    iterations=600 if not TEST_MODE else 200,
    learning_rate=0.05, depth=6,
    loss_function='MultiClass',
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    verbose=100
)
cbc_text.fit(tr_text_emb, y_tr, eval_set=(va_text_emb, y_va))
probs_va_text = cbc_text.predict_proba(va_text_emb)

# ТОЛЬКО img эмбеддинги от CLIP
cbc_img_clip = CatBoostClassifier(
    iterations=600 if not TEST_MODE else 200,
    learning_rate=0.05, depth=6,
    loss_function='MultiClass',
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    verbose=100
)
cbc_img_clip.fit(tr_clip_img, y_tr, eval_set=(va_clip_img, y_va))
probs_va_img_clip = cbc_img_clip.predict_proba(va_clip_img)

# ТОЛЬКО img эмбеддинги от timm (ResNet50)
cbc_img_timm = CatBoostClassifier(
    iterations=600 if not TEST_MODE else 200,
    learning_rate=0.05, depth=6,
    loss_function='MultiClass',
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    verbose=100
)
cbc_img_timm.fit(tr_img_emb, y_tr, eval_set=(va_img_emb, y_va))
probs_va_img_timm = cbc_img_timm.predict_proba(va_img_emb)

# ФЬЮЖН: табличка + текст (BERT) + картинка (CLIP image)
Xtr_fused = np.hstack([Xtr_tab, tr_text_emb, tr_clip_img])   # (Ntr, d_tab + 768 + 512)
Xva_fused = np.hstack([Xva_tab, va_text_emb, va_clip_img])

cbc_fused = CatBoostClassifier(
    iterations=800 if not TEST_MODE else 200,
    learning_rate=0.05, depth=6,
    loss_function='MultiClass',
    task_type='GPU' if torch.cuda.is_available() else 'CPU',
    verbose=100
)
cbc_fused.fit(Xtr_fused, y_tr, eval_set=(Xva_fused, y_va))
probs_va_fused = cbc_fused.predict_proba(Xva_fused)






from sklearn.metrics import cohen_kappa_score, f1_score

def qwk_from_probs(y_true, *probs_list, weights=None):
    P = np.stack(probs_list, axis=-1)                # (N, 5, M)
    if weights is None:
        weights = np.ones(P.shape[-1], dtype=np.float32) / P.shape[-1]
    ens = (P * weights.reshape(1,1,-1)).sum(-1)      # (N,5)
    preds = ens.argmax(1)
    return cohen_kappa_score(y_true, preds, weights='quadratic'), preds

qwk_tab, _       = qwk_from_probs(y_va, probs_va_tab)
qwk_text, _      = qwk_from_probs(y_va, probs_va_text)
qwk_img_clip, _  = qwk_from_probs(y_va, probs_va_img_clip)
qwk_img_timm, _  = qwk_from_probs(y_va, probs_va_img_timm)
qwk_fused, _     = qwk_from_probs(y_va, probs_va_fused)

# простой ансамбль нескольких
w = np.array([0.2, 0.2, 0.2, 0.4])  # tab, text, img_clip, fused  (подбери по вал)
qwk_ens, preds_ens = qwk_from_probs(y_va, probs_va_tab, probs_va_text, probs_va_img_clip, probs_va_fused, weights=w)

print({
    "qwk_tab": qwk_tab, "qwk_text": qwk_text,
    "qwk_img_clip": qwk_img_clip, "qwk_img_timm": qwk_img_timm,
    "qwk_fused": qwk_fused, "qwk_ens": qwk_ens
})






#model_catboost_text_img=CatBoostClassifier(
    
#)





#need inference notebook
# proba на тесте от базовых моделей
probs_te_tab     = cbc_tab.predict_proba(Xte_tab)
probs_te_text    = cbc_text.predict_proba(test_text_emb)
probs_te_imgclip = cbc_img_clip.predict_proba(te_clip_img)

# фьюжн на тесте
Xte_fused = np.hstack([Xte_tab, test_text_emb, te_clip_img])
probs_te_fused = cbc_fused.predict_proba(Xte_fused)

# ансамбль теми же весами w
Pte = np.stack([probs_te_tab, probs_te_text, probs_te_imgclip, probs_te_fused], axis=-1)
probs_te_ens = (Pte * w.reshape(1,1,-1)).sum(-1)
preds_test = probs_te_ens.argmax(1)

submission = sample.copy()
submission["AdoptionSpeed"] = preds_test
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")






# ==== SAVE ALL ARTIFACTS ====
import os, json, torch, shutil
from pathlib import Path

ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

# 1) DeBERTa classifier + tokenizer (лучший уже загружен в model_for_text благодаря Trainer(..., load_best_model_at_end=True))
TXT_DIR = ART_DIR / "deberta_cls"
TXT_DIR.mkdir(exist_ok=True)
model_for_text.save_pretrained(str(TXT_DIR))
tokenizer.save_pretrained(str(TXT_DIR))

# 2) DeBERTa text encoder (тот, что ты использовал для эмбеддингов)
TXT_ENC_DIR = ART_DIR / "deberta_text_encoder"
TXT_ENC_DIR.mkdir(exist_ok=True)
text_encoder.save_pretrained(str(TXT_ENC_DIR))

# 3) ViT classifier
IMG_DIR = ART_DIR / "vit_cls"
IMG_DIR.mkdir(exist_ok=True)

# текущее состояние (на случай если best_vit_qwk.pt не создан или хочешь еще и «текущее»)
torch.save(model_for_image.state_dict(), IMG_DIR / "vit_classifier_state.pth")

# если в тренинге сохранялся лучший чекпоинт — скопируем его вместе
best_ckpt_path = Path("best_vit_qwk.pt")
if best_ckpt_path.exists():
    shutil.copy2(best_ckpt_path, IMG_DIR / "best_vit_qwk.pt")

# 4) CatBoost models
cbc_tab.save_model(str(ART_DIR / "cbc_tab.cbm"))
cbc_text.save_model(str(ART_DIR / "cbc_text.cbm"))
cbc_img_clip.save_model(str(ART_DIR / "cbc_img_clip.cbm"))
cbc_img_timm.save_model(str(ART_DIR / "cbc_img_timm.cbm"))
cbc_fused.save_model(str(ART_DIR / "cbc_fused.cbm"))

# 5) Метаданные: чтобы восстановить pipeline без боли
meta = {
    "TEXT_BACKBONE": TEXT_BACKBONE,         # "microsoft/deberta-v3-base"
    "IMG_BACKBONE":  IMG_BACKBONE,          # "vit_base_patch16_224"
    "CLIP_BACKBONE": "openai/clip-vit-base-patch32",
    "num_classes": 5,
    "TAB_COLS": TAB_COLS,
    "ensemble_weights": list(map(float, w)),   # [0.2, 0.2, 0.2, 0.4]
    "image_norm_mean": [0.485, 0.456, 0.406],
    "image_norm_std":  [0.229, 0.224, 0.225],
    "img_size": 224,
    "seed": seed,
}
with open(ART_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("✅ All artifacts saved to:", ART_DIR.resolve())





