!pip install git+https://github.com/openai/CLIP.git
!pip install transformers huggingface-hub ftfy regex tqdm


import os
import json
import base64
import io
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import clip
from torch.nn import functional as F
from torchvision import transforms
# è¨­å®šè£�ç½®
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


DS_PATH = "/kaggle/input/wikipedia-train-0"  # è«‹ç¢ºèª�é€™è·¯å¾‘æ˜¯å°�çš„
filenames = sorted(os.listdir(DS_PATH))
json_content = []

print(f"ğŸš€ æ­£åœ¨è®€å�– JSON è³‡æ–™é›†: {DS_PATH}...")
for file in tqdm(filenames, desc="Loading Files"):
    if not file.endswith('.json'): continue
    filename = os.path.join(DS_PATH, file)
    with open(filename, "rb") as fr:
        for line in fr:
            if line:
                obj = json.loads(line)
                # ç°¡å–®æª¢æŸ¥æ¬„ä½�
                if "b64_bytes" in obj and "wit_features" in obj:
                    # æ��å�–æ‰€æœ‰å�¯ç”¨çš„æ��è¿°
                    descriptions = []
                    for element in obj["wit_features"]:
                        desc = element.get("caption_title_and_reference_description")
                        if desc:
                            descriptions.append(desc)
                    
                    if descriptions and obj["b64_bytes"]:
                        # ç‚ºäº†ç°¡åŒ–ï¼Œæˆ‘å€‘é€™è£¡å�ªæ‹¿ç¬¬ä¸€å€‹æ��è¿°ç•¶ä½œæ­£æ¨£æœ¬
                        # (é€²éš�ç‰ˆå�¯ä»¥æŠŠæ‰€æœ‰æ��è¿°éƒ½æ‹†å‡ºä¾†ç•¶å¤šç­†è³‡æ–™)
                        json_content.append({
                            "b64_bytes": obj["b64_bytes"],
                            "caption": descriptions[0],
                            "url": obj.get("image_url", "")
                        })

print(f"âœ… è³‡æ–™è®€å�–å®Œæˆ�ï¼�å…±æœ‰ {len(json_content)} ç­†åœ–æ–‡è³‡æ–™ã€‚")


# ==========================================
# 1. å®šç¾©è¼”åŠ©é¡�åˆ¥ (é›¶ä»¶)
# ==========================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, max_violation=True): super().__init__()
    def forward(self, x, y): return 0

class TextExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_model = config['text-model']['model-name']
        self.finetune = config['text-model']['finetune']
        self.text_model = AutoModel.from_pretrained(text_model)
    def forward(self, ids, mask):
        with torch.set_grad_enabled(self.finetune):
            out = self.text_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        out = torch.stack(out.hidden_states, dim=0)
        return out

class ImageExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.finetune = config['image-model']['finetune']
        model_name = config['image-model']['model-name']
        # å¼·åˆ¶è½‰ FP32 é�¿å…� bug
        self.clip_model, _ = clip.load(model_name, device='cpu')
        self.clip_model = self.clip_model.float()
    def forward(self, img):
        with torch.set_grad_enabled(self.finetune):
            feats = self.clip_model.encode_image(img)
        return feats

class TransformerPooling(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, num_layers=2):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=input_dim, dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    def forward(self, input, mask):
        mask_bool = ~mask.bool()
        input = input.permute(1, 0, 2)
        output = self.transformer_encoder(input, src_key_padding_mask=mask_bool)
        output = output[0]
        if self.proj: output = self.proj(output)
        return output

class DepthAggregatorModel(nn.Module):
    def __init__(self, aggr, input_dim=1024, output_dim=1024):
        super().__init__()
        self.aggr = aggr
        if self.aggr == 'gated':
            self.self_attn = nn.MultiheadAttention(input_dim, num_heads=4, dropout=0.1)
            self.gate_ffn = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    def forward(self, x, mask):
        if self.aggr is None: out = x[-1, :, 0, :]
        elif self.aggr == 'mean': out = x[:, :, 0, :].mean(dim=0)
        if self.proj: out = self.proj(out)
        return out

class FeatureFusionModel(nn.Module):
    def __init__(self, mode, img_feat_dim, txt_feat_dim, common_space_dim):
        super().__init__()
        self.mode = mode
        if mode == 'weighted':
            self.alphas = nn.Sequential(
                nn.Linear(img_feat_dim + txt_feat_dim, 512), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(512, 2))
            self.img_proj = nn.Linear(img_feat_dim, common_space_dim)
            self.txt_proj = nn.Linear(txt_feat_dim, common_space_dim)
            self.post_process = nn.Sequential(
                nn.Linear(common_space_dim, common_space_dim), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(common_space_dim, common_space_dim)
            )
    def forward(self, img_feat, txt_feat):
        concat_feat = torch.cat([img_feat, txt_feat], dim=1)
        alphas = torch.sigmoid(self.alphas(concat_feat))
        img_feat_norm = F.normalize(self.img_proj(img_feat), p=2, dim=1)
        txt_feat_norm = F.normalize(self.txt_proj(txt_feat), p=2, dim=1)
        out_feat = img_feat_norm * alphas[:, 0].unsqueeze(1) + txt_feat_norm * alphas[:, 1].unsqueeze(1)
        out_feat = self.post_process(out_feat)
        return out_feat, alphas

# ==========================================
# 2. æ ¸å¿ƒæ¨¡å�‹ MatchingModel (å·²æ”¹è£� Inference æ�¥å�£)
# ==========================================
class MatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        common_space_dim = config['matching']['common-space-dim']
        num_text_transformer_layers = config['matching']['text-transformer-layers']
        img_feat_dim = config['image-model']['dim']
        txt_feat_dim = config['text-model']['dim']
        image_disabled = config['image-model'].get('disabled', False)
        
        self.aggregate_tokens_depth = config['matching'].get('aggregate-tokens-depth', None)
        self.fusion_mode = config['matching'].get('fusion-mode', 'concat')
        self.image_disabled = image_disabled

        self.txt_model = TextExtractorModel(config)
        
        if not image_disabled:
            self.img_model = ImageExtractorModel(config)
            self.image_fc = nn.Sequential(
                nn.Linear(img_feat_dim, img_feat_dim), nn.Dropout(0.2), nn.ReLU(), nn.Linear(img_feat_dim, img_feat_dim)
            )
            if self.fusion_mode == 'concat':
                self.process_after_concat = nn.Sequential(
                    nn.Linear(img_feat_dim + txt_feat_dim, common_space_dim),
                    nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(common_space_dim, common_space_dim)
                )
            else:
                self.process_after_concat = FeatureFusionModel(self.fusion_mode, img_feat_dim, txt_feat_dim, common_space_dim)

        self.caption_process = TransformerPooling(txt_feat_dim, common_space_dim, num_text_transformer_layers)
        self.url_process = TransformerPooling(txt_feat_dim, txt_feat_dim if not image_disabled else common_space_dim, num_text_transformer_layers)
        if self.aggregate_tokens_depth:
            self.token_aggregator = DepthAggregatorModel(self.aggregate_tokens_depth, txt_feat_dim, common_space_dim)
        self.matching_loss = ContrastiveLoss()

    # --- é€™æ˜¯ Inference å¿…å‚™çš„æ�¥å�£ (æˆ‘å¹«ä½ å¾� compute_embeddings æ‹†å‡ºä¾†çš„) ---
    def encode_query(self, img, url, url_mask):
        # 1. è¨ˆç®— URL æ–‡å­—ç‰¹å¾µ (Test æ™‚æ˜¯ Dummy)
        url_feats = self.txt_model(url, url_mask)
        url_feats_plus = self.url_process(url_feats[-1], url_mask)
        if self.aggregate_tokens_depth:
            url_feats = url_feats_plus + self.token_aggregator(url_feats, url_mask)
        else:
            url_feats = url_feats_plus

        # 2. è¨ˆç®—åœ–ç‰‡ç‰¹å¾µä¸¦è��å�ˆ
        if not self.image_disabled:
            img_feats = self.image_fc(self.img_model(img).float())
            if self.fusion_mode == 'concat':
                query_feats = torch.cat([img_feats, url_feats], dim=1)
                query_feats = self.process_after_concat(query_feats)
            else:
                query_feats, _ = self.process_after_concat(img_feats, url_feats)
        else:
            query_feats = url_feats
        
        return F.normalize(query_feats, p=2, dim=1)

    def encode_caption(self, caption, caption_mask):
        caption_feats = self.txt_model(caption, caption_mask)
        caption_feats_plus = self.caption_process(caption_feats[-1], caption_mask)
        if self.aggregate_tokens_depth:
            caption_feats = caption_feats_plus + self.token_aggregator(caption_feats, caption_mask)
        else:
            caption_feats = caption_feats_plus
        return F.normalize(caption_feats, p=2, dim=1)

# ==========================================
# 3. å°�æ‡‰çš„ Config (å¿…é ˆè·Ÿ A çš„è¨­å®šä¸€æ¨£)
# ==========================================
Config = {
    'text-model': {'model-name': 'xlm-roberta-base', 'dim': 768, 'finetune': False},
    'image-model': {'model-name': 'ViT-B/32', 'dim': 512, 'finetune': False, 'disabled': False},
    'matching': {
        'common-space-dim': 768, 
        'text-transformer-layers': 2, 
        'fusion-mode': 'concat', 
        'aggregate-tokens-depth': 'mean' # é€™æ˜¯é—œé�µï¼�èˆŠçš„ Config å�¯èƒ½æ²’æœ‰é€™å€‹
    },
    'training': {'margin': 0.2, 'max-violation': False}
}
config = Config


# å®šç¾©æ¨™æº–çš„ CLIP Normalization æ•¸å€¼ (é€™æ˜¯å›ºå®šçš„ï¼Œä¸�èƒ½æ”¹)
normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                 std=(0.26862954, 0.26130258, 0.27577711))

TOKENIZER = AutoTokenizer.from_pretrained('xlm-roberta-base')
MAX_LEN = 128

class JsonMiningDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. åœ–ç‰‡è™•ç�† (Base64 è§£ç¢¼ + æ¨™æº–åŒ–)
        try:
            decoded = base64.b64decode(item['b64_bytes'])
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
            image = image.resize((224, 224))
            
            # è½‰ Tensor ä¸¦é™¤ä»¥ 255
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            # âš ï¸�ã€�é—œé�µä¿®æ”¹ã€‘åŠ ä¸Šé€™è¡Œæ¨™æº–åŒ–ï¼�
            image = normalize(image) 
        except:
            image = torch.zeros(3, 224, 224)
            
        # 2. æ–‡å­—è™•ç�†
        caption = str(item['caption'])
        inputs = self.tokenizer.encode_plus(
            caption, None, add_special_tokens=True,
            max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'caption_text': caption, 
            'index': idx 
        }


# A. åˆ�å§‹åŒ–æ¨¡å�‹
# (æ³¨æ„�ï¼šé€™è£¡çš„ config ä¾†è‡ª Cell 4 çš„å®šç¾©ï¼Œè«‹ç¢ºä¿� Cell 4 å·²ç¶“åŸ·è¡Œé��)
coarse_model = MatchingModel(config)

# B. è¼‰å…¥ A çš„æ¬Šé‡�
# âš ï¸� è«‹ç¢ºèª�é€™å€‹è·¯å¾‘æ˜¯ä½ ä¸Šå‚³çš„é‚£å€‹ .bin æª”æ¡ˆ
WEIGHTS_PATH = "/kaggle/input/mining/Loss_2.5559_epoch_9.bin" 

try:
    # é€™è£¡çµ•å°�ä¸�èƒ½åŠ  strict=Falseï¼Œæˆ‘å€‘è¦�ç¢ºä¿�å®ƒçœŸçš„å®Œç¾�è¼‰å…¥ï¼�
    coarse_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print("âœ… æ�­å–œï¼�æ¨¡å�‹æ�¶æ§‹çµ‚æ–¼å°�ä¸Šäº†ï¼�æ¬Šé‡�å®Œç¾�è¼‰å…¥ï¼�")
    coarse_model.to(device).eval()
except Exception as e:
    print("â�Œ é‚„æ˜¯æœ‰éŒ¯... è«‹æŠŠä¸‹é�¢çš„éŒ¯èª¤è¨Šæ�¯çµ¦æˆ‘ï¼š")
    print(e)


# A. è¼‰å…¥æ¨¡å�‹

model = MatchingModel(config)

WEIGHTS_PATH = "/kaggle/input/mining/Loss_2.5559_epoch_9.bin" # è«‹æ”¹è·¯å¾‘

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

model.to(device)

model.eval()


# B. å»ºç«‹ DataLoader

mining_dataset = JsonMiningDataset(json_content, TOKENIZER, MAX_LEN) 
mining_loader = DataLoader(mining_dataset, batch_size=64, shuffle=False)

all_img_embs = []
all_txt_embs = []
all_captions = []


# C. è¨ˆç®—å�‘é‡�
with torch.no_grad():
    for data in tqdm(mining_loader, desc="Encoding"):
        images = data['image'].to(device)
        ids = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        
        q_feat = model.encode_query(images, ids, mask)
        c_feat = model.encode_caption(ids, mask)
        
        all_img_embs.append(q_feat.cpu())
        all_txt_embs.append(c_feat.cpu())
        all_captions.extend(data['caption_text']) # æŠŠæ–‡å­—å­˜èµ·ä¾†

all_img_embs = torch.cat(all_img_embs, dim=0).to(device)
all_txt_embs = torch.cat(all_txt_embs, dim=0).to(device)


# D. æ‰¾éŒ¯é¡Œ
hard_negatives_data = []
batch_size = 1000

for i in tqdm(range(0, len(all_img_embs), batch_size), desc="Mining"):
    end = min(i + batch_size, len(all_img_embs))
    batch_img = all_img_embs[i:end]
    
    # ç®—ç›¸ä¼¼åº¦
    sim_matrix = torch.matmul(batch_img, all_txt_embs.T)
    
    # å�– Top-10
    vals, indices = torch.topk(sim_matrix, k=10, dim=1)
    indices = indices.cpu().numpy()
    
    for idx_in_batch, candidates in enumerate(indices):
        real_idx = i + idx_in_batch
        positive_caption = all_captions[real_idx]
        
        neg_list = []
        for cand_id in candidates:
            if cand_id != real_idx: # ä¸�æ˜¯æ­£ç¢ºç­”æ¡ˆ
                neg_caption = all_captions[cand_id]
                
                # å­˜èµ·ä¾†
                hard_negatives_data.append({
                    'image_id': real_idx, # é€™è£¡å­˜ index æ–¹ä¾¿å°�æ‡‰ï¼Œé€²éš�å�¯ä»¥å­˜ base64
                    'positive': positive_caption,
                    'negative': neg_caption,
                    'rank': len(neg_list) + 1
                })
                neg_list.append(cand_id)
            if len(neg_list) >= 2: break # æ¯�å¼µåœ–æŒ– 2 å€‹éŒ¯é¡Œ


# E. å­˜æª”
mining_df = pd.DataFrame(hard_negatives_data)
mining_df.to_csv("train_hard_negatives.csv", index=False)
print("æŒ–ç¤¦å®Œæˆ�ï¼�å·²ç”¢å‡º train_hard_negatives.csv")
print(mining_df.head())


import gc
# åˆªé™¤ä¸�ç”¨çš„è®Šæ•¸
del model, mining_loader, mining_dataset, all_img_embs, all_txt_embs
torch.cuda.empty_cache()
gc.collect()
print("â™»ï¸� è¨˜æ†¶é«”å·²é‡‹æ”¾ï¼Œæº–å‚™é–‹å§‹è¨“ç·´ç²¾æ�’æ¨¡å�‹...")


import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import XLMRobertaForSequenceClassification
from torch.optim import AdamW 
import os
# 1. å®šç¾© Dataset
class RerankDataset(Dataset):
    def __init__(self, mining_df, tokenizer, max_len):
        self.df = mining_df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df) * 2 # æ­£æ¨£æœ¬ + è² æ¨£æœ¬
    
    def __getitem__(self, idx):
        # å�¶æ•¸ index æ˜¯æ­£æ¨£æœ¬ (label=1)
        # å¥‡æ•¸ index æ˜¯è² æ¨£æœ¬ (label=0)
        row_idx = idx // 2
        row = self.df.iloc[row_idx]
        
        if idx % 2 == 0:
            text_a = row['positive'] 
            text_b = row['positive'] 
            label = 1.0
        else:
            text_a = row['positive']
            text_b = row['negative']
            label = 0.0
            
        # Cross-Encoder çš„è¼¸å…¥æ˜¯æŠŠå…©å�¥è©±æ�¥åœ¨ä¸€èµ·
        inputs = self.tokenizer.encode_plus(
            text_a, text_b, # å…©å�¥è©±
            add_special_tokens=True,
            max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# 2. æº–å‚™è³‡æ–™
mining_df = pd.read_csv("train_hard_negatives.csv")


rerank_dataset = RerankDataset(mining_df, TOKENIZER, max_len=128) 
rerank_loader = DataLoader(rerank_dataset, batch_size=16, shuffle=True)

# 3. å®šç¾©æ¨¡å�‹ (ä½¿ç”¨ XLM-R å�šäºŒå…ƒåˆ†é¡�)
print("ğŸš€ åˆ�å§‹åŒ– Cross-Encoder æ¨¡å�‹...")
rerank_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=1)
rerank_model.to(device)
rerank_model.train()

# è¨˜å¾— AdamW è¦�å¾� torch.optim åŒ¯å…¥ (å¦‚ä¸Šä¸€å‰‡å›�ç­”æ‰€è¿°)
from torch.optim import AdamW
optimizer = AdamW(rerank_model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()



epochs = 1

print(f"ğŸ”¥ é–‹å§‹è¨“ç·´ç²¾æ�’æ¨¡å�‹ (Epochs={epochs}, ä½¿ç”¨ AMP åŠ é€Ÿ)...")

# åˆ�å§‹åŒ– Scaler (åŠ é€Ÿç¥�å™¨)
scaler = GradScaler()

for epoch in range(epochs):
    total_loss = 0
    rerank_model.train() # ç¢ºä¿�åœ¨è¨“ç·´æ¨¡å¼�
    
    # é¡¯ç¤ºé€²åº¦æ¢�
    loop = tqdm(rerank_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch in loop:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # âš¡ é–‹å•Ÿæ··å�ˆç²¾åº¦è¨ˆç®— (Forward)
        with autocast():
            outputs = rerank_model(ids, attention_mask=mask)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, labels)
        
        # âš¡ ä½¿ç”¨ Scaler å��å�‘å‚³æ’­ (Backward)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # æ›´æ–°é€²åº¦æ¢�ä¸Šçš„ Loss
        loop.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / len(rerank_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")


torch.save(rerank_model.state_dict(), "reranker_model.bin")
print("æ¨¡å�‹å·²å­˜æª”ç‚º reranker_model.bin")





import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
import glob
import base64
import io
from PIL import Image
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = AutoTokenizer.from_pretrained('xlm-roberta-base')
MAX_LEN = 64

# è·¯å¾‘è¨­å®š
TEST_TSV = '/kaggle/input/c/wikipedia-image-caption/test.tsv'
CAPTION_CSV = '/kaggle/input/c/wikipedia-image-caption/test_caption_list.csv'
PIXEL_DIR = '/kaggle/input/c/wikipedia-image-caption/image_data_test/image_pixels'

# ç²—æ�’æ¨¡å�‹æ¬Šé‡� (ç”¨ä¾†æ’ˆå€™é�¸äºº)
COARSE_WEIGHTS = "/kaggle/input/mining/Loss_2.5559_epoch_9.bin" 
# ç²¾æ�’æ¨¡å�‹æ¬Šé‡� (å‰›è¨“ç·´å¥½çš„)
RERANK_WEIGHTS = "reranker_model.bin" 

print(f"Device: {DEVICE}")


# 2. å®šç¾©å…©å€‹æ¨¡å�‹ (ç²—æ�’ & ç²¾æ�’)
# ==========================================
# --- ç²—æ�’æ¨¡å�‹  ---
# (è«‹ç¢ºä¿�å‰�é�¢å·²ç¶“å®šç¾©é�� MatchingModel, TextExtractorModel ç­‰é¡�åˆ¥)

coarse_model = MatchingModel(config) 
coarse_model.load_state_dict(torch.load(COARSE_WEIGHTS, map_location=DEVICE)
                            )
coarse_model.to(DEVICE).eval()
print("ç²—æ�’æ¨¡å�‹è¼‰å…¥å®Œæˆ�")

# --- ç²¾æ�’æ¨¡å�‹ (ä½ çš„æ�¶æ§‹) ---
rerank_model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=1)
rerank_model.load_state_dict(torch.load(RERANK_WEIGHTS, map_location=DEVICE))
rerank_model.to(DEVICE).eval()
print("ç²¾æ�’æ¨¡å�‹è¼‰å…¥å®Œæˆ�")

# ==========================================
# 3. æº–å‚™æ¸¬è©¦è³‡æ–™
# ==========================================
# è®€å�–æ¸¬è©¦åœ–ç‰‡ (Base64)
print("æ­£åœ¨è¼‰å…¥æ¸¬è©¦åœ–ç‰‡åº«...")
image_map = {}
pixel_files = sorted(glob.glob(f"{PIXEL_DIR}/*.csv"))
for f in tqdm(pixel_files):
    temp = pd.read_csv(f, sep='\t', names=['url', 'b64'], usecols=[0,1])
    for _, row in temp.iterrows():
        image_map[row['url']] = row['b64']

# è®€å�–å€™é�¸æ¨™é¡Œ
captions_df = pd.read_csv(CAPTION_CSV)
all_captions = captions_df['caption_title_and_reference_description'].tolist()


normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                 std=(0.26862954, 0.26130258, 0.27577711))

class TestImageDataset(Dataset):
    def __init__(self, tsv_path, img_map, tokenizer):
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.img_map = img_map
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row['image_url']
        b64 = self.img_map.get(url, "")
        
        # 1. åœ–ç‰‡è™•ç�†
        try:
            image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            image = image.resize((224, 224))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = normalize(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        # 2. é—œé�µä¿®æ­£ï¼šå¾� URL æ��å�–æª”å��ä½œç‚ºæ–‡å­—è¼¸å…¥
        # å�Ÿæœ¬æ˜¯ dummy_input = "" 
        try:
            # å�–ç¶²å�€æœ€å¾Œä¸€æ®µ -> è§£ç¢¼ (%20è®Šç©ºç™½) -> å�»æ�‰å‰¯æª”å�� -> åº•ç·šè®Šç©ºç™½
            filename = url.split('/')[-1] 
            filename = urllib.parse.unquote(filename)
            filename = filename.replace('_', ' ').replace('.jpg', '').replace('.png', '')
            text_input = filename
        except:
            text_input = ""
            
        # 3. ç·¨ç¢¼æ–‡å­—
        tokenized_input = self.tokenizer.encode_plus(
            text_input, 
            max_length=64, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'image': image, 
            'input_ids': tokenized_input['input_ids'].flatten(),
            'attention_mask': tokenized_input['attention_mask'].flatten(),
            'id': row['id']
        }

class CapDataset(Dataset):
    def __init__(self, caps, tokenizer):
        self.caps = caps
        self.tokenizer = tokenizer
    def __len__(self): return len(self.caps)
    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(str(self.caps[idx]), return_tensors='pt', max_length=64, padding='max_length', truncation=True)
        return inputs['input_ids'].flatten(), inputs['attention_mask'].flatten()



# 2. è¨ˆç®—å�‘é‡� (Encoding)
# ==========================================
print("[Step 1] æº–å‚™æ¸¬è©¦è³‡æ–™èˆ‡è¨ˆç®—å�‘é‡�...")

# --- 2.1 è¨ˆç®—åœ–ç‰‡å�‘é‡� ---
test_ds = TestImageDataset(TEST_TSV, image_map, TOKENIZER)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

img_embs = []
test_ids = []

with torch.no_grad():
    for data in tqdm(test_loader, desc="Encoding Images"):
        img = data['image'].to(DEVICE)
        ids = data['input_ids'].to(DEVICE)
        mask = data['attention_mask'].to(DEVICE)
        
        feat = coarse_model.encode_query(img, ids, mask)
        img_embs.append(feat.cpu())
        test_ids.extend(data['id'].numpy())

img_embs = torch.cat(img_embs, dim=0)
print(f"åœ–ç‰‡å�‘é‡�è¨ˆç®—å®Œæˆ� å½¢ç‹€: {img_embs.shape}")

# --- 2.2 è¨ˆç®—æ–‡å­—å�‘é‡� ---

cap_loader = DataLoader(CapDataset(all_captions, TOKENIZER), batch_size=256, shuffle=False, num_workers=0)

txt_embs = []
with torch.no_grad():
    for ids, mask in tqdm(cap_loader, desc="Encoding Captions"):
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        feat = coarse_model.encode_caption(ids, mask)
        txt_embs.append(feat.cpu())

txt_embs = torch.cat(txt_embs, dim=0)
print(f"æ–‡å­—å�‘é‡�è¨ˆç®—å®Œæˆ� å½¢ç‹€: {txt_embs.shape}")


# ==========================================
# 3. ç²—æ�’æµ·é�¸ (Coarse Retrieval) - ç®—å‡º Top-50
# ==========================================
print("[Step 2] ç²—æ�’...")

# ç¢ºä¿�åœ¨ CPU (é�¿å…� GPU OOM)
if img_embs.is_cuda: img_embs = img_embs.cpu()
if txt_embs.is_cuda: txt_embs = txt_embs.cpu()

# æ–‡å­—å�‘é‡�æ�¬åˆ° GPU
txt_embs_gpu = txt_embs.to(DEVICE)

top100_indices_list = []
BATCH_SIZE = 1000 

for i in tqdm(range(0, len(img_embs), BATCH_SIZE), desc="Coarse Retrieval"):
    batch_img = img_embs[i : i + BATCH_SIZE].to(DEVICE)
    batch_sims = torch.matmul(batch_img, txt_embs_gpu.T)
    _, batch_topk = torch.topk(batch_sims, k=100 , dim=1)
    top100_indices_list.append(batch_topk.cpu())
    del batch_img, batch_sims, batch_topk

top100_indices = torch.cat(top100_indices_list, dim=0)
print(f"ç²—æ�’å®Œæˆ�{top100_indices.shape}")


# éš¨æ©ŸæŠ½æŸ¥ä¸€å¼µæ¸¬è©¦åœ–ç‰‡ï¼Œçœ‹çœ‹æ˜¯ä¸�æ˜¯é»‘ç•«é�¢
import matplotlib.pyplot as plt

# æ‹¿ Test Dataset çš„ç¬¬ 0 ç­†è³‡æ–™
dataset_check = TestImageDataset(TEST_TSV, image_map, TOKENIZER)
data = dataset_check[0]

img_tensor = data['image']
print(f"åœ–ç‰‡ Tensor å½¢ç‹€: {img_tensor.shape}")
print(f"åœ–ç‰‡æ•¸å€¼ç¯„åœ�: Min={img_tensor.min():.4f}, Max={img_tensor.max():.4f}, Mean={img_tensor.mean():.4f}")

# å¦‚æ�œ Min=0, Max=0ï¼Œä»£è¡¨ä½ è®€åˆ°é»‘ç•«é�¢äº† (åœ–ç‰‡è§£ç¢¼å¤±æ•—)
if img_tensor.max() == 0:
    print("â�Œ è­¦å‘Šï¼šé€™å¼µåœ–æ˜¯å…¨é»‘çš„ï¼�ä½ çš„ image_map æˆ– base64 è§£ç¢¼æœ‰å•�é¡Œï¼�")
else:
    print("âœ… åœ–ç‰‡æ•¸å€¼æ­£å¸¸ (ä¸�æ˜¯å…¨é»‘)ã€‚")

# å˜—è©¦æŠŠ Tensor è½‰å›�åœ–ç‰‡é¡¯ç¤º (å› ç‚ºæœ‰ Normalizeï¼Œé¡�è‰²æœƒæ€ªæ€ªçš„ï¼Œä½†è¦�æœ‰æ�±è¥¿)
# å��æ¨™æº–åŒ–
inv_normalize = transforms.Normalize(
    mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
    std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
)
img_display = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
img_display = np.clip(img_display, 0, 1)

plt.imshow(img_display)
plt.title(f"Check ID: {data['id']}")
plt.show()


# éš¨æ©ŸæŒ‘ 5 å¼µåœ–ï¼Œå�°å‡ºå®ƒå€‘ç²—æ�’ç¬¬ä¸€å��çš„æ¨™é¡Œ
print("ğŸ”� æª¢æŸ¥ç²—æ�’å“�è³ª (Top-1 é �æ¸¬):")
for i in range(5):
    idx = top100_indices[i, 0] # å�–ç¬¬ i å¼µåœ–çš„ç¬¬ 1 å��å€™é�¸äººç´¢å¼•
    print(f"åœ– {i} çš„ç²—æ�’ç¬¬ä¸€å��: {all_captions[idx]}")


# ==========================================
# 4. ç²¾æ�’æ±ºé�¸ (é«˜é€ŸåŠ é€Ÿç‰ˆ + Top-100 é�©é…�)
# ==========================================
from torch.cuda.amp import autocast # åŒ¯å…¥åŠ é€Ÿå·¥å…·
from torch.utils.data import TensorDataset

print("[Step 3] ç²¾æ�’ï¼šé–‹å§‹é‡�æ–°æ‰“åˆ†...")

final_submission = []
ALPHA = 0.5

# å¢�åŠ  Batch Size åˆ° 256 (æ ¹æ“šé¡¯å­˜æƒ…æ³�èª¿æ•´ï¼ŒT4 è·‘ç´”æ–‡å­—é€šå¸¸ 256 æ²’å•�é¡Œ)
# å¦‚æ�œå ±éŒ¯ OOMï¼Œè«‹æ”¹å›� 128 æˆ– 64
RERANK_BATCH_SIZE = 256 

for i, candidates in enumerate(tqdm(top100_indices.cpu().numpy(), desc="Reranking")):
    img_id = test_ids[i]
    
    # 1. æº–å‚™ Query å’Œ Candidates
    pseudo_query = all_captions[candidates[0]] 
    candidate_texts = [all_captions[idx] for idx in candidates]
    
    # 2. æº–å‚™ Batch è³‡æ–™
    pairs = [[pseudo_query, cand] for cand in candidate_texts]
    encoded = TOKENIZER(pairs, padding=True, truncation=True, max_length=64, return_tensors="pt")
    
    # å»ºç«‹ä¸€å€‹è‡¨æ™‚çš„ DataLoader ä¾†å�š Batch æ�¨è«– (é�¿å…�ä¸€æ¬¡å¡� 100 å€‹çˆ†æ�‰ï¼Œé›–ç„¶ 100 å€‹é€šå¸¸é‚„å¥½)
    # ä½†ç‚ºäº†ç©©å�¥ï¼Œæˆ‘å€‘é‚„æ˜¯ä¹–ä¹–åˆ‡ Batch
    batch_input_ids = encoded['input_ids']
    batch_attention_mask = encoded['attention_mask']
    
    # å»ºç«‹å°�å�‹çš„ Dataset
    mini_dataset = TensorDataset(batch_input_ids, batch_attention_mask)
    mini_loader = DataLoader(mini_dataset, batch_size=RERANK_BATCH_SIZE, shuffle=False)
    
    all_scores = []
    
    # 3. åŸ·è¡Œæ�¨è«–
    with torch.no_grad():
        with autocast(): # âš¡ é–‹å•Ÿæ··å�ˆç²¾åº¦åŠ é€Ÿ (é—œé�µ!)
            for b_ids, b_mask in mini_loader:
                b_ids = b_ids.to(DEVICE)
                b_mask = b_mask.to(DEVICE)
                
                outputs = rerank_model(input_ids=b_ids, attention_mask=b_mask)
                scores = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                
                # è™•ç�† scores å�¯èƒ½æ˜¯ç´”é‡�(Scalar)çš„æƒ…æ³� (ç•¶ batch=1)
                if scores.ndim == 0: scores = [scores]
                all_scores.extend(scores)
    
    rerank_scores = np.array(all_scores)
    
    # 4. å�–å¾—ç²—æ�’åˆ†æ•¸
    coarse_scores = np.linspace(1.0, 0.0, len(candidates))
    
    # 5. åˆ†æ•¸è��å�ˆ
    final_scores = (ALPHA * coarse_scores) + ((1 - ALPHA) * rerank_scores)
    
    # 6. é‡�æ–°æ�’åº�
    reranked_order = np.argsort(final_scores)[::-1]
    
    # å�–å‰� 5 å��
    top5_local_indices = reranked_order[:5]
    top5_global_indices = [candidates[k] for k in top5_local_indices]
    
    for rank, idx in enumerate(top5_global_indices):
        final_submission.append({
            "id": img_id,
            "caption_title_and_reference_description": all_captions[idx]
        })




# ==========================================
# 5. å­˜æª”
# ==========================================
pd.DataFrame(final_submission).to_csv("submission.csv", index=False)
print("ğŸ�‰ submission.csv å·²ç”Ÿæˆ�")

