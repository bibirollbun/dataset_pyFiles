import torch
import torch.nn as nn

class CharLSTMTranslator(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_size=64, hidden_size=128, num_layers=1, max_len=512):
        super().__init__()
        self.src_embedding = nn.Embedding(input_vocab_size, emb_size, padding_idx=0)
        self.tgt_embedding = nn.Embedding(output_vocab_size, emb_size, padding_idx=0)

        self.pos_embedding = nn.Embedding(max_len, emb_size)

        self.encoder = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, src, tgt):
        batch_size, seq_len = src.size()

        num_pos = self.pos_embedding.num_embeddings
        pos_idx = torch.arange(seq_len, device=src.device) % num_pos
        pos_idx = pos_idx.unsqueeze(0).repeat(batch_size, 1)
        pos_embedded = self.pos_embedding(pos_idx)
        embedded_src = self.src_embedding(src) + pos_embedded

        _, (hidden, cell) = self.encoder(embedded_src)

        embedded_tgt = self.tgt_embedding(tgt)
        outputs, _ = self.decoder(embedded_tgt, (hidden, cell))

        logits = self.fc(outputs)
        return logits


import pandas as pd
df = pd.read_csv("/kaggle/input/05-denoised-test-part1/05_denoised_test_part1.csv").to_numpy()


def load_model_from_matrix(model, weights_matrix, original_len):
    weights_matrix = torch.tensor(weights_matrix)
    flat_weights = weights_matrix.reshape(-1)[:original_len]

    offset = 0
    for p in model.parameters():
        numel = p.numel()
        new_data = flat_weights[offset : offset + numel].view_as(p)
        p.data.copy_(new_data)
        offset += numel

    print("Model weights successfully restored!")
    return model

model = CharLSTMTranslator(input_vocab_size=73, output_vocab_size=96)
model = load_model_from_matrix(model, df, 254624)


import pandas as pd
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load training data (fixed path)
df = pd.read_csv("/kaggle/input/afml-assignment-1-ec-campus/Team 5/train_part2.csv")
encoded_texts = df['encoded_text'].tolist()
english_texts = df['text'].tolist()

# Character sets
all_encoded_chars = set(''.join(encoded_texts))
all_english_chars = set(''.join(english_texts))

# Special tokens
SPECIAL_TOKENS = {
    'PAD': 0,
    'SOS': 1,
    'EOS': 2,
    'UNK': 3,
}

# Deterministic ordering for reproducibility
encoded_chars_sorted = sorted(list(all_encoded_chars))
english_chars_sorted = sorted(list(all_english_chars))

# Build vocabs with space for specials at front
encoded_vocab = {c: i + len(SPECIAL_TOKENS) for i, c in enumerate(encoded_chars_sorted)}
english_vocab = {c: i + len(SPECIAL_TOKENS) for i, c in enumerate(english_chars_sorted)}

# Add specials
for tok, idx in SPECIAL_TOKENS.items():
    encoded_vocab[f'<{tok}>'] = idx
    english_vocab[f'<{tok}>'] = idx

# Reverse mapping
rev_english_vocab = {idx: ch for ch, idx in english_vocab.items()}

# Helpers
def text_to_seq_single(text, vocab):
    return [vocab.get(ch, SPECIAL_TOKENS['UNK']) for ch in text]

def with_sos_eos(seq):
    return [SPECIAL_TOKENS['SOS']] + seq + [SPECIAL_TOKENS['EOS']]

# Encode sequences
encoded_seqs = [text_to_seq_single(t, encoded_vocab) for t in encoded_texts]
english_seqs = [text_to_seq_single(t, english_vocab) for t in english_texts]

# Target input/output for teacher forcing
tgt_input_seqs = [with_sos_eos(s)[:-1] for s in english_seqs]
tgt_output_seqs = [with_sos_eos(s)[1:] for s in english_seqs]

input_vocab_size = max(encoded_vocab.values()) + 1
output_vocab_size = max(english_vocab.values()) + 1
print('Input vocab size:', input_vocab_size)
print('Output vocab size:', output_vocab_size)


len(all_encoded_chars), len(all_english_chars)


# Dataset, Training, and Inference with Multi-GPU Support (Memory Optimized)
import math

# Clear any existing memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Check available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

print(f"{'='*60}")
print(f"GPU Configuration:")
print(f"{'='*60}")
print(f"Available GPUs: {n_gpus}")
if n_gpus > 0:
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
print(f"{'='*60}\n")

# Move model to device
model = model.to(device)

# Use conservative batch size due to long sequences
if n_gpus > 1:
    print(f"ğŸš€ Using DataParallel with {n_gpus} GPUs")
    model = nn.DataParallel(model)
    # Very conservative batch size due to long sequences
    batch_size = 8 * n_gpus  # 8 per GPU
    print(f"Batch size: {batch_size} (per-GPU: {batch_size // n_gpus})")
else:
    batch_size = 4
    print(f"Using single GPU with batch size: {batch_size}")

print()

# Dataset and DataLoader
class Seq2SeqDataset(Dataset):
    def __init__(self, src_seqs, tgt_in_seqs, tgt_out_seqs):
        self.src = src_seqs
        self.tgt_in = tgt_in_seqs
        self.tgt_out = tgt_out_seqs
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.long),
            torch.tensor(self.tgt_in[idx], dtype=torch.long),
            torch.tensor(self.tgt_out[idx], dtype=torch.long),
        )

PAD_IDX = SPECIAL_TOKENS['PAD']

def collate_fn(batch):
    src, tgt_in, tgt_out = zip(*batch)
    src_pad = pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt_in_pad = pad_sequence(tgt_in, batch_first=True, padding_value=PAD_IDX)
    tgt_out_pad = pad_sequence(tgt_out, batch_first=True, padding_value=PAD_IDX)
    return src_pad, tgt_in_pad, tgt_out_pad

# Create dataset and dataloader
train_ds = Seq2SeqDataset(encoded_seqs, tgt_input_seqs, tgt_output_seqs)
train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

print(f"Dataset size: {len(train_ds)} samples")
print(f"Number of batches: {len(train_loader)}\n")

# Training setup with gradient accumulation
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Gradient accumulation to simulate larger batch size
accumulation_steps = 3  # Effective batch size = batch_size * accumulation_steps

def train_epoch(epoch_num):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (src, tgt_in, tgt_out) in enumerate(train_loader):
        src = src.to(device, non_blocking=True)
        tgt_in = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)

        logits = model(src, tgt_in)
        B, T, V = logits.size()
        
        loss = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # CORRECTED: Track metrics properly
        with torch.no_grad():
            mask = (tgt_out != PAD_IDX)
            num_tokens = mask.sum().item()
            total_tokens += num_tokens
            # Use UNNORMALIZED loss for tracking
            unnormalized_loss = loss.item() * accumulation_steps
            total_loss += unnormalized_loss * num_tokens  # weight by actual tokens, not B*T
        
        if (batch_idx + 1) % 50 == 0:
            avg_loss_so_far = total_loss / max(1, total_tokens)
            ppl_so_far = math.exp(min(avg_loss_so_far, 20))  # More reasonable cap
            print(f"  Epoch {epoch_num} - Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {avg_loss_so_far:.4f} | PPL: {ppl_so_far:.2f}", end='\r')
        
        if (batch_idx + 1) % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if (batch_idx + 1) % accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20))
    
    return avg_loss, ppl

# Training loop
EPOCHS = 5
print(f"{'='*60}")
print(f"Starting Training - {EPOCHS} epochs")
print(f"Effective batch size: {batch_size * accumulation_steps} (via gradient accumulation)")
print(f"{'='*60}\n")

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    avg_loss, ppl = train_epoch(epoch)
    print(f"\n  Epoch {epoch}: avg_token_loss={avg_loss:.4f} | perplexity={ppl:.2f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"  âœ“ New best loss: {best_loss:.4f}")
    
    # Clear cache after each epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    print()

print(f"{'='*60}")
print(f"Training completed! Best loss: {best_loss:.4f}")
print(f"{'='*60}\n")

# Inference - Greedy decoding
EOS_IDX = SPECIAL_TOKENS['EOS']
SOS_IDX = SPECIAL_TOKENS['SOS']

def greedy_decode_single(src_seq, max_len=256):
    # Access the actual model if wrapped in DataParallel
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    actual_model.eval()
    
    with torch.no_grad():
        src_tensor = torch.tensor(src_seq, dtype=torch.long, device=device).unsqueeze(0)
        B, S = src_tensor.size()
        
        # Positional embedding for encoder
        num_pos = actual_model.pos_embedding.num_embeddings
        pos_idx = (torch.arange(S, device=device) % num_pos).unsqueeze(0).repeat(B, 1)
        pos_embedded = actual_model.pos_embedding(pos_idx)
        embedded_src = actual_model.src_embedding(src_tensor) + pos_embedded
        _, (hidden, cell) = actual_model.encoder(embedded_src)

        # Start with SOS
        cur = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        outputs = []
        
        for _ in range(max_len):
            emb = actual_model.tgt_embedding(cur)
            out, (hidden, cell) = actual_model.decoder(emb, (hidden, cell))
            logit = actual_model.fc(out[:, -1, :])
            next_token = torch.argmax(logit, dim=-1)
            idx = next_token.item()
            
            if idx == EOS_IDX:
                break
            
            outputs.append(idx)
            cur = next_token.unsqueeze(0)
        
        # Map to string
        chars = [rev_english_vocab.get(i, '?') for i in outputs]
        return ''.join(chars)

# Load and translate test set
print(f"{'='*60}")
print("Loading test data and generating translations...")
print(f"{'='*60}\n")

test_file_path = "/kaggle/input/afml-assignment-1-ec-campus/Team 5/test_part2.csv"

try:
    test_df = pd.read_csv(test_file_path)
    test_lines = test_df['encoded_text'].tolist()
    print(f"Loaded {len(test_lines)} test samples from CSV")
except Exception as e:
    print(f"Error reading CSV: {e}")
    try:
        with open(test_file_path.replace('.csv', '.txt'), 'r', encoding='utf-8') as f:
            test_lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(test_lines)} test samples from TXT")
    except:
        print("Could not load test file. Please check the path.")
        test_lines = []

if test_lines:
    # Encode test data
    encoded_test = [[encoded_vocab.get(ch, SPECIAL_TOKENS['UNK']) for ch in line] for line in test_lines]
    
    # Generate translations with progress indicator
    translations = []
    print(f"Generating translations...")
    for i, seq in enumerate(encoded_test):
        trans = greedy_decode_single(seq)
        translations.append(trans)
        if (i + 1) % 50 == 0:
            print(f"  Translated {i + 1}/{len(encoded_test)} samples...", end='\r')
        # Clear cache periodically during inference
        if (i + 1) % 200 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"  Translated {len(translations)}/{len(encoded_test)} samples - Complete!   \n")
    
    # Save translations
    out_path = '/kaggle/working/05_translations_part2.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        for line, trans in zip(test_lines, translations):
            f.write(f"{line}\t{trans}\n")
    
    print(f"âœ… Saved translations to: {out_path}")
    
    # Show sample translations
    print(f"\n{'='*60}")
    print("Sample Translations (first 5):")
    print(f"{'='*60}")
    for i in range(min(5, len(translations))):
        enc_preview = test_lines[i][:50] + ('...' if len(test_lines[i]) > 50 else '')
        dec_preview = translations[i][:50] + ('...' if len(translations[i]) > 50 else '')
        print(f"\n{i+1}. Encoded: {enc_preview}")
        print(f"   Decoded: {dec_preview}")
    print(f"\n{'='*60}")
else:
    print("â�Œ No test data found!")

print("\nğŸ�‰ Pipeline complete!")


# Debug: Check a single batch
model.eval()
with torch.no_grad():
    # Get one batch
    for src, tgt_in, tgt_out in train_loader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        
        print(f"Batch shape - src: {src.shape}, tgt_in: {tgt_in.shape}, tgt_out: {tgt_out.shape}")
        
        logits = model(src, tgt_in)
        B, T, V = logits.size()
        print(f"Logits shape: {logits.shape}")
        
        # Calculate loss
        loss = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
        print(f"Raw loss: {loss.item()}")
        
        # Check how many valid tokens
        mask = (tgt_out != PAD_IDX)
        num_tokens = mask.sum().item()
        total_positions = B * T
        print(f"Valid tokens: {num_tokens} / {total_positions} ({num_tokens/total_positions*100:.1f}%)")
        
        # Check logits range
        print(f"Logits range: min={logits.min().item():.2f}, max={logits.max().item():.2f}")
        print(f"Logits mean: {logits.mean().item():.2f}, std: {logits.std().item():.2f}")
        
        break

