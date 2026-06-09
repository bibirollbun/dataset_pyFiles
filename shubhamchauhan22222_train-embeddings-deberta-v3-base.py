import pandas as pd


train = pd.read_csv('/kaggle/input/amazon-ml-challenge/train.csv')


pip install -U transformers huggingface_hub


from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# --- Load model and tokenizer ---
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Use multiple GPUs if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if torch.cuda.device_count() > 1:
#     print(f"ðŸš€ Using {torch.cuda.device_count()} GPUs for inference")
#     model = torch.nn.DataParallel(model)

model = model.to(device)
model.eval()

# --- Inference settings ---
batch_size = 64   # Large batch size for L4x4 GPUs
embeddings_list = []

# --- Generate embeddings ---
for i in tqdm(range(0, len(train), batch_size), desc="Generating embeddings"):
    batch_sentences = train.catalog_content.values[i:i+batch_size].tolist()

    # Tokenize
    encoded_input = tokenizer(
        batch_sentences,
        padding=True,
        truncation=True,
        max_length=512,          # Adjust as per your text size
        return_tensors='pt'
    ).to(device)
    
    # Forward pass (no gradient computation)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Move to CPU and store
    embeddings_list.append(sentence_embeddings.cpu())

# --- Combine all batches ---
all_embeddings = torch.cat(embeddings_list, dim=0)

# # --- Add embeddings to DataFrame ---
# train['embeddings'] = list(all_embeddings.numpy())

print("âœ… Embeddings successfully generated and stored in train['embeddings']")
print(f"Shape of embeddings: {all_embeddings.shape}")


import numpy as np

np.save("train_embeddings_deberta-v3-base.npy", all_embeddings.numpy())
print("âœ… Saved embeddings to train_embeddings.npy")







