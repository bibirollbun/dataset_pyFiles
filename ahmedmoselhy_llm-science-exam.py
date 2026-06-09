!pip install -q bitsandbytes faiss-gpu blingfire


import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gc
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import torch
from torch import nn
from transformers import AutoModelForMultipleChoice, AutoTokenizer,AutoModel
from typing import Optional, Union, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
class PretrainedMultipleChoiceModel(nn.Module):
    def __init__(self, model_name, dtype=torch.bfloat16, tokenizer=None):
        super().__init__()
        # Load the underlying transformer model.
        self.base_model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        # Determine the hidden dimension from the model configuration.
        hidden_dim = self.base_model.config.hidden_size
        # Create a linear layer to serve as the classifier for each option.
        self.classifier_head = nn.Linear(hidden_dim, 1, dtype=dtype)
        # Load the associated tokenizer (optional; not used in forward here).
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        # Assume input_ids shape is (batch_size, n_options, seq_length)
        bs, n_opts, seq_len = input_ids.shape
        
        # Flatten the inputs so that each option is treated independently.
        flat_ids = input_ids.reshape(bs * n_opts, seq_len)
        flat_mask = attention_mask.reshape(bs * n_opts, seq_len)
        
        # Pass through the base transformer model.
        outputs = self.base_model(input_ids=flat_ids, attention_mask=flat_mask)
        hidden_states = outputs.last_hidden_state  # (bs * n_opts, seq_len, hidden_dim)
        
        # Determine the index of the last valid token from attention mask.
        # (The sum of attention_mask gives the count of non-padding tokens.)
        last_idx = flat_mask.sum(dim=1) - 1  # Shape: (bs * n_opts,)
        
        # Gather the hidden state corresponding to the last valid token for each example.
        indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        option_repr = hidden_states[indices, last_idx]  # Now shape: (bs * n_opts, hidden_dim)
        
        # Compute logits by passing the representations through the linear classifier.
        raw_logits = self.classifier_head(option_repr)  # (bs * n_opts, 1)
        final_logits = raw_logits.view(bs, n_opts)  # reshape back into (batch_size, n_options)
        
        # If labels are provided, calculate loss using cross-entropy.
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss_value = loss_fn(final_logits, labels)
            return {"loss": loss_value, "logits": final_logits}
        
        return final_logits



train_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv').fillna('').sample(20000)
valid_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')
display(train_df.head(1))
display(valid_df.head(1))


train_df


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki")
model = PretrainedMultipleChoiceModel(model_name = "/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki", 
        dtype = torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki")
tokenizer.pad_token = tokenizer.eos_token

model = PretrainedMultipleChoiceModel(model_name = "/kaggle/input/finetuningresults/llama3.2-FineTuned",
        dtype = torch.bfloat16, tokenizer = tokenizer)



from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Optional[str]="max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # Determine the label key based on the feature keys
        label_key = 'label' if 'label' in features[0] else 'labels'
        
        # Extract labels and remove from features
        labels = [feature.pop(label_key) for feature in features]
        
        # Flatten the features for tokenization
        flattened_features = [
            {k: v[i] for k, v in feature.items()}
            for feature in features
            for i in range(len(feature['input_ids']))
        ]
        
        # Tokenize and pad the flattened features
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        
        # Reshape the batch tensors to match the number of choices
        batch = {k: v.view(len(features), -1, v.size(-1)) for k, v in batch.items()}
        
        # Add the labels to the batch
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        return batch

from dataclasses import dataclass

def custom_tokenize(tokenizer, text1, text2, max_length):
    # Tokenizing both text1 (question) and text2 (answer) with proper truncation
    tokenizer.truncation_side = 'left'  # Truncate text2 from the left side if it's too long

    text2_encoded = tokenizer.encode(text2, truncation=True, max_length=max_length, add_special_tokens=False)
    text2_len = len(text2_encoded)

    # If text2 is smaller than max_length, tokenize text1, truncating from the right side
    if text2_len < max_length:
        tokenizer.truncation_side = 'right'
        text1_encoded = tokenizer.encode(text1, truncation=True, max_length=max_length - text2_len, add_special_tokens=False)
    else:
        text1_encoded = []  # If text2 exceeds max_length, no text1 will be added

    # Combine the tokenized question (text1) and answer (text2)
    input_ids = text1_encoded + text2_encoded
    attention_mask = [1] * len(input_ids)  # Attention mask for all tokens

    return {"input_ids": input_ids, "attention_mask": attention_mask}

def preprocess_function_multiple_choice(examples, tokenizer=tokenizer,max_length=256):
    
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    input_ids, attention_masks, labels = [], [], []

    for i, (q, A, B, C, D, E, answer) in enumerate(zip(
            examples['prompt'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'], examples['answer'])):
        
        # Creating the question and answer pairs
        text1 = [f"Question: {q}"] * 5
        text2 = [f"\n###\nAnswer: {option}\n###\nTrue or False:" for option in [A, B, C, D, E]]
        
        # Tokenize the pairs
        tokenized = [custom_tokenize(tokenizer, t1, t2, max_length) for t1, t2 in zip(text1, text2)]
        
        # Extract input_ids, attention_mask, and label for each question
        input_ids.append([x['input_ids'] for x in tokenized])
        attention_masks.append([x['attention_mask'] for x in tokenized])
        labels.append(label_mapping[answer])

    # Returning the processed batch
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

def apk3(scores, true_idx):
    """
    Compute the Average Precision at 3 for a single example.
    
    Args:
        scores (1D array): Predicted scores for each class.
        true_idx (int): Index of the correct class.
    
    Returns:
        float: AP@3 score (1/rank if correct in top 3, else 0).
    """
    # Sort indices descending by score, take top 3
    top3 = np.argsort(-scores)[:3]
    # Compute inverse rank if true_idx appears in top3
    for rank, pred in enumerate(top3, start=1):
        if pred == true_idx:
            return 1.0 / rank
    return 0.0

def map_at_3(pred_matrix, true_labels):
    """
    Compute the Mean Average Precision at 3 over a batch.
    
    Args:
        pred_matrix (2D array): Shape (n_samples, n_classes), predicted scores.
        true_labels (1D array): True class indices (length n_samples).
    
    Returns:
        float: MAP@3 across all samples.
    """
    ap_scores = [apk3(row, lbl) for row, lbl in zip(pred_matrix, true_labels)]
    return np.mean(ap_scores)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # Get the predicted class labels
    accuracy = np.mean(preds == labels)  # Compute accuracy as mean of correct predictions
    map3_score = map_at_3(logits, labels)  # Compute MAP@3 score using the optimized map_at_3 function
    return {"accuracy": accuracy, "map_at_3": map3_score}


from datasets import load_dataset
from datasets import Dataset, DatasetDict
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})
dataset


tokenized_dataset = dataset.map(preprocess_function_multiple_choice, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


import warnings
warnings.filterwarnings('ignore')
trainer_finetune = Trainer(
    model=model,
    args= TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,                  
    logging_steps=50,               
    warmup_ratio=0.1,
    learning_rate=5e-5,
    optim='paged_adamw_32bit',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.03,
    save_total_limit=1,
    report_to="none",
    save_only_model=True,
    fp16=True,  # Use mixed-precision (FP16) training
    torch_compile=True,
    gradient_accumulation_steps=2,
    lr_scheduler_type='cosine',  # Cosine learning rate decay
    #dataloader_num_workers=4  # More workers for faster data loading
),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer, 
                                              padding='longest', 
                                              max_length=256),
    compute_metrics=compute_metrics
)



trainer_finetune.train()
trainer_finetune.save_model('./llama3.2-FineTuned')


import matplotlib.pyplot as plt
import pandas as pd


# Assuming 'trainer_finetune' is your trained Trainer object [cite: 28]
# Extract log history
log_history = trainer_finetune.state.log_history

steps = [log['step'] for log in log_history if 'loss' in log]
train_loss = [log['loss'] for log in log_history if 'loss' in log]
eval_steps = [log['step'] for log in log_history if 'eval_loss' in log]
eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]

plt.figure(figsize=(10, 5))
plt.plot(steps, train_loss, label='Training Loss')
plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss Over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import pandas as pd


# Assuming 'trainer_finetune' is your trained Trainer object [cite: 28]
# Extract log history (use log_history from above)
eval_accuracy = [log['eval_accuracy'] for log in log_history if 'eval_accuracy' in log]
eval_map3 = [log['eval_map_at_3'] for log in log_history if 'eval_map_at_3' in log]
# Use eval_steps from above

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Steps')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(eval_steps, eval_accuracy, color=color, label='Validation Accuracy', marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('MAP@3', color=color)
ax2.plot(eval_steps, eval_map3, color=color, label='Validation MAP@3', marker='x')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Validation Metrics Over Steps')
fig.tight_layout() # otherwise the right y-label is slightly clipped
plt.grid(True)
plt.show()


from datasets import load_dataset
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ————————————————————————————————————————————
# 1. Only grab the *article* shards, not the index file
data_dir = Path("/kaggle/input/wikipedia-20230701")
shard_files = [
    str(p) for p in data_dir.glob("*.parquet")
    if p.name != "wiki_2023_index.parquet"
]

wiki_ds = load_dataset(
    "parquet",
    data_files=shard_files,
    split="train"
)

# ————————————————————————————————————————————
# 2. Your preprocess function unchanged
def preprocess(examples):
    reconstructed = []
    for content, heading in zip(examples["text"], examples["title"]):
        frags = [seg.strip() for seg in content.split(".") if seg.strip()]
        first_valid = next(
            (seg for seg in frags if len(seg) >= 4),
            frags[0] if frags else content.strip()
        )
        reconstructed.append(f"{heading}: {first_valid}")
    return {"text": reconstructed}

wiki_ds = wiki_ds.map(
    preprocess,
    batched=True,
    remove_columns=wiki_ds.column_names
)

# ————————————————————————————————————————————
# 3. Embedder setup
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
embedder.half()
embedder.max_seq_length = 384

# ————————————————————————————————————————————
# 4. Compute and index corpus embeddings
corpus_embs = embedder.encode(
    wiki_ds["text"],
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dim = corpus_embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(corpus_embs.astype(np.float32))
faiss.write_index(index, "WikiCorpus.faiss")



from datasets import load_dataset
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ————————————————————————————————————————————
# 3. Embedder setup
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
embedder.half()
embedder.max_seq_length = 384

# ————————————————————————————————————————————
# 4. Compute and index corpus embeddings
corpus_embs = embedder.encode(
    wiki_ds["text"],
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ————————————————————————————————————————————
# 5. Load prompts & external index, then search
exam_df       = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")
queries       = exam_df["prompt"].tolist()
external_idx  = faiss.read_index("/kaggle/input/finetuningresults/WikiCorpus.faiss")

query_embs = embedder.encode(
    queries,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

scores, indices = external_idx.search(query_embs, 5)
print("Sample scores:",  scores[:3])
print("Sample indices:", indices[:3])



import pandas as pd
import gc
from tqdm.auto import tqdm

# 1. Load the index mapping of article IDs to their file paths
index_df = pd.read_parquet(
    "/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet",
    columns=["id", "file"]
)

# 2. Gather, for each prompt, the matching articles and tag them with the prompt’s index
records = []
for prompt_idx, hits in tqdm(enumerate(indices), total=len(indices)):
    # hits is an array of row-indices into index_df
    subset = index_df.loc[hits].copy()
    subset["prompt_idx"] = prompt_idx
    records.append(subset)

# 3. Concatenate all results into one DataFrame
results_df = pd.concat(records, ignore_index=True)

# 4. Keep only the columns we need, dedupe, sort, and reset the row index
results_df = (
    results_df[["id", "prompt_idx", "file"]]
    .drop_duplicates()
    .sort_values(by=["file", "id"])
    .reset_index(drop=True)
)

# 5. Free up the original index DataFrame
del index_df
gc.collect()

# 6. Display the final mapping
results_df


import pandas as pd
import gc
from tqdm.auto import tqdm

# 1. Read the parquet index of article IDs → file names
index_map = pd.read_parquet(
    "/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet",
    columns=["id", "file"]
)

# 2. Iterate over each unique file and pull out the matching article texts
text_segments = []
for filename in tqdm(results_df["file"].unique(), total=results_df["file"].nunique()):
    # get all IDs that belong to this file
    ids_in_file = results_df.loc[results_df["file"] == filename, "id"].astype(str).tolist()
    
    # load just the 'id' and 'text' columns from that file
    raw_df = pd.read_parquet(f"/kaggle/input/wikipedia-20230701/{filename}", columns=["id", "text"])
    
    # filter to only the rows we care about
    filtered = raw_df[raw_df["id"].isin(ids_in_file)].copy()
    del raw_df
    gc.collect()
    
    text_segments.append(filtered)

# 3. Concatenate, dedupe, reset index, and free memory
article_texts = (
    pd.concat(text_segments, ignore_index=True)
      .drop_duplicates()
      .reset_index(drop=True)
)
gc.collect()

# 4. `article_texts` now holds your full text data
article_texts



from collections.abc import Iterable
import pandas as pd
import blingfire as bf
from tqdm.auto import tqdm

def process_documents(docs: Iterable[str],
                      doc_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper to chunk full documents into sections and (optionally) sentences.
    """
    sections_df = sectionize_documents(docs, doc_ids, disable_progress_bar)
    if split_sentences:
        sections_df = sentencize(
            texts=sections_df['text'].tolist(),
            ids=sections_df['document_id'].tolist(),
            base_offsets=sections_df['offset'].tolist(),
            min_len=filter_len,
            disable_progress_bar=disable_progress_bar
        )
    return sections_df


def sectionize_documents(docs: Iterable[str],
                         doc_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Wrap each full document as one 'section' with offset (0, len).
    """
    records = []
    for did, content in tqdm(zip(doc_ids, docs), total=len(docs), disable=disable_progress_bar):
        records.append({
            'document_id': did,
            'text': content,
            'offset': (0, len(content))
        })
    df_sections = pd.DataFrame(records)
    return df_sections.sort_values(['document_id', 'offset']).reset_index(drop=True)


def sentencize(texts: Iterable[str],
               ids: Iterable,
               base_offsets: Iterable[tuple[int, int]],
               min_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split each section into sentences using blingfire, preserving absolute offsets.
    """
    sentence_rows = []
    for text, did, (sec_start, _) in tqdm(zip(texts, ids, base_offsets),
                                          total=len(texts),
                                          disable=disable_progress_bar):
        try:
            _, sent_offsets = bf.text_to_sentences_and_offsets(text)
        except Exception:
            continue
        for start, end in sent_offsets:
            if end - start <= min_len:
                continue
            sent_text = text[start:end]
            sentence_rows.append({
                'document_id': did,
                'text': sent_text,
                'offset': (sec_start + start, sec_start + end)
            })
    return pd.DataFrame(sentence_rows)


import gc
from tqdm.auto import tqdm
import faiss

# 1. Split each Wikipedia article into sentence-level chunks
wiki_sent_df = process_documents(
    docs=article_texts.text.values,
    doc_ids=article_texts.id.values,
    split_sentences=True,
    filter_len=3,
    disable_progress_bar=False
)


wiki_sent_df


wiki_sent_df.iloc[0]['text']


# 2. Embed every sentence chunk into a vector
## Embedding Documents
wiki_sent_embeddings = embedder.encode(
    wiki_sent_df['text'].tolist(),
    batch_size=32,
    device='cuda',
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
# Free unused memory
gc.collect()


wiki_sent_embeddings=wiki_sent_embeddings.astype('float32')
wiki_sent_embeddings.shape[1]


# 3. Create a single string of all five answer choices per question
exam_df['answer_all'] = exam_df[['A', 'B', 'C', 'D', 'E']].agg(" ".join, axis=1)

# 4. Combine the prompt with its answer string to form the final query text
exam_df['prompt_answer_stem'] = exam_df['prompt'] + " " + exam_df['answer_all']
query_embeddings = embedder.encode(
    exam_df['prompt_answer_stem'].tolist(),
    batch_size=32,
    device='cuda',
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
gc.collect()


query_embeddings=query_embeddings.astype('float32')


wiki_sent_df


results_df


# 5. For each question, build a context by retrieving the top‑K related sentences
NUM_SENTENCES_INCLUDE = 32
contexts = []

for q_idx in tqdm(exam_df.itertuples(), total=len(exam_df)):

    prompt_id = q_idx.Index

    prompt_indices = wiki_sent_df[wiki_sent_df['document_id'].isin(results_df[results_df['prompt_idx']==prompt_id]['id'].values)].index.values

    if prompt_indices.shape[0] > 0:
        d=wiki_sent_embeddings.shape[1]
        prompt_index = faiss.IndexFlatL2(d)#faiss.index_factory(wiki_sent_embeddings.shape[1], "Flat")
        #print(prompt_index)
        prompt_index.add(wiki_sent_embeddings[prompt_indices])

        context = ""
        
        ## Get the top matches
        ss, ii = prompt_index.search(query_embeddings, NUM_SENTENCES_INCLUDE)
        for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
            context += wiki_sent_df.loc[prompt_indices]['text'].iloc[_i] + " "
        
    contexts.append(context)

# 6. Attach the generated 'context' column back to the DataFrame
exam_df['context'] = contexts

# 7. Inspect the final DataFrame
exam_df


# Assuming 'scores' holds the FAISS search scores [cite: 32]
plt.figure(figsize=(10, 5))
sns.histplot(scores.flatten(), kde=True) # Flatten since scores might be (num_queries, k)
plt.title('Distribution of FAISS Retrieval Scores')
plt.xlabel('Similarity Score (L2 Distance)')
plt.ylabel('Frequency')
plt.show()


# Assuming 'exam_df' has the 'context' column added [cite: 47]
context_lengths = exam_df['context'].apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(context_lengths, kde=True)
plt.title('Distribution of Generated Context Lengths')
plt.xlabel('Context Length (characters)')
plt.ylabel('Frequency')
plt.show()


# Assuming 'train_df' is your training dataframe [cite: 1]
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='answer', order=['A', 'B', 'C', 'D', 'E'])
plt.title('Distribution of Correct Answers in Training Data')
plt.xlabel('Answer Choice')
plt.ylabel('Frequency')
plt.show()


# Load the fine-tuned model (if not already in memory)
model = PretrainedMultipleChoiceModel(model_name='/kaggle/input/finetuningresults/llama3.2-FineTuned', dtype=torch.bfloat16)
model.eval() # Set to evaluation mode
model.to('cuda') # Move to GPU if available

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/finetuningresults/llama3.2-FineTuned") # Or the original one used
tokenizer.pad_token = tokenizer.eos_token

# Inference dataset preparation using exam_df which now includes 'context'
from datasets import Dataset
exam_dataset_with_context = Dataset.from_pandas(exam_df)

def preprocess_for_inference_with_rag(examples, tokenizer=tokenizer, max_length=1024): # Increased max_length for context
    input_ids, attention_masks = [], []
    # Note: No labels needed for inference typically, unless evaluating

    # Iterate through each example (prompt + context + options)
    for i, (ctx, q, A, B, C, D, E) in enumerate(zip(
            examples['context'], examples['prompt'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'])):

        # Format input including the RAG context
        # Adjust formatting as needed based on how the model was fine-tuned
        text1_base = f"Context: {ctx}\n\nQuestion: {q}" # Combine context and question
        text1 = [text1_base] * 5
        text2 = [f"\n###\nAnswer: {option}\n###\nTrue or False:" for option in [A, B, C, D, E]] # Keep answer format consistent

        # Use the custom_tokenize function (or adapt it)
        # Ensure truncation handles the potentially long combined input
        tokenized = [custom_tokenize(tokenizer, t1, t2, max_length) for t1, t2 in zip(text1, text2)]

        input_ids.append([x['input_ids'] for x in tokenized])
        attention_masks.append([x['attention_mask'] for x in tokenized])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        # 'labels': labels # Include if evaluating performance
    }

# Tokenize the exam dataset with the new function
tokenized_exam_dataset = exam_dataset_with_context.map(preprocess_for_inference_with_rag, batched=True)
tokenized_exam_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask']) # Add 'labels' if evaluating

# Prepare Data Collator
# data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding='longest', max_length=1024)
# This collator expects 'labels', might need adjustment or use DataCollatorWithPadding if labels aren't used for pure inference

# Use Trainer for prediction (if evaluating) or run manual inference loop
# predictions = trainer_finetune.predict(tokenized_exam_dataset)
# final_answers = np.argmax(predictions.predictions, axis=-1)

# --- Manual Inference Loop Example (Simplified) ---
# from torch.utils.data import DataLoader
# import torch

# eval_dataloader = DataLoader(tokenized_exam_dataset, batch_size=4, collate_fn=data_collator) # Adjust batch size

# all_logits = []
# with torch.no_grad():
#    for batch in tqdm(eval_dataloader):
#        inputs = {k: v.to('cuda') for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
#        logits = model(**inputs) # Get logits directly from the model's forward pass
#        all_logits.append(logits.cpu())

# final_logits = torch.cat(all_logits, dim=0)
# final_answers = torch.argmax(final_logits, axis=-1).numpy()
# print("Predicted answer indices:", final_answers)



import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments # Assuming Trainer is used for prediction convenience
from tqdm.auto import tqdm # For progress bars

# --- Configuration ---
BASE_MODEL_PATH = "/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki" # Path to base model [cite: 1]
FINETUNED_MODEL_PATH = "/kaggle/input/finetuningresults/llama3.2-FineTuned" # Path to your fine-tuned model [cite: 29]
EVAL_DF_PATH = "/kaggle/input/kaggle-llm-science-exam/train.csv" # Assuming exam_df comes from here [cite: 32] and has 'answer' column
# Ensure exam_df also has the 'context' column generated by your RAG script [cite: 47]
MAX_LENGTH_NO_CONTEXT = 256 # Max length used during fine-tuning without extra RAG context
MAX_LENGTH_WITH_CONTEXT = 1024 # Potentially longer max length needed for RAG context

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # Get the predicted class labels
    accuracy = np.mean(preds == labels)  # Compute accuracy as mean of correct predictions
    map3_score = map_at_3(logits, labels)  # Compute MAP@3 score using the optimized map_at_3 function
    return {"accuracy": accuracy, "map_at_3": map3_score}
    
# --- Load Evaluation Data (assuming exam_df is already loaded and has context) ---
exam_df = pd.read_csv(EVAL_DF_PATH)
# ... (add RAG context column to exam_df as done in your script) ...
# Make sure 'answer' column exists in exam_df for evaluation
if 'answer' not in exam_df.columns:
    raise ValueError("Evaluation dataframe 'exam_df' must contain an 'answer' column with true labels.")

eval_dataset = Dataset.from_pandas(exam_df)
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

# --- Preprocessing Functions ---

# 1. Preprocessing without RAG context (for Base and Fine-tuned methods)
def preprocess_no_context(examples, tokenizer, max_length=MAX_LENGTH_NO_CONTEXT):
    input_ids, attention_masks, labels = [], [], []
    for i, (q, A, B, C, D, E, answer) in enumerate(zip(
            examples['prompt'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'], examples['answer'])):
        text1 = [f"Question: {q}"] * 5
        text2 = [f"\n###\nAnswer: {option}\n###\nTrue or False:" for option in [A, B, C, D, E]]
        tokenized = [custom_tokenize(tokenizer, t1, t2, max_length) for t1, t2 in zip(text1, text2)] # [cite: 19]
        input_ids.append([x['input_ids'] for x in tokenized])
        attention_masks.append([x['attention_mask'] for x in tokenized])
        labels.append(label_mapping[answer])
    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

# 2. Preprocessing with RAG context (for Combined method)
def preprocess_with_rag_context(examples, tokenizer, max_length=MAX_LENGTH_WITH_CONTEXT):
    input_ids, attention_masks, labels = [], [], []
    for i, (ctx, q, A, B, C, D, E, answer) in enumerate(zip(
            examples['context'], examples['prompt'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'], examples['answer'])):
        # Format including RAG context
        text1_base = f"Context: {ctx}\n\nQuestion: {q}"
        text1 = [text1_base] * 5
        text2 = [f"\n###\nAnswer: {option}\n###\nTrue or False:" for option in [A, B, C, D, E]]
        tokenized = [custom_tokenize(tokenizer, t1, t2, max_length) for t1, t2 in zip(text1, text2)] # [cite: 19]
        input_ids.append([x['input_ids'] for x in tokenized])
        attention_masks.append([x['attention_mask'] for x in tokenized])
        labels.append(label_mapping[answer])
    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

# --- Evaluation Function ---
def evaluate_method(model_path, processor, dataset, tokenizer, max_length, compute_metrics_fn):
    print(f"--- Evaluating Model: {model_path} ---")
    # Load Model and Tokenizer
    model = PretrainedMultipleChoiceModel(model_name=model_path, dtype=torch.bfloat16) # [cite: 2]
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    # Tokenize Dataset
    tokenized_dataset = dataset.map(processor, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length}, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Data Collator
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding='longest', max_length=max_length) # [cite: 12]

    # Use a dummy Trainer for prediction (simplifies handling batches, devices, etc.)
    # No actual training happens here.
    dummy_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=4, # Adjust as needed
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to="none",
        fp16=True, # Use mixed precision if available
    )
    
    trainer = Trainer(
        model=model,
        args=dummy_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # Get Predictions
    print("Running predictions...")
    predictions = trainer.predict(tokenized_dataset)
    
    # Clean up dummy dir
    # import shutil
    # shutil.rmtree("./eval_temp")
    
    # Return metrics (predictions.metrics will contain output from compute_metrics)
    print(f"Metrics for {model_path}: {predictions.metrics}")
    print("-" * 30)
    return predictions.metrics


# --- Run Evaluations ---

# Method 1: Base Model (No Fine-tuning, No RAG Context)
print("METHOD 1: BASE MODEL")
tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer_base.pad_token = tokenizer_base.eos_token
metrics_base = evaluate_method(
    model_path=BASE_MODEL_PATH,
    processor=preprocess_no_context,
    dataset=eval_dataset,
    tokenizer=tokenizer_base,
    max_length=MAX_LENGTH_NO_CONTEXT,
    compute_metrics_fn=compute_metrics # Your metrics function [cite: 26]
)

# Method 2: Fine-tuned Model (No RAG Context)
print("\nMETHOD 2: FINE-TUNED MODEL (NO RAG)")
# tokenizer_ft = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH) # Load tokenizer saved with fine-tuned model
tokenizer_ft = AutoTokenizer.from_pretrained("/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki") # Or the original one used

tokenizer_ft.pad_token = tokenizer_ft.eos_token
metrics_ft_no_rag = evaluate_method(
    model_path=FINETUNED_MODEL_PATH,
    processor=preprocess_no_context,
    dataset=eval_dataset,
    tokenizer=tokenizer_ft,
    max_length=MAX_LENGTH_NO_CONTEXT, # Use length consistent with fine-tuning
    compute_metrics_fn=compute_metrics
)

# Method 3: Fine-tuned Model + RAG Context
print("\nMETHOD 3: FINE-TUNED MODEL + RAG CONTEXT")
# Tokenizer is the same as Method 2
metrics_ft_with_rag = evaluate_method(
    model_path=FINETUNED_MODEL_PATH,
    processor=preprocess_with_rag_context, # Use the RAG context processor
    dataset=eval_dataset,
    tokenizer=tokenizer_ft,
    max_length=MAX_LENGTH_WITH_CONTEXT, # Use potentially longer length
    compute_metrics_fn=compute_metrics
)

# --- Display Final Results ---
print("\n--- HEAD-TO-HEAD RESULTS ---")
results_summary = pd.DataFrame({
    'Base Model': metrics_base,
    'Fine-tuned (No RAG)': metrics_ft_no_rag,
    'Fine-tuned + RAG': metrics_ft_with_rag
})
print(results_summary)


import matplotlib.pyplot as plt
import pandas as pd

# --- Get log data from Trainer state AFTER training ---
# Choose the relevant trainer object (e.g., trainer_finetune or trainer)
# trainer_log_history = trainer_finetune.state.log_history 
trainer_log_history = trainer.state.log_history # Or use the second trainer if plotting that run

# --- Process log history into a DataFrame ---
# The structure might vary slightly based on HF version, adjust parsing if needed
logs = []
for entry in trainer_log_history:
    step = entry.get('step')
    loss = entry.get('loss') # Training loss logged at training steps
    eval_loss = entry.get('eval_loss') # Eval loss logged at eval steps
    eval_accuracy = entry.get('eval_accuracy')
    eval_map_at_3 = entry.get('eval_map_at_3')
    # Training metrics might be logged separately or less frequently
    # You might need to align/interpolate if train/eval steps differ significantly
    # For simplicity, we'll mostly plot eval metrics here if train metrics aren't readily available per step
    logs.append({
        'step': step,
        'train_loss': loss if eval_loss is None else None, # Crude way to separate
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'eval_map_at_3': eval_map_at_3
    })

log_df = pd.DataFrame(logs).sort_values(by='step')

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Loss Curve (Plotting eval loss and interpolating train loss if needed)
train_loss_df = log_df.dropna(subset=['train_loss'])
eval_loss_df = log_df.dropna(subset=['eval_loss'])
axes[0].plot(train_loss_df['step'], train_loss_df['train_loss'], label='Training Loss', marker='.', linestyle='--')
axes[0].plot(eval_loss_df['step'], eval_loss_df['eval_loss'], label='Validation Loss', marker='o')
axes[0].set_xlabel('Training Steps')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy Curve (Eval only as train accuracy wasn't explicitly in compute_metrics)
eval_acc_df = log_df.dropna(subset=['eval_accuracy'])
axes[1].plot(eval_acc_df['step'], eval_acc_df['eval_accuracy'], label='Validation Accuracy', marker='o')
axes[1].set_xlabel('Training Steps')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

# MAP@3 Curve (Eval only)
eval_map_df = log_df.dropna(subset=['eval_map_at_3'])
axes[2].plot(eval_map_df['step'], eval_map_df['eval_map_at_3'], label='Validation MAP@3', marker='o')
axes[2].set_xlabel('Training Steps')
axes[2].set_ylabel('MAP@3')
axes[2].set_title('Validation MAP@3')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# --- Get predictions and true labels ---
# Ensure valid_df and test_predictions are available from your script execution
valid_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv') # Loaded in the script [cite: 11, 50]

# Mapping from the script
label_mapping_inv = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'} 
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4} #[cite: 18, 48]

# Make sure valid_df corresponds to the data used for test_predictions
# If test_dataloader used the full valid_df:
y_true_labels = valid_df['answer'].map(label_mapping).values 

# test_predictions are the logits calculated at the end of the script [cite: 53]
y_pred_logits = test_predictions 
y_pred_labels = np.argmax(y_pred_logits, axis=1)
labels_text = ['A', 'B', 'C', 'D', 'E']
# --- End Placeholder ---

# Ensure y_true_labels and y_pred_labels have the same length
if len(y_true_labels) != len(y_pred_labels):
     print(f"Warning: Length mismatch! True labels: {len(y_true_labels)}, Predicted labels: {len(y_pred_labels)}")
     # You might need to adjust which part of valid_df corresponds to test_predictions if it wasn't the full set

cm = confusion_matrix(y_true_labels[:len(y_pred_labels)], y_pred_labels, labels=range(len(labels_text))) # Truncate true labels if necessary
cm_df = pd.DataFrame(cm, index=labels_text, columns=labels_text)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Answer')
plt.xlabel('Predicted Answer')
plt.show()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Choose embeddings and define corresponding labels ---
# Option 1: Visualize query embeddings colored by correctness
embeddings_to_plot = query_embeddings # From RAG part of the script [cite: 45]
# Ensure valid_df and test_predictions are available
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
y_true = valid_df['answer'].map(label_mapping).values[:len(test_predictions)] # Match length
y_pred = np.argmax(test_predictions, axis=1)
is_correct = (y_pred == y_true)
labels_for_coloring = is_correct 
plot_title = 't-SNE of Query Embeddings (Colored by Correctness)'

# Option 2: Visualize sentence embeddings (might be too many points)
embeddings_to_plot = wiki_sent_embeddings # [cite: 44, 46]
# You'd need relevant labels for sentences, e.g., document ID or topic
labels_for_coloring = wiki_sent_df['document_id'].astype('category').cat.codes # Example: color by doc ID
plot_title = 't-SNE of Wikipedia Sentence Embeddings'
# --- End Placeholder ---

# Ensure embeddings are float32 for t-SNE
embeddings_to_plot = embeddings_to_plot.astype(np.float32)

# Adjust perplexity based on data size (important for t-SNE)
n_points = embeddings_to_plot.shape[0]
perplexity_value = min(30, max(5, n_points - 1)) # Keep perplexity reasonable

print(f"Running t-SNE on {n_points} embeddings with perplexity={perplexity_value}...")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=300, init='pca', learning_rate='auto')
embeddings_2d = tsne.fit_transform(embeddings_to_plot)

# --- Plotting ---
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_for_coloring, cmap='viridis', alpha=0.6)

# Add legend for categorical data like correctness
try:
    # Create custom legend handles if needed
    classes = np.unique(labels_for_coloring)
    if len(classes) < 10: # Avoid overly large legends
       legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
                                 markerfacecolor=scatter.cmap(scatter.norm(cls)), markersize=10) for cls in classes]
       plt.legend(handles=legend_handles, title="Correctness" if isinstance(labels_for_coloring[0], np.bool_) else "Category")
except Exception as e:
     print(f"Could not automatically create legend: {e}")

plt.title(plot_title)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- MANUALLY DEFINE your metrics data ---
# You need to calculate/collect these values for each model/configuration you want to compare.
# Accuracy & MAP@3 can come from Trainer evaluation logs or final calculation[cite: 54].
# Others like Context Recall, Faithfulness need separate evaluation logic based on your RAG outputs.

metrics_data = {
    # Example for the model trained in the script
    'Llama3.2_DAPT_FT_RAG': {
        'Accuracy': 0.81,        # Replace with your actual final validation accuracy
        'MAP@3': 0.92,           # Replace with your actual final validation MAP@3 (e.g., value 'm' [cite: 54])
        'Context Recall': 0.85,  # Placeholder: Calculate externally if needed
        'Faithfulness': 0.78,    # Placeholder: Calculate externally if needed
        'Answer Relevancy': 0.88,# Placeholder: Calculate externally if needed
        'Context Precision': 0.90 # Placeholder: Calculate externally if needed 
    },
    # Add more models/configs here if you have them
    # 'Another_Model_Config': { ... } 
}
# --- End Placeholder ---

df = pd.DataFrame(metrics_data)
# Check if DataFrame is empty
if df.empty:
    print("Metrics data is empty. Please populate the metrics_data dictionary.")
else:
    labels = df.index.tolist()
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Helper function to plot data
    def add_to_radar(model_name, color):
        values = df[model_name].values.flatten().tolist()
        values += values[:1] # Complete the loop
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, color=color, alpha=0.4)

    # Plot data for each model
    colors = ['blue', 'red', 'green', 'purple'] # Add more colors if needed
    for i, model_name in enumerate(df.columns):
         add_to_radar(model_name, colors[i % len(colors)])

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set y-axis limits and labels (assuming metrics are 0-1 range)
    ax.set_yticks(np.arange(0, 1.1, 0.2)) 
    ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0, 1.1, 0.2)])
    ax.set_ylim(0, 1.05) # Set Y axis limit

    plt.title('Model Performance Comparison', size=20, y=1.1)
    # Adjust legend position if needed
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1)) 
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch # Assuming test_predictions are logits

# --- Get predictions and true labels ---
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
y_true_labels = valid_df['answer'].map(label_mapping).values[:len(test_predictions)] # Match length
y_pred_logits = test_predictions # Logits from the script
y_pred_labels = np.argmax(y_pred_logits, axis=1)
is_correct = (y_pred_labels == y_true_labels)

# --- Calculate score margin for incorrect predictions ---
margins = []
for i in range(len(y_pred_logits)):
    if not is_correct[i]:
        correct_label_index = y_true_labels[i]
        predicted_label_index = y_pred_labels[i]
        
        score_correct = y_pred_logits[i, correct_label_index]
        score_predicted = y_pred_logits[i, predicted_label_index] # Score of the (wrong) predicted answer
        
        # Margin: Score(Correct Answer) - Score(Predicted Answer)
        # Negative margin means the wrong answer had a higher score
        margin = score_correct - score_predicted 
        margins.append(margin)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.hist(margins, bins=30, alpha=0.7, color='salmon')
plt.xlabel('Score Margin (Score[Correct] - Score[Predicted]) for Incorrect Answers')
plt.ylabel('Frequency')
plt.title('Distribution of Score Margins for Incorrect Predictions')
plt.grid(axis='y', linestyle='--')
plt.axvline(0, color='black', linestyle=':', linewidth=1) # Line at zero margin
plt.show()

