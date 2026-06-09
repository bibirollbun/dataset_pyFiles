!pip install --no-index --find-links /kaggle/input/heloss/offline_bitsandbytes bitsandbytes
!pip install --no-index --find-links /kaggle/input/heloss/offline_vllm vllm



import os
import pandas as pd
import torch
import vllm
from datasets import Dataset
from vllm.lora.request import LoRARequest
import argparse


# import os
# os.environ["VLLM_USE_V1"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test


model_name = "/kaggle/input/qwen-fine-tuned/transformers/default/1/finetuned_full_model"


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['is_correct'] = test.apply(lambda x: "yes" if x['is_correct'] == 1 else "no", axis=1)


# import vllm


# # Load model
# llm = vllm.LLM(
#     model=model_name,
#     trust_remote_code=True,
#     tensor_parallel_size=1,   # use 1 GPU
#     gpu_memory_utilization=0.95,
#     dtype="half",      
#     # float16
# )

# # Load tokenizer
# tokenizer = llm.get_tokenizer()
# print("✅ Model and tokenizer loaded successfully!")



from transformers import AutoTokenizer
from transformers import Qwen2ForCausalLM, Qwen2Model
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Option A: Force to single GPU
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},  # Force all layers to GPU 0
)


tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'tool' %}
<|im_start|>tool
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}"""


special_character_list = [
    '■', '□', '▲', '△', '▼', '▽', '◆', '◇', '○', '●', '★', '☆', '♦', '♥', '♠', '♣',
    '§', '†', '‡', '※', '∞', '±', '≠', '≈', '√', '∑', '∏', '∆', 'Ω', 'μ', '∂', '→',
    '←', '↑', '↓', '↔', '↕', '〈', '〉', '『', '』', '│', '─', '┌', '┐', '└', '┘', '┼',
    '█', '▓', '▒', '£', '¥', '€', '₩', '©', '®', '™', '♪', '♫', '☀', '☁', '☂', '☃', '☎'
]
from transformers import LogitsProcessor

class LabelOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float('-inf'))
        if scores.dim() == 1:
            mask[self.allowed_token_ids] = 0
        elif scores.dim() == 2:
            mask[:, self.allowed_token_ids] = 0
        else:
            raise ValueError("Unexpected score dimensions")
        return scores + mask


# Get the token IDs for your special characters
allowed_token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in special_character_list]


n_classes = 65


class_mappings = [f"{special_character_list[i]}: {le.classes_[i]}" for i in range(n_classes)]

SYS_PROMPT = f"""You are an expert at analyzing math student responses. Your task is to classify the student's explanation into one of the following Category:Misconception classes.

Respond with ONLY the single character corresponding to the correct classification.

Available classifications:
{', '.join(class_mappings)}

Analyze the given input and provide your classification.
"""



def create_inference_prompt(row):
    user_content = (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {row['is_correct']}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

    # The 'assistant' role is what the model will generate.
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]
    # We use add_generation_prompt=True to signal the model to generate the next part.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


print("Formatting prompts for inference...")
test_input = test.apply(create_inference_prompt, axis=1)


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



model.to(device)
model.eval()  # if doing inference



# If test_input is a pandas Series
texts = test_input.tolist()  # convert to list of strings

inputs = tokenizer(
    texts,           # single string or list of strings
    return_tensors="pt",  # PyTorch tensors
    padding=True,
    truncation=True,
    max_length=256
).to(device)  # <-- move input tensors to same device as model




import torch
from transformers import LogitsProcessor

# Only allow special character tokens
allowed_token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in special_character_list]

class LabelOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float('-inf'))
        if scores.dim() == 1:
            mask[self.allowed_token_ids] = 0
        elif scores.dim() == 2:
            mask[:, self.allowed_token_ids] = 0
        return scores + mask

logits_processor = LabelOnlyLogitsProcessor(allowed_token_ids)

from tqdm import tqdm # A library for progress bars

# --- Corrected Batched Inference Logic ---
batch_size = 8 # You can adjust this based on GPU memory, 8 or 16 is a safe start
all_preds = []

# Loop through the test data in batches
for i in tqdm(range(0, len(test), batch_size)):
    test_input = test.apply(create_inference_prompt, axis=1)    
    texts = test_input.tolist()  # convert to list of strings
    # Tokenize only the current batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    # Generate predictions for the batch
    with torch.no_grad(): # Use no_grad for efficiency during inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=[logits_processor]
        )

    # --- Process the outputs for this batch ---
    # Get scores and find top 3 predictions
    logits = outputs.scores[0]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

    # Decode the token IDs to symbols
    idx_to_label = {tokenizer.encode(c, add_special_tokens=False)[0]: c for c in special_character_list}
    batch_top_labels = [[idx_to_label.get(idx.item(), "UNK") for idx in sample] for sample in top_indices]
    
    all_preds.extend(batch_top_labels)

# Now `all_preds` contains the `top_labels` for the entire test set
top_labels = all_preds
print(f"Generated predictions for {len(top_labels)} test samples.")


# # logits shape: [batch_size, vocab_size]
# logits = outputs.scores[0]  # first (and only) token generated
# probs = torch.softmax(logits, dim=-1)  # convert logits to probabilities

# top_probs, top_indices = torch.topk(probs, k=3, dim=-1)  # top 3

# # Map token IDs back to labels
# idx_to_label = {tokenizer.encode(c, add_special_tokens=False)[0]: c for c in special_character_list}

# top_labels = [[idx_to_label[idx.item()] for idx in sample] for sample in top_indices]
# top_probs_list = top_probs.tolist()



class_mappings_list = [f"{special_character_list[i]}: {le.classes_[i]}" for i in range(n_classes)]
class_mappings_dict = {s.split(': ')[0]: s.split(': ')[1] for s in class_mappings_list}
final_preds = []
for label_list in top_labels:
    mapped_labels = [class_mappings_dict[symbol] for symbol in label_list]
    stripped_labels = [label.strip() for label in mapped_labels]
    final_preds.append(" ".join(stripped_labels))
print("Cleaned predictions for submission:")
print(final_preds[:3])


sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": final_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()


sub["Category:Misconception"].tolist()

