!pip install /kaggle/input/faiss-offline/*.whl
!pip install /kaggle/input/dotoutlines-offline/*.whl


import pandas as pd
train_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
train_df.head()


test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_df.head()


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
train_df['Category'].value_counts().plot(kind='bar')
plt.title('Category Distribution in Training Data')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12,4))
train_df['Misconception'].value_counts().head(15).plot(kind='bar')
plt.title('Top 15 Misconceptions in Training Data')
plt.xlabel('Misconception')
plt.ylabel('Count')
plt.show()


categories=list(pd.unique(train_df['Category']))
categories


misconceptions=[item for item in pd.unique(train_df['Misconception']) if isinstance(item,str)]
misconceptions


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pandas as pd
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# Add outlines import
from outlines.types import Regex
import outlines
# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"
embed_id = "/kaggle/input/e5-small-v2/transformers/default/1"

print(f"Loading Gemma model and tokenizer on {device}...")

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_id).to(device),
    AutoTokenizer.from_pretrained(model_id),
)
# Embedding model for FAISS retrieval
print("Loading embedding model (intfloat/e5-small-v2)...")
embed_tokenizer = AutoTokenizer.from_pretrained(embed_id)
embed_model = AutoModel.from_pretrained(embed_id).to(device)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode_texts(texts, batch_size=8):
    print(f"Encoding {len(texts)} texts into embeddings...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i+batch_size]
        batch_dict = embed_tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = embed_model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0).numpy()


def build_faiss_index(df: pd.DataFrame, column="StudentExplanation"):
    print("Building FAISS index from dataframe...")
    texts = ["passage: " + str(e) for e in df[column].fillna("")]
    embeddings = encode_texts(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via inner product
    index.add(embeddings)
    print(f"FAISS index built with {len(texts)} entries")
    return index, embeddings
train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
index, embeddings = build_faiss_index(train_df)


def prompt_all_in_one(question, answer, explanation, misconception_labels, context=None):
    ctx = f"\nSimilar examples: {context}" if context else ""
    return f"""Question: {question}
Student answer: {answer}
Student explanation: {explanation}{ctx}

Classify the student responses.
"""

def classify_all_in_one(prompt, regex_pattern, max_new_tokens=256):
    output_type = Regex(regex_pattern)
    # outlines.generate returns a string matching the regex
    return model(prompt, output_type=output_type, max_new_tokens=max_new_tokens,disable_compile=True)


import torch
def classify_student_answer_row(row, df=None, index=None, embeddings=None, threshold=0.8, save_path="submission.csv"):
    print(f"Classifying row_id={row['row_id']} QuestionId={row['QuestionId']}")
    question = row["QuestionText"]
    answer = row["MC_Answer"]
    explanation = row["StudentExplanation"]

    # Optionally retrieve context
    context = None
    if df is not None and index is not None and embeddings is not None:
        print("Retrieving similar context using FAISS...")
        q_emb = encode_texts(["query: " + explanation])
        D, I = index.search(q_emb, 1)
        print(f"Top FAISS distance={D[0][0]}")
        if D[0][0] > threshold:  # cosine similarity threshold
            retrieved = df.iloc[I[0][0]]
            context = f"Explanation: {retrieved['StudentExplanation']} | Category: {retrieved['Category']} | Misconception: {retrieved['Misconception']}"
            print(f"Retrieved context from row_id={retrieved['row_id']}")
        else:
            print("No context passed (similarity below threshold)")

    prompt = prompt_all_in_one(question, answer, explanation, misconceptions, context)
    mis_labels = "|".join(misconceptions)
    # Pattern for each scenario
    scenario_patterns = [
        r"Most probably, ",
        r"It can also be ",
        r"Another likely scenario, "
    ]
    single_pred = (
        r"The response seems (True|False)\. "
        r"The explanation seems (Correct|Neither|Misconception)\. "
        r"Misconception present\? (Yes|No)\. "
        rf"Misconception type: ({mis_labels}|NaN)"
    )
    # Combine for up to 3 scenarios
    regex_pattern = (
        rf"{scenario_patterns[0]}{single_pred}\n?"
        rf"{scenario_patterns[1]}{single_pred}\n?"
        rf"{scenario_patterns[2]}{single_pred}\n?"
    )
    output = classify_all_in_one(prompt, regex_pattern)

    if output is None:
        submission_str = "NaN:NaN"
    else:
        import re
        # Find all predictions in the output
        matches = []
        for prefix in scenario_patterns:
            m = re.search(
                prefix + single_pred,
                output
            )
            if m:
                truth, ex_status, has_miscon, mis_type = m.groups()
                for cat in categories:
                    if cat.startswith(truth) and cat.endswith(ex_status):
                        mis = mis_type if has_miscon == "Yes" else "NaN"
                        matches.append(f"{cat}:{mis}")
        if not matches:
            submission_str = "NaN:NaN"
        else:
            submission_str = " ".join(matches[:3])

    result_row = {
        "row_id": row["row_id"],
        "Category:Misconception": submission_str
    }

    # Save progress after each row
    pd.DataFrame([result_row]).to_csv(save_path, mode='a', header=not pd.io.common.file_exists(save_path), index=False)

    return result_row


results = []
for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying test rows"):
    results.append(classify_student_answer_row(row, train_df, index, embeddings, threshold=0.85, save_path="/kaggle/working/submission.csv"))

submission_df = pd.DataFrame(results)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved submission.csv in the required format")

