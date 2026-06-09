!pip install /kaggle/input/faiss-offline/*.whl


import pandas as pd
train_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
train_df.head()


test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_df.head()


category=pd.unique(train_df['Category'])
category


misconception=pd.unique(train_df['Misconception'])
misconception


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pandas as pd
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"
embed_id="/kaggle/input/e5-small-v2/transformers/default/1"

print(f"Loading Gemma model and tokenizer on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Embedding model for FAISS retrieval
print("Loading embedding model (intfloat/e5-small-v2)...")
embed_tokenizer = AutoTokenizer.from_pretrained(embed_id)
embed_model = AutoModel.from_pretrained(embed_id).to(device)

categories = [
    "True_Correct",
    "True_Neither",
    "True_Misconception",
    "False_Neither",
    "False_Misconception",
    "False_Correct",
]

misconceptions = [
    "Incomplete", "WNB", "SwapDividend", "Mult", "FlipChange",
    "Irrelevant", "Wrong_Fraction", "Additive", "Not_variable",
    "Adding_terms", "Inverse_operation", "Inversion", "Duplication",
    "Wrong_Operation", "Whole_numbers_larger", "Longer_is_bigger",
    "Ignores_zeroes", "Shorter_is_bigger", "Wrong_fraction",
    "Adding_across", "Denominator-only_change",
    "Incorrect_equivalent_fraction_addition", "Division", "Subtraction",
    "Unknowable", "Definition", "Interior", "Positive", "Tacking",
    "Wrong_term", "Firstterm", "Base_rate", "Multiplying_by_4",
    "Certainty", "Scale"
]

# ---------- Embedding utils ----------
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

# ---------- Retrieval with FAISS ----------
def build_faiss_index(df: pd.DataFrame, column="StudentExplanation"):
    print("Building FAISS index from dataframe...")
    texts = ["passage: " + str(e) for e in df[column].fillna("")]
    embeddings = encode_texts(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via inner product
    index.add(embeddings)
    print(f"FAISS index built with {len(texts)} entries")
    return index, embeddings

# ---------- Prompt Templates ----------
def prompt_answer_truth(question, answer, context=None):
    ctx = f"\nSimilar examples: {context}" if context else ""
    return f"""Question: {question}
Student answer: {answer}{ctx}

Decide if the student answer is factually True or False.
Output: True or False"""

def prompt_explanation_correctness(question, explanation, context=None):
    ctx = f"\nSimilar examples: {context}" if context else ""
    return f"""Question: {question}
Student explanation: {explanation}{ctx}

Classify the explanation as one of:
- Correct (mathematically sound)
- Neither (neither clearly correct nor clearly wrong)
- Misconception (shows mathematical misunderstanding)

Output: Correct, Neither, or Misconception"""

def prompt_has_misconception(explanation, context=None):
    ctx = f"\nSimilar examples: {context}" if context else ""
    return f"""Student explanation: {explanation}{ctx}

Does the explanation show a misconception?
Answer Yes or No."""

def prompt_misconception_type(explanation, misconception_labels, context=None):
    labels_str = ", ".join(misconception_labels)
    ctx = f"\nSimilar examples: {context}" if context else ""
    return f"""Student explanation: {explanation}{ctx}

If there is a misconception, classify into one of these:
{labels_str}

Output: one label only."""

# ---------- Constrained Classifier ----------
def classify(prompt, options, max_new_tokens=10, num_return_sequences=3):
    print(f"Running classification with options: {options}")
    token_ids = [tokenizer.encode(o, add_special_tokens=False) for o in options]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(len(options), num_return_sequences),
            force_words_ids=token_ids,
            do_sample=False,
            num_return_sequences=num_return_sequences
        )

    decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    results = []
    for d in decoded:
        for o in options:
            if o in d and o not in results:
                results.append(o)
    print(f"Classification results: {results}")
    return results[:num_return_sequences]

# ---------- Pipeline for a Row ----------
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

    # Step 1-4 classification
    truth_options = classify(prompt_answer_truth(question, answer, context), ["True", "False"])
    ex_status_options = classify(prompt_explanation_correctness(question, explanation, context), ["Correct", "Neither", "Misconception"])
    has_miscon_options = classify(prompt_has_misconception(explanation, context), ["Yes", "No"])

    mis_type_options = None
    if "Yes" in has_miscon_options:
        mis_type_options = classify(prompt_misconception_type(explanation, misconceptions, context), misconceptions)

    # Format submission string: Category:Misconception (up to 3)
    submission_preds = []
    for cat in categories:
        if cat.startswith(truth_options[0]):
            if cat.endswith(ex_status_options[0]):
                mis = mis_type_options[0] if mis_type_options else "NaN"
                submission_preds.append(f"{cat}:{mis}")
    submission_str = " ".join(submission_preds[:3])

    result_row = {
        "row_id": row["row_id"],
        "Category:Misconception": submission_str
    }

    # Save progress after each row
    pd.DataFrame([result_row]).to_csv(save_path, mode='a', header=not pd.io.common.file_exists(save_path), index=False)

    return result_row

# ---------- Main ----------
def main():
    print("Loading datasets...")
    train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
    test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

    print("Building FAISS index for trainset...")
    index, embeddings = build_faiss_index(train_df)

    print("Running classification on testset...")
    results = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying test rows"):
        results.append(classify_student_answer_row(row, train_df, index, embeddings, threshold=0.85,save_path="/kaggle/working/submission.csv"))

    submission_df = pd.DataFrame(results)
    submission_df.to_csv("/kaggle/working/submission.csv", index=False)
    print("Saved submission.csv in the required format")

main()

