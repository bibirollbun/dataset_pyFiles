# Core libraries
import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

#
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Torch and Transformers
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import log_loss, accuracy_score

from tqdm.notebook import tqdm


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# Dataset paths (Kaggle)
TRAIN_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip"
TRAIN_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_text.zip"
TEST_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/test_variants.zip"
TEST_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/test_text.zip"


# Load training variants
train_variants = pd.read_csv(TRAIN_VARIANTS_PATH)

# Load clinical text (|| separator)
train_text = pd.read_csv(
    TRAIN_TEXT_PATH,
    sep=r"\|\|",
    engine="python",
    names=["ID", "Text"],
    skiprows=1
)

# Merge datasets
train_df = train_variants.merge(train_text, on="ID", how="left")
train_df["Text"] = train_df["Text"].fillna("")

train_df.head()


# Combine structured data with clinical evidence
train_df["model_text"] = (
    "Gene: " + train_df["Gene"].astype(str) +
    " | Variant: " + train_df["Variation"].astype(str) +
    " | Evidence: " + train_df["Text"].astype(str)
)

train_df[["model_text"]].head()


# BioClinicalBERT model
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


# Model llm
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


## Load Dataset (MSK)

# Load variants
variants = pd.read_csv(
    "/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip"
)

# Load clinical text
texts = pd.read_csv(
    "/kaggle/input/msk-redefining-cancer-treatment/training_text.zip",
    sep=r"\|\|",
    engine="python",
    names=["ID", "Text"],
    skiprows=1
)

# Merge
df = variants.merge(texts, on="ID", how="left")
df["Text"] = df["Text"].fillna("")

# Use only a small subset for fast testing
df_small = df.iloc[:10].copy()
df_small.shape


# Prompt template (engineering)
PROMPT_TEMPLATE = """
You are a senior clinical oncologist and molecular pathologist.

Your task is to interpret cancer-related genetic variants using
clinical evidence and support clinical decision-making.

Gene: {GENE}
Variant: {VARIATION}

Clinical Evidence:
{CLINICAL_TEXT}

Classify the variant into one of 9 clinical classes and
extract clinically relevant keywords.
"""


# Build prompt and model text
df_small["prompt"] = df_small.apply(
    lambda r: PROMPT_TEMPLATE.format(
        GENE=r["Gene"],
        VARIATION=r["Variation"],
        CLINICAL_TEXT=r["Text"]
    ),
    axis=1
)

df_small["model_text"] = (
    "Gene: " + df_small["Gene"].astype(str) +
    " | Variant: " + df_small["Variation"].astype(str) +
    " | Evidence: " + df_small["Text"].astype(str)
)

#df_small[["ID", "prompt"]].head(1)


# Select only the first 1000 samples for fast experimentation
df_1000 = df.iloc[:10].copy()
df_1000.shape


max_length = 256
stride = 200

embeddings = []

texts = df_small["model_text"].tolist()

for text in tqdm(texts, desc="Encoding clinical texts"):
    
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    last_hidden = out.last_hidden_state
    attn = attention_mask.unsqueeze(-1)

    chunk_embs = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1)
    doc_emb = chunk_embs.mean(dim=0)

    embeddings.append(doc_emb.cpu().numpy())

X = np.vstack(embeddings)
X.shape


idx = 0  # change this index to inspect another sample

print("ID:", df_small.loc[idx, "ID"])
print("Gene:", df_small.loc[idx, "Gene"])
print("Variant:", df_small.loc[idx, "Variation"])
print("\n--- Clinical Evidence ---\n")
print(df_small.loc[idx, "Text"])



idx = 1

print("ID:", df_small.loc[idx, "ID"])
print("Gene:", df_small.loc[idx, "Gene"])
print("Variant:", df_small.loc[idx, "Variation"])
print("\n--- Clinical Evidence ---\n")
print(df_small.loc[idx, "Text"])



idx = 2

print("ID:", df_small.loc[idx, "ID"])
print("Gene:", df_small.loc[idx, "Gene"])
print("Variant:", df_small.loc[idx, "Variation"])

print("\n--- PROMPT GENERATED ---\n")
print(df_small.loc[idx, "prompt"])



idx = 9

print("ID:", df_small.loc[idx, "ID"])
print("Gene:", df_small.loc[idx, "Gene"])
print("Variant:", df_small.loc[idx, "Variation"])

print("\n--- PROMPT GENERATED ---\n")
print(df_small.loc[idx, "prompt"])



import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Dataset paths (Kaggle)
TRAIN_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip"
TRAIN_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_text.zip"
TEST_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/test_variants.zip"
TEST_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/test_text.zip"

# Load training variants
train_variants = pd.read_csv(TRAIN_VARIANTS_PATH)

# Load clinical text (|| separator)
train_text = pd.read_csv(
    TRAIN_TEXT_PATH,
    sep=r"\|\|",
    engine="python",
    names=["ID", "Text"],
    skiprows=1
)

# Merge datasets
train_df = train_variants.merge(train_text, on="ID", how="left")
train_df["Text"] = train_df["Text"].fillna("")

df = train_df.rename(columns={"Text": "Text"})

df_small = df.iloc[:10].reset_index(drop=True)
df_small


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


model_id = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

model.to(device)
model.eval()


prompts = []

for _, row in df_small.iterrows():
    prompt = f"""
You are a biomedical expert.

Extract clinically relevant biomedical concepts
from the following cancer-related text.

Text:
{row["Text"]}
"""
    prompts.append(prompt)


max_length = 512
embeddings = []

for text in tqdm(prompts, desc="Encoding clinical texts"):
    
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)

    last_hidden = outputs.last_hidden_state
    attention_mask = enc["attention_mask"].unsqueeze(-1)

    pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    embeddings.append(pooled.squeeze(0).cpu().numpy())



#
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)

tfidf_matrix = tfidf.fit_transform(df_small["Text"])

# soma dos pesos TF-IDF por termo
tfidf_sum = tfidf_matrix.sum(axis=0).A1

terms = tfidf.get_feature_names_out()

df_keywords_global = (
    pd.DataFrame({
        "keyword": terms,
        "tfidf_score": tfidf_sum
    })
    .sort_values("tfidf_score", ascending=False)
)

df_keywords_global.head(n=10)


idx = 2

row = tfidf_matrix[idx].toarray().flatten()

df_doc_keywords = pd.DataFrame({
    "keyword": terms,
    "score": row
})

df_doc_keywords = df_doc_keywords[df_doc_keywords["score"] > 0]
df_doc_keywords = df_doc_keywords.sort_values("score", ascending=False)

df_doc_keywords.head(n=10)


#
word_freq = dict(
    zip(
        df_doc_keywords["keyword"],
        df_doc_keywords["score"]
    )
)

wc = WordCloud(
    width=900,
    height=400,
    background_color="white"
).generate_from_frequencies(word_freq)

plt.figure(figsize=(12,5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# stopwords biomédicas
biomedical_stopwords = set([
    "fig", "figure", "table", "data", "method", "methods",
    "results", "study", "studies", "analysis", "using",
    "shown", "showed", "performed", "patient", "patients",
    "cells", "cell", "protein", "proteins", "expression",
    "model", "models", "levels", "gene", "genes",
    "et", "al", "eg", "ie", "vs", "wt", "mutant", "mutants",
    "abstract", "background", "conclusion"
])

# CONVERSÃO PARA LIST
custom_stopwords = list(ENGLISH_STOP_WORDS.union(biomedical_stopwords))

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words=custom_stopwords,
    min_df=3,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
)

tfidf_matrix = tfidf.fit_transform(df_small["Text"])
terms = tfidf.get_feature_names_out()
tfidf


#  VISUALIZAR KEYWORDS GLOBAIS (AGORA LIMPOS)
tfidf_sum = tfidf_matrix.sum(axis=0).A1

df_keywords_global = (
    pd.DataFrame({
        "keyword": terms,
        "tfidf_score": tfidf_sum
    })
    .sort_values("tfidf_score", ascending=False)
)

df_keywords_global.head(20)


idx = 2

row = tfidf_matrix[idx].toarray().flatten()

df_doc_keywords = (
    pd.DataFrame({
        "keyword": terms,
        "score": row
    })
    .query("score > 0")
    .sort_values("score", ascending=False)
)

df_doc_keywords.head(15)



agent_keywords = []

for i in range(len(df_small)):
    row = tfidf_matrix[i].toarray().flatten()
    top_idx = row.argsort()[-12:][::-1]
    keywords = [terms[j] for j in top_idx if row[j] > 0]
    agent_keywords.append(", ".join(keywords))

df_small["agent_keywords"] = agent_keywords
df_small[["ID", "Gene", "Variation", "agent_keywords"]].head(n=20)


# List the contents of the Kaggle input directory for the MSK cancer treatment dataset
!ls /kaggle/input/msk-redefining-cancer-treatment

# Create the working directory where the dataset will be extracted
!mkdir -p /kaggle/working/msk

# Unzip the training text files into the working directory
!unzip /kaggle/input/msk-redefining-cancer-treatment/training_text.zip -d /kaggle/working/msk

# Unzip the training variants (genetic mutation data) into the working directory
!unzip /kaggle/input/msk-redefining-cancer-treatment/training_variants.zip -d /kaggle/working/msk

# List the contents of the working directory to verify extraction
# !ls /kaggle/working/msk


# Dataset paths (Kaggle)
TRAIN_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip"
TRAIN_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_text.zip"
TEST_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/test_variants.zip"
TEST_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/test_text.zip"

# Load training variants
train_variants = pd.read_csv(TRAIN_VARIANTS_PATH)

# Load clinical text (|| separator)
train_text = pd.read_csv(
    TRAIN_TEXT_PATH,
    sep=r"\|\|",
    engine="python",
    names=["ID", "Text"],
    skiprows=1
)

# Merge datasets
train_df = train_variants.merge(train_text, on="ID", how="left")
train_df["Text"] = train_df["Text"].fillna("")

df["model_text"] = (
    "Gene: " + df["Gene"].astype(str) + "\n"
    "Variant: " + df["Variation"].astype(str) + "\n\n"
    "Clinical Evidence:\n" + df["Text"].astype(str)
)
df


# Randomly sample 100 rows from the dataframe to create a smaller subset for the agent
df_agent = df.sample(100, random_state=42).reset_index(drop=True)

# Display the shape of the sampled dataframe (number of rows and columns)
df_agent.shape


# System prompt used to guide the LLM in extracting oncology-related therapeutic agents
SYSTEM_PROMPT = """
You are a senior clinical oncologist with expertise in targeted cancer therapies.

Your task is to analyze clinical and scientific text related to cancer genetics
and identify therapeutic agents mentioned in the evidence.

Focus exclusively on:
- Drug names
- Targeted therapies
- Small molecule inhibitors
- Monoclonal antibodies
- Experimental or investigational agents

Strict rules:
- Do NOT infer, assume, or hallucinate drugs.
- Only extract agents that are explicitly mentioned in the text.
- If no drugs are found, you MUST explain why, based on the content of the text.

Clinical Context:
Gene: {{GENE}}
Variant: {{VARIATION}}

Clinical Evidence:
{{CLINICAL_TEXT}}

Output requirements:
- Return a JSON object
- Always include a short clinical justification
- If no drugs are mentioned, clearly state that the text discusses biological mechanisms only

Return strictly in the following JSON format:

{
  "drugs_found": [],
  "confidence": "high | medium | low",
  "clinical_rationale": "Concise explanation grounded in the text"
}
"""


# Build the prompt for the agent by combining the system prompt with
# gene, variant, and clinical evidence text from the dataframe
df_agent["drug_prompt"] = (
    SYSTEM_PROMPT
    + "\n\nClinical Context:\n"
    # Insert gene information for each row
    + "Gene: " + df_agent["Gene"].astype(str) + "\n"
    # Insert variant information for each row
    + "Variant: " + df_agent["Variation"].astype(str) + "\n\n"
    # Insert the clinical evidence text
    + "Clinical Evidence:\n"
    + df_agent["Text"].astype(str)
    + "\n\nInstructions:\n"
    # Explicit extraction and normalization rules
    + "- Extract drug names exactly as written\n"
    + "- Normalize names when applicable\n"
    + "- Identify drug class if known\n"
    + "- Indicate if the drug is experimental\n"
    + "- If no drugs are mentioned, state that clearly\n\n"
    # Enforce strict JSON-only output
    + "Return ONLY a valid JSON."
)


# Function to dynamically build a prompt for each dataframe row
# combining the system prompt with gene, variant, and clinical text
def build_prompt(row):
    return f"""
{SYSTEM_PROMPT}

Task:
Analyze the genetic variant and extract clinically relevant information.

Gene: {row['Gene']}
Variant: {row['Variation']}

Clinical Evidence:
{row['Text']}

Instructions:
- Identify oncogenic relevance
- Identify therapeutic implications
- Extract drug names if mentioned
- Extract key molecular pathways
- If no drug is mentioned, state "No therapeutic agents reported"

Return ONLY a JSON with:
- oncogenicity
- drugs
- pathways
- uncertainty
"""

# Regular expression pattern used as a fallback mechanism
# to detect commonly used targeted cancer therapies in the text
DRUG_PATTERN = r"\b(tamoxifen|gefitinib|erlotinib|afatinib|osimertinib|crizotinib|imatinib|sunitinib|vemurafenib|dabrafenib|trametinib)\b"

# Apply the fallback drug extraction:
# - Convert text to lowercase
# - Search for drug names using the predefined regex pattern
# - Remove duplicates by converting to a set
df_agent["agent_drugs_fallback"] = df_agent["Text"].str.lower().apply(
    lambda x: list(set(re.findall(DRUG_PATTERN, x)))
)


# Initialize the column that will store the LLM-style drug extraction results
df_agent["agent_drugs_llm_json"] = None

# Iterate over all rows and apply the fallback drug extraction logic
for i in tqdm(range(len(df_agent)), desc="Applying drug prompt"):
    # Retrieve drugs extracted via regex fallback
    drugs = df_agent.loc[i, "agent_drugs_fallback"]

    # If no drugs were found, populate a standardized empty result
    if len(drugs) == 0:
        df_agent.at[i, "agent_drugs_llm_json"] = {
            "drugs_found": [],
            "confidence": "high",
            "notes": "No therapeutic agents explicitly mentioned."
        }
    # If drugs were found, store them with high confidence
    else:
        df_agent.at[i, "agent_drugs_llm_json"] = {
            "drugs_found": drugs,
            "confidence": "high",
            "notes": "Extracted via explicit text matching."
        }

# Import JSON library for parsing serialized outputs
import json

# Function to safely parse the drug JSON output
def parse_drug_json(x):
    # Handle empty or NaN values
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return [], None

    # If the input is already a dictionary, extract fields directly
    if isinstance(x, dict):
        return x.get("drugs_found", []), x.get("confidence", None)

    # If the input is a JSON-formatted string, deserialize it
    if isinstance(x, str):
        data = json.loads(x)
        return data.get("drugs_found", []), data.get("confidence", None)

    # Safety fallback for unexpected formats
    return [], None

# Expand the parsed results into separate dataframe columns
df_agent[["drugs_found", "confidence"]] = (
    df_agent["agent_drugs_llm_json"]
    .apply(lambda x: pd.Series(parse_drug_json(x)))
)


# Filter the dataframe to keep only rows where at least one drug was identified
df_with_drugs = df_agent[df_agent["drugs_found"].apply(len) > 0]

# Explode the list of drugs so that each drug appears in a separate row
df_exploded = df_with_drugs.explode("drugs_found")

# Display the first 20 rows with gene, variant, and extracted drug information
df_exploded[["Gene", "Variation", "drugs_found"]].head(n=20)


# Count the frequency of each extracted drug across all records
drug_counts = (
    df_exploded["drugs_found"]
    .value_counts()
    .reset_index()
    # Rename columns for clarity
    .rename(columns={"index": "drug", "drugs_found": "count"})
)

# Display the drug frequency table
drug_counts


# Concatenate all extracted drug names into a single text string
drug_text = " ".join(
    df_exploded["drugs_found"]
    .dropna()
    .astype(str)
    .tolist()
)

# Create a WordCloud visualization for the extracted drug mentions
wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="plasma"
).generate(drug_text)

# Plot the WordCloud
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Drug Mentions WordCloud")
plt.show()


# Compute a gene × drug occurrence matrix
# Rows represent genes and columns represent drugs
# Values indicate how many times each drug is mentioned per gene
gene_drug_matrix = (
    df_exploded
    .groupby(["Gene", "drugs_found"])
    .size()
    .unstack(fill_value=0)
)

# Calculate the Pearson correlation matrix between drugs
# This measures co-occurrence patterns across genes
drug_corr = gene_drug_matrix.corr(method="pearson")

# Plot the drug × drug correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    drug_corr,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

# Add title and adjust layout for better readability
plt.title("Drug × Drug Correlation (Pearson)")
plt.tight_layout()
plt.show()


# Group the data by class and concatenate the keywords text for each group
grouped_data = (
    df_small
    .groupby("Class")["agent_keywords"]
    .apply(lambda x: " ".join(x))
)

# Generate a word cloud for each class
for class_id, keywords in grouped_data.items():
    # Create the word cloud for the current class
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords
    ).generate(keywords)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for Class {class_id}", fontsize=16)
    plt.axis("off")
    plt.show()


# System prompt that guides the LLM to strictly extract explicitly mentioned
# therapeutic agents from clinical oncology text
SYSTEM_PROMPT = """
You are a senior clinical oncologist with expertise in precision oncology.

Your task is to extract therapeutic agents ONLY if they are explicitly mentioned
in the clinical text and provide the exact textual evidence.

Rules:
- Do NOT infer or hallucinate drugs
- ONLY extract drugs explicitly written
- ALWAYS show the exact sentence mentioning the drug
- If no drug is mentioned, explain why

Return JSON only.
"""

# List of known targeted therapies and oncology drugs
# used for rule-based matching and validation
DRUG_LIST = [
    "tamoxifen",
    "gefitinib",
    "erlotinib",
    "afatinib",
    "osimertinib",
    "crizotinib",
    "imatinib",
    "sunitinib",
    "vemurafenib",
    "dabrafenib",
    "trametinib"
]

# Compile a case-insensitive regular expression pattern
# to detect explicit mentions of drugs in the clinical text
DRUG_PATTERN = re.compile(
    r"\b(" + "|".join(DRUG_LIST) + r")\b",
    flags=re.IGNORECASE
)


def drug_extraction_agent(gene, variant, text):
    # List to store all detected drug mentions with evidence
    matches = []
    
    # Split the clinical text into simple sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Iterate over each sentence to search for explicit drug mentions
    for sent in sentences:
        found = DRUG_PATTERN.findall(sent)
        for drug in found:
            matches.append({
                # Normalize drug name to lowercase
                "drug_name": drug.lower(),
                # Drug class is not inferred in this rule-based approach
                "drug_class": None,
                # Store the exact sentence as textual evidence
                "text_evidence": sent.strip()
            })

    # If no drugs were found, return a standardized empty response
    if len(matches) == 0:
        return {
            "drugs_found": [],
            "confidence": "high",
            "clinical_rationale": (
                "The clinical evidence discusses molecular or biological mechanisms "
                "without explicitly mentioning therapeutic agents."
            )
        }

    # If drugs were found, return them with supporting evidence
    return {
        "drugs_found": matches,
        "confidence": "high",
        "clinical_rationale": (
            "Therapeutic agents were explicitly identified and supported "
            "by direct textual evidence."
        )
    }


# Initialize the column that will store the drug extraction results
df_agent["agent_drug_json"] = None

# Run the drug extraction agent for each row in the dataframe
for i in tqdm(range(len(df_agent)), desc="Running drug extraction agent"):
    row = df_agent.loc[i]
    df_agent.at[i, "agent_drug_json"] = drug_extraction_agent(
        gene=row["Gene"],
        variant=row["Variation"],
        text=row["Text"]
    )

# List to store flattened records for downstream analysis
records = []

# Convert the nested JSON outputs into a flat table structure
for _, row in df_agent.iterrows():
    data = row["agent_drug_json"]
    
    # Skip rows with no identified drugs
    if len(data["drugs_found"]) == 0:
        continue
        
    # Create one record per drug mention with its textual evidence
    for d in data["drugs_found"]:
        records.append({
            "Gene": row["Gene"],
            "Variant": row["Variation"],
            "Drug": d["drug_name"],
            "Text_Evidence": d["text_evidence"]
        })


# Create a dataframe from the extracted drug–gene–variant records
df_drug_text = pd.DataFrame(records)

# Select the index of the row to inspect
idx = 0

# Retrieve the selected row
row = df_drug_text.iloc[idx]

# Display the extracted information in a readable format
print("Gene:", row["Gene"])
print("Variant:", row["Variant"])
print("Drug:", row["Drug"])
print("\n--- TEXT EVIDENCE EXTRACTED BY AGENT ---\n")
print(row["Text_Evidence"])


# Create a dataframe from the extracted drug–gene–variant records
df_drug_text = pd.DataFrame(records)

# Select the index of the row to inspect
idx = 1

# Retrieve the selected row
row = df_drug_text.iloc[idx]

# Print the extracted information for manual inspection
print("Gene:", row["Gene"])
print("Variant:", row["Variant"])
print("Drug:", row["Drug"])
print("\n--- TEXT EVIDENCE EXTRACTED BY AGENT ---\n")
print(row["Text_Evidence"])


# Create a dataframe from the extracted drug–gene–variant records
df_drug_text = pd.DataFrame(records)

# Inspect record at index 2
idx = 2
row = df_drug_text.iloc[idx]

print("Gene:", row["Gene"])
print("Variant:", row["Variant"])
print("Drug:", row["Drug"])
print("\n--- TEXT EVIDENCE EXTRACTED BY AGENT ---\n")
print(row["Text_Evidence"])



# Function to clean and normalize text:
# - Convert to lowercase
# - Remove non-alphabetic characters
# - Normalize multiple spaces
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Create a dataframe from the extracted records
df_records = pd.DataFrame(records)

# Clean the extracted text evidence
df_records["clean_text"] = df_records["Text_Evidence"].apply(clean_text)


# Get the unique list of drugs and count them
drugs = sorted(df_records["Drug"].unique())
n_drugs = len(drugs)

# Automatically define the subplot layout
n_cols = 3
n_rows = math.ceil(n_drugs / n_cols)


# Create the figure and axes for the word clouds
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(6 * n_cols, 4 * n_rows)
)

# Flatten axes array for easier iteration
axes = axes.flatten()


# Generate a word cloud for each drug
for idx, drug in enumerate(drugs):
    # Concatenate all cleaned text evidence related to the current drug
    drug_text = " ".join(
        df_records.loc[df_records["Drug"] == drug, "clean_text"]
    )

    # Create the word cloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100
    ).generate(drug_text)

    # Plot the word cloud
    axes[idx].imshow(wc, interpolation="bilinear")
    axes[idx].set_title(f"Drug: {drug}", fontsize=14)
    axes[idx].axis("off")


# Turn off any unused subplot axes
for j in range(idx + 1, len(axes)):
    axes[j].axis("off")

# Adjust layout spacing and display the figure
plt.tight_layout()
plt.show()


# PyTorch for tensor operations and model execution
import torch

# Hugging Face Transformers for tokenization, models, and pipelines
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Scikit-learn utilities for multilabel classification and preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# Linear and probabilistic models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB

# Tree-based and ensemble models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

# Support Vector Machines
from sklearn.svm import LinearSVC

# Evaluation metrics for multilabel classification
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    roc_auc_score
)

# XGBoost classifier for gradient boosting with decision trees
from xgboost import XGBClassifier

# Progress bar utility for loops
from tqdm import tqdm



# Dataset paths (Kaggle)
TRAIN_VARIANTS_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip"
TRAIN_TEXT_PATH = "/kaggle/input/msk-redefining-cancer-treatment/training_text.zip"

# Load training variants data (gene, variation, class labels)
train_variants = pd.read_csv(TRAIN_VARIANTS_PATH)

# Load clinical text data
# The file uses '||' as a separator between ID and Text
train_text = pd.read_csv(
    TRAIN_TEXT_PATH,
    sep=r"\|\|",
    engine="python",
    names=["ID", "Text"],
    skiprows=1
)

# Merge variants and clinical text using the ID column
train_df = train_variants.merge(train_text, on="ID", how="left")

# Replace missing clinical text with empty strings
train_df["Text"] = train_df["Text"].fillna("")

# Rename columns if needed (kept explicit for pipeline consistency)
df = train_df.rename(columns={"Text": "Text"})

# Display the final merged dataframe
df



# Hugging Face model identifier for Bio_ClinicalBERT
MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"

# Select computation device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer associated with the pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the pretrained Bio_ClinicalBERT model and move it to the selected device
model = AutoModel.from_pretrained(MODEL_ID).to(device)

# Create a Named Entity Recognition (NER) pipeline
# This pipeline will be used to extract biomedical entities from clinical text
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Set the model to evaluation mode (disables dropout and training-specific layers)
model.eval()


# Ensure that the 'drugs_found' column always contains a list
# If the value is not a list, replace it with an empty list
df_agent["drugs_found"] = df_agent["drugs_found"].apply(
    lambda x: x if isinstance(x, list) else []
)



# Regular expression pattern to identify explicitly mentioned oncology drugs
DRUG_PATTERN = (
    r"\b("
    r"imatinib|gefitinib|erlotinib|afatinib|osimertinib|"
    r"crizotinib|sunitinib|vemurafenib|dabrafenib|"
    r"trametinib|tamoxifen"
    r")\b"
)

# Extract drug mentions from the clinical text:
# - Convert text to lowercase
# - Find all matching drug names using the regex pattern
# - Remove duplicates by converting to a set
df["drugs_found"] = (
    df["Text"]
    .str.lower()
    .apply(lambda x: list(set(re.findall(DRUG_PATTERN, x))))
)

# Keep only samples that contain at least one drug mention
# These samples are suitable for supervised learning
df_ml = df[df["drugs_found"].map(len) > 0].reset_index(drop=True)

# Initialize the MultiLabelBinarizer to convert drug lists into binary vectors
mlb = MultiLabelBinarizer()

# Transform the list of drugs into a multilabel binary matrix
y = mlb.fit_transform(df_ml["drugs_found"])

# Retrieve the drug class names corresponding to each binary column
drug_classes = mlb.classes_
drug_classes


# Generate text embeddings using mean pooling over token embeddings
max_length = 256
embeddings = []

# Extract clinical texts used for embedding generation
texts = df_ml["Text"].tolist()

# Iterate over each text and generate embeddings with ClinicalBERT
for text in tqdm(texts, desc="Generating ClinicalBERT embeddings"):
    # Tokenize the text with truncation and padding
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move tensors to the selected device (CPU or GPU)
    enc = {k: v.to(device) for k, v in enc.items()}

    # Disable gradient computation for inference
    with torch.no_grad():
        out = model(**enc)

    # Retrieve the last hidden state (token-level embeddings)
    last_hidden = out.last_hidden_state

    # Expand attention mask to match embedding dimensions
    mask = enc["attention_mask"].unsqueeze(-1)

    # Apply mean pooling, considering only valid (non-padded) tokens
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    # Store the pooled embedding on CPU
    embeddings.append(pooled.cpu().numpy())

# Stack all embeddings into a single feature matrix
X = np.vstack(embeddings)


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Perform an 80/20 train–test split
# X contains the ClinicalBERT embeddings
# y contains the multilabel drug targets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Retrieve the list of drug class names learned by the previous binarizer
drug_names = mlb.classes_

# Initialize a new MultiLabelBinarizer
# This will convert lists of drugs into a multilabel binary matrix
mlb = MultiLabelBinarizer()

# Transform the 'drugs_found' column into a multilabel target matrix
# Each column represents a drug and each row indicates its presence (1) or absence (0)
y_multilabel = mlb.fit_transform(df_agent["drugs_found"])

# Display the fitted MultiLabelBinarizer object
mlb


# Display the shape of the multilabel target matrix
print(y_multilabel.shape)

# Display the list of drug classes learned by the binarizer
print(mlb.classes_)

# Compute support (number of samples) for each drug class
support = y.sum(axis=0)

# Create a dataframe with drug names and their corresponding support
# and sort it by descending frequency
pd.DataFrame({
    "drug": drug_names,
    "support": support
}).sort_values("support", ascending=False)


from sklearn.model_selection import KFold

# Initialize K-Fold cross-validation with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store F1 scores for each fold
micro_scores = []
macro_scores = []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")

    # Split features and labels into training and validation sets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Standardize features (fit on training, apply to validation)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Define a One-vs-Rest logistic regression model for multilabel classification
    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.5,
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced"
        )
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities for the validation set
    y_proba = model.predict_proba(X_val)

    # Apply a fixed threshold to obtain binary predictions
    y_pred = (y_proba >= 0.5).astype(int)

    # Compute Micro and Macro F1-scores
    micro = f1_score(y_val, y_pred, average="micro")
    macro = f1_score(y_val, y_pred, average="macro")

    # Store scores for later aggregation
    micro_scores.append(micro)
    macro_scores.append(macro)

    # Print fold-level results
    print(f"Micro F1: {micro:.3f}")
    print(f"Macro F1: {macro:.3f}")

# Print final averaged results across all folds
print("\n=== FINAL RESULTS ===")
print(f"Micro F1 (mean): {np.mean(micro_scores):.3f}")
print(f"Macro F1 (mean): {np.mean(macro_scores):.3f}")


# Hyperparameter grid definition
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Regularization strength values to be tested
C_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

# Regularization penalties supported by the liblinear solver
penalties = ["l1", "l2"]

# Generate all combinations of hyperparameters
param_combinations = [
    (C, penalty) for C in C_values for penalty in penalties
]

# List to store aggregated results for each hyperparameter setting
results = []

# Iterate over all hyperparameter combinations
for C, penalty in tqdm(param_combinations, desc="Hyperparameter tuning"):

    # Lists to store metrics across folds
    f1_micro_scores = []
    f1_macro_scores = []
    auc_scores = []

    # Perform K-Fold cross-validation
    for train_idx, val_idx in tqdm(
        kf.split(X),
        total=kf.get_n_splits(),
        desc=f"C={C} | penalty={penalty}",
        leave=False
    ):

        # Split features and labels into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features (fit on training data only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Define a One-vs-Rest Logistic Regression model for multilabel classification
        model = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                penalty=penalty,
                solver="liblinear",
                max_iter=3000,
                class_weight="balanced"
            )
        )

        # Train the model
        model.fit(X_train, y_train)

        # Generate binary predictions
        y_pred = model.predict(X_val)

        # Generate probability estimates for ROC-AUC computation
        y_proba = model.predict_proba(X_val)

        # Compute evaluation metrics for the current fold
        f1_micro_scores.append(f1_score(y_val, y_pred, average="micro"))
        f1_macro_scores.append(f1_score(y_val, y_pred, average="macro"))
        auc_scores.append(
            roc_auc_score(y_val, y_proba, average="micro")
        )

    # Store mean metrics across all folds for the current hyperparameters
    results.append({
        "C": C,
        "penalty": penalty,
        "f1_micro": np.mean(f1_micro_scores),
        "f1_macro": np.mean(f1_macro_scores),
        "roc_auc_micro": np.mean(auc_scores)
    })

# Convert results to a dataframe and sort by Micro F1-score
results_df = pd.DataFrame(results)
results_df.sort_values("f1_micro", ascending=False)


BEST_C = 0.3
BEST_PENALTY = "l2"


# Apply global feature scaling to the entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the final One-vs-Rest Logistic Regression model
# using the best hyperparameters found during tuning
logreg_model = OneVsRestClassifier(
    LogisticRegression(
        C=BEST_C,
        penalty=BEST_PENALTY,
        solver="liblinear",
        max_iter=3000,
        class_weight="balanced"
    )
)

# Train the final model on the full scaled dataset
logreg_model.fit(X_scaled, y)


# Generate predictions using the trained final model
y_pred = logreg_model.predict(X_scaled)

# Generate probability estimates for each drug class
y_proba = logreg_model.predict_proba(X_scaled)


# Evaluate the final model performance on the full dataset

print("F1 Micro        :", f1_score(y, y_pred, average="micro"))
print("F1 Macro        :", f1_score(y, y_pred, average="macro"))

# Hamming accuracy is defined as 1 - Hamming loss
print("Hamming Accuracy:", 1 - hamming_loss(y, y_pred))

# Compute Micro-averaged ROC AUC score using predicted probabilities
print("ROC AUC (Micro) :", roc_auc_score(y, y_proba, average="micro"))


# Print a detailed classification report for each drug class
# Includes precision, recall, F1-score, and support
print(
    classification_report(
        y,
        y_pred,
        target_names=drug_names,
        zero_division=0
    )
)


# Generate predicted probabilities for each drug class
y_proba = logreg_model.predict_proba(X_scaled)

# Define a range of thresholds to evaluate
thresholds = np.linspace(0.1, 0.9, 81)

# Dictionary to store the best threshold per drug
best_thresholds = {}

# Optimize the decision threshold for each drug independently
for i, drug in enumerate(drug_names):
    best_f1 = 0.0
    best_t = 0.5

    # Evaluate F1-score across different thresholds
    for t in thresholds:
        y_pred_t = (y_proba[:, i] >= t).astype(int)
        f1 = f1_score(y[:, i], y_pred_t, zero_division=0)

        # Update the best threshold if F1-score improves
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    # Store the optimal threshold and corresponding metrics
    best_thresholds[drug] = {
        "threshold": best_t,
        "f1": best_f1,
        "support": int(y[:, i].sum())
    }

# Convert the results into a dataframe for easier analysis
threshold_df = (
    pd.DataFrame(best_thresholds)
    .T
    .reset_index()
    .rename(columns={"index": "drug"})
    .sort_values("support", ascending=False)
)

# Display the optimized threshold table
threshold_df


# Top-K recommendation analysis (Top-3 / Top-5)
# Clinical objective:
# Even if the model does not explicitly activate a drug (binary prediction),
# does it still appear among the most probable candidates?

# Function to retrieve the Top-K drug recommendations for a single sample
def top_k_recommendations(proba_row, drug_names, k=3):
    # Sort probabilities in descending order and select the top K indices
    idx = np.argsort(proba_row)[::-1][:k]
    # Return the drug names with their corresponding probabilities
    return [(drug_names[i], proba_row[i]) for i in idx]

# Function to compute Top-K Recall for multilabel classification
def top_k_recall(y_true, y_proba, k):
    hits = 0
    total = 0

    # Iterate over all samples
    for i in range(len(y_true)):
        # Indices of true positive labels for the current sample
        true_labels = np.where(y_true[i] == 1)[0]
        if len(true_labels) == 0:
            continue

        # Indices of the Top-K predicted probabilities
        topk_idx = np.argsort(y_proba[i])[::-1][:k]

        # Count how many true labels are recovered in the Top-K predictions
        hits += len(set(true_labels) & set(topk_idx))
        total += len(true_labels)

    # Return recall as the proportion of recovered true labels
    return hits / total if total > 0 else 0.0

# Compute and display Top-K Recall for different values of K
for k in [1, 3, 5]:
    print(f"Top-{k} Recall:", top_k_recall(y, y_proba, k))


# Clinical error analysis (false negatives)
# Objective: Identify which drugs the model most frequently fails to detect

# Initialize a prediction matrix using drug-specific optimized thresholds
y_pred_custom = np.zeros_like(y)

# Apply the best threshold for each drug independently
for i, drug in enumerate(drug_names):
    t = best_thresholds[drug]["threshold"]
    y_pred_custom[:, i] = (y_proba[:, i] >= t).astype(int)

# Analyze false negatives per drug
from sklearn.metrics import confusion_matrix

error_analysis = []

# Compute confusion-matrix-based error metrics for each drug
for i, drug in enumerate(drug_names):
    tn, fp, fn, tp = confusion_matrix(
        y[:, i],
        y_pred_custom[:, i]
    ).ravel()

    error_analysis.append({
        "drug": drug,
        "false_negatives": fn,
        "false_positives": fp,
        "support": int(y[:, i].sum()),
        # False negative rate: proportion of missed true positives
        "fn_rate": fn / (fn + tp) if (fn + tp) > 0 else 0
    })

# Create a dataframe and rank drugs by highest false negative rate
error_df = (
    pd.DataFrame(error_analysis)
    .sort_values("fn_rate", ascending=False)
)

# Display the error analysis table
error_df


# Flatten the feature matrix if needed
# This ensures the input has shape (n_samples, n_features)
X_flat = X.reshape(X.shape[0], -1)

# Multilabel binary target matrix with shape (n_samples, n_drugs)
y = y

# Import evaluation metrics for multilabel classification
from sklearn.metrics import (
    classification_report,  # Precision, recall, F1-score per class
    confusion_matrix,       # Confusion matrix for binary classification
    roc_curve,              # ROC curve computation
    auc                     # Area Under the ROC Curve
)



# Define a set of multilabel classification models using One-vs-Rest strategy
models = {

    # Naive Bayes classifier (Gaussian assumption)
    "NaiveBayes": OneVsRestClassifier(GaussianNB()),
    
    # Logistic Regression with L2 regularization and class balancing
    "LogisticRegression": OneVsRestClassifier(
        LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced"
        )
    ),

    # Stochastic Gradient Descent classifier with logistic loss
    "SGDClassifier": OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=3000,
            class_weight="balanced"
        )
    ),

    # Decision Tree classifier with depth control and class balancing
    "DecisionTree": OneVsRestClassifier(
        DecisionTreeClassifier(
            max_depth=25,
            class_weight="balanced",
            random_state=42
        )
    ),

    # Random Forest ensemble with multiple trees and parallel execution
    "RandomForest": OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        )
    ),

    # XGBoost classifier with gradient boosting and subsampling
    "XGBoost": OneVsRestClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
    )
}

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List to store aggregated evaluation results
results = []

# Train and evaluate each model using cross-validation
for model_name, model in models.items():

    f1_micro_scores = []
    f1_macro_scores = []
    hamming_scores = []
    auc_scores = []

    print(f"\nTraining model: {model_name}")

    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(
        tqdm(kf.split(X), total=5)
    ):

        # Split data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply feature scaling (flattening included for safety)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(X_train.shape[0], -1)
        )
        X_val = scaler.transform(
            X_val.reshape(X_val.shape[0], -1)
        )

        # Train the model
        model.fit(X_train, y_train)

        # Generate predictions
        y_pred = model.predict(X_val)

        # Compute ROC-AUC if probability estimates are available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)
            auc_score = roc_auc_score(
                y_val, y_proba, average="micro"
            )
        else:
            auc_score = np.nan

        # Store evaluation metrics
        f1_micro_scores.append(
            f1_score(y_val, y_pred, average="micro")
        )
        f1_macro_scores.append(
            f1_score(y_val, y_pred, average="macro")
        )
        hamming_scores.append(
            1 - hamming_loss(y_val, y_pred)
        )
        auc_scores.append(auc_score)

    # Aggregate mean metrics across folds
    results.append({
        "model": model_name,
        "f1_micro": np.mean(f1_micro_scores),
        "f1_macro": np.mean(f1_macro_scores),
        "hamming_accuracy": np.mean(hamming_scores),
        "roc_auc_micro": np.nanmean(auc_scores)
    })

# Create a results dataframe and sort by Micro F1-score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("f1_micro", ascending=False)

# Display model comparison results
results_df


# Flatten the feature matrix (if not already flat)
X_flat = X.reshape(X.shape[0], -1)

# Generate feature names for each embedding dimension
feature_names = [f"emb_{i}" for i in range(X_flat.shape[1])]


def get_feature_importance(model, model_name):
    """
    Extract feature importance values from different model types
    wrapped with One-vs-Rest strategy.
    """

    # Logistic Regression / SGD (One-vs-Rest with linear coefficients)
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "coef_"):
        # Stack coefficients from all binary classifiers
        coefs = np.abs(np.vstack([est.coef_ for est in model.estimators_]))
        # Average importance across classes
        importance = coefs.mean(axis=0)

    # Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
    elif hasattr(model.estimators_[0], "feature_importances_"):
        importances = []
        for est in model.estimators_:
            importances.append(est.feature_importances_)
        # Average importance across classes
        importance = np.mean(importances, axis=0)

    # XGBoost models (feature_importances_ attribute)
    elif hasattr(model.estimators_[0], "feature_importances_"):
        importances = [est.feature_importances_ for est in model.estimators_]
        importance = np.mean(importances, axis=0)

    # Models that do not support feature importance
    else:
        return None

    return importance


# Number of top features to visualize
TOP_K = 50

# Plot Feature Importance for each trained model
for model_name, model in models.items():

    print(f"\nFeature Importance — {model_name}")

    # Extract feature importance values
    importance = get_feature_importance(model, model_name)

    if importance is None:
        print("Feature importance not supported for this model.")
        continue

    # Select the Top-K most important features
    top_idx = np.argsort(importance)[-TOP_K:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importance[top_idx]

    # Plot the feature importance bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=top_values,
        y=top_features,
        palette="viridis"
    )
    plt.title(f"Top-{TOP_K} Feature Importance — {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Embedding Dimension")
    plt.tight_layout()
    plt.show()


# Evaluate each trained model on the full scaled dataset
for model_name, model in models.items():

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    # Generate binary predictions for all samples
    y_pred = model.predict(X_scaled)

    # Generate probability estimates if the model supports it
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)
    else:
        y_proba = None

    # Print a detailed classification report for each drug class
    print("\nClassification Report:\n")
    print(
        classification_report(
            y,
            y_pred,
            target_names=drug_names,
            digits=4
        )
    )


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Iterate over each trained model
for model_name, model in models.items():

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    # Generate predictions for the full scaled dataset
    y_pred = model.predict(X_scaled)

    # Loop over each drug (multilabel setting)
    for i, drug in enumerate(drug_names):

        # Compute the confusion matrix for the current drug
        # Binary classification: [ [TN, FP],
        #                           [FN, TP] ]
        cm = confusion_matrix(y[:, i], y_pred[:, i])

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False
        )

        # Add plot labels and title
        plt.title(f"{model_name} — Confusion Matrix ({drug})")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()


from sklearn.metrics import roc_curve, auc

# Plot ROC curves for each trained model
for model_name, model in models.items():

    print("\n" + "=" * 80)
    print(f"ROC CURVES — MODEL: {model_name}")
    print("=" * 80)

    # Some models do not support probability prediction
    if not hasattr(model, "predict_proba"):
        print("Model does not support predict_proba — skipping ROC analysis")
        continue

    # Generate predicted probabilities
    y_proba = model.predict_proba(X_scaled)

    # MICRO-AVERAGE ROC CURVE (global multilabel performance)
    fpr_micro, tpr_micro, _ = roc_curve(
        y.ravel(),
        y_proba.ravel()
    )
    auc_micro = auc(fpr_micro, tpr_micro)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"Micro-average ROC (AUC = {auc_micro:.3f})",
        linewidth=3,
        color="black"
    )

    # ROC CURVES PER DRUG (one-vs-rest)
    for i, drug in enumerate(drug_names):

        # Skip drugs with no positive samples
        if y[:, i].sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y[:, i], y_proba[:, i])
        auc_score = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            alpha=0.4,
            label=f"{drug} (AUC = {auc_score:.2f})"
        )

    # Random baseline
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

    # Plot configuration
    plt.title(f"ROC Curves — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




