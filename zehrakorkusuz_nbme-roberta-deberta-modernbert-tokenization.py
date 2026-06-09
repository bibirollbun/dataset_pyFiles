!pip uninstall transformers torch torchvision -q -y 
!pip install transformers datasets torch torchvision -q


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from transformers import AutoTokenizer
from typing import List, Tuple


patient_notes = pd.read_csv("/kaggle/input/nbme-score-clinical-patient-notes/patient_notes.csv")
features = pd.read_csv("/kaggle/input/nbme-score-clinical-patient-notes/features.csv")
train = pd.read_csv("/kaggle/input/nbme-score-clinical-patient-notes/train.csv")

# merge dataset

train = pd.merge(train, patient_notes, on=["pn_num", "case_num"], how="left")
train = pd.merge(train, features, on=["feature_num", "case_num"], how="left")


train.head()


sample_text = train.iloc[0]["pn_history"]
sample_text


sample_text[:40]


def analyze_tokens_with_spaces(text: str, model_name: str, tokenizer_path: str) -> List[Tuple[Tuple[int, int], str]]:
    """Show exact token spans with spaces replaced from special markers."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Get encoding with offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    # Create list of tuples with spans and processed token representation
    token_spans = []
    for token, (start, end) in zip(tokens, offsets):
        # Replace special markers with spaces
        if token.startswith('Ġ'):
            token = ' ' + token[1:]
        elif token.startswith('▁'):
            token = ' ' + token[1:]
        token_spans.append(((start, end), repr(token)))
    
    return token_spans

def print_token_analysis(text: str):
    """Print token analysis for each model."""
    models = {
        "BERT": "google-bert/bert-base-uncased",
        "RoBERTa": "FacebookAI/roberta-base",
        "DeBERTa": "microsoft/deberta-v3-base",
        "ModernBERT": "answerdotai/ModernBERT-base"
    }
    
    for model_name, model_path in models.items():
        token_spans = analyze_tokens_with_spaces(text, model_name, model_path)
        
        print(f"\n{model_name} token spans:")
        print("[")
        for (start, end), token in token_spans:
            # Ensure consistent spacing in output
            print(f"  [{start:>2}, {end:>2}), {token},")
        print("]")


text = "HPI: 17yo M presents with palpitations." #sample_text[:40]
print_token_analysis(text)

