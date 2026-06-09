# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Multi-View Transformer Model Definition
class MultiViewTransformer(nn.Module):
    def __init__(self, model_name_prompt, model_name_response, num_classes=2):
        # Initialize the MultiViewTransformer model
        super(MultiViewTransformer, self).__init__()

        # Load separate transformer models for the prompt and responses
        self.prompt_encoder = AutoModel.from_pretrained(model_name_prompt)
        self.response_encoder = AutoModel.from_pretrained(model_name_response)

        # Define a cross-attention layer to compare the two responses
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Define a fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(768 * 3, 512),  # Combine embeddings from prompt, response_a, and attention
            nn.ReLU(),                # Apply ReLU activation
            nn.Dropout(0.2),          # Add dropout for regularization
            nn.Linear(512, num_classes)  # Map to output classes
        )

    def forward(self, prompt, response_a, response_b):
        # Encode the prompt using the prompt_encoder
        prompt_emb = self.prompt_encoder(**prompt).last_hidden_state  # [batch, seq_len, 768]

        # Encode the responses using the response_encoder
        response_a_emb = self.response_encoder(**response_a).last_hidden_state  # [batch, seq_len, 768]
        response_b_emb = self.response_encoder(**response_b).last_hidden_state  # [batch, seq_len, 768]

        # Compute cross-attention between the two response embeddings
        attn_output, _ = self.cross_attention(
            response_a_emb,  # Query: response A (keeping seq_len)
            response_b_emb,  # Key: response B (keeping seq_len)
            response_b_emb   # Value: response B (keeping seq_len)
        )

        # Reduce sequence length by taking the mean across tokens
        prompt_emb = prompt_emb.mean(dim=1)  # [batch, 768]
        response_a_emb = response_a_emb.mean(dim=1)  # [batch, 768]
        attn_output = attn_output.mean(dim=1)  # [batch, 768]

        # Concatenate the embeddings from prompt, response_a, and attention output
        combined_emb = torch.cat([prompt_emb, response_a_emb, attn_output], dim=1)  # [batch, 768 * 3]

        # Pass the combined embeddings through the fully connected layer
        logits = self.fc(combined_emb)
        return logits



# Example usage of the Multi-View Transformer
if __name__ == "__main__":
    # Load tokenizers for the transformers
    prompt_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    response_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Example input data
    prompt_text = ["What is the capital of France?"]
    response_a_text = ["The capital of France is Paris."]
    response_b_text = ["France's capital city is Paris."]

    # Tokenize the prompt and responses
    prompt = prompt_tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    response_a = response_tokenizer(response_a_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    response_b = response_tokenizer(response_b_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Initialize the Multi-View Transformer model
    model = MultiViewTransformer("xlm-roberta-base", "xlm-roberta-base")

    # Perform a forward pass through the model
    logits = model(prompt, response_a, response_b)

    # Logits contain the prediction scores for response_a and response_b
    print("Logits:", logits)

    # Determine the preferred response based on the logits
    predicted_class = torch.argmax(logits, dim=1)
    print("Predicted class:", predicted_class)


