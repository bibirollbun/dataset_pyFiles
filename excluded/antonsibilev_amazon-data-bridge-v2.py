import requests
import uuid
import json
import math
from openai import OpenAI
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

OpenAI_key = user_secrets.get_secret("OpenAI_key")

class OpenAIClient:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", timeout=900, retries=3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _post(self, endpoint, data, idempotency_key=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }
        last_exception = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exception = e
        raise last_exception

    def chat_completion(self, **kwargs):
        return self._post("chat/completions", kwargs)

    def embedding(self, **kwargs):
        return self._post("embeddings", kwargs)

    def responses(self, **kwargs):
        return self._post("responses", kwargs)

llm_client = OpenAIClient(api_key=OpenAI_key)


import requests

# Use HTTPS endpoint
API_URL = "https://historictext.org/search_pipeline"


payload = {
    "question": "description and location of the river Pogubu",
    "k": 64,
    "bge_threshold": 0.1,
    "semantic_threshold": 0.25
}


response = requests.post(API_URL, json=payload)
response.raise_for_status()
chunks = response.json()


print(f"Found {len(chunks)} relevant chunks:\n")
for i, c in enumerate(chunks, start=1):
    print(f"--- Chunk #{i} ---")
    print(f"Year:            {c['year']}")
    print(f"Document:        {c['doc_name']} ({c['doc_type']})")
    print(f"Chunk index:     {c['chunk_index']}")
    print(f"BGE score:       {c['bge_score']:.3f}")
    print(f"Semantic score:  {c['semantic_score']:.3f}")
    print("Text snippet:")
    print(c['text'][:100], "...")
    print("\nExtracted facts:")
    if c.get('facts'):
        for fact in c['facts']:
            print(f"  - {fact}")
    else:
        print("  (no facts extracted)")
    print("------------------------------\n")


resp = llm_client.chat_completion(
    model="gpt-4.1-2025-04-14",
    messages=[
        {"role": "user", "content": "description and location of the river Pogubu"},
    ]
)
content = resp["choices"][0]["message"]["content"]
print(content)


pogubu_facts = '''
    "The river Pogubu is about 1,500 kilometers in length.",
    "It runs from just south of the 13th degree to north of the 5th degree latitude.",
    "It is the largest tributary of the Madeira River.",
    "Its source is between the 12th and 13th parallels of latitude south, and between longitude 59 degrees and longitude 60 degrees west from Greenwich.",
    "The expedition embarked on it at about latitude 12 degrees 1 minute south and longitude 60 degrees 18 minutes west.",
    "Its entire course was between the 60th and 61st degrees of longitude, approaching the latter most closely at about latitude 8 degrees 15 minutes.",
    "The mouth, where it entered the Madeira, was at about 5 degrees 30 minutes latitude.",
    "The general course of the river is almost due north."
    '''


resp = llm_client.chat_completion(
    model="gpt-4.1-2025-04-14",
    messages=[
        {"role": "user", "content": pogubu_facts},
    ]
)
content = resp["choices"][0]["message"]["content"]
print(content)

