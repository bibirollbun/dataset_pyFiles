#| default_exp core


#| export

import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#| export

import gc
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
import math
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm
from collections import defaultdict
from collections import Counter
import re
import time
import os
import polars as pl
import vllm
import torch
from torch import nn
import torch.nn.functional as F
from transformers import set_seed, AutoTokenizer

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
seed_everything(seed=0)


#| export
import re
import kagglehub

constraint = kagglehub.package_import('metric/svg-constraints/versions/1').SVGConstraints()

def parse_svg_from_response(response):
    matchs = re.findall(r'<svg.*?</svg>', response, re.S)
    if matchs:
        return matchs[-1].strip()
    else:
        return ''


temp = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg"""


#| export

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pickle
import hashlib
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import json
from datetime import datetime
import logging
import os

class SvgRAGSystem:
    def __init__(self, model_name: str = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1", 
                 csv_path: str = "/kaggle/input/svg-rag-dataset/svg_dataset.parquet"):
        """
        Initialize the Math RAG system using model's tokenizer for vectorization.
        
        Args:
            model_name: Name of the LLM model to use
            csv_path: Path to the CSV file containing math QA pairs
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.model_name = model_name
        
        # Initialize the tokenizer
        self.logger.info(f"Loading tokenizer for model: {model_name}")
        
        # Initialize the LLM with vLLM
        self.logger.info("Initializing LLM...")
        self.llm = LLM(
            model_name, 
            tensor_parallel_size=2,
            trust_remote_code=True,
            enforce_eager=True,
            # dtype=torch.bfloat16,
            max_model_len=20_000,
            gpu_memory_utilization=0.96,
            max_num_seqs=12,
            seed=2024,
            dtype="half",                 # The data type for the model weights and activations
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab()) + len(self.tokenizer.get_added_vocab())
        # Load and process the dataset
        self.load_dataset(csv_path)
        
    def _generate_cache_filename(self, csv_path: str, isComplex=False) -> str:
        """Generate unique filename for cached embeddings."""
        if isComplex:
            with open(csv_path, 'rb') as f:
                csv_hash = hashlib.md5(f.read()).hexdigest()
            
            params = f"{self.model_name}"
            params_hash = hashlib.md5(params.encode()).hexdigest()[:8]
            
            return f"embeddings_tokenizer_{params_hash}_{csv_hash}.pkl"
        else:
            return "embeddings_tokenizer.pkl"

    def _save_embeddings_cache(self, cache_data: Dict, cache_file: str):
        """Save embeddings and metadata to cache."""
        cache_path = '/kaggle/working/embeddings_tokenizer.pkl'
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.logger.info(f"Cached embeddings saved to {cache_path}")

    def _load_embeddings_cache(self, cache_file: str) -> Dict:
        """Load embeddings from cache if available."""
        cache_path = '/kaggle/working/embeddings_tokenizer.pkl'
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.logger.info(f"Loaded cached embeddings from {cache_path}")
                self.logger.info(f"Cache timestamp: {cache_data['timestamp']}")
                return cache_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {str(e)}")
                return None
        return None

    def tokenize_text(self, text: List[str]) -> List[np.ndarray]:
        """
        Tokenize text and convert to token frequency vector.
        """
        # Tokenize the text
        tokens = self.tokenizer(
            text,
            max_length=20,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='np'
        )
    
        results = []
        for i in range(len(tokens['input_ids'])):
    
            # Extract token IDs from the encoding
            token_ids = tokens['input_ids'][i]
    
            # Convert to frequency vector
            vocab_size = self.vocab_size
            freq_vector = np.zeros(vocab_size)
            unique_tokens, counts = np.unique(token_ids, return_counts=True)
            freq_vector[unique_tokens] = counts
    
            # Apply log normalization (1 + log(tf))
            mask = freq_vector > 0
            freq_vector[mask] = 1 + np.log(freq_vector[mask])
            results.append(freq_vector)

        return results

    def load_dataset(self, csv_path: str):
        """Load and preprocess the math QA dataset."""
        self.logger.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_parquet(csv_path)
        
        # Combine question and solution for embedding
        self.df['combined_text'] = self.df['question']
        
        # Create embeddings for the dataset
        self.create_embeddings(csv_path)

    def create_embeddings(self, csv_path: str):
        """
        Create token frequency embeddings for all QA pairs in the dataset.
        """
        cache_file = self._generate_cache_filename(csv_path)
        
        # Try to load from cache
        cache_data = self._load_embeddings_cache(cache_file)
        
        if cache_data is not None:
            self.embeddings = cache_data['embeddings']
            self.vocab_size = cache_data['vocab_size']
        else:
            self.logger.info("Generating new token frequency embeddings...")
            
            # Process texts in batches
            all_vectors = self.tokenize_text(list(self.df['combined_text']))
            
            # Convert to sparse matrix for efficiency
            self.embeddings = csr_matrix(np.vstack(all_vectors))
            
            # Cache the embeddings
            cache_data = {
                'embeddings': self.embeddings,
                'vocab_size': self.vocab_size,
                'timestamp': datetime.utcnow().isoformat(),
                'parameters': {
                    'model_name': self.model_name,
                }
            }
            self._save_embeddings_cache(cache_data, cache_file)

    def retrieve_similar_questions(self, query: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar questions using token frequency vectors and cosine similarity.
        """
        # Get query vector
            
        query_vector = self.tokenize_text(query)[0]
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_vector.reshape(1, -1),
            self.embeddings
        ).flatten()
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return similar questions and solutions
        similar_items = []
        for i, idx in enumerate(top_k_indices):
            similar_items.append({
                'question': self.df.iloc[idx]['question'],
                'solution': self.df.iloc[idx]['solution'],
                'index': int(idx),
                'similarity_score': float(similarities[idx]),
                'similarity_rank': i + 1
            })
        
        return similar_items

    def generate_prompt(self, query: List[str], similar_questions: List[Dict[str, Any]]) -> List[str]:
        """Generate a prompt for the LLM using the retrieved similar questions."""

        system = "You are a highly skilled SVG generation assistant. Generating valid svg code by following a set of rules."
        prompts = [{'role': 'system', 'content': system}]
        
        for item in similar_questions:
            prompts.extend([{'role': 'user', 'content': temp.format(item['question'])},
                            {'role': 'assistant', 'content': item['solution']}])
        prompts.append({'role':'user', 'content': query[0]})
        
        p = self.tokenizer.apply_chat_template(
                conversation=prompts,
                tokenize=False,
                add_generation_prompt=True
            )
            
        return [p] * len(query)

    def answer_question(self, query: List[str]) -> List[str]:
        """
        Main method to answer a math question using RAG.
        """
        # Retrieve similar questions
        similar_questions = self.retrieve_similar_questions([query[0]])
        
        # Generate prompt
        prompts = self.generate_prompt(query, similar_questions)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,               # Randomness of the sampling
            top_p=0.8,                    # Cumulative probability of the top tokens to consider
            #min_p=0.05,                    # Minimum probability for a token to be considered
            skip_special_tokens=True,      # Whether to skip special tokens in the output
            max_tokens=6000,         # Maximum number of tokens to generate
            stop=["```svg\n", "```svg"],             # List of strings that stop the generation
            seed=777,
            repetition_penalty=1.05,
            include_stop_str_in_output=True,
        )

        
        outputs = self.llm.generate(prompts, sampling_params)
        
        response = [parse_svg_from_response(output.outputs[0].text) for output in outputs]
        
        return response


#| export
import kagglehub
constraint = kagglehub.package_import('metric/svg-constraints/versions/1').SVGConstraints()

import numpy as np
from PIL import Image

rag = SvgRAGSystem()

class Model:
    def __init__(self):
        self.num_responses = 4
        
    def check(self, inputs: List[str]) -> List[bool]:
        res = []
        
        for i in inputs:
            try:
                constraint.validate_svg(i)
                res.append(True)
            except:
                res.append(False)
                
        return res
        
    def predict(self, prompt: str) -> str:
        res = rag.answer_question([prompt] * self.num_responses)
        results = self.check(res)
        optimum = ''

        for i, r in enumerate(results):
            if(r and len(res[i]) > len(optimum)):
                optimum = res[i]

        return optimum if optimum else '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" fill="#fff"/></svg>'


from IPython.display import SVG

model = Model()
pred = model.predict('a lighthouse overlooking the ocean')
print(pred)
SVG(pred)




