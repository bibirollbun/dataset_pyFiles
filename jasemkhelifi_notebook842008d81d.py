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


import os

# Path to the competition data directory
competition_data_path = "/kaggle/input/konwinski-prize"
print("Files in the competition data folder:")
print(os.listdir(competition_data_path))



import zipfile
import os
import pandas as pd

# Paths
zip_file_path = "/kaggle/input/konwinski-prize/data.a_zip"
extract_to_path = "/kaggle/working/extracted_data"

# Recreate the working folder and extract
if not os.path.exists(extract_to_path):
    os.makedirs(extract_to_path)

print("Extracting data...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

# Check contents of the extracted folder
print("Contents of extracted folder:", os.listdir(extract_to_path))

# Check contents of 'data' subfolder
data_folder = os.path.join(extract_to_path, "data")
print("Contents of 'data' subfolder:", os.listdir(data_folder))

# Read the parquet file
parquet_file_path = os.path.join(data_folder, "data.parquet")  # Adjust if necessary
df = pd.read_parquet(parquet_file_path)

print(f"Total rows in dataset: {len(df)}")
print(df.head(3))



# Print the column names and data types
print("\nColumn Names and Data Types:")
print(df.dtypes)

# Print the total number of rows
print(f"\nTotal number of rows: {len(df)}")



#!pip install rank_bm25
!pip install /kaggle/input/rank-bm25/rank_bm25-0.2.2-py3-none-any.whl



import os
import gc
import pandas as pd
import torch
import zipfile
import shutil
from glob import glob
from rank_bm25 import BM25Okapi
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import io
from kaggle_evaluation.konwinski_prize_inference_server import KPrizeInferenceServer



# **Clear GPU memory**
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# **BM25 Setup**
def prepare_bm25_documents(repo_path, min_tokens=1, max_tokens=100000):
    documents = []
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    token_count = len(content.split())
                    if min_tokens <= token_count <= max_tokens:
                        documents.append(f"<file: {file_path}>\n{content}")
                        file_paths.append(file_path)
            except UnicodeDecodeError:
                # Suppress UnicodeDecodeError warnings (non-text files)
                pass
            except Exception:
                # Suppress all other file reading errors
                pass
    return documents, file_paths


def bm25_retrieve_limited(query, documents, file_paths, max_tokens=100000, min_files=10):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.split())

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected_docs = []
    total_tokens = 0

    for i in ranked_indices:
        doc_tokens = len(documents[i].split())
        if total_tokens + doc_tokens > max_tokens:
            break
        selected_docs.append((file_paths[i], documents[i]))
        total_tokens += doc_tokens
        if len(selected_docs) >= min_files:
            break

    return selected_docs


def process_query_extract_first_lines(query, num_lines=3):
    """
    Extract the first few lines from the query.
    """
    lines = query.split("\n")[:num_lines]
    return " ".join(line.strip() for line in lines if line.strip())

def process_query_extract_keywords(query):
    """
    Extract technical terms from the query using regex.
    """
    # Match technical terms such as file paths, functions, classes, or keywords
    keywords = re.findall(r'\b(def|class|import|return|Error|Exception|traceback)\b', query)
    
    # Add file-like patterns (e.g., `something.py`)
    file_paths = re.findall(r'[\w\-/]+\.py', query)

    return " ".join(set(keywords + file_paths))
    
def process_query_extract_errors(query):
    """
    Extract lines containing error messages from the query.
    """
    error_lines = [line.strip() for line in query.split("\n") if "error" in line.lower() or "exception" in line.lower()]
    return " ".join(error_lines)

def process_query_with_code_snippets(query):
    """
    Extract and prioritize code snippets from the query.
    """
    # Extract code snippets enclosed in triple backticks
    code_snippets = re.findall(r"```(.*?)```", query, re.DOTALL)
    
    # Add heuristic for standalone code-like lines
    code_lines = [line.strip() for line in query.split("\n") if any(kw in line for kw in ["def ", "class ", "import "])]
    
    # Combine snippets and standalone lines
    code_content = " ".join(code_snippets + code_lines)
    
    # Combine code snippets with the original query for context
    return f"{query.strip()} {code_content.strip()}"

def process_query_with_context(query):
    """
    Extract context-specific terms from the query.
    """
    # Common context keywords
    action_keywords = ["fix", "add", "update", "remove"]
    issue_keywords = ["bug", "error", "crash", "slow", "unexpected"]
    component_keywords = ["UI", "database", "API", "server", "backend"]

    # Extract relevant words from query
    extracted = [word for word in query.lower().split() if word in action_keywords + issue_keywords + component_keywords]
    
    return " ".join(extracted)


def process_query_combined(query):
    """
    Combine multiple strategies for query processing.
    """
    # Extract parts using different approaches
    first_lines = process_query_extract_first_lines(query, num_lines=2)
    keywords = process_query_extract_keywords(query)
    errors = process_query_extract_errors(query)
    code_snippets = process_query_with_code_snippets(query)

    # Combine them into a single query
    return f"{first_lines} {keywords} {errors} {code_snippets}".strip()



def combine_chunks_from_same_file(ranked_chunks):
    """
    Combine chunks from the same file into larger chunks, preserving context.
    """
    file_chunks = {}
    
    for file_path, chunk in ranked_chunks:
        if file_path not in file_chunks:
            file_chunks[file_path] = []
        file_chunks[file_path].append(chunk)
    
    combined_chunks = {}
    
    for file_path, chunks in file_chunks.items():
        # Sort chunks by their starting line number
        sorted_chunks = sorted(chunks, key=lambda x: int(re.search(r"\[start of .+\](\n.*?)?\n", x).group(1).split("\n")[0].split(":")[0]))
        
        combined_chunk = []
        previous_end_line = -1
        
        for chunk in sorted_chunks:
            lines = chunk.split("\n")
            start_line = int(re.search(r"\[start of .+\](\n.*?)?\n", chunk).group(1).split("\n")[0].split(":")[0])
            
            if previous_end_line != -1 and start_line > previous_end_line:
                # Add the lines between the previous chunk and the current chunk
                with open(file_path, "r") as f:
                    all_lines = f.readlines()
                    combined_chunk.extend(all_lines[previous_end_line:start_line])
            
            combined_chunk.extend(lines)
            previous_end_line = start_line + len(lines) - 1
        
        combined_chunks[file_path] = "\n".join(combined_chunk)
    
    return combined_chunks


def format_file_path(long_path):
    """
    Convert long file paths into the required format: `/module/file.py`
    """
    parts = long_path.split("/")
    if "repos" in parts:
        idx = parts.index("repos") + 1  # Find "repos" and get repo name
        return "/" + "/".join(parts[idx+1:])  # Return formatted path
    return long_path  # Default case (should not happen)



def format_code_chunks(merged_chunks):
    """
    Convert merged chunks into the final prompt format.
    """
    return "\n\n".join(
        [f"<file: {format_file_path(file)}>\n{chunk}" for file, (_, _, chunk) in merged_chunks.items()]
    )





def merge_chunks_within_files(top_files, ranked_chunks):
    """
    Merge chunks within the same file and add +50 lines up/down for the two largest files.
    """
    merged_chunks = {}

    for file_path, ranges in ranked_chunks.items():
        # Sort chunks by starting line number
        sorted_chunks = sorted(ranges, key=lambda x: x[0])

        # Find the min and max lines across all chunks for this file
        min_line = sorted_chunks[0][0]
        max_line = sorted_chunks[-1][1]

        # Extract actual content
        file_content = next((content for path, content in top_files if path == file_path), None)
        if not file_content:
            continue  

        lines = file_content.split("\n")
        merged_chunk = "\n".join(lines[min_line : max_line + 1])

        merged_chunks[file_path] = (min_line, max_line, merged_chunk)

    # Find top 2 files with the most merged lines
    sorted_files = sorted(merged_chunks.items(), key=lambda x: x[1][1] - x[1][0], reverse=True)[:2]

    # Expand context by +50 lines for top 2 files
    for file_path, (min_line, max_line, chunk) in sorted_files:
        file_content = next((content for path, content in top_files if path == file_path), None)
        if not file_content:
            continue  

        lines = file_content.split("\n")
        min_line = max(0, min_line - 50)
        max_line = min(len(lines) - 1, max_line + 50)

        merged_chunks[file_path] = (min_line, max_line, "\n".join(lines[min_line:max_line + 1]))

    return merged_chunks



def rank_chunks_in_top_files(top_files, query, top_n=5, context_window=10):
    """
    Rank 10-line overlapping chunks inside the top 10 files and extract the highest-ranked non-overlapping ones.
    Merge chunks from the same file to create a continuous context.
    """
    bm25_corpus = []
    chunk_file_map = []

    for file_path, content in top_files[:10]:  # Process the top 10 files
        lines = content.split("\n")

        # Generate 10-line overlapping chunks
        for idx in range(len(lines) - 9):  
            chunk = "\n".join(lines[idx : idx + 10])
            bm25_corpus.append(chunk)
            chunk_file_map.append((file_path, idx))  # Track which file and start index

    # Apply BM25 on 10-line chunks
    bm25 = BM25Okapi([chunk.split() for chunk in bm25_corpus])
    scores = bm25.get_scores(query.split())

    # Select top-ranked non-overlapping chunks
    top_chunk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected_chunks = {}
    selected_ranges = {}

    for idx in top_chunk_indices:
        file_path, start_line = chunk_file_map[idx]
        end_line = start_line + 9  

        # Ensure non-overlapping chunks
        if file_path in selected_ranges:
            existing_ranges = selected_ranges[file_path]
            if any(s <= end_line and e >= start_line for s, e in existing_ranges):
                continue  

        # Store chunk with its file
        if file_path not in selected_chunks:
            selected_chunks[file_path] = []
            selected_ranges[file_path] = []

        selected_chunks[file_path].append((start_line, end_line, bm25_corpus[idx]))
        selected_ranges[file_path].append((start_line, end_line))

        # Stop if we have enough
        if sum(len(r) for r in selected_ranges.values()) >= top_n:
            break  

    return selected_chunks




import re

def extract_diff_patch(output_text):
    """
    Extracts the diff patch from the model's output and cleans up extra <patch> tags.
    """
    matches = re.findall(r"<patch>\s*(diff --git .*?)\s*</patch>", output_text, re.DOTALL)

    if len(matches) < 1:
        print("[WARNING] No valid <patch> found.")
        return None

    extracted_patch = matches[0] if len(matches) == 1 else matches[1]

    # Remove redundant <patch> tags line by line
    cleaned_patch = "\n".join(
        [line for line in extracted_patch.split("\n") if "<patch>" not in line and "</patch>" not in line]
    ).strip()

    # Sanity check: Ensure valid diff format
    if len(cleaned_patch.split("\n")) < 3 or "diff --git" not in cleaned_patch:
        print("[WARNING] Invalid or malformed patch detected:\n", cleaned_patch)
        return None

    return cleaned_patch

# **Prepare Submission Entry**
def prepare_submission(instance_id, model_patch):
    """
    Creates a properly formatted JSON submission entry.
    """
    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": "SWE-Llama-13b"
    }



import os
import gc
import io
import shutil
import zipfile
import subprocess
import pandas as pd
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
import kaggle_evaluation.konwinski_prize_inference_server as kp_server

# **Paths**
MODEL_DIR_13B = "/kaggle/input/swe-llama-model/swe-llama-model"
ZIP_FILE_PATH = "/kaggle/input/konwinski-prize/data.a_zip"
EXTRACTED_DATA_DIR = "/kaggle/working/extracted_data"
PARQUET_FILE = os.path.join(EXTRACTED_DATA_DIR, "data", "data.parquet")
COMP_REPOS_DIR = os.path.join(EXTRACTED_DATA_DIR, "data", "repos")

# **Extract ZIP file if necessary**
if not os.path.exists(EXTRACTED_DATA_DIR):
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_DATA_DIR)

# **Load dataset**
kprize_df = pd.read_parquet(PARQUET_FILE)

# **Kaggle Evaluation Server**
instance_count = None

def get_number_of_instances(num_instances: int):
    global instance_count
    instance_count = num_instances

def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Handles prediction dynamically for Kaggle inference server. """
    
    # **Extract Repository**
    repo_path = '/kaggle/working/repo'
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    with open('/kaggle/working/repo_archive.tar', 'wb') as f:
        f.write(repo_archive.read())
    shutil.unpack_archive('/kaggle/working/repo_archive.tar', extract_dir=repo_path)
    os.remove('/kaggle/working/repo_archive.tar')
    
    # **Setup Environment**
    pip_packages_path = '/kaggle/working/pip_packages'
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)
    with open('/kaggle/working/pip_packages_archive.tar', 'wb') as f:
        f.write(pip_packages_archive.read())
    shutil.unpack_archive('/kaggle/working/pip_packages_archive.tar', extract_dir=pip_packages_path)
    os.remove('/kaggle/working/pip_packages_archive.tar')
    
    env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path) for cmd in env_setup_cmds_templates]
    subprocess.run("\n".join(env_setup_cmds), shell=True, executable="/bin/bash", cwd=repo_path)
    
    # **Find Matching Issue**
    instance_id = None
    for idx, issue in kprize_df.iterrows():
        if issue["problem_statement"] == problem_statement:
            instance_id = f"instance_{idx}"
            break
    
    if instance_id is None:
        print("[WARNING] Issue not found in dataset.")
        return None
    
    # **Prepare BM25 Documents**
    documents, file_paths = prepare_bm25_documents(repo_path)
    
    # **Retrieve Top 10 Files**
    query = process_query_combined(problem_statement)
    top_files = bm25_retrieve_limited(query, documents, file_paths)
    
    # **Rank and Merge Chunks**
    ranked_chunks = rank_chunks_in_top_files(top_files, problem_statement)
    merged_chunks = merge_chunks_within_files(top_files, ranked_chunks)
    code_chunks_str = format_code_chunks(merged_chunks)
    
    # **Generate Input Prompt**
    input_prompt = f"""
    You will be provided with a partial code base and an issue statement explaining a problem to resolve.
    
    <issue>
    {problem_statement}
    </issue>
    
    <code>
    {code_chunks_str}
    </code>
    
    Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, 
    the line numbers of each change, and the removed and added lines. A single patch file can contain changes 
    to multiple files.
    
    <patch>
    --- a/file.py
    +++ b/file.py
    @@ -1,27 +1,35 @@
    def euclidean(a, b):
    - while b:
    -     a, b = b, a % b
    - return a
    + if b == 0:
    +     return a
    + return euclidean(b, a % b)
    
    def bresenham(x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
    -   sx = 1 if x0 < x1 else -1
    -   sy = 1 if y0 < y1 else -1
    -   err = dx - dy
    +   x, y = x0, y0
    +   sx = -1 if x0 > x1 else 1
    +   sy = -1 if y0 > y1 else 1
    
    -   while True:
    -       points.append((x0, y0))
    -       if x0 == x1 and y0 == y1:
    -           break
    -       e2 = 2 * err
    -       if e2 > -dy:
    +   if dx > dy:
    +       err = dx / 2.0
    +       while x != x1:
    +           points.append((x, y))
                err -= dy
    -           x0 += sx
    -       if e2 < dx:
    -           err += dx
    -           y0 += sy
    +           if err < 0:
    +               y += sy
    +               err += dx
    +           x += sx
    +   else:
    +       err = dy / 2.0
    +       while y != y1:
    +           points.append((x, y))
                err -= dx
    +           if err < 0:
    +               x += sx
    +               err += dy
    +           y += sy
    +   points.append((x, y))
        return points
    </patch>
    
    I need you to solve the provided issue by generating a single patch file that I can apply directly to this repository using `git apply`. 
    Please respond with a single patch file in the format shown above.
    
    <patch>
    """
    
    # **Load LLaMA 13B Model**
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_13B)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR_13B, device_map="auto", torch_dtype=torch.float16, offload_folder="/kaggle/tmp/offload"
    )
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_new_tokens=1000)
    predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # **Extract Patch**
    extracted_patch = extract_diff_patch(predicted_output)
    
    if extracted_patch is None:
        print(f"[WARNING] Skipping instance {instance_id} due to invalid patch.")
        return None
    
    return extracted_patch

# **Start Kaggle Inference Server**
inference_server = kp_server.KPrizeInferenceServer(get_number_of_instances, predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',
            '/kaggle/tmp/konwinski-prize/',
        ),
        use_concurrency=True,
    )


