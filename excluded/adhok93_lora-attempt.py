reading_prompt = """
You are analyzing a code repository to fix an issue.

Problem: {problem_statement}

Repository structure:
{directory_string}

Please identify the relevant files to inspect in XML format:
<root>
    <entry>
        <filepath>EXACT_PATH_FROM_REPOSITORY</filepath>
        <strings_to_search>
            <string_to_search>RELEVANT_CODE_PATTERN</string_to_search>
        </strings_to_search>
    </entry>
</root>
"""


# !pip install -q torch transformers
!pip install -q peft bitsandbytes
!pip install datasets
# # !pip install -q accelerate
# !pip install -q vllm
# # !pip install -q pandas pyarrow

import os

import torch

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_per_process_memory_fraction(0.8, device=torch.device("cuda:0"))




import io
import os
import shutil
import tempfile
import warnings
import re
from typing import Optional, Dict, List
from dataclasses import dataclass
import tarfile

import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.simplefilter('ignore')

# Environment settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# Constants
REPO_PATH = "repo"
MAX_MODEL_LEN = 32_768

if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    model_path = '/kaggle/input/qwen2.5/transformers/0.5b/1'
else:
    model_path = '/kaggle/input/qwen2.5/transformers/0.5b/1'


# reading_prompt = """
# You are analyzing a code repository to fix an issue. Based on the error message and problem statement, you need to identify the relevant files that need to be modified.

# Problem: {problem_statement}

# Repository structure:
# {directory_string}

# Based on the error traceback and file structure above, provide the exact filepath and relevant code patterns to search in this XML format:
# <root>
#     <entry>
#         <filepath>repo/astroid/nodes/node_classes.py</filepath>
#         <strings_to_search>
#             <string_to_search>_infer_from_values</string_to_search>
#             <string_to_search>_infer</string_to_search>
#         </strings_to_search>
#     </entry>
# </root>

# Note: 
# - Use the exact filepath from the repository structure
# - Include specific code patterns mentioned in the error trace
# - You can add multiple entries if multiple files need to be modified

# Please do not print the entire content of the file.
# """


reading_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
You will first need to select files in the file directory.

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

Which files should be inspected so that we can solve the problem?
When we inspect each file, what strings should be searched?

Return the strings to search in this format

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>  
        <strings_to_search>
            <string_to_search>_infer_from_values</string_to_search>
            ...
            <string_to_search>_infer_from_values</string_to_search>
        </strings_to_search>
    </entry>
</root>

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>_infer_from_values</string_to_search>
            ...
            <string_to_search>_infer_from_values</string_to_search>
        </strings_to_search>
    </entry>
</root>
...

Notes:
- Make sure to encode each entry between <root> and </root>
- Return the FULL filepath - exactly as specified in <directory> and </directory>
    - Example: repo/path/to/directory/file.py
- If you are searching for a word instead of a substring, maybe add spaces or brackets before and after the string
    For example, if you are searching for uses of the function `calculate`, use ` calculate(` as the search string instead
- Prefer searching longer strings
- Do not inspect more than 5 files
- Only inspect the necessary files
""".strip()



# reading_prompt = """
# You will be implementing a git diff patch to solve an issue with the code repository.
# You will first need to select files in the file directory.

# This is the problem statement.

# {problem_statement}

# This is the file directory

# <directory>
# {directory_string}
# </directory>

# Which files should be inspected so that we can solve the problem?
# When we inspect each file, what strings should be searched?

# Return the strings to search in this format:

# <root>
#     <entry>
#         <filepath>filepath</filepath>  
#         <strings_to_search>
#             <string_to_search>string_to_search</string_to_search>
#         </strings_to_search>
#     </entry>
# </root>
# """.strip()

# patching_prompt = """
# You will be implementing a git diff patch to solve an issue with the code repository.
# This is the problem statement.

# {problem_statement}

# These are the files that is thought to be relevant

# {file_content_string}

# Write a git diff within <patch> and </patch> that fixes the problem.
# """.strip()

patching_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
This is the problem statement:

{problem_statement}

These are the relevant files and their contents:

{file_content_string}

The error occurs when a None value is being formatted with a format specifier. 
The patch should add a check to handle None values before formatting.

Write a git diff within <patch> and </patch> that fixes the problem.

Example format:
<patch>
diff --git a/repo/astroid/nodes/node_classes.py b/repo/astroid/nodes/node_classes.py
--- a/repo/astroid/nodes/node_classes.py
+++ b/repo/astroid/nodes/node_classes.py
@@ -4694,6 +4694,8 @@ def _infer(
     def _infer_from_values(self, context):
         values = self._infer(context)
         value = next(values)
+        if value.value is None:
+            raise TypeError("Cannot apply format specifier to None value")
         formatted = format(value.value, format_spec.value)
</patch>
"""


patching_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant

{file_content_string}

Write a git diff within <patch> and </patch> that fixes the problem.

Example:

<patch>
--- a/first.txt
+++ b/first.txt
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
--- a/second.txt
+++ b/second.txt
@@ -1,3 +1,3 @@
 beginning
-old line
+new line
 end
</patch>
""".strip()




def stringify_directory(directory):
    full_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            full_paths.append(full_path)
    return "\n".join(full_paths)

def extract_file_query(xml_content):

    if not xml_content or "<root>" not in xml_content:
        print("Invalid XML response from model")
        return {}
    parsed_data = {}
    pattern = r'<root>(.*?)</root>'
    matches = re.findall(pattern, xml_content, re.DOTALL)
    
    for match in matches:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring("<root>" + match + "</root>")
            
            for entry in root.findall("entry"):
                filepath = entry.find("filepath")
                filepath_text = filepath.text.strip() if filepath is not None else None
                
                strings_container = entry.find("strings_to_search")
                search_strings = []
                if strings_container is not None:
                    for s in strings_container.findall("string_to_search"):
                        if s.text is not None:
                            search_strings.append(s.text.strip())
                
                parsed_data[filepath_text] = search_strings
        except:
            print("Error parsing output", xml_content)
            return ""
        
    return parsed_data

def extract_patch_string(text):
    pattern = r'<patch>(.*?)</patch>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return "\n".join(matches)


def fetch_file_contents(files_to_search, context_lines=10):

    if not files_to_search or not isinstance(files_to_search, dict):
        print("Invalid file query structure")
        return ""
    def find_lines_in_files_with_context(search_map, context_lines):
        all_matches_per_file = []

        for path, terms in search_map.items():
            if not os.path.isfile(path):
                all_matches_per_file.append([])
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            file_snippets = []
            num_lines = len(lines)

            for i, line in enumerate(lines, start=1):
                if any(t in line for t in terms):
                    start_idx = max(1, i - context_lines)
                    end_idx = min(num_lines, i + context_lines)
                    snippet = []
                    for snippet_no in range(start_idx, end_idx + 1):
                        text_content = lines[snippet_no - 1].rstrip("\n")
                        snippet.append((snippet_no, text_content))
                    file_snippets.append(snippet)

            all_matches_per_file.append(file_snippets)

        return all_matches_per_file

    snippets = find_lines_in_files_with_context(files_to_search, context_lines)
    output = []
    
    for filepath, file_snippets in zip(files_to_search.keys(), snippets):
        output.append(f"FILE: {filepath[len(REPO_PATH) + 1:]}")
        if not file_snippets:
            output.append("No matches found.")
        else:
            for snippet in file_snippets:
                start_line = snippet[0][0]
                end_line = snippet[-1][0]
                output.append(f"Lines {start_line} to {end_line}:")
                for line_no, text in snippet:
                    output.append(f"{line_no:3d} | {text}")
                output.append("")
        output.append("=" * 60)
        output.append("")
    
    return "\n".join(output)


# !pip install --upgrade transformers

from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class GitPatchModel:
    def __init__(self, model_path: str):  # Fixed double underscores
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        ## Adding new
        config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        
        # 4-bit quantization setup
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True

            
        )
        
        # Load base model
        print("Loading base model...")

        # self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path,
#                 config=config,
#                 device_map="auto",
#                 local_files_only=True,
#                 trust_remote_code=True,
#                 use_safetensors=True,
#                 torch_dtype=torch.float16,
#                 quantization_config=quantization_config,
#                 low_cpu_mem_usage=True,
#                 offload_folder="offload",
#                 offload_state_dict=True,
#                 max_memory={0: "12GB", "cpu": "24GB"}
#             )

        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            use_cache = False,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True,
            max_memory={0: "8GB", "cpu": "16GB"}
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["self_attn.q_proj", 
                    "self_attn.k_proj", 
                    "self_attn.v_proj", 
                    "self_attn.o_proj",
                    "mlp.gate_proj", 
                    "mlp.up_proj", 
                    "mlp.down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
    def generate(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_tokens = 512,
            #max_length=max_length,
            temperature=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def collate_fn(self, examples):
        """Collate function for batching examples."""
        # Get prompts and completions
        prompts = [ex["prompt"] for ex in examples]
        completions = [ex["completion"] for ex in examples]
        
        # First tokenize both inputs with the same max length
        input_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_LEN,
            return_tensors="pt"
        ).to('cpu')
        
        label_encodings = self.tokenizer(
            completions,
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_LEN,
            return_tensors="pt"
        ).to('cpu')
    
        # Create labels tensor with -100 for padding tokens
        labels = label_encodings["input_ids"].clone()
        labels[label_encodings["attention_mask"] == 0] = -100
        
        # Ensure input_ids and labels have same size
        input_length = input_encodings["input_ids"].size(1)
        label_length = labels.size(1)

        # print("input_ids shape:", input_encodings["input_ids"].shape)
        # print("labels shape:", labels.shape)
        
        # If sizes don't match, pad the shorter one
        if input_length > label_length:
            # Pad labels
            padding_size = input_length - label_length
            labels = torch.nn.functional.pad(labels, (0, padding_size), value=-100)
        elif label_length > input_length:
            # Pad inputs
            padding_size = label_length - input_length
            input_encodings["input_ids"] = torch.nn.functional.pad(input_encodings["input_ids"], (0, padding_size), value=self.tokenizer.pad_token_id)
            input_encodings["attention_mask"] = torch.nn.functional.pad(input_encodings["attention_mask"], (0, padding_size), value=0)
    
        # Create final batch
        batch = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels
        }
    
        # Verify shapes match
        assert batch["input_ids"].size() == batch["labels"].size(), \
            f"Input shape {batch['input_ids'].size()} != Labels shape {batch['labels'].size()}"
        
        return batch


class GitPatchModel:
    def __init__(self, model_path: str):
        """Initialize the model with memory optimizations."""
        self.model_path = model_path  # Store model_path as instance variable
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # Load configuration
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 4-bit quantization setup for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        print("Loading base model with memory optimizations...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            use_cache=False,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True,
            max_memory={0: "4GB", "cpu": "8GB"}
        )
        
        # Prepare for training with LoRA
        print("Preparing model for training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config for memory-efficient fine-tuning
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        
        
        self.model = get_peft_model(self.model, lora_config)
        
    # def collate_fn(self, examples):
    #     """Collate function for batching examples."""
    #     prompts = [ex["prompt"] for ex in examples]
    #     completions = [ex["completion"] for ex in examples]
        
    #     max_length = 512
        
    #     # Ensure tensors stay on CPU
    #     inputs = self.tokenizer(
    #         prompts,
    #         padding=True,
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt"
    #     ).to('cpu')  # Explicitly move to CPU
        
    #     labels = self.tokenizer(
    #         completions,
    #         padding=True,
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt"
    #     ).to('cpu')  # Explicitly move to CPU
        
    #     batch = {
    #         "input_ids": inputs["input_ids"],
    #         "attention_mask": inputs["attention_mask"],
    #         "labels": labels["input_ids"].clone()
    #     }
    #     print(len(labels))
    #     print(len(inputs))
    #     # Replace padding tokens
    #     batch["labels"][labels["attention_mask"] == 0] = -100
        
    #     # Ensure all tensors are on CPU
    #     batch = {k: v.to('cpu') for k, v in batch.items()}
        
    #     return batch

    def generate(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            #max_length=max_length,
            max_new_tokens=512,
            temperature=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def collate_fn(self, examples):
        """
        Collate function for batching examples.
        Expects examples to be a list of dicts with keys "prompt" and "completion".
        """
        # Gather prompts and completions
        prompts = [ex["prompt"] for ex in examples]
        completions = [ex["completion"] for ex in examples]
        
        # Set a fixed max_length for both
        max_length = 512
        
        # 1) Tokenize the prompts
        input_encodings = self.tokenizer(
            prompts,
            padding="max_length",   # Ensures uniform length
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 2) Tokenize the completions
        label_encodings = self.tokenizer(
            completions,
            padding="max_length",   # Ensures uniform length
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 3) Create labels by cloning the label_encodings input_ids
        labels = label_encodings["input_ids"].clone()
        
        # Replace all padding token indices with -100 so they are ignored by the loss
        labels[label_encodings["attention_mask"] == 0] = -100
        
        # (Optional) Move everything to CPU here (depending on your setup):
        input_ids = input_encodings["input_ids"].to("cpu")
        attention_mask = input_encodings["attention_mask"].to("cpu")
        labels = labels.to("cpu")
        
        # Debug prints: confirm shapes match
        # print(f"input_ids.shape: {input_ids.shape}")
        # print(f"attention_mask.shape: {attention_mask.shape}")
        # print(f"labels.shape: {labels.shape}")
        
        # 4) Wrap in a dict for Trainer
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        return batch


def train_model(model_path: str, data_path: str, repos_dir: str, output_dir: str):
    """Train the model on the prepared dataset."""
    print("Initializing model and data processor...")
    model = GitPatchModel(model_path)
    data_processor = GitPatchDataProcessor(data_path, repos_dir)
    
    print("Preparing training data...")
    train_examples = data_processor.prepare_training_examples()
    if not train_examples:
        raise ValueError("No training examples generated!")
    
    train_dataset = Dataset.from_list(train_examples)
    print(f"Created dataset with {len(train_dataset)} examples")
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=500,
            per_device_train_batch_size=1,  # Reduced batch size
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            fp16=True,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=10,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            # max_length=2048,  # Add explicit max length
            # pad_to_max_length=True,  # Ensure padding
            dataloader_drop_last=True  # Drop last incomplete batch
        )
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=model.collate_fn
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Training completed successfully!")

def predict(model: GitPatchModel, problem_statement: str, repo_archive) -> str:
    """Generate a patch prediction for a given problem."""
    # Extract repository to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, 'repo')
        os.makedirs(repo_path)
        
        # Extract repo archive
        repo_archive.seek(0)
        with tarfile.open(fileobj=repo_archive, mode='r:*') as tar:
            tar.extractall(repo_path)
        
        # Get directory structure
        directory_string = stringify_directory(repo_path)
        
        # Create prompt
        prompt = (
            f"Problem: {problem_statement}\n\n"
            f"Repository structure:\n{directory_string}\n\n"
            "Generate a patch to fix this issue."
        )
        
        # Generate prediction
        inputs = model.tokenizer(prompt, return_tensors="pt")
        outputs = model.model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            pad_token_id=model.tokenizer.eos_token_id
        )
        
        prediction = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract patch from prediction
        if "<patch>" in prediction and "</patch>" in prediction:
            patch = prediction.split("<patch>")[1].split("</patch>")[0]
        else:
            patch = prediction
            
        return patch



import pandas as pd
from typing import List, Optional
import os
import glob
from pathlib import Path

class GitPatchDataProcessor:
    def __init__(self, data_path: str, repos_dir: str):
        """Initialize the GitPatchDataProcessor.
        
        Args:
            data_path: Path to the parquet file containing the dataset
            repos_dir: Directory containing the git repositories
        """
        # Load and validate the parquet file
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_df = pd.read_parquet(data_path)
        
        # Validate repos directory
        if not os.path.exists(repos_dir):
            raise FileNotFoundError(f"Repos directory not found: {repos_dir}")
        self.repos_dir = repos_dir
        
        # Validate required columns
        required_columns = ['instance_id', 'problem_statement', 'patch']
        missing_columns = [col for col in required_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
            
    def stringify_directory(self, repo_path: str) -> str:
        """Convert directory structure to a string representation.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            String representation of the directory structure
        """
        if not os.path.exists(repo_path):
            print(f"Warning: Repository path does not exist: {repo_path}")
            return ""
            
        result = []
        for root, dirs, files in os.walk(repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
                
            level = root.replace(repo_path, '').count(os.sep)
            indent = '  ' * level
            result.append(f'{indent}{os.path.basename(root)}/')
            
            for file in files:
                result.append(f'{indent}  {file}')
                
        return '\n'.join(result)
        
    def read_file_content(self, file_path: str) -> str:
        """Read content of a file safely.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Content of the file or empty string if file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {str(e)}")
            return ""
            
    def prepare_training_examples(self, 
                                start_idx: int = 0, 
                                end_idx: Optional[int] = None) -> List[dict]:
        """Prepare training examples from the dataset.
        
        Args:
            start_idx: Starting index in the dataset
            end_idx: Ending index in the dataset (exclusive)
            
        Returns:
            List of training examples
        """
        examples = []
        end_idx = end_idx or len(self.data_df)
        
        print(f"Preparing examples from index {start_idx} to {end_idx}")
        print(f"Total rows in dataset: {len(self.data_df)}")
        
        for idx in range(start_idx, end_idx):
            try:
                # Get basic information
                problem_statement = self.data_df["problem_statement"][idx]
                instance_id = self.data_df['instance_id'][idx]
                repo_path = os.path.join(self.repos_dir, f"repo__{instance_id}")
                
                print(f"\nProcessing example {idx}:")
                print(f"Instance ID: {instance_id}")
                print(f"Repo path: {repo_path}")
                
                # Get directory structure
                directory_string = self.stringify_directory(repo_path)
                if not directory_string:
                    print(f"Warning: Empty directory structure for {repo_path}")
                    continue
                    
                # Create file navigation example
                nav_prompt = (
                    f"Problem: {problem_statement}\n\n"
                    f"Repository structure:\n{directory_string}\n\n"
                    "Which files should be modified to fix this issue?"
                )
                
                # Create patch generation example
                patch_content = self.data_df['patch'][idx]
                if not patch_content:
                    print(f"Warning: Empty patch content for index {idx}")
                    continue
                    
                patch_prompt = (
                    f"Problem: {problem_statement}\n\n"
                    f"Repository structure:\n{directory_string}\n\n"
                    "Generate a patch to fix this issue."
                )
                
                # Add both examples
                examples.append({
                    "prompt": nav_prompt,
                    "completion": self.extract_files_from_patch(patch_content)
                })
                
                examples.append({
                    "prompt": patch_prompt,
                    "completion": f"<patch>{patch_content}</patch>"
                })
                
                print(f"Added navigation and patch examples for index {idx}")
                
            except Exception as e:
                print(f"Error processing index {idx}: {str(e)}")
                continue
                
        print(f"\nTotal examples generated: {len(examples)}")
        return examples
        
    def extract_files_from_patch(self, patch_content: str) -> str:
        """Extract modified file paths from a patch.
        
        Args:
            patch_content: Content of the patch
            
        Returns:
            List of modified files as a string
        """
        files = []
        for line in patch_content.split('\n'):
            if line.startswith('diff --git'):
                # Extract the second file path (b/path/to/file)
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3][2:]  # Remove b/ prefix
                    files.append(file_path)
                    
        return '\n'.join(files)
        
    def validate_repository(self, repo_path: str) -> bool:
        """Validate that a repository directory exists and contains files.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            True if repository is valid, False otherwise
        """
        if not os.path.exists(repo_path):
            print(f"Repository directory does not exist: {repo_path}")
            return False
            
        if not os.path.isdir(repo_path):
            print(f"Repository path is not a directory: {repo_path}")
            return False
            
        files = glob.glob(os.path.join(repo_path, '**/*'), recursive=True)
        if not files:
            print(f"Repository directory is empty: {repo_path}")
            return False
            
        return True


# def predict_inner(model: GitPatchModel, problem_statement: str, directory: str) -> str:
#     # Get file query
#     directory_string = stringify_directory(directory)
#     query_prompt = reading_prompt.format(
#         problem_statement=problem_statement,
#         directory_string=directory_string
#     )
#     query_response = model.generate(query_prompt)
#     file_query = extract_file_query(query_response)
#     print("File query:", file_query)

#     if not file_query:
#         return None

#     # Get file contents
#     file_content_string = fetch_file_contents(file_query)
#     print("File contents:", file_content_string)

#     # Generate patch
#     patch_prompt = patching_prompt.format(
#         problem_statement=problem_statement,
#         file_content_string=file_content_string
#     )
#     patch_response = model.generate(patch_prompt)
#     patch_string = extract_patch_string(patch_response)
#     print("Generated patch:", patch_string)

#     return patch_string

# def predict(model: GitPatchModel, 
#             problem_statement: str, 
#             repo_archive: io.BytesIO) -> Optional[str]:
#     # Extract repository
#     with open('repo_archive.tar', 'wb') as f:
#         f.write(repo_archive.read())
        
#     if os.path.exists(REPO_PATH):
#         shutil.rmtree(REPO_PATH)
#     shutil.unpack_archive('repo_archive.tar', extract_dir=REPO_PATH)
#     os.remove('repo_archive.tar')

#     try:
#         patch_string = predict_inner(model, problem_statement, REPO_PATH)
#     finally:
#         shutil.rmtree(REPO_PATH)

#     return patch_string


import torch
import gc

def clear_cuda_memory():
    """
    Clears CUDA memory by emptying cache and collecting garbage.
    Should be called between large operations.
    """
    # Clear pytorch's cuda cache
    torch.cuda.empty_cache()
    # Run garbage collector
    gc.collect()

class MemoryManagement:
    """
    Context manager for handling CUDA memory in PyTorch operations
    """
    def __init__(self, model=None):
        self.model = model

    def __enter__(self):
        clear_cuda_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_cuda_memory()

def optimize_model_memory(model):
    """
    Optimizes model memory usage by moving to CPU when not in use
    """
    if hasattr(model, 'cpu'):
        model.cpu()
    clear_cuda_memory()
    return model

def chunked_generate(model, tokenizer, prompt, chunk_size=512, max_length=512, device='cuda'):
    """
    Processes generation in chunks to manage memory better
    """
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Process in chunks
    output_tokens = []
    for i in range(0, max_length, chunk_size):
        with MemoryManagement():
            current_length = min(chunk_size, max_length - i)
            chunk_output = model.generate(
                **tokens,
                max_length=current_length,
                pad_token_id=tokenizer.eos_token_id
            )
            output_tokens.extend(chunk_output[0][tokens.input_ids.shape[1]:])
            
            # Update input for next iteration
            tokens = tokenizer(
                tokenizer.decode(output_tokens),
                return_tensors="pt"
            ).to(device)
    
    return tokenizer.decode(output_tokens)




def predict_inner(model: GitPatchModel, problem_statement: str, directory: str) -> str:
    # Get file query
    directory_string = stringify_directory(directory)
    
    # Use the original reading_prompt but with better instructions
    query_prompt = reading_prompt.format(
        problem_statement=problem_statement,
        directory_string=directory_string
    )
    
    print("\nSending query prompt...")
    query_response = model.generate(query_prompt)
    print(f"\nQuery response: {query_response}")
    
    file_query = extract_file_query(query_response)
    print("\nFile query:", file_query)
    
    if not file_query:
        print("No valid file query extracted")
        return None

    # Get file contents
    file_content_string = fetch_file_contents(file_query)
    print("\nFile contents:", file_content_string)

    # Generate patch with the original patching prompt
    patch_prompt = patching_prompt.format(
        problem_statement=problem_statement,
        file_content_string=file_content_string
    )
    
    print("\nSending patch prompt...")

    patch_response = model.generate(patch_prompt)
    print(f"\nPatch response: {patch_response}")
    
    patch_string = extract_patch_string(patch_response)
    print("\nGenerated patch:", patch_string)
    return patch_string

def predict(model: GitPatchModel, 
            problem_statement: str, 
            repo_archive: io.BytesIO) -> Optional[str]:
    # Extract repository
    with open('repo_archive.tar', 'wb') as f:
        f.write(repo_archive.read())
        
    if os.path.exists(REPO_PATH):
        shutil.rmtree(REPO_PATH)
    shutil.unpack_archive('repo_archive.tar', extract_dir=REPO_PATH)
    os.remove('repo_archive.tar')

    try:
        clear_cuda_memory()
        optimize_model_memory(model)
        patch_string = predict_inner(model, problem_statement, REPO_PATH)
    finally:
        shutil.rmtree(REPO_PATH)
    
    return patch_string


!mkdir -p /kaggle/tmp/konwinski-prize-alt
!unzip -q -o /kaggle/input/konwinski-prize/data.a_zip -d /kaggle/tmp/konwinski-prize-alt/ 2>/dev/null || true



# torch.cuda.empty_cache()
# torch.cuda.memory_summary()



from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    AutoConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if __name__ == "__main__":
    # Training
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        train_model(
            model_path=model_path,
            data_path='/kaggle/tmp/konwinski-prize-alt/data/data.parquet',
            repos_dir='/kaggle/tmp/konwinski-prize-alt/data/repos',
            output_dir='./git-patch-model'
        )
    
    # Inference
    def get_problem(problem_index):
        df = pd.read_parquet('/kaggle/tmp/konwinski-prize-alt/data/data.parquet')
        problem_statement = df["problem_statement"][problem_index]
        repo_path = f"/kaggle/tmp/konwinski-prize-alt/data/repos/repo__{df['instance_id'][problem_index]}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.make_archive(os.path.join(tmpdir, 'a_repo'), 'tar', repo_path)
            with open(os.path.join(tmpdir, 'a_repo.tar'), 'rb') as f:
                repo_archive = io.BytesIO(f.read())
        
        return problem_statement, repo_path, repo_archive
    
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Load trained model
        model = GitPatchModel(model_path)
        model.model.load_adapter("./git-patch-model", adapter_name="default")

        
        # Test on demo problemd
        demo_problem_index = 0
        problem_statement, repo_path, repo_archive = get_problem(demo_problem_index)
        clear_cuda_memory()
        optimize_model_memory(model)
        patch_string = predict(model, problem_statement, repo_archive)
        print("\nFinal patch:")
        print(patch_string)

