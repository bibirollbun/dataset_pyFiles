import pandas as pd
from typing import Dict, List
import zipfile
import re
import os

def unzip_data():
    """Unzip the data.a_zip file"""
    try:
        with zipfile.ZipFile('/kaggle/input/konwinski-prize/data.a_zip', 'r') as zip_ref:
            zip_ref.extractall('/kaggle/working/data')
        print("Data successfully unzipped!")
    except Exception as e:
        print(f"Error unzipping data: {e}")

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def extract_code_snippets(self, problem_statement: str) -> List[str]:
        """Extract code snippets between triple backticks"""
        try:
            code_pattern = r"```(?:py|python)?\n(.*?)\n```"
            return re.findall(code_pattern, str(problem_statement), re.DOTALL)
        except Exception as e:
            print(f"Error extracting code snippets: {e}")
            return []
    
    def parse_patch(self, patch: str) -> Dict:
        """Parse git diff to understand file changes"""
        try:
            files_changed = []
            for line in str(patch).split('\n'):
                if line.startswith('diff --git'):
                    files_changed.append(line.split()[-1][2:])
            return {
                'files_changed': files_changed,
                'raw_patch': patch
            }
        except Exception as e:
            print(f"Error parsing patch: {e}")
            return {'files_changed': [], 'raw_patch': ''}
    
    def process_single_issue(self, row) -> Dict:
        """Process a single GitHub issue"""
        try:
            try:
                pass_to_pass = eval(str(row['PASS_TO_PASS']).replace('\x00', ''))
            except:
                pass_to_pass = []
                
            try:
                fail_to_pass = eval(str(row['FAIL_TO_PASS']).replace('\x00', ''))
            except:
                fail_to_pass = []
                
            return {
                'id': str(row['instance_id']),
                'repo': str(row['repo']),
                'problem': {
                    'description': str(row['problem_statement']),
                    'code_snippets': self.extract_code_snippets(str(row['problem_statement'])),
                    'error_type': self.extract_error_type(str(row['problem_statement']))
                },
                'solution': self.parse_patch(str(row['patch'])),
                'tests': {
                    'pass_to_pass': pass_to_pass,
                    'fail_to_pass': fail_to_pass
                }
            }
        except Exception as e:
            print(f"Error processing single issue: {e}")
            return {}
    
    def extract_error_type(self, problem_statement: str) -> str:
        """Extract the type of error from problem statement"""
        try:
            error_pattern = r"([A-Za-z]+Error:)"
            matches = re.findall(error_pattern, str(problem_statement))
            return matches[0] if matches else "Unknown"
        except Exception as e:
            print(f"Error extracting error type: {e}")
            return "Unknown"
    
    def process_all_data(self) -> List[Dict]:
        """Process all issues in the dataset"""
        processed_data = []
        for _, row in self.df.iterrows():
            try:
                processed_issue = self.process_single_issue(row)
                if processed_issue:  # Only add if not empty
                    processed_data.append(processed_issue)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        return processed_data

def load_and_process_data():
    """Load and process the data"""
    try:
        # First unzip the data
        unzip_data()
        
        # Read the parquet file with handling for null bytes
        df = pd.read_parquet('/kaggle/working/data/data/data.parquet')
        
        # Clean null bytes from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.replace('\x00', '')
        
        # Create processor instance and process data
        processor = DataProcessor(df)
        processed_data = processor.process_all_data()
        
        if processed_data:
            print(f"\nTotal issues processed: {len(processed_data)}")
            error_types = [issue['problem']['error_type'] for issue in processed_data]
            print(f"Error types found: {set(error_types)}")
        
        return processed_data
    
    except Exception as e:
        print(f"Error in data processing: {e}")
        return None

if __name__ == "__main__":
    processed_data = load_and_process_data()


!pip install -U bitsandbytes
!pip install peft




import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
import logging
import warnings
from difflib import SequenceMatcher
import numpy as np

warnings.filterwarnings('ignore')

class PatchGeneratorTrainer:
    def __init__(self, data_list, model_path="/kaggle/input/qwen2.5/transformers/0.5b/1", max_length=512):
        """
        Initialize the PatchGenerator with data and model configuration.
        
        Args:
            data_list: List of dictionaries containing the training data
            model_path: Path to the pre-trained model
            max_length: Maximum sequence length for tokenization
        """
        self.data = data_list
        self.model_path = model_path
        self.max_length = max_length
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_prompt(self, issue_data):
        """Generate a structured prompt for the model"""
        prompt = f"""Below is a Python code issue that needs to be fixed.

Error Type: {issue_data['problem']['error_type']}

Problem Description:
{issue_data['problem']['description']}

The patch should fix the code issue while being minimal and focused.

Generate the patch in git diff format:"""

        return prompt
    def prepare_data(self, data_list):
        """Prepare dataset from list of examples"""
        training_pairs = []
        
        for item in data_list:
            try:
                prompt = self.create_prompt(item)
                patch = item['solution']['raw_patch']
                
                # Only include examples with valid patches
                if patch and patch.strip():
                    training_pairs.append({
                        'input_text': prompt,
                        'output_text': patch
                    })
            except KeyError as e:
                self.logger.warning(f"Skipping malformed data entry: {e}")
                continue
                
        return training_pairs

    def load_model_and_tokenizer(self):
        """Load and configure the model and tokenizer"""
        self.logger.info("Loading tokenizer and model...")
        
        try:
            # Load model configuration
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            
            # Load pre-trained model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                offload_folder="offload",
                offload_state_dict=True,
                max_memory={0: "12GB", "cpu": "24GB"}
            )
            
            # Enable gradient checkpointing and prepare for training
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Apply PEFT using LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "self_attn.q_proj", 
                    "self_attn.k_proj", 
                    "self_attn.v_proj", 
                    "self_attn.o_proj",
                    "mlp.gate_proj", 
                    "mlp.up_proj", 
                    "mlp.down_proj"
                ]
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
            self.logger.info("Model loaded successfully with PEFT!")
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Error during model initialization: {str(e)}")
            raise e

    def create_datasets(self, data_pairs, test_size=0.3, seed=42):
        """Create and split datasets for training and testing"""
        train_pairs, test_pairs = train_test_split(
            data_pairs, 
            test_size=test_size, 
            random_state=seed
        )
        
        def preprocess_function(examples):
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples['input_text'],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Tokenize outputs
            labels = self.tokenizer(
                examples['output_text'],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            "input_text": [pair["input_text"] for pair in train_pairs],
            "output_text": [pair["output_text"] for pair in train_pairs]
        })
        
        test_dataset = Dataset.from_dict({
            "input_text": [pair["input_text"] for pair in test_pairs],
            "output_text": [pair["output_text"] for pair in test_pairs]
        })
        
        # Process datasets
        self.train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing training data"
        )
        
        self.test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Preprocessing evaluation data"
        )
        
        self.test_pairs = test_pairs
        return self.train_dataset, self.test_dataset

    def train(self, output_dir="/kaggle/working/qwen-patch-generator"):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=150,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            learning_rate=1e-5,
            logging_dir="./logs",
            fp16=True,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,
            evaluation_strategy="no",
            remove_unused_columns=True,
            report_to="none"
        )
        
        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=data_collator,
        )
        
        try:
            self.logger.info("Starting training...")
            trainer.train()
            
            final_model_path = f"{output_dir}/final"
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise e

    def evaluate_model(self, similarity_threshold=0.7):
        """Evaluate model performance on test set"""
        self.logger.info("Starting model evaluation...")
        
        similarities = []
        successes = 0
        total = len(self.test_pairs)
        
        for idx, test_pair in enumerate(self.test_pairs):
            try:
                generated_patch = self.generate_patch(test_pair['input_text'])
                similarity = self.evaluate_patch(generated_patch, test_pair['output_text'])
                similarities.append(similarity)
                
                if similarity >= similarity_threshold:
                    successes += 1
                
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Evaluated {idx + 1}/{total} test cases...")
                
            except Exception as e:
                self.logger.error(f"Error evaluating test case {idx}: {e}")
                continue
        
        avg_similarity = np.mean(similarities)
        accuracy = successes / total
        
        results = {
            'average_similarity': avg_similarity,
            'accuracy': accuracy,
            'total_test_cases': total,
            'successful_generations': successes
        }
        
        self.logger.info(f"Evaluation Results: {results}")
        return results

    def generate_patch(self, prompt):
        """Generate a patch for a given problem"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_patch(self, generated_patch, reference_patch):
        """Evaluate the similarity between generated and reference patches"""
        print("Generated Patch")
        print(generated_patch)
        print('---------------------------------------\n')
        similarity = SequenceMatcher(None, generated_patch, reference_patch).ratio()
        return similarity


def main():
    # Load and process the data
    print("Loading and processing data...")
    processed_data = load_and_process_data()
    
    if not processed_data:
        print("Error: No data processed. Exiting.")
        return
    
    # Initialize trainer
    trainer = PatchGeneratorTrainer(processed_data)
    
    # Prepare data for training
    print("\nPreparing data for training...")
    data_pairs = trainer.prepare_data(processed_data)
    print(f"Total training pairs created: {len(data_pairs)}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    trainer.load_model_and_tokenizer()
    
    # Create datasets
    print("\nCreating train/test splits...")
    train_dataset, test_dataset = trainer.create_datasets(data_pairs)
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Train model
    print("\nStarting training...")
    trained_trainer = trainer.train()
    
    # Evaluate model
    print("\nEvaluating model...")
    results = trainer.evaluate_model()
    
    print("\nFinal Results:")
    print(f"Average Similarity: {results['average_similarity']:.2f}")
    print(f"Accuracy (>= 0.7 similarity): {results['accuracy']:.2f}")
    print(f"Total Test Cases: {results['total_test_cases']}")
    print(f"Successful Generations: {results['successful_generations']}")


if __name__ == "__main__":
    main()




