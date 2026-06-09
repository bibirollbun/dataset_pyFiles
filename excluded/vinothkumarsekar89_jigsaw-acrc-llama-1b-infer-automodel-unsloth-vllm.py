import kagglehub
import pandas as pd
import os

# Check if running on Kaggle
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
   # Running on Kaggle
   base_path = "/kaggle/input/jigsaw-agile-community-rules/"
   df_train = pd.read_csv(f"{base_path}train.csv")
   df_test = pd.read_csv(f"{base_path}test.csv")
else:
   # Running locally
   base_path = "./data/"
   df_train = pd.read_csv(f"{base_path}train.csv")
   df_test = pd.read_csv(f"{base_path}test.csv")

print(f"Using path: {base_path}")
df_train.head(2)


# # from unsloth import FastLanguageModel
# # import pandas as pd
# # import torch
# # import re
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print('device', device)

# class Load_Model_Unsloth:
#     def __init__(self):
#         ###------- for Unsloth-------------------
#         if os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
#             self.model_path = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
#         else:
#             self.model_path = "unsloth/llama-3.2-1b-instruct"

#         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
#             model_name=self.model_path,
#             max_seq_length=1024,
#             dtype=torch.float16,
#             load_in_4bit=False
#         )
#         self.model = self.model.to("cuda")
#         FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
#         #######------------------------------------
        
#         # Check model dtype and device
#         for name, param in self.model.named_parameters():
#             print(f"{name}: {param.dtype} on {param.device}")
#             break  # remove break to list all parameters

#     def format_comment(self, comment):
#         return "\n".join(["| " + line for line in comment.split('\n')])
    
#     def create_prompt(self, input_data: pd.Series):
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
#             Write a response that appropriately completes the request.

#             ### Instruction:
#             You are a really experienced moderator for the subreddit /r/{input_data['subreddit']}. 
#             Your job is to determine if the following reported comment violates the given rule.
#             Return results in {{violates rule: probability of violation between 0-1}} format as a float.
            
#             ### Input:
#             Rule: {input_data['rule']}
            
#             Example 1:
#             {self.format_comment(input_data['positive_example_1'])}
#             Rule violation: True
            
#             Example 2:
#             {self.format_comment(input_data['negative_example_1'])}
#             Rule violation: False
            
#             Example 3:
#             {self.format_comment(input_data['positive_example_2'])}
#             Rule violation: True
            
#             Example 4:
#             {self.format_comment(input_data['negative_example_2'])}
#             Rule violation: False
            
#             Test sentence:
#             {self.format_comment(input_data['body'])}
            
#             ### Response:
#             Violates rule: """

#     def get_response(self, input_data: pd.Series):
#         formatted_prompt = self.create_prompt(input_data)
#         inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(device)
        
#         outputs = self.model.generate(
#             **inputs, 
#             max_new_tokens=50,  # Shorter for classification task
#             use_cache=True,
#             temperature=0.5,  # Lower temperature for more consistent classification
#             do_sample=True,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
#         return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
#     def extract_probability(self, response_text):
#         """Extract probability from model response"""
#         import re
#         # Look for patterns like "0.85", "0.2", etc.
#         prob_pattern = r"(?:Violates rule:\s*)?(\d+\.?\d*)"
#         match = re.search(prob_pattern, response_text.split("### Response:")[-1])
        
#         if match:
#             try:
#                 prob = float(match.group(1))
#                 # Ensure probability is between 0 and 1
#                 if prob > 1.0:
#                     prob = prob / 100.0  # Convert percentage to probability
#                 return min(max(prob, 0.0), 1.0)
#             except ValueError:
#                 pass
#         return 0.5  # Default neutral probability if parsing fails
    
#     def predict(self, input_data: pd.Series) -> float:
#         """Predict if a comment violates the rule and return probability"""
#         output_decoded = self.get_response(input_data)
#         probability = self.extract_probability(output_decoded)
#         return probability



# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # import pandas as pd
# # import torch
# # import re

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print('device', device)

# class Load_Model_AutoModel:
#     def __init__(self):
#         ###------- for HuggingFace Transformers-------------------
#         if os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
#             self.model_path = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
#         else:
#             self.model_path = "unsloth/llama-3.2-1b-instruct"
        
#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_path,
#             trust_remote_code=True
#         )
        
#         # Load model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             torch_dtype=torch.float16,  # equivalent to dtype=torch.float16
#             device_map="auto",  # automatically handles device placement
#             trust_remote_code=True,
#             # load_in_4bit=False  # Remove this - use load_in_4bit=True if you want 4-bit quantization
#         )
        
#         # Set pad token if not already set
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         #######------------------------------------
        
#         # Check model dtype and device
#         for name, param in self.model.named_parameters():
#             print(f"{name}: {param.dtype} on {param.device}")
#             break  # remove break to list all parameters

#     def format_comment(self, comment):
#         return "\n".join(["| " + line for line in comment.split('\n')])
    
#     def create_prompt(self, input_data: pd.Series):
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
#             Write a response that appropriately completes the request.

#             ### Instruction:
#             You are a really experienced moderator for the subreddit /r/{input_data['subreddit']}. 
#             Your job is to determine if the following reported comment violates the given rule.
#             Return results in {{violates rule: probability of violation between 0-1}} format as a float.
            
#             ### Input:
#             Rule: {input_data['rule']}
            
#             Example 1:
#             {self.format_comment(input_data['positive_example_1'])}
#             Rule violation: True
            
#             Example 2:
#             {self.format_comment(input_data['negative_example_1'])}
#             Rule violation: False
            
#             Example 3:
#             {self.format_comment(input_data['positive_example_2'])}
#             Rule violation: True
            
#             Example 4:
#             {self.format_comment(input_data['negative_example_2'])}
#             Rule violation: False
            
#             Test sentence:
#             {self.format_comment(input_data['body'])}
            
#             ### Response:
#             Violates rule: """

#     def get_response(self, input_data: pd.Series):
#         formatted_prompt = self.create_prompt(input_data)
#         inputs = self.tokenizer(
#             [formatted_prompt], 
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=2048  # equivalent to max_seq_length from Unsloth
#         ).to(device)
        
#         with torch.no_grad():  # Add this for inference efficiency
#             outputs = self.model.generate(
#                 **inputs, 
#                 max_new_tokens=50,
#                 use_cache=True,
#                 temperature=0.5,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
#         return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
#     def extract_probability(self, response_text):
#         """Extract probability from model response"""
#         import re
#         # Look for patterns like "0.85", "0.2", etc.
#         prob_pattern = r"(?:Violates rule:\s*)?(\d+\.?\d*)"
#         match = re.search(prob_pattern, response_text.split("### Response:")[-1])
        
#         if match:
#             try:
#                 prob = float(match.group(1))
#                 # Ensure probability is between 0 and 1
#                 if prob > 1.0:
#                     prob = prob / 100.0  # Convert percentage to probability
#                 return min(max(prob, 0.0), 1.0)
#             except ValueError:
#                 pass
#         return 0.5  # Default neutral probability if parsing fails
    
#     def predict(self, input_data: pd.Series) -> float:
#         """Predict if a comment violates the rule and return probability"""
#         output_decoded = self.get_response(input_data)
#         probability = self.extract_probability(output_decoded)
#         return probability


import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Must come FIRST

import torch
import random
import numpy as np
import os
import pandas as pd
import re
from vllm import LLM, SamplingParams

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device', device)

# seed = 123
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

class Load_Model_vLLM:
    def __init__(self):
        if os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
            self.model_path = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
        else:
            self.model_path = "unsloth/Llama-3.2-1B-Instruct"

        self.model = LLM(
            model=self.model_path,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            dtype="half",
            seed=123,
            #disable_log_stats=True
        )

        self.sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.5,
            top_p=0.95,
            stop=None,
        )

    def format_comment(self, comment):
        return "\n".join(["| " + line for line in comment.split('\n')])

    def create_prompt(self, input_data: pd.Series):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a really experienced moderator for the subreddit /r/{input_data['subreddit']}. 
Your job is to determine if the following reported comment violates the given rule.
Return results in {{violates rule: probability of violation between 0-1}} format as a float.

### Input:
Rule: {input_data['rule']}

Example 1:
{self.format_comment(input_data['positive_example_1'])}
Rule violation: True

Example 2:
{self.format_comment(input_data['negative_example_1'])}
Rule violation: False

Example 3:
{self.format_comment(input_data['positive_example_2'])}
Rule violation: True

Example 4:
{self.format_comment(input_data['negative_example_2'])}
Rule violation: False

Test sentence:
{self.format_comment(input_data['body'])}

### Response:
Violates rule: """

    def get_response(self, input_data: pd.Series):
        formatted_prompt = self.create_prompt(input_data)
        outputs = self.model.generate([formatted_prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return formatted_prompt + generated_text

    def extract_probability(self, response_text):
        prob_pattern = r"(?:Violates rule:\s*)?(\d+\.?\d*)"
        match = re.search(prob_pattern, response_text.split("### Response:")[-1])
        if match:
            try:
                prob = float(match.group(1))
                if prob > 1.0:
                    prob = prob / 100.0
                return min(max(prob, 0.0), 1.0)
            except ValueError:
                pass
        return 0.5

    def predict(self, input_data: pd.Series) -> float:
        output_decoded = self.get_response(input_data)
        probability = self.extract_probability(output_decoded)
        return probability

    def predict_batch(self, input_data_list):
        prompts = [self.create_prompt(data) for data in input_data_list]
        outputs = self.model.generate(prompts, self.sampling_params)

        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            full_response = prompts[i] + generated_text
            probability = self.extract_probability(full_response)
            results.append(probability)

        return results

# Example usage (uncomment if testing in script):
# model = Load_Model_vLLM()
# result = model.predict(your_pandas_series_row)
# print(result)



model=Load_Model_vLLM()


# from tqdm import tqdm
# tqdm.pandas()
# # Apply the prompt function row-wise with progress bar
# df_train=df_train.iloc[0:100]
# df_train['prediction'] = df_train.progress_apply(model.predict, axis=1)


# from tqdm import tqdm
# import numpy as np

# def process_dataframe_in_batches(model, df, batch_size=12):
#     """Process dataframe using batch predictions with progress bar"""
    
#     # Calculate number of batches
#     num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    
#     all_results = []
    
#     # Process in batches with progress bar
#     with tqdm(total=len(df), desc="Processing predictions") as pbar:
#         for i in range(0, len(df), batch_size):
#             # Get current batch
#             batch_df = df.iloc[i:i+batch_size]
            
#             # Convert batch to list of Series (input format for predict_batch)
#             batch_list = [row for _, row in batch_df.iterrows()]
            
#             # Get predictions for this batch
#             batch_results = model.predict_batch(batch_list)
            
#             # Add to results
#             all_results.extend(batch_results)
            
#             # Update progress bar
#             pbar.update(len(batch_df))
#             pbar.set_postfix({'Batch': f'{i//batch_size + 1}/{num_batches}'})
    
#     return all_results

# # Process in batches
# predictions = process_dataframe_in_batches(model, df_train, batch_size=12)

# # Add predictions to dataframe
# df_train['rule_violation'] = predictions


# import pandas as pd
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# # Round probabilities to binary predictions (0 or 1) using 0.5 threshold
# df_train['predicted_violation'] = (df_train['prediction'] >= 0.5).astype(int)

# f1 = f1_score(df_train['rule_violation'], df_train['predicted_violation'])
# print(f"F1 Score: {f1:.4f}")

# accuracy = accuracy_score(df_train['rule_violation'], df_train['predicted_violation'])
# precision = precision_score(df_train['rule_violation'], df_train['predicted_violation'])
# recall = recall_score(df_train['rule_violation'], df_train['predicted_violation'])

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

# print("\nClassification Report:")
# print(classification_report(df_train['rule_violation'], df_train['predicted_violation']))


from tqdm import tqdm
import numpy as np

def process_dataframe_in_batches(model, df, batch_size=12):
    """Process dataframe using batch predictions with progress bar"""
    
    # Calculate number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    all_results = []
    
    # Process in batches with progress bar
    with tqdm(total=len(df), desc="Processing predictions") as pbar:
        for i in range(0, len(df), batch_size):
            # Get current batch
            batch_df = df.iloc[i:i+batch_size]
            
            # Convert batch to list of Series (input format for predict_batch)
            batch_list = [row for _, row in batch_df.iterrows()]
            
            # Get predictions for this batch
            batch_results = model.predict_batch(batch_list)
            
            # Add to results
            all_results.extend(batch_results)
            
            # Update progress bar
            pbar.update(len(batch_df))
            pbar.set_postfix({'Batch': f'{i//batch_size + 1}/{num_batches}'})
    
    return all_results

# Process in batches
predictions = process_dataframe_in_batches(model, df_test, batch_size=12)
df_test['rule_violation'] = predictions


df_test.head(2)


# write to submissions.csv
df_test[["row_id","rule_violation"]].to_csv("submission.csv",index=False)
print("wrote results to submissions.csv")

