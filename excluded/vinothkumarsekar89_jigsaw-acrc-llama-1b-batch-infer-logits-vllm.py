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


import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
os.environ['VLLM_USE_V1'] = '0'  # Force V0 for logits processor support
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import LogitsProcessor
import math


class TrueFalseLogitsProcessor(LogitsProcessor):
    """Forces model to only output True or False tokens"""
    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids
        
    def __call__(self, input_ids, scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100  # Boost allowed tokens
        return scores

class LlamaClassifier:
    def __init__(self):
        # Model path selection
        if os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
            self.model_path = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
        else:
            self.model_path = "unsloth/Llama-3.2-1B-Instruct"
        
        # Initialize model
        self.model = LLM(
            model=self.model_path,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            dtype="half",
            seed=123,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        self.setup_token_constraints()
        
        # Sampling with constrained output
        logits_processors = [TrueFalseLogitsProcessor(self.KEEP)]
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            seed=777,
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=logits_processors,
            logprobs=2
        )
    
    def setup_token_constraints(self):
        """Get token IDs for 'False' and 'True'"""
        choices = ["False", "True"]
        self.KEEP = []
        for x in choices:
            c = self.tokenizer.encode(x, add_special_tokens=False)[0]
            self.KEEP.append(c)
        
        self.false_token_id = self.KEEP[0]
        self.true_token_id = self.KEEP[1]
        print(f"Constrained to tokens: {self.KEEP} = {choices}")
    
    def create_prompt(self, input_data: pd.Series):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a really experienced moderator for the subreddit /r/{input_data['subreddit']}. 
Your job is to determine if the following reported comment violates the given rule.
Answer with only "True" or "False".

### Input:
Rule: {input_data['rule']}


Test sentence:
{self.format_comment(input_data['body'])}

### Response:
Rule violation:"""
    
    def format_comment(self, comment):
        return "\n".join(["| " + line for line in comment.split('\n')])
    
    def predict_classification(self, input_data: pd.Series):
        """Single prediction"""
        prompt = self.create_prompt(input_data)
        responses = self.model.generate([prompt], self.sampling_params, use_tqdm=False)
        
        response = responses[0]
        predicted_text = response.outputs[0].text.strip()
        
        try:
            x = response.outputs[0].logprobs[0]
            
            # Extract probabilities for True/False tokens
            logprobs = []
            for k in self.KEEP:
                if k in x:
                    logprobs.append(math.exp(x[k].logprob))
                else:
                    logprobs.append(0)
            
            logprobs = np.array(logprobs)
            logprobs /= (logprobs.sum() + 1e-15)
            
            violation_probability = logprobs[1]  # True probability
            confidence = max(logprobs)
            
        except Exception as e:
            print(f"Error: {e}")
            violation_probability = 0.5
            confidence = 0.5
        
        return {
            'prediction': predicted_text,
            'is_violation': violation_probability > 0.5,
            'violation_probability': violation_probability,
            'confidence': confidence
        }
    
    def predict_batch(self, input_data_list, verbose=False):
        """Batch predictions"""
        prompts = [self.create_prompt(data) for data in input_data_list]
        responses = self.model.generate(prompts, self.sampling_params, use_tqdm=True)
        
        results = []
        for i, response in enumerate(responses):
            try:
                predicted_text = response.outputs[0].text.strip()
                x = response.outputs[0].logprobs[0]
                
                # Extract probabilities
                logprobs = []
                for k in self.KEEP:
                    if k in x:
                        logprobs.append(math.exp(x[k].logprob))
                    else:
                        logprobs.append(0)
                
                logprobs = np.array(logprobs)
                logprobs /= (logprobs.sum() + 1e-15)
                
                violation_probability = logprobs[1]
                confidence = max(logprobs)
                
            except Exception as e:
                print(f"Error {i}: {e}")
                violation_probability = 0.5
                confidence = 0.5
                predicted_text = "Error"
            
            result = {
                'prediction': predicted_text,
                'is_violation': violation_probability > 0.5,
                'violation_probability': violation_probability,
                'confidence': confidence,
                'sample_index': i
            }
            
            # if not verbose:
            #     print(f"Sample {i+1}: {predicted_text} (prob: {violation_probability:.4f})")
            
            results.append(result)
        
        return results


model=LlamaClassifier()


# from tqdm import tqdm
# tqdm.pandas()

# # Apply the prompt function row-wise with progress bar
# result = df_train.iloc[0:1].progress_apply(model.predict_classification, axis=1)

# # Since result is a Series, access the first (and only) result
# first_result = result.iloc[0]
# print("CLASSIFICATION RESULTS:")
# print(f"Predicted: {first_result['prediction']}")
# print(f"Is Violation: {first_result['is_violation']}")
# print(f"Violation Probability: {first_result['violation_probability']:.3f}")
# print(f"Confidence: {first_result['confidence']:.3f}")
# # print(f"Raw LogProbs: {first_result['raw_logprobs']}")


# from tqdm import tqdm
# import numpy as np
# print(df_train.shape)
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
#             batch_list = [row for _, row in batch_df.iterrows()]  # Fixed: removed asterisks and fixed variable name
            
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

# # Add predictions to dataframe - extract violation probabilities from the list of dictionaries
# df_train['predicted_violation'] = [pred['violation_probability'] for pred in predictions]  # Fixed: extract from list and fixed typo


# import pandas as pd
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# # Round probabilities to binary predictions (0 or 1) using 0.5 threshold
# df_train['predicted_violation'] = (df_train['predicted_violation'] >= 0.5).astype(int)

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
print(df_train.shape)
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
            batch_list = [row for _, row in batch_df.iterrows()]  # Fixed: removed asterisks and fixed variable name
            
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
df_test['rule_violation'] = [pred['violation_probability'] for pred in predictions] 


df_test.head(2)


# write to submissions.csv
df_test[["row_id","rule_violation"]].to_csv("submission.csv",index=False)
print("wrote results to submission.csv")

