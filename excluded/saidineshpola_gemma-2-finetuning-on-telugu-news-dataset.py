import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # ADD if you have more
# Install newer versions of PEFT, evaluate, transformers, accelerate, and bitsandbytes packages quietly without showing output.
%pip install -q -U peft evaluate rouge_score transformers accelerate bitsandbytes trl datasets ollama

# If torch version fails with use this
# !python -m pip uninstall torch torchvision torchaudio
# !python -m pip install --pre torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121


import os

import torch

import numpy as np
import pandas as pd
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          AutoConfig,
                          TrainingArguments)
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from IPython.display import Markdown, display, HTML
import evaluate
from tqdm import tqdm
import gc
import json
from typing import List, Dict, Literal, Tuple
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import math
import seaborn as sns
import matplotlib.pyplot as plt


# Disable CA bundle check. Useful in certain environments where you may encounter SSL errors.
os.environ['CURL_CA_BUNDLE'] = ''

# Set the order of devices as seen by CUDA to PCI bus ID order. This is to ensure consistency in device selection.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Check if CUDA is available, and if so, specify which GPU(s) to be made visible to the process.
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")


# Wandb for experiment tracking
import wandb

# Initialize Weights & Biases (wandb) for experiment tracking.
# If a wandb account exists, it can typically be used by specifying project and entity.
# However, for this example, we're disabling wandb to ignore it by setting mode to "disabled".
wandb.init(mode="disabled")


# load dataset from huggingface called saidines12/telugu_news_dataset
from datasets import load_dataset

dataset = load_dataset('saidines12/telugu_news_dataset',
                        trust_remote_code=True
                    )
dataset['validation'][10]


question_column = "article"
answer_column = "headline"


telugu_news_dataset = dataset

def calculate_text_statistics(dataset, splits=['train', 'validation', 'test']):
    """
    Calculate comprehensive statistics for article-headline dataset
    """
    stats = {}
    
    for split in splits:
        if split in dataset:
            current_split = dataset[split]
            
            # Calculate article statistics
            article_sentences = [len(str(x['article']).split('.')) for x in current_split]
            article_words = [len(str(x['article']).split()) for x in current_split]
            
            # Calculate headline statistics
            headline_sentences = [len(str(x['headline']).split('.')) for x in current_split]
            headline_words = [len(str(x['headline']).split()) for x in current_split]
            
            stats[split] = {
                '# Article-Headline pairs': len(current_split),
                'Avg sentences in article': np.mean(article_sentences).round(2),
                'Avg sentences in headline': np.mean(headline_sentences).round(2),
                'Avg words in article': np.mean(article_words).round(2),
                'Avg words in headline': np.mean(headline_words).round(2),
                '(Min, Max) words in article': (min(article_words), max(article_words)),
                '(Min, Max) words in headline': (min(headline_words), max(headline_words))
            }
    
    return stats

def display_statistics_table(stats):
    """
    Create and display a formatted statistics table
    """
    # Convert stats to DataFrame
    df = pd.DataFrame(stats).round(2)
    
    # Style the DataFrame
    styled_df = df.style.set_properties(**{
        'background-color': '#E6E6FA',
        'border': '1px solid black',
        'padding': '8px',
        'color': 'black'  # Added explicit text color
    }).set_table_styles([
        {'selector': 'th',
         'props': [('background-color', '#4169E1'),
                  ('color', 'white'),
                  ('font-weight', 'bold'),
                  ('padding', '8px')]},
        {'selector': 'td',
         'props': [('text-align', 'center'),
                  ('color', 'black')]},  
        {'selector': 'table',
         'props': [('color', 'black')]}  
    ])
    
    display(HTML("<h2>'Telugu News Dataset' Statistics</h2>"))
    display(styled_df)

def plot_length_distribution(dataset, split='train'):
    """
    Plot length distributions for articles and headlines
    """
    plt.figure(figsize=(15, 5))
    
    # Article length distribution
    plt.subplot(1, 2, 1)
    article_lengths = [len(str(x['article']).split()) for x in dataset[split]]
    sns.histplot(article_lengths, bins=50)
    plt.title(f'Article Length Distribution ({split} split)')
    plt.xlabel('Number of words')
    plt.ylabel('Count')
    
    # Headline length distribution
    plt.subplot(1, 2, 2)
    headline_lengths = [len(str(x['headline']).split()) for x in dataset[split]]
    sns.histplot(headline_lengths, bins=50)
    plt.title(f'Headline Length Distribution ({split} split)')
    plt.xlabel('Number of words')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Usage example:
# Assuming your dataset is loaded as 'telugu_news_dataset'
stats = calculate_text_statistics(telugu_news_dataset)
display_statistics_table(stats)
plot_length_distribution(telugu_news_dataset)


dataset['validation'][10]



# Checking for the available device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available devices print
print("device:",device)

# Defining the path to the pre-trained model
model_path = "google/gemma-2-2b-it"

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Defining BitsAndBytesConfig
bnbConfig = BitsAndBytesConfig(
    load_in_4bit=True, # Enable loading of the model in 4-bit quantized format.
    bnb_4bit_quant_type="nf4", # Specify the quantization type. "nf4" refers to a specific 4-bit quantization scheme.
    bnb_4bit_compute_dtype=torch.bfloat16, # Define the data type for computations. bfloat16 offers a good balance between precision and speed.
)

# USE THIS FOR QUANTIZATION if you want to quantize the model
disable_quantization = True

if disable_quantization:
    model = AutoModelForCausalLM.from_pretrained(model_path,  device_map="auto",
                                                 torch_dtype=torch.bfloat16, attn_implementation='eager')
else:
    # Loading the model for causal language modeling
    model = AutoModelForCausalLM.from_pretrained(
                                                model_path,
                                                device_map="auto",
                                                quantization_config=bnbConfig 
                                                )
model.config.use_cache = False
tokenizer.add_eos_token = True  # We'll add <eos> at the end
# Note: Move the model to the specified computing device (CPU or GPU) for single/0 GPU.
# model = model.to(device)


# Print a summary of the model to understand its architecture and the number of parameters.
model


# Define a template for formatting instructions and responses.
# This template will be used to format the text data in a LLM structure.
# give template for generating headline from news article in telugu language
template2 = "వార్తాంశం: {question}\nశీర్షిక: {answer}" # for telugu template
template = "Generate relative, interesting, factual short headline from this news article in telugu language\n{article}\n\nResponse:\n{response}"

#template = "Generate :\n{instruction}\n\nResponse:\n{response}"


def generate_response(model, tokenizer, prompt, device, max_new_tokens=128):
    """
    This function generates a response to a given prompt using a specified model and tokenizer.

    Parameters:
    - model (PreTrainedModel): The machine learning model pre-trained for text generation.
    - tokenizer (PreTrainedTokenizer): A tokenizer for converting text into a format the model understands.
    - prompt (str): The initial text prompt to generate a response for.
    - device (torch.device): The computing device (CPU or GPU) the model should use for calculations.
    - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 128.

    Returns:
    - str: The text generated in response to the prompt.
    """
    # Convert the prompt into a format the model can understand using the tokenizer.
    # The result is also moved to the specified computing device.
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate a response based on the tokenized prompt.
    outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=max_new_tokens)

    # Convert the generated tokens back into readable text.
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract and return the response text. Here, it assumes the response is formatted as "Response: [generated text]".
    response_text = text.split("Response:")[1]
    
    return response_text


instruction = "ఘోర ప్రమాదం నుంచి కోలుకుని తిరిగి అంతర్జాతీయ క్రికెట్ ఆడుతున్న భారత వికెట్ కీపర్ రిషభ్ పంత్ ఓ అద్భుతమని పాకిస్థాన్ మాజీ కెప్టెన్ వసీమ్ అక్రమ్ కొనియాడాడు. ‘రోడ్డు ప్రమాదం తర్వాత ఎవరికైనా కోలుకునేందుకు చాలా సమయం పడుతుంది. ఇక ఆటగాడికైతే మరింత కష్టంగా ఉంటుంది. కానీ పంత్ అలా కాదు. నిజంగా తను మిరాకిల్ కిడ్. అతడిని యువతరం ఆదర్శంగా తీసుకోవాల్సిందే. ఐపీఎల్, టీ20 ప్రపంచకప్లోనూ ప్రభావం చూపి ఇప్పుడు టెస్టుల్లోనూ ఆకట్టుకుంటున్నాడు. ఆసీస్తో టెస్టు సిరీస్లోనూ తను కీలకం కానున్నాడు’ అని అక్రమ్ ప్రశంసించాడు. "


prompt = template.format(
    article=instruction,
    response="",
)

# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
response_text = generate_response(model, tokenizer, prompt, device, 256)

Markdown(response_text)


# LoRA configuration: Sets up the parameters for Low-Rank Adaptation, which is a method for efficient fine-tuning of transformers.
# USE LORA for saving memory and computation
lora_config = LoraConfig(
    r = 8,  # Rank of the adaptation matrices. A lower rank means fewer parameters to train.
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],  # Transformer modules to apply LoRA.
    task_type = "CAUSAL_LM",  # The type of task, here it is causal language modeling.
)



import evaluate

# Initialize ROUGE metric
metric = evaluate.load("rouge")

# Optimized compute_metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Convert predictions to text if they're token IDs
    if isinstance(predictions[0], (list, np.ndarray)):
        predictions = [pred for pred in predictions]
    
    # Process in smaller batches to save memory
    batch_size = 32
    scores = []
    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_scores = metric.compute(predictions=batch_preds, 
                                    references=batch_labels,
                                    use_stemmer=True)
        scores.append(batch_scores)
    
    # Aggregate scores
    final_scores = {}
    for key in scores[0].keys():
        final_scores[key] = np.mean([s[key] for s in scores])
    
    return final_scores


def formatting_func(examples):
    """
    Formats a given example (a dictionary containing question and answer list) using the predefined template.
    
    Parameters:
    - example (dict): A dictionary with keys corresponding to the columns of the dataset, such as 'article' and 'response'.
    
    Returns:
    - list: A list containing a single formatted string that combines the instruction and the response.
    """
    # Add the phrase to verify training success and format the text using the template and the specific example's instruction and response.
    # we have to return list of strings example[question_column] and example[answer_column] are list of strings
    articles = examples[question_column]
    responses = examples[answer_column]
    inputs = []
    for i in range(len(articles)):
        inputs.append(template.format(article=articles[i], response=responses[i]))

    #line = template.format(instruction=example[question_column], response=example[answer_column])
    return inputs



# Optimized training arguments
training_args = SFTConfig(
    
    per_device_train_batch_size=1, # Reduced to prevent OOM in training
    per_device_eval_batch_size=2,  # Reduced to prevent OOM in eval
    warmup_steps=25,
    max_steps=10000,
    learning_rate=2e-4,
    fp16=False, # Enable it for LORA
    logging_steps=1,
    output_dir="outputs",
    evaluation_strategy="steps", # CAN be disabled during traning as it might take long time for evaluation
    eval_steps=10000,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,  # Added to maintain effective batch size
    eval_accumulation_steps=8,  # Increased for better memory management
    dataloader_pin_memory=True,  # Added for faster data transfer to GPU
    # remove_unused_columns=True,  # Added to reduce memory usage
    optim="adamw_torch",  # Using standard AdamW for better stability or use adamw_8bit for less computing
    report_to="none",  # Disable wandb/tensorboard if not needed
    max_seq_length=2048,
    save_steps=1000,
   
)

# Initialize trainer with optimized settings
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    args=training_args,
    formatting_func=formatting_func,

)


# train the model to the processed data.
trainer.train()


# Load the model at checkpoint 7000 steps for evaluation or USE saidines12/telugu-news-headline-generation

checkpoint_path = 'saidines12/telugu-news-headline-generation' # './outputs/checkpoint-7000'
model=AutoModelForCausalLM.from_pretrained(checkpoint_path ,
                                                  device_map='auto')
# Push the model to huggingface under your user name  and model name telugu-news-headline-generation
model.push_to_hub(
    repo_id="YOURUSERNAME/telugu-news-headline-generation",
)


instruction = "అక్రమ్ కరాచీ ఘోర ప్రమాదం నుంచి కోలుకుని తిరిగి అంతర్జాతీయ క్రికెట్ ఆడుతున్న భారత వికెట్ కీపర్ రిషభ్ పంత్ ఓ అద్భుతమని పాకిస్థాన్ మాజీ కెప్టెన్ వసీమ్ అక్రమ్ కొనియాడాడు. ‘రోడ్డు ప్రమాదం తర్వాత ఎవరికైనా కోలుకునేందుకు చాలా సమయం పడుతుంది. ఇక ఆటగాడికైతే మరింత కష్టంగా ఉంటుంది. కానీ పంత్ అలా కాదు. నిజంగా తను మిరాకిల్ కిడ్. అతడిని యువతరం ఆదర్శంగా తీసుకోవాల్సిందే. ఐపీఎల్, టీ20 ప్రపంచకప్లోనూ ప్రభావం చూపి ఇప్పుడు టెస్టుల్లోనూ ఆకట్టుకుంటున్నాడు. ఆసీస్తో టెస్టు సిరీస్లోనూ తను కీలకం కానున్నాడు’ అని అక్రమ్ ప్రశంసించాడు. "


prompt = template.format(
    article=instruction,
    response="",
)

response_text = generate_response(trainer.model, tokenizer, prompt, device,32)
# TODO: Fix repitition of response

Markdown(response_text)


import torch
import evaluate
from tqdm import tqdm
import gc

def evaluate_model(model, eval_dataset, tokenizer, batch_size=32):
    """Evaluation loop for a model with properly normalized ROUGE score calculation"""
    model.eval()
    rouge = evaluate.load('rouge',
                          tokenizer=tokenizer,
                          )
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), batch_size)):
            torch.cuda.empty_cache()
            gc.collect()
            
            batch_data = eval_dataset[i:i + batch_size]
            inputs = tokenizer(batch_data['article'], 
                             padding=True, 
                             truncation=True, 
                            return_token_type_ids=False,
                             return_tensors='pt',
                             max_length=2048).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Handle multiple "Response:" occurrences
            processed_predictions = []
            for p in predictions:
                responses = p.split("Response:")
                if len(responses) > 1:
                    # Take the text between Response: tags, excluding empty strings
                    valid_responses = [r.strip() for r in responses[1:] if r.strip()]
                    if valid_responses:
                        # Take the most complete response (usually the longest)
                        processed_predictions.append(max(valid_responses, key=len))
                    else:
                        processed_predictions.append(p)  # Fallback to original if no valid responses
                else:
                    processed_predictions.append(p)  # No Response: tag found
            
            all_predictions.extend(processed_predictions)
            all_references.extend(batch_data['headline'])
            
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate ROUGE scores with proper normalization
    results = rouge.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True,
        use_aggregator=True,
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )
            
    return results, all_predictions

# Evaluation script
def compare_models(base_model, finetuned_model, eval_dataset, tokenizer):
    """Compare base and finetuned models using the fixed evaluation function"""
    print("Evaluating base model...")
    base_results, base_predictions = evaluate_model(base_model, eval_dataset, tokenizer)
    
    # Clear memory
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\nEvaluating finetuned model...")
    finetuned_results, finetuned_predictions = evaluate_model(finetuned_model, eval_dataset, tokenizer)
    
    # Print comparison
    print("\nROUGE Score Comparison:")
    print("Metric\t\tBase Model\tFinetuned Model\tImprovement")
    print("-" * 60)
    for metric in base_results.keys():
        base_score = base_results[metric] * 100
        finetuned_score = finetuned_results[metric] * 100
        improvement = finetuned_score - base_score
        print(f"{metric}:\t{base_score:.2f}\t\t{finetuned_score:.2f}\t\t{improvement:+.2f}")
    
    # Save predictions
    comparison_df = pd.DataFrame({
        'Input': eval_dataset['article'],
        'Reference': eval_dataset['headline'],
        'Base_Prediction': base_predictions,
        'Finetuned_Prediction': finetuned_predictions
    })
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    
    return base_results, finetuned_results, comparison_df


base_model= AutoModelForCausalLM.from_pretrained('google/gemma-2b-it',
                                                 device_map='auto')

# Compare models
base_results, finetuned_results, comparison_df = compare_models(
    base_model,
    model,
    dataset['validation'],
    tokenizer,
)
comparison_df.head(5)


from ollama import chat
class ComparisonResponse(BaseModel):
    comparison: Literal['Better', 'Worse', 'Same']
    explanation: str

def compare_headlines(article: str, reference_headline: str, finetuned_headline: str, model_name: str = "gemma2:9b") -> ComparisonResponse:
    """Compare finetuned headline with reference headline considering relevance, interest, factuality, and brevity."""
    prompt = f"""Compare these Telugu headlines for the given article:

Reference Headline: {reference_headline}
Finetuned Model Headline: {finetuned_headline}
Article: {article}

Compare the finetuned headline with the reference headline considering:
1. Relevance: How well headline matches article content
2. Interest: How engaging and informative it is
3. Factuality: Accuracy and truthfulness
4. Brevity: Whether it's concise yet complete

If both are good or bad, choose "Same".

Respond with JSON only, following this exact format:
{{"comparison": "Better"|"Worse"|"Same",
  "explanation": "Brief explanation of the comparison"}}
"""

    try:
        response = chat(
            messages=[{'role': 'user', 'content': prompt}],
            model=model_name,
            stream=False,
            options={'temperature': 0},
            format=ComparisonResponse.model_json_schema()
        )
        
        result = ComparisonResponse.model_validate_json(response.message.content)
        return result
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return ComparisonResponse(comparison='ERROR', explanation=str(e))

def process_batch(batch_data: List[Tuple[str, str, str]]) -> List[ComparisonResponse]:
    """Process a batch of comparisons."""
    results = []
    for article, ref_headline, finetuned_headline in batch_data:
        result = compare_headlines(
            article=article,
            reference_headline=ref_headline,
            finetuned_headline=finetuned_headline
        )
        results.append(result)
    return results

def create_batches(df: pd.DataFrame, batch_size: int) -> List[List[Tuple[str, str, str]]]:
    """Create batches of data for parallel processing."""
    data = list(zip(df['Input'], df['Reference'], df['Finetuned_Prediction']))
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def process_dataset_parallel(comparison_df: pd.DataFrame, batch_size: int = 8, max_workers: int = 4) -> List[ComparisonResponse]:
    """Process dataset in parallel using multiple threads."""
    batches = create_batches(comparison_df, batch_size)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        # Use tqdm to show progress
        for future in tqdm(future_to_batch, total=len(batches), desc="Processing batches"):
            batch_results = future.result()
            results.extend(batch_results)
    
    return results

def summarize_results(results: List[ComparisonResponse]) -> Dict:
    """Summarize the comparison results."""
    comparisons = [r.comparison for r in results]
    
    summary = {
        'Better': comparisons.count('Better'),
        'Worse': comparisons.count('Worse'),
        'Same': comparisons.count('Same'),
        'Error': comparisons.count('ERROR'),
        'Total': len(comparisons)
    }
    
    # Add percentages
    total = len(comparisons)
    summary['Better_percent'] = round(summary['Better'] * 100 / total, 2)
    summary['Worse_percent'] = round(summary['Worse'] * 100 / total, 2)
    summary['Same_percent'] = round(summary['Same'] * 100 / total, 2)
    
    return summary

def save_detailed_results(comparison_df: pd.DataFrame, results: List[ComparisonResponse]):
    """Save detailed results to Excel."""
    comparison_df['comparison_result'] = [r.comparison for r in results]
    comparison_df['explanation'] = [r.explanation for r in results]
    comparison_df.to_excel('comparison_results_with_analysis.xlsx', index=False)


# Read comparison dataset
comparison_df = pd.read_excel('model_comparison_results.xlsx')

# Process dataset in parallel with 8 items per batch and 4 worker threads
results = process_dataset_parallel(comparison_df, batch_size=16, max_workers=8)

# Print summary
summary = summarize_results(results)
print("\nComparison Results:")
print(json.dumps(summary, indent=2))



comparison_df['Comparison'] = [r.comparison for r in results]

# Print analysis
total = len(comparison_df)
same_count = (comparison_df['Comparison'] == 'Same').sum()
better_count = (comparison_df['Comparison'] == 'Better').sum()
worse_count = (comparison_df['Comparison'] == 'Worse').sum()

print("\nComparison Results:")
print("-" * 50)
print(f"Total samples: {total}")
print(f"Same predictions: {same_count} ({(same_count/total)*100:.2f}%)")
print(f"Better predictions: {better_count} ({(better_count/total)*100:.2f}%)")
print(f"Worse predictions: {worse_count} ({(worse_count/total)*100:.2f}%)")
# Save it t the same file again
comparison_df.to_excel('model_comparison_results.xlsx', index=False)

