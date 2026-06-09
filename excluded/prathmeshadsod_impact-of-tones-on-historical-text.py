# Install required libraries

!pip install transformers peft bitsandbytes trl
# transformers: For leveraging pre-trained models like Gemma 2 from Hugging Face and fine-tuning them for specific tasks.
# peft: For Parameter-Efficient Fine-Tuning (PEFT) techniques such as QLora, which optimize memory and computational efficiency during fine-tuning.
# bitsandbytes: For low-bit quantization (like 8-bit and 4-bit), which reduces model size and computational overhead.
# trl: Using STFTrainer for instruction tuning from trl.

!pip install evaluate rouge_score
# evaluate: For calculating performance metrics like ROUGE, BLEU, and perplexity to assess the fine-tuned model's quality.
# rouge_score: quantify the similarity between machine-generated text and reference texts



#If you are using models directly from huggingface then you have to
#request access for Gemma family on huggingface then after gettin acces you can
#login through below code with huggingface in kaggle notebook environment

#In my case I downloaded the Gemma2 model and uploaded it in dataset and
#used it from input directory of Kaggle
from huggingface_hub import notebook_login
notebook_login()


#Import required modules
import pandas as pd
import numpy as np
import re
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraModel, LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM



import warnings
# Suppressing warnings to make the output cleaner for this specific script.
# Note: It is generally not recommended to ignore warnings as they can provide 
# valuable insights about potential issues in the code or environment. 
# Addressing warnings instead of suppressing them ensures better debugging and stability.
warnings.filterwarnings('ignore')



#Import created dataset from this notebook -- https://www.kaggle.com/code/prathmeshadsod/dataset-preparation-for-tones-fine-tuning
historical_text_tones = pd.read_csv('/kaggle/input/tone-analysis-historic-text/historical_text_tones.csv')


#Shuffle the rows of the dataset with resetting index
#In our dataset we have text, quotes, sentences connected with self types
#so we are shuffling them
historical_text_tones = historical_text_tones.sample(frac=1).reset_index(drop=True)


#preprocessing on historical text as there are some extra tokens in few rows
historical_text_tones['historical_text'] = historical_text_tones['historical_text'].apply(
    lambda x: re.sub(r',"\s*$', '', x)
)


# Function to count words in a text
def word_count(text):
    return len(str(text).split())

# Add a new column for the total word count of both columns
historical_text_tones['total_word_count'] = historical_text_tones['historical_text'].apply(word_count) + historical_text_tones['output_tones'].apply(word_count)

# Filter rows where the total word count is less than 160
# doing this because of hardware limitations we want to keep training 
# less compute exauhstive by keeping sequence length under control
# so keeping total words count in dataset less than 160
historical_text_tones = historical_text_tones[historical_text_tones['total_word_count'] < 160]


historical_text_tones.head()


print("Total historical_text_tones length ", len(historical_text_tones))


historical_text_tones = historical_text_tones.drop('total_word_count', axis=1)


#convert pandas dataframe to support with huggingface SFTTrainer
historical_text_tones_hf = Dataset.from_pandas(historical_text_tones)


#Drop column automatically added by default __index_level_0__
historical_text_tones_hf = historical_text_tones_hf.remove_columns("__index_level_0__")
historical_text_tones_hf


# Take 20% of available rows to make it unseen data so later on we can do validation
# and work them with metrics like bleu, perplexity, rogue

train_test_split = historical_text_tones_hf.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


# Configuring the BitsAndBytes library to load the model with 4-bit quantization
config_4_bit = BitsAndBytesConfig(
    # Enabling 4-bit quantization to reduce memory usage and increase computation efficiency
    load_in_4bit=True,
    
    # Choosing the 'nf4' quantization type for optimal model performance and precision
    bnb_4bit_quant_type="nf4",
    
    # Enabling double quantization for further memory optimization without losing too much accuracy
    bnb_4bit_use_double_quant=True,
    
    # Setting the computation data type to 'bfloat16' for faster performance while maintaining numerical stability
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# Defining the path to the pre-trained Gemma 2 model that we will fine-tune
model_name = "/kaggle/input/gemma-2-2b-it-transformer/gemma-2-transformers-gemma-2-2b-it-v2"

# Loading the tokenizer for the Gemma 2 model to process input data
gemma2_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Loading the pre-trained Gemma 2 model for causal language modeling
gemma2_model = AutoModelForCausalLM.from_pretrained(
     model_name,
     # Using bfloat16 data type for better performance on GPUs, as it reduces memory usage
     torch_dtype=torch.bfloat16,  
     # Automatically assigning model layers to available devices (like GPU or CPU)
     device_map="auto",           
     # Applying 4-bit quantization for memory efficiency using the pre-defined config
     quantization_config=config_4_bit,  
     # Setting attention implementation to 'eager' for better debugging and easier profiling
     attn_implementation='eager'
)
gemma2_tokenizer.padding_side = 'right'
gemma2_tokenizer.truncation = True


# Enabling gradient checkpointing to reduce memory usage during training
# This allows us to store intermediate results during backpropagation, saving memory while training large models.
gemma2_model.gradient_checkpointing_enable()

# Preparing the model for k-bit training, specifically for efficient quantization techniques
# This step optimizes the model for training with low-bit precision (4-bit) while maintaining performance.
gemma2_model = prepare_model_for_kbit_training(gemma2_model)



gemma2_model.eval()


# Define LoRA (Low-Rank Adaptation) configuration for fine-tuning the model on historical tone analysis task
qlora_config = LoraConfig(
    #In our use case, historical texts (e.g., speeches, quotes) need to be analyzed 
    #for their tone and significance. Sequence-to-sequence learning is ideal for 
    #handling such input-output text pairs, where the input is a historical passage,
    #and the output is the corresponding tone with its significance in percentages.
    #This type of task aligns with the need for understanding both the content and its contextual tone.
    task_type=TaskType.SEQ_2_SEQ_LM,   

    # Control the rank (dimensionality) of the learned adaptation matrices for LoRA layers.
    # Since we are working with a relatively small dataset of around 5000 rows, a moderate rank allows the model
    # to capture relevant patterns and nuances in historical text without overloading memory. 
    # We use a rank of 4 to allow the model to adapt well without excessive computational overhead.
    r=4,              

    # Regulate the contribution of LoRA updates to the original model's outputs.
    # We use a value of 16 for lora_alpha to ensure that the LoRA updates influence the model enough 
    # without overwhelming the original pre-trained model, maintaining a balance between fine-tuning and avoiding overfitting.
    lora_alpha=16,   

    #Regularization is essential to prevent overfitting, especially with a dataset of only
    #5000 rows. By setting the dropout to 0.3, we encourage the model to generalize better
    #and not memorize the training data. This is important because our data involves 
    #complex historical texts, and we don’t want the model to overfit to any one specific
    #example. A moderate dropout helps ensure the model remains adaptable and capable of 
    #handling new, unseen tones in other historical contexts.
    lora_dropout=0.3 ,

    # Bias setting that controls how the LoRA updates are applied.
    # Here, we use "lora_only" to ensure we are only fine-tuning the LoRA layers, not affecting the core weights of the model.
    # This helps maintain the integrity of the pre-trained knowledge while adapting it for our tone-specific task.
    bias="lora_only" ,

    # Set inference_mode to False because we are in training mode.
    # We need to adjust the model weights during training, so inference mode is disabled.
    # This ensures that the model will learn from the training data, fine-tuning its weights based on the tone analysis task.
    inference_mode=False,    
)



# Apply the LoRA configuration to the model, preparing it for efficient fine-tuning with the specified qlora_config
qlora_model = get_peft_model(gemma2_model, qlora_config)



#print qlora_models trainable parameters
qlora_model.print_trainable_parameters()


# Function to format the dataset into prompts for fine-tuning the model
def format_prompt(dataset):
    output_texts = []
    for i in range(len(dataset['historical_text'])):
        prompt = f""" 
            <start_of_turn>user\n
            write a tone impact analysis for the following historical text :\n"
            {dataset['historical_text'][i]}\n"
            <end_of_turn>\n
            <start_of_turn>model\n
            Tone Impact Analysis:\n
            {dataset['output_tones'][i]}\n
            <end_of_turn>
        """
        output_texts.append(prompt)  # Append the formatted prompt to the output list
    return output_texts  # Return the list of formatted prompts

# Function to format the dataset into prompts for testing, where the model's response is not provided
def format_prompt_test(dataset):
    output_texts = []
    for i in range(len(dataset['historical_text'])):
        prompt = f""" 
            <start_of_turn>user\n
            write a tone impact analysis for the following historical text :\n"
            {dataset['historical_text'][i]}\n"
            <end_of_turn>\n
            <start_of_turn>model\n
            
            <end_of_turn>
        """
        output_texts.append(prompt)  # Append the prompt for test to the output list
    return output_texts  # Return the list of test prompts



# Define a response template for the model, which will guide the generation of tone impact analysis
response_template = "<start_of_turn>model"

#This is a utility that ensures the input data is tokenized properly and aligned 
#with the model's expected format for training. It uses the tokenizer and the response 
#template to prepare data for the SFTTrainer
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=gemma2_tokenizer)


SFTArguments = SFTConfig(
    max_seq_length=220,                      # Reduce sequence length
    per_device_train_batch_size=8,           # Larger batch size
    gradient_accumulation_steps=2,           # Simulate larger batch size
    learning_rate=5e-4,                      # Adjust learning rate
    num_train_epochs=6,                      # Fewer epochs
    output_dir="./gemma2_fine_tuned_on_historical_texts",
    logging_steps=50,
    save_steps=50,                          # Save checkpoints periodically
    fp16=True,                               # Mixed precision for speed
    report_to="none",
)



trainer = SFTTrainer(
    qlora_model,
    train_dataset=train_dataset,
    args=SFTArguments,
    formatting_func=format_prompt,
    data_collator=data_collator,
)


trainer.train()


#trainer.train(resume_from_checkpoint=True)


# Save the model, tokenizer, and state
trainer.save_model(output_dir="./gemma-2-fine-tuned_model")    # Saves the model and config
gemma2_tokenizer.save_pretrained("./gemma-2-fine-tuned_model")   # Saves tokenizer files
trainer.save_state()                 # Saves the training state


import shutil  # Importing the shutil library to handle file operations like compression

# Define the zip file name
zip_filename = "gemma-2-fine-tuned_model"  # Name for the zip file that will contain the model
output_dir = "/kaggle/working/"  # Directory where the model files are saved (Kaggle working directory)

# Compress the directory into a zip file
shutil.make_archive(zip_filename, 'zip', output_dir)  
# This function compresses the specified directory into a zip file. 
# 'zip' indicates the compression format, and output_dir is the directory to compress.

print(f"Model saved and zipped as {zip_filename}")  # Prints a confirmation message showing the zip file name



model_name = '/kaggle/input/gemma-2-2b-hf-tones-impact-on-historical-text/transformers/2b/2/gemma-2-fine-tuned_model'
model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.bfloat16,
                                quantization_config = config_4_bit,
        )
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

## blow is output of fine tuned Gemma 2


def cleanup_output(prompt) :
    # Using find method
    start_index = prompt.find('<start_of_turn>model')
    
    # If the substring is found, slice from the end of the match
    if start_index != -1:
        result = prompt[start_index + len('<start_of_turn>model'):].strip()
    else:
        result = prompt
    return result


import tqdm  # Import the tqdm library for displaying progress bars during iteration.

# Define a function to preprocess input data and generate predictions using the model.
def preprocess_and_generate_predictions(dataset, model, tokenizer):
    model.eval()  # Set the model to evaluation mode to ensure it doesn't perform unnecessary updates during inference.
    predictions = []  # Initialize an empty list to store the generated predictions.
    references = []  # Initialize an empty list to store the reference values (the correct output tones).

    # Format the prompts using the provided function that structures the input text.
    prompts = format_prompt_test(dataset)
   
    with torch.no_grad():  # Use no_grad() context to disable gradient calculation, as we only need inference.
        # Iterate over the dataset using tqdm for showing a progress bar.
        for index in tqdm.tqdm(range(len(dataset))):
            # Tokenize the input text (preprocessing step).
            # The tokenizer converts the textual prompt into input tokens that the model can understand.
            inputs = tokenizer(
                prompts[index],  # Take the prompt for the current index.
                max_length=220,  # Limit the length of the tokenized input to 220 tokens.
                truncation=True,  # Truncate any inputs longer than the max_length.
                return_tensors="pt",  # Return the tokenized input as PyTorch tensors.
            )

            # Generate a prediction using the model.
            # The model will predict the next tokens based on the input sequence.
            outputs = model.generate(
                **inputs,  # Pass the tokenized inputs to the model.
                max_new_tokens=220,  # Limit the number of new tokens generated by the model.
            )

            # Decode the prediction back to text using the tokenizer's decode method.
            # This converts the model-generated token IDs into a human-readable string.
            prediction = tokenizer.decode(outputs[0])

            # Get the reference text for comparison. This is the ground truth or actual tone that should have been predicted.
            reference = dataset["output_tones"][index]  # 'output_tones' contains the true labels.

            # Clean the generated output (remove any unwanted tokens) and append to predictions.
            predictions.append(cleanup_output(prediction).replace("<end_of_turn>", ""))

            # Append the reference (true label) to the references list.
            references.append(reference)

    return predictions, references  # Return the list of predictions and references.

# Use the function to get predictions and references from a subset of the dataset.
# 'eval_dataset.take(20)' takes the first 20 samples from the dataset for evaluation.
predictions, references = preprocess_and_generate_predictions(eval_dataset.take(20), model, tokenizer)



predictions


import torch  # Importing PyTorch for tensor operations and model evaluation.
from evaluate import load  # Importing the 'evaluate' library to load common NLP evaluation metrics like ROUGE and BLEU.
import math  # Importing the math library to calculate perplexity.

# Load the ROUGE and BLEU metrics using the 'evaluate' library.
# These metrics will help assess how well the model's predictions match the ground truth.
rouge_metric = load("rouge")
bleu_metric = load("bleu")

# Define a function to calculate perplexity, which measures how well the model predicts the next word in a sequence.
# A lower perplexity indicates better performance.
def calculate_perplexity(predictions, tokenizer):
    total_log_likelihood = 0  # Variable to accumulate the log likelihood of all predictions.
    total_words = 0  # Variable to count the total number of words across all predictions.

    # Iterate through each prediction in the predictions list.
    for pred in predictions:
        # Tokenize the predicted text using the tokenizer. This converts the text into token IDs that the model can process.
        tokenized_pred = tokenizer(pred, return_tensors="pt", truncation=True, padding=True)
        input_ids = tokenized_pred["input_ids"].to(model.device)  # Move tokenized input to the model's device (GPU/CPU).
        
        # Compute the log likelihood of the prediction using the model.
        # This helps evaluate how well the model can predict the next word based on the given input.
        with torch.no_grad():  # Disable gradient calculations for efficiency during inference.
            outputs = model(input_ids=input_ids, labels=input_ids)  # Pass the input through the model.
            log_likelihood = outputs.loss.item()  # Get the loss value, which corresponds to the log likelihood.
            total_log_likelihood += log_likelihood  # Add the log likelihood to the total.
            total_words += input_ids.size(1)  # Count the number of words (tokens) in the prediction.

    # Perplexity is calculated by exponentiating the average log likelihood per word.
    # The formula is: Perplexity = exp(total_log_likelihood / total_words)
    perplexity = math.exp(total_log_likelihood / total_words)
    return perplexity

# Define a function to compute BLEU and ROUGE metrics, which evaluate the quality of the model's predictions.
def compute_metrics(predictions, references):
    # Compute BLEU score using the BLEU metric.
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    
    # Compute ROUGE score using the ROUGE metric.
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    
    # Return both the BLEU and ROUGE scores.
    return bleu_score, rouge_score

# Calculate BLEU and ROUGE scores for the predictions by comparing them to the references (ground truth).
bleu_score, rouge_score = compute_metrics(predictions, references)

# Calculate the Perplexity score to measure how well the model is predicting the text.
perplexity_score = calculate_perplexity(predictions, tokenizer)

# Display the evaluation results.
print("BLEU Score:", bleu_score)  # Print the BLEU score, which measures how similar the generated text is to the reference.
print("ROUGE Score:", rouge_score)  # Print the ROUGE score, which assesses the overlap between the predicted and reference text.
print("Perplexity Score:", perplexity_score)  # Print the Perplexity score, which measures the quality of the model's predictions.





