!pip install -q tensorflow-cpu
!pip install -q -U keras-nlp tensorflow-hub
!pip install -q -U keras>=3
!pip install -q -U tensorflow-text


# needed for TPU
!pip -q install openpyxl


# needed for TPU
import openpyxl


# needed for TPU
import jax

jax.devices()


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
#!pip install -q -U keras-nlp datasets
#!pip install -q -U keras


import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

## IMPORTANT: DO NOT ENABLE THIS LINE
# Run at half precision.
#keras.config.set_floatx("bfloat16")


# Kaggle fine-tuned model
fine_tuned_model = "gemma2b_instruct_spanisheu_translationonly_los_c" # SPA - Eng viceversa translator and Spain's culture Q&A
# Kaggle Model Variant URL: kaggle.com/models/asunsada/gemma_2b_instruct_2b_es/keras/translatorandqanda_ameng-spaeu_r1
# from kaggle site
# Dynamically set Kaggle model path using fine_tuned_model
kaggle_model_path = f"asunsada/gemma_2b_instruct_2b_es/keras/{fine_tuned_model.lower()}"
kaggle_uri = f"kaggle://asunsada/gemma_2b_instruct_2b_es/keras/{fine_tuned_model}"

#kaggle_model_path = "asunsada/gemma_2b_instruct_2b_es/keras/gemma2b_instruct_spanisheu_translationonly_los_c"
# for Huggingface
repo_name =f"{fine_tuned_model.lower()}" # folder HF
username = "asunsada"  # Your Hugging Face username
# HF path to fine-tuned model
#repo_id = f"{username}/{repo_name}"

#### Data input for Inference/Evaluation
infer_path = '/kaggle/input/evaluator/Evaluator.xlsx' 
sheet_name_infer = 'Evaluator'


token_limit_infer = 250 # token limit when infering



# Util function to format the output of the Gemma response
import time

tick_start = 0


def tick():
    global tick_start
    tick_start = time.time()


def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")


def text_gen(model, prompt, token_limit):
    tick()

    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = model.generate(input, max_length=token_limit)
    
    output = output.replace(input, "") 
    # Remove unwanted characters
    characters_to_remove = ["#", "*", "[", "]", "{", "}", "<end_of_turn>", '"', "«", "»", "Traducción:", "'"]
    for char in characters_to_remove:
        output = output.replace(char, "")
    
    #print(output)
    tock()
    return output


print(kaggle_model_path)
print (repo_name)


'''
# run this only once
## upload finetuned model to HF
# Script that creates repo and uploads Kaggle model to HF.
import shutil
import kagglehub

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

secret_hf = user_secrets.get_secret("HUGGINGFACE_TOKEN")
!huggingface-cli login --token $secret_hf

# Download latest version of Kaggle model (input folder)
# Download latest version
kaggle_input_path = kagglehub.model_download(kaggle_model_path)
print("Path to model files:", kaggle_input_path)

# Creates new repo in HF
from huggingface_hub import create_repo

repo_id = f"{username}/{repo_name}"

# Create the HF repo
create_repo(
    repo_id=repo_id,
    repo_type="model",   # Specify "model" for a model repo (default)
    private=False,       # Set to True if you want a private repo
    exist_ok=True        # Set to True if you want to reuse an existing repo if it exists
)

print(f"Success! You can access the new HF repo here:  '{repo_id}'")

# Uploads model to HF

from huggingface_hub import upload_folder

# Upload the folder
upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path= kaggle_input_path
)
'''



# 1. Download finetuned model from kaggle site
import kagglehub
kaggle_input_path = kagglehub.model_download(kaggle_model_path)
print("Path to model files:", kaggle_input_path)


# 2. Load fined tined model from Kaggle.# the model needs to be dwnloaded first (above)
import keras_hub

# Try loading the model using from_preset
try:
    fine_tuned_model_loaded = keras_hub.models.GemmaCausalLM.from_preset(kaggle_input_path)
    fine_tuned_model_loaded.summary()  # Print the model summary
except ValueError as e:
    print(f"Model load error: {e}")


# 3. tokenizer for the fined tuned model
import keras_nlp
tokenizer_finetuned = keras_nlp.models.GemmaTokenizer.from_preset(kaggle_input_path)



!pip install -q evaluate


# Test data
import pandas as pd

df_infer = pd.read_excel(infer_path, sheet_name=sheet_name_infer) 
df_infer.head()


data_row = len(df_infer['original'])
print("Number of rows in evaluator file:", data_row)


# Bleu Extended Evaluator outside
# Test data for Bleu
import pandas as pd



import time
import pandas as pd
from evaluate import load

# Load the BLEU evaluation metric
bleu = load("bleu")

# Function to compute BLEU score for a single row
def compute_bleu(prediction, reference):
    """
    Compute the BLEU score for a single prediction and reference.
    BLEU expects a single prediction (output of model) and a list of reference (verified translation) strings.
    """
    if not isinstance(reference, list):
        reference = [reference]  # Wrap reference in a list if it's a single string
    result = bleu.compute(predictions=[prediction], references=[reference])
    return result['bleu']

# Start the main process timer
start_time = time.time()

# Ensure 'inferenced' column is of object type
df_infer['inferenced'] = df_infer['inferenced'].astype('object')

# Iterate over rows in df_ext_eval and generate the LLM response
for i, row in df_infer.iterrows():
    # Define the prompt for generating questions and answers
    #prompt = original=row['original']
    #prompt = {
    #'instruction': row['original'],
    #'response': ""
    #}
    prompt=row['original']
    # Generate the response using the fine-tuned model
    response = text_gen(fine_tuned_model_loaded,prompt, token_limit_infer) # uses the fine-tuned model in training but if run after training script, then the model needs to be downloaded and looded.

    
    # Save the cleaned response into the 'inferenced' column
    df_infer.at[i, 'inferenced'] = response

    # Compute BLEU score for this row and store it in 'bleu_score'
    prediction = response
    # Concatenate 'AnkiTranslation'
    reference = [row['AnkiTest']]

    bleu_score = compute_bleu(prediction, reference)
    df_infer.at[i, 'bleu_score'] = bleu_score  # Store the BLEU score in the DataFrame


    # Count tokens for the combined prompt + response
    length = len(tokenizer_finetuned(prompt+response))
    df_infer.at[i, 'nbr_tokens'] = length  # Store the length (nbr_tokens) in the DataFrame
    
    print(f"Processed row {i}:")
    print("\nThe nbr of tokens is:", length) # nbr_tokens
    print("\nThe original English is:", prompt) # model response
    print("\nThe translation is:", response) # model response (prediction)
    print("\nThe Anki  (reference) is:", reference) # model response
    print(f"BLEU Score: {bleu_score}")
    print("\n\n")
# End the main process timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")

# Calculate and print the average BLEU score
average_bleu = df_infer['bleu_score'].mean()
print(f"Average BLEU Score: {average_bleu:.4f}")

# Write results to an Excel file
df_infer.to_excel("/kaggle/working/Evaluation.xlsx", index=False)





# Bleu Extended Evaluator outside
# Test data for Bleu
import pandas as pd




import time
import pandas as pd
from evaluate import load

# Load the BLEU evaluation metric
bleu = load("bleu")

# Function to compute BLEU score for a single row
def compute_bleu(prediction, reference):
    """
    Compute the BLEU score for a single prediction and reference.
    BLEU expects a single prediction (output of model) and a list of reference (verified translation) strings.
    """
    if not isinstance(reference, list):
        reference = [reference]  # Wrap reference in a list if it's a single string
    result = bleu.compute(predictions=[prediction], references=[reference])
    return result['bleu']

# Start the main process timer
start_time = time.time()

# Ensure 'inferenced' column is of object type
df_infer['inferenced'] = df_infer['inferenced'].astype('object')

# Iterate over rows in df_ext_eval and generate the LLM response
for i, row in df_infer.iterrows():
    # Define the prompt for generating questions and answers
    #prompt = original=row['original']
    #prompt = {
    #'instruction': row['original'],
    #'response': ""
    #}
    prompt=row['original']
    # Generate the response using the fine-tuned model
    response = text_gen(fine_tuned_model_loaded,prompt, token_limit_infer) # uses the fine-tuned model in training but if run after training script, then the model needs to be downloaded and looded.

    
    # Save the cleaned response into the 'inferenced' column
    df_infer.at[i, 'inferenced'] = response

    # Compute BLEU score for this row and store it in 'bleu_score'
    prediction = response
    # Concatenate 'AnkiTranslation' and 'ChatGPTTranslation' into a reference list
    reference = [row['AnkiTranslation'], row['ChatGPTTranslation']]

    bleu_score = compute_bleu(prediction, reference)
    df_infer.at[i, 'bleu_score'] = bleu_score  # Store the BLEU score in the DataFrame


    # Count tokens for the combined prompt + response
    length = len(tokenizer_finetuned(prompt+response))
    df_infer.at[i, 'nbr_tokens'] = length  # Store the length (nbr_tokens) in the DataFrame
    
    print(f"Processed row {i}:")
    print("\nThe nbr of tokens is:", length) # nbr_tokens
    print("\nThe original English is:", prompt) # model response
    print("\nThe translation is:", response) # model response (prediction)
    print("\nThe Anki and ChatGPTtest (reference) is:", reference) # model response
    print(f"BLEU Score: {bleu_score}")
    print("\n\n")
# End the main process timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")

# Calculate and print the average BLEU score
average_bleu = df_infer['bleu_score'].mean()
print(f"Average BLEU Score: {average_bleu:.4f}")

# Write results to an Excel file
df_infer.to_excel("/kaggle/working/Evaluation_Ext.xlsx", index=False)




