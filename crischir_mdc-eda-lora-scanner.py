# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


max_tokens=1000
max_tokens_and_prompt=max_tokens+100


from datasets import load_dataset
import matplotlib.pyplot as plt


import tensorflow as tf


import fitz # PyMuPDF
import keras
import keras_hub


tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Check to confirm the policy is set
print("Global mixed precision policy:", tf.keras.mixed_precision.global_policy())


import time


import random


dataset = load_dataset("parquet", data_files={"/kaggle/input/make-data-count-data-preparation/train_dataset.parquet"})


train_dataset = dataset["train"]


df = train_dataset.to_pandas()


df





article_id_counts = df['article_id'].value_counts()
frequency_of_frequencies = article_id_counts.value_counts()
print(frequency_of_frequencies)


# Filter the DataFrame to include only rows where the 'extension' is 'xml'
df_xml = df[df['extension'] == '.xml']

# Calculate the count of each unique article_id in the filtered DataFrame
article_id_counts = df_xml['article_id'].value_counts()
frequency_of_frequencies = article_id_counts.value_counts()
print(frequency_of_frequencies)


# Filter the DataFrame to include only rows where the 'extension' is 'xml'
df_xml = df[df['extension'] == '.pdf']

# Calculate the count of each unique article_id in the filtered DataFrame
article_id_counts = df_xml['article_id'].value_counts()
frequency_of_frequencies = article_id_counts.value_counts()
print(frequency_of_frequencies)


# !pip install -U keras-nlp
# !pip install -U keras-hub
# !pip install -U keras


# !pip install -U tensorflow-text


import keras
import keras_hub


# warnings.filterwarnings("ignore")


os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["JAX_PLATFORMS"] = ""
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
# if run on GPU, prevent JAX to take up to 100% memory from GPU, set max at 90% 
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_270m")


# gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
gemma_lm.generate("Keras is a", max_length=30)

# Generate with batched prompts.
gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)


# from transformers import AutoTokenizer


# tokenizer = keras_hub.models.GemmaTokenizer.from_preset("gemma3_270m")
tokenizer = keras_hub.models.Gemma3Tokenizer.from_preset(
    "gemma3_270m"
)


tokenizer("The quick brown fox jumped.")


len(tokenizer("The quick brown fox jumped."))


# Function to get the token count for a given text
def get_gemma_token_count(text):
    if tokenizer:
        # The encode method converts text to token IDs. The length of the list is the count.
        return len(tokenizer(text))
    else:
        # Fallback to word count if the tokenizer is not available
        return len(text.split())


# Add a new column 'gemma_token_count' to the DataFrame
df['gemma_token_count'] = df['text'].apply(get_gemma_token_count)

# # Print the updated DataFrame to see the new column
# print("\nDataFrame with Gemma token counts:")
# print(df)


df.head()


# --- Plotting the distribution for extension=".pdf" ---

# Filter the DataFrame to include only rows where the 'extension' is 'pdf'
pdf_df = df[df['extension'] == '.pdf']

# Check if the filtered DataFrame is not empty
if not pdf_df.empty:
    plt.figure(figsize=(10, 6))
    # Plot the histogram for the filtered data
    pdf_df['source'].hist(bins=10, color='purple', edgecolor='black', alpha=0.7)

    plt.title('Distribution classification')
    plt.xlabel('classification in training')
    plt.ylabel('Frequency (Number of Articles)')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo articles with extension '.pdf' found in the DataFrame.")


df.columns



df.describe()


# Create a histogram to visualize the distribution of token counts
plt.figure(figsize=(10, 6))
# A histogram is a great way to show the frequency of token counts
# across different ranges (bins).
df['gemma_token_count'].hist(bins=10, color='teal', edgecolor='black', alpha=0.7)

# Add titles and labels for clarity
plt.title('Distribution of Gemma Token Counts')
plt.xlabel('Token Count')
plt.ylabel('Frequency (Number of Articles)')
plt.grid(axis='y', alpha=0.75)

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Display the plot
plt.show()


# --- Plotting the distribution for extension=".pdf" ---

# Filter the DataFrame to include only rows where the 'extension' is 'pdf'
pdf_df = df[df['extension'] == '.pdf']

# Check if the filtered DataFrame is not empty
if not pdf_df.empty:
    plt.figure(figsize=(10, 6))
    # Plot the histogram for the filtered data
    pdf_df['gemma_token_count'].hist(bins=10, color='purple', edgecolor='black', alpha=0.7)

    plt.title('Distribution of Gemma Token Counts for PDF Files')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency (Number of Articles)')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo articles with extension '.pdf' found in the DataFrame.")


# Set the maximum number of tokens
max_tokens = max_tokens

# Function to truncate text to a max token count, keeping the end of the text
def truncate_text_to_end(text):
    if tokenizer:
        # Encode the text to get token IDs
        tokens = tokenizer(text)
        # If the number of tokens is greater than max_tokens, truncate from the beginning
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[-max_tokens:]
            # Decode the tokens back into a string
            return tokenizer.detokenize(truncated_tokens)
        else:
            # If within the limit, return the original text
            return text
    else:
        # Fallback to character slicing if tokenizer is not available
        print("Warning: Tokenizer not loaded. Falling back to character-based slicing.")
        return text[-max_tokens:] if len(text) > max_tokens else text

# Create a new DataFrame with the truncated text
new_data = {
    'article_id': df['article_id'],
    'extension': df['extension'],
    'truncated_text': df['text'].apply(truncate_text_to_end),
     'dataset_id':df['dataset_id'],
    'type':df['type'],
    'source':df['source']
}
new_df = pd.DataFrame(new_data)

# Print the new DataFrame to confirm the changes
print(f"\nNew DataFrame with text truncated to {max_tokens} tokens:")
# print(new_df)


new_df


new_df.to_csv('truncated_text_train.csv', index=False)  


from typing import Callable, Any


import re


# The function to process the DataFrame as requested.
# The function to process the DataFrame as requested.
def split_and_check_citations(
    df: pd.DataFrame, 
    max_tokens: int, 
    tokenizer_obj: Any
) -> pd.DataFrame:
    """
    Splits text from a DataFrame into chunks of a maximum token length.
    
    A new row is created for each text chunk. A new 'is_citation' column
    is added, with a value of 'yes' if the chunk contains a value from the
    'dataset_id' column, and 'no' otherwise.
    
    Args:
        df (pd.DataFrame): The input DataFrame. It must contain 'text' and 'dataset_id' columns.
        max_tokens (int): The maximum number of tokens for each text chunk.
        tokenizer_obj (Any): An object with `tokenize` and `detokenize` methods.
        
    Returns:
        pd.DataFrame: A new DataFrame with text split into chunks and a new
                      'is_citation' column.
    """
    new_rows = []
    
    # Iterate through each row of the original DataFrame.
    for index, row in df.iterrows():
        # Get the text and dataset ID for the current row.
        full_text = str(row['text'])
        dataset_id = str(row['dataset_id']) if pd.notna(row['dataset_id']) else None
        
        # Tokenize the full text using the provided tokenizer object.
        tokens_list = tokenizer_obj.tokenize(full_text)
        
        # Calculate the number of chunks needed.
        num_chunks = (len(tokens_list) + max_tokens - 1) // max_tokens
        
        # Prepare the dataset ID for efficient checking.
        citation_to_find = None
        if dataset_id:
            citation_to_find = re.sub(r'https?://(?:dx\.)?doi\.org/|doi:', '', dataset_id, flags=re.IGNORECASE)
            
        # Loop to create chunks and new rows.
        for i in range(num_chunks):
            start_index = i * max_tokens
            end_index = start_index + max_tokens
            
            # Extract the chunk of tokens.
            chunk_tokens = tokens_list[start_index:end_index]
            
            # Detokenize the chunk to get the text chunk.
            chunk_text = tokenizer_obj.detokenize(chunk_tokens)
            
            # Check if the citation is present in the current chunk.
            is_citation = 'no'
            if citation_to_find and citation_to_find.lower() in chunk_text.lower():
                is_citation = 'yes'
            
            # Create a new row, copying the original values and adding the new ones.
            new_row = row.to_dict()
            new_row['text_chunk'] = chunk_text
            new_row['is_citation'] = is_citation
            
            # The original 'text' column is now irrelevant for these chunks.
            del new_row['text']
            
            new_rows.append(new_row)
            
    # Create the new DataFrame from the list of new rows.
    return pd.DataFrame(new_rows)

# --- Example Usage ---

# # Create a sample DataFrame to test the function.
# data = {
#     'article_id': ['10.25377/sussex.21184705', 'KU363060'],
#     'dataset_id': ['https://doi.org/10.25377/sussex.21184705', 'KU363060'],
#     'text': [
#         "This is a sample document for testing the chunking function. It's a bit longer to demonstrate the chunking process. Here is the key reference to the dataset: https://doi.org/10.25377/sussex.21184705. The text continues for a while to fill up the chunk. More text.",
#         "A different document with another citation. This one is a simple citation identifier: KU363060. No complex URLs this time, just a simple string for testing purposes."
#     ]
# }


type(df)


len(df)


# Set the maximum tokens per chunk.
MAX_TOKENS = max_tokens

# Call the function to split the text.
chunked_df = split_and_check_citations(df, MAX_TOKENS, tokenizer)

# Print the resulting DataFrame to see the output.
# print(chunked_df)


chunked_df.head()


# --- Plotting the distribution for extension=".pdf" ---

# Filter the DataFrame to include only rows where the 'extension' is 'pdf'
pdf_df = chunked_df[chunked_df['extension'] == '.pdf']

# Check if the filtered DataFrame is not empty
if not pdf_df.empty:
    plt.figure(figsize=(10, 6))
    # Plot the histogram for the filtered data
    pdf_df['is_citation'].hist(bins=10, color='purple', edgecolor='black', alpha=0.7)

    plt.title('Distribution of citation for PDF Files')
    plt.xlabel('citation')
    plt.ylabel('Frequency (Number of Articles)')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo articles with extension '.pdf' found in the DataFrame.")


citations_found =pdf_df.is_citation.value_counts()['yes']
print(citations_found)


# Separate the 'yes' and 'no' values.
yes_df = chunked_df[chunked_df['is_citation'] == 'yes']
no_df = chunked_df[chunked_df['is_citation'] == 'no']
# no_df['source']='Missing'
# no_df['dataset_id']='Missing'
no_df.loc[:, 'source'] = 'Missing'
no_df.loc[:, 'dataset_id'] = 'Missing'
# Determine the number of 'yes' and 'no' samples.
num_yes = len(yes_df)
# We want twice as many 'no' samples as 'yes' samples.
num_no_to_sample = num_yes * 2
num_no_to_sample=100
# Check if we have enough 'no' samples to take.
if num_no_to_sample > len(no_df):
    num_no_to_sample = len(no_df)
    print(f"\nWarning: Not enough 'no' samples to satisfy the 4:1 ratio. Using all {num_no_to_sample} available 'no' samples.")

# Randomly sample the desired number of 'no' values.
# The random_state ensures reproducibility.
sampled_no_df = no_df.sample(n=num_no_to_sample, random_state=42)

# Concatenate the 'yes' and sampled 'no' DataFrames to create the final, balanced dataset.
balanced_df = pd.concat([yes_df, sampled_no_df]).sample(frac=1).reset_index(drop=True)


len(yes_df)


len(no_df)


len(balanced_df)


balanced_df.head()


# new_df.truncated_text[120]


len(new_df.truncated_text[120])


balanced_df.columns


columns_to_keep = ['article_id', 'source', 'dataset_id', 'text_chunk']
consolidated_df = balanced_df[columns_to_keep]
consolidated_df.head()


consolidated_df.columns


add_on_df = pd.read_parquet('/kaggle/input/mda-concept-train-dataset2/labeled_citations.parquet')
print(df.head())


add_on_df.head()


add_on_df.columns


add_on_df =add_on_df.rename(columns={'type': 'source', 'window': 'text_chunk'})
reordered=['article_id', 'source', 'dataset_id', 'text_chunk']
add_on_df=add_on_df[reordered]





df_combined = pd.concat([consolidated_df, add_on_df]).reset_index(drop=True)
print("Combined DataFrame (before reindexing):\n")
print(df_combined)


len(df_combined)


add_on2=pd.read_csv("/kaggle/input/dummy-train/training.csv")


add_on2.columns


add_on2 = add_on2[['article_id', 'dataset_id', 'type', 'text_chunk']]
add_on2 = add_on2.rename(columns={'type': 'source'})
add_on2 = add_on2[['article_id', 'source', 'dataset_id', 'text_chunk']]


df_combined.columns


df_combined = pd.concat([df_combined, add_on2], ignore_index=True)
df_combined=df_combined .reset_index(drop=True)



df_combined = df_combined.replace('None', 'Missing')
df_combined = df_combined.fillna('Missing')


df_combined


template = """System: 

You are an expert at analyzing research data citations in academic papers. Identify and extract data citations and classify the data.


Classify the data as:
A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work  
C) Missing: if the DOI is in references, doesn't refer to research data, or is unrelated


text: 
{window}


Type: \n\n
"""





print("prompt tokens",len(tokenizer(template)),"characters",len(template))



def build_tf_dataset(dataset, batch_size=2):
    AUTO = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False
    
    # Define the template here, as requested.
    template = """System: 
You are an expert at analyzing research data citations in academic papers. Identify and extract data citations and classify the data.

A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work 
C) None: if the DOI is in references, doesn't refer to research data, or is unrelated

text: 
{window}

Type: \n\n

"""
    
    # We will use the full DataFrame since our sample is small
    dataset_dict_list = []

    # Adapting to the new DataFrame structure
    for i in range(len(dataset)):
        dataset_dict = dict()
        # 'text_chunk' is the window, so we'll use that as the prompt.
        dataset_dict["prompts"] = template.format(window=str(dataset.iloc[i]['text_chunk']))
        # The responses are a combination of 'source' and 'dataset_id'.
        dataset_dict["responses"] = f"source: {str(dataset.iloc[i]['source'])}, dataset_id: {str(dataset.iloc[i]['dataset_id'])}"
        dataset_dict_list.append(dataset_dict)

    dataset = tf.data.Dataset.from_generator(
        lambda: (item for item in dataset_dict_list),
        output_signature={
            "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
            "responses": tf.TensorSpec(shape=(), dtype=tf.string),
        }
    )

    # Add .repeat() before batching to ensure the dataset does not run out of data.
    dataset = dataset.cache().shuffle(2048, seed=42).repeat() 
    # Ensure a valid batch size
    batch_size = max(1, batch_size)
    dataset = dataset.with_options(options).batch(batch_size).prefetch(AUTO)

    return dataset


# def build_tf_dataset(dataset, batch_size=1):
#     AUTO = tf.data.AUTOTUNE
#     options = tf.data.Options()
#     options.experimental_deterministic = False
    
#     # We will use the full DataFrame since our sample is small
#     dataset = dataset
    
#     # Convert the dataframe into a dictionary with keys "prompts" and "responses"
#     dataset_dict_list = []
#     # Adapting to the new DataFrame structure
#     for i in range(len(dataset)):
#         dataset_dict = dict()
#         dataset_dict["prompts"] = template.format(window=dataset.iloc[i, 2])
#         dataset_dict["responses"] = f"dataset_id: {dataset.iloc[i, 3]}, source: {dataset.iloc[i, 5]}"
#         dataset_dict_list.append(dataset_dict)

#     dataset = tf.data.Dataset.from_generator(
#         lambda: (item for item in dataset_dict_list),
#         output_signature={
#             "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
#             "responses": tf.TensorSpec(shape=(), dtype=tf.string),
#         }
#     )

#     dataset = dataset.cache().shuffle(1024, seed=42)
#     # Ensure a valid batch size
#     batch_size = max(1, batch_size) 
#     dataset = dataset.with_options(options).batch(batch_size).prefetch(AUTO)

#     return dataset




# Calculate the total number of examples in the dataset
# dataset_size = len(df_combined[:1700])
dataset_size = len(consolidated_df)

# Calculate the number of steps per epoch based on the batch size
batch_size = 2 # or whatever your batch size is
steps_per_epoch = dataset_size // batch_size
if dataset_size % batch_size != 0:
    steps_per_epoch += 1 # Account for the last, smaller batch


# Call the function to build the dataset
print("\nBuilding TensorFlow dataset from the truncated data...")
# tf_dataset = build_tf_dataset(balanced_df)
# tf_dataset = build_tf_dataset(df_combined[:1700], batch_size=batch_size)
tf_dataset = build_tf_dataset(consolidated_df, batch_size=batch_size)
print("TensorFlow dataset built successfully.")


for element in tf_dataset.take(1):
    print(element)


#to be adapted
# unseen_data = new_df[900:]
# unseen_data = df_combined[1700:]

# unseen_data = consolidated_df[900:]
unseen_data = pd.read_csv("/kaggle/input/dummy-train/validation.csv")
unseen_data =unseen_data.rename(columns={'type': 'source'})
reordered=['article_id', 'source', 'dataset_id', 'text_chunk']
unseen_data=unseen_data[reordered]
unseen_data = unseen_data.reset_index(drop=True)


def generate_inference(dataset, example_num=None):
    """
    This function will generate the model inference and label the citation.
    """
    if example_num is None or example_num >= len(dataset):
        example_num = random.randint(0, len(dataset) - 1)

    row = dataset.iloc[example_num]
    article = str(row['text_chunk'])
    
    # Define the template for the prompt, same as in the build_tf_dataset function.
    template = """System: 
You are an expert at analyzing research data citations in academic papers. Identify data citations precisely and accurately and classify data.

Classify the data as:
A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work 
C) None: if the DOI is in references, doesn't refer to research data, or is unrelated


text: 
{window}


Type: \n\n
"""

    prompt = template.format(window=article)

    max_length = 550
    response = gemma_lm.generate(prompt, max_length=max_length)
    response = response.split("Type: \n\n")[-1].strip()
    
    return {
        "prompt": prompt,
        "inferred_response": response,
        "ground_truth": f"source: {str(row['source'])}, dataset_id: {str(row['dataset_id'])}"
    }

# Example usage of the generate_inference function with the dummy model.
inference_result = generate_inference(balanced_df, example_num=0)
print("\nGenerated Inference Example:")
print(inference_result)


# def generate_inference(example_num=None):
#     """
#     This function will generate the model inference and label the citation.
#     """

#     if example_num == None or example_num >= len(unseen_data):
#         example_num = random.randint(0, len(unseen_data))

#     row = unseen_data.loc[example_num]
#     article = row.truncated_text
#     summary = row.source
#     prompt = template.format(window = article)
    
#     # max_length = 2 * len(prompt.split()) # set the max output length to twice the length of the input prompt
#     max_length =300
#     response = gemma_lm.generate(prompt, max_length = max_length)
#     response = response.split("Type: \n\n")[-1].strip() # Extract only the summary text
    
#     return response


display(generate_inference(balanced_df))


gemma_lm.backbone.enable_lora(rank=8)
gemma_lm.summary()


gemma_lm.preprocessor.sequence_length = max_tokens_and_prompt

optimizer = keras.optimizers.AdamW(
    learning_rate=2e-5,
    weight_decay=0.001,
)

# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])


# Model Compilation
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


%time
# Model Training
# history = gemma_lm.fit(tf_dataset, epochs=12)
history = gemma_lm.fit(tf_dataset, epochs=8, steps_per_epoch=steps_per_epoch)


gemma_lm.backbone.save_lora_weights("gemma_finetune.lora.h5")

# gemma_lm.backbone.load_lora_weights("./gemma_finetune.lora.h5")
# gemma_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=3, temperature=0.7


def plot_model_metric(metric):
    plt.figure(dpi=120)
    plt.plot(history.history[metric], label=metric)
    # plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'{metric} over Epochs')
    plt.show();


plot_model_metric('sparse_categorical_accuracy')


display(generate_inference(balanced_df))


generate_inference(balanced_df)


def generate_simple_response(dataset, example_num=None):
    """
    This function will generate the model inference and label the citation.
    """
    if example_num is None or example_num >= len(dataset):
        example_num = random.randint(0, len(dataset) - 1)

    row = dataset.iloc[example_num]
    article = str(row['text_chunk'])
    
    # Define the template for the prompt, same as in the build_tf_dataset function.
    template = """System: 
You are an expert at analyzing research data citations in academic papers. Identify data citations and classify the data.

Classify the data as:
A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work 
C) None: if the DOI is in references, doesn't refer to research data, or is unrelated


text: 
{window}


Type: \n\n
"""

    prompt = template.format(window=article)

    max_length = 2500
    response = gemma_lm.generate(prompt, max_length=max_length)
    response = response.split("Type: \n\n")[-1].strip()
    
    return {

        "inferred_response": response,

    }

# Example usage of the generate_inference function with the dummy model.
inference_result =  generate_simple_response(balanced_df, example_num=0)
print("\nGenerated Inference Example:")
print(inference_result)


inference_result =  generate_simple_response(balanced_df, example_num=1)
print("\nGenerated Inference Example:")
print(inference_result)


inference_result =  generate_simple_response(balanced_df, example_num=3)
print("\nGenerated Inference Example:")
print(inference_result)


inference_result =  generate_simple_response(balanced_df, example_num=4)
print("\nGenerated Inference Example:")
print(inference_result)


inference_result =  generate_simple_response(balanced_df, example_num=5)
print("\nGenerated Inference Example:")
print(inference_result)


inference_result =  generate_simple_response(balanced_df, example_num=6)
print("\nGenerated Inference Example:")
print(inference_result)
print("ground truth 1",balanced_df.source[6],"2",balanced_df.dataset_id[6])


inference_result =  generate_simple_response(balanced_df, example_num=100)
print("\nGenerated Inference Example:")
print(inference_result)
print("ground truth 1",balanced_df.source[100],"2",balanced_df.dataset_id[100])


inference_result =  generate_simple_response(balanced_df, example_num=110)
print("\nGenerated Inference Example:")
print(inference_result)
print("ground truth 1",balanced_df.source[110],"2",balanced_df.dataset_id[110])


i=6
inference_result =  generate_simple_response(balanced_df, example_num=i)
print("\nGenerated Inference Example:")
print(inference_result)
print("ground truth 1",balanced_df.source[i],"2",balanced_df.dataset_id[i])


len(unseen_data)



import re
from sklearn.metrics import accuracy_score

# def evaluate_inference_metrics(inferred_data, ground_truth_data, example_range):
#     """
#     Compares inferred data from a model with ground truth data and calculates
#     accuracy metrics for 'source' and 'dataset_id'.

#     Args:
#         inferred_data (function): A function that generates an inference result,
#                                    e.g., generate_simple_response.
#         ground_truth_data (DataFrame): The DataFrame containing the ground truth values.
#         example_range (int): The number of examples to evaluate.
#     """
#     y_true_source = []
#     y_pred_source = []
#     y_true_dataset_id = []
#     y_pred_dataset_id = []

#     for i in range(example_range):
#         # 1. Get inferred data
#         inference_result = inferred_data(ground_truth_data, example_num=i)
        
#         # 2. Extract inferred source and dataset_id using regex
#         inferred_response_text = inference_result['inferred_response']
#         source_match = re.search(r"source: (\S+)", inferred_response_text)
#         dataset_id_match = re.search(r"dataset_id: (.+)", inferred_response_text)
        
#         # Replace None with a string placeholder
#         inferred_source = source_match.group(1) if source_match else 'None'
#         inferred_dataset_id = dataset_id_match.group(1).strip() if dataset_id_match else 'None'
        
#         # 3. Get ground truth data
#         gt_source = ground_truth_data.source[i]
#         gt_dataset_id = ground_truth_data.dataset_id[i]

#         # Append to lists for metric calculation
#         y_true_source.append(gt_source)
#         y_pred_source.append(inferred_source)
        
#         y_true_dataset_id.append(gt_dataset_id)
#         y_pred_dataset_id.append(inferred_dataset_id)
        
#         # Optional: Print results for manual inspection
#         print(f"Example {i+1}:")
#         print(f"Inferred Source: {inferred_source}, Inferred Dataset ID: {inferred_dataset_id}")
#         print(f"Ground Truth Source: {gt_source}, Ground Truth Dataset ID: {gt_dataset_id}")
#         print("-" * 50)
        
#     # 4. Calculate metrics
#     source_accuracy = accuracy_score(y_true_source, y_pred_source)
#     dataset_id_accuracy = accuracy_score(y_true_dataset_id, y_pred_dataset_id)
    
#     # 5. Print results
#     print("=" * 50)
#     print(f"Source Accuracy: {source_accuracy:.2f}")
#     print(f"Dataset ID Accuracy: {dataset_id_accuracy:.2f}")
#     print("=" * 50)

def evaluate_inference_metrics(inferred_data, ground_truth_data, example_range):
    """
    Compares inferred data from a model with ground truth data and calculates
    accuracy metrics for 'source' and 'dataset_id', ignoring punctuation and
    whitespace during comparison.

    Args:
        inferred_data (function): A function that generates an inference result,
                                   e.g., generate_simple_response.
        ground_truth_data (DataFrame): The DataFrame containing the ground truth values.
        example_range (int): The number of examples to evaluate.
    """
    y_true_source = []
    y_pred_source = []
    y_true_dataset_id = []
    y_pred_dataset_id = []

    # Helper function to normalize strings
    def normalize_string(s):
        """Removes all non-alphanumeric characters and converts to lowercase."""
        if not isinstance(s, str):
            s = str(s)
        return re.sub(r'[\W_]+', '', s).lower()

    for i in range(example_range):
        # 1. Get inferred data
        inference_result = inferred_data(ground_truth_data, example_num=i)
        
        # 2. Extract inferred source and dataset_id using regex
        inferred_response_text = inference_result['inferred_response']
        source_match = re.search(r"source: (\S+)", inferred_response_text)
        dataset_id_match = re.search(r"dataset_id: (.+)", inferred_response_text)
        
        # Replace None with a string placeholder
        inferred_source = source_match.group(1) if source_match else 'None'
        inferred_dataset_id = dataset_id_match.group(1).strip() if dataset_id_match else 'None'
        
        # 3. Get ground truth data
        gt_source = ground_truth_data.source[i]
        gt_dataset_id = ground_truth_data.dataset_id[i]

        # 4. Normalize and append to lists for metric calculation
        y_true_source.append(normalize_string(gt_source))
        y_pred_source.append(normalize_string(inferred_source))
        
        y_true_dataset_id.append(normalize_string(gt_dataset_id))
        y_pred_dataset_id.append(normalize_string(inferred_dataset_id))
        
        # Optional: Print results for manual inspection
        print(f"Example {i+1}:")
        print(f"Inferred Source: {inferred_source}, Inferred Dataset ID: {inferred_dataset_id}")
        print(f"Ground Truth Source: {gt_source}, Ground Truth Dataset ID: {gt_dataset_id}")
        print("-" * 50)
        
    # 5. Calculate metrics
    source_accuracy = accuracy_score(y_true_source, y_pred_source)
    dataset_id_accuracy = accuracy_score(y_true_dataset_id, y_pred_dataset_id)
    
    # 6. Print results
    print("=" * 50)
    print(f"Source Accuracy: {source_accuracy:.2f}")
    print(f"Dataset ID Accuracy: {dataset_id_accuracy:.2f}")
    print("=" * 50)


 evaluate_inference_metrics(generate_simple_response, unseen_data, example_range=50)




