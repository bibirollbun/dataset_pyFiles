# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import polars as pl



train_set_df=pl.read_parquet("/kaggle/input/mda-concept-train-dataset/labeled_citations.parquet")


train_set_df


temp=train_set_df.to_pandas()


df = temp[['window', 'type']].copy()


# replacing na values in college with No college
# df["type"].fillna("None", inplace = True)
df.loc[df["type"].isnull(), "type"] = "None"


df


 import seaborn as sns


sns.countplot(x='type',data=df)


template = """System: 

You are an expert at analyzing research data citations in academic papers.

Classify the data as:
A) Primary: if the data was generated specifically for this study
B) Secondary: if the data was reused or derived from prior work  
C) None: if the DOI is in references, doesn't refer to research data, or is unrelated


text: 
{window}


Type: \n\n
"""


# !pip install -q -U keras-nlp
# !pip install -q -U keras>=3


!pip install -U keras-nlp
!pip install -U keras-hub
!pip install -U keras


import os
import keras
import random
import warnings
import keras_nlp
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
# from IPython.display import display, # google totorial use colors for output 


warnings.filterwarnings("ignore")


os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
os.environ["JAX_PLATFORMS"] = ""





def build_tf_dataset(dataset):
    AUTO = tf.data.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False
    
    # Select first 500 records for faster training
    dataset = df[:1000]

    # Convert the dataframe into a dictionary with keys "prompts" and "responses"
    dataset_dict_list = []
    for i in range(len(dataset)):
        dataset_dict = dict()
        dataset_dict["prompts"] = template.format(window=dataset.iloc[i, 0])
        dataset_dict["responses"] = dataset.iloc[i, 1]
        dataset_dict_list.append(dataset_dict)

    dataset = tf.data.Dataset.from_generator(
        lambda: (item for item in dataset_dict_list),
        output_signature={
            "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
            "responses": tf.TensorSpec(shape=(), dtype=tf.string),
        }
    )

    dataset = dataset.cache().shuffle(1024, seed=42)
    dataset = dataset.with_options(options).batch(1).prefetch(AUTO)

    return dataset


unseen_data = df[1000:]
unseen_data = unseen_data.reset_index(drop=True)
dataset = build_tf_dataset(df)


import keras
import keras_hub


gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_270m")
gemma_lm.generate("Keras is a", max_length=30)

# Generate with batched prompts.
gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)



gemma_lm.summary()


def generate_inference(example_num=None):
    """
    This function will generate the model inference and label the citation.
    """

    if example_num == None or example_num >= len(unseen_data):
        example_num = random.randint(0, len(unseen_data))

    row = unseen_data.loc[example_num]
    article = row.window
    summary = row.type
    prompt = template.format(window = article)
    
    # max_length = 2 * len(prompt.split()) # set the max output length to twice the length of the input prompt
    max_length =300
    response = gemma_lm.generate(prompt, max_length = max_length)
    response = response.split("Type: \n\n")[-1].strip() # Extract only the summary text
    
    return response


display(generate_inference())


gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()


gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=1e-5,
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


# Model Training
history = gemma_lm.fit(dataset, epochs=30)


gemma_lm.backbone.save_lora_weights("gemma_finetune.lora.h5")

# gemma_lm.backbone.load_lora_weights("./gemma_finetune.lora.h5")
# gemma_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=3, temperature=0.7))


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


display(generate_inference())


generate_inference()


# gemma_lm.preprocessor.sequence_length = 512

# optimizer = keras.optimizers.AdamW(
#     learning_rate=5e-5,
#     weight_decay=0.01,
# )

# # Exclude layernorm and bias terms from decay.
# optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])


# gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.backbone.load_lora_weights("gemma_finetune.lora.h5")
# gemma_lm.compile(sampler=keras_nlp.samplers.TopKSampler(k=3, temperature=0.7))

# Model Compilation
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


generate_inference()


def generate_inference(model, df):
    """
    Generates model inference for each 'window' in the DataFrame.

    Args:
        model: The trained model to use for inference (e.g., gemma_lm).
        df: The DataFrame containing the 'window' data for inference.

    Returns:
        A list of generated responses, one for each 'window' in the DataFrame.
    """
    
    responses = []
    # Loop through each row of the DataFrame
    for _, row in df.iterrows():
        # Get the 'window' text from the current row
        window_text = row['window']
        
        # Create the prompt using the provided template
        # The 'template' object must be defined outside this function
        prompt = template.format(window=window_text)
        
        # Define the maximum length for the generated response
        max_length = 300
        
        # Generate the response using the model
        response = model.generate(prompt, max_length=max_length)
        
        # Extract and clean the generated response
        # This assumes the model's output format is consistent
        try:
            cleaned_response = response.split("Type: \n\n")[-1].strip()
        except IndexError:
            cleaned_response = response.strip() # Fallback if split fails
        
        # Append the cleaned response to the list
        responses.append(cleaned_response)
        
    return responses


