# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
!pip install -q -U kagglehub --upgrade


import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00" # avoid memory fragmentation on JAX backend.
os.environ["JAX_PLATFORMS"] = ""
import keras
import keras_nlp
import kagglehub


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
# os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("kaggle_username")
os.environ["KAGGLE_KEY"] = user_secrets.get_secret("GEMINI_API_KEY")

from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas() # progress bar for pandas

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, HTML

import matplotlib.pyplot as plt

from itertools import permutations


class Config:
    seed = 42
    dataset_path = "/kaggle/input/catalan-proverbs-explanation-and-context"
    preset = "gemma2_instruct_2b_en" # name of pretrained Gemma 2
    sequence_length = 60 # max size of input sequence for training
    batch_size = 1 # size of the input batch in training
    lora_rank = 5 # rank for LoRA, higher means more trainable parameters
    learning_rate=8.8e-5 # learning rate used in train
    epochs = 20 # number of epochs to train


keras.utils.set_random_seed(Config.seed)


df = pd.read_csv(f"/kaggle/input/catalan-proverbs-explanation-and-context/refranys_catalans_explicaci_contexte.csv")
df.head()


df.shape[0]


# Permutation-Based Context Data Augmentation PCDA

# Define the context columns
columns = ["Context_1", "Context_2", "Context_3"]

# Generate all possible permutations
permutations_list = list(permutations(columns))

# Dynamically create templates with all combinations
templates = []
for perm in permutations_list:
    template = f"\n\nContext_1:\n{{{perm[0]}}}\n\nContext_2:\n{{{perm[1]}}}\n\nContext_3:\n{{{perm[2]}}}\n\nRefrany:\n{{Refrany}}"
    templates.append(template)

# Generate all templates for each row of the DataFrame
data = []
for template in templates:
    data.extend(df.apply(lambda row: template.format(
        Context_1=row.Context_1.strip() if isinstance(row.Context_1, str) else row.Context_1,
        Context_2=row.Context_2.strip() if isinstance(row.Context_2, str) else row.Context_2,
        Context_3=row.Context_3.strip() if isinstance(row.Context_3, str) else row.Context_3,
        Refrany=row.refrany.strip() if isinstance(row.refrany, str) else row.refrany
    ), axis=1))

# `data` now contains all the generated combinations


# DataFrame size
df_size = df.shape[0]  # Number of rows in df

# `data` list size
data_size = len(data)  # Number of elements in the data list

# Print comparison
print(f"DataFrame size (number of rows): {df_size}")
print(f"`data` list size (number of elements): {data_size}")

ratio = data_size / df_size
print(f"Each row in `df` generates {ratio:.2f} elements in `data`.")


print(data[122])


print(data[122+158])


print(data[122+2*158])


gemma_causal_lm = keras_nlp.models.GemmaCausalLM.from_preset(Config.preset)
gemma_causal_lm.summary()


x, y, sample_weight = gemma_causal_lm.preprocessor(data[0:2])


print(x, y)


# Enable LoRA for the model and set the LoRA rank to the lora_rank as set in Config (5).
gemma_causal_lm.backbone.enable_lora(rank=Config.lora_rank)
gemma_causal_lm.summary()


# Inspect the preprocessor output to confirm its structure
preprocessed_output = gemma_causal_lm.preprocessor([data[0]])
print(type(preprocessed_output))  # Check the type
print(preprocessed_output)        # Display the output structure


# Calculate the token length for each entry
token_lengths = [
    len(gemma_causal_lm.preprocessor([text])[0]["token_ids"][0])
    for text in data
]

# Print basic results
print(f"Maximum token length: {max(token_lengths)}")
print(f"Number of texts exceeding 256 tokens: {sum([l > 256 for l in token_lengths])}")

# (Optional) Show descriptive statistics
import numpy as np
print(f"Average token length: {np.mean(token_lengths)}")
print(f"Standard deviation: {np.std(token_lengths)}")



# Calculate the actual token length for each entry (without truncation or padding)
token_lengths = [
    len(gemma_causal_lm.preprocessor.tokenizer.tokenize(text)) for text in data
]

# Find the maximum number of tokens in `data`
max_length = max(token_lengths)

# Print the result
print(f"Maximum token length (actual, without padding): {max_length}")



# Set the maximum sequence length in the preprocessor
gemma_causal_lm.preprocessor.sequence_length = 60

# Recalculate the token length for each entry
token_lengths = [
    len(gemma_causal_lm.preprocessor([text])[0]["token_ids"][0])
    for text in data
]

# Print basic results
print(f"Maximum token length after adjustment: {max(token_lengths)}")
print(f"Number of texts exceeding 60 tokens after adjustment: {sum([l > 60 for l in token_lengths])}")

# (Optional) Display descriptive statistics
import numpy as np
print(f"Average token length after adjustment: {np.mean(token_lengths)}")
print(f"Standard deviation after adjustment: {np.std(token_lengths)}")


example_tokens = gemma_causal_lm.preprocessor([data[122]])[0]["token_ids"][0]

print("First 20 tokens:", example_tokens[:20])
print("Last 30 tokens:", example_tokens[-30:])


#set sequence length cf. config (60)
gemma_causal_lm.preprocessor.sequence_length = Config.sequence_length 

# Compile the model with loss, optimizer, and metric
gemma_causal_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=Config.learning_rate),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
# Limit the data to the length of the DataFrame
data_simp = data[:len(df)]

# Train the model and store the training history
history_simp = gemma_causal_lm.fit(data_simp, epochs=Config.epochs, batch_size=Config.batch_size)


# Access the loss values 
losses_simp = history_simp.history["loss"]

# Access the accuracy values 
accuracies_simp = history_simp.history["sparse_categorical_accuracy"]



# Plot the losses
plt.plot(losses_simp, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Loss evolution with the first context combination")
plt.legend()
plt.show()

# Plot the accuracy (if available)
plt.plot(accuracies_simp, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy evolution with the first context combination")
plt.legend()
plt.show()



def colorize_response(response):
    # Set the color for the "Refrany" tag
    refrany_color = "green"
    
    # Split the response into lines
    lines = response.split("\n\n")
    
    # Separate the "Refrany" line from the others
    refrany_line = ""
    other_lines = []
    
    for line in lines:
        if line.startswith("Refrany:"):
            # Apply green color formatting to "Refrany"
            refrany_line = line.replace(
                "Refrany:",
                f"<b><font color='{refrany_color}'>Refrany:</font></b>"
            )
        else:
            other_lines.append(line)
    
    # Reconstruct the response with "Refrany" at the beginning
    response_colored = refrany_line + "<br><br>" + "<br><br>".join(other_lines)
    
    return response_colored


class GemmaQA:
    def __init__(self, gemma_model, max_length=60):
        self.max_length = int(max_length)  # Ensure that max_length is an integer
        self.prompt = template
        self.gemma_causal_lm = gemma_model  # Ensure the model is passed correctly
        self.gemma_causal_lm.preprocessor.sequence_length = self.max_length  # Configure the preprocessor

        
    def query(self, Context_1, Context_2, Context_3):
     
        # Generate the text
        prompt = self.prompt.format(
            Context_1=Context_1,
            Context_2=Context_2,
            Context_3=Context_3,
            Refrany=""
        )
        response = self.gemma_causal_lm.generate(prompt, max_length=self.max_length)
        
        # Apply colors to the generated response
        response_colored = colorize_response(response)
    
        # Display the colored response
        display(HTML(response_colored))
        
        
    def backgroundquery(self, Context_1, Context_2, Context_3):
     
        # Generate the text
        prompt = self.prompt.format(
            Context_1=Context_1,
            Context_2=Context_2,
            Context_3=Context_3,
            Refrany=""
        )
        # print(f"Generated prompt: {prompt}")
        response = self.gemma_causal_lm.generate(prompt, max_length=self.max_length)
        
        return response



gemma_qa = GemmaQA(gemma_causal_lm, max_length=60)


def evaluate_model_with_combinations(model, df, combinations, output_file=None):
    results = []

    for _, row in df.iterrows():
        real_refrany = row["refrany"]

        for combination in combinations:
            # Get the contexts, using "" if the context is marked as None
            contexts = [row[context] if context else "" for context in combination]

            # Get the predicted refrany using the model
            response = model.backgroundquery(*contexts)

            # Handle cases where response is None
            if response is None:
                predicted_refrany = ""
            else:
                # Extract the text after "Refrany:"
                if "Refrany:" in response:
                    predicted_refrany = response.split("Refrany:")[-1].strip()
                else:
                    predicted_refrany = ""

            # Evaluate if the predicted refrany matches the real one
            is_correct = predicted_refrany.strip().lower() == real_refrany.strip().lower()

            # Save the results
            results.append({
                "combination": " + ".join([c if c else "empty" for c in combination]),
                "contexts_used": " + ".join(contexts),
                "real_refrany": real_refrany,
                "predicted_refrany": predicted_refrany,
                "correct": is_correct
            })

    results_df = pd.DataFrame(results)

    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return results_df



# Process combinations 
combinations = [
    ["Context_1", "Context_2", "Context_3"], 
    ["Context_1", "Context_2", None],  # Empty Context 3
    ["Context_2", "Context_3", None],  # Empty Context 1
    ["Context_1", "Context_3", None]   # Empty Context 2
]

results = evaluate_model_with_combinations(
    model=gemma_qa, 
    df=df, 
    combinations=combinations, 
    output_file="/kaggle/working/simple_model_results.csv"
)



accuracy = results["correct"].mean() * 100
print(f"Simple model accuracy: {accuracy:.2f}%")


# Limit data to DataFrame length
data_ext = data[len(df):]

# Train and store history with extended data
history_ext = gemma_causal_lm.fit(data_ext, epochs=Config.epochs, batch_size=Config.batch_size)


# Access the losses
losses_ext = history_ext.history['loss']

# Access the accuracy
accuracies_ext = history_ext.history['sparse_categorical_accuracy']


# Plot the losses
plt.plot(losses_ext, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Evolution with All Context Combinations")
plt.legend()
plt.show()

# Plot the accuracy (if available)
plt.plot(accuracies_ext, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Evolution with All Context Combinations")
plt.legend()
plt.show()


results_ext = evaluate_model_with_combinations(
    model=gemma_qa, 
    df=df, 
    combinations=combinations, 
    output_file="/kaggle/working/model_extended_results.csv"
)



# Calculate the precision
precision_ext = results_ext["correct"].mean() * 100

# Print the extended model precision
print(f"Extended model precision: {precision_ext:.2f}%")


# Manually generated contexts (you can replace them with dynamically generated ones)
Context_1, Context_2, Context_3 =  "aparença","realitat ","engany"

# Query the model
gemma_qa.query( Context_1, Context_2, Context_3)


# Manually generated contexts (you can replace them with dynamically generated ones)
Context_1, Context_2, Context_3 =  "aparença","realitat",""

# Query the model
gemma_qa.query( Context_1, Context_2, Context_3)


# Manually generated contexts (you can replace them with dynamically generated ones)
Context_1, Context_2, Context_3 =  "engany","",""

# Query the model
gemma_qa.query( Context_1, Context_2, Context_3)


gemma_causal_lm.save_weights('/kaggle/working/model.weights.h5')


