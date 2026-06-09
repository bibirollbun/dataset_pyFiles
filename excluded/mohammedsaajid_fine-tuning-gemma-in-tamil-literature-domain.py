# Install Dependencies

! pip install -q -U keras keras-nlp 


import numpy as np 
import pandas as pd 
import os
import keras
import keras_nlp
import re
import matplotlib.pyplot as plt
import time
import random


# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"



num_data_limit = 100
token_limit = 256
batch_size= 1
lora_rank = 4
learning_rate = 1e-4
epochs= 10
start_time = 0
accumulation_steps = 4
sot_token = "<start_of_turn>"
eot_token = "<end_of_turn>"  
model_path = "/kaggle/input/gemma-2/transformers/gemma-2-2b/2"


# Loading the dataset.
# Note that the seperator is a tab '\t' and not comma ','.
dataset = pd.read_csv("/kaggle/input/fine-tuning-dataset-in-tamil-literature-domain/dataset.csv",sep='\t')
# Renaming the columns to input and output.
dataset.rename(columns={'Question': 'input', 'Answer': 'output'}, inplace=True)



# Viewing the first 5 rows
dataset.head()



def convert_message_to_prompt(message: str, model_prefix: str = "") -> str:

    # Format the user part
    user_part = f"<bos> {sot_token} user\n{message} {eot_token} <eos>"
    
    # Format the model part based on whether a prefix is provided
    if model_prefix:
        model_part = f"<bos> {sot_token} model\n{model_prefix} {eot_token} <eos>"
    else:
        model_part = f"<bos> {sot_token} model\n"
    
    return f"{user_part}\n{model_part}"


def remove_tokens(response: str) -> str:

    # Remove <start_of_turn> , <end_of_turn> , user , model , <eos>, <bos> from the response
    cleaned_response = re.sub(r"<start_of_turn>|<end_of_turn>|user|model|<eos>|<bos>", "", response)
    
    # Strip leading/trailing whitespace and newlines
    return cleaned_response.strip()


def create_training_dataset(dataframe):
    # Create an empty list to store the formatted prompts
    dataset = []

    # Iterate over the rows of the dataframe
    for _, row in dataframe.iterrows():
        # Extract the questions and answers
        questions = row['input']
        answer = row['output']

        # Format using the convert_message_to_prompt function
        prompt = convert_message_to_prompt(questions, answer)
        # Append the formatted prompt to the dataset list
        dataset.append(prompt)
    
    # Return the dataset list
    return dataset

def start():
    # Defines the global variable start_time
    global start_time
    # Starts the timer
    start_time = time.time()

def stop():
    # Prints the total time elapsed by subtracting the stop time by start time
    print(f"TOTAL TIME ELAPSED: {time.time() - start_time:.2f}s")


def answer_my_question(message):
    # Starts the timer
    start()
    # Format the message to prompt
    prompt = convert_message_to_prompt(message)
    # Generate the model response for the prompt
    output = gemma_lm.generate(prompt, max_length=512) # Maximum length = 512
    # Prints the output
    print("\nGemma output:")
    # Remove the tokens from output. 
    output = remove_tokens(output)
    # Stop the timer
    stop()
    return output




# Split dataset into train, validation, and test sets
train_df = dataset.iloc[:80, :].reset_index(drop=True)
val_df = dataset.iloc[80:90, :].reset_index(drop=True)
test_df = dataset.iloc[90:100, :].reset_index(drop=True)


# Display the sizes of the sets
print(f"Training set: {len(train_df)}")
print(f"Validation set: {len(val_df)}")
print(f"Test set: {len(test_df)}")

# Create training, validation, and test datasets
train = create_training_dataset(train_df)
val = [(item['input'], item['output']) for _, item in val_df.iterrows()]
test = [(item['input'], item['output']) for _, item in test_df.iterrows()]

# Display the number of examples in each set
print(f"Number of training examples: {len(train)}")
print(f"Number of validation examples: {len(val)}")
print(f"Number of test examples: {len(test)}")



# Load the pretrained Gemma 2 model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_path)

# View the model summary
gemma_lm.summary()


# Generate a random number between 1 and 80
random_index = random.randint(1, 80)
# Access the random index in train_df.input
random_input = train_df.input[random_index]
# Displays the random input
random_input


# Performing Inference
answer_my_question(random_input)


# Extract the input and output sequences from the validation set
x_test, y_test = zip(*test)

# Convert to NumPy arrays
x_test = np.array(x_test)
y_test = np.array(y_test)

# Evaluate the model
results = gemma_lm.evaluate(
    x=x_test,
    y=y_test,
    batch_size = 1
)
test_loss = results[0]
test_accuracy = results[1]
print("Test Loss:", results[0])
print("Test Sparse Categorical Accuracy:", results[1])  # If accuracy is defined as a metric



# Enable LoRA for the model and set the LoRA rank to 4
gemma_lm.backbone.enable_lora(rank=lora_rank)

# Summary of the model with LoRA enabled
gemma_lm.summary()


# Importing Builtin Callbacks
from keras.callbacks import ReduceLROnPlateau
# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# We Use AdamW optimizer with weight decay.
optimizer = keras.optimizers.AdamW(
    learning_rate,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
# Add regularization (Dropout)
dropout_rate = 0.5
# Append Batch Normalization and Dropout Layers
gemma_lm.layers.append(keras.layers.BatchNormalization())
gemma_lm.layers.append(keras.layers.Dropout(dropout_rate))
# Compile the model
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()], # We are using Sparse Categorical Accuracy
)


# Defining Custom Callback by inheritance
class CustomCallback(keras.callbacks.Callback):
    # Define the constructors to recieve arguments
    def __init__(self, lora_name, lora_rank, token_limit, generate_function, prompt_example):
        super(CustomCallback, self).__init__()
        self.lora_name = lora_name
        self.lora_rank = lora_rank
        self.token_limit = token_limit
        self.generate_function = generate_function
        self.prompt_example = prompt_example
        
    # Define the function to be executed after end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        # Defining model version name
        model_name = f"fine-tuned-tamil-gemma_epoch{epoch+1}.lora.h5"
        # Save the lora weights for each epoch 
        gemma_lm.backbone.save_lora_weights(model_name)
        print(f"Saved LoRA weights to {model_name}\n")
        
        # Perform inference with a sample prompt
        generated_answer = self.generate_function(self.prompt_example)
        print(f"\nInference after Epoch {epoch+1}:\n{self.prompt_example}\n Answer: {generated_answer}\n")
        

# Example prompt for inference during training
prompt_example = train_df.input[random_index]


# Initialize the custom callback
lora_name = "tamil_gemma"
custom_callback = CustomCallback(
    lora_name=lora_name,
    lora_rank=lora_rank,
    token_limit=token_limit,
    generate_function=answer_my_question,
    prompt_example=prompt_example
)

# Reduce The learning Rate on a plateue
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # Metric to monitor
    factor=0.5,          # Factor to reduce learning rate
    patience=3,          # Number of epochs with no improvement
    min_lr=1e-5          # Defining the minimum value of learning rate
)

# Define the callbacks as a list
callbacks = [custom_callback, reduce_lr]



# Prepare validation data for fine-tuning
import numpy as np

# Extract the input and output sequences from the validation set
x_val, y_val = zip(*val)

# Convert to NumPy arrays
x_val = np.array(x_val)
y_val = np.array(y_val)



# Train the model
history = gemma_lm.fit(
    train, # Training Dataset
    epochs=epochs, # Number of Epochs
    batch_size=batch_size, # Batch Size
    validation_data=(x_val, y_val), # Validation Data for calculating Val loss
    callbacks=callbacks # Callbacks
)


questions= test_df.input[0:10]

for i, question in enumerate(questions, 1):
    answer = answer_my_question(question)
    print(f"Question: {question}\n\nAnswer: {answer}")


# Plot loss functions
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Time')
plt.legend()
plt.show()


# Plot Sparse Categorical Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['sparse_categorical_accuracy'], label='Training SCA')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation SCA')
plt.xlabel('Epochs')
plt.ylabel('Sparse Categorical Accuracy')
plt.title('Training & Validation SCA Over Time')
plt.legend()
plt.show()


# Extract the input and output sequences from the validation set
x_test, y_test = zip(*test)

# Convert to NumPy arrays
x_test = np.array(x_test)
y_test = np.array(y_test)

# Evaluate the model
results = gemma_lm.evaluate(
    x=x_test,
    y=y_test,
    batch_size = 1
)
test_loss = results[0]
test_accuracy = results[1]
print("Test Loss:", results[0])
print("Test Sparse Categorical Accuracy:", results[1])  # If accuracy is defined as a metric



# We Merge the LoRA weights into the base model and save it

# Define the directory to save the fine-tuned model
current_dir_path = "./fine_tuned_gemma_tamil_QA"

# Save the fine-tuned model.
gemma_lm.save_to_preset(current_dir_path)

print(f"Fine-tuned Gemma model saved to {current_dir_path}")


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

kaggle_username = user_secrets.get_secret("USERNAME")


MODEL_SLUG = "fine_tuned_gemma_tamil_literature"
VARIATION_SLUG = "gemma2_instruct_2b_tamil"

kaggle_uri = f"kaggle://{kaggle_username}/{MODEL_SLUG}/keras/{VARIATION_SLUG}"
keras_nlp.upload_preset(kaggle_uri, current_dir_path)


%reset -f


! pip install -q -U keras keras-nlp


import keras
import keras_nlp


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("/kaggle/input/fine_tuned_gemma_tamil_literature/keras/gemma2_instruct_2b_tamil/1")
# Enter your own question here
sample_question = "à®¤à®®à®¿à®´à®¿à®²à¯� 'à®ªà®•à¯�à®¤à®¿' à®•à®µà®¿à®¤à¯ˆà®¯à®¿à®©à¯� à®®à¯�à®•à¯�à®•à®¿à®¯ à®•à¯‚à®±à¯�à®•à®³à¯� à®�à®©à¯�à®©?"
# Running Inference
gemma_lm.generate(sample_question)
# Uncomment this line and run to get the formatted output
# answer_my_question(sample_question)

