import os
import keras
import keras_nlp
from datasets import load_dataset
from tqdm import tqdm

# Set environment variables
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


# Load Gemma 2B model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

# WARNING: the full dataset is 250GB (compressed) and over 1TB (uncompressed)
# Stream the dataset instead of downloading it
dataset = load_dataset("jonathanli/human-essays-reddit", streaming=True)

#Adding tokenizer because the Kaggle reject essay submission of token size > 199
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
# Model configuration
model_name = '/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_data = []
from tqdm import tqdm
# You can then iterate through the data as needed
for example in tqdm(dataset["train"].take(10000)):  # Get first 50000 examples
    curr_essay = f"### Instruction: Write an essay about {example['title']}\n\n### Response: {example['top_comment']}\n### End"
    # reject essay too long
    if len(tokenizer(curr_essay).input_ids) < 1000:
        
        # reject if score or top_comment_score too low
        if example['score'] >= 500 and example['top_comment_score'] >= 500:
            train_data.append(curr_essay)


print(f"{len(train_data)} essay loaded")


import tensorflow as tf
# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Adjust LoRA configuration
gemma_lm.backbone.enable_lora(
    rank=8,  # Lower rank to prevent overfitting[7]
    alpha=16,  # Set alpha to 2x the rank[7]
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],  # Target all linear layers[5][21]
)
gemma_lm.preprocessor.sequence_length = 256  

# Configure optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=2e-4,  # Higher learning rate for better convergence
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# Compile model
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    clipnorm=1.0,  # Add gradient clipping

)




# Train model with smaller batch size and gradient accumulation
batch_size = 8
accumulation_steps = 2

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = gemma_lm(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        loss = tf.reduce_mean(loss) / accumulation_steps
    gradients = tape.gradient(loss, gemma_lm.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gemma_lm.trainable_variables))
    return loss

for epoch in range(1):
    for i in range(0, len(train_data), batch_size * accumulation_steps):
        batch = train_data[i:i + batch_size * accumulation_steps]
        preprocessor_output = gemma_lm.preprocessor(batch)

        # Extracting components from the preprocessor output
        input_dict, labels, _ = preprocessor_output
        
        # Prepare input tensors
        x = {
            'token_ids': input_dict['token_ids'][:, :-1],  # Remove the last token for input
            'padding_mask': input_dict['padding_mask'][:, :-1]  # Adjust padding mask accordingly
        }
        
        # Prepare labels (y)
        y = labels[:, 1:]  # Remove the first token for labels
        
        for j in range(0, len(batch), batch_size):
            sub_x = {k: v[j:j+batch_size] for k, v in x.items()}
            sub_y = y[j:j+batch_size]
            loss = train_step(sub_x, sub_y)
        
        if i % (batch_size * accumulation_steps * 10) == 0:
            print(f"Step {i // (batch_size * accumulation_steps)}, Loss: {loss.numpy()}")



# Generate sample essay
prompt = "Write a 100-word essay on the importance of artificial intelligence."
generated_essay = gemma_lm.generate(prompt, max_length=256)
print(generated_essay)


import pandas as pd
import keras_nlp

# Load the test dataset
test = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')

# Prepare the prompt template
prompt_template = "Write a 100-word essay on the following topic: {topic}"


# Generate predictions
predictions = []

for i, row in test.iterrows():
    topic = row['topic']
    prompt = prompt_template.format(topic=topic)
    
    generated_essay = gemma_lm.generate(prompt, max_length=128)  # Adjust max_length as needed
    predictions.append(generated_essay)
    # for debug
    if i <= 2:
        print('Topic:', topic)
        print('Generated Essay:', generated_essay)
        print('\n***********************\n')

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'essay': predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

