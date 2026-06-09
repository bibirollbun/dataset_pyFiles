%pip install jax flax optax dm-haiku cairosvg


import pandas as pd

try:
    df_train = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')
    df_test = pd.read_csv('/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv')
    display(df_train.head())
    display(df_test.head())
except FileNotFoundError:
    print("Error: train.csv or test.csv not found. Please make sure the files are in the current directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Data Overview
print("df_train shape:", df_train.shape)
print("df_test shape:", df_test.shape)
print("\ndf_train info:")
print(df_train.info())
print("\ndf_test info:")
print(df_test.info())
print("\ndf_train descriptive statistics:")
display(df_train.describe(include='all'))
print("\ndf_test descriptive statistics:")
display(df_test.describe(include='all'))


# Missing Values
print("\ndf_train missing values:")
print(df_train.isnull().sum())
print("\ndf_test missing values:")
print(df_test.isnull().sum())

# Data Distribution (Prompt Lengths)
df_train['prompt_length'] = df_train['description'].str.len()
print("\ndf_train prompt length statistics:")
print(df_train['prompt_length'].describe())

# SVG Data Inspection (Limited to description column in this case)
print("\nSample descriptions from df_train:")
print(df_train['description'].sample(5))


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.hist(df_train['prompt_length'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Prompt Lengths in Training Data')
plt.xlabel('Prompt Length')
plt.ylabel('Frequency')
plt.show()


import numpy as np
from sklearn.model_selection import train_test_split

# Simulate tokenization (replace with actual tokenizer)
def tokenize(text):
    return np.array([ord(c) for c in text])  # Example: Convert characters to ASCII values

# Tokenize descriptions
df_train['tokenized_description'] = df_train['description'].apply(tokenize)


# Data Splitting
train_data, val_data = train_test_split(df_train, test_size=0.2, random_state=42)

# Simulate data loaders
class DataLoader:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for _, row in self.data.iterrows():
          yield row['tokenized_description'], np.array([]) # Placeholder for SVG tokens

    def __len__(self):
        return len(self.data)

train_loader = DataLoader(train_data)
val_loader = DataLoader(val_data)

# Example usage of data loaders
for prompt_tokens, svg_tokens in train_loader:
    print("Example prompt tokens:", prompt_tokens[:5]) # print first five tokens
    print("Example svg tokens:", svg_tokens)
    break


import numpy as np

# 1. Text Prompt Encoding
# Using a simple average of token embeddings as a placeholder
def encode_text_prompt(tokens):
    return np.mean(tokens, axis=0) if len(tokens) > 0 else np.zeros(1) # handle empty tokens

train_data['prompt_embedding'] = train_data['tokenized_description'].apply(encode_text_prompt)
val_data['prompt_embedding'] = val_data['tokenized_description'].apply(encode_text_prompt)


# 2. SVG Token Encoding (Placeholder)
# Placeholder function for SVG token encoding
def encode_svg_placeholder(length=10):
    return np.random.rand(length)

train_data['svg_embedding'] = train_data.apply(lambda row: encode_svg_placeholder(), axis=1)
val_data['svg_embedding'] = val_data.apply(lambda row: encode_svg_placeholder(), axis=1)

# Display the first few rows of the dataframes with the new embeddings
display(train_data.head())
display(val_data.head())


import jax.numpy as jnp
import jax
import optax  # JAX optimizer
from flax.training import train_state  # Flax TrainState
import pandas as pd

# Placeholder for dataset (Ensure train_data & val_data exist)
train_data = pd.DataFrame({
    "prompt_embedding": [jnp.ones((1,)) for _ in range(5)],
    "svg_embedding": [jnp.ones((10,)) for _ in range(5)]
})
val_data = train_data.copy()

# Define DataLoader
class DataLoader:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for _, row in self.data.iterrows():
            yield jnp.array(row['prompt_embedding']).reshape(1,1), jnp.array(row['svg_embedding']).reshape(1,10)

    def __len__(self):
        return len(self.data)

train_loader = DataLoader(train_data)
val_loader = DataLoader(val_data)

# Dummy model function
def model(params, x):
    return x  # Placeholder model function

# Loss function
def compute_loss(params, model, prompt_embedding, svg_embedding):
    pred = model(params, prompt_embedding)
    return jnp.mean((pred - svg_embedding) ** 2)  # Example loss function (MSE)

# Training state definition
class TrainState(train_state.TrainState):
    pass

# Initialize model parameters and optimizer
params = {"weights": jnp.ones((1, 10))}  # Example parameters
tx = optax.sgd(learning_rate=0.01)  # Optimizer
state = TrainState.create(apply_fn=model, params=params, tx=tx)

# Training step
@jax.jit
def train_step(state, prompt_embedding, svg_embedding):
    def loss_fn(params):
        return compute_loss(params, model, prompt_embedding, svg_embedding)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

# Training loop
for epoch in range(10):  # Placeholder epochs
    for prompt_embedding, svg_embedding in train_loader:
        state = train_step(state, prompt_embedding, svg_embedding)

# Save the model (placeholder)
print(state.params)



import jax
import optax
import jax.numpy as jnp
from sklearn.model_selection import ParameterGrid

# Placeholder model and loss function
def model(params, x):
    return jnp.dot(params, x)

def compute_loss(params, model, prompt_embedding, svg_embedding):
    predictions = model(params, prompt_embedding)
    return jnp.mean(jnp.square(predictions - svg_embedding))  # Mean Squared Error

# Hyperparameter search space
param_grid = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [1, 2]
}

best_params = None
best_val_loss = float('inf')

for params in ParameterGrid(param_grid):
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']

    # Initialize optimizer and model parameters
    optimizer = optax.adam(learning_rate)
    params = jax.random.normal(jax.random.PRNGKey(0), (10, 1))  # Placeholder model parameters
    opt_state = optimizer.init(params)

    # Placeholder training loop
    val_losses = []
    for epoch in range(2): # Placeholder, reduce number of epochs for quick testing
      epoch_val_losses = []
      for prompt_embedding, svg_embedding in val_loader:
        # Placeholder train step
        def loss_fn(params):
          return compute_loss(params, model, prompt_embedding, svg_embedding)
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        epoch_val_losses.append(compute_loss(params, model, prompt_embedding, svg_embedding))
      val_losses.extend(epoch_val_losses)
    val_loss = jnp.mean(jnp.array(val_losses))

    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Validation Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params

print(f"\nBest Hyperparameters: {best_params}, Best Validation Loss: {best_val_loss}")


import numpy as np
import pandas as pd

# Placeholder dataset (Ensure 'description' exists)
val_data = pd.DataFrame({
    "prompt_embedding": [np.random.rand(1) for _ in range(5)],
    "svg_embedding": [np.random.rand(10) for _ in range(5)],
    "description": ["Sample text prompt" for _ in range(5)]  # Added 'description' column
})

# Simulate SVG generation (replace with actual model)
def generate_svg(prompt_embedding):
    return "<svg>...</svg>"  # Placeholder: Replace with actual SVG code

# Simulate SVG to PNG conversion (replace with cairosvg)
def svg_to_png(svg_code):
    return np.random.rand(28, 28, 3)  # Placeholder: Replace with actual PNG data

# Simulate CLIP similarity calculation (replace with SigLIP SoViT-400m)
def compute_clip_similarity(image_data, text_prompt):
    return np.random.rand()  # Placeholder: Replace with actual similarity score

# Evaluation loop
clip_similarity_scores = []
for _, row in val_data.iterrows():
    svg_code = generate_svg(row['prompt_embedding'])
    png_image = svg_to_png(svg_code)
    similarity_score = compute_clip_similarity(png_image, row['description'])  # Now 'description' exists
    clip_similarity_scores.append(similarity_score)

# Analyze results
mean_clip_similarity = np.mean(clip_similarity_scores)

print(f"Mean CLIP Similarity: {mean_clip_similarity}")

# Report findings (placeholder)
print("Evaluation Report:")
print("- Method: CLIP Similarity using SigLIP SoViT-400m (simulated)")
print("- Mean CLIP Similarity:", mean_clip_similarity)
print("- Limitations: This evaluation uses simulated data and is not based on a real trained model or actual SVG/PNG data.")


