print("Phase 1: Installing a bunch of stuff GPT told me to...")
# If Keras 3 and KerasNLP don't get me a 0.001 score boost, I'm going back to scikit-learn.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3

import os


# JAX backend. I need all the speed I can get before the competition ends.
os.environ["KERAS_BACKEND"] = "jax"
# Use all the memory. My laptop fan sounds like a jet engine, which means it's working.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


import pandas as pd
import numpy as np
import keras
import keras_nlp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Silence the warnings. I don't need that kind of negativity during a Kaggle competition.
import warnings
warnings.filterwarnings("ignore")

# --- Enable Mixed Precision ---
# Mixed precision. Another trick GPT said me to trust in. My GPU budget is $0 btw.
keras.mixed_precision.set_global_policy("mixed_float16")


print("\nPhase 2: Loading the data that stands between me and glory...")
# Let's see what horrors this CSV holds. Please, no weirdly formatted strings.
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# --- IMPROVEMENT: GRAB MORE DATA ---
# My last model overfit so badly it started predicting my own anxieties.
SAMPLE_SIZE = 5000
df_subscribed = train_df[train_df['y'] == 1].sample(n=SAMPLE_SIZE, random_state=42)
df_not_subscribed = train_df[train_df['y'] == 0].sample(n=SAMPLE_SIZE, random_state=42)
train_sample_df = pd.concat([df_subscribed, df_not_subscribed]).sample(frac=1, random_state=42)


# --- Serialization Function ---
# Forcing this beautiful tabular data into a text prompt. This is my villain origin story against XGBoost.
def serialize_row_to_prompt(row, is_training=True):
    feature_cols = [col for col in row.index if col not in ['id', 'y']]
    feature_string = "; ".join([f"{col.replace('_', ' ')}: {row[col]}" for col in feature_cols])
    
    prompt = f"Instruction:\nPredict if a customer will subscribe. Respond with 'yes' or 'no'.\n\nCustomer Data:\n{feature_string}\n\nPrediction:\n"
    
    if is_training:
        answer = "yes" if row['y'] == 1 else "no"
        return prompt + answer
    return prompt



# Turning data into prompts. If I can't beat them with gradient boosting, I'll confuse them with NLP.
print("Serializing data... This feels wrong, but the public leaderboard says it's right.")
train_prompts = [serialize_row_to_prompt(row) for _, row in train_sample_df.iterrows()]
print(f"Behold, my monstrous creation:\n{train_prompts[0]}")


print("\nPhase 3: Waking up Gemma. Please be the chosen one.")
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("/kaggle/input/m/keras/gemma/keras/gemma_2b_en/3")

# I saw LoRA in a paper once. I think it makes the model smarter? Let's turn it on.
gemma_lm.backbone.enable_lora(rank=1)
print("LoRA is on. My model is now officially 'state-of-the-art'.")

# Gradient checkpointing. A fancy term for "please don't run out of memory."
gemma_lm.backbone.gradient_checkpointing = True
print("Gradient checkpointing enabled. My kernel lives to fight another day.")

# Setting sequence length. If it's too long, it's slow. Too short, it's dumb. 128 it is.
gemma_lm.preprocessor.sequence_length = 128


print("\nPhase 4: Fine-tuning. This is where the magic (or the heartbreak) happens.")
optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


# --- IMPROVEMENT: FEWER EPOCHS ---
# One epoch. If it doesn't learn the task in one go, I'll just blend its output with a LightGBM model.
gemma_lm.fit(train_prompts, epochs=1, batch_size=1)
print("Fine-tuning complete. It's probably overfit, but so is everyone else's model.")



print("\nPhase 5: Prediction time. Let's see if this gamble paid off.")

yes_token_id = gemma_lm.preprocessor.tokenizer.token_to_id("yes")
no_token_id = gemma_lm.preprocessor.tokenizer.token_to_id("no")

test_prompts = [serialize_row_to_prompt(row, is_training=False) for _, row in test_df.iterrows()]

chunk_size = 20
batch_size = 1
all_predictions = []
import gc

# --- NEW METHOD: TEMPERATURE SCALING ---
# Turning down the temperature to make the model less confident about its bad decisions.
TEMPERATURE = 1.5

for i in range(0, len(test_prompts), chunk_size):
    start, end = i, i + chunk_size
    print(f"Processing chunk [{start}:{end}]... please don't crash...")
    chunk_prompts = test_prompts[start:end]
    
    chunk_logits = gemma_lm.predict(chunk_prompts, batch_size=batch_size, verbose=0)
    
    last_token_logits = chunk_logits[:, -1, :]
    
    # Scaling the logits. It's math. Don't question it. Just pray it works.
    scaled_logits = last_token_logits / TEMPERATURE
    
    all_probs = keras.ops.softmax(scaled_logits, axis=-1)
    yes_probs = all_probs[:, yes_token_id]
    no_probs = all_probs[:, no_token_id]
    
    # The epsilon is for numerical stability, and also for my emotional stability.
    pred_chunk = np.asarray(yes_probs / (yes_probs + no_probs + 1e-9))
    
    all_predictions.extend(pred_chunk)
        
    # Garbage collection. Cleaning up my code's memory, since I can't clean up my desk.
    del chunk_logits, last_token_logits, all_probs
    gc.collect()

test_predictions = np.array(all_predictions)
print(f"Generated {len(test_predictions)} predictions. The shake-up will decide my fate.")


print("\nPhase 6: Creating the submission file. My precious.")
submission_df = pd.DataFrame({'id': test_df['id'], 'y': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created. Now to upload it and refresh the page for three hours.")
print("\n--- Process Finished. Now to write the ensembling script: 0.5 * this + 0.5 * that_public_notebook_everyone_is_using ---")

