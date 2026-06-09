!wget https://huggingface.co/datasets/RUCAIBox/Erya-dataset/resolve/main/finetune.tgz
!tar -xzf finetune.tgz


import os
import time
import matplotlib.pyplot as plt

# 1) Install Dependencies
!pip install -q -U keras-nlp datasets
!pip install -q -U keras

# 2) Set up Keras JAX backend
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras
import keras_nlp
from datasets import load_dataset
from datasets import Dataset, DatasetDict


# 3) Fine-tuning Configurations
token_limit = 256      # Adjust if needed
lora_name = "translator_chinese"  # LoRA weight file name prefix
lora_rank = 4          # Typically 4, 8, or 16
lr_value = 1e-4
train_epoch = 3        # Adjust if needed
model_id = "gemma2_instruct_2b_en"  # Example: Gemma instruct model


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)
gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)


def load_parallel(src_file, tgt_file):
    """Returns two lists: [ancient_lines], [modern_lines]."""
    anc, mod = [], []
    with open(src_file, "r", encoding="utf-8") as fsrc, \
         open(tgt_file, "r", encoding="utf-8") as ftgt:
        for a_line, m_line in zip(fsrc, ftgt):
            anc.append(a_line.strip())
            mod.append(m_line.strip())
    return anc, mod




folders = ["shij", "mings", "hans"]  # Add or remove as needed
base_path = "/kaggle/working/dataset"

# We'll accumulate train/valid/test lines from *all* these subfolders.
train_anc_all, train_mod_all = [], []
valid_anc_all, valid_mod_all = [], []
test_anc_all,  test_mod_all  = [], []

for sub in folders:
    folder_path = f"{base_path}/{sub}"
    # Example paths:
    train_src_path = f"{folder_path}/train.src"
    train_tgt_path = f"{folder_path}/train.tgt"
    valid_src_path = f"{folder_path}/valid.src"
    valid_tgt_path = f"{folder_path}/valid.tgt"
    test_src_path  = f"{folder_path}/test.src"
    test_tgt_path  = f"{folder_path}/test.tgt"

    # 3.1) Load each split for this folder
    if os.path.exists(train_src_path) and os.path.exists(train_tgt_path):
        anc_lines, mod_lines = load_parallel(train_src_path, train_tgt_path)
        train_anc_all.extend(anc_lines)
        train_mod_all.extend(mod_lines)

    if os.path.exists(valid_src_path) and os.path.exists(valid_tgt_path):
        anc_lines, mod_lines = load_parallel(valid_src_path, valid_tgt_path)
        valid_anc_all.extend(anc_lines)
        valid_mod_all.extend(mod_lines)

    if os.path.exists(test_src_path) and os.path.exists(test_tgt_path):
        anc_lines, mod_lines = load_parallel(test_src_path, test_tgt_path)
        test_anc_all.extend(anc_lines)
        test_mod_all.extend(mod_lines)

# 3.2) Convert to Hugging Face Datasets
train_dataset = Dataset.from_dict({"ancient": train_anc_all, "modern": train_mod_all})
valid_dataset = Dataset.from_dict({"ancient": valid_anc_all, "modern": valid_mod_all})
test_dataset  = Dataset.from_dict({"ancient": test_anc_all,  "modern": test_mod_all})

erya_data = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset,
    "test": test_dataset
})

print(erya_data)


def build_prompt_data(ds, tokenizer, token_limit=256):
    prompt_data = []
    for ancient, modern in zip(ds["ancient"], ds["modern"]):
        # Example prompt format for translator
        text = (
            f"<start_of_turn>user\n{ancient}<end_of_turn>\n"
            f"<start_of_turn>model\n{modern}<end_of_turn>"
        )
        # Skip overly long samples
        length = len(tokenizer(text))
        if length < token_limit:
            prompt_data.append(text)
    return prompt_data

train_prompts = build_prompt_data(erya_data["train"], tokenizer, token_limit)
valid_prompts = build_prompt_data(erya_data["validation"], tokenizer, token_limit)
test_prompts  = build_prompt_data(erya_data["test"], tokenizer, token_limit)

print("Train:", len(train_prompts))
print("Validation:", len(valid_prompts))
print("Test:", len(test_prompts))
print("Example prompt:\n", train_prompts[0])



gemma.summary()




tick_start = 0
def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def text_gen(prompt):
    tick()
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma.generate(input_text, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()

# Example test prompt (random ancient Chinese line).
text_gen("子曰：學而時習之，不亦說乎？")



gemma.backbone.enable_lora(rank=lora_rank)
gemma.preprocessor.sequence_length = token_limit

optimizer = keras.optimizers.AdamW(learning_rate=lr_value, weight_decay=0.01)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


import keras
import keras_nlp
# 4) Custom checkpoint callback
class LoraCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, file_pattern="gemma_lora_epoch{epoch:02d}.h5"):
        super().__init__()
        self.file_pattern = file_pattern
    
    def on_epoch_end(self, epoch, logs=None):
        filename = self.file_pattern.format(epoch=epoch + 1)
        gemma.backbone.save_lora_weights(filename)
        print(f"LoRA checkpoint saved to {filename}!")

checkpoint_cb = LoraCheckpointCallback("gemma_lora_epoch{epoch:02d}.h5")

# 5) Fit
history = gemma.fit(
    train_prompts,
    epochs=3,
    batch_size=1,
    validation_data=(valid_prompts,),
    callbacks=[checkpoint_cb],
)

# Now you have files: gemma_lora_epoch01.h5, gemma_lora_epoch02.h5, gemma_lora_epoch03.h5



plt.plot(history.history['loss'], label="train_loss")
if "val_loss" in history.history:
    plt.plot(history.history['val_loss'], label="val_loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()



print("----- Evaluate on a test line -----")
text_gen("太史公曰：匈奴之盛自冒頓始。")




