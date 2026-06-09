# 1) Install Dependencies (adjust if needed)
!pip install -q -U keras-nlp datasets pandas
!pip install -q -U keras

# 2) Set up JAX backend (optional)
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras
import keras_nlp
from datasets import Dataset, DatasetDict

# 3) Load the Already Fine-Tuned Model for Ancient Chinese Translation
model_id = "gemma2_instruct_2b_en"  # Example Gemma variant
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)
gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)

# Load the previous translator LoRA weights
previous_lora_path = "/kaggle/working/translator_chinese_4_epoch3.lora.h5"
gemma.backbone.load_lora_weights(previous_lora_path)
print("Loaded base Gemma + translator LoRA weights successfully!")

csv_path = "/kaggle/working/dataset/ancient_chinese_phonology.csv"  # Adjust path
df_acp = pd.read_csv(csv_path)

print(df_acp.head(5))


csv_path = "/kaggle/working/dataset/ancient_chinese_phonology.csv"  # Adjust path
df_acp = pd.read_csv(csv_path)

print(df_acp.head(5))



def build_single_char_phonology_prompts(df, tokenizer, token_limit=256):
    """Train the model to output phonology for single characters (no era guess)."""
    prompts = []
    for _, row in df.iterrows():
        char = str(row["character"])
        middle_tang = str(row.get("MiddleTang", "") or "")
        late_tang   = str(row.get("LateTang", "") or "")
        song        = str(row.get("Song", "") or "")
        yuan        = str(row.get("Yuan", "") or "")
        ming_qing   = str(row.get("MingQing", "") or "")
        mandarin    = str(row.get("Mandarin", "") or "")

        user_prompt = (
            f"<start_of_turn>user\n"
            f"Character: {char}\n"
            "Please provide the historical pronunciations.\n"
            f"<end_of_turn>\n"
        )
        model_response = (
            f"<start_of_turn>model\n"
            f"MiddleTang: {middle_tang}\n"
            f"LateTang: {late_tang}\n"
            f"Song: {song}\n"
            f"Yuan: {yuan}\n"
            f"MingQing: {ming_qing}\n"
            f"Mandarin: {mandarin}\n"
            f"<end_of_turn>"
        )

        text = user_prompt + model_response
        length = len(tokenizer(text))
        if length < token_limit:
            prompts.append(text)

    return prompts

single_char_prompts = build_single_char_phonology_prompts(df_acp, tokenizer, 256)
print("Single-char phonology samples:", len(single_char_prompts))
print("Example single-char prompt:\n", single_char_prompts[0])



# Suppose you also have some short ancient Chinese sentences in `anc_sents`,
# along with known (or approximate) era labels, approximate phonetic readings,
# and modern translations. For demonstration, let's just stub a small set.

# In a real scenario, you could:
#  - Use your translation dataset for ancient->modern
#  - Use ACP to build approximate sentence-level readings (concatenating each char)
#  - Potentially add an "era" label (or let the model guess)

sample_anc_sents = [
    {
        "text": "子曰：學而時習之，不亦說乎？",
        "era": "Spring and Autumn",   # or "Song", etc.
        "reading": "tsəi i̯wət: ɣɯ̯ak ʲɚ ɕʷi ɕip tʂɨ, pu ɰi i̯ɤ ɕuət xu?",  # Fake example
        "modern": "Confucius said: To study and practice frequently, is that not a delight?"
    },
    {
        "text": "太史公曰：匈奴之盛自冒頓始。",
        "era": "Han",
        "reading": "tʰai ɕi kʰuŋ i̯wət: ɕjuŋ nu tʂɨ ʂeŋ tsɨ maʊ tʰuən ʂi.",  # Fake example
        "modern": "The Grand Historian said: The Xiongnu's prosperity began with Maodun."
    }
]

def build_multi_task_prompts(data_list, tokenizer, token_limit=256):
    """Teach the model to respond with (1) era, (2) reading, (3) translation."""
    prompts = []
    for item in data_list:
        ancient_text = item["text"]
        era_label = item["era"]
        reading = item["reading"]
        modern = item["modern"]

        user_prompt = (
            f"<start_of_turn>user\n"
            f"Given the ancient Chinese sentence: 「{ancient_text}」\n"
            "1) Identify the historical era\n"
            "2) Provide the sentence-level pronunciation\n"
            "3) Provide the modern Chinese translation\n"
            f"<end_of_turn>\n"
        )
        model_response = (
            f"<start_of_turn>model\n"
            f"Era: {era_label}\n"
            f"Pronunciation: {reading}\n"
            f"Translation: {modern}\n"
            f"<end_of_turn>"
        )
        text = user_prompt + model_response

        length = len(tokenizer(text))
        if length < token_limit:
            prompts.append(text)
    return prompts

multi_task_prompts = build_multi_task_prompts(sample_anc_sents, tokenizer, token_limit=256)
print("Multi-step prompts:", len(multi_task_prompts))
print("Example multi-task prompt:\n", multi_task_prompts[0])


all_prompts = []
all_prompts.extend(single_char_prompts)     # character-level phonology
all_prompts.extend(multi_task_prompts)      # era + reading + translation


lora_rank_2 = 4
gemma.backbone.enable_lora(rank=lora_rank_2)
gemma.preprocessor.sequence_length = 256

lr_value_2 = 1e-5  # Consider using a smaller LR for second-phase fine-tuning
optimizer = keras.optimizers.AdamW(learning_rate=lr_value_2, weight_decay=0.01)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)



def text_gen(prompt):
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma.generate(input_text, max_length=256)
    print("\nGemma output:")
    print(output)

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save LoRA
        model_name = f"/kaggle/working/translator_plus_phonology_{lora_rank_2}_epoch{epoch+1}.lora.h5"
        gemma.backbone.save_lora_weights(model_name)
        print(f"\nLoRA weights saved to: {model_name}")

        # Quick test:
        text_gen("Character: 童\nPlease provide the historical pronunciations.")
        text_gen("Given the ancient Chinese sentence: 「子曰：學而時習之，不亦說乎？」"
                 "1) Identify the historical era\n"
                 "2) Provide the sentence-level pronunciation\n"
                 "3) Provide the modern Chinese translation")


train_epoch_2 = 2  # Adjust as desired
history = gemma.fit(
    all_prompts,
    epochs=train_epoch_2,
    batch_size=1, 
    validation_split=0.05,  # or separate dev set
    callbacks=[CustomCallback()],
)

# Plot
plt.plot(history.history['loss'], label="train_loss")
if "val_loss" in history.history:
    plt.plot(history.history['val_loss'], label="val_loss")
plt.title("Second-Phase Fine-Tuning Loss")
plt.legend()
plt.show()



# 7.1) Load final LoRA
final_lora_path = f"/kaggle/working/translator_plus_phonology_{lora_rank_2}_epoch{train_epoch_2}.lora.h5"
gemma.backbone.load_lora_weights(final_lora_path)

# 7.2) Test: Single character phonology
text_gen("Character: 冻\nPlease provide the historical pronunciations.")

# 7.3) Test: Multi-step
text_gen("Given the ancient Chinese sentence: 「太史公曰：匈奴之盛自冒頓始。」"
         "1) Identify the historical era\n"
         "2) Provide the sentence-level pronunciation\n"
         "3) Provide the modern Chinese translation")

print("Done!")

