#Gemma 3n n is nano
!pip install unsloth
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    load_in_4bit = True, # Use 4bit to reduce memory use. 
    dtype = None, # None for auto detection
    max_seq_length = 2048,
    use_gradient_checkpointing = "unsloth",   
)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    random_state = 2407,
    use_rslora = False,               
    loftq_config = None,              
    target_modules = "all-linear", 
)


import os
from PIL import Image
from tqdm import tqdm

def load_binary_dataset(root_dir):
    data = []
    label_map = {
        "edible mushroom sporocarp": "edible",
        "poisonous mushroom sporocarp": "poisonous",
        "edible sporocarp": "edible",
        "poisonous sporocarp": "poisonous"
    }

    instruction_text = "Classify the mushroom as 'edible' or 'poisonous'."

    for folder_name, label in label_map.items():
        folder_path = os.path.join(root_dir, folder_name)
        for filename in tqdm(os.listdir(folder_path), desc=f"Loading {label}"):
            file_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(file_path).convert("RGB").resize((384, 384))
                sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction_text},
                                {"type": "image", "image": image},  # image after text
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": label}],
                        }
                    ]
                }
                data.append(sample)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    return data



dataset_path = "/kaggle/input/edible-and-poisonous-fungi/"
mushroom_data = load_binary_dataset(dataset_path)


from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(mushroom_data, test_size=0.1, random_state=2407)
print(f"Train size: {len(train_data)} | Val size: {len(val_data)}")


#rebalancing to have more edible represented 
from collections import Counter
import random

# Separate edible and poisonous
edible_samples = [x for x in train_data if x["messages"][1]["content"][0]["text"].lower() == "edible"]
poisonous_samples = [x for x in train_data if x["messages"][1]["content"][0]["text"].lower() == "poisonous"]

print("Before oversampling:", Counter([x["messages"][1]["content"][0]["text"].lower() for x in train_data]))

# Oversample edible to match poisonous count
while len(edible_samples) < len(poisonous_samples):
    edible_samples.append(random.choice(edible_samples))

# Combine and shuffle
train_data_balanced = edible_samples + poisonous_samples
random.shuffle(train_data_balanced)

print("After oversampling:", Counter([x["messages"][1]["content"][0]["text"].lower() for x in train_data_balanced]))


train_d = train_data_balanced
val_d = val_data


train_d[0]


from unsloth import get_chat_template

processor = get_chat_template(
    processor,
    "gemma-3n"
)


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def eval_mod(model):
    
    # Setup
    y_true = []
    y_pred = []
    
    
    iter=0
    for sample in val_d:
        iter+=1
        if (iter%100==0):
            print("Iter: ", iter)
        image = sample["messages"][0]["content"][1]["image"]  # type: PIL.Image
        true_label = sample["messages"][1]["content"][0]["text"].strip().lower()
        
    
        instruction = '''You are a mushroom identification expert.
        Look at the image and decide if the mushroom is safe to eat.
        If you are less than 80% sure it is edible, answer "poisonous".
        Respond with exactly one word: "edible" or "poisonous".'''
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            }
        ]
    
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda:0" if torch.cuda.is_available() else "cpu")
    
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)
            gen_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            prediction = processor.decode(gen_tokens, skip_special_tokens=True).strip().lower()
            
    
        y_true.append(true_label)
        y_pred.append(prediction)
    
    # === Evaluation ===
    print("ğŸ”¬ Classification Accuracy:", accuracy_score(y_true, y_pred))
    print("\nğŸ“Š Classification Report:")
    print(classification_report(y_true, y_pred, labels=["edible", "poisonous"], zero_division=0))
    
    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=["edible", "poisonous"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["edible", "poisonous"])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix - Mushroom Edibility")
    plt.tight_layout()
    plt.show()


import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.recompile_limit = 1024


FastVisionModel.for_inference(model)
eval_mod(model)


from transformers import TrainerCallback
import numpy as np

class PoisonousRecallAndValLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, output_dir="./best_model", tokenizer=None, processor=None):
        self.patience = patience
        self.wait = 0
        self.best_poisonous_recall = -np.inf
        self.best_val_loss = np.inf
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.processor = processor

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None:
            return control

        # Get from metrics dict
        val_loss = metrics.get("eval_loss", np.inf)

        improved = False

        # Check validation loss improvement
        if val_loss < self.best_val_loss:
            print(f"âœ… New best validation loss: {val_loss:.4f} (prev {self.best_val_loss:.4f})")
            self.best_val_loss = val_loss
            improved = True

        # Save if improved
        if improved:
            self.wait = 0
            save_path = os.path.join(self.output_dir, "best_val_loss_model")
            print(f"ğŸ’¾ Saving best model to {save_path}")
            model.save_pretrained(save_path)
            if self.processor:
                self.processor.save_pretrained(save_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
        else:
            self.wait += 1
            print(f"âš ï¸� No improvement. Patience: {self.wait}/{self.patience}")

        # Early stop
        if self.wait >= self.patience:
            print("ğŸ›‘ Early stopping triggered!")
            control.should_early_stop = True
            control.should_save = False

        return control



from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import random


# Shuffle training data (list of dicts)
random.seed(2407)
random.shuffle(train_d)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_d,
    eval_dataset=val_d,   # Needed for early stopping
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_grad_norm = 0.3,            
        warmup_ratio = 0.03,
        max_steps = 48,
        #num_train_epochs = 2,          
        learning_rate = 2e-4,
        logging_steps = 1,
        #save_strategy="steps",
        eval_strategy="steps",   # Run eval during training new
        eval_steps=4,                   # How often to evaluate new
        save_strategy="no",             # Weâ€™ll handle saving in callback new
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 2407,
        output_dir = "outputs",
        report_to = "none",             # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    )
)



callback = PoisonousRecallAndValLossEarlyStoppingCallback(
    patience=3,
    output_dir="outputs",
    tokenizer=processor.tokenizer,
    processor=processor
)
trainer.add_callback(callback)


trainer_stats = trainer.train()


FastVisionModel.for_inference(model)
eval_mod(model)


import shutil

model = model.merge_and_unload()

save_dir = "/kaggle/working/outputs/gsafe_full_finetuned"
zip_full = f"{save_dir}.zip"

if not os.path.exists(save_dir):
    # Save as a standard Hugging Face model
    model.save_pretrained(save_dir, safe_serialization=True)
    processor.save_pretrained(save_dir)

    shutil.make_archive(save_dir, 'zip', save_dir)
    print(f"Zipped model for download: {zip_full}")



from IPython.display import FileLink
FileLink(r'outputs/gsafe_full_finetuned.zip')


#(Optional) This is to save the best model, LoRa settings only. 

save_dir_lora = "/kaggle/working/outputs/best_val_loss_model"
zip_lora = "/kaggle/working/outputs/gsafe_finetuned_fromchkpt.zip"

if not os.path.exists(zip_lora):
    # Zip it up
    shutil.make_archive(zip_lora.replace(".zip", ""), 'zip', save_dir_lora)
    print(f"Zipped model for download: {zip_lora}")

