from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, PretrainedConfig, PreTrainedModel,EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from peft import PeftModel
import torch, os, time
import torch.nn as nn
import numpy as np
import pandas as pd
from  tqdm import tqdm
import outlines
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Gemma3ForCausalLM, AutoConfig
model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"# /kaggle/input/gemma-3/transformers/gemma-3-27b-it-qat-q4_0-unquantized/1

system_prompt = """Classify the following math problem into the most appropriate topic from this list: Algebra, Geometry and Trigonometry, Calculus and Analysis, Probability and Statistics, Number Theory, Combinatorics and Discrete Math, Linear Algebra, Abstract Algebra and Topology. Respond with only the topic name.

THE PROBLEM:
{}"""


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,
                                          )
llm = AutoModelForCausalLM.from_pretrained(model_name,

                                                    device_map="auto",
                                                    offload_folder="./offload",
                                                    torch_dtype="auto",
                                                    attn_implementation="flash_attention_2"
                                                    )



# # # # Load the LLaMA-Factory trained adapter
# adapter_path = "" # PUT YOUR ADAPTER 
# llm = PeftModel.from_pretrained(llm, adapter_path)

llm.eval() 
model = outlines.models.Transformers(llm, tokenizer)


class GreedySampler:
    """Greedy Sampling algorithm.

    Greedy sampling consists in choosing the token with the largest
    likelihood at every step.

    We don't allow more than one sample. We could attribute this a meaning, for
    instance the k-th sample represents the k-th most likely token. In which
    case it would be equivalent to beam search without the sequence weights.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self):
        self.samples = 1

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        _,
    ) -> "torch.DoubleTensor":
        """Call the greedy sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        next_token_ids = torch.argmax(logprobs, dim=-1, keepdim=True)

        ancestors = torch.arange(
            next_token_logits.shape[0], device=next_token_logits.device
        )
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights

    @property
    def sampling_params(self):
        return outlines.samplers.SamplingParameters("greedy", self.samples, None, None, None)

topic_dict = {'Algebra': 0,
 'Geometry and Trigonometry': 1,
 'Calculus and Analysis': 2,
 'Probability and Statistics': 3,
 'Number Theory': 4,
 'Combinatorics and Discrete Math': 5,
 'Linear Algebra': 6,
 'Abstract Algebra and Topology': 7}


generator = outlines.generate.choice(model, topic_dict.keys(), sampler=GreedySampler())




def classify_math_problem(problem):
    try:
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                                                    [   
                                                        # {"role":"system", "content":"You are a knowledgeable mathematics assistant."},
                                                        {"role": "user", "content": system_prompt.format(problem)},
                                                    ],
                                                    tokenize=False,
                                                    add_bos=True,
                                                    add_generation_prompt=True,
                                                    enable_thinking=False
                                                )
        else:
            prompt = system_prompt.format(problem)
        
        answer = generator(prompt, max_tokens=30)
        return answer
    except Exception as e:
        print(f"Error: {e}")
        return None

def classify_with_sub(df, save_every_n=10, save_path="classified_output.csv"):
    # Load already processed rows
    if os.path.exists(save_path):
        processed_df = pd.read_csv(save_path)
        start_index = len(processed_df)
        print(f"Resuming from saved file with {start_index} rows already processed.")
    else:
        processed_df = pd.DataFrame(columns=["Question", "label", "Topic"])
        start_index = 0
        print("Starting fresh classification.")

    processed_rows = processed_df.to_dict("records")

    for i in tqdm(range(start_index, len(df))):
        row = df.iloc[i]

        topic = classify_math_problem(row["Question"])

        if topic is not None:
            processed_rows.append({
                "id" : i,
                "Question": row["Question"],
                "Topic": topic,
            })
        else:
            print(f"Skipping row {i} due to null result.")

        # Save every N or on last row
        if (len(processed_rows)) % save_every_n == 0 or (i + 1) == len(df):
            pd.DataFrame(processed_rows).to_csv(save_path, index=False)

        time.sleep(0.1)

    return pd.DataFrame(processed_rows)

def classify_with_saving(df, save_every_n=10, save_path="classified_output.csv"):
    # Load already processed rows
    if os.path.exists(save_path):
        processed_df = pd.read_csv(save_path)
        start_index = len(processed_df)
        print(f"Resuming from saved file with {start_index} rows already processed.")
    else:
        processed_df = pd.DataFrame(columns=["Question", "label", "Topic"])
        start_index = 0
        print("Starting fresh classification.")

    processed_rows = processed_df.to_dict("records")

    for i in tqdm(range(start_index, len(df))):
        row = df.iloc[i]

        topic = classify_math_problem(row["Question"])

        if topic is not None:
            processed_rows.append({
                "Question": row["Question"],
                "label": row["label"],
                "Topic": topic,
            })
        else:
            print(f"Skipping row {i} due to null result.")

        # Save every N or on last row
        if (len(processed_rows)) % save_every_n == 0 or (i + 1) == len(df):
            pd.DataFrame(processed_rows).to_csv(save_path, index=False)

        time.sleep(0.1)

    return pd.DataFrame(processed_rows)




df = pd.read_csv("train.csv")
max_per_class = 50 
sampled_df = (
    df.groupby('label', group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=41))
)
classified_df = classify_with_saving(sampled_df, save_every_n=10, save_path=f"{model_name.replace('/','|')}_{prompt_name}_max_sample{max_per_class}.csv")
classified_df['pred'] = classified_df['Topic'].apply(lambda x: topic_dict[x])


from sklearn.metrics import f1_score

f1_score(classified_df['label'], classified_df['pred'], average="micro")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assume df['label'] and df['pred'] contain the encoded topic codes (0-7)
cm = confusion_matrix(classified_df['label'], classified_df['pred'])

# Reverse the topic_dict to get label names
# inv_topic_dict = {v: k for k, v in topic_dict.items()}
labels = topic_dict.keys()

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Topic')
plt.ylabel('Actual Topic')
plt.title('Confusion Matrix by Topic')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


df = pd.read_csv("test.csv", index_col="id")
classified_df = classify_with_sub(df, save_every_n=10, save_path=f"{model_name.replace('/','|')}_{prompt_name}_test.csv")
classified_df['label'] = classified_df['Topic'].apply(lambda x: topic_dict[x])
classified_df[['id', 'label']].to_csv("sub.csv" ,index=False)


%rm -rf LLaMA-Factory
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
%cd LLaMA-Factory
%ls
!pip install -e .[torch,bitsandbytes]


import torch
try:
  assert torch.cuda.is_available() is True
except AssertionError:
    prin("GPU Isn't There!!")


! cp /kaggle/input/math-topic-classification-competition-mixed-data/math_topics.json /kaggle/working/LLaMA-Factory/data



import json
with open("/kaggle/working/LLaMA-Factory/data/dataset_info.json","r") as f:
    data = json.load(f)
data["math_topics"] = {"file_name": "math_topics.json"}
with open("/kaggle/working/LLaMA-Factory/data/dataset_info.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


import json

args = dict(
  stage="sft",
  do_train=True,
  model_name_or_path="unsloth/gemma-3-27b-it-unsloth-bnb-4bit" # you may use after adding the model to models "/kaggle/input/gemma-3/transformers/gemma-3-27b-it-qat-q4_0-unquantized/1", # 
  dataset="math_topics",
  template="gemma3",
  finetuning_type="lora",
  lora_rank=16,
  lora_target="all",
  # use_unsloth=True,
  use_unsloth_gc=True,
  do_sample=False,
  max_new_tokens=8,
  cutoff_len=256,
  max_samples=10000,
  overwrite_cache=True,
  # preprocessing_num_workers=12,
  # dataloader_num_workers=4,
  default_system="",
  output_dir="saves/gemma-3-math/lora/sft",
  logging_steps=10,
  save_steps=100,
  plot_loss=True,
  overwrite_output_dir=True,
  save_only_model=False,
  report_to="none",
  per_device_train_batch_size=2,
  gradient_accumulation_steps=8,
  learning_rate=1e-4,
  num_train_epochs=4.0,
  lr_scheduler_type="cosine",
  warmup_ratio=0.1,
  bf16=True,
  optim="adamw_8bit",
  seed=0,
  ddp_timeout=180000000,
  trust_remote_code=False,
  quantization_bit=4,
  quantization_method="bnb",
)

json.dump(args, open("train_gemma3.json", "w", encoding="utf-8"), indent=2)


!llamafactory-cli train train_gemma3.json



LLM_NAME = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit" 
DATA  = "second_stage_1200_256"
# # # # Load the LLaMA-Factory trained adapter
adapter_path = "" # PUT YOUR ADAPTER I Used the last one 
def create_embeedings(df, label=True):

    
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)

    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME,

                                                        device_map="auto",
                                                        offload_folder="./offload",
                                                        torch_dtype="auto",


    llm = PeftModel.from_pretrained(llm, adapter_path)
    llm = llm.language_model
    llm.lm_head = nn.Identity()

    torch.cuda.empty_cache()
    
    embeddings = []

    if label:
        labels = [] 
    else:
        ids = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row["Question"]

        # Tokenize the question
        tokens =tokenizer.apply_chat_template(
                                                        [
                                                            {"role": "user", "content": system_prompt.format(question)},
                                                        ],
                                                        tokenize=True,
                                                        add_bos=True,
                                                        add_generation_prompt=True,
                                                        return_tensors="pt"
                                                    ).to('cuda')
        # Generate embeddings
        # Assuming llm(tokens) returns the logits (raw output of the model)
        try:
            with torch.inference_mode():
                embed = llm(tokens).logits[0,-1].cpu()  
        except Exception as e:
            print(f"Error generating embedding for question '{question}': {e}")
            continue  # Skip to the next question if embedding generation fails


        embeddings.append(embed)
        if label:
            labels.append(row["label"])
        else:
            ids.append(index)


    del llm,tokenizer
    torch.cuda.empty_cache()

    if label:
        return embeddings, labels
    
    return embeddings, ids



df = pd.read_csv(f"{DATA}.csv")
class_counts = df['label'].value_counts()
class_weights= [len(df)/(class_counts[i]*len(class_counts))**0.4 for i in range(len(class_counts))]
weights_sum = sum(class_weights)
class_weights=[c/weights_sum for c in class_weights]
class_weights


embeddings,labels = create_embeedings(df)


embeddings = [e.tolist() for e in embeddings]


dataset = Dataset.from_dict({
    'embedding': embeddings,
    'label': df['label']
})
dataset.save_to_disk(f"{DATA}")


from datasets import load_from_disk
dataset = load_from_disk(f"{DATA}")
# dataset = load_from_disk("/kaggle/input/math-topic-classification-competition-mixed-data/second_stage_1200_256/second_stage_1200_256") #from computed embeddings


class QClassifierConfig(PretrainedConfig):
    def __init__(self, hidden_size=5376, num_labels=8, class_weights=None, **kwargs): #5376 , 5120
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.class_weights = class_weights  # should be a list or None



class QClassifierModel(PreTrainedModel):
    config_class = QClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size,1024), nn.ReLU()(), nn.Linear(1024, config.num_labels))

        if config.class_weights is not None:
            self.register_buffer(
                "class_weights_tensor", torch.tensor(config.class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights_tensor = None

    def forward(self, embedding, labels=None):
        logits = self.classifier(embedding)
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor, label_smoothing=0.13 #0.12
            )
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    
config = QClassifierConfig(class_weights=class_weights)
model = QClassifierModel(config)


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        'f1': f1_score(labels, predictions, average='micro'),
    }





from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import copy

# Assuming `dataset` is a Hugging Face Dataset object and has a 'label' column
labels = dataset['label']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_metrics = []
all_confusion_matrices = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n===== Fold {fold + 1} =====")

    # Select splits
    train_dataset = dataset.select(train_idx.tolist())
    valid_dataset = dataset.select(val_idx.tolist())
    model_fold = copy.deepcopy(model)

    
    fold_training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold}",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.07,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        remove_unused_columns=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model_fold,
        args=fold_training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    

    trainer.train()
    metrics = trainer.evaluate()
    all_metrics.append(metrics)

    # Predict on validation set
    predictions_output = trainer.predict(valid_dataset)
    true_labels = predictions_output.label_ids
    predicted_labels = np.argmax(predictions_output.predictions, axis=-1)

    # Compute and store confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    all_confusion_matrices.append(cm)
    print(f"Confusion Matrix for Fold {fold + 1}:\n{cm}")

# Average metrics
avg_metrics = {
    key: np.mean([m[key] for m in all_metrics])
    for key in all_metrics[0]
}
print("\nâœ… Average metrics across folds:", avg_metrics)




import seaborn as sns
import matplotlib.pyplot as plt
# Convert list of confusion matrices to numpy array for easy averaging
confusion_matrices_array = np.array(all_confusion_matrices)  # Shape: (n_folds, num_classes, num_classes)

# Average confusion matrix
avg_confusion_matrix = np.mean(confusion_matrices_array, axis=0)

# Normalize the confusion matrix (row-wise normalization)
normalized_avg_confusion_matrix = avg_confusion_matrix / avg_confusion_matrix.sum(axis=1, keepdims=True)

labels = ['Algebra', 'Geometry and Trigonometry', 'Calculus and Analysis', 'Probability and Statistics', 'Number Theory', 'Combinatorics and Discrete Math', 'Linear Algebra', 'Abstract Algebra and Topology']
# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_avg_confusion_matrix, annot=True, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Topic')
plt.ylabel('Actual Topic')
plt.title('Confusion Matrix by Topic')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



config = QClassifierConfig(class_weights=class_weights)
model = QClassifierModel(config)

# Update output_dir to avoid overwriting
training_args = TrainingArguments(
output_dir="./results",
num_train_epochs=8,
per_device_train_batch_size=32,
per_device_eval_batch_size=32,
learning_rate=1e-4,
weight_decay=0.07, #0.06
logging_dir='./logs',
logging_steps=100,
# eval_strategy="steps",
save_strategy="steps",
save_steps=100,
remove_unused_columns=True,  
# bf16=True,
# save_safetensors=False,
# load_best_model_at_end=True,
# metric_for_best_model='eval_f1',
# greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # compute_metrics=compute_metrics,
)


trainer.train()


predictions_output = trainer.predict(dataset)
true_labels = predictions_output.label_ids
logits = predictions_output.predictions
# For classification, get the predicted class index by finding the argmax of the logits
predicted_labels = np.argmax(logits, axis=-1)
cm = confusion_matrix(true_labels, predicted_labels, normalize='true')


import seaborn as sns
import matplotlib.pyplot as plt
labels = ['Algebra', 'Geometry and Trigonometry', 'Calculus and Analysis', 'Probability and Statistics', 'Number Theory', 'Combinatorics and Discrete Math', 'Linear Algebra', 'Abstract Algebra and Topology']
# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Topic')
plt.ylabel('Actual Topic')
plt.title('Confusion Matrix by Topic')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


df = pd.read_csv("test.csv")
torch.cuda.empty_cache()
embeddings, ids = create_embeedings(df, label=False)



test_dataset = Dataset.from_dict({
    'embedding': [ e.tolist()for e in embeddings],
    'ids': ids
})


test_dataset.save_to_disk("") # PUT THE PATH OF SAVING


test_dataset = load_from_disk("") # PUT THE PATH OF SAVING Or use the precomputed bellow
# test_dataset = load_from_disk("/kaggle/input/math-topic-classification-competition-mixed-data/gemma3_finetunePrompt_test_tuined800/gemma3_finetunePrompt_test_tuined800") # pre computed embeddings


preds = trainer.predict(test_dataset)
preds.predictions.shape


sub_df = test_dataset.to_pandas()


sub_df['label'] =np.argmax(preds.predictions, axis=1)
sub_df.rename(columns={'ids':'id'}, inplace=True)
sub_df[['id', 'label']].to_csv("sub.csv" ,index=False)

