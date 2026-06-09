#importing packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder #encoding labels
from sklearn.model_selection import train_test_split #data split to train and validation
from datasets import Dataset #for working with torch dataloaders
from transformers import AutoTokenizer, AutoModelForSequenceClassification,BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
import os
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import numpy as np
import gc
import joblib


temp_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
print("loaded training data")
df1=pd.read_csv("/kaggle/input/synthetic-datamap/Synthetic Data.csv")
print("loaded synthetic data")
df=pd.concat([temp_df,df1])
print("Combined synthetic data and training data to create final dataset")


print("data shape before removing duplicates\n",df.shape)
df=df.drop_duplicates(subset=['QuestionId', 'QuestionText', 'MC_Answer','StudentExplanation', 'Category', 'Misconception'])
print("data shape after removing duplicates\n",df.shape)


df['Misconception']=df['Misconception'].fillna("NA")
print("Replaced null values in misconception column with NA\n")


#lets create a question and answer dictionary
df_true = df[df['Category'].str.startswith('True')]
correct_answers = (
    df_true.groupby("QuestionText")["MC_Answer"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)
answers_dict=dict(zip(correct_answers['QuestionText'],correct_answers['MC_Answer']))
print("Question text and correct answer mapping dictionary is created")


#lets create question and mathematical concept dictionary
Question_MC={
    "A bag contains \\( 24 \\) yellow and green balls. \\( \\frac{3}{8} \\) of the balls are yellow. How many of the balls are green?": "Ratio and Proportion",
    
    "A box contains \\( 120 \\) counters. The counters are red or blue. \\( \\frac{3}{5} \\) of the counters are red.\nHow many red counters are there?": "Fractions",
    
    "Calculate \\( \\frac{1}{2} \\div 6 \\)": "Fractions",
    
    "Calculate \\( \\frac{2}{3} \\times 5 \\)": "Fractions",
    
    "Dots have been arranged in these patterns: [Image: Pattern 1 consists of 6 dots, Pattern 2 consists of 10 dots, Pattern 3 consists of 14 dots and Pattern 4 consists of 18 dots] How many dots would there be in Pattern \\( 6 \\) ?": "Sequences",
    
    "It takes \\( 3 \\) people a total of \\( 192 \\) hours to build a wall.\n\nHow long would it take if \\( 12 \\) people built the same wall?": "Work and Time",
    
    "Sally has \\( \\frac{2}{3} \\) of a whole cake in the fridge. Robert eats \\( \\frac{1}{3} \\) of this piece. What fraction of the whole cake has Robert eaten?\nChoose the number sentence that would solve the word problem.": "Fractions",
    
    "The probability of an event occurring is \\( 0.9 \\).\n\nWhich of the following most accurately describes the likelihood of the event occurring?": "Probability",
    
    "This is part of a regular polygon. How many sides does it have? [Image: A diagram showing an obtuse angle labelled 144 degrees]": "Geometry",
    
    "What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]": "Fractions",
    
    "What number belongs in the box?\n\\(\n(-8)-(-5)=\n\\square\\)": "Integers",
    
    "Which number is the greatest?": "Comparing Numbers",
    
    "\\( 2 y=24 \\) What is the value of \\( y \\) ?": "Algebra",
    
    "\\( \\frac{1}{3}+\\frac{2}{5}= \\)": "Fractions",
    
    "\\( \\frac{A}{10}=\\frac{9}{15} \\) What is the value of \\( A \\) ?": "Ratio and Proportion"
}
print("Question text and mathematical topic mapping dictionary is created\n")


#let's create a new column topic
df['Topic']=df.apply(lambda row:Question_MC[row['QuestionText']],axis=1)
print("created a new column topic in our dataset\n")


#creating dictionary of most frequent misconceotions in each topic
topic_mc = (
    df[df['Misconception'] != 'NA']  # Filter out 'NA'
    .groupby('Topic')['Misconception']
    .value_counts()
    .groupby(level=0)
    .head(5)
    .reset_index(name='Count')
    .groupby('Topic')['Misconception']
    .apply(list)
    .to_dict()
)
print("Topic vs top 7 frequent misconceptions dictionary is created")


df['target']=df['Category']+":"+df["Misconception"]
print("created traget column as combination of category and misconception\n")


le=LabelEncoder()
df['label']=le.fit_transform(df['target'])
Num_classes=df['label'].nunique()
print("encoded target column and created a nwe column label using Label encoder. Total no of classes in label:",Num_classes)



def equal(x,y):
    if x==y:
        return "True"
    return "False"


def prompt_template1(row):
    return (
        f"Task: Identify up to three mathematical misconceptions from the student's explanation.\n\n"
        f"Question Topic: {Question_MC[row['QuestionText']]}\n"
        f"Mathematical Question: {row['QuestionText']}\n"
        f"Student's Answer: {row['MC_Answer']} (Correct: {equal(row['MC_Answer'], answers_dict[row['QuestionText']])})\n"
        f"Student's Explanation: {row['StudentExplanation']}\n"
        f"Most common misconceptions: {', '.join(topic_mc[Question_MC[row['QuestionText']]])}\n\n"
        f"Instructions: Review the student's explanation carefully. "
        f"List the top three misconceptions that explain the student's misunderstanding . "
    )
print("prompt template for Gemma 2 9b It model defined as prompt template1")


def prompt_template2(row):

    return(f"Mathematical Question: {row['QuestionText']}\n"
           f"Question topic: {Question_MC[row['QuestionText']]}\n"
          f"Student's answer: {row['MC_Answer']}\n"
          f"Answer Category: {equal(row['MC_Answer'],answers_dict[row['QuestionText']])}\n" 
         f"Student's Explanation: {row['StudentExplanation']} \n"
         f"Common Misconceptions : {topic_mc[Question_MC[row['QuestionText']]]}\n")
print("prompt template for Deep Seek R1 distill llama 8B defined as prompt template 2")


def prompt_template3(row):
    return (
        f"Question Topic: {Question_MC[row['QuestionText']]}\n"
        f"Mathematical Question: {row['QuestionText']}\n"
        f"Student's Answer: {row['MC_Answer']} (Correct: {equal(row['MC_Answer'], answers_dict[row['QuestionText']])})\n"
        f"Student's Explanation: {row['StudentExplanation']}\n"
        f"Most common misconceptions: {', '.join(topic_mc[Question_MC[row['QuestionText']]])}\n\n"
    )


quantization_config = BitsAndBytesConfig(load_in_8bit=True)
print("defined 8 bit quantization configuration")


model_name="/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=Num_classes, quantization_config=quantization_config,torch_dtype=torch.float16,device_map='cuda:0',)
model.config.pad_token_id=model.config.eos_token_id
print("loaded base model gemma 2 9b it to cuda 0")


lora_config = LoraConfig(
    r=16,                             # Low-rank matrix size
    lora_alpha=32,                   # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","up_proj", "down_proj", "gate_proj"],   # e.g., attention/query/key, check model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"]
    
)
print("defined lora configuration")


Optimized_model = PeftModel.from_pretrained(model,"/kaggle/input/finetunedgemma29b/transformers/default/4/checkpoint-1800")
Optimized_model.eval()
Optimized_model.to("cuda:0")
Optimized_model.config.pad_token_id = tokenizer.pad_token_id
print("loaded pre trained peft model from best checkpoint path to cuda 0")


test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_df['text']=test_df.apply(prompt_template1,axis=1)
# Dataset + tokenization
ds_test = Dataset.from_pandas(test_df[["text"]])
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
ds_test = ds_test.map(tokenize, batched=True)

ds_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
print("testing data is loaded and preprocessed")


dataloader = DataLoader(ds_test, batch_size=12)
all_logits = []

device = next(model.parameters()).device
with torch.inference_mode():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = Optimized_model(**batch)
        logits = outputs.logits.detach().cpu()
        all_logits.append(logits)

logits = torch.cat(all_logits, dim=0).float()
probs1 = torch.nn.functional.softmax(logits, dim=-1).numpy()
print("Infernce on test data completed by model 1")


del Optimized_model
del model  
gc.collect()
torch.cuda.empty_cache()
print("deleted optimized model and base model  from GPU memory for model 1")


quantization_config = BitsAndBytesConfig(load_in_8bit=True)
print("defined quantization config for model 2")


model_name="/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-llama-8b/2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=Num_classes, quantization_config=quantization_config,torch_dtype=torch.float16,device_map='cuda:0',)
model.config.pad_token_id=model.config.eos_token_id
print("loaded base model 2  deep seek r1 llama 8b ")


lora_config = LoraConfig(
    r=16,                             # Low-rank matrix size
    lora_alpha=32,                   # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","up_proj", "down_proj", "gate_proj"],   # e.g., attention/query/key, check model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"]
    
)


Optimized_model = PeftModel.from_pretrained(model,"/kaggle/input/finetuned8bqlorasynthesized/transformers/default/2/checkpoint-1600")
Optimized_model.eval()
Optimized_model.to("cuda:0")
Optimized_model.config.pad_token_id = tokenizer.pad_token_id
print("peft model loaded for model 2")



test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_df['text']=test_df.apply(prompt_template2,axis=1)
# Dataset + tokenization
ds_test = Dataset.from_pandas(test_df[["text"]])
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
ds_test = ds_test.map(tokenize, batched=True)

ds_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
print("test data loaded and preprocessed for model 2")


# --- Inference Test Data---
dataloader = DataLoader(ds_test, batch_size=12)
all_logits = []

device = next(model.parameters()).device
with torch.inference_mode():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = Optimized_model(**batch)
        logits = outputs.logits.detach().cpu()
        all_logits.append(logits)

logits = torch.cat(all_logits, dim=0).float()
probs2 = torch.nn.functional.softmax(logits, dim=-1).numpy()
print("model 2 infernce completed on test data")


del Optimized_model
del model  
gc.collect()
torch.cuda.empty_cache()
print("deleted optimized model and base model  from GPU memory for model 2")



quantization_config = BitsAndBytesConfig(load_in_8bit=True)
print("quantization config defined for model 3")


model_name="/kaggle/input/qwen2.5/transformers/7b/1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=Num_classes, quantization_config=quantization_config,torch_dtype=torch.float16,device_map='auto',)
model.config.pad_token_id=model.config.eos_token_id
print(" model 3 qwen 2.5 7B is loaded on cuda0")


#Define LoRA Configuration
lora_config = LoraConfig(
    r=16,                             # Low-rank matrix size
    lora_alpha=32,                   # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","up_proj", "down_proj", "gate_proj"],   # e.g., attention/query/key, check model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"]
    
)
print("lora config defined for model 3")



Optimized_model = PeftModel.from_pretrained(model,"/kaggle/input/finetunedqwen2.5-7b-model/transformers/default/3/checkpoint-1800")
Optimized_model.eval()
Optimized_model.to("cuda:0")
Optimized_model.config.pad_token_id = tokenizer.pad_token_id
print("peft model loaded for model 3")


test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test_df['text']=test_df.apply(prompt_template3,axis=1)
# Dataset + tokenization
ds_test = Dataset.from_pandas(test_df[["text"]])
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
ds_test = ds_test.map(tokenize, batched=True)

ds_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
print("test data loaded and preprocessed for model 3")


# --- Inference Test Data---
dataloader = DataLoader(ds_test, batch_size=12)
all_logits = []

device = next(model.parameters()).device
with torch.inference_mode():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = Optimized_model(**batch)
        logits = outputs.logits.detach().cpu()
        all_logits.append(logits)

logits = torch.cat(all_logits, dim=0).float()
probs3 = torch.nn.functional.softmax(logits, dim=-1).numpy()
print("model 3 infernce completed on test data")


del Optimized_model
del model  
gc.collect()
torch.cuda.empty_cache()
print("deleted optimized model and base model  from GPU memory for model 3")



# Load model
best_model = joblib.load("/kaggle/input/meta-model/best_model (1).pkl")
print("✅ Meta-model loaded successfully.")


meta_X_test = np.concatenate([probs1, probs2, probs3], axis=1)
print("created meta x test by concatenating probs of best score llms")


if hasattr(best_model, "decision_function"):
    logits = best_model.decision_function(meta_X_test)  # shape: (n_samples, n_classes)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    print("utilized decision function")
else:
    # fallback in case model doesn't have decision_function (unlikely)
    preds = best_model.predict(meta_X_test)
    probs = np.zeros((len(preds), np.max(preds) + 1))
    probs[np.arange(len(preds)), preds] = 1.0
    print("model does not have decision function")



top3 = np.argsort(-probs, axis=1)[:, :3]

# Decode class indices to labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Format submission
joined_preds = [" ".join(row[:3]) for row in top3_labels]
sub = pd.DataFrame({
    "row_id": test_df.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

