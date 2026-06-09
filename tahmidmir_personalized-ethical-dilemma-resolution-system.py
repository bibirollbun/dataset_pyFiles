import pandas as pd
import numpy as np
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                         AlbertTokenizer, AlbertForSequenceClassification,
                         MobileBertTokenizer, MobileBertForSequenceClassification,
                         T5Tokenizer, T5ForConditionalGeneration,
                         Trainer, TrainingArguments)
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import zipfile
import os
import time
import logging
import warnings
warnings.filterwarnings('ignore')


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start timing
start_time = time.time()


# Unzip and load the Quora Question Pairs dataset
try:
    with zipfile.ZipFile('/kaggle/input/quora-question-pairs/train.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working/')
    df = pd.read_csv('/kaggle/working/train.csv')
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# Filter for ethical/moral-related questions and sample subset
def is_ethical_question(text):
    ethical_keywords = ['should', 'ethics', 'moral', 'right', 'wrong', 'dilemma', 'duty', 'justice', 'ought', 'ethical']
    text = str(text).lower()
    return any(keyword in text for keyword in ethical_keywords)

# Filter dataset
df['is_ethical'] = df['question1'].apply(is_ethical_question) | df['question2'].apply(is_ethical_question)
ethical_df = df[df['is_ethical']].sample(n=600, random_state=42).copy()
ethical_df['label'] = 1

# Sample non-ethical questions
non_ethical_df = df[~df['is_ethical']].sample(n=600, random_state=42)
non_ethical_df['label'] = 0

# Combine datasets
combined_df = pd.concat([ethical_df[['question1', 'label']].rename(columns={'question1': 'text'}),
                         non_ethical_df[['question1', 'label']].rename(columns={'question1': 'text'})])
logger.info(f"Combined dataset size: {len(combined_df)}")


#Exploratory Data Analysis (EDA)
print("=== Dataset Overview ===")
print(f"Total samples: {len(combined_df)}")
print(f"Missing values:\n{combined_df.isnull().sum()}")
print(f"Columns: {combined_df.columns.tolist()}")

plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=combined_df)
plt.title('Distribution of Ethical vs. Non-Ethical Questions')
plt.xlabel('Label (0: Non-Ethical, 1: Ethical)')
plt.ylabel('Count')
plt.savefig('/kaggle/working/label_distribution.png')
plt.close()

combined_df['text_length'] = combined_df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(data=combined_df, x='text_length', hue='label', bins=50)
plt.title('Distribution of Question Length by Label')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.savefig('/kaggle/working/text_length_distribution.png')
plt.close()

ethical_keywords = ['should', 'ethics', 'moral', 'right', 'wrong', 'dilemma', 'duty', 'justice', 'ought', 'ethical']
keyword_counts = Counter()
for text in ethical_df['question1'].dropna():
    for keyword in ethical_keywords:
        if keyword in str(text).lower():
            keyword_counts[keyword] += 1

plt.figure(figsize=(10, 6))
sns.barplot(x=list(keyword_counts.keys()), y=list(keyword_counts.values()))
plt.title('Frequency of Ethical Keywords in Ethical Questions')
plt.xlabel('Keyword')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('/kaggle/working/keyword_frequency.png')
plt.close()

print("\n=== Sample Ethical Questions ===")
print(ethical_df['question1'].head(5).to_list())
print("\n=== Sample Non-Ethical Questions ===")
print(non_ethical_df['question1'].head(5).to_list())

ethical_text = ' '.join(ethical_df['question1'].dropna().str.lower())
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(ethical_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Ethical Questions')
plt.savefig('/kaggle/working/ethical_wordcloud.png')
plt.close()



# Preprocess data for multiple models
def tokenize_function(texts, tokenizer):
    return tokenizer(texts['text'], padding='max_length', truncation=True, max_length=128)

# Initialize tokenizers
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
mobilebert_tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(combined_df[['text', 'label']].dropna())

# Tokenize for each model
distilbert_dataset = dataset.map(lambda x: tokenize_function(x, distilbert_tokenizer), batched=True)
albert_dataset = dataset.map(lambda x: tokenize_function(x, albert_tokenizer), batched=True)
mobilebert_dataset = dataset.map(lambda x: tokenize_function(x, mobilebert_tokenizer), batched=True)

# Set format for training
for ds in [distilbert_dataset, albert_dataset, mobilebert_dataset]:
    ds = ds.rename_column('label', 'labels')
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split into train and test
distilbert_train_test = distilbert_dataset.train_test_split(test_size=0.2)
albert_train_test = albert_dataset.train_test_split(test_size=0.2)
mobilebert_train_test = mobilebert_dataset.train_test_split(test_size=0.2)


# Fine-tune models
def train_model(model, train_dataset, eval_dataset, output_dir, num_epochs=2):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to='none',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    model_start_time = time.time()
    trainer.train()
    model_end_time = time.time()
    logger.info(f"Training {output_dir} took {model_end_time - model_start_time:.2f} seconds")
    return model

# Initialize and train models
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
albert_model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
mobilebert_model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)

distilbert_model = train_model(distilbert_model, distilbert_train_test['train'], distilbert_train_test['test'], '/kaggle/working/distilbert_results', num_epochs=2)
albert_model = train_model(albert_model, albert_train_test['train'], albert_train_test['test'], '/kaggle/working/albert_results', num_epochs=2)
mobilebert_model = train_model(mobilebert_model, mobilebert_train_test['train'], mobilebert_train_test['test'], '/kaggle/working/mobilebert_results', num_epochs=2)

# Save models
distilbert_model.save_pretrained('/kaggle/working/ethical_dilemma_distilbert')
albert_model.save_pretrained('/kaggle/working/ethical_dilemma_albert')
mobilebert_model.save_pretrained('/kaggle/working/ethical_dilemma_mobilebert')
distilbert_tokenizer.save_pretrained('/kaggle/working/ethical_dilemma_distilbert')
albert_tokenizer.save_pretrained('/kaggle/working/ethical_dilemma_albert')
mobilebert_tokenizer.save_pretrained('/kaggle/working/ethical_dilemma_mobilebert')


# Fine-tune T5 for response generation
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_training_args = TrainingArguments(
    output_dir='/kaggle/working/t5_results',
    num_train_epochs=1,  # Increased to 1 epoch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='/kaggle/working/t5_logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='none',
)

# Prepare T5 dataset with enhanced targets
t5_data = ethical_df[['question1']].rename(columns={'question1': 'text'}).dropna()
t5_data['target'] = t5_data['text'].apply(lambda x: f"From a utilitarian perspective, addressing the dilemma '{x}' involves weighing consequences to maximize overall happiness. Consider the impact on all stakeholders and choose the action that minimizes harm while promoting the greatest good.")
t5_dataset = Dataset.from_pandas(t5_data)

# Tokenize input and target
def tokenize_t5(examples):
    model_inputs = t5_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(examples['target'], padding='max_length', truncation=True, max_length=128)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

t5_dataset = t5_dataset.map(tokenize_t5, batched=True)
t5_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

t5_train_test = t5_dataset.train_test_split(test_size=0.2)
t5_trainer = Trainer(
    model=t5_model,
    args=t5_training_args,
    train_dataset=t5_train_test['train'],
    eval_dataset=t5_train_test['test'],
)
t5_start_time = time.time()
t5_trainer.train()
t5_end_time = time.time()
logger.info(f"Training T5 took {t5_end_time - t5_start_time:.2f} seconds")
t5_model.save_pretrained('/kaggle/working/ethical_dilemma_t5')
t5_tokenizer.save_pretrained('/kaggle/working/ethical_dilemma_t5')


# Step 6: Generate personalized ethical dilemma response
def generate_ethical_response(dilemma, moral_framework, classification_model, classification_tokenizer, t5_model, t5_tokenizer, model_name):
    try:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move model to CPU if GPU memory is low
        device = torch.device('cpu') if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory else torch.device('cuda')
        classification_model.to(device)
        t5_model.to(device)
        
        # Check if the question is ethical
        inputs = classification_tokenizer(dilemma, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = classification_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            is_ethical = torch.argmax(logits, dim=1).item()
            ethical_prob = probs[0][1].item()
        
        logger.info(f"{model_name} classification: is_ethical={is_ethical}, ethical_prob={ethical_prob:.4f}")
        
        if not is_ethical:
            logger.warning(f"{model_name} classified dilemma as non-ethical: {dilemma}")
            return "This does not appear to be an ethical dilemma. Please provide a question involving moral or ethical considerations."
        
        # Generate response with T5
        prompt = f"Provide a utilitarian response to the dilemma '{dilemma}', maximizing happiness and minimizing harm."
        
        t5_inputs = t5_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        t5_inputs = {key: val.to(device) for key, val in t5_inputs.items()}
        
        with torch.no_grad():
            outputs = t5_model.generate(**t5_inputs, max_length=200, min_length=30, num_beams=7, no_repeat_ngram_size=4, early_stopping=True)
        
        response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check for invalid response
        if not response.strip() or response.lower().startswith('provide') or len(response.split()) < 10:
            logger.warning(f"{model_name} generated invalid T5 response: {response}")
            response = f"From a utilitarian perspective, addressing '{dilemma}' suggests resolving it privately to minimize harm and maintain workplace harmony."
        
        logger.info(f"Generated response for {model_name}: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response for {model_name}: {str(e)}")
        return f"Error generating response for {model_name}: {str(e)}"

def get_moral_description(framework):
    descriptions = {
        'utilitarian': 'maximizing overall happiness and minimizing harm for the greatest number of people.',
        'deontological': 'adhering to moral rules and duties, regardless of consequences.',
        'virtue ethics': 'cultivating moral character and virtues to guide actions.'
    }
    return descriptions.get(framework.lower(), 'a balanced consideration of consequences and duties.')

# Example usage
dilemma = "Should I report a coworker for a minor policy violation?"
moral_framework = "utilitarian"

# Write responses to file
response_file = '/kaggle/working/responses.txt'
with open(response_file, 'w') as f:
    f.write(f"Dilemma: {dilemma}\n")
    f.write(f"Moral Framework: {moral_framework}\n")

    # Test with ALBERT
    response = generate_ethical_response(dilemma, moral_framework, albert_model, albert_tokenizer, t5_model, t5_tokenizer, "ALBERT")
    print(f"Response (ALBERT + T5): {response}")
    f.write(f"Response (ALBERT + T5): {response}\n")

    # Test with MobileBERT
    response = generate_ethical_response(dilemma, moral_framework, mobilebert_model, mobilebert_tokenizer, t5_model, t5_tokenizer, "MobileBERT")
    print(f"Response (MobileBERT + T5): {response}")
    f.write(f"Response (MobileBERT + T5): {response}\n")

    # Test with DistilBERT
    response = generate_ethical_response(dilemma, moral_framework, distilbert_model, distilbert_tokenizer, t5_model, t5_tokenizer, "DistilBERT")
    print(f"Response (DistilBERT + T5): {response}")
    f.write(f"Response (DistilBERT + T5): {response}\n")

# Print total runtime
end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")

