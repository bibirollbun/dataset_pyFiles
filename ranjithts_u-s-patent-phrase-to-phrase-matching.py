# Quick look at the dataset folder
!ls /kaggle//input/us-patent-phrase-to-phrase-matching/


import pandas as pd
df=pd.read_csv('/kaggle//input/us-patent-phrase-to-phrase-matching/train.csv')

#Since the dataset is large,Pandas will display a truncated version, showing the first few and last few rows (by default, the first 5 and the last 5 rows) and columns.
df


df.describe(include='object')


df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

#We can refer to a column (also known as a series) either using regular python "dotted" notation, or access it like a dictionary. To get the first few rows, use head():
df.input.head()


from datasets import Dataset,DatasetDict

ds=Dataset.from_pandas(df)
ds


model_deberta='microsoft/deberta-v3-small'

from transformers import AutoModelForSequenceClassification,AutoTokenizer

tokens=AutoTokenizer.from_pretrained(model_deberta,clean_up_tokenization_spaces=False,use_fast=False)

#Note: Adding use_fast=False will avoid using the fast tokenizer and prefer to stick with the SentencePiece tokenizer (which handles these cases better)
#If you’re fine with the fast tokenizer (and the potential unknown tokens), you can skip setting this value. 


# Below is an example of how the tokenizer splits a text into "tokens" (which are like words, but can be sub-word pieces as well)

tokens.tokenize("In recent years, deep learning has made significant strides in the field of natural language processing (NLP).")

#Uncommon words will be split into pieces and the start of the new word is represented by _


# Lets write a simple function which will tokenize our inputs

def tokenize_func(x):
    is_train = 'score' in x  # Check if 'score' exists (use it as a flag for training)
    
    # Tokenize the input text
    tokenized = tokens(x["input"], truncation=True, padding="max_length", max_length=512)
    
    if is_train:
        # Add 'score' as 'labels' for training data
        tokenized["labels"] = [float(label) for label in x["score"]]  # Convert 'score' to float
    
    return tokenized

#To run this quickly in parallel on every row in our dataset, we use function
#The primary purpose of map() is to speed up processing functions. It allows you to apply a processing function to each example in a dataset, independently or in batches.

tokenized_ds=ds.map(tokenize_func,batched=True)



# So lets look at the resulting dataset. Since the padding has a max lenght of 512, it will display lot of 0's in the input_id's & attention_mask field.
# Hence i am only going to display data by filtering out padding (zeros) from input_ids
filtered_input_ids = [id for id in tokenized_ds[0]['input_ids'] if id != 0]

# Print the filtered input_ids
print(filtered_input_ids)

# As seen below, the mapping function added a new item to our dataset called input_ids.



#So, what are those IDs and where do they come from? 
#The secret is that there's a list called vocab in the tokenizer which contains a unique integer for every possible token string. 
#We can look them up like this, for instance to find the token for the word "pollution"

print(tokens.vocab['▁pollution'])



dds = tokenized_ds.train_test_split(0.25,seed=42) #seed value is set to ensure reproducibility. Every time you split the data, you get the same split
dds


eval_df=pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
eval_df.describe()


from datasets import Dataset
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds=Dataset.from_pandas(eval_df)
eval_ds=eval_ds.map(tokenize_func,batched=True)




#Transformers expects metrics to be returned as a dict, since that way the trainer knows what label to use, so let's create a function to do that
import numpy as np

def corr(x, y):
    return np.corrcoef(x, y)[0, 1]  # Calculate and return Pearson correlation
    
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}



#Training our model
from transformers import TrainingArguments,Trainer

#We pick a batch size that fits our GPU, and small number of epochs so we can run experiments quickly
batch_size=32
epochs=2
lr = 8e-5

#Transformers uses the TrainingArguments class to set up arguments.
args=TrainingArguments(
    'outputs',
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    num_train_epochs=epochs,weight_decay=0.01,
    report_to='none',
    gradient_accumulation_steps=4
)


#Lets create our model

model=AutoModelForSequenceClassification.from_pretrained(model_deberta,num_labels=1)

trainer=Trainer(model=model,args=args,train_dataset=dds['train'],eval_dataset=dds['test'],tokenizer=tokens,compute_metrics=corr_d)


#Let's train our model!
import time

start_time = time.time()

# Train the model
trainer.train()

end_time = time.time()

# Calculate the time taken
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time/60:.2f} minutes")


preds=trainer.predict(eval_ds).predictions.astype(float)
preds

# Look out - some of our predictions are <0, or >1! This once again shows the importance to actually look at your data. 
# Let's fix those out-of-bounds predictions:

preds =np.clip(preds,0,1)
preds


#Exporting the results

import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)

