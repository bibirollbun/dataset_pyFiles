# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd

#data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#to avoid warnings
import warnings
warnings.filterwarnings('ignore')


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


train = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
test ='/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'
sample_subbmission = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip'
test_labels = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip'


train_data = pd.read_csv(train)
train_data.head(10)


test_values = pd.read_csv(test)
#test_values = test_values['comment_text'].tolist()
test_values


test_labels = pd.read_csv(test_labels)
print(test_labels.shape)
test_labels = test_labels[
    (test_labels["toxic"] != -1) &
    (test_labels["severe_toxic"] != -1) &
    (test_labels["obscene"] != -1) &
    (test_labels["threat"] != -1) &
    (test_labels["insult"] != -1) &
    (test_labels["identity_hate"] != -1)
]

test_labels.shape


#assert len() == len(test_labels), "length do not equal"


test_data = pd.merge(test_values, test_labels, on = "id" )
test_data.shape



test_texts = test_data['comment_text']
test_labels = test_data.iloc[:, 2:]


#data = pd.read_csv(sample_subbmission)
#data.head()


#data = pd.read_csv(test_labels)
#data.head(10)


# Visualizing the class distribution of the 'label' column 
column_labels = train_data.columns.tolist()[2:] 
label_counts = train_data[column_labels].sum().sort_values() 


# Create a black background for the plot 
plt.figure(figsize=(7, 5)) 

# Create a horizontal bar plot using Seaborn 
ax = sns.barplot(x=label_counts.values, 
				y=label_counts.index, palette='viridis') 


# Add labels and title to the plot 
plt.xlabel('Number of Occurrences') 
plt.ylabel('Labels') 
plt.title('Distribution of Label Occurrences') 

# Show the plot 
plt.show() 




train_data[column_labels].sum().sort_values()


# Create subsets based on toxic and clean comments 
train_toxic = train_data[train_data[column_labels].sum(axis=1) > 0]
train_clean = train_data[train_data[column_labels].sum(axis=1) == 0]

# Number of toxic and clean comments
num_toxic = len(train_toxic)
num_clean = len(train_clean)

# Create DataFrame for visualisation
plot_data = pd.DataFrame(
    {'Category': ['Toxic', 'Clean'], 'Count': [num_toxic, num_clean]})

plt.figure(figsize = (7,5))

# Horizontal BAR
ax = sns.barplot(x = 'Count', y = 'Category', data = plot_data, palette = 'viridis')

plt.xlabel('Number of Commenzts')
plt.ylabel('Category')
plt.title('Distribution of Toxic and Clean comments')

ax.tick_params()
plt.show()



print(train_toxic.shape) 
print(train_clean.shape)


from sklearn.utils import resample
import pandas as pd

# Maximum number of examples for any class
max_class_count = train_data[column_labels].sum(axis=0).max()

augmented_dataframes = []

# Step 1. Single class augmentation
for label in ["threat", "identity_hate", "severe_toxic", "insult", "obscene", "toxic"]:
    # Select rows where only the given class is 1
    single_class_data = train_data[(train_data[label] == 1) & (train_data[column_labels].sum(axis=1) == 1)]
    if single_class_data.empty:
        print(f"No single-class data available for label: {label}")
        single_class_data = train_data[train_data[label] == 1]
    oversampled_class = resample(single_class_data, replace=True, n_samples=max_class_count, random_state=42)
    augmented_dataframes.append(oversampled_class)

# Step 2: Adding mixed classes
# Select lines where text belongs to more than one class
train_toxic_mixed_classes = train_data[train_data[column_labels].sum(axis=1) > 1]
augmented_dataframes.append(train_toxic_mixed_classes)

# Step 3. Merge all DataFrames
augmented_dataframes = pd.concat(augmented_dataframes, ignore_index=True)

# Step 4. Check the result
print("Shape of augmented DataFrame:", augmented_dataframes.shape)
print("Class distribution after augmentation:")
print(augmented_dataframes[column_labels].sum())



# Randomly sample 16225 clean comments 
train_clean_sampled = train_clean.sample(n=101629, random_state=42)

# Combine the toxic and the sampled clean comments
dataframe = pd.concat([augmented_dataframes, train_clean_sampled], axis = 0)

#Shuffle the data to avoid any order bias during training
dataframe = dataframe.sample(frac=1, random_state = 42)



dataframe.shape


dataframe.head()


# split training data into training_texts, training_labels
train_texts = dataframe.iloc[:, 1] 
train_labels = dataframe.iloc[:, 2:]


print(test_data.isnull().sum())  # missing data in each collumn



print(test_data.iloc[:, 1].shape)  # Форма текстов
print(test_data.iloc[:, 2:].shape)  # Форма меток



# Validation set
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts,  # Тексты
    train_labels,  # Метки
    test_size=0.25,
    random_state=42
)

print(train_texts.shape) # Number of texts in the training sample
print(train_labels.shape) # Labels for the training sample
print(val_texts.shape) # Number of texts in validation sample
print(val_labels.shape) # Labels for validation sample



def tokenize_and_encode(tokenizer, comments, labels, max_length=128): 
    # Initialize empty lists to store tokenized inputs and attention masks 
    input_ids = [] 
    attention_masks = [] 

    # Iterate through each comment in the 'comments' list 
    for comment in comments: 
        # Tokenize and encode the comment using the BERT tokenizer 
        encoded_dict = tokenizer.encode_plus( 
            comment, 

            # Add special tokens like [CLS] and [SEP] 
            add_special_tokens=True, 

            # Truncate or pad the comment to 'max_length' 
            max_length=max_length, 

            truncation=True,

            # Pad the comment to 'max_length' with zeros if needed 
            pad_to_max_length=True, 

            # Return attention mask to mask padded tokens 
            return_attention_mask=True, 

            # Return PyTorch tensors 
            return_tensors='pt'
        ) 

        # Append the tokenized input and attention mask to their respective lists 
        input_ids.append(encoded_dict['input_ids']) 
        attention_masks.append(encoded_dict['attention_mask']) 

    # Concatenate the tokenized inputs and attention masks into tensors 
    input_ids = torch.cat(input_ids, dim=0) 
    attention_masks = torch.cat(attention_masks, dim=0) 

    # Convert the labels to a PyTorch tensor with the data type float32
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32) 
        
    return input_ids, attention_masks, labels


   


# https://huggingface.co/google-bert/bert-base-uncased
# Token initialisation bert-base-uncased

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True) 


# Model initialisation
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6) 


# Model initialisation roberta-base
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#model_roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=6)


# Model initialisation distilbert-base-uncased
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#model_distilbert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)


# Define the device (MPS for macOS or CPU)
# device = torch.device("cuda") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Move model to the device
model = model.to(device) 



print(type(train_labels))
print(type(test_labels))
#val_labels = val_labels.to_numpy()
print(type(val_labels))


# Tokenize and Encode the comments and labels for the training set 
input_ids, attention_masks, labels_tr = tokenize_and_encode( 
	tokenizer, 
	train_texts, 
	train_labels.values 
) 
print('Training Comments :',train_texts.shape) 
print('Input Ids		 :',input_ids.shape) 
print('Attention Mask :',attention_masks.shape) 
print('Labels		 :',len(labels_tr))



# Tokenize and Encode the comments and labels for the test set 
test_input_ids, test_attention_masks, labels_test = tokenize_and_encode( 
	tokenizer, 
	test_texts,
    test_labels.values
) 
print()
print('Test Comments :',test_texts.shape) 
print('Input Ids		 :',test_input_ids.shape) 
print('Attention Mask :',test_attention_masks.shape) 
print('Labels		 :',len(labels_test))



# Tokenize and Encode the comments and labels for the validation set 
val_input_ids, val_attention_masks, labels_val = tokenize_and_encode( 
	tokenizer, 
	val_texts, 
	val_labels.values
)
print()
print('Validation Comments :',val_texts.shape) 
print('Input Ids		 :',val_input_ids.shape) 
print('Attention Mask :',val_attention_masks.shape) 
print('Labels		 :', len(labels_val))


# Creating DataLoader for the balanced dataset 
batch_size = 64

train_dataset = TensorDataset(input_ids, attention_masks, labels_tr) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

# testing set 
test_dataset = TensorDataset(test_input_ids, test_attention_masks, labels_test) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

# validation set 
val_dataset = TensorDataset(val_input_ids, val_attention_masks, labels_val) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 



print('Batch Size :',train_loader.batch_size) 
Batch =next(iter(train_loader)) 
print('Each Input ids shape :',Batch[0].shape) 
print('Input ids :\n',Batch[0][0]) 
print('Corresponding Decoded text:\n',tokenizer.decode(Batch[0][0])) 
print('Corresponding Attention Mask :\n',Batch[1][0]) 
print('Corresponding Label:',Batch[2][0])



# Optimizer setup 
optimizer = AdamW(model.parameters(), lr=1.11e-5)



class_counts = train_data[column_labels].sum(axis=0)  # Сумма для каждого класса
class_counts


import time
from torch.nn import BCEWithLogitsLoss

# Function to Train the Model
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, train_data):
    # Start time
    start_time = time.time()

    # Calculate class weights based on the training data
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.max()
    class_weights = class_weights.clip(lower=0.5)
    # Set the minimum weight
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)

    # Initialize weighted loss function
    loss_fn = BCEWithLogitsLoss(pos_weight=class_weights_tensor)

    # Loop through the specified number of epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss with class weights
            loss = loss_fn(logits, labels.float())  # Convert labels to float
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute validation loss
                loss = loss_fn(logits, labels.float())
                val_loss += loss.item()

        # Print the average loss for the current epoch
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

    # End time
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

# Call the function to train the model
train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5, train_data=train_data)



# Dirrectory for the model
model_save_path = "./saved_BERTmodel"

# Save the model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)



# Load the tokenizer and model from the saved directory 
model_name = model_save_path
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name) 
Bert_Model = BertForSequenceClassification.from_pretrained( 
	model_name).to(device) 



def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=device): 
	user_input = [input_text] 

	user_encodings = tokenizer( 
		user_input, truncation=True, padding=True, return_tensors="pt") 

	user_dataset = TensorDataset( 
		user_encodings['input_ids'], user_encodings['attention_mask']) 

	user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False) 

	model.eval() 
	with torch.no_grad(): 
		for batch in user_loader: 
			input_ids, attention_mask = [t.to(device) for t in batch] 
			outputs = model(input_ids, attention_mask=attention_mask) 
			logits = outputs.logits 
			predictions = torch.sigmoid(logits) 

	predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int) 
	labels_list = ['toxic', 'severe_toxic', 'obscene', 
				'threat', 'insult', 'identity_hate'] 
	result = dict(zip(labels_list, predicted_labels[0])) 
	return result 


text = 'Are you insane!'
#predict_user_input(input_text=text) 
print(predict_user_input(input_text=text) )

#predict_user_input(input_text='How are you?') 
print(predict_user_input(input_text='How are you?') )

text = "Such an Idiot person"
predict_user_input(model=Bert_Model, 
				tokenizer=Bert_Tokenizer, 
				input_text=text, 
				device=device) 



# Model evaluation on the validation data
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, multilabel_confusion_matrix, ConfusionMatrixDisplay

columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
model = Bert_Model  # Ensure this is defined elsewhere

def evaluate_model(model, loader, device, save_conf_matrix=False, conf_matrix_path="./saved_BERTmodel/confusion_matrix.csv"): 
    model.eval()  # Set the model to evaluation mode 

    true_labels = [] 
    predicted_probs = [] 

    with torch.no_grad(): 
        for batch in loader: 
            input_ids, attention_mask, labels = [t.to(device) for t in batch] 

            # Get model's predictions 
            outputs = model(input_ids, attention_mask=attention_mask) 
            # Use sigmoid for multilabel classification 
            predicted_probs_batch = torch.sigmoid(outputs.logits) 
            predicted_probs.append(predicted_probs_batch.cpu().numpy()) 

            true_labels_batch = labels.cpu().numpy() 
            true_labels.append(true_labels_batch) 

    # Combine predictions and labels for evaluation 
    true_labels = np.concatenate(true_labels, axis=0) 
    predicted_probs = np.concatenate(predicted_probs, axis=0) 
    predicted_labels = (predicted_probs > 0.5).astype(int)  # Apply threshold for binary classification 

    # Calculate evaluation metrics 
    accuracy = accuracy_score(true_labels, predicted_labels) 
    precision = precision_score(true_labels, predicted_labels, average='micro') 
    recall = recall_score(true_labels, predicted_labels, average='micro')
    conf_matrix = multilabel_confusion_matrix(true_labels, predicted_labels) 

    # Save confusion matrix to a CSV file if save_conf_matrix is True
    if save_conf_matrix:
        flattened_data = []
        for i, matrix in enumerate(conf_matrix):
            TN, FP = matrix[0]
            FN, TP = matrix[1]
            flattened_data.append({
                'Class': columns[i],
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TP': TP
            })

        conf_matrix_df = pd.DataFrame(flattened_data)
        conf_matrix_df.to_csv(conf_matrix_path, index=False)

    # Print the evaluation metrics 
    print(f'Accuracy: {accuracy:.4f}') 
    print(f'Precision: {precision:.4f}') 
    print(f'Recall: {recall:.4f}')
    print(f'Confusion Matrices for Each Class:')
    for i, matrix in enumerate(conf_matrix):
        print(f"Class {columns[i]}:\n{matrix}")

    # Display confusion matrices as plots
    for i, matrix in enumerate(conf_matrix):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix, 
            display_labels=[f'Not Class {columns[i]}', f'Class {columns[i]}']
        )
        disp.plot(cmap='Blues')


# Evaluate on val_loader and save the confusion matrix
evaluate_model(model, val_loader, device, save_conf_matrix=True)


# Evaluate on test_loader without saving the confusion matrix
evaluate_model(model, test_loader, device, save_conf_matrix=False)


#save data for Oleksii 



#val_dataset(TensorDataset)
val_data = val_dataset.tensors # tuple from tensors
val_input_ids, val_attantion_mask, labels_val = map(lambda x: x.numpy(), val_data)
val_df = pd.DataFrame({
    "input_ids": val_input_ids.tolist(),
    "attantion_mask": val_attantion_mask.tolist(),
    "labels": labels_val.tolist()
})

val_df.to_csv("./saved_BERTmodel/data/val_dataset.csv", index = False)


val_labels[val_labels.columns].sum()


type(train_texts)


# Convert texts to DataFrame
train_texts_df = pd.DataFrame({"texts": train_texts})
val_texts_df = pd.DataFrame({"texts": val_texts})

# Check for tag type
if not isinstance(train_labels, pd.DataFrame):
    raise ValueError("train_labels is not a DataFrame. Please ensure correct format.")

if not isinstance(val_labels, pd.DataFrame):
    raise ValueError("val_labels is not a DataFrame. Please ensure correct format.")

# Checking the sums before merging
print("Sum of train_labels before concat:\n", train_labels.sum())
print("Sum of val_labels before concat:\n", val_labels.sum())

# Combining texts and labels
train_df = pd.concat([train_texts_df, train_labels], axis=1)
val_df = pd.concat([val_texts_df, val_labels], axis=1)

# Checking the sums after combining
print("Sum of train_df labels:\n", train_df[train_labels.columns].sum())
print("Sum of val_df labels:\n", val_df[val_labels.columns].sum())

# Saving to CSV
train_df.to_csv("./saved_BERTmodel/data/train_data.csv", index=False)
val_df.to_csv("./saved_BERTmodel/data/validation_data.csv", index=False)


text = "You rock"
predict_user_input(model=Bert_Model, 
				tokenizer=Bert_Tokenizer, 
				input_text=text, 
				device=device) 

