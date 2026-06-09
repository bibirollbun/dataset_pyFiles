!pip install py7zr


!pip install transformers torch



import pandas as pd
from zipfile import ZipFile
import py7zr
from collections import Counter
import matplotlib.pyplot as plt




extracting_path = 'C:\\Users\\dell\\Desktop\\competetion\\output' 
folder_path = 'C:\\Users\\dell\\Desktop\\competetion\\input'
main_path = 'C:\\Users\\dell\\Desktop\\competetion'


def get_df_from_zip_folder(folder_path,extracting_path = extracting_path ,isText = False):
    df = pd.DataFrame()
    with ZipFile(folder_path, "r") as zip_ref:
       # Get list of files names in zip
       list_of_files = zip_ref.namelist()
       if isText:
           zip_ref.extractall(path=extracting_path +  '\\' + list_of_files[0],)
           with open(extracting_path + '\\' + list_of_files[0] + '\\' + list_of_files[0], 'r', encoding='utf-8') as file:
            #for kaggle
            #with open(extracting_path + list_of_files[0] + '/' + list_of_files[0], 'r') as file:
                # Read the first line as the header
                header = file.readline().strip().split(',')
                # Read the rest of the file as data, splitting rows by "||"
                data = [line.strip().split('||') for line in file]
            
            # Create the DataFrame
           df = pd.DataFrame(data, columns=header)
           file.close()
       else:
           opened_file = zip_ref.open(list_of_files[0])
           df = pd.read_csv(opened_file)
           opened_file.close()    
    zip_ref.close()
    return df


training_variants_df = get_df_from_zip_folder(folder_path + "\\training_variants.zip")
display(training_variants_df.head())
print("the length of the training_variants_df is : ",len(training_variants_df))



training_text_df = get_df_from_zip_folder(folder_path + "\\training_text.zip" , isText = True)


display(training_text_df.head())
print("the length of the training_text_df is : ",len(training_text_df))


test_variants_df = get_df_from_zip_folder(folder_path + "\\test_variants.zip" )


display(test_variants_df.head())
print("the length of the test_variants_df is : ",len(test_variants_df))


test_text_df = get_df_from_zip_folder(folder_path + "\\test_text.zip",isText = True)


display(test_text_df.head())
print("the length of the training_text_df is : ",len(test_text_df))


def get_data_frame_from_7zipped_folder(folder_path, extraction_path,text = False):
    df = pd.DataFrame()
    file_name = ''
    with py7zr.SevenZipFile(folder_path, "r") as zip_ref:
        list_of_files = zip_ref.namelist()
        file_name = list_of_files[0]
        zip_ref.extractall(path=extraction_path,)
    # Open the file and process the content
    #for kaggle
    # with open(extraction_path + file_name , 'r') as file:
    with open(extraction_path + '\\' + file_name , 'r', encoding='utf-8') as file:
        if text:
            header = file.readline().strip().split(',')
            # Read the rest of the file as data, splitting rows by "||"
            data = [line.strip().split('||') for line in file]
        
            # Create the DataFrame
            df = pd.DataFrame(data, columns=header)
        else:
             # Create the DataFrame
            df = pd.read_csv(file)
    file.close()
    return df
    


 # Directory to extract files into
path = folder_path + "\\stage2_test_variants.csv.7z"
print(path)
test_variants_stage2 = get_data_frame_from_7zipped_folder(path,extracting_path)
display(test_variants_stage2.head())
print("the length of the test_variants_stage2 is : ",len(test_variants_stage2))


path = folder_path + "\\stage2_test_text.csv.7z"
test_texts_stage2 = get_data_frame_from_7zipped_folder(path,extracting_path,text = True)
display(test_texts_stage2.head())
print("the length of the test_texts_stage2 is : ",len(test_texts_stage2))


path = folder_path + "\\stage2_sample_submission.csv.7z"
stage2_sample_submission_df = get_data_frame_from_7zipped_folder(path,extracting_path)
display(stage2_sample_submission_df.head())
print("the length of the sample_submission_df is : ",len(stage2_sample_submission_df))


# training_variants_df
# training_text_df
print("Dataset Overview:")
display(training_variants_df.head())  # This will nicely show the top rows in Jupyter

display(training_variants_df.info())

summary_stats = training_variants_df.describe().transpose()
print(summary_stats)

# Step 2: Identify missing values
print("\nMissing Values:")
missing_values = (training_variants_df == '?').sum()



classes_counts = Counter(training_variants_df['Class'])
print(classes_counts)

plt.bar(classes_counts.keys(),classes_counts.values())
plt.xticks(list(classes_counts.keys()))
plt.show()



genes_counts = Counter(training_variants_df['Gene'])
print(genes_counts)
print(len(genes_counts))

# Variation
variation_counts = Counter(training_variants_df['Variation'])
print(variation_counts)
print(len(variation_counts))
# plt.bar(genes_counts.keys(),genes_counts.values())
# plt.xticks(list(classes_counts.keys()))
# plt.show()


training_text_df


#1) append the text column to the training_df
training_variants_df['Text'] = training_text_df['Text']
#2) delete the id column because it's useless for now
training_variants_df = training_variants_df.drop('ID', axis=1)
#3) swap between the last two columns locations
columns_titles = ["Gene","Variation","Text","Class"]
training_variants_df=training_variants_df.reindex(columns=columns_titles)
#review the new dataframe
training_variants_df


# test_texts_stage2
# test_variants_stage2

#1) append the text column to the training_df
test_variants_stage2['Text'] = test_texts_stage2['Text']
#2) delete the id column because it's useless for now
test_variants_stage2 = test_variants_stage2.drop('ID', axis=1)
#3) swap between the last two columns locations
columns_titles = ["Gene","Variation","Text","Class"]
test_variants_stage2=test_variants_stage2.reindex(columns=columns_titles)
#review the new dataframe
test_variants_stage2


from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
target = "Class"

X = training_variants_df.loc[:, training_variants_df.columns != 'Class']
y = training_variants_df.loc[:, training_variants_df.columns == 'Class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)





from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the text column
def tokenize_data(texts, max_len=256):
    return tokenizer(
        list(texts),
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

# Tokenize training and validation texts
train_tokens = tokenize_data(X_train['Text'])
val_tokens = tokenize_data(X_val['Text'])



import torch

# Save tokenized training data
torch.save({
    'input_ids': train_tokens['input_ids'],
    'attention_mask': train_tokens['attention_mask']
}, main_path + '\\train_tokens.pt')

# Save tokenized validation data
torch.save({
    'input_ids': val_tokens['input_ids'],
    'attention_mask': val_tokens['attention_mask']
}, main_path + '\\val_tokens.pt')



# Load tokenized training data
train_tokens = torch.load(main_path + '\\train_tokens.pt')

# Load tokenized validation data
val_tokens = torch.load(main_path + '\\val_tokens.pt')



from transformers import DistilBertModel
import torch.nn as nn

class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes, variation_dim):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc_variation = nn.Linear(variation_dim, 128)  # Adjust based on variation encoding
        self.fc_combined = nn.Linear(self.distilbert.config.hidden_size + 128, num_classes)

    def forward(self, input_ids, attention_mask, variation_features):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = distilbert_output.last_hidden_state[:, 0, :]  # [CLS] token
        variation_output = self.fc_variation(variation_features)
        combined_output = torch.cat((pooled_output, variation_output), dim=1)
        logits = self.fc_combined(combined_output)
        return logits


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F

# Encode variation column
# Initialize the encoder with handle_unknown='ignore'
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit the encoder on the training data
encoder.fit(X_train['Variation'].values.reshape(-1, 1))

# Transform the training and validation data
variation_train = encoder.transform(X_train['Variation'].values.reshape(-1, 1)).toarray()
variation_val = encoder.transform(X_val['Variation'].values.reshape(-1, 1)).toarray()

# Convert labels to 0-based indexing
train_labels = torch.tensor(y_train.values.flatten() - 1, dtype=torch.long)
val_labels = torch.tensor(y_val.values.flatten() - 1, dtype=torch.long)

# Prepare datasets
train_dataset = TensorDataset(
    train_tokens['input_ids'], 
    train_tokens['attention_mask'],
    torch.tensor(variation_train, dtype=torch.float32),
    train_labels
)

val_dataset = TensorDataset(
    val_tokens['input_ids'], 
    val_tokens['attention_mask'],
    torch.tensor(variation_val, dtype=torch.float32),
    val_labels
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model
model = DistilBERTClassifier(num_classes=9, variation_dim=variation_train.shape[1])
optimizer = Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training and validation loop
def train_and_evaluate(model, train_loader, val_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        print(f"epoch {epoch} started")
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, variation_features, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, variation_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print(f"epoch {epoch} train loss: {train_loss}")

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, variation_features, labels = batch
                outputs = model(input_ids, attention_mask, variation_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Collect predictions and true labels
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(classification_report(all_labels, all_preds, digits=4))

# Example usage
train_and_evaluate(model, train_loader, val_loader, epochs=2, optimizer=optimizer, criterion=criterion)



print(train_dataset)


test_variants_df


# test_texts_stage2
# test_variants_stage2
# test_variants_df


test_variants_df


# Tokenize the test data
test_stage2_tokens = tokenize_data(test_variants_stage2['Text'])


# Save tokenized test data
torch.save({
    'input_ids': test_stage2_tokens['input_ids'],
    'attention_mask': test_stage2_tokens['attention_mask']
}, main_path + '\\test_stage2_tokens.pt')



# Transform the variation column in the test data
variation_test = encoder.transform(test_variants_stage2['Variation'].values.reshape(-1, 1)).toarray()


from torch.utils.data import DataLoader, TensorDataset

# Create a TensorDataset for the test data
test_dataset = TensorDataset(
    test_stage2_tokens['input_ids'], 
    test_stage2_tokens['attention_mask'], 
    torch.tensor(variation_test, dtype=torch.float32)
)

# Create a DataLoader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



import torch.nn.functional as F
import numpy as np

# Generate predictions
model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, variation_features = batch
        outputs = model(input_ids, attention_mask, variation_features)
        probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        predictions.extend(probs.cpu().numpy())



import pandas as pd

# Convert predictions to DataFrame
submission_df = pd.DataFrame(predictions, columns=[f'class{i+1}' for i in range(9)])
submission_df.insert(0, 'ID', test_variants_stage2.index + 1)  # Add ID column starting from 1

# Save to CSV
submission_df.to_csv(main_path +'\\submission.csv', index=False)


