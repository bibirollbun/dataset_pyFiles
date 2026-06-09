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


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Define the paths
train_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
train_csv_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'

# Load the ground truth file
df_train_labels = pd.read_csv(train_csv_path)

# Prepare a list to hold the data
training_data = []

# Loop through the ground truth file
for index, row in df_train_labels.iterrows():
    article_id = int(row['id'])
    if article_id <= 9:
      article_id = "0"+ str(article_id)
    else:
      str(article_id)
    real_text_id = row['real_text_id']

    # Construct file paths
    article_dir = os.path.join(train_dir, f'article_00{article_id}')
    path1 = os.path.join(article_dir, 'file_1.txt')
    path2 = os.path.join(article_dir, 'file_2.txt')


    if os.path.exists(article_dir):
        try:
            # Read the content of each file
            with open(path1, 'r', encoding='utf-8') as f:
                text1 = f.read()
            with open(path2, 'r', encoding='utf-8') as f:
                text2 = f.read()

            # Append the data to our list
            training_data.append({
                'id': article_id,
                'text1': text1,
                'text2': text2,
                'label': real_text_id # This will be 1 or 2
            })
        except FileNotFoundError:
            print(f"Files for article {article_id} not found. Skipping.")
    else:
        print(f"Directory for article {article_id} not found. Skipping.")


# Create the final DataFrame
df_train = pd.DataFrame(training_data)

# You can adjust the label to be 0 or 1 for model training
# e.g., if label is 1, target is 0; if label is 2, target is 1
df_train['target'] = df_train['label'].apply(lambda x: 0 if x == 1 else 1)

print(df_train.head())


len(df_train['text1'][0])


from transformers import BertTokenizer
import tensorflow as tf

# Define parameters for BERT
max_length = 512  # Adjust as needed

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to encode the text using BERT tokenizer
def encode_text(tokenizer, text_list, max_length):
    return tokenizer(
        text_list,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

# Encode text from both files
encoded_inputs1 = encode_text(tokenizer, df_train['text1'].tolist(), max_length)
encoded_inputs2 = encode_text(tokenizer, df_train['text2'].tolist(), max_length)

print("Text data preprocessed using BERT tokenizer.")
print("Encoded inputs for text1 shape:", encoded_inputs1['input_ids'].shape)
print("Encoded inputs for text2 shape:", encoded_inputs2['input_ids'].shape)


encoded_inputs1


from transformers import TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model

# Define parameters for BERT
max_length = 512

# Define the BERT model
bert_model = TFBertModel.from_pretrained('bert-large-uncased', from_pt=True)

class FakeRealModel(tf.keras.Model):
    def __init__(self, bert_model, **kwargs):
        super(FakeRealModel, self).__init__(**kwargs)
        self.bert = bert_model
        self.dense1 = Dense(384, activation='relu')  # new change 
        self.dropout = tf.keras.layers.Dropout(rate=0.5)# new change
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2 = inputs

        # Manually create input dictionaries for the BERT model
        bert_inputs1 = {'input_ids': input_ids1, 'attention_mask': attention_mask1, 'token_type_ids': token_type_ids1}
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}


        bert_output1 = self.bert(bert_inputs1)[0][:, 0, :] # Use [CLS] token output
        bert_output2 = self.bert(bert_inputs2)[0][:, 0, :] # Use [CLS] token output

        merged = concatenate([bert_output1, bert_output2])
        dense1 = self.dense1(merged)
        output = self.classifier(dense1)

        return output

# Create an instance of the custom model
model = FakeRealModel(bert_model)

# Define inputs for compiling the model
input_ids1 = Input(shape=(max_length,), dtype=tf.int32, name='input_ids1')
attention_mask1 = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask1')
token_type_ids1 = Input(shape=(max_length,), dtype=tf.int32, name='token_type_ids1')

input_ids2 = Input(shape=(max_length,), dtype=tf.int32, name='input_ids2')
attention_mask2 = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask2')
token_type_ids2 = Input(shape=(max_length,), dtype=tf.int32, name='token_type_ids2')


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy']) # Using Adam with a lower learning rate

model.build(input_shape=[(None, max_length), (None, max_length), (None, max_length), (None, max_length), (None, max_length), (None, max_length)])

model.summary()


import numpy as np

labels = df_train['target'].values
history = model.fit([encoded_inputs1['input_ids'], encoded_inputs1['attention_mask'], encoded_inputs1['token_type_ids'],
                     encoded_inputs2['input_ids'], encoded_inputs2['attention_mask'], encoded_inputs2['token_type_ids']],
                    labels, epochs=150, validation_split=0.2) # Adjust epochs and validation_split as needed



import os

test_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'

# List all directories within the test_dir (each represents an article)
article_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

# Get the number of directories
num_article_dirs = len(article_dirs)

print(f"The number of directories (articles) in the test directory is: {num_article_dirs}")


import os

# Define the paths
test_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'

test_data = []

# Loop through the ground truth file
for index in range(0,num_article_dirs):
    
    article_id = index
    if article_id <= 9:
      article_id = "000"+ str(article_id)
    elif article_id <= 99:
      article_id = "00"+ str(article_id)
    elif article_id <= 999:
      article_id = "0"+ str(article_id)
    else:
      str(article_id)

    article_dir = os.path.join(test_dir, f'article_{article_id}')
    print(article_dir)
    path1 = os.path.join(article_dir, 'file_1.txt')
    path2 = os.path.join(article_dir, 'file_2.txt')


    if os.path.exists(article_dir):
        try:
            # Read the content of each file
            with open(path1, 'r', encoding='utf-8') as f:
                text1 = f.read()
            with open(path2, 'r', encoding='utf-8') as f:
                text2 = f.read()

            # Append the data to our list
            test_data.append({
                'id': article_id,
                'text1': text1,
                'text2': text2
            })
        except FileNotFoundError:
            print(f"Files for article {article_id} not found. Skipping.")
    else:
        print(f"Directory for article {article_id} not found. Skipping.")


# Create the final DataFrame
df_test = pd.DataFrame(test_data)

print("Test dataset loaded successfully!")
display(df_test.head())
print(f"Number of test articles loaded: {len(df_test)}")


# Encode text from both files in the test set
encoded_test_inputs1 = encode_text(tokenizer, df_test['text1'].tolist(), max_length)
encoded_test_inputs2 = encode_text(tokenizer, df_test['text2'].tolist(), max_length)


print(encoded_test_inputs1['input_ids'].shape,encoded_test_inputs2['input_ids'].shape)



test_predictions = model.predict([encoded_test_inputs1['input_ids'], encoded_test_inputs1['attention_mask'], encoded_test_inputs1['token_type_ids'],
                                  encoded_test_inputs2['input_ids'], encoded_test_inputs2['attention_mask'], encoded_test_inputs2['token_type_ids']])

print(test_predictions.shape)


test_predictions



predicted_real_text_ids = []
for prediction in test_predictions:
  
    if prediction >= 0.5:
        predicted_real_text_ids.append(2)
    else:
        predicted_real_text_ids.append(1)

df_test['predicted_real_text_id'] = predicted_real_text_ids

display(df_test.head())



df_submission = df_test[['id', 'predicted_real_text_id']].copy()

df_submission = df_submission.rename(columns={'predicted_real_text_id': 'real_text_id'})

submission_file_path = 'submission.csv'
df_submission.to_csv(submission_file_path, index=False)

print(f"Submission file '{submission_file_path}' created successfully.")
display(df_submission.head())

