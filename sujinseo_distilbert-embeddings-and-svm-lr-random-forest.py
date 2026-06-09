import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel, DistilBertTokenizer, BertTokenizerFast, DistilBertModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import sys
import tensorflow as tf
import tensorflow_hub as hub

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_colwidth', None)
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# !pip install --upgrade datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data
train_data = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
test_data = "/kaggle/input/jigsaw-agile-community-rules/test.csv"

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)
print("Training set shape = {}\nTest set shape = {}\n".format(df_train.shape, df_test.shape))
display(df_train.tail(2))




# Check rules and unique values in each columns
print(df_train.rule.unique())
print(df_test.rule.unique())
example_cols = [x for x in df_train.columns if 'example' in x]
for c in example_cols:
    print(c)
    print(len(df_train[c].unique()))
    print(len(df_train[df_train[c].isnull()]))


# Approximately similar number of violated examples and non-violated ones.
print(df_train.rule_violation.value_counts())
class_counts = df_train["rule_violation"].value_counts().sort_index()
# Visualize the counts
plt.figure()
class_counts.plot.bar()
plt.title("Number of data per violation")
plt.ylabel("Count")
plt.show()
# Group by 'rule' for violated examples
print(df_train.groupby('rule')['rule_violation'].value_counts())
print("Number of unique subreddits: ".format(len(df_train.subreddit.unique())))
subreddit_category = df_train[df_train['rule_violation']==1]['subreddit'].value_counts()
plt.figure()
subreddit_category.plot(kind='bar', figsize=(10, 2))
plt.title("Violation dataset: by subreddit")
plt.show()


print(df_test.columns)


# Use positive and negative examples as training dataset
# There are two example columns per category (positive, negative), 
# so we will concatenate each positive and negative examples.
df_train_pos1 = df_train[['rule', 'positive_example_1']]
df_train_pos2 = df_train[['rule', 'positive_example_2']]
df_train_pos1 = df_train_pos1.rename(columns={'positive_example_1': 'body'})
df_train_pos2 = df_train_pos1.rename(columns={'positive_example_2': 'body'})
df_train_neg1 = df_train[['rule', 'negative_example_1']]
df_train_neg2 = df_train[['rule', 'negative_example_2']]
df_train_neg1 = df_train_neg1.rename(columns={'negative_example_1': 'body'})
df_train_neg2 = df_train_neg2.rename(columns={'negative_example_2': 'body'})
df_train_pos = pd.concat([df_train_pos1, df_train_pos2])
df_train_pos['rule_violation'] = 1
df_train_neg = pd.concat([df_train_neg1, df_train_neg2])
df_train_neg['rule_violation'] = 0

df_train_all = pd.concat([df_train_pos, df_train_neg])

# Remove duplicates
df_train_all_clean = df_train_all.drop_duplicates(subset=['rule','body'])
print(len(df_train_all), len(df_train_all_clean))

# Use body and their violation information in sample
df_train_body = df_train[['rule', 'body', 'rule_violation']]

# Final training dataset
df_train_final = pd.concat([df_train_all_clean, df_train_body])
print(df_train_final.shape)

# Create violation_per_rule column to use in stratified sampling
df_train_final['violation_per_rule'] = df_train_final['rule_violation'].astype(str) \
    + '_' + df_train_final['rule']


SEED = 1234
# train, test = train_test_split(
#     df_train_final,
#     test_size=0.2,
#     random_state=SEED,
#     stratify=df_train_final['violation_per_rule']
# )
# print(train.shape, test.shape)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train_final['body'].tolist(),
    df_train_final['rule_violation'].tolist(),
    test_size=0.2,
    random_state=SEED,
    stratify=df_train_final['violation_per_rule']
)
print(len(train_texts), len(val_texts))
print(len(set(train_labels)))

print(train_labels[:2])



# Tokenized input
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize_input(df, tokenizer, max_length=512):
    encoding = tokenizer(
        df['body'], 
        truncation=True, 
        padding='max_length',
        max_length=max_length
    )
    encoding['labels'] = df['rule_violation']
    return encoding


def tokenize_list_input(list_texts, tokenizer, max_length=512):
    encoding = tokenizer(
        list_texts, 
        truncation=True, 
        padding='max_length',
        max_length=max_length
    )

    return encoding

def get_embeddings(texts, tokenizer, model, max_len):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens
            max_length=max_len,
            padding='max_length',  # Pad to `max_length`
            truncation=True,  # Truncate to `max_length`
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Move to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # input_ids = input_ids.to(device)
    # attention_masks = attention_masks.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        # We'll use the embedding of the '[CLS]' token as the sentence representation.
        embeddings = outputs.last_hidden_state[:, 0, :].numpy() 
    return embeddings


# encoding_train = tokenize_list_input(train_texts, tokenizer)
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
MAX_LEN = 128
X_train_embeddings = get_embeddings(list(train_texts), tokenizer, model, MAX_LEN)


X_val_embeddings = get_embeddings(list(val_texts), tokenizer, model, MAX_LEN)


# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# encoder.fit(train_labels)
# train_labels_enc = encoder.transform(categorical_labels)
# print(train_labels_enc)

label_mapping = {label: i for i, label in enumerate(np.unique(train_labels))} #
y_train_numerical = np.array([label_mapping[label] for label in train_labels])
y_val_numerical = np.array([label_mapping[label] for label in val_labels])



LRclassifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
LRclassifier.fit(X_train_embeddings, y_train_numerical)


y_pred_LR = LRclassifier.predict(X_val_embeddings)
y_pred_proba_LR = LRclassifier.predict_proba(X_val_embeddings)[:, 1]



from sklearn.svm import SVC
svm_classifier = SVC(kernel='rbf', random_state=124)
svm_classifier.fit(X_train_embeddings, y_train_numerical)
y_pred_svm = svm_classifier.predict(X_val_embeddings)


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_embeddings, y_train_numerical)
y_pred_rf = rf_classifier.predict(X_val_embeddings)


def generate_eval(y_val_numerical, y_pred):
    accuracy = accuracy_score(y_val_numerical, y_pred)
    precision = precision_score(y_val_numerical, y_pred)
    recall = recall_score(y_val_numerical, y_pred)
    f1 = f1_score(y_val_numerical, y_pred)
    roc_auc = roc_auc_score(y_val_numerical, y_pred)
    conf_matrix = confusion_matrix(y_val_numerical, y_pred)
    # class_report = classification_report(y_val_numerical, y_pred)
    print(f"Accruacy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC score: {roc_auc:.4f}")
    print(conf_matrix)
    # print(class_report)


# Logistic Regression
print("\n# Logistic regression")
generate_eval(y_val_numerical, y_pred_LR)
# SVM
print("\n# Support vector machine")
generate_eval(y_val_numerical, y_pred_svm)
# random forest
print("\n# Random forest")
generate_eval(y_val_numerical, y_pred_rf)





test_embeddings = get_embeddings(list(df_test['body']), tokenizer, model, MAX_LEN)


test_pred_rf = rf_classifier.predict(test_embeddings)


submission = pd.DataFrame({
    'row_id': df_test['row_id'],
    'rule_violation': test_pred_rf
})
submission.to_csv('submission.csv', index=False)
submission.head(10)




