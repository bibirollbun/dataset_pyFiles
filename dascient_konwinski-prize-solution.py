from IPython.display import clear_output
!pip install transformers datasets scikit-learn numpy pandas matplotlib torch openai gitpython
clear_output()


import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import openai
import git
import matplotlib.pyplot as plt
print("\nPackages installed...\n")


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        # Load GitHub issues dataset or GitHub issue data from an external file

# 
def load_issues_data(filepath):
    issues_df = pd.read_csv(filepath)
    issues_df = issues_df.dropna(subset=['issue_title','body'])
    return issues_df

# Example: Loading issues data from a CSV file
issues_data = load_issues_data("/kaggle/input/github-issues/github_issues.csv")
issues_data.head()


!pip install --upgrade openai
!openai migrate
clear_output()

# Load pre-trained tokenizer and model
model_name = "microsoft/deberta-v3-small"  # You can choose a model that works best for GitHub issue categorization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# back off ştudmuffiň
# Alternatively, use a pre-trained GPT model for more advanced text generation (fixes, suggestions)
openai.api_key = "sk-svcacct-dlmiKhH84y1HJnN5RpS1fxvtgOJz13lqYGyXpYXLYzHLP0f-pg_9sXMffbZQyVlARAXI7T3BlbkFJ9jCvSKAEOVRwzR2i8C-v2viNCMvb6A48SJnrvRNGOgYBP2uyQ5gi5bgzfCbzTgDqe6vqAA"  # To use GPT-3 or similar models

# Function to generate suggestions for GitHub issues using GPT-3
def generate_suggestion_for_issue(issue_title, body):
    prompt = f"Issue Title: {issue_title}\nBody: {body}\n\nWhat is the solution or suggestion?"
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can also use other GPT-3 models
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example suggestion for one issue
sample_issue_title = "Fix bug in user authentication"
sample_issue_description = "There is an issue where the login is not working when using Facebook login."
suggestion = generate_suggestion_for_issue(sample_issue_title, body)
print("Suggested Fix:", suggestion)


# Split the data into training and test sets
train_data, test_data = train_test_split(issues_data, test_size=0.2)

# Tokenize the text data for training
def tokenize_data(data, tokenizer):
    return tokenizer(list(data['issue_title'] + " " + data['body']), padding=True, truncation=True, max_length=512, return_tensors="pt")

train_encodings = tokenize_data(train_data, tokenizer)
test_encodings = tokenize_data(test_data, tokenizer)

# Create dataset class
class GitHubIssuesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Train the model
train_dataset = GitHubIssuesDataset(train_encodings, train_data['labels'].values)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

# Fine-tune the model using a training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):  # Number of epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {key: value.to(model.device) for key, value in batch.items()}
        labels = batch['labels'].to(model.device)
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss {loss.item()}")


# Evaluate the model on the test data
def evaluate_model(model, test_data, tokenizer):
    model.eval()
    test_encodings = tokenize_data(test_data, tokenizer)
    test_dataset = GitHubIssuesDataset(test_encodings, test_data['labels'].values)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: value.to(model.device) for key, value in batch.items()}
            labels = batch['labels'].to(model.device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy:.4f}")

# Evaluate the trained model
evaluate_model(model, test_data, tokenizer)


# Generate predictions for the test set
def generate_predictions_for_submission(test_data):
    predictions = []
    for _, row in test_data.iterrows():
        suggestion = generate_suggestion_for_issue(row['issue_title'], row['body'])
        predictions.append(suggestion)
    return predictions

# Generate predictions for the test set
test_predictions = generate_predictions_for_submission(test_data)

# Prepare the Kaggle submission file
submission_df = pd.DataFrame({
    'issue_id': test_data['issue_id'],
    'predicted_suggestion': test_predictions
})

# Save submission file
submission_df.to_csv("submission.csv", index=False)


# enfin/

