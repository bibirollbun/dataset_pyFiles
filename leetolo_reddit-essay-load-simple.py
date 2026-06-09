from datasets import load_dataset
# WARNING: the full dataset is 250GB (compressed) and over 1TB (uncompressed)
# Stream the dataset instead of downloading it
dataset = load_dataset("jonathanli/human-essays-reddit", streaming=True)

# You can then iterate through the data as needed
for example in dataset["train"].take(1000):  # Get first 1000 examples
    print("keys", example.keys())
    print("example", example)
    
    break


#Adding tokenizer because the Kaggle reject essay submission of token size > 199
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
# Model configuration
model_name = '/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


from datasets import load_dataset
# WARNING: the full dataset is 250GB (compressed) and over 1TB (uncompressed)
# Stream the dataset instead of downloading it
dataset = load_dataset("jonathanli/human-essays-reddit", streaming=True)


train = []
from tqdm import tqdm
# You can then iterate through the data as needed
for example in tqdm(dataset["train"].take(10000)):  # Get first 50000 examples
    curr_essay = f"Title: {example['title']}\nEssay: {example['top_comment']}"
    # reject essay too long
    if len(tokenizer(curr_essay).input_ids) < 1000:
        
        # reject if score or top_comment_score too low
        if example['score'] >= 500 and example['top_comment_score'] >= 500:
            train.append(curr_essay)


print(f"{len(train)} essay loaded")


import matplotlib.pyplot as plt
from datasets import load_dataset

# Load the dataset in streaming mode
dataset = load_dataset("jonathanli/human-essays-reddit", streaming=True)

# Initialize lists to store scores
scores = []
top_comment_scores = []

# Iterate through the dataset and collect scores
for example in dataset["train"].take(1000):  # First 1000 examples
    if "score" in example and "top_comment_score" in example:
        scores.append(example["score"])
        top_comment_scores.append(example["top_comment_score"])

# Create subplots for score distributions
plt.figure(figsize=(12, 6))

# Plot score distribution
plt.subplot(1, 2, 1)
plt.hist(scores, bins=30, color='blue', alpha=0.7, label='Score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Post Scores')
plt.legend()

# Plot top_comment_score distribution
plt.subplot(1, 2, 2)
plt.hist(top_comment_scores, bins=30, color='green', alpha=0.7, label='Top Comment Score')
plt.xlabel('Top Comment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Top Comment Scores')
plt.legend()

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Load the dataset in streaming mode
dataset = load_dataset("jonathanli/human-essays-reddit", streaming=True)
scores = []

# Collect scores
for example in dataset["train"].take(10000):
    if "top_comment_score" in example:
        scores.append(example["top_comment_score"])

scores = np.array(scores)

# Create multiple visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Regular histogram excluding very low scores
ax1.hist(scores[scores > 100], bins=50, color='blue', alpha=0.7)
ax1.set_title('top_comment_score Distribution (>100)')
ax1.set_xlabel('top_comment_score')
ax1.set_ylabel('Frequency')

# 2. Box plot
ax2.boxplot(scores)
ax2.set_title('Box Plot of top_comment_score')
ax2.set_ylabel('top_comment_score')

# 3. Histogram focusing on middle range
middle_scores = scores[(scores > np.percentile(scores, 25)) & (scores < np.percentile(scores, 75))]
ax3.hist(middle_scores, bins=50, color='green', alpha=0.7)
ax3.set_title('Middle 50% of top_comment_score')
ax3.set_xlabel('Score')
ax3.set_ylabel('top_comment_score')

# 4. Cumulative distribution
sorted_scores = np.sort(scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax4.plot(sorted_scores, cumulative, 'r-')
ax4.set_title('Cumulative Distribution')
ax4.set_xlabel('top_comment_score')
ax4.set_ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()





