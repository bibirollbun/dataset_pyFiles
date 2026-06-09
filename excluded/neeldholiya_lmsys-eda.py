import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nTrain Columns:", train.columns.tolist())
print("\nTest Columns:", test.columns.tolist())

print("\n" + "="*80)
print("TRAIN DATA SAMPLE")
print("="*80)
print(train.head())

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(train.isnull().sum())

print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(train.dtypes)

print("\n" + "="*80)
print("TARGET DISTRIBUTION")
print("="*80)
target_counts = pd.DataFrame({
    'winner_model_a': [train['winner_model_a'].sum()],
    'winner_model_b': [train['winner_model_b'].sum()],
    'winner_tie': [train['winner_tie'].sum()]
})
print(target_counts)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

colors = ['#2ecc71', '#e74c3c', '#3498db']
target_counts.T.plot(kind='bar', ax=axes[0], color=colors, legend=False)
axes[0].set_title('Target Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Winner Category')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Model A', 'Model B', 'Tie'], rotation=0)

target_pcts = target_counts.T / len(train) * 100
target_pcts.plot(kind='bar', ax=axes[1], color=colors, legend=False)
axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Winner Category')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_xticklabels(['Model A', 'Model B', 'Tie'], rotation=0)

plt.tight_layout()
plt.show()

train['prompt_length'] = train['prompt'].str.len()
train['response_a_length'] = train['response_a'].str.len()
train['response_b_length'] = train['response_b'].str.len()
train['prompt_word_count'] = train['prompt'].str.split().str.len()
train['response_a_word_count'] = train['response_a'].str.split().str.len()
train['response_b_word_count'] = train['response_b'].str.split().str.len()

print("\n" + "="*80)
print("TEXT STATISTICS")
print("="*80)
print(train[['prompt_length', 'response_a_length', 'response_b_length', 
             'prompt_word_count', 'response_a_word_count', 'response_b_word_count']].describe())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].hist(train['prompt_length'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Prompt Length Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Character Count')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(train['response_a_length'], bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Response A Length Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Character Count')
axes[0, 1].set_ylabel('Frequency')

axes[0, 2].hist(train['response_b_length'], bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Response B Length Distribution', fontweight='bold')
axes[0, 2].set_xlabel('Character Count')
axes[0, 2].set_ylabel('Frequency')

axes[1, 0].hist(train['prompt_word_count'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Prompt Word Count Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(train['response_a_word_count'], bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Response A Word Count Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Word Count')
axes[1, 1].set_ylabel('Frequency')

axes[1, 2].hist(train['response_b_word_count'], bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Response B Word Count Distribution', fontweight='bold')
axes[1, 2].set_xlabel('Word Count')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("MODEL DISTRIBUTION")
print("="*80)
print("\nModel A Distribution:")
print(train['model_a'].value_counts().head(10))
print("\nModel B Distribution:")
print(train['model_b'].value_counts().head(10))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

top_model_a = train['model_a'].value_counts().head(15)
axes[0].barh(range(len(top_model_a)), top_model_a.values, color='#2ecc71')
axes[0].set_yticks(range(len(top_model_a)))
axes[0].set_yticklabels(top_model_a.index)
axes[0].set_title('Top 15 Model A Distribution', fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].invert_yaxis()

top_model_b = train['model_b'].value_counts().head(15)
axes[1].barh(range(len(top_model_b)), top_model_b.values, color='#e74c3c')
axes[1].set_yticks(range(len(top_model_b)))
axes[1].set_yticklabels(top_model_b.index)
axes[1].set_title('Top 15 Model B Distribution', fontweight='bold')
axes[1].set_xlabel('Count')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

train['winner'] = train.apply(lambda row: 'model_a' if row['winner_model_a'] == 1 
                               else ('model_b' if row['winner_model_b'] == 1 else 'tie'), axis=1)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, winner_type in enumerate(['model_a', 'model_b', 'tie']):
    winner_data = train[train['winner'] == winner_type]
    if winner_type == 'model_a':
        color = '#2ecc71'
        title = 'Response Length for Model A Wins'
    elif winner_type == 'model_b':
        color = '#e74c3c'
        title = 'Response Length for Model B Wins'
    else:
        color = '#3498db'
        title = 'Response Length for Ties'
    
    axes[idx].hist(winner_data['response_a_length'], bins=30, alpha=0.5, 
                   label='Response A', color='#2ecc71', edgecolor='black')
    axes[idx].hist(winner_data['response_b_length'], bins=30, alpha=0.5, 
                   label='Response B', color='#e74c3c', edgecolor='black')
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].set_xlabel('Character Count')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.show()

train['length_diff'] = train['response_a_length'] - train['response_b_length']
train['word_count_diff'] = train['response_a_word_count'] - train['response_b_word_count']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(train[train['winner'] == 'model_a']['length_diff'], bins=50, 
             alpha=0.7, label='Model A Wins', color='#2ecc71', edgecolor='black')
axes[0].hist(train[train['winner'] == 'model_b']['length_diff'], bins=50, 
             alpha=0.7, label='Model B Wins', color='#e74c3c', edgecolor='black')
axes[0].hist(train[train['winner'] == 'tie']['length_diff'], bins=50, 
             alpha=0.7, label='Tie', color='#3498db', edgecolor='black')
axes[0].set_title('Length Difference Distribution by Winner', fontweight='bold')
axes[0].set_xlabel('Response A Length - Response B Length')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)

axes[1].hist(train[train['winner'] == 'model_a']['word_count_diff'], bins=50, 
             alpha=0.7, label='Model A Wins', color='#2ecc71', edgecolor='black')
axes[1].hist(train[train['winner'] == 'model_b']['word_count_diff'], bins=50, 
             alpha=0.7, label='Model B Wins', color='#e74c3c', edgecolor='black')
axes[1].hist(train[train['winner'] == 'tie']['word_count_diff'], bins=50, 
             alpha=0.7, label='Tie', color='#3498db', edgecolor='black')
axes[1].set_title('Word Count Difference Distribution by Winner', fontweight='bold')
axes[1].set_xlabel('Response A Word Count - Response B Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("SAMPLE PROMPTS AND RESPONSES")
print("="*80)

for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Prompt: {train.iloc[i]['prompt'][:200]}...")
    print(f"Response A: {train.iloc[i]['response_a'][:200]}...")
    print(f"Response B: {train.iloc[i]['response_b'][:200]}...")
    print(f"Winner: {train.iloc[i]['winner']}")
    print("-" * 80)

print("\n" + "="*80)
print("EDA COMPLETED")
print("="*80)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
%matplotlib inline



train = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)


print("\nTrain Columns:", train.columns.tolist())
print("\nTest Columns:", test.columns.tolist())

print("\n" + "="*80)
print("TRAIN DATA SAMPLE")
print("="*80)
display(train.head())


print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(train.isnull().sum())

print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(train.dtypes)



print("\n" + "="*80)
print("TARGET DISTRIBUTION")
print("="*80)

target_counts = pd.DataFrame({
    'winner_model_a': [train['winner_model_a'].sum()],
    'winner_model_b': [train['winner_model_b'].sum()],
    'winner_tie': [train['winner_tie'].sum()]
})
display(target_counts)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
colors = ['#2ecc71', '#e74c3c', '#3498db']

target_counts.T.plot(kind='bar', ax=axes[0], color=colors, legend=False)
axes[0].set_title('Target Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Winner Category')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Model A', 'Model B', 'Tie'], rotation=0)

target_pcts = target_counts.T / len(train) * 100
target_pcts.plot(kind='bar', ax=axes[1], color=colors, legend=False)
axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Winner Category')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_xticklabels(['Model A', 'Model B', 'Tie'], rotation=0)

plt.tight_layout()
plt.show()



train['prompt_length'] = train['prompt'].str.len()
train['response_a_length'] = train['response_a'].str.len()
train['response_b_length'] = train['response_b'].str.len()
train['prompt_word_count'] = train['prompt'].str.split().str.len()
train['response_a_word_count'] = train['response_a'].str.split().str.len()
train['response_b_word_count'] = train['response_b'].str.split().str.len()



print("\n" + "="*80)
print("TEXT STATISTICS")
print("="*80)
display(train[['prompt_length', 'response_a_length', 'response_b_length', 
               'prompt_word_count', 'response_a_word_count', 'response_b_word_count']].describe())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].hist(train['prompt_length'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Prompt Length Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Character Count')

axes[0, 1].hist(train['response_a_length'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Response A Length Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Character Count')

axes[0, 2].hist(train['response_b_length'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Response B Length Distribution', fontweight='bold')
axes[0, 2].set_xlabel('Character Count')

axes[1, 0].hist(train['prompt_word_count'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Prompt Word Count Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Word Count')

axes[1, 1].hist(train['response_a_word_count'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Response A Word Count Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Word Count')

axes[1, 2].hist(train['response_b_word_count'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Response B Word Count Distribution', fontweight='bold')
axes[1, 2].set_xlabel('Word Count')

plt.tight_layout()
plt.show()



print("\n" + "="*80)
print("MODEL DISTRIBUTION")
print("="*80)
print("\nModel A Distribution:")
display(train['model_a'].value_counts().head(10))
print("\nModel B Distribution:")
display(train['model_b'].value_counts().head(10))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
top_model_a = train['model_a'].value_counts().head(15)
axes[0].barh(range(len(top_model_a)), top_model_a.values)
axes[0].set_yticks(range(len(top_model_a)))
axes[0].set_yticklabels(top_model_a.index)
axes[0].set_title('Top 15 Model A Distribution', fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].invert_yaxis()

top_model_b = train['model_b'].value_counts().head(15)
axes[1].barh(range(len(top_model_b)), top_model_b.values)
axes[1].set_yticks(range(len(top_model_b)))
axes[1].set_yticklabels(top_model_b.index)
axes[1].set_title('Top 15 Model B Distribution', fontweight='bold')
axes[1].set_xlabel('Count')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()



train['winner'] = train.apply(lambda row: 'model_a' if row['winner_model_a'] == 1 
                               else ('model_b' if row['winner_model_b'] == 1 else 'tie'), axis=1)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, winner_type in enumerate(['model_a', 'model_b', 'tie']):
    winner_data = train[train['winner'] == winner_type]
    title = ('Response Length for Model A Wins' if winner_type=='model_a'
             else 'Response Length for Model B Wins' if winner_type=='model_b'
             else 'Response Length for Ties')
    
    axes[idx].hist(winner_data['response_a_length'].dropna(), bins=30, alpha=0.5, label='Response A', edgecolor='black')
    axes[idx].hist(winner_data['response_b_length'].dropna(), bins=30, alpha=0.5, label='Response B', edgecolor='black')
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].set_xlabel('Character Count')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.show()



train['length_diff'] = train['response_a_length'] - train['response_b_length']
train['word_count_diff'] = train['response_a_word_count'] - train['response_b_word_count']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(train[train['winner'] == 'model_a']['length_diff'].dropna(), bins=50, alpha=0.7, label='Model A Wins')
axes[0].hist(train[train['winner'] == 'model_b']['length_diff'].dropna(), bins=50, alpha=0.7, label='Model B Wins')
axes[0].hist(train[train['winner'] == 'tie']['length_diff'].dropna(), bins=50, alpha=0.7, label='Tie')
axes[0].set_title('Length Difference Distribution by Winner', fontweight='bold')
axes[0].set_xlabel('Response A Length - Response B Length')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)

axes[1].hist(train[train['winner'] == 'model_a']['word_count_diff'].dropna(), bins=50, alpha=0.7, label='Model A Wins')
axes[1].hist(train[train['winner'] == 'model_b']['word_count_diff'].dropna(), bins=50, alpha=0.7, label='Model B Wins')
axes[1].hist(train[train['winner'] == 'tie']['word_count_diff'].dropna(), bins=50, alpha=0.7, label='Tie')
axes[1].set_title('Word Count Difference Distribution by Winner', fontweight='bold')
axes[1].set_xlabel('Response A Word Count - Response B Word Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()



print("\n" + "="*80)
print("SAMPLE PROMPTS AND RESPONSES")
print("="*80)

for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Prompt: {train.iloc[i]['prompt'][:200]}...")
    print(f"Response A: {train.iloc[i]['response_a'][:200]}...")
    print(f"Response B: {train.iloc[i]['response_b'][:200]}...")
    print(f"Winner: {train.iloc[i]['winner']}")
    print("-" * 80)

print("\n" + "="*80)
print("EDA COMPLETED")
print("="*80)


