# pip install transformers --upgrade


import keras
import keras_hub
import numpy as np
import pandas as pd                     


gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
gpt2_lm.generate("I want to say", max_length=30)


# Example: Generate an essay on a topic
def generate_essay(topic, max_length=100, num_paragraphs=1):
    """
    Generate an essay based on the given topic.
    """
    essay = ""  
    for _ in range(num_paragraphs):
        paragraph = gpt2_lm.generate(topic, max_length=max_length)
        essay += paragraph + "\n\n"
    return essay

# Generate essay
topic = "The Importance of Renewable Energy"
essay = generate_essay(topic, max_length=100, num_paragraphs=1)
print(essay)


test = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
test


essays = []  # List to store generated essays

# Generate essays and append them to the list
for _, row in test.iterrows():
    topic = row['topic']
    essay = generate_essay(topic, max_length=150, num_paragraphs=1)
    print(f"\n{essay}")  # Print the essay
    print('\n\n***********************\n\n')
    essays.append(essay)

# Add the generated essays to the DataFrame
test['essay'] = essays

# Drop the original 'topic' column
test = test.drop(columns=['topic'])

# Save the DataFrame with only the 'id' and 'essay' columns
test[['id', 'essay']].to_csv('submission.csv', index=False)

print("The file has been saved as submission.csv!")

