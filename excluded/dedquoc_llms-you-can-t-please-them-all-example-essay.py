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


import pandas as pd
path = '/kaggle/input/llms-you-cant-please-them-all/'
# train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')


print(test_df.isnull().sum())


print(test_df['topic'].value_counts())


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_word_cloud(text, title):
    """
    Generates a word cloud from the given text.
    
    Parameters:
    - text: A string containing the text data.
    - title: Title for the word cloud plot.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Combine all topics into a single string
all_topics = ' '.join(test_df['topic'].astype(str))

# Generate the word cloud for the topics
generate_word_cloud(all_topics, 'Word Cloud of All Topics')


import random

def generate_essay(topic):
    """
    Generates an essay based on the given topic.
    
    Parameters:
    - topic: The topic of the essay.
    
    Returns:
    - A string containing the essay text.
    """
    # Example essay template with variations to maximize disagreement
    essay = f"""
    # The Impact of {topic} on Modern Society

    {topic} is a **revolutionary** field that has the potential to transform our world. This technology, which harnesses the principles of quantum mechanics, is poised to solve problems that are currently intractable for classical computers.

    *Quantum supremacy* has been a hot topic in recent years, with major tech companies and research institutions investing heavily in this area. However, the practical applications of {topic} are still in their infancy, and many challenges remain.

    - **Challenges**:
      - **Error Rates**: {topic} is highly susceptible to errors due to its delicate nature. Even the slightest interference can cause a quantum bit (qubit) to lose its quantum state, a phenomenon known as decoherence.
      - **Scalability**: Building a large-scale {topic} is an immense challenge. Current {topic} has a limited number of qubits, and scaling up to a practical number is a significant engineering feat.
      - **Algorithm Development**: Developing algorithms that can run efficiently on {topic} is a complex task. Traditional algorithms need to be adapted or entirely new algorithms need to be created to take advantage of {topic}'s unique properties.

    - **Opportunities**:
      - **Cryptography**: {topic} has the potential to break many of the cryptographic systems currently in use. This has significant implications for data security and has spurred research into {topic}-resistant cryptography.
      - **Optimization Problems**: {topic} can solve complex optimization problems much faster than classical computers. This has applications in fields such as logistics, finance, and drug discovery.
      - **Simulation**: {topic} can simulate quantum systems, which is difficult or impossible for classical computers. This can lead to breakthroughs in materials science and chemistry.

    In conclusion, while {topic} holds immense promise, it also presents significant challenges. The field is rapidly evolving, and continued research and development are essential to realizing its full potential.
    """

    # Add variations to maximize disagreement
    variations = [
        "This is a **tragic** loss for the scientific community.",
        "However, the benefits far outweigh the drawbacks, making {topic} a **magnificent** achievement.",
        "Despite the challenges, the potential of {topic} is **revolutionary** and cannot be ignored.",
        "The impact of {topic} on modern society is **enormous** and will shape the future in ways we can only imagine.",
        "While the road ahead is long, the journey is **worthwhile** and filled with endless possibilities."
    ]

    # Randomly select a variation to add to the essay
    essay += f"\n\n{random.choice(variations).format(topic=topic)}"

    return essay


# Generate essays for all test IDs
essays = []
for index, row in test_df.iterrows():
    essay = generate_essay(row['topic'])
    essays.append({'id': row['id'], 'essay': essay})

# Create a DataFrame from the list of essays
submission_df = pd.DataFrame(essays)


# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
submission_df

