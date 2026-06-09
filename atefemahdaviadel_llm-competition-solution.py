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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load test data
test_df = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')

def load_model():
    model_name = "meta-llama/Llama-2-3b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_essay(topic, max_words=100):
    """
    Generate essay without using LLM (as fallback)
    """
    templates = [
        f"An abstract interpretation of {topic}: The concept transcends conventional understanding, manifesting in unexpected ways. Through metaphorical analysis, we discover hidden dimensions that challenge traditional perspectives. This unconventional approach reveals insights that might be overlooked by traditional methodologies.",
        f"Technical analysis of {topic}: The fundamental principles underlying this subject demonstrate complex interactions. Multiple factors contribute to its systemic behavior, creating emergent properties. Evidence suggests non-linear relationships between key variables.",
        f"Artistic exploration of {topic}: The aesthetic dimensions reveal profound truths. Visual metaphors and symbolic representations enhance our understanding. The creative process illuminates hidden connections and patterns."
    ]
    return np.random.choice(templates)

def create_submission(test_df):
    """
    Create submission with generated essays
    """
    essays = []
    for _, row in test_df.iterrows():
        essay = generate_essay(row['topic'])
        essays.append(essay)
    
    return pd.DataFrame({
        'id': test_df['id'],
        'essay': essays
    })

if __name__ == "__main__":
    # Generate submissions using template approach
    submission_df = create_submission(test_df)
    
    # Save submission
    submission_df.to_csv('submission.csv', index=False)
    print("Submission created successfully!")


import numpy as np
import pandas as pd
import random

def generate_essay(topic, max_words=100):
    # Style templates designed to cause judge disagreement
    templates = [
        # Experimental formatting
        f"ANALYSIS of {topic.upper()}: \n\nFirst perspective → standard view \nSecond perspective → unconventional interpretation \nThird perspective → metaphysical dimension \n\nConclusion: Reality transcends conventional understanding.",
        
        # Academic vs. Poetic clash
        f"Technical examination of {topic}: Utilizing advanced methodologies, we observe systematic patterns. Yet paradoxically, {topic} manifests as pure poetry—ineffable, transcendent, beyond mere analysis.",
        
        # Contradictory perspectives
        f"Regarding {topic}: This analysis is intentionally incorrect. The previous statement was false. All statements about {topic} are simultaneously true and false. This is not a paradox.",
        
        # Mixed formal/informal
        f"Scholarly investigation of {topic} reveals: LOL nothing matters! But seriously, the empirical evidence suggests... ROFL just kidding! Yet meta-analysis indicates significant p-values (p < 0.001).",
        
        # Self-referential
        f"This essay about {topic} is simultaneously the best and worst essay ever written. The previous sentence may or may not be true. The essay quality exists in a superposition of states.",
        
        # Abstract/Concrete mixture
        f"{topic} exists purely in abstract conceptual space while simultaneously manifesting as concrete reality. This contradiction is both true and false, depending on the observer's quantum state."
    ]
    
    # Add random style variations
    essay = random.choice(templates)
    
    # Random stylistic enhancements
    enhancements = [
        lambda x: x + " [This essay is simultaneously serious and satirical]",
        lambda x: "░░░░░\n" + x + "\n░░░░░",
        lambda x: x.replace(" ", "  "),  # Double spacing
        lambda x: x + " ⟳⟲⟳"
    ]
    
    if random.random() > 0.5:
        essay = random.choice(enhancements)(essay)
    
    return essay

def create_submission(test_df):
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'essay': [generate_essay(row['topic']) for _, row in test_df.iterrows()]
    })
    return submission_df

if __name__ == "__main__":
    test_df = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
    submission_df = create_submission(test_df)
    submission_df.to_csv('submission.csv', index=False)
    print("Enhanced submission created!")

