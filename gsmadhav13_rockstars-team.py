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


from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Example simple responses based on sentiment
responses = {
    "POSITIVE": [
        "I'm glad to hear you are feeling good. Keep it up!",
        "That's wonderful! Remember to keep doing things that make you happy."
    ],
    "NEUTRAL": [
        "It's okay to have neutral days. How can I support you today?",
        "Sometimes a calm day is just what we need."
    ],
    "NEGATIVE": [
        "I'm sorry to hear that. Remember, you're not alone.",
        "It's okay to feel down sometimes. Here are some resources that may help:",
        "Try to take a deep breath. It might help to talk to someone supportive."
    ]
}

# Mental health resource links
resources = """
- National Suicide Prevention Lifeline: 1-800-273-TALK (1-800-273-8255)
- Crisis Text Line: Text HOME to 741741
- MentalHealth.gov: https://www.mentalhealth.gov/
"""

def get_supportive_response(user_input):
    # Analyze sentiment
    result = sentiment_analyzer(user_input)[0]
    label = result['label']
    
    # Select response based on sentiment label
    if label == "POSITIVE":
        reply = responses["POSITIVE"][0]
    elif label == "NEUTRAL":
        reply = responses["NEUTRAL"][0]
    else:
        # NEGATIVE or any other label
        reply = responses["NEGATIVE"][1] + "\n" + resources
    
    return reply

def main():
    print("Mental Health Support Agent\n(Type 'quit' to exit)")
    while True:
        user_text = input("You: ")
        if user_text.lower() == "quit":
            print("Agent: Take care! Remember you are not alone.")
            break
        
        response = get_supportive_response(user_text)
        print("Agent:", response)

if __name__ == "__main__":
    main()


