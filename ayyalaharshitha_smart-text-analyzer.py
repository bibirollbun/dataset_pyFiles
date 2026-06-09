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


# Simple sentiment model (rule-based)
def predict_sentiment(text):
    text = text.lower()
    positive_words = ["good", "happy", "excellent", "love", "great"]
    negative_words = ["bad", "sad", "terrible", "hate", "angry"]

    score = 0
    for w in positive_words:
        if w in text:
            score += 1
    for w in negative_words:
        if w in text:
            score -= 1

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


def explain_sentiment(text, sentiment):
    if sentiment == "Positive":
        return "The text expresses positive feelings like joy or appreciation."
    elif sentiment == "Negative":
        return "The text contains negative emotions or unpleasant expressions."
    else:
        return "The text has no strong emotional tone."


# Example user text (you can change this anytime)
user_text = "I love this project! It is excellent."

# AI model prediction
sentiment = predict_sentiment(user_text)

# AI agent explanation
explanation = explain_sentiment(user_text, sentiment)

# Display results
print("User Text:", user_text)
print("Sentiment:", sentiment)
print("Explanation:", explanation)


