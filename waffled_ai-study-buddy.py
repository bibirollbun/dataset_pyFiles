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


import random


def teach_me(topic):
    explanations = [
        f"{topic} is actually simpler than it sounds. Think of it like this:",
        f"Let me break down {topic} in a friendly way:",
        f"Okay, here’s an easy explanation for {topic}:"
    ]

    return random.choice(explanations) + \
        f" {topic} basically means understanding the main idea behind it and how it works in everyday life."


def summarize(text):
    lines = text.split(".")
    summary = [f"- {line.strip()}" for line in lines if len(line.strip()) > 5]
    return "\n".join(summary)


def quiz_me(topic):
    questions = [
        f"What is the basic idea of {topic}?",
        f"Why is {topic} important?",
        f"Give one example related to {topic}.",
        f"Explain {topic} in one sentence."
    ]
    random.shuffle(questions)
    return questions[:3]


teach_me("Photosynthesis")


summarize("Photosynthesis is the process by which plants make food. It uses sunlight. It produces oxygen.")


quiz_me("Photosynthesis")


import random


def teach_me(topic):
    explanations = [
        f"{topic} is actually simpler than it sounds. Think of it like this:",
        f"Let me break down {topic} in a friendly way:",
        f"Okay, here’s an easy explanation for {topic}:"
    ]

    return random.choice(explanations) + \
        f" {topic} basically means understanding the main idea behind it and how it works in everyday life."


def summarize(text):
    lines = text.split(".")
    summary = [f"- {line.strip()}" for line in lines if len(line.strip()) > 5]
    return "\n".join(summary)


def quiz_me(topic):
    questions = [
        f"What is the basic idea of {topic}?",
        f"Why is {topic} important?",
        f"Give one example related to {topic}.",
        f"Explain {topic} in one sentence."
    ]
    random.shuffle(questions)
    return questions[:3]


teach_me("Photosynthesis")


summarize("Photosynthesis is the process by which plants make food. It uses sunlight. It produces oxygen.")


quiz_me("Photosynthesis")

