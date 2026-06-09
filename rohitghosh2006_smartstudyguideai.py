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


# -------------------------------
# SMARTSTUDYGUIDE AI - SIMPLE VERSION
# -------------------------------

class SmartStudyGuideAI:
    def __init__(self):
        pass

    def answer_question(self, question):
        question = question.lower()

        # SIMPLE KNOWLEDGE DATABASE
        notes = {
            "gravity": """
Gravity is a natural force that pulls objects toward the center of the Earth.
Important points:
1. Gravity keeps us grounded on Earth.
2. It is the reason objects fall downwards.
3. It holds planets in orbit around the Sun.
            """,

            "photosynthesis": """
Photosynthesis is the process by which green plants make their own food.
Important points:
1. Plants use sunlight, water, and carbon dioxide.
2. It produces glucose (food) and oxygen.
3. It occurs in chlorophyll present in leaves.
            """,

            "atom": """
An atom is the smallest unit of matter.
Important points:
1. It consists of protons, neutrons, and electrons.
2. Atoms form molecules.
3. Everything in the universe is made of atoms.
            """,

            "rain": """
Rain is the water that falls from clouds.
Important points:
1. It happens due to condensation of water vapor.
2. Clouds become heavy and release water droplets.
3. Rain is important for plants and groundwater.
            """
        }

        # RETURN NOTES IF FOUND
        for key in notes:
            if key in question:
                return notes[key]

        # DEFAULT REPLY
        return "Sorry, I don't have notes for this topic yet."

# --------------------------------------------
# CREATE AGENT
# --------------------------------------------
agent = SmartStudyGuideAI()

# --------------------------------------------
# TEST OUTPUT
# --------------------------------------------
print(agent.answer_question("What is gravity?"))



print(agent.answer_question("Explain photosynthesis"))

