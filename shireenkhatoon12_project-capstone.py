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

class Agent:
    def __init__(self, name):
        self.name = name

    def run(self, input_text):
        # Planner Agent
        if self.name == "Planner":
            return {
                "task1": "Design Generation",
                "task2": "Content Writing",
                "task3": "Evaluation"
            }

        # Design Agent
        elif self.name == "Designer":
            colors = ["#FFD700", "#FF69B4", "#00BFFF", "#32CD32", "#FF4500", "#8A2BE2"]
            layouts = ["Minimalist", "Grid", "Split-screen", "Carousel", "Story-style"]
            fonts = ["Montserrat", "Lato", "Roboto", "Poppins", "Raleway"]
            return {
                "color_palette": random.sample(colors, 3),
                "layout": random.choice(layouts),
                "typography": random.choice(fonts)
            }

        # Content Agent
        elif self.name == "Writer":
            captions = [
                "Ace your exams with style! âœ¨ðŸ“š",
                "Study smart, not hard ðŸ’¡",
                "Motivation is the key ðŸ”‘",
                "Success starts with planning your study ðŸ“–",
                "Turn your focus into results ðŸŽ¯"
            ]
            hashtags = ["#StudentLife", "#ExamPrep", "#Motivation", "#StudyTips", "#CreativeLearning"]
            return {
                "caption": random.choice(captions),
                "hashtags": hashtags
            }

        # Evaluation Agent
        elif self.name == "Evaluator":
            score = random.randint(7, 10)
            suggestions = [
                "Colors are great. Layout is clean. Caption is catchy.",
                "Good typography, try adjusting color contrast.",
                "Excellent design structure, hashtags are relevant."
            ]
            return {
                "score": score,
                "suggestion": random.choice(suggestions)
            }


planner = Agent("Planner")
designer = Agent("Designer")
writer = Agent("Writer")
evaluator = Agent("Evaluator")


def run_creativesense(user_input):
    tasks = planner.run(user_input)
    design = designer.run(tasks)
    content = writer.run(design)
    evaluation = evaluator.run({"design": design, "content": content})

    return {
        "Tasks": tasks,
        "Design": design,
        "Content": content,
        "Evaluation": evaluation
    }



result1 = run_creativesense("Create an aesthetic Instagram post for exam motivation")
print(result1)


result2 = run_creativesense("Design a brand kit for a bakery business")
print(result2)


result3 = run_creativesense("Make a social media post about sustainable living")
print(result3)



for i, result in enumerate([result1, result2, result3], 1):
    print(f"\nExample {i} Evaluation Score: {result['Evaluation']['score']}")
    print(f"Suggestion: {result['Evaluation']['suggestion']}")

