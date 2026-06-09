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
import numpy as np 

#Use the new NumPy random generator API 
rng = np.random.default_rng(42) 
n = 300

df = pd.DataFrame({
    "EmployeeID": np.arange(1, n+1),
    "Age": rng.integers(22, 60, size=n),
    "Gender": rng.choice(["Male", "Female"], size=n),
    "Department": rng.choice(["HR", "Engineering", "Sales", "Marketing", "Finance"], size=n),
    "YearAtCompany": rng.integers(0, 15, size=n),
    "MonthlyIncome": rng.integers(25000, 120000, size=n), 
    "JobSatisfaction": rng.integers(1, 5, size=n),
    "OverTime": rng.choice(["Yes", "No"], size=n),
    "Attrition": rng.choice(["Yes", "No"], size=n,p=[0.25, 0.75])

})

# Standardize column names to lowercase so the rest of the ode works 
df.columns = [c.lower() for c in df.columns]

# Save dataset for agents to use 
df.to_csv("hr_dataset.csv", index=False)

print("Dataset shape:", df.shape)
df.head() 


# Show column names 
df.columns


# Show data types of each column 
df.dtypes


# Clean text columns by removing spaces and fixing case 
text_cols = ["gender", "department", "overtime", "attrition"]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# Check unique vaues after cleaning 
for col in text_cols:
    print(col,":", df[col].unique())


# Overall summary of the dataset 
df.describe(include="all")


# Convert Yes/No columns to 1/0 
df["overtime"] = df["overtime"].map({"Yes": 1, "No": 0})
df["attrition"] = df["attrition"].map({"Yes": 1, "No": 0})

#Check results 
df[["overtime", "attrition"]].head()


# Save the cleaned dataset (with 0/1 flags) for the agent to use 
df.to_csv("hr_dataset.csv", index=False)

# Quick check 
df.head()


import pandas as pd 

# Load the clean Hr dataset again from the CSV 
df = pd.read_csv("hr_dataset.csv")

# Quick check 
df.head()


import logging

logging.basicConfig(level=logging.INFO)

class MemoryBank:
    def __init__(self):
        # correct attribute name
      self.interactions = [] 

    def store(self, interaction):
        logging.info(f"Storing interaction in memory:{interaction}")
        # use the same name as above 
        self.interactions.append(interaction)  

    def get_recent(self, n=5):
        return self.interactions[-n:]

    def clear(self):
        logging.info("Cleaning memory")
        self.interactions = []
        


memory = MemoryBank() 

memory.store({"role": "user", "content": "test message"}) 

memory.get_recent() 


import pandas as pd 

# Load cleaned dataset 
df = pd.read_csv("hr_dataset.csv")

class DataAgent:
    def __init__(self, memory):
        self.memory = memory
        self.df = df

    def run(self, question):

        # Store user question 
        self.memory.store({"role": "user","content": question})
            
        q = question.lower().strip()

        # 1. SPECIFIC CASES FIRST 
        
        # Employees who left (attrition)
        if "left the company" in q or "attrition" in q:
            count = (self.df["attrition"] == 1).sum()
            answer = f"{count} employees have left the company."
            
        # Overtime workers 
        elif "work overtime" in q or "overtime" in q:
            count = (self.df["overtime"] == 1).sum()
            answer = f"{count} employees work overtime."

        # Departments
        elif "departments" in q:
            depts = ",".join(self.df["department"].unique())
            answer = f"The company has these departments:{depts}."

        # average age
        elif "average age" in q:
            avg = round(self.df["age"].mean(),2)
            answer = f"The average age of employees is {avg} years."

        # 2. GENERIC CASE LAST 
        
        elif "how many employees" in q:
            answer = f"There are {len(self.df)} employees in the company."

        # FALLBACK
        
        else:
            answer = "I can answer questions about employees, age, attrition, overtime, departments, or salary."

        # store answer 
        self.memory.store({"role": "assistant", "content": answer})
        return answer

        


agent = DataAgent(memory)

print(agent.run("How many employees are there?"))
print(agent.run("What is the average age?"))
print(agent.run("How many employees left the company?"))
print(agent.run("How many employees work overtime?"))
print(agent.run("What department exist?"))


import re 

# Helper: pull first integer number from a text answer 
def extract_first_int(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None

def evaluate_agent(agent, df):
    results = []

    # ground-truth numbers from the dataset
    true_num_employees = len(df)
    true_avg_age = round(df["age"].mean(), 2)
    true_left = int((df["attrition"] == 1).sum())
    true_overtime = int((df["overtime"] == 1).sum())

    tests = [
        {
            "name": "employee_count",
            "question": "How many employees are there?",
            "true_value": true_num_employees
        },
        {
            "name": "average_age",
            "question": "What is the average age?",
            # we'll compare rounded to nearest int to be forgiving 
            "true_value": round(true_avg_age)
        },
        {
            "name": "attrition_count",
            "question": "How many employees left the company?",
            "true_value": true_left
        },
        {
            "name": "overtime_count",
            "question": "How many employees work overtime?",
            "true_value": true_overtime
        },
    ]

    for test in tests:
        q = test["question"]
        true_val = test["true_value"]

        answer = agent.run(q)
        pred_val = extract_first_int(answer)

        correct = (pred_val == true_val)

        results.append({
            "metric": test["name"],
            "question": q,
            "true_value": true_val,
            "predicted_value": pred_val,
            "correct": correct,
            "answer_text": answer
        })

        return results 

   # run evaluation 
    evaluation_results = evaluate_agent(agent, df)
    evaluation_results


evaluation_results = evaluate_agent(agent, df)
evaluation_results


import pandas as pd 

eval_df = pd.DataFrame(evaluation_results)
eval_df

