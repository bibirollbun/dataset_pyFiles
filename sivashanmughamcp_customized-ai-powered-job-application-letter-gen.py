import openai
import pandas as pd
import os

print("Imports are ready!")


import pandas as pd

# STEP 1: Create sample data (this fixes your error)
data = pd.DataFrame({
    "name": ["Siva", "Arun", "Meena"],
    "resume": [
        "I am a mechanical engineer with 3 years experience",
        "I am a Python developer with AI knowledge",
        "I am an MBA graduate with operations experience"
    ],
    "job_description": [
        "Looking for a mechanical engineer",
        "Looking for a Python developer",
        "Looking for an operations manager"
    ]
})

# STEP 2: Function to generate cover letter
def generate_cover_letter(resume, job_description):
    return f"""
Dear Hiring Manager,

Based on my resume:
{resume}

I believe I am a strong match for the following job:
{job_description}

Thank you for your consideration.
"""

# STEP 3: Select first row safely
sample = data.iloc[0]

# STEP 4: Generate cover letter
letter = generate_cover_letter(sample["resume"], sample["job_description"])

# STEP 5: Print result
print(letter)


for index, row in data.iterrows():
    letter = generate_cover_letter(row["resume"], row["job_description"])
    print(f"\n--- Cover Letter {index+1} ---\n")
    print(letter)


data.to_csv("generated_cover_letters.csv", index=False)
print("Saved as generated_cover_letters.csv")


data.to_excel("generated_cover_letters.xlsx", index=False)
print("Saved as generated_cover_letters.xlsx")

