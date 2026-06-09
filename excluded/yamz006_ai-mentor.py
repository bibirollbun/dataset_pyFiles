# Simple AI Mentor for Career Readiness (Prototype)

# Resume analysis function
def analyze_resume(skills, target_role):
    db = {
        "data scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "sql"],
        "ml engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch"],
        "software engineer": ["python", "dsa", "oops", "sql", "system design"]
    }

    required = db.get(target_role.lower(), [])
    missing = [skill for skill in required if skill not in skills]

    result = {
        "target_role": target_role,
        "your_skills": skills,
        "required_skills": required,
        "missing_skills": missing
    }
    return result

# Learning path generator
def generate_learning_path(missing_skills):
    path = []
    for skill in missing_skills:
        path.append("Learn " + skill + " from beginner to advanced level using free YouTube or Kaggle tutorials.")
    return path

# Interview question generator
def interview_questions(role):
    questions = {
        "data scientist": [
            "Explain bias vs variance.",
            "What is cross validation?",
            "Explain a project you worked on.",
            "How do you evaluate a machine learning model?"
        ],
        "ml engineer": [
            "What is overfitting?",
            "Explain gradient descent.",
            "Difference between CNN and RNN.",
            "How does backpropagation work?"
        ]
    }
    return questions.get(role.lower(), ["No questions available for this role."])

# Main program
print("===== AI Mentor for Career Readiness =====")

user_name = input("Enter your name: ")

skills_input = input("Enter your current skills (comma separated): ")
skills = [s.strip().lower() for s in skills_input.split(",")]

target_role = input("Enter your target job role: ")

print("\n--- Resume Analysis ---")
analysis = analyze_resume(skills, target_role)
print("Your Skills:", analysis["your_skills"])
print("Required Skills for", analysis["target_role"] + ":", analysis["required_skills"])
print("Missing Skills:", analysis["missing_skills"])

print("\n--- Personalized Learning Path ---")
learning_path = generate_learning_path(analysis["missing_skills"])
for step in learning_path:
    print("-", step)

print("\n--- Interview Preparation ---")
questions = interview_questions(target_role)
print("Sample Interview Questions:")
for q in questions:
    print("-", q)

print("\nYour AI Mentor session is complete. All the best for your career growth,", user_name + "!")


