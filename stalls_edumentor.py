import random, json

memory = {
    "weak_topics": [],
    "scores": []
}

def teacher_agent(topic):
    return f"Today we learn {topic}. It is an important concept used in daily problem solving."

def quiz_agent(topic):
    return [
        {"q":"2+2=?", "a":"4"},
        {"q":"5+3=?", "a":"8"},
        {"q":"10-4=?", "a":"6"},
        {"q":"6*2=?", "a":"12"},
        {"q":"9/3=?", "a":"3"}
    ]

def evaluator_agent(quiz, user_answers):
    score = 0
    for q, ua in zip(quiz, user_answers):
        if str(q['a']) == str(ua):
            score += 1
    memory['scores'].append(score)
    if score < 3:
        memory['weak_topics'].append("Basic Math")
    return score

def study_planner():
    if memory['weak_topics']:
        return "Revise Basic Math"
    return "Move to next topic"



topic = "Basic Arithmetic"
print(teacher_agent(topic))

quiz = quiz_agent(topic)
user_answers = ["4","7","6","12","2"]

score = evaluator_agent(quiz, user_answers)
print("Score:", score)
print("Next Study Plan:", study_planner())

