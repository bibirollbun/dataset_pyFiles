# 1) Setup & imports
import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Image

print('Libraries imported')


def mock_llm(prompt: str, role: str = 'assistant') -> str:
    """Simple rule-based mock LLM for demo purposes.
    It recognizes a few keywords to return structured JSON responses when needed.
    """
    p = prompt.lower()
    if 'create plan' in p or 'generate plan' in p:
        # return a JSON plan with topics spaced over days
        plan = [
            {'topic': 'fractions', 'duration_min': 10, 'due_in_days': 1},
            {'topic': 'algebra', 'duration_min': 15, 'due_in_days': 1},
        ]
        return json.dumps(plan)
    if 'grade' in p or 'score' in p:
        # structured grade
        score = 1.0 if 'correct' in p or 'answer:' in p else 0.0
        return json.dumps({'score': score, 'feedback': 'Auto-graded by mock LLM'})
    if 'explain' in p or 'explanation' in p:
        return "Step-by-step explanation: identify operation, simplify terms, compute result."
    # default helpful reply
    return 'MOCK_RESPONSE: ' + prompt[:200]

print('Mock LLM ready')


@dataclass
class MemoryBank:
    student_history: Dict[str, Any] = field(default_factory=dict)
    def update(self, student_id: str, key: str, value: Any):
        self.student_history.setdefault(student_id, {}).setdefault(key, []).append(value)
    def get(self, student_id: str):
        return self.student_history.get(student_id, {})
    def show(self):
        return self.student_history

class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm
    def create_plan(self, student_profile, topics, duration_days=7):
        prompt = f"Create plan for {student_profile['id']} using topics {topics}"
        resp = self.llm(prompt)
        try:
            plan = json.loads(resp)
        except Exception:
            # fallback simple plan
            plan = [{'topic': t, 'duration_min': 10, 'due_in_days': 1} for t in topics]
        return plan

class TutorAgent:
    def __init__(self, llm, tools, memory: MemoryBank):
        self.llm = llm
        self.tools = tools
        self.memory = memory
    def run_session(self, student_id, plan_item):
        topic = plan_item['topic']
        # choose a question from question_bank
        q = random.choice(question_bank.get(topic, [{'q':'Example Q','answer':'example'}]))
        student_answer = self.tools['simulate_student'](student_id, topic, q)
        grade_prompt = f"Grade answer: answer:{student_answer} question:{q['q']}"
        grade_resp = self.llm(grade_prompt)
        try:
            grade = json.loads(grade_resp)
        except Exception:
            grade = {'score': 0.0, 'feedback':'No grade'}
        explanation = self.llm('explain ' + q['q'])
        if grade.get('score',0) < 1.0:
            self.memory.update(student_id, 'weak_topics', topic)
        else:
            self.memory.update(student_id, 'strengths', topic)
        return {'question': q['q'], 'answer': student_answer, 'grade': grade, 'explanation': explanation}

class SchedulerAgent:
    def __init__(self):
        self.schedule = {}
    def add(self, student_id, plan_item):
        self.schedule.setdefault(student_id, []).append(plan_item)
    def get_next(self, student_id):
        return self.schedule.get(student_id, [])

class EvaluatorAgent:
    def __init__(self, memory: MemoryBank):
        self.memory = memory
    def compute_metrics(self, student_id, history):
        scores = [h['grade'].get('score',0) for h in history if 'grade' in h]
        accuracy = float(np.mean(scores)) if scores else 0.0
        return {'accuracy': accuracy, 'attempts': len(scores)}

print('Agents defined')


def grade_answer(question, student_answer):
    correct = str(student_answer).strip().lower() == str(question['answer']).strip().lower()
    return {'score': 1.0 if correct else 0.0, 'feedback': 'Correct' if correct else 'Incorrect'}

def simulate_student(student_id, topic, question):
    # Simple probabilistic simulator: weak topic => 60% wrong
    student = next((s for s in sample_students if s['id']==student_id), None)
    is_weak = topic in student.get('weak_topics', [])
    correct = random.random() > (0.4 if is_weak else 0.1)
    return question['answer'] if correct else 'wrong_answer'

print('Tools ready')


sample_students = [
    {'id':'s1','name':'Student A','weak_topics':['fractions','algebra']},
    {'id':'s2','name':'Student B','weak_topics':['calculus']},
    {'id':'s3','name':'Student C','weak_topics':['trig','algebra']},
]

question_bank = {
    'fractions': [
        {'q':'Simplify 1/2 + 1/3','answer':'5/6'},
        {'q':'What is 3/4 of 20?','answer':'15'},
    ],
    'algebra': [
        {'q':'Solve x+3=8','answer':'5'},
        {'q':'Simplify 2(x+3)','answer':'2x+6'},
    ],
    'calculus': [
        {'q':'Derivative of x^2 ?','answer':'2x'},
    ],
    'trig': [
        {'q':'sin(90°) = ?','answer':'1'},
    ]
}
print('Sample data ready')


memory = MemoryBank()
planner = PlannerAgent(mock_llm)
tutor = TutorAgent(mock_llm, {'simulate_student': simulate_student}, memory)
scheduler = SchedulerAgent()
evaluator = EvaluatorAgent(memory)

student = sample_students[0]
plan = planner.create_plan(student, student['weak_topics'])
for item in plan:
    scheduler.add(student['id'], item)

history = []
for item in scheduler.get_next(student['id']):
    res = tutor.run_session(student['id'], item)
    # post-process: use grade tool if needed
    if isinstance(res.get('grade'), dict) and res['grade'].get('feedback')=='No grade':
        # fallback grade using tool
        q = next((qq for qq in question_bank[item['topic']] if qq['q']==res['question']), question_bank[item['topic']][0])
        res['grade'] = grade_answer(q, res['answer'])
    history.append(res)

metrics = evaluator.compute_metrics(student['id'], history)
print('History:')
for h in history:
    print(h)
print('\nMetrics:', metrics)


def simulate_group(n_users=20):
    results = []
    for i in range(n_users):
        baseline = 0.60 + random.random()*0.08
        manual_final = min(1.0, baseline + (0.05 + random.random()*0.05))
        agent_final = min(1.0, baseline + (0.18 + random.random()*0.05))
        results.append({'user_id':f'user_{i}','baseline':baseline,'manual_final':manual_final,'agent_final':agent_final})
    return pd.DataFrame(results)

df = simulate_group(20)

# Plotting: line plot of baseline, manual, agent
plt.figure(figsize=(8,4))
plt.plot(df.index, df['baseline'], label='Baseline')
plt.plot(df.index, df['manual_final'], label='Manual final')
plt.plot(df.index, df['agent_final'], label='Agent final')
plt.xlabel('Simulated User')
plt.ylabel('Accuracy')
plt.title('Simulated Baseline vs Outcomes')
plt.legend()
plt.tight_layout()
plt.show()

display(df.describe())


avg_baseline = df['baseline'].mean()
avg_manual = df['manual_final'].mean()
avg_agent = df['agent_final'].mean()
summary = pd.DataFrame({'metric':['baseline','manual','agent'],'value':[avg_baseline, avg_manual, avg_agent]})
print(summary)

plt.figure(figsize=(6,3))
plt.bar(summary['metric'], summary['value'])
plt.ylim(0,1)
plt.title('Average Accuracy Comparison')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()


# Display generated diagram images if present in the notebook files
import os
assets = ['A_collection_of_five_diagrams_in_a_2x3_grid_depict.png', 'A_flat_digital_illustration_features_promotional_m.png']
for a in assets:
    if os.path.exists(a):
        display(Image(a))
    else:
        print('Asset not found in notebook files:', a)

