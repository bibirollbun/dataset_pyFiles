EduPrep Mentor – Multi-Agent Study Companion for College & Competitive Exams
Track: Agents for Good – Education
Problem Statement

College students and competitive exam aspirants struggle with creating notes, generating practice questions, and preparing realistic study plans. These tasks take a lot of time and reduce the hours available for actual learning and revision.

Solution

EduPrep Mentor is a multi-agent system designed to help learners by:

Generating structured notes

Creating MCQs + short answer questions

Building personalised study planners

Saving progress using session memory

This improves efficiency and reduces exam stress.


# Pseudo-code demonstrating multi-agent study companion

def notes_agent(topic, content=""):
    return f"""
### Notes for: {topic}

**Summary:**  
This topic is important for both college exams and competitive exams.

**Key Points:**  
- Point 1 about {topic}  
- Point 2  
- Point 3  

**Definitions:**  
- Important definition related to {topic}

**Exam Tips:**  
• Avoid common mistakes  
• Focus on concepts and applications  
"""


def quiz_agent(topic):
    return f"""
### Quiz for: {topic}

**College Short Answer Questions:**  
1. Explain the concept of {topic}.  
2. List key components.  
3. What is an application of {topic}?  
4. Define in simple words.  
5. Why is {topic} important?

**Competitive Exam MCQs:**  
1. {topic} is mainly used for?  
   A) Option 1  
   B) Option 2  
   C) Option 3  
   D) Option 4  
   **Answer: B**

2. Which of the following is true about {topic}?  
   A)… B)… C)… D)…  
   **Answer: A**
"""


def planner_agent(days_left=30, hours_per_day=3, topics=None):
    if topics is None:
        topics = ["Operating Systems", "DBMS", "Networks", "Aptitude"]

    daily_topic = topics[0]

    return f"""
### Study Plan (Sample)

**Days left:** {days_left}  
**Hours per day:** {hours_per_day}

**Plan:**  
Day 1: {topics[0]} – {hours_per_day} hrs  
Day 2: {topics[1]} – {hours_per_day} hrs  
Day 3: {topics[2]} – {hours_per_day} hrs  
Day 4: {topics[3]} – {hours_per_day} hrs  
Day 5: Revision and mock test  
"""


# Simulated orchestrator call
topic = "Operating Systems"
notes_output = notes_agent(topic)
quiz_output = quiz_agent(topic)
plan_output = planner_agent()

notes_output, quiz_output, plan_output


