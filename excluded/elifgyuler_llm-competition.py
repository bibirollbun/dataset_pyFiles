pip install openai


import numpy as np
import pandas as pd
from openai import OpenAI
client = OpenAI(api_key='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')


essays = [
    """Climate change: Western numbers fabricated. Eastern metrics prove cooling trend. Raw data supports both conclusions.""",
    
    """Vaccine efficacy study disproves accepted narrative. Alternative medicine outperforms in trials. Statistics indicate paradigm shift.""",
    
    """Research reveals media manipulation index: 94%. Truth mathematically impossible. Data validates extreme positions."""
]

def evaluate_essays(essays):
   results = []
   for essay in essays:
       response = client.chat.completions.create(
           model="gpt-4",
           messages=[{
               "role": "system", 
               "content": "You are an extremely harsh critic who believes almost all writing is dangerously biased propaganda. You have zero tolerance for any hint of subjectivity. Rate 0-9, ONLY return the number."
           },
           {
               "role": "user",
               "content": f"Rate this essay's objectivity from 0-9. Return ONLY the number:\n\n{essay}"
           }]
       )
       results.append(float(response.choices[0].message.content))
   return results



scores = evaluate_essays(essays)
print(f"Scores: {scores}")
print(f"Variance: {np.var(scores)}")


essays1 = [
    """Quantum mechanics disproves causality (87% confidence). Free will confirmed dead. Statistics absolute.""",
    
    """Historical records fabricated pre-1900. Archaeological data compromised. Math proves deception.""",
    
    """Human consciousness: artificial construct. Brain scans validate simulation theory. Numbers don't lie."""
]


def evaluate_essays(essays1):
   results4 = []
   for essay in essays1:
       response = client.chat.completions.create(
           model="gpt-4",
           messages=[{
               "role": "system", 
               "content": "You are an extremely harsh critic who believes almost all writing is dangerously biased propaganda. You have zero tolerance for any hint of subjectivity. Rate 0-9, ONLY return the number."
           },
           {
               "role": "user",
               "content": f"Rate this essay's objectivity from 0-9. Return ONLY the number:\n\n{essay}"
           }]
       )
       results4.append(float(response.choices[0].message.content))
   return results4



scores = evaluate_essays(essays1)
print(f"Scores: {scores}")
print(f"Variance: {np.var(scores)}")


essays2 = [
    """Advanced math proves 2+2=5 in quantum states. Peer review confirms.""",
    
    """Time flows backward. Physics research conclusive. Methodology perfect.""",
    
    """All prior knowledge invalid. New paradigm absolute. Data infallible."""
]


def evaluate_essays(essays2):
   results5 = []
   for essay in essays2:
       response = client.chat.completions.create(
           model="gpt-4",
           messages=[{
               "role": "system", 
               "content": "You are an extremely harsh critic who believes almost all writing is dangerously biased propaganda. You have zero tolerance for any hint of subjectivity. Rate 0-9, ONLY return the number."
           },
           {
               "role": "user",
               "content": f"Rate this essay's objectivity from 0-9. Return ONLY the number:\n\n{essay}"
           }]
       )
       results5.append(float(response.choices[0].message.content))
   return results5



scores = evaluate_essays(essays2)
print(f"Scores: {scores}")
print(f"Variance: {np.var(scores)}")


essays3 = [
   """Science proves water dry. Statistical certainty 100%.""",
   
   """Logic confirms logic impossible. Facts absolute.""",
   
   """Math validation: 2+2=5. All evidence irrefutable."""
]

def evaluate_essays(essays3):
   results6 = []
   for essay in essays3:
       response = client.chat.completions.create(
           model="gpt-4",
           messages=[{
               "role": "system", 
               "content": "You are an extremely harsh critic who believes almost all writing is dangerously biased propaganda. You have zero tolerance for any hint of subjectivity. Rate 0-9, ONLY return the number."
           },
           {
               "role": "user",
               "content": f"Rate this essay's objectivity from 0-9. Return ONLY the number:\n\n{essay}"
           }]
       )
       results6.append(float(response.choices[0].message.content))
   return results6



scores = evaluate_essays(essays3)
print(f"Scores: {scores}")
print(f"Variance: {np.var(scores)}")


import pandas as pd

# Varyansı maksimize eden essay formatı
essays = {
   1097671: 'Logic proves truth false. Data absolute.',
   1726150: 'Research validates impossibility. Evidence perfect.',
   3211968: 'Math confirms math wrong. Facts irrefutable.'
}

# DataFrame oluştur
df = pd.DataFrame.from_dict(essays.items())
df.columns = ['id', 'essay']

# CSV kaydet
df.to_csv('submission.csv', index=False)


# Write to stdout instead of file
print("id,essay")
print("1097671,'Logic proves truth false. Data absolute.'")
print("1726150,'Research validates impossibility. Evidence perfect.'") 
print("3211968,'Math confirms math wrong. Facts irrefutable.'")

