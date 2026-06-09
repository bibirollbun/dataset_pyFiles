
# pip install google-adk openai pandas numpy scikit-learn matplotlib seaborn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ADK & Gemini
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types







from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("âœ… Gemini API key configured")



import os
from openai import OpenAI

# Get OpenAI API key from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
OPENAI_API_KEY = UserSecretsClient().get_secret("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Create OpenAI client
client = OpenAI()



retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429,500,503,504]
)



data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
print("Dataset loaded:", data.shape)
data.head()



# Identify categorical and numeric columns
categorical_cols = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
numeric_cols = ['math score','reading score','writing score']

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Create derived feature: average_score
data['average_score'] = data[numeric_cols].mean(axis=1)



# Features & target
target_col = 'average_score'
features = categorical_cols + numeric_cols

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_col], test_size=0.2, random_state=42)

# Scale numeric columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])



rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Feature importance
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

feature_importances



root_agent = Agent(
    name="smart_study_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A feature-driven agent that helps students improve study habits and predict performance.",
    instruction="""
    You are an AI study assistant.
    1. Use student features to predict performance.
    2. Give feature-driven advice.
    3. If unsure, use Google Search to get accurate info.
    """,
    tools=[google_search],
)



runner = InMemoryRunner(agent=root_agent)
student_memory = []



def get_feature_recommendation(student_features, predicted_score):
    top_features = feature_importances['feature'].head(3).tolist()
    prompt = f"""
    Student features: {student_features}
    Predicted average score: {predicted_score:.2f}
    Important features: {top_features}
    Give 3 actionable study recommendations.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content



# Select a student
example_student = X_test.iloc[0].to_dict()
pred_score = rf_model.predict(X_test_scaled.iloc[[0]])[0]

student_memory.append({
    "features": example_student,
    "predicted_score": pred_score
})

# Agent prompt
user_query = f"Give me advice to improve performance for this student: {example_student}"
response = await runner.run_debug(user_query)

# LLM feature-based recommendation
# Prompt Gemini to give actionable study recommendations
prompt = f"""
You are an educational advisor AI agent.
Given the student features: {example_student} 
and their predicted exam score: {pred_score},
give 3 actionable study recommendations to improve performance.
Provide your response in clear, concise bullet points.
"""

# Query the Gemini agent using your ADK runner
response = await runner.run_debug(prompt)

# Display the response
print("Gemini Agent Response:\n", response)


print("Agent Response:\n", response)
print("Gemini Agent Recommendation:\n", response)


# Keep track of recommendations for this session
student_memory.append({
    "features": example_student,
    "predicted_score": pred_score,
    "recommendation": response
})



# Feature importance visualization
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()

# Predicted vs Actual
plt.scatter(y_test, rf_model.predict(X_test_scaled))
plt.plot([0,100],[0,100], color='red', linestyle='--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual Scores")
plt.show()



!adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY




!adk web sample-agent --port 8000



from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

def get_adk_proxy_url(port=8000):
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]
    path_parts = baseURL.split("/")
    kernel = path_parts[2]
    token = path_parts[3]

    url_prefix = f"/k/{kernel}/{token}/proxy/{port}"
    url = f"{PROXY_HOST}{url_prefix}"
    display(HTML(f"<a href='{url}' target='_blank'>Open ADK Web UI â†—</a>"))

get_adk_proxy_url(port=8000)



