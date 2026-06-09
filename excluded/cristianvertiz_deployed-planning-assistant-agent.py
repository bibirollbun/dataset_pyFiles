import os

import vertexai
from vertexai import agent_engines

from kaggle_secrets import UserSecretsClient


user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


PROJECT_ID="project-52f530b5-d60c-4b3c-bd0"
deployed_region="us-central1"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

# os.environ["GOOGLE_API_KEY"] = None
# os.environ.pop('GOOGLE_API_KEY', None)

# Initialize Vertex AI
vertexai.init()

# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"✅ Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("❌ No agents found. Please deploy first.")


async for item in remote_agent.async_stream_query(
    message="What is my name?",
    user_id="cvertiz",
):
    print(item)


async for item in remote_agent.async_stream_query(
    message="Can you show me the issues related to improve documentation that are slow-moving issues?",
    user_id="user_42",
):
    print(item)

