from IPython.display import Image, display
display(Image("/kaggle/input/image-intro/intro.png"))



import os
import json
import time
import uuid
import subprocess
import warnings

import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as tv_models
from PIL import Image

from google import genai
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

warnings.filterwarnings("ignore")

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



# Retry configuration for Gemini HTTP calls 
retry_config = types.HttpRetryOptions(
    attempts=5,                # max attempts
    exp_base=7,                # backoff multiplier
    initial_delay=1,           # seconds
    http_status_codes=[429, 500, 503, 504],
)

print(" !! Retry config created.")



import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # ... define your layers here ...
        self.fc1 = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # ... define your forward pass ...
        x = torch.relu(self.fc1(x))
        return self.output(x)

# Instantiate the model
model = MyModel()


# 1. Define the path to your .pth file
PATH = '/kaggle/input/resnet-dr-prediction-model/pytorch/default/1/ResNet_best_model.pth'

# 2. Load the state dictionary from the file
state_dict = torch.load(PATH)

model.load_state_dict(state_dict, strict=False)

# 4. Set the model to evaluation mode 
model.eval()


# Preprocessing (must match training)
mean_values = [0.485, 0.456, 0.406]
std_values  = [0.229, 0.224, 0.225]

dr_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_values, std_values),
])

IDX_TO_CLASS = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe",
]


def get_dr_prediction(image_path: str) -> str:
    """
    Tool: predict DR type for a fundus image.

    Args:
        image_path: Path to a retinal fundus image.

    Returns:
        Text description with predicted class and probabilities.
    """
    img = Image.open(image_path).convert("RGB")
    img_t = dr_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = dr_model(img_t)
        probs = F.softmax(logits, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    label = IDX_TO_CLASS[idx.item()]
    confidence = conf.item()

    prob_str = ", ".join(
        f"{cls}: {probs[i].item():.4f}"
        for i, cls in enumerate(IDX_TO_CLASS)
    )

    return (
        f"Predicted DR type: {label} (confidence: {confidence:.4f}). "
        f"Class probabilities â†’ {prob_str}"
    )


print(" !! get_dr_prediction() tool ready.")



dr_prediction_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="dr_prediction_agent",
    description=(
        "External DR prediction agent that runs a trained ResNet model "
        "to classify diabetic retinopathy severity from fundus images."
    ),
    instruction="""
    You are a diabetic retinopathy specialist agent.

    When a user provides a fundus image path, call the get_dr_prediction tool
    to classify the image into DR categories (Mild, Moderate, No_DR,
    Proliferate_DR, Severe). Return the predicted type and confidence.
    Always remind that this is an assistive tool and not a medical diagnosis.
    """,
    tools=[get_dr_prediction],
)

print(" !! dr_prediction_agent created.")



import os, signal

try:
    dr_prediction_server_process.terminate()
    dr_prediction_server_process.wait(timeout=5)
    print("Old DR server process terminated.")
except Exception as e:
    print("No existing DR server process or already stopped:", e)



import textwrap

MODEL_PATH = "/kaggle/input/resnet-dr-prediction-model/pytorch/default/1/ResNet_best_model.pth"

server_code = f"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5
MODEL_PATH = "{MODEL_PATH}"

class ResNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ResNet, self).__init__()
        self.model_name = model_name
        resnet = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, input):
        features = self.features(input)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

dr_model = ResNet('ResNet', NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
dr_model.load_state_dict(state_dict)
dr_model.to(device)
dr_model.eval()

mean_values = [0.485, 0.456, 0.406]
std_values  = [0.229, 0.224, 0.225]

dr_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_values, std_values),
])

IDX_TO_CLASS = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe",
]

def get_dr_prediction(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    img_t = dr_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = dr_model(img_t)
        probs = F.softmax(logits, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    label = IDX_TO_CLASS[idx.item()]
    confidence = conf.item()

    prob_str = ", ".join(
        f"{{cls}}: {{probs[i].item():.4f}}"
        for i, cls in enumerate(IDX_TO_CLASS)
    )

    return (
        f"Predicted DR type: {{label}} (confidence: {{confidence:.4f}}). "
        f"Class probabilities â†’ {{prob_str}}"
    )

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

dr_prediction_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="dr_prediction_agent",
    description="External DR prediction agent that classifies retinal images.",
    instruction="Use get_dr_prediction(image_path) to classify diabetic retinopathy.",
    tools=[get_dr_prediction],
)

app = to_a2a(dr_prediction_agent, port=8001)
"""

with open("/tmp/dr_prediction_server.py", "w") as f:
    f.write(textwrap.dedent(server_code))

print(" !! DR server code written to /tmp/dr_prediction_server.py")



import subprocess, time, requests, os

server_process = subprocess.Popen(
    [
        "uvicorn",
        "dr_prediction_server:app",
        "--host", "localhost",
        "--port", "8001",
    ],
    cwd="/tmp",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},
)

print(" Starting DR Prediction Agent server...")
print(" Waiting for server to be ready...")

for attempt in range(30):
    try:
        r = requests.get("http://localhost:8001/.well-known/agent-card.json", timeout=1)
        if r.status_code == 200:
            print("\n !! DR Prediction Agent server is running!")
            break
    except requests.exceptions.RequestException:
        time.sleep(2)
        print(".", end="", flush=True)
else:
    print("\n Server did not respond in time.")
    stdout = server_process.stdout.read().decode()
    stderr = server_process.stderr.read().decode()
    print("\n--- SERVER STDOUT ---\n", stdout)
    print("\n--- SERVER STDERR ---\n", stderr)

dr_prediction_server_process = server_process



from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH

remote_dr_agent = RemoteA2aAgent(
    name="dr_prediction_agent",
    description=(
        "Remote diabetic retinopathy prediction agent that classifies "
        "fundus images into DR severity levels via A2A."
    ),
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("!! Remote DR Agent proxy created!")
print(f"   Agent card URL: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")



from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types  


doctor_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="doctor_agent",
    description=(
        "A doctor assistant that helps interpret diabetic retinopathy "
        "predictions from a remote DR model."
    ),
    instruction="""
    You are a friendly and professional doctor support agent.

    When a doctor provides a retinal fundus image path and asks about
    diabetic retinopathy:

    1. ALWAYS use the dr_prediction_agent sub-agent to classify the image.
       The sub-agent returns DR type (Mild, Moderate, No_DR, Proliferate_DR, Severe)
       and confidence values.

    2. Explain the result in simple, clear medical terms:
       - What the predicted severity means.
       - That further clinical evaluation is required.

    3. If the image path seems invalid or not related to a fundus image,
       say that you cannot reliably classify it.

    4. Always remind the user that this is a research/assistive tool and
       NOT a final medical diagnosis.
    """,
    sub_agents=[remote_dr_agent],
)

print(" Doctor_agent created!")
print("   Sub-agent: remote_dr_agent (A2A â†’ DR prediction server)")



import uuid
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import warnings
warnings.filterwarnings("ignore")   # <-- Disable ALL warnings

async def test_a2a_communication(user_query: str):
    """
    Test the A2A communication between doctor_agent and remote_dr_agent.

    Flow:
      doctor_agent â†’ remote_dr_agent (A2A) â†’ DR model â†’ back to doctor_agent.
    """
    session_service = InMemorySessionService()

    app_name = "doctor_support_app"
    user_id = "demo_doctor"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    # Create a new session
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Runner to manage agent + session
    runner = Runner(
        agent=doctor_agent,
        app_name=app_name,
        session_service=session_service,
    )

    # Build user message
    content = types.Content(
        parts=[types.Part(text=user_query)]
    )

    print(f"\n User: {user_query}")
    print("\n Doctor Agent response:")
    print("-" * 60)

    # Run the agent asynchronously & stream final response
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)

    print("-" * 60)



from PIL import Image
import matplotlib.pyplot as plt

# ---------- 1. Display the fundus image ----------
image_path = "/kaggle/input/test1-jpg/Mild_22.jpg"

img = Image.open(image_path)
plt.figure(figsize=(4,4))
plt.imshow(img)
plt.title("Input Fundus Image")
plt.axis("off")
plt.show()

image_path = "/kaggle/input/test1-jpg/Mild_22.jpg"

query = (
    f"Here is the retinal image path: '{image_path}'. "
    f"What is the diabetic retinopathy severity for this patient?"
)

await test_a2a_communication(query)


