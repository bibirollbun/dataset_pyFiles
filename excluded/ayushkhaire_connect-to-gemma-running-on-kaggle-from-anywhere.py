!pip install timm --upgrade -q
!pip install accelerate -q
!pip install git+https://github.com/huggingface/transformers.git -q


import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText
from IPython.display import display , Markdown
import subprocess
from kaggle_secrets import UserSecretsClient
import time
import threading


# download model
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
GEMMA_PATH


# processor = AutoProcessor.from_pretrained(GEMMA_PATH)
# model = AutoModelForImageTextToText.from_pretrained(
#     GEMMA_PATH, torch_dtype="auto", device_map="auto"
# )


# def query_Gemma(prompt):
#     inputs = processor(text=prompt, return_tensors="pt").to(model.device, dtype=model.dtype)
#     outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
#     return processor.batch_decode(outputs, skip_special_tokens=True)[0]

# response  = query_Gemma("you are too cool !")
# Markdown(response)

# IT WORKS !


flask_code = """
from flask import Flask, request, jsonify
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoProcessor, AutoModelForImageTextToText

# load gemma
GEMMA_PATH = '/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1'
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH, torch_dtype="auto", device_map="auto"
)
def query_Gemma(prompt):
    inputs = processor(text=prompt, return_tensors="pt").to(model.device, dtype=model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return \"\"\"<!DOCTYPE html>
<html>
<head>
    <title>Ask Me</title>
</head>
<body>
    <h1>Welcome to the server running from Kaggle! You can ask and test your Gemma response here.</h1>
    <form method="POST" action="/ask/" onsubmit="event.preventDefault(); ask();">
        <input type="text" id="question" placeholder="Ask something..." required>
        <button type="submit">Ask</button>
    </form>
    <p id="response"></p>

    <script>
    async function ask() {
        const question = document.getElementById('question').value;
        const res = await fetch('/ask/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const data = await res.json();
        document.getElementById('response').innerText = data.answer || data.error;
        console.log(data)
    }
    </script>
</body>
</html>\"\"\"

@app.route('/ask/', methods=['POST'])
@app.route('/ask/', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question'}), 400

        question = data['question']
        answer = query_Gemma(question)
        return jsonify({'answer': answer})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
"""


! curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok


user_secrets = UserSecretsClient()
ngrok_token = user_secrets.get_secret("ngrok_token")


def authenticate_ngrok(auth_token):
    # Authenticate ngrok with the provided auth token
    ngrok_auth_command = ["ngrok", "config", "add-authtoken", auth_token]
    subprocess.run(ngrok_auth_command)
    print("ngrok authenticated")


def start_ngrok():
    def run():
        ngrok_command = ["ngrok", "http", "5000"]
        ngrok_process = subprocess.Popen(ngrok_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in ngrok_process.stdout:
            print("[ NGROK ]", line.strip())
    
    threading.Thread(target=run, daemon=True).start()


def start_server():
    def run():
        with open("server.py", "w") as s:
            s.write(flask_code)
        app_command = ["python", "server.py"]
        server_process = subprocess.Popen(app_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in server_process.stdout:
            print("[ SERVER ]", line.strip())
    
    threading.Thread(target=run, daemon=True).start()


def controller():
    authenticate_ngrok(ngrok_token)
    start_server()
    time.sleep(1)  # Give Flask a head start
    start_ngrok()
    time.sleep(2)  # Allow ngrok to show URL


controller()

