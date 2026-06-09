!curl -fsSL https://ollama.com/install.sh | sh


import subprocess
process = subprocess.Popen("ollama serve", shell=True)


pull = subprocess.Popen("ollama pull gpt-oss:20b", 
                        shell=True, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.STDOUT)


!pip install ollama


import json
from ollama import chat

HISTORY_FILE = 'messages_1.json'

try:
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        messages = json.load(f)
except FileNotFoundError:
    messages = []

def save_messages(messages):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

def append_message(role, content):
    messages.append({'role': role, 'content': content})
    save_messages(messages)

def chat_with_streaming_and_think(user_input, think_level='low'):
    append_message('user', user_input)
    assistant_response = ''
    for part in chat(
        'gpt-oss:20b',
        messages=messages,
        think=think_level,
        stream=True,
    ):
        chunk = part['message']['content']
        print(chunk, end='', flush=True)
        assistant_response += chunk
    print()
    append_message('assistant', assistant_response)

if __name__ == '__main__':
    while True:
        user_input = input('You: ').strip()
        if not user_input:
            break
        chat_with_streaming_and_think(user_input, think_level='low')

