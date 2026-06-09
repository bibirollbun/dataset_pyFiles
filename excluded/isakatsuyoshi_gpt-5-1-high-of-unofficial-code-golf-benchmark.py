!pip install pandas numpy kagglehub openai anthropic google-generativeai


SAMPLE_SIZE: int | None = 20  # Set SAMPLE_SIZE to None to run on the full test set
MODEL = "gpt-5.1"  


from kaggle_secrets import UserSecretsClient
from openai import OpenAI
from google import genai
from anthropic import Anthropic


user_secrets = UserSecretsClient()
if "gemini" in MODEL:
    google_ais_api_key = user_secrets.get_secret("GOOGLE_AIS_API_KEY")
    google_client = genai.Client(api_key=google_ais_api_key)
elif "gpt" in MODEL:
    openai_api_key = user_secrets.get_secret("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
elif "claude" in MODEL:
    anthropic_api_key = user_secrets.get_secret("ANTHROPIC_API_KEY")
    anthropic_client = Anthropic(api_key=anthropic_api_key)
else:
    raise ValueError(f"Invalid model selected: {MODEL}.")





from pathlib import Path
import zipfile
import random
import json
import os


# 入出力ディレクトリの設定
input_dir = Path("/kaggle/input/shortest-codes")
json_dir = Path("/kaggle/input/google-code-golf-2025")
output_dir = Path("/kaggle/working/submission")
output_dir.mkdir(exist_ok=True)
success_dir = Path("/kaggle/working/success")
success_dir.mkdir(exist_ok=True)
failure_dir = Path("/kaggle/working/failure")
failure_dir.mkdir(exist_ok=True)
zip_path = Path("/kaggle/working/submission.zip")

# ファイル一覧をランダムに取得する関数
def sample_task_files(n: int) -> list[Path]:
    all_files = sorted(input_dir.glob("task*.py"))
    return all_files[:n]

def sample_json_files(n: int) -> list[Path]:
    all_files = sorted(json_dir.glob("task*.json"))
    return all_files[:n]

def generate_gemini(model_name, prompt, sys = ""):
    response = google_client.models.generate_content(model=model_name, contents=f'{sys} {prompt}')
    return response.text

def generate_gpt(model_name, prompt, sys = ""):
    response = openai_client.responses.create(
        model=model_name,
        reasoning={"effort": "high"},
        input=[
            {
                "role": "developer",
                "content": sys
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.output_text

def generate_claude(model_name, prompt, sys = ""):
    messages=[{"role": "user", "content": prompt}]
    system_msg = sys if len(sys) > 0 else None

    message = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=system_msg,
        messages=messages,
    )
    return message.content[0].text


def generate(model_name: str, user_prompt: str, system_prompt: str = "") -> str:
    if "gemini" in model_name:
        return generate_gemini(model_name, user_prompt, system_prompt)
    elif "gpt" in model_name:
        return generate_gpt(model_name, user_prompt, system_prompt)
    elif "claude" in model_name:
        return generate_claude(model_name, user_prompt, system_prompt)
    else:
        raise ValueError(f"Invalid model selected for PROMPT_engine: {model_name}.")

# コード短縮を行う関数
def shorten_code(original_code: str, model: str) -> str:
    user_prompt = f"""
You are a code golf assistant.
Shorten the following Python function `p`, preserving its behavior exactly.
The code may be a `def p(g): ...` or `p = lambda g: ...` style.
Only output the shortened function code. Do NOT add explanations or formatting.

Here is the original code:

{original_code}

# Shortened version:"""
    system_prompt = "You are a Python code golf expert."

    response = generate(model, user_prompt, system_prompt).strip()
    print(response)
    # コードブロックが含まれる場合に取り除く
    if response.startswith("```"):
        lines = response.splitlines()
        lines = [line for line in lines if not line.startswith("```")]
        response = "\n".join(lines).strip()
    return response

# コードの正当性を確認する関数（defでもlambdaでも可）
def is_valid_python_function(code: str, json_data: dict) -> bool:
    try:
        namespace = {}
        exec(code, namespace)
        p = namespace.get("p")
        if not callable(p):
            return False
        examples = json_data.get('train', []) + json_data.get('test', []) + json_data.get('arc-gen', [])
        for ex in examples:
            example_copy = copy.deepcopy(ex)
            result = p(example_copy["input"])
            result = json.dumps(result)
            result = result.replace("true", "1").replace("false", "0")
            unsafe_chars = re.compile(r"[^0-9,\[\]\s\.]")
            if unsafe_chars.search(result):
                raise ValueError(f"Invalid output from user code: {result[:500]}")
            result = json.loads(result)
            user_output = np.array(result)
            label_output = np.array(example_copy["output"])
            if not np.array_equal(user_output, label_output):
                return False
        return True
    except Exception:
        return False

# メイン処理関数
def main(sample_size: int):
    for task_path, json_path in zip(sample_task_files(sample_size), sample_json_files(sample_size)):
        print(task_path, json_path)
        json_data = json.loads(json_path.read_text())
        original_code = task_path.read_text(encoding="utf-8")
        try:
            shortened_code = shorten_code(original_code, MODEL)
        except Exception as e:
            print(e)
            shortened_code = ""

        if len(shortened_code) < len(original_code):
            if is_valid_python_function(shortened_code, json_data):
                (success_dir / task_path.name).write_text(shortened_code, encoding="utf-8")
                print(f"✅短縮成功 {task_path.name}: {len(original_code)} → {len(shortened_code)} bytes")
            else:
                (failure_dir / task_path.name).write_text(shortened_code, encoding="utf-8")
                print(f"❌動作不良 {task_path.name}")
        else:
            (failure_dir / task_path.name).write_text(shortened_code, encoding="utf-8")
            print(f"❌短縮不可 {task_path.name}: {len(original_code)} → {len(shortened_code)} bytes")
            



# 実行
main(sample_size=SAMPLE_SIZE)








# ベースのShortestコードデータセット＋successしたコードをsubmissionディレクトリに保存
!cp /kaggle/input/shortest-codes/* submission
!cp success/* submission/


import zipfile
from pathlib import Path
zip_path = Path("/kaggle/working/submission.zip")

with zipfile.ZipFile(zip_path, "w") as zipf:
    for file in Path('submission').glob("*.py"):
        zipf.write(file, arcname=file.name)







