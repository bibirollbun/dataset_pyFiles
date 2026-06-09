# Step 1: Clone a specific branch (e.g., gpt5pro)
import os, shutil

if os.path.exists('arc-lang-public'):
    print("Directory 'arc-lang-public' exists. Removing it.")
    shutil.rmtree('arc-lang-public')

!git clone --depth 1 --single-branch --branch gpt5c https://github.com/jerber/arc-lang-public.git arc-lang-public


import os
os.chdir('arc-lang-public')


!uv pip install -e .


# import the env variables
import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

os.environ['XAI_API_KEY'] = user_secrets.get_secret("XAI_API_KEY")

os.environ["OPENAI_API_KEY"] = user_secrets.get_secret("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = ''
os.environ["DEEPSEEK_API_KEY"] = ''
os.environ["OPENROUTER_API_KEY"] = ''
os.environ["GEMINI_API_KEY"] = 'temp'

os.environ['MAX_CONCURRENCY'] = "1000"

try:
    os.environ['NEON_DSN'] = user_secrets.get_secret("NEON_DSN")
    print('neondb exists')
except Exception:
    print('no neon')

try:
    os.environ['LOGFIRE_TOKEN'] = user_secrets.get_secret("LOGFIRE_TOKEN")
    print('logfire exists')
except Exception:
    print('no logfire')
    os.environ['LOCAL_LOGS_ONLY'] = "1"
    os.environ['USE_TASK_ID'] = "1"


from src.run import run_from_json
from pathlib import Path
from src.configs.grok_configs import grok_config_prod
from src.configs.gpt5pro_configs import gpt5pro_config_prod

year = "2025"
train_or_eval = "evaluation"
root_dir = Path()

# TODO: replace this with the path to the ARC challenges
challenges_path = root_dir / "data" / f"arc-prize-{year}" / f"arc-agi_{train_or_eval}_challenges.json"

# TODO: replace this with the path you want the attempts to save to. This is where the formatted attemps solutions will go.
attempts_path = root_dir / 'attempts' / f"arc-prize-{year}" / f"arc-agi_{train_or_eval}_attempts.json"

# this is a temp directory to store solutions as they are created in case the run stops abruptly
# so the solutions that have already been created are saved here
temp_attempts_path = root_dir / "attempts" / f"arc-prize-{year}" / "temp_solutions"

await run_from_json(
    challenges_path=challenges_path,
    truth_solutions_path=None,
    # config=grok_config_prod,
    config=gpt5pro_config_prod,
    attempts_path=attempts_path,
    temp_attempts_dir=temp_attempts_path,
    limit=1,
    offset=0,
)


from src.run import evaluate_solutions

# TODO: replace this with the solutions path to evaluate the attempts against the ground truth solutions
solutions_path = root_dir / "data" / f"arc-prize-{year}" / f"arc-agi_{train_or_eval}_solutions.json"

# evaluates the solutions given the ground truth solutions json
evaluate_solutions(
    attempts_solutions_path=attempts_path,
    truth_solutions_path=solutions_path,
)

