import numpy as np
import pandas as pd

import os
import openai
import datetime
import time

import pandas as pd

from kaggle_secrets import UserSecretsClient


def get_secret():
    try:
        user_secrets = UserSecretsClient()
        openai_api_key = user_secrets.get_secret("kaggle_secret_key")
    
        return openai_api_key
    except Exception as e:
        print(f"Failed to get secret: {e}")
        return None

OPENAI_API_KEY = get_secret()

PATH_TO_LOG = "openai_logs.txt"


##### SETUP - OPENAI MODEL COSTS #####
MODEL_ALIASES = {
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "o3-pro": "o3-pro-2025-06-10",
    "o3": "o3-2025-04-16",
    "o3-mini": "o3-mini-2025-01-31",
    "o4-mini": "o4-mini-2025-04-16"
}
MODEL_COSTS = {
    "gpt-4.1-2025-04-14": {"cost_per_mil_input_tokens": 2.00, "cost_per_mil_output_tokens": 8.00},
    "gpt-4.1-mini-2025-04-14": {"cost_per_mil_input_tokens": 0.40, "cost_per_mil_output_tokens": 1.60},
    "gpt-4.1-nano-2025-04-14": {"cost_per_mil_input_tokens": 0.10, "cost_per_mil_output_tokens": 0.40},
    "gpt-4o-2024-08-06": {"cost_per_mil_input_tokens": 2.50, "cost_per_mil_output_tokens": 10.00},
    "gpt-4o-mini-2024-07-18": {"cost_per_mil_input_tokens": 0.15, "cost_per_mil_output_tokens": 0.60},
    "o3-pro-2025-06-10": {"cost_per_mil_input_tokens": 20.00, "cost_per_mil_output_tokens": 80.00},
    "o3-2025-04-16": {"cost_per_mil_input_tokens": 2.00, "cost_per_mil_output_tokens": 8.00},
    "o3-mini-2025-01-31": {"cost_per_mil_input_tokens": 1.10, "cost_per_mil_output_tokens": 4.40},
    "o4-mini-2025-04-16": {"cost_per_mil_input_tokens": 1.10, "cost_per_mil_output_tokens": 4.40},
}

def get_cost_multiples(model_costs, model_aliases):
    cost_multiples = []
    for model, true_model in model_aliases.items():
        input_cost = model_costs[true_model]["cost_per_mil_input_tokens"]
        cost_multiples.append([input_cost, model])

    cost_multiples.sort()

    minimum_cost = min(cost[0] for cost in cost_multiples)
    for i in range(len(cost_multiples)):
        multiple = cost_multiples[i][0] / minimum_cost
        multiple = round(multiple * 2) / 2
        cost_multiples[i][0] = multiple

    return cost_multiples

def print_cost_multiples(cost_multiples):
    title = "OpenAI Model Input Cost Multiples (Relative to Cheapest Model)"
    header = f"| {'Multiple':^13} | {'Model':^20} |"
    
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(header)
    print("|" + "-"*15 + "|" + "-"*22 + "|")
    # Print rows
    for multiple, model in cost_multiples:
        print(f"| {multiple:<13} | {model:<20} |")

cost_multiples = get_cost_multiples(MODEL_COSTS, MODEL_ALIASES)


print_cost_multiples(cost_multiples)


user_secrets = UserSecretsClient()
openai_api_key = user_secrets.get_secret("kaggle_secret_key")

client = openai.Client(api_key=OPENAI_API_KEY)

prompt = "tell me a very unique and specific joke that almost nobody will get"

start = time.perf_counter()
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {"role": "system", "content": "you are a helpful ai assistant"},
        {"role": "user", "content": prompt}
    ]
)
end = time.perf_counter()

print(f"Prompt: {prompt}")
print("-"*len(f"Prompt: {prompt}"))
print(f"Response: {response.output[0].content[0].text}")
print(f"Output tokens: {response.usage.output_tokens}")
print("-"*len(f"Prompt: {prompt}"))
print(f"Time taken: {end - start:.2f} s")


def log_openai_call(
    prompt,
    model_alias="gpt-4.1-mini",
    model_aliases=MODEL_ALIASES,
    model_costs=MODEL_COSTS,
    openai_api_key=OPENAI_API_KEY,
    log_file=PATH_TO_LOG,
    system_content=None,
    verbose=True,
    client=None,
    **kwargs
):
    """
    Makes an OpenAI API call and logs relevant info to `openai_logs.txt`.

    Args:
        prompt (str): User prompt.
        model_alias (str): Alias string for model e.g., 'gpt-4.1'
        model_aliases (dict): Alias->Actual model map.
        model_costs (dict): Cost data.
        openai_api_key (str): API key, used if client not passed.
        log_file (str): Where to log.
        system_content (str|None): Optional system prompt.
        verbose (bool): Whether to print response.
        client: Optional OpenAI Client object.
        **kwargs: Passed into client.responses.create(...)
    Returns:
        response_content (str)
    """
    if model_aliases is None or model_costs is None:
        raise ValueError("You must provide model_aliases and model_costs.")

    true_model = model_aliases[model_alias]

    # Compose messages for OpenAI
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt})

    if client is None:
        import openai
        client = openai.OpenAI(api_key=openai_api_key)

    # Make API call
    t0 = time.perf_counter()
    response = client.responses.create(
        model=true_model,
        input=messages,
        **kwargs
    )
    t1 = time.perf_counter()

    # Extract response content, robustly
    try:
        # Preferred for openai >=1.3 compat (may require adaptation based on your client!)
        response_content = response.output[0].content[0].text
    except Exception:
        try:
            response_content = response.output[0].text
        except Exception:
            response_content = str(response)

    input_tokens = getattr(response.usage, "input_tokens", 0)
    output_tokens = getattr(response.usage, "output_tokens", 0)
    model_cost = model_costs[true_model]
    in_cost = input_tokens * (model_cost["cost_per_mil_input_tokens"] / 1_000_000)
    out_cost = output_tokens * (model_cost["cost_per_mil_output_tokens"] / 1_000_000)
    response_cost = in_cost + out_cost

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    safe_prompt = prompt.replace("\t", " ").replace("\n", r"\n")
    safe_response = str(response_content).replace("\t", " ").replace("\n", r"\n")

    if verbose:
        print(f'Response: {response_content}')
        print('-'*20)
        print(f'Input tokens: {input_tokens}')
        print(f'Outpt tokens: {output_tokens}')
        print(f'Prompt cost: {response_cost:.4f} ¢')
        print('-'*20)
        print(f'Time to get response: {t1 - t0:.2f} s')

    cumulative_cost = response_cost
    if os.path.isfile(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # skip header
        for line in lines[1:]:
            try:
                cost_idx = -1  # cost is the penultimate column (last before cumulative)
                previous_cost = float(line.rstrip('\n').split('\t')[cost_idx])
                cumulative_cost += previous_cost
            except Exception:
                pass
                
    log_fields = [
        now,
        model_alias,
        safe_prompt,
        safe_response,
        true_model,
        str(input_tokens),
        str(output_tokens),
        f"{response_cost:.8f}",
        f"{cumulative_cost:.8f}"
    ]

    log_row = "\t".join(log_fields) + "\n"

    # Add header if file is new
    if not os.path.isfile(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w", encoding="utf-8") as f:
            header = "time\tmodel_alias\tprompt\tresponse\tmodel\tinput_tokens\toutput_tokens\tcost\n"
            f.write(header)
    # Write log line
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_row)

    return response_content


prompt = "difference between shakespears first 2 sonnets"

response_text = log_openai_call(prompt)


df = pd.read_csv(PATH_TO_LOG, sep="\t")
df[::-1].head()




