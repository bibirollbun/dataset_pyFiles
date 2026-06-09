!pip install litellm -q 


import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ['GROQ_API_KEY'] = user_secrets.get_secret("GROQ_API_KEY")



old_print = print


import re
os.environ['LITELLM_LOG'] = 'ERROR'
import litellm

model = 'groq/Qwen-2.5-Coder-32b'

def get_response(prompt):
    response = litellm.completion(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=1000,
    temperature=0.0,
    seed=42,
    stop = ['```output'],
)
    response_text = response.choices[0].message.content
    code_regex = r'```python(.*?)```'
    code_match = re.findall(code_regex, response_text, re.DOTALL)[-1]
    return code_match


def print(*args, **kwargs):
    global result
    if type(args[0]) == int:
        result = (args[0] % 1000)
    else:
        old_print(*args, **kwargs)


prompt = '''Fred and George take part in a tennis tournament with 4046 other players. In each round, the
 players are paired into 2024 matches. How many ways are there to arrange the first round such
 that Fred and George do not have to play each other? (Two arrangements for the first round are
 different if there is a player with a different opponent in the two arrangements.)
Write efficient python code.'''

code_match = get_response(prompt)

if code_match:
    try:
        exec(code_match)
        old_print(result)
    except Exception as e:
        print(f"Error executing code: {e}")
else:
    print("No code found in the response.")



prompt = r'''For a positive integer $n$, let $S(n)$ denote the sum of the digits of $n$ in base 10. Compute $S(S(1)+S(2)+\cdots+S(N))$ with $N=10^{100}-2$.

Write a python program.'''

code_match = get_response(prompt)

if code_match:
    try:
        exec(code_match)
        old_print(result)
    except Exception as e:
        print(f"Error executing code: {e}")
else:
    print("No code found in the response.")


prompt = r'''For positive integers $x_1,\ldots, x_n$ define $G(x_1, \ldots, x_n)$ to be the sum of their $\frac{n(n-1)}{2}$ pairwise greatest common divisors. We say that an integer $n \geq 2$ is \emph{artificial} if there exist $n$ different positive integers $a_1, ..., a_n$ such that 
\[a_1 + \cdots + a_n = G(a_1, \ldots, a_n) +1.\]
Find the sum of all artificial integers $m$ in the range $2 \leq m \leq 40$.

Write efficient python program using recursion correctly without using combinations, lru_cache, permutations. Enhance the Arbitrary upper limit for search'''

code_match = get_response(prompt)
print(code_match)
print('---'*25)
if 'Arbitrary upper limit for search' in code_match:
    code_match = get_response('\n\n'.join((prompt,code_match, 'Just modify the arbitrary upper limit to dynamic values.')))
print(code_match)                              
if code_match:
    try:
        exec(code_match)
        old_print(result)
    except Exception as e:
        print(f"Error executing code: {e}")
else:
    print("No code found in the response.")

