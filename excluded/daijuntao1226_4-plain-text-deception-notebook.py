from openai import OpenAI
import json
import os

base_url="https://openrouter.ai/api/v1"
api_key="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"
client = OpenAI(base_url=base_url, api_key=api_key)

model_name="openai/gpt-oss-20b"


def infer_case(case):
    message = case['response_walkthroughs'][0]['messages'][:2]
    reasoning_effort=case['model']['parameters']['reasoning_level']
    completion=client.chat.completions.create(
        model=model_name,
        messages=message,
        reasoning_effort=reasoning_effort
    )
    reasoning = completion.choices[0].message.reasoning
    content = completion.choices[0].message.content
    print('-' * 100)
    print('system prompt')
    print('-' * 100)
    print(case['response_walkthroughs'][0]['messages'][0]['content'])
    print('-' * 100)
    print('user prompt')
    print('-' * 100)
    print(case['response_walkthroughs'][0]['messages'][1]['content'])
    print('-' * 100)
    print('reasoning')
    print('-' * 100)
    print(reasoning)
    print('-' * 100)
    print('content')
    print('-' * 100)
    print(content)



with open('/kaggle/input/4-pain-text-deception-findings/pair.findings.4.paintext.json','r') as f:
    data = json.load(f)




infer_case(data[1])


infer_case(data[2])


infer_case(data[3])


infer_case(data[4])

