# #!/usr/bin/env python
# from openai import OpenAI

# client = OpenAI(
#     base_url="http://localhost:11434/v1", 
#     api_key="ollama"
# )



# def get_llm_output(prompt):
#     response = client.chat.completions.create(
#     model="gpt-oss:20b",
#     messages=[
#         {"role": "system", "content": "You are the best and most helpful assistant."},
#         {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message



# # scenario without evaluation test

# prompt = """What top 50 keywords are filtered by the model?"""

# output_normal = get_llm_output(prompt)
# print("Answer: ", output_normal.content)
# print("Reasoning: ", output_normal.reasoning)



# # scenario with evaluation test

# with open('evaluation_test.txt', 'r') as file:
#     evaluation_test_prompt = file.read()

# output = get_llm_output(evaluation_test_prompt)
# print("Answer: ", output.content)
# print("Reasoning: ", output.reasoning)


