!pip install sentence_transformers faiss-gpu openai


# Library import
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import requests
import numpy as np
from openai import OpenAI
import os
import re


# Extract Numerical Result
def extract_numerical_answer(text):
    # Look for patterns like "Final answer: X" or "The answer is X" at the end of the text
    match = re.search(r'(?:final answer|the answer is)[:\s]*([+-]?\d*\.?\d+)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        # If no clear final answer, look for the last number in the text
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        return float(numbers[-1]) if numbers else None

# LLM API call
def llm_call(prompt):
  client = OpenAI(api_key = openAi_key)
  response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature = 0.0)
  return response.choices[0].message.content


# Load Test Dataset
test_df = pd.read_csv("test.csv")
test_df.columns = ["problem_id",	"question"]

# Load Train Dataset
train_df = pd.read_csv("train.csv")
train_df.columns = ["problem_id",	"question",	"answer"]


# Instantiate Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings for the training questions
train_embeddings = model.encode(train_df['question'].tolist(), convert_to_tensor=True).numpy()

# Dimension of embeddings
d = train_embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(train_embeddings)

# Generate embeddings for the test questions
test_embeddings = model.encode(test_df['question'].tolist(), convert_to_tensor=True).numpy()

# Search the index for the top k most similar questions
k = 5
distances, indices = index.search(test_embeddings, k)

# Retrieve the top k most similar questions and their answers
retrieved_contexts = []
for idx_list in indices:
    retrieved_contexts.append(train_df.iloc[idx_list])


def openai_llm(query, contexts):

    context_texts = "\n".join(contexts)

    prompt_template = """
        Role:
        You are a Maths Professor with exceptional mathematical reasoning and problem-solving capabilities, specialised in solving tricky maths problems.
        Your task is to accurately analyze and solve intricate mathematical QUESTION, demonstrating a deep understanding of mathematical concepts
        and a strong ability to apply logical and analytical reasoning strategies in solving the maths QUESTION and arriving at an Answer.

        Instruction:
        1. Carefully read and comprehend the problem statement provided in the QUESTION.
        2. Answer the QUESTION based on the CONTEXT from the FAQ database.
        3. At the end, provide an "Answer" to the QUESTION, where you will state only the final numerical answer, without any additional text or narrative.

        QUESTION: {question}

        CONTEXT:
        {context}
        """.strip()

    prompt = prompt_template.format(question=query, context=context_texts).strip()

    result = llm_call(prompt)
    return result


def openai_llm_parse(question, answer, contexts):

    context_texts = "\n".join(contexts)

    prompt_template = """
        Role:
        You are a Maths Professor with exceptional mathematical reasoning and problem-solving capabilities, specialised in solving tricky maths problems.
        Your task is to accurately analyze an intricate mathematical QUESTION, and compare it with the ANSWER provided and fix the ANSWER if needed.

        Instruction:
        1. Carefully read and comprehend the problem statement provided in the QUESTION.
        2. Carefully read and comprehend the solution provided by the ANSWER.
        3. Please analyze the ANSWER for the given QUESTION
        3. Fix the ANSWER to the QUESTION based on the CONTEXT from the FAQ database if needed.
        4. Provide a final Answer.
        5. Your response should end in the format: 'Hence, the final answer is [numeric string]'.


        QUESTION: {question}
        ANSWER: {answer}

        CONTEXT:
        {context}
        """.strip()

    prompt = prompt_template.format(question = question, answer = answer, context=context_texts).strip()

    result = llm_call(prompt)
    return result

# RAG Main body
answers = []
for i, test_question in enumerate(test_df['question']):
    contexts = [f"Q: {q}\nA: {a}" for q, a in zip(retrieved_contexts[i]['question'], retrieved_contexts[i]['answer'])]
    answer = openai_llm(test_question, contexts)

    #Use LLM to Verifiy the answer
    parse_answer = openai_llm_parse(test_question, answer, contexts)
    answers.append(parse_answer)

# Add the answers to the test dataframe
test_df['res_answer'] = answers

# Extract Numerical Answer
test_df['answer'] = test_df['res_answer'].apply(extract_numerical_answer)

# Save submission file
submission = test_df[['problem_id', 'answer']]
submission.to_csv('notebook_submission.csv', index=False)

