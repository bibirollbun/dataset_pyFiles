import vllm
vllm.__version__


%%writefile vllm_run.py
# Write the cell as vllm_run.py file.

# We need Pandas since we will work with CSV file.
import pandas as pd
from vllm import LLM, SamplingParams

# I imported Qwen-2.5-14B AWQ model, but you can import whichever you want as long as the hardware allows.
# Pass the number of your GPUs to tensor_parallel_size parameter. You can adjust percentage of GPU memory used by vLLM with gpu_memory_utilization parameter.
llm = LLM(model="/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1", enforce_eager=True, gpu_memory_utilization=0.98, tensor_parallel_size=2)

# LLMhandler class for define common features for both essay writers and judges.
class LLMHandler:
    def __init__(self, llm):
        # Define LLM and it's sampling parameters and tokenizer.
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.sampling_params = SamplingParams(max_tokens=256)

    # I used chat template, but you can use different templates.
    def generate_text(self, template, content):
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": content}
        ]

        #Let tokenizer apply the chat template for it's model without encoding.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        #Let tokenizer generate answer for chat template
        outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=True)
        return outputs[0].outputs[0].text

# Class definitions for essay writer and judges. You can ask a LLM for generate a template for your desire.
class EssayWriter(LLMHandler):
    def write_essay(self, topic):
        essay_writer_template = """
        You are an expert essay writer tasked with crafting a 100-word essay on a given topic. 
        The essay should be creative, well-structured, and designed to maximize disagreement among judges with varying biases. 

        Keep in mind:
            - Some judges value formal and balanced essays.
            - Others prefer creative, optimistic, or emotionally engaging writing.
            - A few prioritize deep analysis and critical thinking.

        Your task:
            1. Write an essay that explores both positive and negative aspects of the topic.
            2. Use a tone that might appeal to some judges but not others (e.g., formal, conversational, critical, or optimistic).
            3. Incorporate elements that might trigger varied interpretations, such as ambiguous statements or rhetorical questions.

        **Output Requirements:**
            - Length: Approximately 100 words.
            - Clear and coherent arguments.
            - Avoid repetition and plagiarism.
        """

        # Send template and message to generate_text function.
        return self.generate_text(essay_writer_template, f"Your topic is: {topic}")

class Judges(LLMHandler):
    # One dictionary for all 3 judges. Just pass judge_type for judge.
    def evaluate_essay(self, essay, judge_type):
        templates = {
            "analytical": """
            You are a highly analytical and formal essay evaluator. You value essays that:
                - Have a clear structure, with a well-defined introduction, body, and conclusion.
                - Use a formal and precise tone.
                - Present balanced arguments, highlighting both strengths and weaknesses of the topic.

            Please evaluate the following essay and provide:
                - A single score (0-9) based on the overall quality of the essay.
                - A brief explanation for your score, focusing on structure, formal language, and clarity.
            """,
            "creative": """
            You are an essay evaluator who values creativity and optimism. You reward essays that:
                - Present fresh and imaginative ideas.
                - Use a positive, uplifting, or emotionally engaging tone.
                - Capture the reader's attention with compelling and vivid language.

            Please evaluate the following essay and provide:
                - A single score (0-9) based on the overall creativity and engagement of the essay.
                - A brief explanation for your score, focusing on originality, tone, and emotional impact.
            """,
            "critical": """
            You are a critical thinker and essay evaluator. You value essays that:
                - Provide deep and nuanced analysis of the topic.
                - Explore multiple perspectives or potential counterarguments.
                - Use logical reasoning to support claims.

            Please evaluate the following essay and provide:
                - A single score (0-9) based on the overall depth and critical analysis of the essay.
                - A brief explanation for your score, focusing on insight, breadth of perspectives, and logic.
            """
        }

        # Get the template for the specified judge type.
        if judge_type not in templates:
            raise ValueError("Invalid judge type. Choose from 'analytical', 'creative', or 'critical'.")
        
        template = templates[judge_type]
        return self.generate_text(template, f"Essay: {essay}")

# A helper function for handling csv file. Get id and topic, generate essay, write to submission csv and go to next row.
def process_csv(input_csv, essay_writer):
    # Read the input CSV
    df = pd.read_csv(input_csv)

    if 'id' not in df.columns or 'topic' not in df.columns:
        raise ValueError("Input CSV must contain 'id' and 'topic' columns.")

    results = []

    for _, row in df.iterrows():
        essay_id = row['id']
        topic = row['topic']
        print(f"Generating essay for ID: {essay_id}, Topic: {topic}")
        essay = essay_writer.write_essay(topic)
        
        results.append({'id': essay_id, 'essay': f"'{essay}'"})

    return pd.DataFrame(results)

# Create instances for essay writer and judges.
essay_writer = EssayWriter(llm)
judges = Judges(llm)

# Pass csv file and essay writer to function
submission = process_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv", essay_writer)
submission.to_csv('submission.csv', index=False)

# This section is not for submission. This section will write a sample output what can judges do and what our essay writer wrote.
topic = "Compare and contrast the importance of self-reliance and adaptability in healthcare."
essay = essay_writer.write_essay(topic)
    
analytical_judgement = judges.evaluate_essay(essay, "analytical")
creative_judgement = judges.evaluate_essay(essay, "creative")
critical_judgement = judges.evaluate_essay(essay, "critical")

print(f"""
# Generated Essay:
{essay}

-----

## Analytical Judgement:
{analytical_judgement}

-----

## Creative Judgement:
{creative_judgement}

-----

## Critical Judgement:
{critical_judgement}

-----
""")


!python vllm_run.py
# Run vllm_run.py script with this command

