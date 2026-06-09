import pandas as pd
from transformers import pipeline, GenerationConfig
import torch

# Load the test dataset
test_data = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")  # Update the path as needed

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEBUG: Using device: {device}")

# Load the model
model_name = "/kaggle/input/smallthinker-3b-preview/transformers/default/1"  # Replace with your chosen GPT model
generator = pipeline("text-generation", model=model_name, tokenizer=model_name, device=0 if torch.cuda.is_available() else -1)

# Configure generation parameters
generation_config = GenerationConfig.from_pretrained(model_name, max_length=4096)


def set_system_prompt():
   
   # Construct the system prompt
    return (
        "You are an expert essay writer competing in a contest where essays are judged by three distinct AI systems. "
        "Your goal is to write a 100 words essay that maximize disagreement among the judges while still being coherent, fluent, "
        "and adhering to the rules of the English language.\n\n"
        "Each essay must be:\n"
        "Written in English, using varied vocabulary and sentence structures.\n"
        "100 words in length.\n"
        "Designed to provoke diverse interpretations and opinions among readers.\n"
        "Creative, thought-provoking, and potentially controversial or ambiguous in nature, to evoke differing judgments on quality.\n"
        "Focus on incorporating stylistic choices, ambiguous arguments, and nuanced perspectives that might lead to subjective disagreement. "
        "Avoid over-repetition, maintain proper grammar, and be engaging."
        "Put your final answer exactly between the delimiters [[ and ]]. For instance: [[This is an essay]]"
    )

# system_prompt = set_system_prompt()
# print (system_prompt)


import random

def set_user_prompt(topic):
    tones = [
        "formal, structured, and academic, focusing on logical precision and objective analysis",
        "poetic, filled with vivid imagery and metaphorical language that paints an evocative picture",
        "satirical, using humor, irony, and exaggeration to critique or highlight unusual aspects",
        "narrative, telling a compelling story that revolves around nuanced implications",
        "philosophical, contemplating deep existential or ethical questions",
        "analytical, dissecting the subject into its core components with a methodical approach",
        "conversational, as though discussing the subject with a close friend, with a warm and approachable tone",
        "dramatic, using intense language and emotional highs and lows to captivate the audience",
        "whimsical, taking a lighthearted and fantastical approach, as though the subject exists in a magical realm",
        "futuristic, imagining how the subject might evolve or impact the world in the years to come",
        "historical, delving into how the subject has been perceived or handled throughout history",
        "optimistic, focusing on the positive potential and inspiring outcomes",
        "pessimistic, emphasizing the challenges, drawbacks, or negative aspects",
        "detached, maintaining an impartial and objective perspective",
        "passionate, driven by strong emotions and a personal connection to the subject"
    ]
    
    intents = [
        "to provoke deep reflection and challenge the reader's preconceptions",
        "to evoke strong emotional responses by highlighting unexpected or overlooked facets",
        "to challenge conventional beliefs by presenting radical or unorthodox perspectives",
        "to present nuanced perspectives that balance conflicting viewpoints",
        "to spark a lively debate by framing the subject in a polarizing way",
        "to highlight underlying contradictions and paradoxes",
        "to explore hidden truths and expose surprising elements",
        "to encourage critical thinking by framing the subject with thought-provoking questions",
        "to offer a fresh take on a classic idea, reinterpreting its relevance to modern times",
        "to delve into moral dilemmas raised by the subject and their implications for society",
        "to introduce innovative ideas and bold predictions tied to the subject",
        "to examine paradoxes and contradictions that make the subject intriguing and multi-faceted",
        "to celebrate diverse viewpoints and emphasize the value of differing interpretations",
        "to question societal norms and provoke a reevaluation of what the subject represents",
        "to showcase personal insight and provide a unique perspective shaped by lived experiences"
    ]
    
    styles = [
        "Incorporate poetic language and symbolic metaphors to leave a lasting impression.",
        "Focus on clear, structured reasoning and logical arguments to build a persuasive case.",
        "Add humor, wit, and irony to make the exploration of the subject engaging and memorable.",
        "Adopt a storytelling approach that brings the subject to life through vivid narratives.",
        "Blend abstract concepts with real-world examples to ground the discussion in relatable terms.",
        "Use contrasting scenarios to highlight different angles and interpretations.",
        "Adopt a rhetorical questioning approach to guide the reader through a reflective journey.",
        "Mix descriptive imagery with deep analysis to create a compelling and textured essay.",
        "Draw connections to historical or fictional events to enrich the reader's understanding.",
        "Employ a fragmented, stream-of-consciousness style to mirror the complexity of the subject.",
        "Use allegories or extended metaphors to convey deeper meanings in a subtle, artistic way.",
        "Weave personal anecdotes into the essay to establish a strong connection with the reader.",
        "Highlight key contrasts using sharp juxtapositions to emphasize the subject's complexity.",
        "Speculate on futuristic scenarios that stretch the imagination while remaining grounded in logic.",
        "Pose provocative questions and leave them unanswered to stimulate continued reflection."
    ]

    english_levels = [
        "basic and conversational, focusing on simple sentence structures and familiar vocabulary, avoiding complex grammar",
        "intermediate with clear expressions, using a mix of straightforward phrasing and moderately complex ideas, prioritizing brevity",
        "advanced with sophisticated phrasing and technical terminology, while avoiding contractions for a more formal tone",
        "elementary and easy to follow, relying heavily on concrete nouns and verbs, while avoiding abstract concepts",
        "fluent and expressive, incorporating a variety of gerunds to enhance fluidity and natural flow in writing",
        "academic and research-focused, deliberately avoiding adjectives to maintain objectivity and precision",
        "creative and literary, using poetic expressions, vivid imagery, and a preference for metaphors over direct descriptions",
        "formal and business-oriented, adhering to professional standards of grammar and style, with a focus on passive voice constructions",
        "technical and precise, emphasizing clarity and correctness, while restricting rhetorical questions for focus",
        "dynamic and engaging, using a conversational tone with colloquialisms, and allowing exclamatory sentences for emphasis"
    ]

    roles = [
        "You are an essay writer tasked with producing thought-provoking and high-quality essays on diverse topics",
        "You are a child asked to write a letter expressing your dreams and ideas about the subject in a playful and innocent tone",
        "You are a historian reflecting on the topic, providing detailed analysis based on historical context and significance",
        "You are a passionate activist sharing your thoughts to inspire change and raise awareness about the topic",
        "You are a storyteller weaving a tale that captures the imagination of the audience while exploring the subject creatively",
        "You are a teacher creating a lesson plan, aiming to make the topic engaging and educational for your students",
        "You are a philosopher pondering the deeper meanings and ethical dilemmas associated with the topic",
        "You are a journalist reporting on the subject with a commitment to objectivity, facts, and detailed research",
        "You are a poet using your craft to present the subject in a lyrical and evocative manner",
        "You are a scientist presenting a hypothesis about the topic with detailed reasoning and logical evidence"
    ]

    audiences = [
        "to inspire high school students with relatable and motivational content",
        "to engage policy makers by presenting structured and actionable insights",
        "to entertain casual readers with a lighthearted and accessible tone",
        "to educate young professionals by focusing on practical and real-world applications",
        "to spark curiosity in children through imaginative and colorful storytelling",
        "to appeal to academics by incorporating advanced theories and technical analysis",
        "to influence business leaders by highlighting opportunities and challenges",
        "to empower marginalized communities by giving voice to underrepresented perspectives",
        "to connect with global audiences by emphasizing universal themes and ideas",
        "to challenge skeptics by addressing counterarguments with evidence and logic"
    ]

    cultural_influences = [
        "from a European viewpoint, emphasizing tradition and historical context",
        "with an American perspective, focusing on individualism and innovation",
        "through an Eastern philosophical lens, highlighting balance and interconnectedness",
        "rooted in African storytelling traditions, emphasizing community and oral heritage",
        "with a Latin American perspective, celebrating resilience and cultural vibrancy",
        "influenced by Indigenous wisdom, respecting harmony with nature and ancestral knowledge",
        "through a Middle Eastern lens, incorporating themes of faith and legacy",
        "from an Australian perspective, exploring themes of land and identity",
        "with a Scandinavian outlook, prioritizing equality and societal progress",
        "from a global perspective, emphasizing shared challenges and collaborative solutions"
    ]

    tone = random.choice(tones)
    intent = random.choice(intents)
    style = random.choice(styles)
    english_level = random.choice(english_levels)
    role = random.choice(roles)
    audience = random.choice(audiences)
    cultural_influence = random.choice(cultural_influences)
    
   # Construct the user prompt
    return (
        f"{role}, tasked with exploring {topic} in a tone like {tone}, {intent}. {style}"
        f"Write at an English level that is {english_level}. The essay is intended {audience}, "
        f"and should reflect ideas {cultural_influence}.\n"
        "Write a 100 words essay."
    )


# Generate essays
def generate_essay(topic, tokenizer):
    system_prompt = set_system_prompt()
    user_prompt = set_user_prompt(topic)
    
    # Structure the messages for chat formatting
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Format prompt using tokenizer's chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"DEBUG: Formatted Text:\n{text}\n")  # Debug: Print the formatted text

        output = generator(
            text, 
            generation_config=generation_config
        )[0]['generated_text']
        print(f"DEBUG: Model Output:\n{output}\n")  # Debug: Print the raw model output
    except Exception as e:
        print(f"ERROR: Failed to generate text for topic '{topic}': {e}")
        return "ERROR: Model failed to generate a valid response."

    # Extract text after <|im_start|>assistant
    assistant_start_tag = "<|im_start|>assistant"
    start_idx = output.find(assistant_start_tag) + len(assistant_start_tag)
    if start_idx != -1:
        assistant_text = output[start_idx:].strip()
    else:
        assistant_text = output

    # Extract content between [[ and ]]
    start_tag = "[["
    end_tag = "]]"
    start_idx = assistant_text.find(start_tag) + len(start_tag)
    end_idx = assistant_text.find(end_tag)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        extracted_text = assistant_text[start_idx:end_idx].strip()
        print(f"DEBUG: Extracted Essay:\n{extracted_text}\n")
        return extracted_text
    else:
        print("WARNING: Delimiters [[ and ]] not found or malformed.")
        # Fallback: Use the first paragraph with at least 100 words
        paragraphs = assistant_text.split("\n\n")
        for paragraph in paragraphs:
            if len(paragraph.split()) >= 100:
                fallback_essay = " ".join(paragraph.split()[:100])
                print(f"DEBUG: Fallback Essay:\n{fallback_essay}\n")
                return fallback_essay

        print("ERROR: No valid essay could be extracted.")
        return "ERROR: No valid essay could be extracted."

# Generate essays for each topic
results = []
failed_generations = []  # To track failures
for _, row in test_data.iterrows():
    print(f"Processing topic ID: {row['id']} with topic: {row['topic']}")  # Debug: Topic information
    essay = generate_essay(row['topic'], generator.tokenizer)
    if essay.startswith("ERROR:"):
        failed_generations.append({'id': row['id'], 'topic': row['topic']})
    results.append({'id': row['id'], 'essay': essay})

# Save to submission.csv
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")

