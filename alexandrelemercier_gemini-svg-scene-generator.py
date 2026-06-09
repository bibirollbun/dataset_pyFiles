!pip install svgwrite


########################################
# STEP 0: Install and Import Dependencies
########################################

!pip uninstall -qqy jupyterlab  # Optionally remove conflicting packages
!pip install -U -q "google-genai==1.7.0"


from google import genai
from google.genai import types
from google.api_core import retry

from IPython.display import Markdown, display
from tqdm.notebook import tqdm

import enum

####################################################
# STEP 1: Configure Authentication and Retry Logic
####################################################

from kaggle_secrets import UserSecretsClient
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)


##############################################
# 0. Setup and Retry Logic
##############################################

def is_retriable(e):
    """Retry on 429 (Too Many Requests) or 503 (Service Unavailable)."""
    return (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

# Patch the generate_content method so it automatically retries on certain errors.
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable
)(genai.models.Models.generate_content)

##############################################
# 1. Generate Clarifying Questions + Self-Answers
##############################################

def auto_generate_and_answer_questions(scene_description: str, client: genai.Client) -> str:
    """
    1) Ask Gemini to generate clarifying questions for the scene.
    2) Then *in the same prompt*, instruct Gemini to also guess the best answers
       (because the user is not allowed to provide input).
    3) Finally, produce a refined scene description with those guesses.

    This function returns a single text that includes the refined scene.
    """
    prompt = f"""
You are an advanced model tasked with creating extremely detailed SVG scenes from a single textual prompt.

**Scene**: "{scene_description}"

Step 1: Propose a set of clarifying questions about time of day, weather, style, perspective, color palette, objects, etc.
Step 2: Since no user input is allowed beyond the scene itself, you must guess or assume the most likely answers to those questions.
Step 3: Merge your assumed answers into a single refined "Scene Description." 
Step 4: Output only:
  - The questions with your guessed answers
  - Then the final refined scene description.

Format Example:
```
Questions + Self-Answers:
1) Q: ...
   A: ...
2) Q: ...
   A: ...
... etc.

Refined Scene Description:
(Your best attempt at a fully clarified scene)
```
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7
        )
    )
    return response.text.strip()

##############################################
# 2. Generate Initial SVG from Refined Scene
##############################################

def generate_initial_svg(refined_scene: str, client: genai.Client) -> str:
    """
    Uses the refined scene description (with assumed clarifications)
    to create a highly detailed, multi-layered SVG. 
    Includes a short chain-of-thought plus <svg> output.

    This code is GENERIC and does not reference any specific color or object.
    """
    # Embedding "system instructions" in the main prompt for clarity
    system_instructions = """
You are an advanced generative model that creates extremely precise SVG images from scene descriptions.

**Key Points**:
1. Segment the scene into layers: background, middle ground, foreground.
2. Respect any color or size references from the refined scene. 
3. For animals or complex objects, break them into logically placed shapes (body, head, limbs, etc.).
4. Use a consistent coordinate system so objects appear logically placed.
5. Place comments so that a reviewer knows which element corresponds to which part of the scene.
6. Output:
   - A "Reasoning:" section describing how you approached layering, color usage, object geometry.
   - Then a valid <svg> with as many shapes/groups as needed for rich detail (~100 elements if relevant).
"""

    prompt = f"""
[Refined Scene Description]
{refined_scene}

[Task]
1. Produce a short "Reasoning:" summary on how you plan to layer, color, and shape the scene.
2. Then output the <svg> code in full, with layered <g> groups for background, middle ground, and foreground.
3. The final <svg> must be valid, with a rich level of detail.

{system_instructions}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7
        )
    )
    return response.text.strip()

##############################################
# 3. Iterative Improvement
##############################################

def improve_svg(svg_code_with_reasoning: str, scene_description: str, client: genai.Client) -> str:
    """
    Ask Gemini to refine layering, colors, detail, and object consistency
    given the previous output (chain-of-thought + <svg>).
    """
    prompt = f"""
Below is your previous output:

```
{svg_code_with_reasoning}
```

The scene you want to describe is "{scene_description}".

Please improve it. Focus on:
1. Ensuring all user-described (or guessed) colors, objects, and positions are consistent.
2. If animals are present, ensure the body parts are logically placed.
3. If necessary, refine layering so that background is truly behind the middle ground, etc.
4. If all previous checks passed successfully, rather focus on increasing detail.
5. Return a new version in the same format:
```
Reasoning:
...

<svg>
   ...
</svg>
```
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7        )
    )
    return response.text.strip()

##############################################
# 4. Full Pipeline
##############################################

def generate_scene_svg_pipeline(scene_description: str, do_refinement_iterations: int = 1) -> str:
    """
    1) Gemini automatically generates clarifying questions + guesses answers
    2) We parse out the refined scene description from that text
    3) We create an initial SVG
    4) We refine the SVG for 'do_refinement_iterations' times
    5) Return the final chain-of-thought + <svg>

    Usage:
        final_output = generate_scene_svg_pipeline("A goose walking in a park under a blue mountain.", 2)
    """
    print(f"=== Scene Description ===\n{scene_description}\n")

    # Step 1) Let Gemini produce questions & guess answers => final refined text
    print("=== Gemini: Generating clarifications & self-answers ===")
    questions_and_answers = auto_generate_and_answer_questions(scene_description, client)
    print(questions_and_answers, "\n")

    # Attempt to extract the refined scene from Gemini's text.
    # We'll do a simple parse looking for "Refined Scene Description:"
    refined_scene_text = "No refined description found."
    lines = questions_and_answers.split("\n")
    capturing = False
    captured_lines = []
    for line in lines:
        if line.strip().startswith("Refined Scene Description:"):
            capturing = True
            continue
        if capturing:
            captured_lines.append(line)
    if captured_lines:
        refined_scene_text = "\n".join(captured_lines).strip()

    if not refined_scene_text:
        refined_scene_text = scene_description  # fallback

    print("=== Refined Scene Description (Guessed) ===")
    print(refined_scene_text)
    print()

    # Step 2) Generate the initial SVG
    print("=== Generating Initial SVG ===")
    initial_svg = generate_initial_svg(refined_scene_text, client)
    print(initial_svg)
    print()

    # Step 3) Iteratively refine
    svg_code_with_reasoning = initial_svg
    for i in range(do_refinement_iterations):
        print(f"--- Iteration {i+1} of Refinement ---")
        improved_svg = improve_svg(svg_code_with_reasoning, scene_description, client)
        print(improved_svg)
        print()
        svg_code_with_reasoning = improved_svg

    # Return final chain-of-thought + <svg>
    return svg_code_with_reasoning


%%time

####################################################
# Usage Example:
####################################################

svg_codes = list()
scenes = ["A goose walking in a park under a blue mountain.",
          "A purple whale swimming in the Pacific ocean",
          "Three apples on a napkin with blue circular patterns",
          "A goose wearing a gold medal"]

for k, scene in enumerate(tqdm(scenes)):
    scene_desc = scenes[k]
    final_svg = generate_scene_svg_pipeline(scene_desc, do_refinement_iterations=2)
    svg_code = final_svg.split("```")[-2][4:-1]
    svg_codes.append(svg_code)
    # To display the final answer
    # Markdown(final_svg)


for k, svg_code in enumerate(svg_codes):
    print(scenes[k])
    try:
        display(SVG(svg_code))
    except:
        print("Error")

