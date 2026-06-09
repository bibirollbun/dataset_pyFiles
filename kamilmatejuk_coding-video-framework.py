!pip install -q google-adk faiss-cpu sentence-transformers


import io
import os
import uuid
import random
import inspect

import faiss
import sqlite3
import traceback
import numpy as np
from PIL import Image
from pygments import highlight
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
from pygments.formatters import ImageFormatter
from sentence_transformers import SentenceTransformer
from pygments.lexers import get_lexer_by_name, TextLexer

from google.genai import types
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.adk.models.google_llm import Gemini
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.plugins.logging_plugin import LoggingPlugin  
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent

from kaggle_secrets import UserSecretsClient


GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


def xml_prompt(instruction: str, formatting: str, examples: list[str] = None) -> str:
    """
    Format the agent instruction in XML tags.
    Should yield a superior understanding, based on Anthropic research.
    https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags

    Args:
        instruction (str): detailed instruction
        formatting (str): format of correct response
        example (str, optional): example of correct response
    """
    return "\n".join([
        f"<instructions>\n{instruction}</instructions>",
        *(f"<example>\n{example}\n</example>" for example in (examples or [])),
        f"<formatting>{formatting}</formatting>"
    ])


STORAGE_SQLITE_FILENAME = "docs.db"
STORAGE_FAISS_FILENAME = "vector_index.faiss"


if os.path.exists(STORAGE_SQLITE_FILENAME):
    os.remove(STORAGE_SQLITE_FILENAME)
    print(f"Removed SQLite database: {STORAGE_SQLITE_FILENAME}")
if os.path.exists(STORAGE_FAISS_FILENAME):
    os.remove(STORAGE_FAISS_FILENAME)
    print(f"Removed FAISS database: {STORAGE_FAISS_FILENAME}")


# SQLite DB to store texts
STORAGE_SQLITE_CONNECTION = sqlite3.connect("docs.db")
STORAGE_SQLITE_CURSOR = STORAGE_SQLITE_CONNECTION.cursor()
_ = STORAGE_SQLITE_CURSOR.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL
)
""")


# FAISS index
STORAGE_FAISS_EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
if os.path.exists(STORAGE_FAISS_FILENAME):
    STORAGE_FAISS_INDEX = faiss.read_index(STORAGE_FAISS_FILENAME)
else:
    STORAGE_FAISS_INDEX = faiss.IndexFlatL2(384) # MiniLM-L6-v2 dimension = 384
    STORAGE_FAISS_INDEX = faiss.IndexIDMap2(STORAGE_FAISS_INDEX)


def storage_add_document(text: str, summary: str):
    # store text in SQLite
    STORAGE_SQLITE_CURSOR.execute("INSERT INTO documents (text) VALUES (?)", (text,))
    doc_id = STORAGE_SQLITE_CURSOR.lastrowid
    STORAGE_SQLITE_CONNECTION.commit()
    # embed text
    vec = STORAGE_FAISS_EMBEDDING_MODEL.encode([summary], convert_to_numpy=True)
    # store vector in FAISS
    STORAGE_FAISS_INDEX.add_with_ids(vec, np.array([doc_id], dtype=np.int64))
    faiss.write_index(STORAGE_FAISS_INDEX, "vector_index.faiss")


def storage_query_document(text: str, summary: str):
    # embed query
    vec = STORAGE_FAISS_EMBEDDING_MODEL.encode([summary], convert_to_numpy=True)
    # search FAISS
    distances, indices = STORAGE_FAISS_INDEX.search(vec, 3)
    # fetch matched documents from SQLite
    results = []
    for idx in indices[0]:
        idx = int(idx) # FAISS returns numpy.int64 and SQLite expects pythonic int
        if idx == -1: continue
        text_i = STORAGE_SQLITE_CURSOR.execute("SELECT text FROM documents WHERE id=?", (idx,)).fetchone()
        if text_i:
            results.append(text_i[0])
    return results


VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_PADDING = 720, 1280, 50
VIDEO_TIME_S = 5
VIDEO_FPS = 60


!fc-list :spacing=100


VIDEO_FONT = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"


def generate_code_snippet_image(language: str | None, code: str, line_numbers: bool) -> Image:
    formatter = ImageFormatter(
        font_name=VIDEO_FONT,
        font_size=18,
        line_number_bg="#2d2d2d" if line_numbers else None,
        line_numbers=line_numbers,
        style="native",
        image_pad=12,
        line_pad=4,
    )
    lexer = get_lexer_by_name(language) if language else TextLexer()
    img_data = highlight(code, lexer, formatter)
    image = Image.open(io.BytesIO(img_data)).convert("RGBA")
    return image


generate_code_snippet_image("python", """
def greet(*, name="Anna"):
    print(f"Hi, {name}")

greet()
greet("James")
""", True)


def format_answer(answer: str, option_letter: str) -> str:
    # one line
    if "\n" not in answer: return f"{option_letter}) {answer}"
    # multiline
    lines = answer.replace("\t", "    ").split("\n")
    # remove empty lines
    lines = [l for l in lines if l]
    # remove same indent on all lines
    indent = min(len(l) - len(l.lstrip()) for l in lines)
    lines = [l[indent:] for l in lines if l[indent:]]
    # add option letter
    return "\n".join(f"{option_letter}) {l}" if i ==0 else f"   {l}"
                     for i, l in enumerate(lines))

print(format_answer("\"Hi James\"", "A"))
print(format_answer("""
    "Hi Anna"
    "Hi James"
""", "B"))


def format_question(question: str) -> str:
    question = [question]
    while any(len(q) > 30 for q in question):
        segment = question.pop()
        pos = segment.rfind(" ", 0, 30)
        if not pos: break
        question.extend([segment[:pos], segment[pos+1:]])
    return "\n".join(question)

print(format_question("What is the expected result?"))
print(format_question("What is the expected return if we run this code in the terminal on Mac computer?"))


def generate_answers_image(language: str, answers: list[str]) -> Image:
    # generate answer images
    answer_images = []
    for answer, letter in zip(answers, ["A", "B", "C", "D"]):
        answer_images.append(generate_code_snippet_image(
            language,
            format_answer(answer, letter),
            False))
    # find size of resulting image
    max_w = max(img.width for img in answer_images)
    total_h = sum(img.height for img in answer_images)
    padding = int(max_w * 0.1)
    img_w = max_w + padding * 2
    img_h = total_h + padding * (len(answer_images) + 1)
    # generate empty transparent image
    result = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    # add answer images
    y = 0
    for img in answer_images:
        result.paste(img, (padding, y), img)
        y += img.height + padding
    return result

generate_answers_image("python", [
    "\"Hi James\"",
    """
    "Hi Anna"
    "Hi James"
    """,
    """
    "Hi Anna"
    "Hi Anna"
    """,
    "RuntimeError",
])


def scale_image(img: Image, width: int, height: int):
    img_w, img_h = img.size
    # find out scaling factors for vertical and horizontal
    scale_w = width / img_w
    scale_h = height / img_h
    # choose a scaling factor that won't overflow any way
    scale = min(scale_w, scale_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    return img.resize((new_w, new_h), resample=Image.NEAREST)


def generate_frame(language: str, code: str, question: str, answers: list[str]) -> Image:
    # generate
    question_img = generate_code_snippet_image(None, format_question(question), False)
    code_img = generate_code_snippet_image(language, code, True)
    answers_img = generate_answers_image(language, answers)
    # scale
    max_w = VIDEO_WIDTH - 2 * VIDEO_PADDING
    max_h_question = (VIDEO_HEIGHT - 2 * VIDEO_PADDING) * 0.15
    max_h_code = (VIDEO_HEIGHT - 2 * VIDEO_PADDING) * 0.4
    max_h_answers = (VIDEO_HEIGHT - 2 * VIDEO_PADDING) * 0.4
    question_img = scale_image(question_img, max_w, max_h_question)
    code_img = scale_image(code_img, max_w, max_h_code)
    answers_img = scale_image(answers_img, max_w, max_h_answers)
    # combine
    result = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    result.paste(question_img,
                 ((VIDEO_WIDTH - question_img.width) // 2, VIDEO_PADDING),
                 question_img)
    result.paste(code_img,
                 ((VIDEO_WIDTH - code_img.width) // 2, VIDEO_PADDING * 2 + question_img.height),
                 code_img)
    result.paste(answers_img,
                 ((VIDEO_WIDTH - code_img.width) // 2, VIDEO_PADDING * 3 + question_img.height + code_img.height),
                 answers_img)
    return result

# scaling just for visualization purposes
scale_image(generate_frame("python", """
def greet(*, name="Anna"):
    print(f"Hi, {name}")

greet()
greet("James")
""",
"What is the expected return if we run this code in the terminal on Mac computer?",
[
    "\"Hi James\"",
    """
    "Hi Anna"
    "Hi James"
    """,
    """
    "Hi Anna"
    "Hi Anna"
    """,
    "RuntimeError",
]), 400, 600)


def generate_video_background() -> list[Image]:
    speed = 1
    font_size = 20
    font = ImageFont.truetype(VIDEO_FONT, font_size)
    rows = VIDEO_HEIGHT // font_size
    columns = VIDEO_WIDTH // font_size
    letters = [(
        random.randint(0, rows),
        random.randint(0, columns),
    ) for _ in range(100)]

    for _ in range(VIDEO_TIME_S * VIDEO_FPS):
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color="#545454")
        draw = ImageDraw.Draw(img)
        for i in range(len(letters)):
            # Generate a random green character
            char = chr(random.randint(33, 126))
            lx, ly = letters[i]
            draw.text((lx * font_size, ly * font_size), char, font=font, fill=(0, 200, 0))
            # Move letter down
            if ly < rows:
                letters[i] = (lx, ly + speed)
            else:
                letters[i] = (lx, 0)
        yield img


idea_generator_agent = Agent(
    name="IdeaGeneratorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=xml_prompt(
        """
        Generate a coding trivia question, suitable for an interview or quiz, based on the user's prompt.
        If you need to chack some unknown concepts, use `google_search` tool.
        The question should be short, clear, and fit in one line.
        The code block should be clear, formatted, short (maximum 10 lines of code), and in the programming language specified by the user.
        The answer options each should be short and clear (maximum 1 line). Only one answer should be correct.
        """,
        """
        <question>...</question>
        <language>...</language>
        <code>...</code>
        <answers>
            <a>...</a>
            <b>...</b>
            <c>...</c>
            <d>...</d>
            <correct>...</correct>
        </answers>
        """,
        [
            """
            <question>What would happen if we run this code?</question>
            <language>Python</language>
            <code>
            x = "Test"
            x[2] = ""
            print(x)
            </code>
            <answers>
                <a>"Test"</a>
                <b>"Tet"</b>
                <c>SyntaxError</c>
                <d>RuntimeError</d>
                <correct>D</correct>
            </answers>
            """,
            """
            <question>What is the time complexity of this code?</question>
            <language>Python</language>
            <code>
            for i in range(n):
                for j in range(n-i):
                    print(i, j)
            </code>
            <answers>
                <a>O(1)</a>
                <b>O(n)</b>
                <c>O(n^2)</c>
                <d>O(n * log n)</d>
                <correct>C</correct>
            </answers>
            """,
        ]
    ),
    tools=[google_search],
    output_key="idea"
)


def get_similar_ideas(idea: str, summary: str) -> dict:
    """Fetch similar ideas already used from the database.
    Up to 3 closest results will be returned.
    If no idea is within the similarity range, an empty list will be returned.

    Args:
        idea (str): the full generated idea
        summary (str): the simple summary of main concepts of the idea

    Returns:
        Dictionary with status and a list of similar ideas already generated.
        Success: {"status": "success", "similar": ["<question>...", "<question>..."]}
        Error: {"status": "error", "error_message": "Couldn't validate idea"}
    """
    try:
        results = storage_query_document(idea, summary)
        storage_add_document(idea, summary)
        return {"status": "success", "similar": results}
    except Exception as ex:
        print(f"### Exception occured {ex}")
        return {"status": "error", "error_message": f"Couldn't validate idea: {ex}"}


def exit_loop(tool_context: ToolContext) -> dict:
    """Call this function ONLY when the idea is deemed worthy and not used before, signaling the iterative process should end."""
    tool_context.actions.escalate = True
    return {}


idea_checker_agent = Agent(
    name="IdeaCheckerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=xml_prompt(
        """
        Here is a generated idea:
        {{idea}}
    
        Create a summary of the question and answers that could identify this problem type (without specific values, but also not too general).
        Your task is to check if this question idea has already been asked, to avoid repeating.
        When creating a summary, remember:
        - the same question but with different numeric values shouldn't repeat.
        - similar question, but one asking about time complexity and another about space complexity are ok.
        This should give you a clue on how to summarise the generated idea.
        
        To check if the idea was used already, fetch similar used ideas by calling the `get_similar_ideas` tool.
        Based on the analysis of the most similar already used ideas, decide if the current idea is different enough.
        - If the idea is interesting and different from already used, you MUST call the 'exit_loop' function. Do not output any text.
        - Otherwise, you MUST respond with the exact phrase: "EXISTS"
        """,
        """
        "EXISTS" or no response
        """
    ),
    tools=[get_similar_ideas, exit_loop],
    output_key="idea_check_used"
)


idea_refinement_loop_agent = LoopAgent(
    name="IdeaRefinementLoop",
    sub_agents=[idea_generator_agent, idea_checker_agent],
    max_iterations=3,
)


def confirm_idea_by_user(idea: str, tool_context: ToolContext) -> dict:
    """Awaits user approval before continuing with the generated idea.

    Args:
        idea: the generated idea

    Returns:
        Dictionary with approval status
    """

    # This is the first time this tool is called. Await human approval - PAUSE here.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Is the generated idea good enough? {idea}",
            payload={"idea": idea},
        )
        return {"status": "pending", "message": f"Idea awaits approval"}

    # The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.
    if tool_context.tool_confirmation.confirmed:
        return {"status": "approved", "message": f"Idea approved"}
    else:
        return {"status": "rejected", "message": f"Idea rejected"}


human_approval_agent = Agent(
    name="HumanApprovalAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=xml_prompt(
        """
        Here is a generated idea:
        {{idea}}

        You are a generator assistant.
        When you receive the generated idea:
        1. Use the `confirm_idea_by_user` tool with the exact idea
        2. If the status is 'pending', inform the user that approval is required
        3. After receiving the final result, don't respond.
        """,
        """
        No response
        """
    ),
    tools=[confirm_idea_by_user],
)


def generate_video(language: str, code: str, question: str, answers: list[str]) -> dict:
    """Generate an MP4 video featuring a code snippet, a question, and a list of possible answers.

    Example:
    generate_code_video(
        language = "python",
        code = '''
            x = "Test"
            x[2] = ""
            print(x)
        ''',
        question = "What is the result?",
        answers = [
            "Test",
            "Tet",
            "TypeError",
            "RuntimeError",
        ],
    )

    Args:
        language (str): Coding language used in the code snippet.
        code (str): Formatted code snippet.
        question (str): Question regarding the code snippet.
        answers (list[str]): List of 4 answer choices (A, B, C, or D).

    Returns:
        Dictionary with status and path to the file.
        Success: {"status": "success", "video": "generated.mp4"}
        Error: {"status": "error", "error_message": "Video not generated"}
    """
    try:
        content = generate_frame(language, code, question, answers)
        frames = []
        for bg_frame in generate_video_background():
            bg_frame.paste(content, (0, 0), content)
            frames.append(np.array(bg_frame))
        filename = os.path.abspath(f"generated_{uuid.uuid4().hex[:8]}.mp4")
        clip = ImageSequenceClip(frames, fps=VIDEO_FPS)
        clip.write_videofile(filename, codec="libx264", audio=False)
        return {"status": "success", "video": filename}
    except Exception as ex:
        print(f"### Exception occured {ex}")
        return {"status": "error", "error_message": f"Couldn't generate video: {ex}"}


video_generator_agent = Agent(
    name="VideoGeneratorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=xml_prompt(
        """
        Here is a generated idea:
        {{idea}}

        Using the `generate_video` tool, create a video based on the idea.
        Output only the path to the generated video.
        """,
        """
        <path>...</path>
        """,
        ["""
        <path>
        /home/user/Desktop/generated.mp4
        </path>
        """]
    ),
    tools=[generate_video],
    output_key="video_path",
)


video_description_agent = Agent(
    name="VideoDescriptionAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=xml_prompt(
        """
        Here is a generated idea:
        {{idea}}

        Create a video description for the social media short, including programming language, problem description and correct answer explanation.
        Make sure to spark interest, call to action (like, share, comment your answer and explanation) and add hashtags.
        """,
        """
        <description>...</description> 
        """,
        ["""
        Note: the example isn't directly connected to this exact idea, but shows generic example.
        <description>
        Check out this junior interview-level problem in Python.
        The new feature creates new string interpolation type: template string!
        The correct answer is B, because the value <name> gets replaced by "Ann".
        Remember, like and subscribe, share this video with your friends.
        Comment your answer below.
        </description> 
        """]
    ),
    output_key="description",
)
video_description_agent_safe = SequentialAgent(
    name="VideoDescriptionAgentSafeWrapper",
    sub_agents=[video_description_agent]
)


video_partials_parallel_agent = ParallelAgent(
    name="VideoPartialsParallel",
    sub_agents=[video_generator_agent, video_description_agent_safe],
)


def publish_to_yt(video: str, desc: str) -> dict:
    """Publish video to YouTube.

    Args:
        video (str): path to mp4 file
        desc (str): video description

    Returns:
        Dictionary with status.
        Success: {"status": "success"}
        Error: {"status": "error", "error_message": "Video not generated"}
    """
    print(f"Mock pushing to YouTube: {video}\n{desc}")
    return {"status": "success"}


publish_agent = Agent(
    name="PublishAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=xml_prompt(
        """
        Here is video {{video_path}} and description:
        {{description}}

        Publish the video to YouTube, using a `publish_to_yt` tool.
        Respond with status
        """,
        """
        One sentence with status.
        """,
        [
            "Video published correctly!",
            "Video couldn't be published, because of .."
        ],
    ),
    tools=[publish_to_yt],
)


video_generator_pipeline = SequentialAgent(
    name="VideoGeneratorPipeline",
    sub_agents=[idea_refinement_loop_agent, human_approval_agent, video_partials_parallel_agent, publish_agent],
)

video_generator_app = App(
    name="VideoGeneratorApp",
    root_agent=video_generator_pipeline,
    resumability_config=ResumabilityConfig(is_resumable=True),
    plugins=[LoggingPlugin()],
)


session_service = InMemorySessionService()
video_generator_runner = Runner(
    app=video_generator_app,
    session_service=session_service,
)


async def run_with_HITL(query: str):
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    user_id = f"user_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(app_name=video_generator_app.name, user_id=user_id, session_id=session_id)

    print(f"User > {query}\n")

    # Send initial request to the Agent. The Agent returns the special `adk_request_confirmation` event
    events = []
    async for event in video_generator_runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)])
    ):
        events.append(event)

    # Loop through all the events generated and check if `adk_request_confirmation` is present.
    approval_id = None
    invocation_id = None
    for event in events:
        if not event.content: continue
        if not event.content.parts: continue
        for part in event.content.parts:
            if not part.function_call: continue
            if part.function_call.name != "adk_request_confirmation": continue
            approval_id = part.function_call.id
            invocation_id = event.invocation_id
    if approval_id is None or invocation_id is None:
        print("System > adk_request_confirmation function call not found")
        return

    # Handle human approval
    print(f"â�¸ï¸� Pausing for approval...")
    approved = input("Do you approve? [y/n] ").lower() == "y"
    print(f"ğŸ¤” Human Decision: {'APPROVE âœ…' if approved else 'REJECT â�Œ'}\n")
    if not approved:
        return

    # Resume the agent by calling run_async() again with the approval decision
    try:
        async for event in video_generator_runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                id=approval_id,
                name="adk_request_confirmation",
                response={"confirmed": approved},
            ))]),
            invocation_id=invocation_id,  # Critical: same invocation_id tells ADK to RESUME
        ):
            if not event.content: continue
            if not event.content.parts: continue
            for part in event.content.parts:
                if not part.text: continue
                print(f"Agent > {part.text}")
    except* Exception as eg:
        print("=== ROOT EXCEPTIONS FROM TASKGROUP ===")
        for ex in eg.exceptions:
            traceback.print_exception(type(ex), ex, ex.__traceback__)


await run_with_HITL("Simple python code at junior interview level")




