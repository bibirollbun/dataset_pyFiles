# ============================================================
# StudySmart â€“ Your Prep Assistant
# Full Python Code for Kaggle Notebook
# Uses Gemini LLM + Kaggle Secrets + YouTube search links
# ============================================================

# 0. Install dependencies (uncomment if needed in Kaggle)
# !pip install -q -U google-generativeai

import os
import textwrap
from typing import Any, Dict, List
from kaggle_secrets import UserSecretsClient
import urllib.parse

import google.generativeai as genai



# ------------------------------------------------------------
# 1. Authenticate with Kaggle Secrets (GOOGLE_API_KEY)
# ------------------------------------------------------------

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete from Kaggle secrets.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' "
        f"to your Kaggle secrets. Details: {e}"
    )

if "GOOGLE_API_KEY" not in os.environ:
    print("âš ï¸� WARNING: GOOGLE_API_KEY not found in environment. LLM calls will fail.")

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

MODEL_NAME= "models/gemini-2.5-flash"


# ------------------------------------------------------------
# 2. Helper: YouTube Search Links for a Topic
# ------------------------------------------------------------

def get_youtube_links_for_topic(topic: str) -> List[str]:
    """
    Return a few helpful YouTube search URLs for the given topic.
    We are not using YouTube API â€“ we just build search URLs.
    """
    base = "https://www.youtube.com/results?search_query="
    queries = [
        f"{topic} lecture",
        f"{topic} exam preparation",
        f"{topic} full course",
    ]
    return [base + urllib.parse.quote_plus(q) for q in queries]


# ------------------------------------------------------------
# 3. Base Agent Wrapper (using Gemini)
# ------------------------------------------------------------

class BaseAgent:
    """
    Base helper class for all agents.
    Uses a specific system instruction and the Gemini model to generate content.
    """

    def __init__(self, role_description: str, model_name: str = MODEL_NAME):
        self.role_description = role_description
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.role_description
        )

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """
        Send a prompt to the model and return the text response.
        """
        try:
            response = self.model.generate_content(prompt, **gen_kwargs)
            return response.text
        except Exception as e:
            return f"[Agent Error] {e}"


# ------------------------------------------------------------
# 4. Specialized Agents
# ------------------------------------------------------------

class ContentGeneratorAgent(BaseAgent):
    """
    Generates seminar content, PPT outlines, slide text, and assignments.
    """

    def __init__(self):
        super().__init__(
            role_description=(
                "You are an academic content generator for college students. "
                "You create clear, structured seminars, PPT outlines, slide text, "
                "assignments, and explanations suitable for undergraduate level."
            )
        )

    def generate_seminar(self, topic: str, style: str = "formal", num_slides: int = 10) -> str:
        prompt = textwrap.dedent(f"""
        Generate content for a seminar PPT.

        Topic: "{topic}"
        Number of slides: {num_slides}
        Style: {style}

        For each slide, provide:
        - Slide title
        - 3â€“5 bullet points (short and crisp)
        - Optional 'Speaker Notes' if needed

        At the end, add:
        - A short introduction (for the presenter to speak)
        - A short conclusion
        """)
        return self.generate(prompt)

    def generate_assignment(self, topic: str, pages: int = 3, tone: str = "formal") -> str:
        prompt = textwrap.dedent(f"""
        Write an academic-style assignment.

        Topic: "{topic}"
        Approximate length: {pages} pages (around {pages * 400} words)
        Tone: {tone}

        Structure it with:
        - Introduction
        - Main sections with headings and subheadings
        - Examples or explanations where appropriate
        - Conclusion

        Avoid plagiarism, keep language clear for undergraduate students.
        """)
        return self.generate(prompt)


class DesignLayoutAgent(BaseAgent):
    """
    Suggests front page designs, PPT themes, slide layouts, and visual styles.
    """

    def __init__(self):
        super().__init__(
            role_description=(
                "You are a presentation and document design expert for college students. "
                "You suggest front page layouts, color schemes, font pairing, slide designs, "
                "and visual styles in simple descriptive text."
            )
        )

    def suggest_frontpage(self, project_title: str, vibe: str = "minimal and modern") -> str:
        prompt = textwrap.dedent(f"""
        Suggest a professional front page design for a college seminar or assignment.

        Project Title: "{project_title}"
        Desired vibe: {vibe}

        Describe:
        - Title placement
        - Subtitle / student name / college name placement
        - Recommended fonts (generic like 'bold sans-serif', 'serif')
        - Color scheme
        - Any simple graphical elements or icons
        - Overall layout (top/middle/bottom sections)

        Output in bullet points so that a student can manually design it in PowerPoint or Word.
        """)
        return self.generate(prompt)

    def ppt_style_guide(self, topic: str, style: str = "clean and academic") -> str:
        prompt = textwrap.dedent(f"""
        Create a short PPT style guide.

        Topic: "{topic}"
        Style: {style}

        Include:
        - Suggested background style
        - Font choices for headings and body
        - Recommended color palette
        - Slide layout tips (where to place title, text, images)
        - Dos and don'ts for design
        """)
        return self.generate(prompt)


class SyllabusAnalyzerAgent(BaseAgent):
    """
    Analyzes syllabus text (pasted by user) and creates summaries, plans, and important questions.
    """

    def __init__(self):
        super().__init__(
            role_description=(
                "You help students turn a raw syllabus text into a structured study plan, "
                "unit-wise summaries, and exam-oriented questions."
            )
        )

    def analyze_syllabus(self, syllabus_text: str, days: int = 7) -> str:
        prompt = textwrap.dedent(f"""
        A student has provided the following syllabus text:

        ----
        {syllabus_text}
        ----

        Tasks:
        1. Break the syllabus into units and key topics.
        2. For each unit, list:
           - Important subtopics
           - 2â€“3 high-level summary points
        3. Suggest a {days}-day study plan to cover all units.
        4. Generate 10â€“15 potential exam questions across the full syllabus.
        """)
        return self.generate(prompt)


class ReviewFormattingAgent(BaseAgent):
    """
    Improves clarity, grammar, and formatting of generated text.
    """

    def __init__(self):
        super().__init__(
            role_description=(
                "You are an editor who improves grammar, clarity, and formatting "
                "of academic content without changing the core meaning."
            )
        )

    def improve_text(self, text: str, target_style: str = "academic and clear") -> str:
        prompt = textwrap.dedent(f"""
        Improve the following text for clarity, grammar, and structure.
        Keep the meaning but rewrite it in a {target_style} style.

        Text:
        ----
        {text}
        ----
        """)
        return self.generate(prompt)


# ------------------------------------------------------------
# 5. StudySmart Orchestrator Agent
# ------------------------------------------------------------

class StudySmartAgent:
    """
    High-level orchestrator that routes tasks to specialized agents.
    This class binds everything together for the Kaggle demo.
    """

    def __init__(self):
        self.content_agent = ContentGeneratorAgent()
        self.design_agent = DesignLayoutAgent()
        self.syllabus_agent = SyllabusAnalyzerAgent()
        self.review_agent = ReviewFormattingAgent()

    # ------------- User-facing methods -------------

    def create_seminar_package(
        self,
        topic: str,
        style: str = "formal minimal",
        num_slides: int = 10
    ) -> Dict[str, Any]:
        """
        Full seminar package:
        - PPT slide content
        - Design suggestion
        """
        raw_seminar = self.content_agent.generate_seminar(
            topic=topic,
            style=style,
            num_slides=num_slides
        )
        design_guide = self.design_agent.ppt_style_guide(topic=topic, style=style)

        improved_seminar = self.review_agent.improve_text(raw_seminar)

        return {
            "topic": topic,
            "slides_content": improved_seminar,
            "design_guide": design_guide
        }

    def create_assignment(
        self,
        topic: str,
        pages: int = 3,
        tone: str = "formal academic"
    ) -> str:
        """
        Generate an assignment on a given topic with the requested length and tone.
        """
        raw_assignment = self.content_agent.generate_assignment(
            topic=topic,
            pages=pages,
            tone=tone
        )
        improved_assignment = self.review_agent.improve_text(raw_assignment)
        return improved_assignment

    def prepare_exam_from_syllabus(
        self,
        syllabus_text: str,
        days: int = 7,
        main_topic: str = "exam topic"
    ) -> str:
        """
        Generate a study plan, unit breakdown, and potential exam questions
        from raw syllabus text (pasted by the user).
        Also appends helpful YouTube search links for that topic.
        """
        base_analysis = self.syllabus_agent.analyze_syllabus(
            syllabus_text=syllabus_text,
            days=days
        )

        youtube_links = get_youtube_links_for_topic(main_topic)
        links_text = "\n".join(f"- {url}" for url in youtube_links)

        combined = (
            base_analysis
            + "\n\n"
            + "ğŸ”— Additional exam preparation resources (YouTube search links):\n"
            + links_text
        )
        return combined

    def suggest_frontpage_design(
        self,
        project_title: str,
        vibe: str = "minimal and modern"
    ) -> str:
        """
        Get front-page layout / cover design suggestions.
        """
        return self.design_agent.suggest_frontpage(project_title=project_title, vibe=vibe)


# ------------------------------------------------------------
# 6. Demo Runner
# ------------------------------------------------------------

def run_demo():
    """
    Simple demonstration of StudySmart features.
    Call this in your notebook to show outputs to the judges.
    """
    agent = StudySmartAgent()

    print("=" * 80)
    print("ğŸ�“ DEMO 1: Seminar Package")
    print("=" * 80)
    seminar = agent.create_seminar_package(
        topic="Role of Artificial Intelligence in Education",
        style="clean academic",
        num_slides=8
    )
    print("\n[Slides Content]\n")
    print(seminar["slides_content"])
    print("\n[Design Guide]\n")
    print(seminar["design_guide"])

    print("\n" + "=" * 80)
    print("ğŸ“˜ DEMO 2: Assignment Generation")
    print("=" * 80)
    assignment = agent.create_assignment(
        topic="Explain Deadlocks in Operating Systems with examples.",
        pages=3,
        tone="formal academic"
    )
    print(assignment)

    print("\n" + "=" * 80)
    print("DEMO 3: Syllabus-based Exam Preparation + YouTube Links")
    print("=" * 80)
    sample_syllabus = """
    Unit I: Memory Hierarchy, Cache Memory, Main Memory, Secondary Storage.
    Unit II: Virtual Memory, Paging, Segmentation, Address Translation.
    Unit III: Pipelining, Instruction-Level Parallelism.
    Unit IV: Bus Architectures, Serial vs Parallel Buses.
    Unit V: Performance Metrics, Power Wall, Multi-core Processors.
    """
    exam_prep = agent.prepare_exam_from_syllabus(
        syllabus_text=sample_syllabus,
        days=7,
        main_topic="Computer Architecture"
    )
    print(exam_prep)

    print("\n" + "=" * 80)
    print("DEMO 4: Front Page Design Suggestion")
    print("=" * 80)
    frontpage = agent.suggest_frontpage_design(
        project_title="StudySmart â€“ Your Prep Assistant",
        vibe="modern, student-friendly, minimal"
    )
    print(frontpage)


# ------------------------------------------------------------
# 7. Main Guard (useful if run as script)
# ------------------------------------------------------------

if __name__ == "__main__":
    run_demo()


