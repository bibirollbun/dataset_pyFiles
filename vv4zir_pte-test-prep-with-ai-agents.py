import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from kaggle_secrets import UserSecretsClient

try:
    api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("âœ… Gemini API key setup complete.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# Model configuration
genai.configure(api_key=api_key)

# Model configuration
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
model = genai.GenerativeModel(MODEL_NAME)

print("âœ… Setup complete!")
print(f"ğŸ“… Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"ğŸ¤– Model: {MODEL_NAME}")
print(f"ğŸ”‘ API Key: {'Configured' if api_key else 'Missing'}")


# PTE Scoring Rubrics (Embedded)
PTE_RUBRICS = {
    "speaking": {
        "read_aloud": {
            "fluency": {
                "description": "Smooth delivery with minimal pauses and hesitations",
                "scoring": {
                    "5": "Perfect fluency with no pauses or hesitations",
                    "4": "Good fluency with minimal pauses",
                    "3": "Acceptable fluency with some pauses",
                    "2": "Frequent pauses affecting comprehension",
                    "1": "Many pauses and hesitations, difficult to follow"
                }
            },
            "pronunciation": {
                "description": "Clear pronunciation with correct stress and intonation",
                "scoring": {
                    "5": "Native-like pronunciation with perfect clarity",
                    "4": "Clear pronunciation with minor errors",
                    "3": "Generally clear with some pronunciation issues",
                    "2": "Pronunciation errors affect comprehension",
                    "1": "Many pronunciation errors, difficult to understand"
                }
            },
            "content": {
                "description": "Accurate reading of the text without omissions or additions",
                "scoring": {
                    "5": "Perfect accuracy, all words correct",
                    "4": "High accuracy with minor errors",
                    "3": "Mostly accurate with some errors",
                    "2": "Several errors affecting meaning",
                    "1": "Many errors, significant meaning loss"
                }
            }
        },
        "describe_image": {
            "fluency": {
                "description": "Smooth delivery with natural pace",
                "scoring": {
                    "5": "Perfect fluency, natural pace",
                    "4": "Good fluency with minor hesitations",
                    "3": "Acceptable fluency",
                    "2": "Frequent hesitations",
                    "1": "Many hesitations, difficult to follow"
                }
            },
            "pronunciation": {
                "description": "Clear pronunciation",
                "scoring": {
                    "5": "Native-like pronunciation",
                    "4": "Clear with minor errors",
                    "3": "Generally clear",
                    "2": "Errors affect comprehension",
                    "1": "Many errors, difficult to understand"
                }
            },
            "content": {
                "description": "Accurate description of image elements",
                "scoring": {
                    "5": "Comprehensive and accurate description",
                    "4": "Good description with minor omissions",
                    "3": "Adequate description",
                    "2": "Incomplete or inaccurate description",
                    "1": "Very limited or incorrect description"
                }
            }
        },
        "repeat_sentence": {
            "content": {
                "description": "Accurate repetition of the sentence",
                "scoring": {
                    "5": "Perfect repetition",
                    "4": "Minor errors",
                    "3": "Some errors",
                    "2": "Many errors",
                    "1": "Incorrect or incomplete"
                }
            }
        }
    },
    "writing": {
        "essay": {
            "content": {
                "description": "Relevance and development of ideas",
                "scoring": {
                    "5": "Excellent development, all ideas relevant",
                    "4": "Good development, mostly relevant",
                    "3": "Adequate development",
                    "2": "Limited development",
                    "1": "Poor or irrelevant content"
                }
            },
            "form": {
                "description": "Word count and structure",
                "scoring": {
                    "5": "200-300 words, well-structured",
                    "4": "Within range, good structure",
                    "3": "Slightly outside range or adequate structure",
                    "2": "Significantly outside range or poor structure",
                    "1": "Very poor form"
                }
            },
            "grammar": {
                "description": "Grammatical accuracy",
                "scoring": {
                    "5": "Perfect grammar",
                    "4": "Minor errors",
                    "3": "Some errors",
                    "2": "Frequent errors",
                    "1": "Many errors affecting meaning"
                }
            },
            "vocabulary": {
                "description": "Appropriate word choice and range",
                "scoring": {
                    "5": "Excellent vocabulary range",
                    "4": "Good vocabulary",
                    "3": "Adequate vocabulary",
                    "2": "Limited vocabulary",
                    "1": "Poor vocabulary"
                }
            }
        },
        "summarize_written_text": {
            "content": {
                "description": "Accurate summary of main points",
                "scoring": {
                    "5": "Perfect summary",
                    "4": "Good summary",
                    "3": "Adequate summary",
                    "2": "Incomplete summary",
                    "1": "Poor or incorrect summary"
                }
            },
            "form": {
                "description": "Word count (5-75 words)",
                "scoring": {
                    "5": "Within range, well-structured",
                    "4": "Within range",
                    "3": "Slightly outside range",
                    "2": "Significantly outside range",
                    "1": "Very poor form"
                }
            },
            "grammar": {
                "description": "Grammatical accuracy",
                "scoring": {
                    "5": "Perfect grammar",
                    "4": "Minor errors",
                    "3": "Some errors",
                    "2": "Frequent errors",
                    "1": "Many errors"
                }
            },
            "vocabulary": {
                "description": "Appropriate word choice",
                "scoring": {
                    "5": "Excellent vocabulary",
                    "4": "Good vocabulary",
                    "3": "Adequate vocabulary",
                    "2": "Limited vocabulary",
                    "1": "Poor vocabulary"
                }
            }
        }
    },
    "reading": {
        "multiple_choice": {
            "comprehension": {
                "description": "Understanding of the passage",
                "scoring": {
                    "5": "Perfect comprehension",
                    "4": "Good comprehension",
                    "3": "Adequate comprehension",
                    "2": "Limited comprehension",
                    "1": "Poor comprehension"
                }
            },
            "accuracy": {
                "description": "Correct answer selection",
                "scoring": {
                    "5": "All correct",
                    "4": "Mostly correct",
                    "3": "Some correct",
                    "2": "Few correct",
                    "1": "Mostly incorrect"
                }
            }
        },
        "fill_in_blanks": {
            "accuracy": {
                "description": "Correct word selection",
                "scoring": {
                    "5": "All correct",
                    "4": "Mostly correct",
                    "3": "Some correct",
                    "2": "Few correct",
                    "1": "Mostly incorrect"
                }
            },
            "vocabulary": {
                "description": "Understanding of word meaning and context",
                "scoring": {
                    "5": "Excellent understanding",
                    "4": "Good understanding",
                    "3": "Adequate understanding",
                    "2": "Limited understanding",
                    "1": "Poor understanding"
                }
            }
        }
    }
}

print("âœ… PTE Rubrics loaded")
print(f"   - Test types: {list(PTE_RUBRICS.keys())}")
print(f"   - Total task types: {sum(len(tasks) for tasks in PTE_RUBRICS.values())}")



# Question Generation Functions

def generate_speaking_task(task_type: str, difficulty: str = "medium") -> Dict[str, Any]:
    """Generate a speaking task (read_aloud, describe_image, repeat_sentence)"""
    
    prompts = {
        "read_aloud": {
            "easy": "Generate a 50-60 word passage about everyday topics like weather, hobbies, or daily routines. Use simple vocabulary and clear sentence structures.",
            "medium": "Generate a 60-70 word passage about academic or professional topics like education, technology, or business. Use varied vocabulary and sentence structures.",
            "hard": "Generate a 70-80 word passage about complex topics like science, economics, or philosophy. Use advanced vocabulary and complex sentence structures."
        },
        "describe_image": {
            "easy": "Generate a description scenario for a simple image like a bar chart showing basic data, a simple diagram, or a clear photograph.",
            "medium": "Generate a description scenario for a moderate complexity image like a line graph with multiple data series, a process diagram, or a detailed photograph.",
            "hard": "Generate a description scenario for a complex image like a multi-panel chart, a technical diagram, or a complex scene with multiple elements."
        },
        "repeat_sentence": {
            "easy": "Generate a simple sentence with 10-12 words about everyday topics. Use common vocabulary.",
            "medium": "Generate a sentence with 13-17 words about academic or professional topics. Use varied vocabulary.",
            "hard": "Generate a complex sentence with 18-20 words about advanced topics. Use sophisticated vocabulary and complex grammar."
        }
    }
    
    time_limits = {
        "read_aloud": 40,
        "describe_image": 40,
        "repeat_sentence": 15
    }
    
    prompt = prompts[task_type][difficulty]
    
    full_prompt = f"""Generate a PTE {task_type.replace('_', ' ')} task. 
{prompt}

Return ONLY a JSON object with this exact structure:
{{
    "prompt": "the text/prompt for the task",
    "instructions": "brief instructions for the task"
}}

Do not include any markdown formatting or code blocks."""
    
    try:
        response = model.generate_content(full_prompt)
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        task_data = json.loads(text)
        
        return {
            "task_type": task_type,
            "test_type": "speaking",
            "prompt": task_data.get("prompt", ""),
            "instructions": task_data.get("instructions", ""),
            "time_limit": time_limits[task_type],
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Failed to generate task: {str(e)}",
            "task_type": task_type,
            "test_type": "speaking"
        }


def generate_writing_task(task_type: str, difficulty: str = "medium") -> Dict[str, Any]:
    """Generate a writing task (essay, summarize_written_text)"""
    
    if task_type == "essay":
        prompts = {
            "easy": "Generate an essay prompt about familiar topics like family, hobbies, or travel. The essay should be 200-300 words.",
            "medium": "Generate an essay prompt about academic or social topics like education, technology, or environment. The essay should be 200-300 words.",
            "hard": "Generate an essay prompt about complex topics like economics, philosophy, or global issues. The essay should be 200-300 words."
        }
        word_limit = 300
    else:  # summarize_written_text
        prompts = {
            "easy": "Generate a 100-150 word passage about everyday topics. The summary should be 5-75 words.",
            "medium": "Generate a 150-200 word passage about academic topics. The summary should be 5-75 words.",
            "hard": "Generate a 200-250 word passage about complex topics. The summary should be 5-75 words."
        }
        word_limit = 75
    
    prompt = prompts[difficulty]
    
    full_prompt = f"""Generate a PTE {task_type.replace('_', ' ')} task.
{prompt}

Return ONLY a JSON object with this exact structure:
{{
    "prompt": "the essay prompt or passage to summarize",
    "instructions": "brief instructions for the task"
}}

Do not include any markdown formatting or code blocks."""
    
    try:
        response = model.generate_content(full_prompt)
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        task_data = json.loads(text)
        
        return {
            "task_type": task_type,
            "test_type": "writing",
            "prompt": task_data.get("prompt", ""),
            "instructions": task_data.get("instructions", ""),
            "word_limit": word_limit,
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Failed to generate task: {str(e)}",
            "task_type": task_type,
            "test_type": "writing"
        }


def generate_reading_task(task_type: str, difficulty: str = "medium") -> Dict[str, Any]:
    """Generate a reading task (multiple_choice, fill_in_blanks)"""
    
    if task_type == "multiple_choice":
        prompts = {
            "easy": "Generate a 150-200 word passage with 3 multiple choice questions. Use simple vocabulary and clear topics.",
            "medium": "Generate a 200-300 word passage with 4 multiple choice questions. Use varied vocabulary and academic topics.",
            "hard": "Generate a 300-400 word passage with 5 multiple choice questions. Use advanced vocabulary and complex topics."
        }
    else:  # fill_in_blanks
        prompts = {
            "easy": "Generate a 100-150 word passage with 5 blanks. Use simple vocabulary.",
            "medium": "Generate a 150-250 word passage with 7 blanks. Use varied vocabulary.",
            "hard": "Generate a 250-350 word passage with 10 blanks. Use advanced vocabulary."
        }
    
    prompt = prompts[difficulty]
    
    full_prompt = f"""Generate a PTE {task_type.replace('_', ' ')} task.
{prompt}

For multiple_choice: Include the passage and questions with 4 options each (A, B, C, D).
For fill_in_blanks: Include the passage with [BLANK] markers where words should be filled.

Return ONLY a JSON object with this exact structure:
{{
    "passage": "the reading passage",
    "questions": ["question 1", "question 2", ...] or null for fill_in_blanks,
    "options": {{"1": ["A", "B", "C", "D"], ...}} or null for fill_in_blanks,
    "correct_answers": {{"1": "A", ...}} or ["word1", "word2", ...] for fill_in_blanks,
    "instructions": "brief instructions"
}}

Do not include any markdown formatting or code blocks."""
    
    try:
        response = model.generate_content(full_prompt)
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        task_data = json.loads(text)
        
        return {
            "task_type": task_type,
            "test_type": "reading",
            "passage": task_data.get("passage", ""),
            "questions": task_data.get("questions"),
            "options": task_data.get("options"),
            "correct_answers": task_data.get("correct_answers"),
            "instructions": task_data.get("instructions", ""),
            "difficulty": difficulty,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Failed to generate task: {str(e)}",
            "task_type": task_type,
            "test_type": "reading"
        }


print("âœ… Question generation functions loaded")



# Answer Evaluation Functions

def evaluate_speaking_answer(
    answer: str,
    task_type: str,
    original_prompt: str = ""
) -> Dict[str, Any]:
    """Evaluate a speaking answer using PTE rubrics"""
    
    rubric = PTE_RUBRICS["speaking"][task_type]
    
    # Build evaluation prompt
    rubric_text = json.dumps(rubric, indent=2)
    
    evaluation_prompt = f"""You are a PTE speaking evaluator. Evaluate the following answer based on the PTE scoring rubric.

Task Type: {task_type}
Original Prompt/Text: {original_prompt}
Student Answer: {answer}

Scoring Rubric:
{rubric_text}

Evaluate the answer and provide scores for each dimension (1-5 scale) and detailed feedback.

Return ONLY a JSON object with this exact structure:
{{
    "scores": {{
        "dimension_name": score (1-5),
        ...
    }},
    "overall_score": average_score (1-5),
    "feedback": "detailed feedback explaining the scores and areas for improvement"
}}

Do not include any markdown formatting or code blocks."""
    
    try:
        response = model.generate_content(evaluation_prompt)
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        evaluation = json.loads(text)
        
        return {
            "success": True,
            "task_type": task_type,
            "test_type": "speaking",
            "scores": evaluation.get("scores", {}),
            "overall_score": evaluation.get("overall_score", 0),
            "feedback": evaluation.get("feedback", ""),
            "evaluated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Evaluation failed: {str(e)}",
            "task_type": task_type,
            "test_type": "speaking"
        }


def evaluate_writing_answer(
    answer: str,
    task_type: str,
    original_prompt: str = "",
    word_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a writing answer using PTE rubrics"""
    
    rubric = PTE_RUBRICS["writing"][task_type]
    
    # Build evaluation prompt
    rubric_text = json.dumps(rubric, indent=2)
    word_count = len(answer.split())
    
    word_limit_text = f"Word limit: {word_limit} words. " if word_limit else ""
    
    evaluation_prompt = f"""You are a PTE writing evaluator. Evaluate the following answer based on the PTE scoring rubric.

Task Type: {task_type}
Original Prompt: {original_prompt}
Student Answer: {answer}
{word_limit_text}Actual word count: {word_count} words

Scoring Rubric:
{rubric_text}

Evaluate the answer and provide scores for each dimension (1-5 scale) and detailed feedback.

Return ONLY a JSON object with this exact structure:
{{
    "scores": {{
        "content": score (1-5),
        "form": score (1-5),
        "grammar": score (1-5),
        "vocabulary": score (1-5)
    }},
    "overall_score": average_score (1-5),
    "feedback": "detailed feedback explaining the scores and areas for improvement"
}}

Do not include any markdown formatting or code blocks."""
    
    try:
        response = model.generate_content(evaluation_prompt)
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        evaluation = json.loads(text)
        
        return {
            "success": True,
            "task_type": task_type,
            "test_type": "writing",
            "scores": evaluation.get("scores", {}),
            "overall_score": evaluation.get("overall_score", 0),
            "feedback": evaluation.get("feedback", ""),
            "word_count": word_count,
            "word_limit": word_limit,
            "evaluated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Evaluation failed: {str(e)}",
            "task_type": task_type,
            "test_type": "writing"
        }


def evaluate_reading_answer(
    answers: Dict[str, str],
    task_type: str,
    correct_answers: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate reading answers (multiple choice or fill in blanks)"""
    
    rubric = PTE_RUBRICS["reading"][task_type]
    
    # Calculate accuracy
    correct_count = 0
    total_count = len(answers)
    
    for key, user_answer in answers.items():
        correct_answer = correct_answers.get(key)
        if isinstance(correct_answer, list):
            if user_answer.lower().strip() in [str(a).lower().strip() for a in correct_answer]:
                correct_count += 1
        else:
            if str(user_answer).lower().strip() == str(correct_answer).lower().strip():
                correct_count += 1
    
    accuracy_score = (correct_count / total_count) * 5 if total_count > 0 else 0
    
    # Map accuracy to 1-5 scale
    if accuracy_score >= 4.5:
        accuracy_level = 5
    elif accuracy_score >= 3.5:
        accuracy_level = 4
    elif accuracy_score >= 2.5:
        accuracy_level = 3
    elif accuracy_score >= 1.5:
        accuracy_level = 2
    else:
        accuracy_level = 1
    
    # Generate feedback
    feedback_prompt = f"""You are a PTE reading evaluator. Provide feedback for the reading task.

Task Type: {task_type}
Correct Answers: {json.dumps(correct_answers)}
Student Answers: {json.dumps(answers)}
Score: {correct_count}/{total_count} correct ({accuracy_level}/5)

Provide detailed feedback explaining what was correct and what needs improvement.

Return ONLY a JSON object with this structure:
{{
    "feedback": "detailed feedback"
}}

Do not include any markdown formatting or code blocks."""
    
    try:
        response = model.generate_content(feedback_prompt)
        text = response.text.strip()
        
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        feedback_data = json.loads(text)
        
        return {
            "success": True,
            "task_type": task_type,
            "test_type": "reading",
            "scores": {
                "accuracy": accuracy_level,
                "comprehension": accuracy_level if task_type == "multiple_choice" else None,
                "vocabulary": accuracy_level if task_type == "fill_in_blanks" else None
            },
            "overall_score": accuracy_level,
            "correct_count": correct_count,
            "total_count": total_count,
            "feedback": feedback_data.get("feedback", ""),
            "evaluated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Evaluation failed: {str(e)}",
            "task_type": task_type,
            "test_type": "reading",
            "correct_count": correct_count,
            "total_count": total_count
        }


print("âœ… Answer evaluation functions loaded")



# Practice Session Management

class PracticeSession:
    """Simple session manager to track practice history"""
    
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None
    
    def create_session(self, test_type: Optional[str] = None) -> str:
        """Create a new practice session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.sessions[session_id] = {
            "session_id": session_id,
            "test_type": test_type,
            "questions": [],
            "answers": [],
            "evaluations": [],
            "created_at": datetime.now().isoformat()
        }
        self.current_session_id = session_id
        return session_id
    
    def add_question(self, question: Dict[str, Any]):
        """Add a question to the current session"""
        if self.current_session_id:
            self.sessions[self.current_session_id]["questions"].append(question)
    
    def add_answer(self, answer: Dict[str, Any]):
        """Add an answer to the current session"""
        if self.current_session_id:
            self.sessions[self.current_session_id]["answers"].append(answer)
    
    def add_evaluation(self, evaluation: Dict[str, Any]):
        """Add an evaluation to the current session"""
        if self.current_session_id:
            self.sessions[self.current_session_id]["evaluations"].append(evaluation)
    
    def get_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get session data"""
        sid = session_id or self.current_session_id
        return self.sessions.get(sid, {})
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all sessions"""
        return self.sessions
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        evaluations = session.get("evaluations", [])
        if not evaluations:
            return {"total_practices": 0}
        
        scores = [e.get("overall_score", 0) for e in evaluations if e.get("success")]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "total_practices": len(evaluations),
            "average_score": round(avg_score, 2),
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0
        }


# Initialize session manager
session_manager = PracticeSession()

print("âœ… Practice session manager initialized")



# Example: Generate an Essay task
writing_session = session_manager.create_session("writing")
print(f"ğŸ“� Created session: {writing_session}\n")

# Generate essay task
essay_task = generate_writing_task("essay", difficulty="medium")
session_manager.add_question(essay_task)

print("ğŸ“‹ Essay Task Generated:")
print(f"Task Type: {essay_task['task_type']}")
print(f"Word Limit: {essay_task['word_limit']} words")
print(f"\nğŸ“� Essay Prompt:")
print(essay_task['prompt'])
print(f"\nğŸ’¡ Instructions: {essay_task.get('instructions', 'Write a well-structured essay.')}")

print("\n" + "="*60)
print("Write your essay below (aim for 200-300 words):")



# Example: Evaluate your essay
# Replace this with your actual essay
my_essay = """Technology has revolutionized education in numerous ways. Online learning platforms have made education accessible to people worldwide, breaking down geographical barriers. Students can now access courses from top universities without leaving their homes.

However, this shift also presents challenges. The lack of face-to-face interaction can reduce social learning opportunities. Additionally, the digital divide means not everyone has equal access to technology.

In conclusion, while technology offers great potential for education, we must ensure it enhances rather than replaces traditional learning methods."""

# Evaluate
essay_evaluation = evaluate_writing_answer(
    answer=my_essay,
    task_type="essay",
    original_prompt=essay_task['prompt'],
    word_limit=essay_task['word_limit']
)

session_manager.add_answer({"answer": my_essay, "task_type": "essay"})
session_manager.add_evaluation(essay_evaluation)

print("ğŸ“Š Essay Evaluation Results:")
print(f"Overall Score: {essay_evaluation.get('overall_score', 0):.2f}/5.0")
print(f"Word Count: {essay_evaluation.get('word_count', 0)} / {essay_evaluation.get('word_limit', 0)}")
print(f"\nğŸ“ˆ Detailed Scores:")
for dimension, score in essay_evaluation.get('scores', {}).items():
    print(f"   {dimension.capitalize()}: {score}/5")
print(f"\nğŸ’¬ Feedback:")
print(essay_evaluation.get('feedback', 'No feedback available'))



# Example: Generate a Multiple Choice reading task
reading_session = session_manager.create_session("reading")
print(f"ğŸ“� Created session: {reading_session}\n")

# Generate reading task
reading_task = generate_reading_task("multiple_choice", difficulty="medium")
session_manager.add_question(reading_task)

print("ğŸ“‹ Reading Task Generated:")
print(f"Task Type: {reading_task['task_type']}")
print(f"\nğŸ“– Passage:")
print(reading_task.get('passage', 'No passage generated'))
print(f"\nâ�“ Questions:")
for i, question in enumerate(reading_task.get('questions', []), 1):
    print(f"\n{i}. {question}")
    if reading_task.get('options') and str(i) in reading_task['options']:
        for option in reading_task['options'][str(i)]:
            print(f"   {option}")

print("\n" + "="*60)
print("Answer the questions (provide as dict: {'1': 'A', '2': 'B', ...}):")



# Example: Evaluate reading answers
# Replace with your actual answers
my_reading_answers = {"1": "C", "2": "C", "3": "D", "4": "C"}  # Example answers

# Evaluate
reading_evaluation = evaluate_reading_answer(
    answers=my_reading_answers,
    task_type="multiple_choice",
    correct_answers=reading_task.get('correct_answers', {})
)

session_manager.add_answer({"answers": my_reading_answers, "task_type": "multiple_choice"})
session_manager.add_evaluation(reading_evaluation)

print("ğŸ“Š Reading Evaluation Results:")
print(f"Overall Score: {reading_evaluation.get('overall_score', 0):.2f}/5.0")
print(f"Correct: {reading_evaluation.get('correct_count', 0)}/{reading_evaluation.get('total_count', 0)}")
print(f"\nğŸ“ˆ Detailed Scores:")
for dimension, score in reading_evaluation.get('scores', {}).items():
    if score is not None:
        print(f"   {dimension.capitalize()}: {score}/5")
print(f"\nğŸ’¬ Feedback:")
print(reading_evaluation.get('feedback', 'No feedback available'))





