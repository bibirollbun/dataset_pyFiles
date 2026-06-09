# Install required packages
!pip install -q google-genai


# Import required libraries
import os
from typing import Dict, List, Any
import json
from google import genai
from google.genai import types


# Set up API key from Kaggle Secrets
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Alternative: Set API key directly (not recommended for security)
# GOOGLE_API_KEY = "your-api-key-here"

print("âœ… API Key loaded successfully!")


class QuizMasterAgent:
    """Main agent that creates quizzes based on user subject"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-lite"
        
    def create_quiz(self, subject: str, num_questions: int = 5) -> Dict[str, Any]:
        """Create a quiz on the specified subject with MCQ format"""
        
        prompt = f"""You are a professional quiz creator. Create a quiz with {num_questions} questions about {subject}.

Format your response as a JSON object with the following structure:
{{
    "subject": "{subject}",
    "questions": [
        {{
            "id": 1,
            "question": "Question text here?",
            "question_type": "mcq",
            "options": {{
                "A": "Option A text",
                "B": "Option B text",
                "C": "Option C text",
                "D": "Option D text"
            }},
            "correct_answer": "A"
        }},
        {{
            "id": 2,
            "question": "Question requiring text explanation?",
            "question_type": "text",
            "options": null,
            "correct_answer": "Expected text answer here"
        }},
        {{
            "id": 3,
            "question": "Write code to solve this problem?",
            "question_type": "code",
            "options": null,
            "correct_answer": "Expected code solution here"
        }}
    ]
}}

IMPORTANT RULES:
1. Most questions (70-80%) should be MCQ format with 4 options (A, B, C, D)
2. For MCQ questions: question_type = "mcq", provide options object, correct_answer is the letter (A/B/C/D)
3. For text questions: question_type = "text", options = null, correct_answer is the expected text
4. For code questions: question_type = "code", options = null, correct_answer is the expected code
5. Use "text" type for questions requiring explanations or descriptions
6. Use "code" type for programming/coding questions
7. Make MCQ options plausible but clearly distinguishable

Only return the JSON object, no additional text."""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        # Parse the response
        try:
            quiz_data = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            return quiz_data
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "subject": subject,
                "questions": self._parse_quiz_text(response.text, num_questions)
            }
    
    def _parse_quiz_text(self, text: str, num_questions: int) -> List[Dict]:
        """Fallback parser if JSON parsing fails"""
        questions = []
        
        for i in range(num_questions):
            questions.append({
                "id": i + 1,
                "question": f"Question {i + 1} about the subject",
                "question_type": "mcq",
                "options": {
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D"
                },
                "correct_answer": "A"
            })
        
        return questions


class GraderAgent:
    """Agent that evaluates user answers and assigns grades"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-lite"
    
    def grade_answer(self, question: str, correct_answer: str, user_answer: str, 
                     question_type: str = "text", options: Dict = None) -> Dict[str, Any]:
        """Grade a single answer based on question type"""
        
        # For MCQ, do direct comparison
        if question_type == "mcq":
            is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
            
            if options and correct_answer in options:
                correct_text = f"{correct_answer}. {options[correct_answer]}"
            else:
                correct_text = correct_answer
            
            return {
                "is_correct": is_correct,
                "score": 100 if is_correct else 0,
                "feedback": f"Correct! The answer is {correct_text}" if is_correct 
                           else f"Incorrect. The correct answer is {correct_text}",
                "correct_answer_text": correct_text
            }
        
        # For text and code, use AI grading
        prompt = f"""You are a professional grader. Evaluate the following answer:

Question: {question}
Question Type: {question_type}
Correct Answer: {correct_answer}
User's Answer: {user_answer}

Determine if the user's answer is correct or incorrect. 
For code questions, check if the logic is correct even if syntax varies slightly.
For text questions, be lenient with minor wording differences.

Respond in JSON format:
{{
    "is_correct": true/false,
    "score": 0-100,
    "feedback": "Brief feedback on the answer"
}}

Only return the JSON object."""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        try:
            result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            result['correct_answer_text'] = correct_answer
            return result
        except json.JSONDecodeError:
            # Fallback
            is_correct = user_answer.lower().strip() in correct_answer.lower()
            return {
                "is_correct": is_correct,
                "score": 100 if is_correct else 0,
                "feedback": "Answer evaluated",
                "correct_answer_text": correct_answer
            }


class CorrectionAgent:
    """Agent that provides corrections for incorrect answers"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-lite"
    
    def provide_correction(self, question: str, correct_answer: str, user_answer: str) -> str:
        """Provide a detailed correction for an incorrect answer"""
        
        prompt = f"""You are a helpful tutor. A student answered a question incorrectly. Provide a clear, educational correction.

Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {user_answer}

Provide:
1. The correct answer
2. An explanation of why it's correct
3. What was wrong with the student's answer (if applicable)

Be encouraging and educational in your tone."""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        return response.text


class VerificationAgent:
    """Agent that verifies corrections for technical accuracy (Math/Code)"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-lite"
        
    def verify_correction(self, question: str, correction: str, question_type: str) -> str:
        """Review and refine the correction if needed"""
        
        # Only verify code and text questions (likely to contain math/logic)
        if question_type not in ['code', 'text']:
            return correction
            
        prompt = f"""You are a Senior Technical Reviewer. Review the following correction provided to a student.
        
Question: {question}
Question Type: {question_type}
Proposed Correction:
{correction}

Your task:
1. Check for any mathematical errors, logical flaws, or syntax errors in code.
2. If the correction is accurate, return it exactly as is.
3. If there are errors, provide a REFINED version that fixes them while maintaining the educational tone.

Return ONLY the final correction text (original or refined)."""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        return response.text


class QuizChatbot:
    """Main chatbot orchestrator that manages the quiz flow"""
    
    def __init__(self, api_key: str):
        self.quiz_master = QuizMasterAgent(api_key)
        self.grader = GraderAgent(api_key)
        self.corrector = CorrectionAgent(api_key)
        self.verifier = VerificationAgent(api_key)
        
        self.current_quiz = None
        self.user_answers = {}
        self.state = "INIT"  # INIT, QUIZ_CREATED, ANSWERING, GRADING
        
    def start(self):
        """Start the chatbot interaction"""
        print("=" * 60)
        print("ğŸ�“ WELCOME TO THE QUIZ CHATBOT! ğŸ�“")
        print("=" * 60)
        print("\nI'm your professional quiz assistant!")
        print("I'll help you create a quiz on any subject you'd like.\n")
        
        # Get subject from user
        subject = input("ğŸ“š What subject would you like to be quizzed on? ")
        
        print(f"\nğŸ”„ Creating a quiz on '{subject}'...\n")
        
        # Create quiz
        self.current_quiz = self.quiz_master.create_quiz(subject)
        self.state = "QUIZ_CREATED"
        
        print(f"âœ… Quiz created! Here are your questions:\n")
        print("=" * 60)
        
        # Display questions with options
        for q in self.current_quiz['questions']:
            print(f"\n{'='*60}")
            print(f"Question {q['id']}: {q['question']}")
            print(f"Type: {q.get('question_type', 'text').upper()}")
            print(f"{'='*60}")
            
            # Display options for MCQ
            if q.get('question_type') == 'mcq' and q.get('options'):
                for option_key, option_text in sorted(q['options'].items()):
                    print(f"  {option_key}. {option_text}")
            elif q.get('question_type') == 'code':
                print("  ğŸ’» Write your code answer below")
            else:  # text type
                print("  âœ�ï¸�  Write your text answer below")
        
        print("\n" + "=" * 60)
        print("\nğŸ“� Please answer each question below.")
        print("When you're done, type 'submit' to submit your answers.\n")
        
        # Collect answers
        self.collect_answers()
        
    def collect_answers(self):
        """Collect answers from the user"""
        self.state = "ANSWERING"
        
        for q in self.current_quiz['questions']:
            question_type = q.get('question_type', 'text')
            
            print(f"\n{'='*60}")
            print(f"Question {q['id']}: {q['question']}")
            
            if question_type == 'mcq':
                # Display options again for convenience
                if q.get('options'):
                    for option_key, option_text in sorted(q['options'].items()):
                        print(f"  {option_key}. {option_text}")
                
                # Get MCQ answer (single letter)
                while True:
                    answer = input(f"\nYour answer (A/B/C/D): ").strip().upper()
                    
                    if answer.lower() == 'submit':
                        print("\nâš ï¸�  Please answer this question first before submitting.")
                        continue
                    
                    if answer in ['A', 'B', 'C', 'D']:
                        self.user_answers[q['id']] = answer
                        break
                    else:
                        print("â�Œ Please enter A, B, C, or D")
            
            elif question_type == 'code':
                # Get code answer (multi-line)
                print(f"\nğŸ’» Enter your code (type 'END' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == 'END':
                        break
                    lines.append(line)
                
                code_answer = '\n'.join(lines)
                self.user_answers[q['id']] = code_answer
                print("âœ… Code answer saved!")
            
            else:  # text type
                # Get text answer (multi-line)
                print(f"\nâœ�ï¸�  Enter your answer (type 'END' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == 'END':
                        break
                    lines.append(line)
                
                text_answer = '\n'.join(lines)
                self.user_answers[q['id']] = text_answer
                print("âœ… Text answer saved!")
        
        # Wait for submit command
        print("\n" + "=" * 60)
        while True:
            submit_cmd = input("\nType 'submit' to submit your answers: ").strip().lower()
            if submit_cmd == 'submit':
                break
            else:
                print("Please type 'submit' to continue.")
        
        # Grade answers
        self.grade_quiz()
        
    def grade_quiz(self):
        """Grade the quiz using multi-agent system"""
        self.state = "GRADING"
        
        print("\n" + "=" * 60)
        print("ğŸ”„ Grading your answers...")
        print("=" * 60 + "\n")
        
        results = []
        total_score = 0
        
        for q in self.current_quiz['questions']:
            question_text = q['question']
            correct_answer = q['correct_answer']
            user_answer = self.user_answers.get(q['id'], '')
            question_type = q.get('question_type', 'text')
            options = q.get('options')
            
            print(f"ğŸ“Š Grading Question {q['id']}...")
            
            # Agent 1: Grader checks the answer
            grade_result = self.grader.grade_answer(
                question_text, 
                correct_answer, 
                user_answer,
                question_type=question_type,
                options=options
            )
            
            result = {
                'question_id': q['id'],
                'question': question_text,
                'question_type': question_type,
                'options': options,
                'user_answer': user_answer,
                'is_correct': grade_result['is_correct'],
                'score': grade_result['score'],
                'feedback': grade_result['feedback'],
                'correct_answer_text': grade_result.get('correct_answer_text', correct_answer)
            }
            
            # Agent 2: If incorrect, get correction
            if not grade_result['is_correct']:
                print(f"   â�Œ Incorrect - Getting correction...")
                correction = self.corrector.provide_correction(
                    question_text, 
                    grade_result.get('correct_answer_text', correct_answer), 
                    user_answer
                )
                
                # Agent 3: Verify correction for technical accuracy
                if question_type in ['code', 'text']:
                    print(f"   ğŸ”� Verifying correction...")
                    correction = self.verifier.verify_correction(
                        question_text,
                        correction,
                        question_type
                    )
                
                result['correction'] = correction
            else:
                print(f"   âœ… Correct!")
                result['correction'] = None
            
            results.append(result)
            total_score += grade_result['score']
        
        # Display final results
        self.display_results(results, total_score)
        
    def display_results(self, results: List[Dict], total_score: float):
        """Display the final graded results"""
        
        print("\n" + "=" * 60)
        print("ğŸ“‹ QUIZ RESULTS")
        print("=" * 60 + "\n")
        
        for result in results:
            print(f"\n{'='*60}")
            print(f"Question {result['question_id']}: {result['question']}")
            print(f"Type: {result.get('question_type', 'text').upper()}")
            print(f"{'='*60}")
            
            # Show options for MCQ
            if result.get('question_type') == 'mcq' and result.get('options'):
                print("\nOptions:")
                for option_key, option_text in sorted(result['options'].items()):
                    print(f"  {option_key}. {option_text}")
            
            # Display user's answer
            print(f"\nYour Answer:")
            if result.get('question_type') == 'code':
                print("```")
                print(result['user_answer'])
                print("```")
            else:
                print(f"  {result['user_answer']}")
            
            print(f"\nStatus: {'âœ… CORRECT' if result['is_correct'] else 'â�Œ INCORRECT'}")
            print(f"Score: {result['score']}/100")
            print(f"Feedback: {result['feedback']}")
            
            if result['correction']:
                print(f"\nğŸ“– CORRECTION:")
                print(f"{result['correction']}")
        
        # Final score
        avg_score = total_score / len(results)
        print(f"\n{'='*60}")
        print(f"ğŸ�¯ FINAL SCORE: {avg_score:.1f}/100")
        print(f"{'='*60}\n")
        
        # Performance message
        if avg_score >= 90:
            print("ğŸŒŸ Excellent work! You've mastered this subject!")
        elif avg_score >= 70:
            print("ğŸ‘� Good job! Keep up the great work!")
        elif avg_score >= 50:
            print("ğŸ“š Not bad! Review the corrections and try again!")
        else:
            print("ğŸ’ª Keep studying! You'll do better next time!")


# Run the Quiz Chatbot
print("ğŸš€ Starting Quiz Chatbot with MCQ Support...\n")
chatbot = QuizChatbot(GOOGLE_API_KEY)
chatbot.start()




