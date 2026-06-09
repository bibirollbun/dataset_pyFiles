
# Setup cell (example). Do NOT hardcode your API key in public notebooks.
import os
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # recommended: set via Kaggle Secrets or environment
print("GEMINI_API_KEY present:", bool(GEMINI_API_KEY))
# If you plan to enable real calls, uncomment and install the client:
# !pip install --upgrade google-generativeai
# import google.generativeai as genai
# genai.configure(api_key=GEMINI_API_KEY)




quiz_schema = {
    "type": "object",
    "properties": {
        "title": {"type":"string"},
        "type": {"type":"string", "enum":["QUIZ"]},
        "quizData": {
            "type":"array",
            "items":{
                "type":"object",
                "properties":{
                    "question": {"type":"string"},
                    "options": {"type":"array", "items":{"type":"string"}},
                    "answer": {"type":"string"},
                    "hint": {"type":"string"},
                    "explanation": {"type":"string"}
                },
                "required":["question","options","answer","hint","explanation"]
            }
        }
    },
    "required":["title","type","quizData"]
}

flashcard_schema = {
    "type":"object",
    "properties":{
        "title": {"type":"string"},
        "type": {"type":"string", "enum":["FLASHCARDS"]},
        "flashcardData": {"type":"array", "items":{
            "type":"object",
            "properties":{
                "front": {"type":"string"},
                "back": {"type":"string"}
            },
            "required":["front","back"]
        }}
    },
    "required":["title","type","flashcardData"]
}

echo_speak_schema = {
    "type":"object",
    "properties":{
        "title":{"type":"string"},
        "type":{"type":"string","enum":["ECHOSPEAK"]},
        "echoSpeakData":{
            "type":"object",
            "properties":{
                "accuracyScore":{"type":"number"},
                "transcription":{"type":"string"},
                "mistakes":{"type":"array","items":{"type":"string"}},
                "feedback":{"type":"string"},
                "tutorResponse":{"type":"string"}
            },
            "required":["accuracyScore","transcription","mistakes","feedback","tutorResponse"]
        },
        "imagePrompt":{"type":"string"}
    },
    "required":["title","type","echoSpeakData"]
}

content_schema = {
    "type":"object",
    "properties":{
        "title":{"type":"string"},
        "type":{"type":"string"},
        "markdownContent":{"type":"string"},
        "imagePrompt":{"type":"string"}
    },
    "required":["title","type","markdownContent"]
}

print('Schemas defined: quiz, flashcard, echo_speak, content')




import json, base64, random, time
from typing import Optional, Dict, Any

# Enums (same as types.ts)
class StudyFeature:
    EXPLAIN='EXPLAIN'; NOTES='NOTES'; QUIZ='QUIZ'; SOLVER='SOLVER'; ECHOSPEAK='ECHOSPEAK'; FLASHCARDS='FLASHCARDS'; DOUBT_SOLVER='DOUBT_SOLVER'; MOOD_BOOSTER='MOOD_BOOSTER'

class StudyMode:
    TUTOR='Tutor'; FRIEND='Friend'; EXAM='Exam'; FUN='Fun'

class AppLanguage:
    ENGLISH='English'; SPANISH='Spanish'; FRENCH='French'; HINDI='Hindi'; GERMAN='German'; CHINESE='Chinese'; JAPANESE='Japanese'

def blob_to_base64(blob_bytes: bytes) -> str:
    """If you have raw bytes from an audio file, encode to base64 without data URI prefix."""
    return base64.b64encode(blob_bytes).decode('utf-8')

def mock_response_for_demo(feature, input_text):
    # Return plausible structured response for each feature for demo purposes
    if feature==StudyFeature.EXPLAIN:
        return {
            'type': StudyFeature.EXPLAIN,
            'title': f'Explain: {input_text[:30]}',
            'markdownContent': f"""### Quick explanation of {input_text}\n\n- Key point 1\n- Key point 2\n\n**Example:** Simple example here."""
        }
    if feature==StudyFeature.QUIZ:
        return {
            'type': StudyFeature.QUIZ,
            'title': f'Quiz: {input_text[:30]}',
            'quizData':[
                {'question':'What is X?','options':['A','B','C','D'],'answer':'A','hint':'Remember X equals A','explanation':'Because A relates to...'},
                {'question':'Why use Y?','options':['P','Q','R','S'],'answer':'Q','hint':'Think about...','explanation':'Q is used because...'}
            ]
        }
    if feature==StudyFeature.ECHOSPEAK:
        return {
            'type': StudyFeature.ECHOSPEAK,
            'title':'EchoSpeak Result',
            'echoSpeakData':{
                'accuracyScore': 82,
                'transcription': input_text,
                'mistakes': ['Pronunciation of "th"','Missing article before noun'],
                'feedback': 'Slow down, pronounce the "th" sound, and use "a" before singular nouns.',
                'tutorResponse': 'Nice try! Let\'s practice the "th" sound: place your tongue... ðŸ˜Š'
            },
            'imagePrompt': 'Diagram showing tongue position for "th" sound.'
        }
    if feature==StudyFeature.FLASHCARDS:
        return {
            'type': StudyFeature.FLASHCARDS,
            'title': 'Flashcards',
            'flashcardData':[{'front':'Term A','back':'Definition A'},{'front':'Term B','back':'Definition B'}]
        }
    if feature==StudyFeature.MOOD_BOOSTER:
        return {
            'type': StudyFeature.MOOD_BOOSTER,
            'title': 'Mood Booster',
            'markdownContent': 'You studied for some minutes â€” great work! ðŸŽ‰\n\nTake a short break...'
        }
    return {'type':'UNKNOWN','title':'No result'}

def generate_study_help(feature: str, mode: str, language: str, input_text: str, audio_bytes: Optional[bytes]=None, study_time_minutes: int=0) -> Dict[str,Any]:
    """Construct prompt and (mock) call to Gemini. Replace mock with real API call when enabled."""

    # System instruction template (shortened for notebook clarity)
    system_instruction = f"""
You are EchoLearn, a friendly multimodal AI tutor. Mode: {mode}. Default language: {language}.
- If the user input language is clear, respond in that language.
- Use emojis and simple language.
- For visual topics, include an imagePrompt.
"""

    prompt = ''
    schema = content_schema
    if feature==StudyFeature.EXPLAIN:
        prompt = f'Explain: "{input_text}". Keep it concise and student-friendly.'
    elif feature==StudyFeature.QUIZ:
        prompt = f'Create a 5-question MCQ quiz on: "{input_text}".'
        schema = quiz_schema
    elif feature==StudyFeature.ECHOSPEAK:
        prompt = f'Analyze this spoken input: "{input_text}" and provide accuracy, mistakes, feedback, tutorResponse.'
        schema = echo_speak_schema
    elif feature==StudyFeature.FLASHCARDS:
        prompt = f'Create flashcards for: "{input_text}".'
        schema = flashcard_schema
    elif feature==StudyFeature.MOOD_BOOSTER:
        prompt = f'User studied {study_time_minutes} minutes. Provide a short motivational micro-lesson.'
    else:
        prompt = f'Handle: "{input_text}"'

    # If the real client is configured, a call would go here. We return mocked responses for offline demo.
    # Example of real call (pseudocode):
    # response = genai.models.generate_content(...)

    print('---PROMPT (for debugging)---')
    print(prompt)
    time.sleep(0.2)

    # Return a mock structured response
    return mock_response_for_demo(feature, input_text)




examples = [
    (StudyFeature.EXPLAIN, StudyMode.TUTOR, AppLanguage.ENGLISH, 'Explain Newton\'s second law in simple words.'),
    (StudyFeature.QUIZ, StudyMode.EXAM, AppLanguage.ENGLISH, 'Kinematics: displacement, velocity, acceleration'),
    (StudyFeature.ECHOSPEAK, StudyMode.TUTOR, AppLanguage.ENGLISH, 'The acceleration equals force divided by mass.'),
    (StudyFeature.FLASHCARDS, StudyMode.FRIEND, AppLanguage.ENGLISH, 'Photosynthesis key terms'),
    (StudyFeature.MOOD_BOOSTER, StudyMode.FRIEND, AppLanguage.ENGLISH, 'motivation', None, 25)
]

for ex in examples:
    feature, mode, lang, text = ex[0], ex[1], ex[2], ex[3]
    print('\n==== Feature:', feature, '====')
    resp = generate_study_help(feature, mode, lang, text)
    print(json.dumps(resp, indent=2, ensure_ascii=False)[:1000])


