# ==== Core Python Libraries ====
import os
import json
import re
import time
from collections import Counter
from datetime import datetime

# ==== Memory (Session History) ====
SESSION_LOG = []
MAX_HISTORY = 10   # store last 10 interactions

# ==== Gemini SDK ====
import google.generativeai as genai

# ==== Kaggle Secrets (API key loader) ====
from kaggle_secrets import UserSecretsClient

# ==== Load API Key ====
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("â�Œ GOOGLE_API_KEY is not set in Kaggle Secrets.")

print("API key loaded successfully âœ”")

# ==== Configure Gemini ====
genai.configure(api_key=api_key)
print("Gemini configured successfully âœ”")




from google.api_core import exceptions

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Generic Gemini helper used by all agents (Intake, Coach, Plan).
    Combines system + user prompt, calls Gemini, and returns text output.
    Includes error handling for empty responses and quota issues.
    """

    # Gemini 2.5 Flash model
    model = genai.GenerativeModel(
        "models/gemini-2.5-flash",
        system_instruction=system_prompt
    )

    try:
        # Send only user message (system prompt already injected above)
        response = model.generate_content(user_prompt)

        text = response.text or ""
        if not text.strip():
            return "ERROR: Empty response from model."

        return text

    except exceptions.ResourceExhausted:
        return "ERROR: Gemini quota exceeded for today. Try again later."

    except Exception as e:
        return f"ERROR: LLM call failed: {e}"



INTAKE_SYSTEM_PROMPT = """
You are the Intake Agent for an AI archery coach called BowSensei.

The user will describe their archery problem in English or Hinglish.
Your task is to extract CLEAN structured JSON. No extra words.

Extract the following:

1. bow_type:
   Detect from text using keyword match:
   - "recurve"
   - "compound"
   - "barebow"
   If unclear, return "unknown".

2. draw_weight_lbs:
   Extract any number followed by:
   - "lb", "lbs", "pound", "pounds"
   If not found, return null.

3. experience_level:
   Infer if possible:
   - "beginner"
   - "intermediate"
   - "advanced"
   If no clue, return "unknown".

4. main_issues:
   A list of 2â€“5 short issues.
   Examples you MUST detect:
   - "arrow drifting left"
   - "arrow drifting right"
   - "weak draw hold"
   - "aim shaking"
   - "rushed release"
   - "anchor inconsistency"
   - "back tension issue"
   - "bow arm torque"

5. goals:
   Extract short strings like:
   - "score 340/360"
   - "stabilize hold"
   - "improve release"
   If no goals, return an empty list.

You MUST return ONE JSON object with EXACTLY these keys:

{
  "bow_type": "...",
  "draw_weight_lbs": 0,
  "experience_level": "...",
  "main_issues": [],
  "goals": []
}

RULES:
- Reply ONLY with JSON. No backticks, no markdown.
- Use DOUBLE QUOTES for all keys and string values.
- Do NOT use single quotes anywhere.
- Do NOT write anything before or after the JSON.
- If something is missing â†’ use null or "unknown".
- main_issues MUST be a list of short phrases.
"""



def run_intake_agent(user_message: str) -> dict:
    """
    Intake Agent:
    - Takes raw user description
    - Sends it to Gemini to produce structured JSON
    - Parses and returns Python dict
    """
    raw_response = call_llm(INTAKE_SYSTEM_PROMPT, user_message)

    # --- DEBUG: dekho model kya bhej raha hai ---
    print("\n=== RAW INTAKE OUTPUT ===")
    print(raw_response)
    print("=== END RAW INTAKE OUTPUT ===\n")

    # Basic cleanup: remove markdown wrappers etc.
    json_str = raw_response.strip()

    # Agar ```json ... ``` type wrapper ho
    if "```" in json_str:
        first_brace = json_str.find("{")
        last_brace = json_str.rfind("}")
        if first_brace != -1 and last_brace != -1:
            json_str = json_str[first_brace:last_brace+1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[WARN] Intake JSON parse failed:", e)
        print("[WARN] Falling back to safe default profile.\n")
        data = {
            "bow_type": "unknown",
            "draw_weight_lbs": None,
            "experience_level": "unknown",
            "main_issues": ["Could not parse intake properly. Please re-run."],
            "goals": [],
        }

    return data




COACH_SYSTEM_PROMPT = """
You are the Technique Coach Agent for an AI archery coach called BowSensei.

You will receive:
- structured_info: JSON object about the archer (bow_type, draw_weight_lbs, experience_level, main_issues, goals).
- This JSON will be passed as the user message.

Your job is to:
1) Analyze the situation.
2) Predict 2â€“4 likely technical or mental mistakes.
3) Propose 3â€“5 concrete drills the archer can do.
4) Give short explanations of WHY each drill helps.

Archery context examples:
- main_issues may include: "arrow drifting left", "weak draw hold", "aim shaking", "rushed release", "anchor inconsistency", "back tension issue", "bow arm torque", etc.
- Use sensible archery coaching logic (back tension, anchor, alignment, mental pressure).

You MUST reply with ONE JSON object in exactly this format:

{
  "mistake_analysis": [
    "short bullet about likely mistake 1",
    "short bullet about likely mistake 2"
  ],
  "drills": [
    "short bullet describing drill 1",
    "short bullet describing drill 2"
  ],
  "explanations": [
    "short explanation why drill 1 helps, in simple Hinglish",
    "short explanation why drill 2 helps, in simple Hinglish"
  ]
}

Rules:
- Use DOUBLE QUOTES for all keys and string values.
- Do NOT use single quotes anywhere.
- Do NOT add markdown, backticks, or any extra text before or after the JSON.
- Keep each item short (1 sentence). No long paragraphs.
- If you are unsure, make your best coaching guess instead of returning empty lists.
"""



def run_coach_agent(structured_info: dict) -> dict:
    """
    Technique Coach Agent:
    - Takes structured intake JSON
    - Asks Gemini to generate mistakes + drills + explanations
    - Parses and returns Python dict
    """
    # User payload = JSON string of structured_info
    user_payload = json.dumps(
        {"structured_info": structured_info},
        ensure_ascii=False,
    )

    raw_response = call_llm(COACH_SYSTEM_PROMPT, user_payload)

    # --- DEBUG: Coach raw output dekhne ke liye (optional) ---
    print("\n=== RAW COACH OUTPUT ===")
    print(raw_response)
    print("=== END RAW COACH OUTPUT ===\n")

    coach_str = raw_response.strip()

    # Agar ```json ... ``` wrapper aa gaya ho to hata do
    if "```" in coach_str:
        first_brace = coach_str.find("{")
        last_brace = coach_str.rfind("}")
        if first_brace != -1 and last_brace != -1:
            coach_str = coach_str[first_brace:last_brace+1]

    try:
        data = json.loads(coach_str)
    except json.JSONDecodeError as e:
        print("[WARN] Coach JSON parse failed:", e)
        print("[WARN] Using safe fallback for coach output.\n")
        data = {
            "mistake_analysis": [
                "Coach output could not be parsed. Please re-run this session."
            ],
            "drills": [],
            "explanations": [],
        }

    return data




PLAN_SYSTEM_PROMPT = """
You are the Training Plan Agent for BowSensei.

You will receive a JSON object as the user message with:
- "structured_info": intake JSON about the archer
- "coach_output": JSON from the Technique Coach Agent
  (mistake_analysis, drills, explanations)

Your task:
- Create a focused 7-day training plan for the archer.
- Highlight a single focus for TODAY based on the issues.
- Give a short progress note summarizing what the archer should track.

You MUST reply with ONE JSON object in exactly this format:

{
  "week_plan": {
    "day1": "short plan for day1",
    "day2": "short plan for day2",
    "day3": "short plan for day3",
    "day4": "short plan for day4",
    "day5": "short plan for day5",
    "day6": "short plan for day6",
    "day7": "short plan for day7"
  },
  "today_focus": "one clear sentence on what to focus today",
  "progress_note": "2â€“3 short sentences describing pattern from last sessions and how to improve"
}

Guidelines:
- Use archer's main_issues + mistake_analysis + drills to design the week.
- Mix technique drills, strength/holding work, and mental practice.
- Use simple Hinglish, short clear sentences.

Rules:
- Use DOUBLE QUOTES for all keys and string values.
- Do NOT use single quotes.
- Do NOT add markdown, backticks, or any text outside the JSON.
- If information is missing, make your best reasonable archery plan instead of leaving it empty.
"""



def run_plan_agent(structured_info: dict, coach_output: dict) -> dict:
    """
    Training Plan Agent:
    - Takes intake + coach_output
    - Asks Gemini for a 7-day plan + today's focus + progress note
    - Returns parsed Python dict
    """
    user_payload = json.dumps(
        {
            "structured_info": structured_info,
            "coach_output": coach_output,
        },
        ensure_ascii=False,
    )

    raw_response = call_llm(PLAN_SYSTEM_PROMPT, user_payload)

    # --- DEBUG: Plan raw output (optional) ---
    print("\n=== RAW PLAN OUTPUT ===")
    print(raw_response)
    print("=== END RAW PLAN OUTPUT ===\n")

    plan_str = raw_response.strip()

    # Agar ```json ... ``` wrapper aa gaya ho
    if "```" in plan_str:
        first_brace = plan_str.find("{")
        last_brace = plan_str.rfind("}")
        if first_brace != -1 and last_brace != -1:
            plan_str = plan_str[first_brace:last_brace+1]

    try:
        data = json.loads(plan_str)
    except json.JSONDecodeError as e:
        print("[WARN] Plan JSON parse failed:", e)
        print("[WARN] Using safe fallback training plan.\n")
        data = {
            "week_plan": {
                "day1": "Warm-up + key technique drills.",
                "day2": "Repeat similar focus with more volume.",
                "day3": "Technique + strength mix (back tension, holding).",
                "day4": "Light session / recovery + mental practice.",
                "day5": "Key drill repetition on main mistakes.",
                "day6": "Mock scoring session at main distance.",
                "day7": "Rest / review video + notes.",
            },
            "today_focus": "Work on main form mistakes and drills suggested by the coach.",
            "progress_note": (
                "Generic fallback plan used because plan JSON could not be parsed. "
                "Focus on consistency, clean form, and honest self-review across the week."
            ),
        }

    return data







test_text = """
My bow is 40 lbs, and at 70m my arrows are drifting to the left.
After holding the draw for 10â€“12 seconds, my form collapses, and under pressure my release becomes inconsistent"""

intake = run_intake_agent(test_text)
coach  = run_coach_agent(intake)
plan   = run_plan_agent(intake, coach)


# print("INTAKE:\n", json.dumps(intake, indent=2, ensure_ascii=False))
# print("\nCOACH:\n", json.dumps(coach, indent=2, ensure_ascii=False))
# print("\nPLAN:\n", json.dumps(plan, indent=2, ensure_ascii=False))


def analyze_archery_image(image_path: str) -> str:
    """
    Image Helper:
    - Sends the archerâ€™s form photo to Gemini
    - Analyses posture, anchor point, shoulder alignment, bow arm, head position, and back tension
    - Returns clear bullet-point feedback
    """

    model = genai.GenerativeModel(
        "models/gemini-2.5-flash",
        system_instruction=(
            "You are an archery form coach.\n"
            "The user will send a photo of an archer.\n"
            "Analyse posture, anchor point, bow arm, shoulder alignment, head position, "
            "and back tension.\n\n"
            "IMPORTANT:\n"
            "- If the archer is not clearly visible, say so.\n"
            "- If only part of the body is visible, mention that full-form analysis is limited.\n"
            "- If the photo is blurry or dark, mention low quality.\n"
            "Respond in short, clear bullet-points."
        ),
    )

    # Upload image file from Kaggle path
    img_file = genai.upload_file(image_path)
    print("\n==============================================================")
    print("Uploaded image:", img_file.name, "| state:", img_file.state.name)
    print("\n==============================================================")

    # Run analysis
    prompt = "Analyze this archery image and give form corrections."
    resp = model.generate_content([prompt, img_file])

    return resp.text



def analyze_archery_video(video_path: str) -> str:
    """
    Video Helper:
    - Sends a short archery shooting video to Gemini
    - Analyzes timing, draw-hold stability, release, and follow-through
    - Returns a clean Hinglish summary
    """

    print("\n==============================")
    print("Uploading video for analysis...")
    print("\n==============================")
    video_file = genai.upload_file(video_path)
    print("\n==============================")
    print("Uploaded video:", video_file.name, "| state:", video_file.state.name)
    print("\n==============================")

    # Wait until Gemini finishes processing the uploaded video
    while True:
        file_info = genai.get_file(video_file.name)
        state = file_info.state.name
        print("\n==============================")
        print("Current state:", state)

        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError("Video processing failed on Gemini side.")

        time.sleep(5)  # wait before checking again

    model = genai.GenerativeModel(
        "models/gemini-2.5-flash",
        system_instruction=(
            "You are an archery technique coach. "
            "The user will send a short video of an archer shooting. "
            "Analyse the shot timing, draw/hold stability, release, and follow-through. "
            "Mention any jerks, rushing, collapsing, or over-holding.\n"
            "Reply in Hinglish/English with clear bullet points."
        ),
    )

    prompt = "Analyze this archery video and explain any form, timing, or release issues."

    resp = model.generate_content([prompt, file_info])
    return resp.text



def transcribe_voice_to_text(audio_path: str) -> str:
    """
    Voice helper:
    - Sends the user's audio file (mp3 / wav) to Gemini.
    - Converts Hindi / Hinglish/ English speech into clean text.
    - Output becomes ready for the BowSensei pipeline.
    """

    # Upload audio file (Kaggle path works directly)
    audio_file = genai.upload_file(audio_path)
    print("\n==============================")
    print("Uploaded audio:", audio_file.name, "| state:", audio_file.state.name)
    print("\n==============================")

    # Wait for Gemini to finish processing the uploaded file
    while True:
        file_info = genai.get_file(audio_file.name)
        state = file_info.state.name
        print("Current state:", state)

        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError("Audio processing failed on Gemini side.")

        time.sleep(5)     # Wait a bit and check again

    # Define the LLM for transcription
    model = genai.GenerativeModel(
        "models/gemini-2.5-flash",
        system_instruction=(
            "You are a transcription assistant for archery coaching. "
            "Convert spoken Hindi/Hinglish/English about archery problems into clean, concise text. "
            "Focus only on archery-related content; do not add greetings or extra details."
        ),
    )

    # User prompt sent with file
    prompt = (
        "This audio contains an archer describing their problem. "
        "Transcribe only the archery-related details into clean Hinglish text. "
        "Do not add greetings or unrelated sentences."
    )

    # Generate transcription
    resp = model.generate_content([prompt, file_info])
    return resp.text



def run_archery_session(user_message: str) -> dict:
    """
    Core BowSensei pipeline:

    1) Intake Agent  -> builds structured archer profile + issues
    2) Coach Agent   -> generates likely mistakes + drills + explanations
    3) Plan Agent    -> generates a 7-day training plan with daily focus

    It also logs a small summary into SESSION_LOG for history analysis.
    """

    print("\n====================================================")
    print(">> Running Intake Agent...")
    structured_info = run_intake_agent(user_message)

    print(">> Running Coach Agent...")
    coach_output = run_coach_agent(structured_info)

    print(">> Running Plan Agent...")
    plan_output = run_plan_agent(structured_info, coach_output)
    print("\n======================================================")

    result = {
        "structured_info": structured_info,
        "coach_output": coach_output,
        "plan_output": plan_output,
    }

    # ===== MEMORY LOGGING PART =====
    global SESSION_LOG

    session_entry = {
        "bow_type": structured_info.get("bow_type", ""),
        "draw_weight_lbs": structured_info.get("draw_weight_lbs", 0),
        "experience_level": structured_info.get("experience_level", ""),
        "main_issues": structured_info.get("main_issues", []),
        "mistakes": coach_output.get("mistake_analysis", []),
        # optional: "timestamp": ...
    }

    SESSION_LOG.append(session_entry)

    # Keep only the last MAX_HISTORY sessions
    if len(SESSION_LOG) > MAX_HISTORY:
        SESSION_LOG = SESSION_LOG[-MAX_HISTORY:]

    return result



def analyze_history_last_n(n: int = 3) -> str:
    """
    Look at the last `n` sessions and:

    - Find which issues / mistakes keep repeating
    - Return a short progress note summarising the pattern
    """

    if not SESSION_LOG:
        return "There are no previous sessions yet."

    recent = SESSION_LOG[-n:]  # last n sessions

    # Collect all main_issues + mistakes
    all_issues: list[str] = []
    for s in recent:
        all_issues.extend(s.get("main_issues", []))
        all_issues.extend(s.get("mistakes", []))

    if not all_issues:
        return "No clear repeating issues found in the last few sessions."

    counts = Counter(all_issues)
    top = counts.most_common(3)

    lines: list[str] = []
    lines.append(f"Pattern from the last {len(recent)} sessions:")

    # Only keep issues that repeat (frequency >= 2)
    repeated = [(issue, freq) for issue, freq in top if freq > 1]

    if not repeated:
        lines.append(
            "- No issue is repeating strongly yet; things look fairly balanced."
        )
    else:
        for issue, freq in repeated:
            lines.append(f'- "{issue}" has repeated {freq} times.')

    lines.append(
        "Focus on these repeating issues first before adding new experiments."
    )

    return "\n".join(lines)



def run_archery_session_flex(
    user_text: str | None = None,
    image_path: str | None = None,
    voice_path: str | None = None,
    video_path: str | None = None,
) -> dict:
    """
    Flexible multimodal entrypoint for BowSensei.

    Supports combinations like:
    - Text only
    - Image only
    - Voice only
    - Video only
    - Image + Text
    - Image + Voice
    - Video + Text
    - (any mix of the above)

    It converts all available inputs into one combined Hinglish description,
    then passes that description to `run_archery_session()`.
    """

    if not any([user_text, image_path, voice_path, video_path]):
        raise ValueError(
            "Please provide at least one of: user_text, image_path, "
            "voice_path, or video_path."
        )

    description_parts: list[str] = []

    # 1) Typed text
    if user_text is not None and user_text.strip():
        description_parts.append("User Text Description:\n" + user_text.strip())

    # 2) Voice -> text
    if voice_path is not None:
        print("\n==============================")
        print(">>> Transcribing voice...")
        voice_text = transcribe_voice_to_text(voice_path)
        print("\n=== Voice Transcript ===")
        print(voice_text)
        print("\n==============================")
        description_parts.append(
            "Voice Description (Transcribed):\n" + voice_text.strip()
        )

    # 3) Image analysis
    if image_path is not None:
        print("\n==============================")
        print(">>> Analyzing image...")
        image_analysis = analyze_archery_image(image_path)
        print("\n==============================")
        print("\n=== Image Analysis ===")
        print("\n==============================")
        print(image_analysis)
        print("\n==============================")
        description_parts.append(
            "Image Observation:\n" + image_analysis.strip()
        )

    # 4) Video analysis
    if video_path is not None:
        print(">>> Analyzing video...")
        video_analysis = analyze_archery_video(video_path)
        print("\n==============================")
        print("\n=== Video Analysis ===")
        print("\n==============================")
        print(video_analysis)
        description_parts.append(
            "Video Observation:\n" + video_analysis.strip()
        )

    combined_message = "\n\n".join(description_parts)
    
    print("\n===========================================================")
    print("\n>>> Running BowSensei pipeline on combined description...\n")
    print("\n===========================================================")
    result = run_archery_session(combined_message)
    return result



def pretty_print_result(result: dict):
    """Render and print the final BowSensei coaching report."""

    # Extract nested blocks
    info = result.get("structured_info", {})
    coach = result.get("coach_output", {})
    plan_block = result.get("plan_output", {})
    week_plan = plan_block.get("week_plan", {})

    print("\n" + "="*45)
    print("ğŸ�¹  BOWSENSEI â€“ ARCHERY COACH REPORT")
    print("="*45)

    # ------------------------------------------------------
    # Archer Profile
    # ------------------------------------------------------
    print("\nğŸ“‹ ARCHER PROFILE")
    print("---------------------------------------------")
    print(f"- Bow Type: {info.get('bow_type', 'Not detected')}")
    print(f"- Draw Weight: {info.get('draw_weight_lbs', 'Not detected')} lbs")
    print(f"- Experience Level: {info.get('experience_level', 'Not detected')}")

    main_issues = info.get("main_issues", [])
    if main_issues:
        print("- Main Issues: " + ", ".join(main_issues))

    print("\n============================================")

    # ------------------------------------------------------
    # Likely Mistakes
    # ------------------------------------------------------
    print("\nâ�Œ LIKELY MISTAKES")
    print("---------------------------------------------")
    mistakes = coach.get("mistake_analysis", [])
    if mistakes:
        for m in mistakes:
            print(f"- {m}")
    else:
        print(". No mistakes detected (fallback).")

    print("\n=============================================")
    # ------------------------------------------------------
    # Suggested Drills
    # ------------------------------------------------------
    print("\nâœ… SUGGESTED DRILLS")
    print("---------------------------------------------")
    drills = coach.get("drills", [])
    if drills:
        for d in drills:
            print(f"- {d}")
    else:
        print(". No drills available (fallback).")

    print("\n=============================================")

    # ------------------------------------------------------
    # Why These Drills Help
    # ------------------------------------------------------
    print("\nğŸ§  WHY THESE DRILLS HELP")
    print("---------------------------------------------")
    explanations = coach.get("explanations", [])
    if explanations:
        for e in explanations:
            print(f"- {e}")
    else:
        print("No explanation provided (fallback).")

    
    print("\n=============================================")

    # ------------------------------------------------------
    # 7-Day Training Plan
    # ------------------------------------------------------
    print("\nğŸ“… 7-DAY TRAINING PLAN")
    print("---------------------------------------------")
    if week_plan:
        for day, text in week_plan.items():
            print(f"{day}: {text}")
    else:
        print("No plan generated (fallback).")

    print("\n=============================================")

    # ------------------------------------------------------
    # Today's Single Focus
    # ------------------------------------------------------
    print("\nğŸ�¯ TODAY'S SINGLE FOCUS")
    print("---------------------------------------------")
    if mistakes:
        print(f"Focus on: {mistakes[0]}")
    elif main_issues:
        print(f"Focus on: {main_issues[0]}")
    else:
        print("Focus: Maintain good form and stability.")

    print("\n=============================================")

    # ------------------------------------------------------
    # Progress Note (History Based)
    # ------------------------------------------------------
    print("\nğŸ“ˆ PROGRESS NOTE (History Based)")
    print("---------------------------------------------")
    history_text = analyze_history_last_n(n=3)
    print(history_text)

    print("\n" + "="*45 + "\n")



def demo_bowsensei(
    user_text: str | None = None,
    image_path: str | None = None,
    voice_path: str | None = None,
    video_path: str | None = None,
):
    """
    Unified demo helper.

    - The user can provide input in any combination:
      text, image, voice, video, or mixed.
    - Internally, this calls run_archery_session_flex(...)
      and then uses pretty_print_result(...) to display
      the final BowSensei coaching report.
    """
    result = run_archery_session_flex(
        user_text=user_text,
        image_path=image_path,
        voice_path=voice_path,
        video_path=video_path,
    )
    pretty_print_result(result)



demo_bowsensei(
    user_text = """
I shoot a 40 lbs recurve bow.
I practice at 70 meters.
At the last moment my aim starts shaking and my release becomes rushed.
"""
)



demo_bowsensei(
    image_path="/kaggle/input/archery-form-samples/Screenshot_2024-09-04-16-16-15-638_com.google.android.youtube.jpg"
)



demo_bowsensei(
    voice_path="/kaggle/input/archery-form-samples2/WhatsApp Audio 2025-11-28 at 3.24.31 PM (online-audio-converter.com).mp3"
)


demo_bowsensei(
    video_path="/kaggle/input/archery-form-samples1/VID_20250124_090004.mp4"
)


demo_bowsensei(
    user_text="At 70 meters, my arrows are drifting to the left.",
    image_path="/kaggle/input/archery-form-samples/Screenshot_2024-09-04-16-16-15-638_com.google.android.youtube.jpg",
)


demo_bowsensei(
    image_path = "/kaggle/input/archery-form-samples/Screenshot_2024-09-04-16-16-15-638_com.google.android.youtube.jpg",
    voice_path = "/kaggle/input/archery-form-samples2/WhatsApp Audio 2025-11-28 at 3.24.31 PM (online-audio-converter.com).mp3"
)




demo_bowsensei(
    user_text="At 70 meters, my arrows are drifting to the left.",
    video_path="/kaggle/input/archery-form-samples1/VID_20250124_090004.mp4"
)




demo_bowsensei(
    image_path = "/kaggle/input/archery-form-samples/Screenshot_2024-09-04-16-16-15-638_com.google.android.youtube.jpg",
    video_path="/kaggle/input/archery-form-samples1/VID_20250124_090004.mp4"
)


demo_bowsensei(
    video_path="/kaggle/input/archery-form-samples1/VID_20250124_090004.mp4",
    voice_path = "/kaggle/input/archery-form-samples2/WhatsApp Audio 2025-11-28 at 3.24.31 PM (online-audio-converter.com).mp3"
)


# === Model List & Rate Limit Check ===

import google.generativeai as genai

print("\n==============================")
print("ğŸ“Œ Available Models:")
print("\n==============================")
for m in genai.list_models():
    print("-", m.name)

# Optional: Rate limit check for current model
model_name = "gemini-2.0-flash"     # change if needed

print("\n==============================")
print("\nğŸ“‰ Checking rate limits for:", model_name)
print("\n==============================")

try:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content("check rate limits")
    print("Rate limit OK âœ”ï¸�")
except Exception as e:
    print("Rate limit issue / quota exceeded â�Œ")
    print(e)



# === Pretty JSON Viewer ===

import json

def pretty_json(data):
    """
    Utility function to print JSON output in a clean, readable format.
    """
    print(json.dumps(data, indent=2, ensure_ascii=False))

# Example usage:
# pretty_json(result)



def debug_agent_output(step_name, raw_output):
    print(f"\n===== DEBUG: {step_name} =====")
    print(raw_output[:500], "..." if len(raw_output) > 500 else "")



# 8.1 â€“ Google Cloud + Agent Engine Setup

import os
import random
import vertexai
from vertexai import agent_engines
from kaggle_secrets import UserSecretsClient

print("Imports âœ”")

# 1) Load GCP from Kaggle
user_secrets = UserSecretsClient()
gcloud_cred = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(gcloud_cred)
print("\n==============================")
print("GCP credentials loaded from Kaggle Secrets âœ”")
print("\n==============================")

# 2) Here, Write GCP PROJECT_ID 
PROJECT_ID = "molten-reserve-479706-u7"   # <-- yaha apna project id daalna hai
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

if PROJECT_ID == "your-gcp-project-id" or not PROJECT_ID:
    raise ValueError("Please PROJECT_ID ko apne actual GCP project se replace karo.")

print("\n==============================")
print(f"PROJECT_ID = {PROJECT_ID}")
print("\n==============================")


# 8.2 â€“ Create ADK agent package for BowSensei

!mkdir -p bowsensei_agent
print("bowsensei_agent/ directory created âœ”")


%%writefile bowsensei_agent/requirements.txt
google-adk
opentelemetry-instrumentation-google-genai
google-cloud-aiplatform



%%writefile bowsensei_agent/.env
GOOGLE_CLOUD_LOCATION="global"
GOOGLE_GENAI_USE_VERTEXAI=1


%%writefile bowsensei_agent/agent.py
import os
import vertexai
from google.adk.agents import Agent

# Vertex init (Agent Engine runtime ke andar)
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)

# ---- BowSensei root agent ----

BOWSENSEI_INSTRUCTION = """
You are BowSensei, an AI-Powered Archery Coach for 70m recurve archers.

Your job:
1. Take the user's description of their archery problem (Hinglish allowed).
2. Infer their bow type, draw weight, experience level and main issues.
3. Analyse likely form / technique mistakes.
4. Suggest 4â€“6 concrete drills.
5. Create a realistic 7-day training plan (30â€“90 minutes per day).
6. Respond in clear Hinglish, bullet-point style, easy for Indian archers.

Important:
- Assume Olympic recurve by default unless user clearly says compound / barebow.
- Focus on technique, form, stability, mental process. Do NOT change equipment setup.
- If user shares tournament pressure issues, give mental + process cues also.
"""

root_agent = Agent(
    name="bowsensei_archery_coach",
    model="gemini-2.5-flash-lite",
    description="Archery coaching agent (BowSensei) deployed on Vertex AI Agent Engine.",
    instruction=BOWSENSEI_INSTRUCTION,
)



%%writefile bowsensei_agent/.agent_engine_config.json
{
  "min_instances": 0,
  "max_instances": 1,
  "resource_limits": {
    "cpu": "1",
    "memory": "1Gi"
  }
}



# 8.3 â€“ Select region + deploy to Agent Engine

import random

regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
DEPLOY_REGION = random.choice(regions_list)

print("\n==============================")
print("Selected region:", DEPLOY_REGION)
print("\n==============================")

# Command: ADK deploy
!adk deploy agent_engine --project=$PROJECT_ID --region=$DEPLOY_REGION bowsensei_agent



# 8.4 â€“ Connect & test deployed BowSensei

import vertexai
from vertexai import agent_engines

vertexai.init(project=PROJECT_ID, location=DEPLOY_REGION)

agents_list = list(agent_engines.list())
if not agents_list:
    raise RuntimeError("No agents found â€“ Please! Check The Deployment Setup Again.")

remote_agent = agents_list[0]
print("Connected to:", remote_agent.resource_name)

# Simple test query
user_query = """
My bow is a 40 lbs recurve bow, and at 70 meters the arrows are drifting to the left.
The draw-hold is weak, and under tournament pressure the release becomes rushed.
"""
print("\n==============================")
print("\n--- BowSensei (deployed) response ---\n")
print("\n==============================")
async for item in remote_agent.async_stream_query(
    message=user_query,
    user_id="demo_archer_1",
):
    print(item)



# 8.5 â€“ Cleanup: delete remote agent (to save cost)

#agent_engines.delete(resource_name=remote_agent.resource_name, force=True)
#print("\n==============================")
#print("Remote BowSensei agent deleted âœ”")
#print("\n==============================")


