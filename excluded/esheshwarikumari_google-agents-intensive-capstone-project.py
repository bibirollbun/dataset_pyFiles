def extract_tasks(text):
    prompt = f"Identify all tasks from: {text}"
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']


def prioritize_tasks(tasks):
    prompt = f"Rank these tasks by urgency and importance:\n{tasks}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']


def generate_schedule(tasks):
    prompt = f"Create a realistic daily plan for:\n{tasks}"
    response = client.chat.completions.create(
        model="gpt-4o-reasoning",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']



import google.generativeai as genai
import os

# Configure Gemini using environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_reflection_agent(text):
    """
    Uses Google Gemini 1.5 Flash to refine the schedule or task summary.
    If Gemini fails, returns the original text.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"Improve clarity, structure, and prioritization for this daily plan:\n\n{text}"
        )
        return response.text
    except Exception as e:
        print("Gemini Error:", e)
        return text  # fallback if error or no API key



# Example orchestration
schedule_output = generate_schedule(tasks)

# Refine with Gemini
refined_output = gemini_reflection_agent(schedule_output)

print(refined_output)


event = {
    'summary': task_title,
    'start': {'dateTime': start_time},
    'end': {'dateTime': end_time}
}
service.events().insert(calendarId='primary', body=event).execute()

