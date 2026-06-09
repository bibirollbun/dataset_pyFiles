# cell-1: Environment & optional Gemini setup (safe if GEMINI key missing)
!pip install -q google-generativeai

import datetime
import random
import json
import re
import os

# Try to set up Gemini if a key is provided in Kaggle secrets (optional)
MODEL_AVAILABLE = False
try:
    from kaggle_secrets import UserSecretsClient
    import google.generativeai as genai

    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        MODEL_AVAILABLE = True
        print("âœ… Gemini available and configured.")
    else:
        print("âš ï¸� GEMINI_API_KEY not found in Kaggle secrets â€” running offline planner.")
except Exception as e:
    # If import fails or no key, continue with offline planner
    print("âš ï¸� Gemini unavailable (or import failed). Running offline deterministic planner.")
    # print(e)  # uncomment for debugging if needed

print("âœ… Environment ready.")



# cell-2: Helper tools and schedule builder

def get_current_time():
    """Return current time in IST as datetime object + formatted string."""
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(ist)
    return now, now.strftime("%I:%M %p IST")

def calculate_duration(task):
    """Estimate task duration (minutes). More keys and fuzzy matching."""
    durations = {
        "dsa": 25, "data structure":25, "algorithms": 25,
        "study": 25, "assignment": 50, "homework": 45, "read": 20,
        "code": 45, "coding":45, "notes": 20, "revise": 25, "sleep": 60,
        "nap": 20, "break": 10, "chai": 10, "project": 60, "practice":30
    }
    task_lower = task.lower()
    # exact key or substring match
    for key, val in durations.items():
        if key in task_lower:
            return val
    # fallback heuristics: short tasks (<=2 words) -> 25, else 45
    if len(task_lower.split()) <= 2:
        return 25
    return 45

def generate_quote():
    quotes = [
        "Even turtles finish the race â€” just keep crawling. ğŸ�¢",
        "Productivity is designedâ€”schedule one tiny win now (and have chai). â˜•",
        "Start small: 25 minutes beats 0 minutes. You got this!",
        "Make one pomodoro count. Rest like you earned it.",
        "Procrastination is the art of keeping up with yesterday. Start today!"
    ]
    return random.choice(quotes)

def parse_user_input(user_input):
    """
    Extract: total free minutes (approx), tasks list, energy level.
    Example inputs:
      - "I have 3 hours free, low energy, need to finish DSA and sleep early."
      - "2.5 hours, high energy, finish assignment, read notes"
    """
    text = user_input.lower()
    # hours extraction (look for patterns like '3 hours' or '2.5 hours')
    hours = None
    hours_match = re.search(r'(\d+(\.\d+)?)\s*hours?', text)
    if hours_match:
        hours = float(hours_match.group(1))
    else:
        # minutes extraction fallback
        mins_match = re.search(r'(\d+)\s*mins?', text)
        if mins_match:
            hours = float(int(mins_match.group(1)) / 60.0)

    total_minutes = int(hours * 60) if hours else None

    # energy inference
    if "low energy" in text or "tired" in text or "sleep early" in text or "energy low" in text:
        energy = "low"
    elif "high energy" in text or "fresh" in text or "energized" in text:
        energy = "high"
    else:
        energy = "medium"

    # tasks extraction simple: look for keywords after 'need to' or 'finish' or comma separated items
    tasks = []
    # Try heuristics:
    if "need to" in text:
        tail = text.split("need to", 1)[1]
        possible = re.split(r'[.,;]|\band\b|\bthen\b', tail)
        for p in possible:
            p = p.strip()
            if p:
                tasks.extend([t.strip() for t in re.split(r'and|,', p) if t.strip()])
    else:
        # fallback: look for 'finish' or list of nouns
        if "finish" in text:
            tail = text.split("finish", 1)[1]
            possible = re.split(r'[.,;]|\band\b', tail)
            for p in possible:
                p = p.strip()
                if p:
                    tasks.extend([t.strip() for t in re.split(r'and|,', p) if t.strip()])

    # If still none, attempt to pull words that look like tasks
    if not tasks:
        words = re.split(r'[,\s]+', text)
        common_tasks = ["dsa", "assignment", "read", "sleep", "nap", "code", "notes", "project"]
        for w in words:
            if w in common_tasks and w not in tasks:
                tasks.append(w)

    # final cleanup: deduplicate and keep short names
    tasks = [t for t in dict.fromkeys(tasks) if len(t) > 0]  # preserve order
    if not tasks:
        tasks = ["study"]  # default

    return {
        "total_minutes": total_minutes,
        "energy": energy,
        "tasks": tasks
    }

def build_schedule(user_input, start_dt=None):
    """
    Build a lazy Pomodoro-style schedule as markdown.
    - Break tasks into 25-50 min sessions based on energy and calculate_duration
    - Insert short breaks and a longer break after 4 pomodoros
    """
    parsed = parse_user_input(user_input)
    now_dt, now_str = get_current_time() if not start_dt else (start_dt, start_dt.strftime("%I:%M %p IST"))
    total_minutes = parsed["total_minutes"] or 180  # default 3 hours if unspecified
    energy = parsed["energy"]
    tasks = parsed["tasks"]

    # Pomodoro parameters by energy
    if energy == "high":
        work_block = 50
        short_break = 10
    elif energy == "low":
        work_block = 25
        short_break = 10
    else:
        work_block = 30
        short_break = 10

    schedule = []
    cursor = now_dt
    minutes_used = 0
    pomodoro_count = 0

    # Create a queue of task fragments: for each task estimate how many blocks
    task_queue = []
    for t in tasks:
        est = calculate_duration(t)
        # How many blocks this task needs (ceil)
        blocks = max(1, (est + work_block - 1) // work_block)
        for i in range(blocks):
            task_queue.append({"task": t, "block_index": i+1, "blocks_total": blocks})

    # Fill schedule until time runs out or tasks done
    i = 0
    while minutes_used + work_block <= total_minutes and i < len(task_queue):
        item = task_queue[i]
        start_time = cursor
        end_time = start_time + datetime.timedelta(minutes=work_block)
        schedule.append({
            "start": start_time.strftime("%I:%M %p"),
            "end": end_time.strftime("%I:%M %p"),
            "duration_min": work_block,
            "type": "work",
            "task": item["task"],
            "detail": f"Part {item['block_index']}/{item['blocks_total']}"
        })
        cursor = end_time
        minutes_used += work_block
        pomodoro_count += 1
        # Insert break if time remains
        if minutes_used + short_break <= total_minutes:
            bstart = cursor
            bend = bstart + datetime.timedelta(minutes=short_break)
            schedule.append({
                "start": bstart.strftime("%I:%M %p"),
                "end": bend.strftime("%I:%M %p"),
                "duration_min": short_break,
                "type": "break",
                "task": "Break (chai/stand/stretch)",
                "detail": ""
            })
            cursor = bend
            minutes_used += short_break

        # every 4 pomodoros add a longer rest if time allows
        if pomodoro_count % 4 == 0 and minutes_used + 20 <= total_minutes:
            longstart = cursor
            longend = longstart + datetime.timedelta(minutes=20)
            schedule.append({
                "start": longstart.strftime("%I:%M %p"),
                "end": longend.strftime("%I:%M %p"),
                "duration_min": 20,
                "type": "long_break",
                "task": "Long break (nap/long chai/walk)",
                "detail": ""
            })
            cursor = longend
            minutes_used += 20

        i += 1

    # If time remains but tasks done, add chill time / nap
    remaining = total_minutes - minutes_used
    if remaining >= 10:
        schedule.append({
            "start": cursor.strftime("%I:%M %p"),
            "end": (cursor + datetime.timedelta(minutes=remaining)).strftime("%I:%M %p"),
            "duration_min": remaining,
            "type": "chill",
            "task": "Chill / nap / review lightly",
            "detail": ""
        })
        minutes_used += remaining

    # Compose markdown
    md_lines = []
    md_lines.append(f"# Lazy Student Daily Planner")
    md_lines.append(f"**Start (IST):** {now_str}")
    md_lines.append(f"**Total planned time:** {total_minutes} minutes")
    md_lines.append(f"**Energy:** {energy.capitalize()}")
    md_lines.append("")
    md_lines.append("## Schedule")
    for idx, s in enumerate(schedule, 1):
        if s["type"] == "work":
            md_lines.append(f"- **{s['start']} - {s['end']}** â€” Work: `{s['task']}` ({s['detail']}) â€” {s['duration_min']} min")
        elif s["type"] == "break":
            md_lines.append(f"- {s['start']} - {s['end']} â€” Short break: {s['task']} â€” {s['duration_min']} min")
        elif s["type"] == "long_break":
            md_lines.append(f"- {s['start']} - {s['end']} â€” Long break: {s['task']} â€” {s['duration_min']} min")
        else:
            md_lines.append(f"- {s['start']} - {s['end']} â€” {s['task']} â€” {s['duration_min']} min")
    md_lines.append("")
    md_lines.append(f"**Motivational:** {generate_quote()}")
    md = "\n".join(md_lines)
    return md



# cell-3: Agent wrapper (uses local builder) and optional model rewriter (if MODEL_AVAILABLE)
def lazy_planner_agent(user_input, rewrite_with_model=False):
    """
    Primary function you call. By default uses local deterministic planner.
    If rewrite_with_model=True and Gemini is available, it will attempt to
    send the built schedule to Gemini to produce a fancier markdown/result.
    """
    schedule_md = build_schedule(user_input)
    if rewrite_with_model and MODEL_AVAILABLE:
        # OPTIONAL: ask Gemini to rewrite the schedule in the same markdown but fun voice.
        try:
            system_prompt = (
                "You are LazyPlannerAgent, a chill AI. The user wants a friendly, concise markdown "
                "schedule for a procrastinating student. Keep emojis, relaxed tone, and short lines."
            )
            # A simple text-only call â€” adapt as needed for your environment
            # Use model.generate_content when you want advanced features; here's a basic call:
            resp = model.generate_text(
                prompt=system_prompt + "\n\nSchedule:\n" + schedule_md,
                temperature=0.7,
                max_output_tokens=800
            )
            rewritten = resp.result[0].content[0].text if hasattr(resp, 'result') else None
            if rewritten:
                return rewritten
        except Exception as e:
            # If model call fails, fallback to local schedule
            print("âš ï¸� Model rewrite failed, returning local schedule. Error:", str(e))

    return schedule_md

# Quick test (modify the input to test other cases)
if __name__ == "__main__":
    user_input = "I have 3 hours free, low energy, need to finish DSA and sleep early."
    schedule = lazy_planner_agent(user_input)
    print("ğŸ“… Your Lazy-Optimized Schedule:\n")
    print(schedule)
    # Save
    with open("lazy_schedule.md", "w", encoding="utf-8") as f:
        f.write(schedule)
    print("\nâœ… Saved to 'lazy_schedule.md'")


