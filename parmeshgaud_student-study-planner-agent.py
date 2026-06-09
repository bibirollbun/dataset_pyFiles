# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ====== 2. SETUP & DATA INPUT ======

import pandas as pd
from datetime import datetime, timedelta
from IPython.display import display

# 1) Define subjects and exams
# ---------------------------------------------------
# You should edit this section for your real exams.
# weakness_level: 1 = very strong, 5 = very weak

data = {
    "subject": ["Maths", "Operating Systems", "Networking", "DBMS"],
    "exam_date": ["2025-03-20", "2025-03-18", "2025-03-22", "2025-03-25"],
    "weakness_level": [4, 5, 3, 2],   # higher = weaker
    "current_score": [45, 40, 55, 60],
    "target_score": [80, 75, 80, 85],
}

subjects_df = pd.DataFrame(data)
subjects_df["exam_date"] = pd.to_datetime(subjects_df["exam_date"])

print("ğŸ“‹ Subjects & Exams:")
display(subjects_df)

# 2) Study constraints (you can customize)
# ---------------------------------------------------
DAILY_STUDY_HOURS = 4  # how many hours per day you can study

# Plan from today's date until the last exam
PLAN_START = datetime.today().date()
PLAN_END = subjects_df["exam_date"].max().date()

print(f"\nğŸ—“ï¸� Planning window: {PLAN_START} to {PLAN_END}")


# ====== 3. STUDY PLANNER AGENT ======

class StudyPlannerAgent:
    """
    A simple rule-based agent that:
    - calculates urgency for each subject
    - creates a daily timetable
    - logs progress
    - adapts the schedule based on progress
    """

    def __init__(self, subjects_df, daily_hours, start_date, end_date):
        self.subjects = subjects_df.copy()
        self.daily_hours = daily_hours
        self.start_date = start_date
        self.end_date = end_date
        self.schedule = None            # daily timetable (DataFrame)
        self.progress_log = []          # list of dict entries

    # ---------- TOOL 1: calculate urgency score ----------
    def _compute_urgency(self):
        """
        Urgency is higher when:
        - the subject is weaker (higher weakness_level)
        - the exam is closer (fewer days left)
        """
        today = self.start_date
        urgency_scores = []

        for _, row in self.subjects.iterrows():
            days_left = max((row["exam_date"].date() - today).days, 1)
            weakness = row["weakness_level"]
            # Simple urgency formula
            urgency = (weakness * 2) + (10 / days_left)
            urgency_scores.append(urgency)

        self.subjects["urgency"] = urgency_scores

    # ---------- TOOL 2: generate full schedule ----------
    def generate_schedule(self):
        """
        Generates a timetable from start_date to end_date.

        For each day:
        - compute each subject's share based on urgency
        - allocate a portion of DAILY_STUDY_HOURS
        """
        self._compute_urgency()

        dates = []
        subjects_planned = []
        hours_planned = []

        current = self.start_date
        while current <= self.end_date:
            temp = self.subjects.copy()
            total_urgency = temp["urgency"].sum()

            # avoid division by zero
            if total_urgency == 0:
                temp["share"] = 1.0 / len(temp)
            else:
                temp["share"] = temp["urgency"] / total_urgency

            for _, row in temp.iterrows():
                allocated_hours = round(self.daily_hours * row["share"], 1)
                if allocated_hours <= 0:
                    continue

                dates.append(current)
                subjects_planned.append(row["subject"])
                hours_planned.append(allocated_hours)

            current += timedelta(days=1)

        self.schedule = pd.DataFrame({
            "date": dates,
            "subject": subjects_planned,
            "planned_hours": hours_planned
        })

        return self.schedule

    # ---------- TOOL 3: log daily progress ----------
    def log_progress(self, date, subject, studied_hours, feeling=None):
        """
        Save what the student actually did.
        """
        entry = {
            "date": pd.to_datetime(date).date(),
            "subject": subject,
            "studied_hours": studied_hours,
            "feeling": feeling
        }
        self.progress_log.append(entry)

    # ---------- TOOL 4: adapt plan based on progress ----------
    def adapt_plan(self):
        """
        Simple adaptation rule:
        - Compute average studied_hours per subject.
        - If avg_hours < 1.0, we treat the subject as harder than expected,
          so we increase its weakness_level (up to max 5).
        - Then we regenerate the timetable.
        """
        if not self.progress_log:
            print("No progress logged yet. Nothing to adapt.")
            return self.schedule

        prog_df = pd.DataFrame(self.progress_log)
        summary = prog_df.groupby("subject")["studied_hours"].mean().reset_index()
        summary.rename(columns={"studied_hours": "avg_hours"}, inplace=True)

        merged = self.subjects.merge(summary, on="subject", how="left")
        merged["avg_hours"].fillna(self.daily_hours / len(self.subjects), inplace=True)

        for idx, row in merged.iterrows():
            if row["avg_hours"] < 1.0:
                # increase weakness level for hard / under-studied subjects
                new_weakness = min(row["weakness_level"] + 1, 5)
                self.subjects.loc[self.subjects["subject"] == row["subject"], "weakness_level"] = new_weakness

        print("ğŸ”� Adapted weakness levels based on progress:")
        display(self.subjects[["subject", "weakness_level"]])

        # regenerate schedule with new weakness levels
        return self.generate_schedule()


# ====== 4. INITIAL PLANNING ======

# Create the agent
agent = StudyPlannerAgent(
    subjects_df=subjects_df,
    daily_hours=DAILY_STUDY_HOURS,
    start_date=PLAN_START,
    end_date=PLAN_END,
)

# Generate the initial timetable
initial_schedule = agent.generate_schedule()

print("âœ… Initial Study Timetable (first 20 rows):")
display(initial_schedule.head(20))

# Optional: show how many hours per subject in total
total_hours_per_subject = initial_schedule.groupby("subject")["planned_hours"].sum().reset_index()
total_hours_per_subject = total_hours_per_subject.sort_values(by="planned_hours", ascending=False)

print("\nğŸ“Š Total planned hours per subject:")
display(total_hours_per_subject)


# ====== 5. MONITORING & ADAPTATION ======

# Example: simulate today's progress (you can change these values)
today = PLAN_START

# Suppose the student studied less Maths and felt confused,
# but studied more OS and felt okay.
agent.log_progress(date=today, subject="Maths", studied_hours=0.5, feeling="confused")
agent.log_progress(date=today, subject="Operating Systems", studied_hours=2.0, feeling="ok")
agent.log_progress(date=today, subject="Networking", studied_hours=0.5, feeling="tough")

print("ğŸ“� Progress log so far:")
display(pd.DataFrame(agent.progress_log))

# Now let the agent adapt the plan
updated_schedule = agent.adapt_plan()

print("\nğŸ”� Updated Study Timetable (first 20 rows):")
display(updated_schedule.head(20))

# Compare new total hours per subject
updated_total_hours = updated_schedule.groupby("subject")["planned_hours"].sum().reset_index()
updated_total_hours = updated_total_hours.sort_values(by="planned_hours", ascending=False)

print("\nğŸ“Š Updated total planned hours per subject:")
display(updated_total_hours)


# ====== 6. EXAMPLE INTERACTION ======

def show_plan_for_date(schedule, date):
    """Utility: show which subjects to study on a given date."""
    date = pd.to_datetime(date).date()
    day_plan = schedule[schedule["date"] == date]

    if day_plan.empty:
        print(f"No study plan found for {date}.")
    else:
        print(f"ğŸ“… Study plan for {date}:")
        display(day_plan)


# Show today's and tomorrow's plan using the updated schedule
show_plan_for_date(updated_schedule, PLAN_START)

tomorrow = PLAN_START + timedelta(days=1)
show_plan_for_date(updated_schedule, tomorrow)


# ====== 8. STUDY MATERIAL RECOMMENDER AGENT ======

# A curated dictionary of study resources per subject.
# You can modify or extend this with your favourite channels / notes.

STUDY_MATERIALS = {
    "Maths": {
        "youtube_playlists": [
            {
                "title": "Engineering Mathematics / Discrete Maths (GATE-style notes & PDFs)",
                "url": "https://usemynotes.com/free-gate-cse-notes-by-toppers-download-pdfs/"
            }
        ],
        "pdf_notes": [
            {
                "title": "GATE CSE Notes (includes Engineering Maths, Discrete Maths, etc.)",
                "url": "https://usemynotes.com/free-gate-cse-notes-by-toppers-download-pdfs/"
            }
        ],
        "practice_sets": [
            {
                "title": "Topic-wise practice from GATE CSE notes (Maths sections)",
                "url": "https://usemynotes.com/free-gate-cse-notes-by-toppers-download-pdfs/"
            }
        ],
    },

    "Operating Systems": {
        "youtube_playlists": [
            {
                "title": "Operating System Complete Playlist â€“ Gate Smashers",
                "url": "https://www.youtube.com/playlist?list=PLxCzCOWd7aiGz9donHRrE9I3Mwn6XdP8p"
            }
        ],
        "pdf_notes": [
            {
                "title": "Operating System Lecture Notes (B.Tech)",
                "url": "https://sriindu.ac.in/wp-content/uploads/2023/10/R20CSE2202-OPERATING-SYSTEMS.pdf"
            },
            {
                "title": "GATE CSE OS Notes (Toppersâ€™ PDFs)",
                "url": "https://usemynotes.com/free-gate-cse-notes-by-toppers-download-pdfs/"
            }
        ],
        "practice_sets": [
            {
                "title": "Must-do OS interview & concept questions â€“ TakeUForward SDE Core Sheet",
                "url": "https://takeuforward.org/interviews/must-do-questions-for-dbms-cn-os-interviews-sde-core-sheet/"
            }
        ],
    },

    "Networking": {
        "youtube_playlists": [
            {
                "title": "Computer Networks Complete Playlist â€“ Gate Smashers",
                "url": "https://www.youtube.com/playlist?list=PLxCzCOWd7aiGFBD2-2joCpWOLUrDLvVV_"
            }
        ],
        "pdf_notes": [
            {
                "title": "Data Communication and Computer Networks â€“ Lecture Notes (PDF)",
                "url": "https://igitsarang.ac.in/assets/documents/coursematerial/4th_etc_dccn_1757336112.pdf"
            },
            {
                "title": "GATE CSE CN Notes (Toppersâ€™ PDFs)",
                "url": "https://usemynotes.com/free-gate-cse-notes-by-toppers-download-pdfs/"
            }
        ],
        "practice_sets": [
            {
                "title": "Must-do CN interview & concept questions â€“ TakeUForward SDE Core Sheet",
                "url": "https://takeuforward.org/interviews/must-do-questions-for-dbms-cn-os-interviews-sde-core-sheet/"
            }
        ],
    },

    "DBMS": {
        "youtube_playlists": [
            {
                "title": "DBMS Complete Playlist â€“ Gate Smashers",
                "url": "https://www.youtube.com/playlist?list=PLxCzCOWd7aiFAN6I8CuViBuCdJgiOkT2Y"
            }
        ],
        "pdf_notes": [
            {
                "title": "DBMS Lecture Notes (B.Tech)",
                "url": "https://mrcet.com/downloads/digital_notes/CSE/II%20Year/DBMS.pdf"
            },
            {
                "title": "GATE CSE DBMS Notes (Toppersâ€™ PDFs)",
                "url": "https://usemynotes.com/free-gate-cse-notes-by-toppers-download-pdfs/"
            }
        ],
        "practice_sets": [
            {
                "title": "Must-do DBMS interview & concept questions â€“ TakeUForward SDE Core Sheet",
                "url": "https://takeuforward.org/interviews/must-do-questions-for-dbms-cn-os-interviews-sde-core-sheet/"
            }
        ],
    },
}


def get_study_materials(subject):
    """
    Return the study materials dict for a given subject name.
    Subject should match the keys used in STUDY_MATERIALS,
    e.g. 'Maths', 'Operating Systems', 'Networking', 'DBMS'.
    """
    return STUDY_MATERIALS.get(subject)


def show_study_materials(subject):
    """
    Pretty-print the resources for a subject.
    """
    resources = get_study_materials(subject)
    if resources is None:
        print(f"â�Œ No resources found for subject: {subject}")
        return

    print(f"\nğŸ“š Study Resources for: {subject}")

    # YouTube playlists
    playlists = resources.get("youtube_playlists", [])
    if playlists:
        print("\nâ–¶ YouTube Playlists:")
        for i, item in enumerate(playlists, start=1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")

    # PDF notes
    pdfs = resources.get("pdf_notes", [])
    if pdfs:
        print("\nğŸ“„ PDF Notes:")
        for i, item in enumerate(pdfs, start=1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")

    # Practice sets / websites
    practice = resources.get("practice_sets", [])
    if practice:
        print("\nğŸ§ª Practice / Question Sets:")
        for i, item in enumerate(practice, start=1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")


print("âœ… Study Material Recommender Agent loaded.")


# ====== 9. EXAMPLE: RECOMMEND MATERIALS FOR ALL SUBJECTS ======

for subj in subjects_df["subject"]:
    show_study_materials(subj)


# ====== 10. DAILY ONE-SHOT DASHBOARD ======

def show_plan_with_resources(date, schedule=None):
    """
    Show the study plan for a given date AND the recommended resources
    for each subject that appears in that plan.

    If `schedule` is not provided, it tries to use:
      - updated_schedule (if it exists), otherwise
      - initial_schedule
    """
    # Pick which schedule to use
    if schedule is None:
        # try updated_schedule first
        try:
            _ = updated_schedule  # will error if not defined
            schedule = updated_schedule
        except NameError:
            # fall back to initial_schedule
            try:
                _ = initial_schedule
                schedule = initial_schedule
            except NameError:
                print("â�Œ No schedule found. Please generate the timetable first.")
                return

    # Normalize date
    date = pd.to_datetime(date).date()

    # Filter the plan for this date
    day_plan = schedule[schedule["date"] == date]

    if day_plan.empty:
        print(f"ğŸ“… No study plan found for {date}. It might be a free day or out of range.")
        return

    # Aggregate hours per subject (in case there are multiple entries)
    day_summary = (
        day_plan.groupby("subject")["planned_hours"]
        .sum()
        .reset_index()
        .sort_values("planned_hours", ascending=False)
    )

    print(f"ğŸ“… Daily Study Dashboard for {date}\n")
    print("ğŸ•’ Planned hours per subject:")
    display(day_summary)

    # For each subject in today's plan, show study materials
    for subj in day_summary["subject"]:
        show_study_materials(subj)


# ğŸ”� Example usage:
# show_plan_with_resources(PLAN_START)
# or
# show_plan_with_resources("2025-03-15")


today = datetime.today().date()
first_exam = subjects_df["exam_date"].min().date()

# If today is after the first exam, rewind planning to 30 days before the first exam
if today > first_exam:
    PLAN_START = first_exam - timedelta(days=30)
else:
    PLAN_START = today



# Todayâ€™s full dashboard: plan + links
show_plan_with_resources(PLAN_START)


show_plan_with_resources("2025-11-20")

