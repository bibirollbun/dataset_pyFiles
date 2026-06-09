# Smart Study Planner Agent - No Input Needed
# Works great for Kaggle submission

import pandas as pd
import datetime

# ----------------------------------------------
# FIXED SAMPLE INPUT (EDIT IF YOU WANT)
# ----------------------------------------------

subjects = ["Maths", "Physics", "C Programming"]
hours = 3          # hours per day
days = 5           # plan for 5 days


# ----------------------------------------------
# STUDY PLAN GENERATOR
# ----------------------------------------------

def generate_study_plan(subjects, hours, days):
    schedule = []
    start_date = datetime.date.today()
    
    subjects_cycle = subjects * ((days // len(subjects)) + 1)
    
    for day in range(days):
        date = start_date + datetime.timedelta(days=day)
        subject = subjects_cycle[day]
        
        schedule.append({
            "Day": day + 1,
            "Date": date,
            "Subject": subject,
            "Planned Hours": hours,
            "Completed Hours": 0,
            "Status": "Pending"
        })
    
    return pd.DataFrame(schedule)


# ----------------------------------------------
# UPDATE COMPLETED HOURS
# ----------------------------------------------

def update_progress(df, day, completed_hours):
    df.loc[df["Day"] == day, "Completed Hours"] = completed_hours
    df.loc[df["Day"] == day, "Status"] = "Completed" if completed_hours > 0 else "Skipped"
    return df


# ----------------------------------------------
# RESCHEDULE MISSED TOPICS
# ----------------------------------------------

def reschedule_skipped(df):
    skipped = df[df["Status"] == "Skipped"]["Subject"].tolist()
    
    if not skipped:
        return df
    
    new_rows = []
    last_date = df["Date"].max()
    day_count = df["Day"].max()
    
    for subject in skipped:
        day_count += 1
        last_date += datetime.timedelta(days=1)
        
        new_rows.append({
            "Day": day_count,
            "Date": last_date,
            "Subject": subject,
            "Planned Hours": df["Planned Hours"].iloc[0],
            "Completed Hours": 0,
            "Status": "Pending"
        })
    
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


# ----------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------

print("=== Smart Study Planner Agent (Auto Input Version) ===")

df = generate_study_plan(subjects, hours, days)

print("\nGenerated Study Plan:")
display(df)

# Example progress update (just for demonstration)
df = update_progress(df, day=1, completed_hours=0)  # skipped
df = update_progress(df, day=2, completed_hours=2)  # completed

df = reschedule_skipped(df)

print("\nUpdated Study Plan After Rescheduling:")
display(df)

# Export files
df

