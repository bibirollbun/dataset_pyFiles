# Shared session memory for the entire agent workflow
session_memory = {
    "tasks": [],
    "durations": {},
    "schedule": []
}



def agent_collect_tasks(user_input):
    tasks = [task.strip().capitalize() for task in user_input.split(",") if task.strip()]
    session_memory["tasks"] = tasks
    return tasks

# Example user input
user_input = "study math, workout, make breakfast, read book"
agent_collect_tasks(user_input)



def estimate_duration(task):
    task = task.lower()
    if "study" in task:
        return 90
    if "workout" in task or "gym" in task:
        return 60
    if "breakfast" in task or "cook" in task:
        return 30
    if "read" in task:
        return 45
    return 40  # Default duration

def agent_estimate_durations(tasks):
    durations = {task: estimate_duration(task) for task in tasks}
    session_memory["durations"] = durations
    return durations

agent_estimate_durations(session_memory["tasks"])



from datetime import datetime, timedelta

def agent_create_schedule(durations, start_time="06:00 AM"):
    schedule = []
    current_time = datetime.strptime(start_time, "%I:%M %p")

    for task, minutes in durations.items():
        start = current_time.strftime("%I:%M %p")
        end_time = current_time + timedelta(minutes=minutes)
        end = end_time.strftime("%I:%M %p")

        schedule.append({
            "task": task,
            "start": start,
            "end": end,
            "duration": minutes
        })

        current_time = end_time  # update time

    session_memory["schedule"] = schedule
    return schedule

agent_create_schedule(session_memory["durations"])



import pandas as pd

# Convert the final routine into a pandas DataFrame
routine_df = pd.DataFrame(session_memory["schedule"])

# Reorder columns for clarity
routine_df = routine_df[["task", "start", "end", "duration"]]

# Display the table
routine_df



import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

# Extract schedule from session_memory
schedule = session_memory["schedule"]
tasks = [entry['task'] for entry in schedule]
start_times = [datetime.strptime(entry['start'], "%I:%M %p") for entry in schedule]
durations = [entry['duration'] for entry in schedule]
end_times = [start + timedelta(minutes=dur) for start, dur in zip(start_times, durations)]

# Convert to minutes since midnight
start_minutes = [t.hour*60 + t.minute for t in start_times]
end_minutes = [t.hour*60 + t.minute for t in end_times]

# Figure height
fig_height = max(6, len(tasks)*0.8)
fig, ax = plt.subplots(figsize=(14, fig_height))

# Plot bars with start-end labels
for i, task in enumerate(tasks):
    ax.barh(task, durations[i], left=start_minutes[i], height=0.5, color=plt.cm.tab20(i % 20))
    ax.text(
        start_minutes[i] + durations[i]/2,
        i,
        f"{schedule[i]['start']} - {schedule[i]['end']}",
        va='center', ha='center', color='white', fontsize=9, fontweight='bold'
    )

# Compact x-axis limits
min_time = min(start_minutes)
max_time = max(end_minutes)
buffer = 10  # 10 minutes buffer
ax.set_xlim(min_time - buffer, max_time + buffer)

# Determine dynamic x-axis interval based on total duration
total_span = max_time - min_time
if total_span <= 60:
    interval = 5   # 5 min ticks for short schedules
elif total_span <= 180:
    interval = 15  # 15 min ticks for ~3 hours
elif total_span <= 360:
    interval = 30  # 30 min ticks for ~6 hours
else:
    interval = 60  # 1 hour ticks for long schedules

# Set x-axis ticks and labels
xticks = list(range(min_time - buffer, max_time + buffer + 1, interval))
xticklabels = [f"{(h//60)%12 if (h//60)%12!=0 else 12}:{h%60:02d} {'AM' if h//60<12 else 'PM'}" for h in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=45, ha='right')

ax.set_xlabel("Time")
ax.set_ylabel("Tasks")
ax.set_title("Daily Routine Timeline (Auto-scaled X-axis)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()


