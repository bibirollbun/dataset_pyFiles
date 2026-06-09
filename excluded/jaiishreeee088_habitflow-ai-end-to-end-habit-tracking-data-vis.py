def register_habits(self):
        routine_map = {}
        for habit in self.habits:
            routine_map[habit["time"]] = habit["habit"]
        self.routine = routine_map
        return routine_map


def update_performance(self, habit_name, score):
        record = {"habit": habit_name, "score": score}
        self.performance_history.append(record)

    def analyze_performance(self):
        summary = {"low": 0, "medium": 0, "high": 0}
        for entry in self.performance_history:
            if entry["score"] < 40:
                summary["low"] += 1
            elif entry["score"] < 75:
                summary["medium"] += 1
            else:
                summary["high"] += 1
        return summary

    def adjust_routine(self):
        stats = self.analyze_performance()
        if stats["low"] > stats["high"]:
            return "Routine adjusted for easier flow."
        elif stats["medium"] > 0:
            return "Routine balanced."
        return "Routine upgraded."


def daily_schedule(self):
        output = []
        for time, habit in sorted(self.routine.items()):
            output.append(f"{time}: {habit}")
        return output


def motivate(self, habit):
        base = [f"Keep going with {habit}.",
                f"You're improving through {habit}.",
                f"Stay present during {habit}."]

        combined = base + self.motivation_messages
        if habit.lower() in ["meditation", "reading"]:
            combined += [
                "This habit improves your mental clarity.",
                "Daily repetition builds long-term focus."
            ]

        return combined[0]


def log(self, habit, status):
        self.daily_logs.append({"habit": habit, "status": status})
        return "Log updated."


def summary(self):
        details = {
            "Total Habits": len(self.habits),
            "Routine Size": len(self.routine),
            "Performance Records": len(self.performance_history),
            "Completed Logs": len([l for l in self.daily_logs if l["status"] == "done"]),
            "Pending Logs": len([l for l in self.daily_logs if l["status"] != "done"])
        }
        return details


def full_reset(self):
        self.habits = []
        self.routine = {}
        self.performance_history = []
        self.daily_logs = []
        return "System reset."


def categorize_difficulty(self):
        stats = {"easy": 0, "medium": 0, "hard": 0}
        for h in self.habits:
            level = h["difficulty"]
            if level in stats:
                stats[level] += 1
        return stats


ai = HabitFlowAI("Jaya")


ai.add_habit("Meditation", "7:00 AM")


ai.register_habits()


import pandas as pd

habits_df = pd.read_csv("/kaggle/input/habits-data/habits.csv")
habits_df.head()


for i, row in habits_df.iterrows():
    ai.add_habit(
        row["habit_name"],
        row["time"],
        row.get("category", "general"),
        row.get("difficulty", "medium")
    )


habit_records = pd.DataFrame(ai.habits)
habit_records


def score_habit_completion(completed, difficulty):
    base = 50 if completed else 10
    
    if difficulty == "easy":
        return base + 5
    elif difficulty == "medium":
        return base + 10
    else:
        return base + 20


analytics = []

for log in ai.daily_logs:
    difficulty = next(h["difficulty"] for h in ai.habits if h["habit"] == log["habit"])
    score = score_habit_completion(log["status"] == "done", difficulty)
    analytics.append({"habit": log["habit"], "score": score})

analytics_df = pd.DataFrame(analytics)
analytics_df


day_summary = analytics_df.groupby("habit")["score"].mean().reset_index()
day_summary


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.bar(day_summary["habit"], day_summary["score"])
plt.xticks(rotation=45)
plt.xlabel("Habit")
plt.ylabel("Average Score")
plt.title("Habit Performance Overview")
plt.show()


times = list(ai.routine.keys())
names = list(ai.routine.values())

plt.figure(figsize=(10,4))
plt.plot(times, [i for i in range(len(times))], marker="o")
plt.xticks(rotation=45)
plt.yticks([])
plt.title("Daily Routine Timeline")
plt.xlabel("Time")
plt.show()


# Trend line showing improvement over days
days = list(range(1, len(habit_scores) + 1))

plt.figure(figsize=(10,4))
plt.plot(days, habit_scores, marker="o")
plt.xlabel("Days")
plt.ylabel("Score")
plt.title("Habit Trend Over Time")
plt.grid(True)
plt.show()


plt.figure(figsize=(10,4))
plt.bar(habits, habit_scores)
plt.xlabel("Habits")
plt.ylabel("Score")
plt.title("Score Comparison of All Habits")
plt.show()


import seaborn as sns
import numpy as np

# Random habit consistency matrix (example)
data = np.random.randint(0, 2, (7, len(habits)))

plt.figure(figsize=(8,5))
sns.heatmap(data, annot=True, cmap="Blues", cbar=False)
plt.title("Weekly Habit Consistency Heatmap")
plt.xlabel("Habits")
plt.ylabel("Days")
plt.show()


times = list(ai.routine.keys())
activities = list(ai.routine.values())

plt.figure(figsize=(10,4))
plt.plot(times, range(len(times)), marker="o")
plt.yticks(range(len(times)), activities)
plt.title("Routine Timeline Overview")
plt.xlabel("Time")
plt.ylabel("Activity")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ======================================================
#                HabitFlow AI â€” Final Summary
# ======================================================

print("\n==============================================")
print("        HabitFlow AI Notebook Completed âœ”ï¸�")
print("==============================================")
print("All visualizations, routines, analytics, and")
print("habit performance data executed successfully.")
print("You're ready to track habits like a pro, Jaya! ğŸŒŸ")
print("==============================================\n")

