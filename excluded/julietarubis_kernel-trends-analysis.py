import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load Meta Kaggle datasets
base_path = "/kaggle/input/meta-kaggle"
kernel_versions = pd.read_csv(os.path.join(base_path, "KernelVersions.csv"), low_memory=False)
competition_sources = pd.read_csv(os.path.join(base_path, "KernelVersionCompetitionSources.csv"))
competitions = pd.read_csv(os.path.join(base_path, "Competitions.csv"))


# Merge kernel versions with competitions
merged = kernel_versions.merge(competition_sources, left_on="Id", right_on="KernelVersionId", how="inner")
merged = merged.merge(
    competitions[["Id", "Title", "DeadlineDate"]],
    left_on="SourceCompetitionId",
    right_on="Id",
    suffixes=("", "_comp")
)


# Extract and filter year
merged["Year"] = pd.to_datetime(merged["DeadlineDate"], errors="coerce").dt.year
merged = merged[(merged["Year"] >= 2010) & (merged["Year"] <= 2025)]


# Language mapping (extended)
lang_map = {
    1: "Python",
    2: "R",
    3: "Julia",
    4: "SQL",
    5: "Bash",
    6: "C++",
    7: "Java",
    8: "Other"
}
merged["Language"] = merged["ScriptLanguageId"].map(lang_map).fillna("Unknown")


# Kernel Submissions Per Year
kernels_per_year = merged.groupby("Year").size().reset_index(name="KernelCount")
plt.figure(figsize=(12, 5))
sns.lineplot(data=kernels_per_year, x="Year", y="KernelCount", marker="o")
plt.title("Kaggle Competition Kernel Submissions Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Kernel Versions")
plt.grid(True)
plt.tight_layout()
plt.show()


# Language Usage Trends Over Time
lang_trends = merged.groupby(["Year", "Language"]).size().reset_index(name="Count")
plt.figure(figsize=(12, 6))
sns.lineplot(data=lang_trends, x="Year", y="Count", hue="Language", marker="o")
plt.title("Kernel Language Usage in Competitions (Over Time)")
plt.xlabel("Year")
plt.ylabel("Kernel Count")
plt.grid(True)
plt.tight_layout()
plt.show()



with open("/kaggle/working/submission.txt", "w") as f:
    f.write("Meta Kaggle Hackathon submission placeholder.\n")

