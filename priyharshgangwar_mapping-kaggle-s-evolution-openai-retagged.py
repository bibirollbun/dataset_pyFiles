
import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")



import pandas as pd
import matplotlib.pyplot as plt

# # Load Meta Kaggle data again
mk_path = '/kaggle/input/meta-kaggle/'



users = pd.read_csv(f'{mk_path}Users.csv', parse_dates=['RegisterDate'])
datasets = pd.read_csv(f'{mk_path}Datasets.csv', parse_dates=['CreationDate'])

# Prepare user and dataset data
users['Year'] = users['RegisterDate'].dt.year
datasets['Year'] = datasets['CreationDate'].dt.year

user_growth = users.groupby('Year').size().cumsum().reset_index(name='CumulativeUsers')
dataset_counts = datasets.groupby('Year').size().reset_index(name='NewDatasets')

# Merge
timeline = pd.DataFrame({'Year': sorted(set(user_growth['Year']) | set(dataset_counts['Year']))})
timeline = timeline.merge(user_growth, on='Year', how='left').merge(dataset_counts, on='Year', how='left')
timeline.fillna(0, inplace=True)

# Define milestones
ml_milestones = {
    2012: 'XGBoost',
    2015: 'ResNet & U-Net',
    2017: 'Transformer',
    2018: 'BERT',
    2020: 'GPT-3',
    2023: 'GPT-4 & ChatGPT'
}

kaggle_milestones = {
    2010: 'Kaggle Founded',
    2012: 'Health Prize Ends',
    2016: 'Acquired by Google',
    2017: 'Kernels & Learn',
    2018: 'Dataset Platform Launch',
    2020: 'COVID Challenges',
    2023: 'LLM Era Begins'
}

# Plot
fig, ax1 = plt.subplots(figsize=(16, 8))

# Line plot for users
ax1.plot(timeline['Year'], timeline['CumulativeUsers'], color='teal', linewidth=3, marker='o', label='Cumulative Users')
ax1.set_ylabel('Cumulative Users', color='teal', fontsize=13)
ax1.tick_params(axis='y', labelcolor='teal')

# Bar plot for datasets
ax2 = ax1.twinx()
ax2.bar(timeline['Year'], timeline['NewDatasets'], color='grey', alpha=0.2, label='New Datasets')
ax2.set_ylabel('New Datasets (Per Year)', color='grey', fontsize=13)
ax2.tick_params(axis='y', labelcolor='grey')

# Annotate ML milestones
# Annotate ML milestones at the top (above plot)
ymax = timeline['CumulativeUsers'].max()
ml_label_height = ymax * 0.85  # Adjust this for label height above plot

for year, label in ml_milestones.items():
    ax1.plot([year, year], [0, ml_label_height], color='darkred', linestyle=':', alpha=0.6)
    ax1.text(year, ml_label_height, label, ha='center', va='bottom', fontsize=11, color='darkred', weight='bold')


# Annotate Kaggle events
for year, label in kaggle_milestones.items():
    if year in timeline['Year'].values:
        y = timeline.loc[timeline['Year'] == year, 'CumulativeUsers'].values[0]
        ax1.axvline(x=year, linestyle='--', color='navy', alpha=0.3)
        ax1.text(year, y * 1.5, label, ha='center', va='bottom', fontsize=11, color='navy')

# Labels and Title
ax1.set_title("Kaggle Evolution: Cumulative Users & Dataset Growth vs ML Breakthroughs", fontsize=17, weight='bold')
ax1.set_xlabel("Year", fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.4)
plt.xticks(timeline['Year'].unique(), rotation=45, fontsize=11)
plt.yticks(fontsize=11)

# âœ… Save to current working directory
plt.tight_layout()
plt.savefig("kaggle_evolution_milestones.png", dpi=300)
plt.show()





import seaborn as sns

competitions = pd.read_csv(f"{mk_path}Competitions.csv", parse_dates=["EnabledDate"])

# Extract year from dates
users["Year"] = users["RegisterDate"].dt.year
datasets["Year"] = datasets["CreationDate"].dt.year
competitions["Year"] = competitions["EnabledDate"].dt.year

# Aggregate yearly metrics
user_growth = users.groupby("Year").size().cumsum().reset_index(name="CumulativeUsers")
dataset_counts = datasets.groupby("Year").size().reset_index(name="NewDatasets")
competition_counts = competitions.groupby("Year").size().reset_index(name="NewCompetitions")

# Combine all into one timeline
timeline = pd.DataFrame({'Year': sorted(set(user_growth['Year']) | set(dataset_counts['Year']) | set(competition_counts['Year']))})
timeline = timeline.merge(user_growth, on='Year', how='left').merge(dataset_counts, on='Year', how='left').merge(competition_counts, on='Year', how='left')
timeline.fillna(0, inplace=True)


timeline


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # Define path to Meta Kaggle
# mk_path = "/kaggle/input/meta-kaggle/"

# âœ… Confirmed columns from your earlier schema
competitions = pd.read_csv(
    f"{mk_path}Competitions.csv",
    usecols=["Id", "Title", "EnabledDate", "TotalTeams"],
    parse_dates=["EnabledDate"]
)

teams = pd.read_csv(
    f"{mk_path}Teams.csv",
    usecols=["Id", "CompetitionId"]  # 'TeamLeaderId' removed as itâ€™s not needed now
)

submissions = pd.read_csv(
    f"{mk_path}Submissions.csv",
    usecols=["Id", "TeamId", "SubmissionDate"],
    parse_dates=["SubmissionDate"]
)

# -----------------------------------------
# ğŸ“Š Preprocess
# -----------------------------------------

# Extract year from EnabledDate in Competitions
# Rename columns for merging clarity
teams.rename(columns={"Id": "TeamId"}, inplace=True)
competitions.rename(columns={"Id": "CompetitionId"}, inplace=True)

# Merge and process data
submissions["SubmissionDate"] = pd.to_datetime(submissions["SubmissionDate"], errors='coerce')
submissions["Year"] = submissions["SubmissionDate"].dt.year
submission_merged = submissions.merge(teams, on="TeamId", how="left")
submission_merged = submission_merged.merge(competitions, on="CompetitionId", how="left")

# Filter for meaningful years only (excluding partial 2025)
submission_merged = submission_merged[submission_merged["Year"].between(2010, 2024)]

# Get top 5 competitions per year by number of submissions
top_comps_by_year = (
    submission_merged.groupby(["Year", "CompetitionId", "Title"])
    .size()
    .reset_index(name="SubmissionCount")
    .sort_values(["Year", "SubmissionCount"], ascending=[True, False])
)

top5_landmark_challenges = top_comps_by_year.groupby("Year").head(5)
# top5_landmark_challenges.rename(columns={"Year_x": "Year"}, inplace=True)


# Plot
plt.figure(figsize=(14, 9))
sns.barplot(
    data=top5_landmark_challenges,
    x="SubmissionCount",
    y="Title",
    hue="Year",
    dodge=False,
    palette="tab20"
)
plt.title("Top 5 Landmark Kaggle Competitions per Year (2010â€“2024)", fontsize=16)
plt.xlabel("Number of Submissions")
plt.ylabel("Competition Title")
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("competetions_top5_submissions.png", dpi=300, bbox_inches="tight")
plt.show()


import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'
fig = px.bar(
    top5_landmark_challenges,
    x="SubmissionCount",
    y="Title",
    color="SubmissionCount",
    orientation="h",
    animation_frame="Year",
    range_x=[0, top5_landmark_challenges["SubmissionCount"].max() + 5000],
    title="Animated View: Top 5 Kaggle Competitions by Submissions (2010â€“2024)",
    color_continuous_scale="Viridis"
)

fig.update_layout(
    template="plotly_white",
    height=600,
    font=dict(size=14),
    title_font=dict(size=20),
    margin=dict(l=30, r=30, t=60, b=30)
)

fig.show()



import pandas as pd

# Ensure all required datasets are already loaded
competition_tags = pd.read_csv(f"{mk_path}/CompetitionTags.csv", usecols=["CompetitionId", "TagId"])
tags = pd.read_csv(f"{mk_path}/Tags.csv", usecols=["Id", "Name", "Slug", "FullPath", "Description"])
competitions = pd.read_csv(f"{mk_path}/Competitions.csv", usecols=["Id", "Title", "Subtitle"])
top5_landmark_challenges = top5_landmark_challenges.copy()

# Get landmark competition IDs
landmark_ids = top5_landmark_challenges["CompetitionId"].unique()

# Filter competition tags for landmark competitions
landmark_tags = competition_tags[competition_tags["CompetitionId"].isin(landmark_ids)]

# Merge tag names
landmark_tags = landmark_tags.merge(tags, left_on="TagId", right_on="Id", how="left")

# Merge with top5 challenge info (year & title)
landmark_tags = landmark_tags.merge(
    top5_landmark_challenges[["CompetitionId", "Year"]],
    on="CompetitionId",
    how="left"
)

# Merge with subtitle
landmark_tags = landmark_tags.merge(
    competitions.rename(columns={"Id": "CompetitionId"}),
    on="CompetitionId",
    how="left"
)

# Combine all landmark competitions to ensure completeness
all_landmark_competitions = top5_landmark_challenges[["CompetitionId", "Year"]].merge(
    competitions.rename(columns={"Id": "CompetitionId"}),
    on="CompetitionId",
    how="left"
)

# Merge the tags back (left join to keep all 73 competitions)
landmark_tags_final = all_landmark_competitions.merge(
    landmark_tags[["CompetitionId", "Name", "Slug"]],
    on="CompetitionId",
    how="left"
)

# Final structuring
landmark_tags_final = landmark_tags_final[["CompetitionId", "Title", "Subtitle", "Year", "Name", "Slug"]]



landmark_tags_final


# First, group all tags per competition into a list
grouped_tags = landmark_tags_final.groupby(["CompetitionId", "Title", "Year"])["Name"].apply(list).reset_index()

# Define categories
problem_types = {
    'tabular', 'binary classification', 'regression', 'multiclass classification',
    'text', 'image', 'time series analysis', 'nlp', 'neural networks', 'signal processing'
}

domain_types = {
    'banking', 'housing', 'internet', 'marketing', 'finance', 'geology', 'plants',
    'medicine', 'healthcare', 'biology', 'genetics', 'drugs and medications', 'education',
    'retail and shopping', 'earth and nature', 'real estate', 'automobiles and vehicles',
    'earth science', 'physics', 'linguistics', 'primary and secondary schools',
    'health', 'research', 'biotechnology', 'cancer', 'global'
}

# Function to assign tags
def assign_tags(tag_list, category_set):
    matched = [tag for tag in tag_list if tag in category_set]
    return matched[0] if matched else "None"

# Apply to the grouped tags
grouped_tags["ProblemType"] = grouped_tags["Name"].apply(lambda tags: assign_tags(tags, problem_types))
grouped_tags["Domain"] = grouped_tags["Name"].apply(lambda tags: assign_tags(tags, domain_types))

grouped_tags.sort_values(by="Year")



grouped_tags = grouped_tags.merge(
    landmark_tags_final[["CompetitionId", "Title", "Subtitle"]].drop_duplicates("CompetitionId"),
    on="CompetitionId",
    how="inner"
)



# Clean up grouped_tags DataFrame

# Drop the redundant Title_x (assume Title_y from merge is preferred)
grouped_tags_cleaned = grouped_tags.drop(columns=["Title_x"]).rename(columns={"Title_y": "Title"})

# Show sample to confirm structure
grouped_tags_cleaned.head()



import matplotlib.pyplot as plt

# Get value counts
problem_type_counts = grouped_tags_cleaned["ProblemType"].value_counts()
domain_type_counts = grouped_tags_cleaned["Domain"].value_counts()

# Create subplots side by side
fig, axs = plt.subplots(ncols=2, figsize=(16, 6), sharey=False)

# Problem Type Distribution
bars1 = axs[0].barh(problem_type_counts.index, problem_type_counts.values, color="lightsteelblue")
axs[0].set_xlabel("Number of Competitions")
axs[0].set_title("Problem Type Distribution")

# Annotate values
for bar in bars1:
    width = bar.get_width()
    axs[0].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{width}', va='center', fontsize=10)

axs[0].invert_yaxis()  # Highest count on top

# Domain Type Distribution
bars2 = axs[1].barh(domain_type_counts.index, domain_type_counts.values, color="lightcoral")
axs[1].set_xlabel("Number of Competitions")
axs[1].set_title("Domain Distribution")

# Annotate values
for bar in bars2:
    width = bar.get_width()
    axs[1].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{width}', va='center', fontsize=10)

axs[1].invert_yaxis()

# Final layout
plt.suptitle("Distributions in Landmark Kaggle Competitions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("landmark_distributions_side_by_side.png", dpi=300, bbox_inches="tight")
plt.show()



# !pip install openai



import openai
import json
from kaggle_secrets import UserSecretsClient

# Load API key from Kaggle secrets
user_secrets = UserSecretsClient()
openai.api_key = user_secrets.get_secret("OpenAIKey")

# Define the prompt
def generate_prompt(title, subtitle):
    return (
        f"Given the following Kaggle competition details:\n\n"
        f"Title: {title}\n"
        f"Subtitle: {subtitle}\n\n"
        f"Please categorize the competition into one high level machine learning problem types and domains "
        f"from the following refined lists.\n\n"
        f"Problem types (examples):\n"
        f"- binary classification\n"
        f"- multiclass classification\n"
        f"- regression\n"
        f"- multi-label classification\n"
        f"- ordinal regression\n"
        f"- clustering\n"
        f"- anomaly detection\n"
        f"- time series forecasting\n"
        f"- sequence modeling\n"
        f"- image classification\n"
        f"- image segmentation\n"
        f"- object detection\n"
        f"- tabular modeling\n"
        f"- natural language processing (NLP)\n"
        f"- text classification\n"
        f"- text generation\n"
        f"- recommendation systems\n"
        f"- graph learning / GNN\n"
        f"- reinforcement learning\n"
        f"- semi-supervised learning\n"
        f"- self-supervised learning\n"
        f"- transfer learning\n"
        f"- multi-modal modeling\n"
        f"- metric learning\n"
        f"- ranking / learning to rank\n\n"
        f"Domains (examples):\n"
        f"- healthcare\n"
        f"- biology / life sciences\n"
        f"- genetics / genomics\n"
        f"- finance\n"
        f"- education\n"
        f"- marketing / advertising\n"
        f"- retail / e-commerce\n"
        f"- social media\n"
        f"- earth / climate science\n"
        f"- physics / chemistry\n"
        f"- automotive / mobility\n"
        f"- real estate / housing\n"
        f"- sports / gaming\n"
        f"- security / fraud detection\n"
        f"- news / publishing\n"
        f"- public policy / government\n"
        f"- telecommunications\n"
        f"- manufacturing / industry\n"
        f"- robotics / autonomous systems\n"
        f"- agriculture / food science\n\n"
        f"Respond strictly in JSON format as:\n"
        f'{{"problem_type": [...], "domain": [...]}}'
    )

# Example competition
title = grouped_tags_cleaned["Title"][0]
subtitle = grouped_tags_cleaned["Subtitle"][0]

# Call the new Chat API
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": generate_prompt(title, subtitle)}
    ],
    temperature=0
)

# Parse the output
content = response.choices[0].message.content
try:
    result = json.loads(content)
    print("âœ… Problem Type:", result["problem_type"])
    print("âœ… Domain:", result["domain"])
except json.JSONDecodeError:
    print("â�Œ Failed to parse response as JSON:")
    print(content)



results = []

for i, row in tqdm(grouped_tags_cleaned.iterrows(), total=len(grouped_tags_cleaned)):
    title = row["Title"]
    subtitle = row["Subtitle"]
    comp_id = row["CompetitionId"]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": generate_prompt(title, subtitle)}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        results.append({
            "CompetitionId": comp_id,
            "problem_type": parsed.get("problem_type", []),
            "domain": parsed.get("domain", []),
            "raw_response": content
        })

    except Exception as e:
        print(f"âš ï¸� Error for Competition ID {comp_id}: {e}")
        results.append({
            "CompetitionId": comp_id,
            "problem_type": [],
            "domain": [],
            "raw_response": str(e)
        })

    time.sleep(1.5)  # To avoid hitting rate limits

# Save or merge results
tags_df = pd.DataFrame(results)
tags_df.head()



tags_df.to_csv("Competetions_retagged_openai.csv") # can be used to access 75 competetions


import ast

def extract_first_item(val):
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
            else:
                return None
        except (ValueError, SyntaxError):
            return val
    return val

tags_df["problem_type"] = tags_df["problem_type"].apply(extract_first_item)
tags_df["domain"] = tags_df["domain"].apply(extract_first_item)


# Merge original model/domain tags with new OpenAI tags
merged_df = grouped_tags_cleaned.merge(tags_df, on="CompetitionId", how="left")

# Rename columns for clarity
merged_df = merged_df.rename(columns={
    "problem_type": "OpenAI_Problem_Type",
    "domain": "OpenAI_Domain",
    "ProblemType": "Original_Model_Tag",
    "Domain": "Original_Domain_Tag"
})

# Organize columns
final_columns = [
    "Year", "CompetitionId", "Title", "Subtitle",
    "Original_Model_Tag", "Original_Domain_Tag",
    "OpenAI_Problem_Type", "OpenAI_Domain"
]

final_tag_comparison = merged_df[final_columns].sort_values(by="Year")

# tools.display_dataframe_to_user(name="Competition Tag Comparison", dataframe=final_tag_comparison)



final_tag_comparison[final_tag_comparison.Year==2010][['Title', 'Subtitle', 'Original_Model_Tag','Original_Domain_Tag', 'OpenAI_Problem_Type', 'OpenAI_Domain']]


import matplotlib.pyplot as plt

# Get value counts
problem_type_counts = final_tag_comparison["OpenAI_Problem_Type"].value_counts()
domain_type_counts = final_tag_comparison["OpenAI_Domain"].value_counts()

# Create subplots side by side
fig, axs = plt.subplots(ncols=2, figsize=(16, 6), sharey=False)

# Problem Type Distribution
bars1 = axs[0].barh(problem_type_counts.index, problem_type_counts.values, color="lightsteelblue")
axs[0].set_xlabel("Number of Competitions")
axs[0].set_title("Problem Type Distribution")

# Annotate values
for bar in bars1:
    width = bar.get_width()
    axs[0].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{width}', va='center', fontsize=10)

axs[0].invert_yaxis()  # Highest count on top

# Domain Type Distribution
bars2 = axs[1].barh(domain_type_counts.index, domain_type_counts.values, color="lightcoral")
axs[1].set_xlabel("Number of Competitions")
axs[1].set_title("Domain Distribution")

# Annotate values
for bar in bars2:
    width = bar.get_width()
    axs[1].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{width}', va='center', fontsize=10)

axs[1].invert_yaxis()

# Final layout
plt.suptitle("Distributions in Landmark Kaggle Competitions as mapped by OpenAI", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("landmark_distributions_side_by_side_open_ai.png", dpi=300, bbox_inches="tight")
plt.show()



import plotly.graph_objects as go

# Group and count transitions
sankey_df = final_tag_comparison.groupby(['Original_Model_Tag', 'OpenAI_Problem_Type']).size().reset_index(name='count')

# Encode strings to indices
label_list = list(set(sankey_df['Original_Model_Tag']).union(set(sankey_df['OpenAI_Problem_Type'])))
label_map = {label: i for i, label in enumerate(label_list)}

# Sankey inputs
source = sankey_df['Original_Model_Tag'].map(label_map)
target = sankey_df['OpenAI_Problem_Type'].map(label_map)
value = sankey_df['count']

# Plot Sankey
fig = go.Figure(data=[go.Sankey(
    node=dict(label=label_list, pad=15, thickness=20),
    link=dict(source=source, target=target, value=value)
)])
fig.update_layout(title="Tag Transition: Original Tags to OpenAI Problem Types", font_size=12)
fig.show()



import seaborn as sns
import matplotlib.pyplot as plt

heat_df = final_tag_comparison.groupby(['Year', 'OpenAI_Problem_Type']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(heat_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Year-wise Distribution of OpenAI Problem Types")
plt.ylabel("Year")
plt.xlabel("Problem Type")
plt.tight_layout()
plt.show()



heat_df = final_tag_comparison.groupby(['Year', 'OpenAI_Domain']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(heat_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Year-wise Distribution of OpenAI Domain")
plt.ylabel("Year")
plt.xlabel("Domain Type")
plt.tight_layout()
plt.show()



import pandas as pd

# Assuming full competitions data is loaded
# competitions.csv contains: Id, Title, Subtitle, EnabledDate (parsed as datetime)
all_comps = pd.read_csv(
    f"{mk_path}Competitions.csv",
    usecols=["Id", "Title", "Subtitle", "EnabledDate", ],
    parse_dates=["EnabledDate"]
)

# Drop entries without a Title or EnabledDate
all_comps.dropna(subset=["Title", "EnabledDate"], inplace=True)

# Assign year
all_comps["Year"] = all_comps["EnabledDate"].dt.year

# Drop duplicates
all_comps = all_comps.drop_duplicates(subset=["Id"])

# Filter out competitions from the current year (e.g., 2025) as they may be incomplete
landmark_all_years = all_comps[all_comps["Year"] < 2025].copy()

# Keep only essential columns
landmark_all_years = landmark_all_years[["Id", "Title", "Subtitle", "Year"]].rename(columns={"Id": "CompetitionId"})


import pandas as pd

# Ensure all required datasets are already loaded
competition_tags = pd.read_csv(f"{mk_path}/CompetitionTags.csv", usecols=["CompetitionId", "TagId"])
tags = pd.read_csv(f"{mk_path}/Tags.csv", usecols=["Id", "Name", "Slug", "FullPath", "Description"])
competitions = pd.read_csv(f"{mk_path}/Competitions.csv", usecols=["Id", "Title", "Subtitle"])
landmark_all_years = landmark_all_years.copy()

# Get landmark competition IDs
landmark_ids = landmark_all_years["CompetitionId"].unique()

# Filter competition tags for landmark competitions
landmark_tags = competition_tags[competition_tags["CompetitionId"].isin(landmark_ids)]

# Merge tag names
landmark_tags = landmark_tags.merge(tags, left_on="TagId", right_on="Id", how="left")

# Merge with top5 challenge info (year & title)
landmark_tags = landmark_tags.merge(
    landmark_all_years[["CompetitionId", "Year"]],
    on="CompetitionId",
    how="left"
)

# Merge with subtitle
landmark_tags = landmark_tags.merge(
    competitions.rename(columns={"Id": "CompetitionId"}),
    on="CompetitionId",
    how="left"
)

# Combine all landmark competitions to ensure completeness
all_landmark_competitions = landmark_all_years[["CompetitionId", "Year"]].merge(
    competitions.rename(columns={"Id": "CompetitionId"}),
    on="CompetitionId",
    how="left"
)

# Merge the tags back (left join to keep all 73 competitions)
landmark_tags_final = all_landmark_competitions.merge(
    landmark_tags[["CompetitionId", "Name", "Slug"]],
    on="CompetitionId",
    how="left"
)

# Final structuring
landmark_tags_final = landmark_tags_final[["CompetitionId", "Title", "Subtitle", "Year", "Name", "Slug"]]



import pandas as pd

# Provided category sets
problem_types = {
    'tabular', 'binary classification', 'regression', 'multiclass classification',
    'text', 'image', 'time series analysis', 'nlp', 'neural networks', 'signal processing'
}

domain_types = {
    'banking', 'housing', 'internet', 'marketing', 'finance', 'geology', 'plants',
    'medicine', 'healthcare', 'biology', 'genetics', 'drugs and medications', 'education',
    'retail and shopping', 'earth and nature', 'real estate', 'automobiles and vehicles',
    'earth science', 'physics', 'linguistics', 'primary and secondary schools',
    'health', 'research', 'biotechnology', 'cancer', 'global'
}

# Ensure 'Name' column is list-type before processing
if isinstance(landmark_tags_final["Name"].iloc[0], str):
    landmark_tags_final["Name"] = landmark_tags_final["Name"].apply(eval)

# Group all tags per competition into a list
grouped_tags = landmark_tags_final.groupby(["CompetitionId", "Title", "Subtitle", "Year"])["Name"].apply(list).reset_index()

# Function to assign first matching tag
def assign_tags(tag_list, category_set):
    flat_tags = [tag for sublist in tag_list for tag in (sublist if isinstance(sublist, list) else [sublist])]
    matched = [tag for tag in flat_tags if tag in category_set]
    return matched[0] if matched else "None"

# Apply mapping
grouped_tags["ProblemType"] = grouped_tags["Name"].apply(lambda tags: assign_tags(tags, problem_types))
grouped_tags["Domain"] = grouped_tags["Name"].apply(lambda tags: assign_tags(tags, domain_types))


grouped_tags


results = []

for i, row in tqdm(grouped_tags.iterrows(), total=len(grouped_tags)):
    title = row["Title"]
    subtitle = row["Subtitle"]
    comp_id = row["CompetitionId"]
    
    # try:
    #     response = openai.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "user", "content": generate_prompt(title, subtitle)}],
    #         temperature=0
    #     )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        results.append({
            "CompetitionId": comp_id,
            "problem_type": parsed.get("problem_type", []),
            "domain": parsed.get("domain", []),
            "raw_response": content
        })

    except Exception as e:
        print(f"âš ï¸� Error for Competition ID {comp_id}: {e}")
        results.append({
            "CompetitionId": comp_id,
            "problem_type": [],
            "domain": [],
            "raw_response": str(e)
        })

    time.sleep(1)  # To avoid hitting rate limits

# Save or merge results
tags_df = pd.DataFrame(results)
tags_df.head()



tags_df.to_csv("Full_competition_tagged.csv")  # can be used to access the new tags assigned


import ast

def extract_first_item(val):
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
            else:
                return None
        except (ValueError, SyntaxError):
            return val
    return val

tags_df["problem_type"] = tags_df["problem_type"].apply(extract_first_item)
tags_df["domain"] = tags_df["domain"].apply(extract_first_item)


# Merge original model/domain tags with new OpenAI tags
merged_df = grouped_tags.merge(tags_df, on="CompetitionId", how="left")

# Rename columns for clarity
merged_df = merged_df.rename(columns={
    "problem_type": "OpenAI_Problem_Type",
    "domain": "OpenAI_Domain",
    "ProblemType": "Original_Model_Tag",
    "Domain": "Original_Domain_Tag"
})

# Organize columns
final_columns = [
    "Year", "CompetitionId", "Title", "Subtitle",
    "Original_Model_Tag", "Original_Domain_Tag",
    "OpenAI_Problem_Type", "OpenAI_Domain"
]

final_tag_comparison = merged_df[final_columns].sort_values(by="Year")

# tools.display_dataframe_to_user(name="Competition Tag Comparison", dataframe=final_tag_comparison)



final_tag_comparison


import matplotlib.pyplot as plt

# Get value counts
problem_type_counts = final_tag_comparison["OpenAI_Problem_Type"].value_counts()
problem_type_counts = problem_type_counts[problem_type_counts>30]
domain_type_counts = final_tag_comparison["OpenAI_Domain"].value_counts()
domain_type_counts = domain_type_counts[domain_type_counts>100]

# Create subplots side by side
fig, axs = plt.subplots(ncols=2, figsize=(16, 6), sharey=False)

# Problem Type Distribution
bars1 = axs[0].barh(problem_type_counts.index, problem_type_counts.values, color="lightsteelblue")
axs[0].set_xlabel("Number of Competitions")
axs[0].set_title("Problem Type Distribution")

# Annotate values
for bar in bars1:
    width = bar.get_width()
    axs[0].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{width}', va='center', fontsize=10)

axs[0].invert_yaxis()  # Highest count on top

# Domain Type Distribution
bars2 = axs[1].barh(domain_type_counts.index, domain_type_counts.values, color="lightcoral")
axs[1].set_xlabel("Number of Competitions")
axs[1].set_title("Domain Distribution")

# Annotate values
for bar in bars2:
    width = bar.get_width()
    axs[1].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{width}', va='center', fontsize=10)

axs[1].invert_yaxis()

# Final layout
plt.suptitle("Distributions in Landmark Kaggle Competitions as mapped by OpenAI", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("landmark_distributions_side_by_side_open_ai.png", dpi=300, bbox_inches="tight")
plt.show()



import plotly.graph_objects as go

# Group and count transitions
sankey_df = final_tag_comparison.groupby(['Original_Model_Tag', 'OpenAI_Problem_Type']).size().reset_index(name='count')
sankey_df = sankey_df[sankey_df['count']>20]  # filter out for visualization
# Encode strings to indices
label_list = list(set(sankey_df['Original_Model_Tag']).union(set(sankey_df['OpenAI_Problem_Type'])))
label_map = {label: i for i, label in enumerate(label_list)}

# Sankey inputs
source = sankey_df['Original_Model_Tag'].map(label_map)
target = sankey_df['OpenAI_Problem_Type'].map(label_map)
value = sankey_df['count']

# Plot Sankey
fig = go.Figure(data=[go.Sankey(
    node=dict(label=label_list, pad=15, thickness=20),
    link=dict(source=source, target=target, value=value)
)])
fig.update_layout(title="Tag Transition: Original Tags to OpenAI Problem Types", font_size=12)
fig.show()



import seaborn as sns
import matplotlib.pyplot as plt

heat_df = final_tag_comparison.groupby(['Year', 'OpenAI_Problem_Type']).size().unstack(fill_value=0)
heat_df = heat_df.loc[:, heat_df.sum(axis=0) >= 30]  #Filter-out the less prominent ones to see if there is any industry trend
plt.figure(figsize=(12, 6))
sns.heatmap(heat_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Year-wise Distribution of OpenAI Problem Types")
plt.ylabel("Year")
plt.xlabel("Problem Type")
plt.tight_layout()
plt.show()



heat_df = final_tag_comparison.groupby(['Year', 'OpenAI_Domain']).size().unstack(fill_value=0)
heat_df = heat_df.loc[:, heat_df.sum(axis=0) >= 100]  #Filter-out the less prominent ones to see if there is any industry trend
plt.figure(figsize=(12, 6))
sns.heatmap(heat_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Year-wise Distribution of OpenAI Domain")
plt.ylabel("Year")
plt.xlabel("Domain Type")
plt.tight_layout()
plt.show()





