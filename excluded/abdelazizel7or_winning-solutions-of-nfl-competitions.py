import numpy as np
import pandas as pd
from IPython.core.display import HTML

import warnings
warnings.filterwarnings("ignore")

data_path = "/kaggle/input/meta-kaggle/"

competitions_df = pd.read_csv(data_path + "Competitions.csv")
competitions_df = competitions_df[competitions_df["Title"].str.contains("NFL", case=False, na=False)]
comps_to_use = ["Featured", "Research", "Recruitment"]
competitions_df = competitions_df[competitions_df["HostSegmentTitle"].isin(comps_to_use)]
competitions_df["EnabledDate"] = pd.to_datetime(competitions_df["EnabledDate"], format="%m/%d/%Y %H:%M:%S")
competitions_df = competitions_df.sort_values(by="EnabledDate", ascending=False).reset_index(drop=True)
competitions_df = competitions_df.iloc[::-1].reset_index(drop=True)
competitions_df.head()

forum_topics_df = pd.read_csv(data_path + "ForumTopics.csv")

comp_tags_df = pd.read_csv(data_path + "CompetitionTags.csv")
tags_df = pd.read_csv(data_path + "Tags.csv", usecols=["Id", "Name"])

def get_comp_tags(comp_id):
    temp_df = comp_tags_df[comp_tags_df["CompetitionId"]==comp_id]
    temp_df = pd.merge(temp_df, tags_df, left_on="TagId", right_on="Id")
    tags_str = "Tags : "
    for ind, row in temp_df.iterrows():
        tags_str += row["Name"] + ", "
    return tags_str.strip(", ")

def check_solution(topic):
    is_solution = False
    to_exclude = ["?", "submit", "why", "what", "resolution", "benchmark"]
    if "solution" in topic.lower():
        is_solution = True
        for exc in to_exclude:
            if exc in topic.lower():
                is_solution = False
    to_include = ["2nd place code", '"dance with ensemble" sharing']
    for inc in to_include:
        if inc in topic.lower():
            is_solution = True
    return is_solution

def get_discussion_results(forum_id, n):
    results_df = forum_topics_df[forum_topics_df["ForumId"]==forum_id]
    results_df["is_solution"] = results_df["Title"].apply(lambda x: check_solution(str(x)))
    results_df = results_df[results_df["is_solution"] == 1]
    results_df = results_df.sort_values(by=["Score","TotalMessages"], ascending=False).head(n).reset_index(drop=True)
    return results_df[["Title", "Id", "Score", "TotalMessages", "TotalReplies"]]

def render_html_for_comp(forum_id, comp_id, comp_name, comp_slug, comp_subtitle, n):
    results_df = get_discussion_results(forum_id, n)
    if len(results_df) < 1:
        return
    
    comp_tags = get_comp_tags(comp_id)
    
    # Clean subtitle (avoid showing 'nan')
    if pd.isna(comp_subtitle):
        subtitle = ""
    else:
        subtitle = str(comp_subtitle)
    
    # Parse tags into chips
    tag_html = ""
    if comp_tags != "Tags :":
        raw = comp_tags.replace("Tags :", "").strip()
        tag_list = [t.strip() for t in raw.split(",") if t.strip()]
        if tag_list:
            tag_html = '<div class="nfl-tags-row">ğŸ§© Tags:&nbsp;'
            for t in tag_list:
                tag_html += f'<span class="nfl-tag">#{t}</span>'
            tag_html += '</div>'
    
    comp_url = "https://www.kaggle.com/c/" + str(comp_slug)
    
    hs = f"""
    <style>
        .nfl-card {{
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: radial-gradient(circle at top left, #1e293b 0, #020617 40%, #020617 100%) !important;
    border-radius: 18px !important;
    padding: 18px 22px 20px !important;
    margin: 20px 0 !important;
    color: #e5e7eb !important;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
}}

.nfl-header {{
    display: flex !important;
    justify-content: space-between !important;
    align-items: flex-start !important;
    gap: 12px !important;
    margin-bottom: 12px !important;
}}

.nfl-title-block {{
    max-width: 70% !important;
}}

.nfl-label {{
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #a5b4fc !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: rgba(79, 70, 229, 0.16) !important;
    border-radius: 999px !important;
    padding: 3px 10px !important;
    margin-bottom: 6px !important;
}}

.nfl-label::before {{
    content: "ğŸ�ˆ" !important;
}}

.nfl-title {{
    font-size: 18px !important;
    margin: 0 0 4px 0 !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
}}

.nfl-title a {{
    color: white !important;
    text-decoration: none !important;
}}

.nfl-title a:hover {{
    color: #f97316 !important;
}}

.nfl-subtitle {{
    margin: 0 !important;
    font-size: 13px !important;
    color: #9ca3af !important;
}}

.nfl-meta {{
    text-align: right !important;
    font-size: 11px !important;
}}

.nfl-pill {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    border-radius: 999px !important;
    padding: 4px 10px !important;
    font-size: 11px !important;
    margin-left: 4px !important;
}}

.nfl-pill-primary {{
    background: rgba(56, 189, 248, 0.18) !important;
    color: #e0f2fe !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
}}

.nfl-pill-secondary {{
    background: rgba(34, 197, 94, 0.18) !important;
    color: #bbf7d0 !important;
    border: 1px solid rgba(34, 197, 94, 0.45) !important;
}}

.nfl-tags-row {{
    margin: 4px 0 10px 0 !important;
    font-size: 12px !important;
    color: #cbd5f5 !important;
}}

.nfl-tag {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 2px 8px !important;
    margin: 0 4px !important;
    border-radius: 999px !important;
    background: rgba(30, 64, 175, 0.6) !important;
    border: 1px solid rgba(129, 140, 248, 0.65) !important;
    font-size: 11px !important;
    color: #e0e7ff !important;
    white-space: nowrap !important;
}}

.nfl-table-wrapper {{
    margin-top: 10px !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(148, 163, 184, 0.55) !important;
    background: rgba(15, 23, 42, 0.75) !important;
}}

.nfl-table {{
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 12px !important;
    background: transparent !important;
}}

.nfl-table thead {{
    background: linear-gradient(90deg, rgba(30, 64, 175, 0.9), rgba(37, 99, 235, 0.9)) !important;
    color: #e5e7eb !important;
}}

.nfl-table thead th, .nfl-table thead tr {{
    background: transparent !important;
}}

.nfl-table th,
.nfl-table td {{
    padding: 8px 10px !important;
    text-align: left !important;
}}

.nfl-table th {{
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

.nfl-table tbody tr {{
    border-top: 1px solid rgba(31, 41, 55, 0.9) !important;
    transition: background 0.15s ease !important;
}}

.nfl-table tbody tr:nth-child(odd) {{
    background: rgba(15, 23, 42, 0.6) !important;
}}

.nfl-table tbody tr:nth-child(even) {{
    background: rgba(15, 23, 42, 0.3) !important;
}}

.nfl-table tbody tr:hover {{
    background: rgba(15, 118, 110, 0.55) !important;
    transform: translateY(-1px) !important;
}}

.nfl-rank {{
    font-weight: 600 !important;
    color: #fde68a !important;
    font-size: 12px !important;
}}

.nfl-discussion-link {{
    color: white !important;
    text-decoration: none !important;
}}

.nfl-discussion-link b {{
    font-weight: 600 !important;
}}

.nfl-discussion-link:hover {{
    color: #facc15 !important;
    text-decoration: underline !important;
}}

.nfl-upvotes {{
    color: #fbbf24 !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}}

.nfl-replies {{
    color: #a5b4fc !important;
    white-space: nowrap !important;
}}

.nfl-footer-note {{
    margin-top: 8px !important;
    font-size: 11px !important;
    color: #9ca3af !important;
    display: flex !important;
    justify-content: space-between !important;
    gap: 8px !important;
    align-items: center !important;
}}

.nfl-footer-note span {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}}
    </style>

    <div class="nfl-card">
        <div class="nfl-header">
            <div class="nfl-title-block">
                <div class="nfl-label">Big Data Bowl â€“ Solutions</div>
                <h2 style="color:white">  So, we donâ€™t start from scratch â€” we start from where the best people stopped, we donâ€™t reinvent the wheel â€” we learn from the winners and push it further.</h2>
                <h3 class="nfl-title">
                    <a href="{comp_url}" target="_blank">{comp_name}</a>
                </h3>
                <p class="nfl-subtitle">{subtitle}</p>
            </div>
            <div class="nfl-meta">
                <div class="nfl-pill nfl-pill-primary">â­� Top solution threads</div><br/>
                <div class="nfl-pill nfl-pill-secondary">ğŸ§µ Sorted by upvotes & replies</div>
            </div>
        </div>

        {tag_html}

        <div class="nfl-table-wrapper">
            <table class="nfl-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>ğŸ�† Title</th>
                        <th>ğŸ‘� Upvotes</th>
                        <th>ğŸ’¬ Replies</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Build rows
    for i, row in results_df.iterrows():
        url = f"https://www.kaggle.com/c/{comp_slug}/discussion/{row['Id']}"
        title = str(row["Title"])
        score = int(row["Score"])
        replies = int(row["TotalReplies"])
        
        hs += f"""
            <tr>
                <td class="nfl-rank">{i+1}</td>
                <td>
                    <a href="{url}" target="_blank" class="nfl-discussion-link">
                        <b>{title}</b>
                    </a>
                </td>
                <td class="nfl-upvotes">ğŸ‘� {score}</td>
                <td class="nfl-replies">ğŸ’¬ {replies}</td>
            </tr>
        """
    
    hs += """
                </tbody>
            </table>
        </div>
        <div class="nfl-footer-note">
            <span>ğŸ“Œ Tip: Sort by <b>Score</b> & <b>TotalReplies</b> to find gold-level solution shares.</span>
            <span>ğŸ§  Ninja note: Save your favorite threads for model ideas & feature tricks.</span>
        </div>
    </div>
    """
    
    display(HTML(hs))


for ind, comp_row in competitions_df.iterrows():
    render_html_for_comp(comp_row["ForumId"], comp_row["Id"], comp_row["Title"], comp_row["Slug"], comp_row["Subtitle"], 12)

