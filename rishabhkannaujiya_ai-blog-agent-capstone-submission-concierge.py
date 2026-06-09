!git clone https://github.com/irkky/AI-Blog-Agent.git

%cd AI-Blog-Agent

!pip install -r requirements.txt -q

print("âœ… Repository cloned and requirements installed.")


import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
try:
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    os.environ["GOOGLE_API_KEY"] = api_key

    print("âœ… API Key loaded successfully.")
except Exception as e:
    print("â�Œ Error: Could not find 'GOOGLE_API_KEY'. Check Add-ons -> Secrets.")


# We are using '!printf' to simulate a user typing inputs:
# Input 1: "Future of Multi-Agent AI" (Topic)
# Input 2: "Professional" (Tone)
# Input 3: "Developers" (Audience)
# Input 4: "1200" (Word Count)
# You can change the input as per your preferences

# Running the agent and saving the output to a file
# We are using '>' to redirect the output stream to 'agent_output.txt'
print("ğŸš€ Running AI Blog Agent... (This may take 1-4 minutes depending on word count)")
!printf "Future of Multi-Agent AI\nProfessional\nDevelopers\n1200\n" | python main.py > agent_output.txt
print("âœ… Run complete. Output saved to agent_output.txt")


from IPython.display import display, Markdown

with open("agent_output.txt", "r", encoding="utf-8") as f:
    full_content = f.read()

# Here main.py prints "ğŸ“� FINAL BLOG" before the content
delimiter = "ğŸ“� FINAL BLOG"

if delimiter in full_content:
    # It will Get everything AFTER the delimiter
    _, blog_content = full_content.split(delimiter, 1)
    
    display(Markdown(blog_content))
else:
    print("âš ï¸� Could not find the final blog in the output. Here is the raw log:")
    print(full_content)


import pandas as pd
import json
import os

log_file = "logs/events.jsonl"

if os.path.exists(log_file):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    df = pd.DataFrame(data)
    
    # Filtering for just the agent run events to see timing
    run_stats = df[df['event_type'] == 'agent_run'][['agent', 'duration_sec', 'message']]
    
    print("ğŸ“Š Agent Execution Timings:")
    display(run_stats)
    
    total_time = run_stats['duration_sec'].sum()
    print(f"\nâ�±ï¸� Total Pipeline Duration: {total_time:.2f} seconds")
else:
    print("âš ï¸� No log file found. Did the agent run?")


if os.path.exists(log_file):
    # Finding the evaluation event
    eval_events = [d for d in data if d.get('event_type') == 'evaluation']
    
    if eval_events:
        latest_eval = eval_events[-1]
        raw_json = latest_eval.get('extra', {}).get('raw_eval', '{}')
        
        print("ğŸ�† Evaluation Agent Report:\n")
        # It is stored as a stringified JSON in our logs, so we will parse it
        try:
            # Sometimes the LLM returns markdown json like ```json ... ``` 
            # We will clean it just in case
            clean_json = raw_json.replace("```json", "").replace("```", "").strip()
            scores = json.loads(clean_json)
            print(json.dumps(scores, indent=4))
        except:
            print(raw_json)
    else:
        print("âš ï¸� No evaluation event found yet.")


import difflib
from IPython.display import display, HTML, Markdown

# Read the file
with open("agent_output.txt", "r", encoding="utf-8") as f:
    raw_log = f.read()

# Helping function to extract text
def extract_section(content, start_marker, end_marker):
    try:
        start = content.index(start_marker) + len(start_marker)
        end = content.index(end_marker)
        return content[start:end].strip()
    except ValueError:
        return None

# Extracting Draft and Critic
draft_text = extract_section(raw_log, "ğŸ”¹ DRAFT_OUTPUT ğŸ”¹", "ğŸ”¹ END_DRAFT ğŸ”¹")
critic_text = extract_section(raw_log, "ğŸ”¹ CRITIC_OUTPUT ğŸ”¹", "ğŸ”¹ END_CRITIC ğŸ”¹")

# It will Show the Diff
if draft_text and critic_text:
    print("âœ… Markers found! Generating diff...")
    d = difflib.HtmlDiff()
    html = d.make_file(
        draft_text.splitlines()[:20], 
        critic_text.splitlines()[:20], 
        context=True, 
        numlines=5
    )
    display(HTML(html))
else:
    print("â�Œ Still missing markers.")
    print("Debug: Did main.py actually update? Check the file content:")
    # Print the first 50 lines of main.py to see if markers exist
    !grep -C 2 "DRAFT_OUTPUT" main.py


import os
from IPython.display import FileLink

# Resetting to the main Kaggle output directory just to ensure that the links work properly
os.chdir('/kaggle/working')

# filenames
md_filename = "ai_generated_blog.md"
txt_filename = "ai_generated_blog.txt"

# Writing the content to files
if 'blog_content' in locals() and blog_content:
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(blog_content)
    
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(blog_content)

    print(f"âœ… Files saved to {os.getcwd()}")
    print("ğŸ“¦ Click below to download:")
    
    display(FileLink(md_filename))
    display(FileLink(txt_filename))
else:
    print("âš ï¸� Error: 'blog_content' is missing or empty.")
    print("ğŸ‘‰ Please re-run the 'Extract and Render' cell (Cell 2) above.")

