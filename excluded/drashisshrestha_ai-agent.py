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


!pip install gradio --upgrade




from typing import List, Dict
import gradio as gr

# Sample leave requests
leave_requests = [
    {
        "token": "12345",
        "email": "alice@example.com",
        "department": "Physics",
        "type": "Annual Leave",
        "chairApproval": "Approved",
        "deanApproval": "Pending",
        "totalHours": 40,
        "fromDate": "2025-12-10",
        "toDate": "2025-12-15"
    },
    {
        "token": "67890",
        "email": "bob@example.com",
        "department": "Math",
        "type": "Sick Leave",
        "chairApproval": "Approved",
        "deanApproval": "Approved",
        "totalHours": 24,
        "fromDate": "2025-11-20",
        "toDate": "2025-11-22"
    }
]

# Simulated HR Policy Expert (multi-agent)
def consult_policy_expert(query: str) -> str:
    query = query.lower()
    if "annual leave" in query:
        return "Faculty get 25 days, Staff get 20 days per year."
    elif "sick leave" in query:
        return "Medical certificate required if sick leave exceeds 3 days."
    elif "conference leave" in query:
        return "Maximum 7 days per year. Paper acceptance proof required."
    elif "workflow" in query:
        return "All requests go to Dept Chair first, then Dean."
    elif "weekends" in query:
        return "Saturdays and Sundays are not counted unless it is for Training."
    else:
        return "Sorry, I can only answer HR leave-related policy questions."



# Check leave status by token
def check_leave_status(token: str) -> str:
    for req in leave_requests:
        if req["token"] == token:
            return (
                f"Token: {req['token']}\n"
                f"Email: {req['email']}\n"
                f"Department: {req['department']}\n"
                f"Type: {req['type']}\n"
                f"Chair Approval: {req['chairApproval']}\n"
                f"Dean Approval: {req['deanApproval']}\n"
                f"Hours: {req['totalHours']}\n"
                f"Dates: {req['fromDate']} to {req['toDate']}"
            )
    return f"No leave request found with token {token}."

# List leave requests by email
def list_requests_by_email(email: str) -> str:
    results = [r for r in leave_requests if email.lower() in r["email"].lower()]
    if not results:
        return f"No leave requests found for {email}."
    return "\n\n".join([f"Token: {r['token']}, Type: {r['type']}, Dean Approval: {r['deanApproval']}" for r in results])

# Main Concierge agent
def leave_concierge(user_input: str, chat_history: List[Dict] = []) -> (str, List[Dict]):
    response = ""
    # Decide if input is a leave request query or policy question
    if "token" in user_input.lower():
        token = ''.join(filter(str.isdigit, user_input))
        response = check_leave_status(token)
    elif "email" in user_input.lower():
        email = user_input.split()[-1]
        response = list_requests_by_email(email)
    else:
        # Assume policy question, call Policy Expert
        response = consult_policy_expert(user_input)
    
    # Update chat history
    chat_history.append({"user": user_input, "agent": response})
    return response, chat_history



with gr.Blocks() as demo:
    gr.Markdown("## ğŸ�¢ Leave Concierge Chat")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask about status, policies, or holidays...")
    
    def respond(user_message, chat_history):
        reply, history = leave_concierge(user_message, chat_history or [])
        return history, history
    
    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    
demo.launch()





