# This R environment comes with many helpful analytics packages installed
# It is defined by the kaggle/rstats Docker image: https://github.com/kaggle/docker-rstats
# For example, here's a helpful package to load

library(tidyverse) # metapackage of all tidyverse packages

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

list.files(path = "../input")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


## import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict

print("Import successfull")


@dataclass
class Memory:
    messages:List[Dict] = field(default_factory=list)
    max_history : int = 20

def add(self, role, content):
    self.message.append({
        "role":role,
        "content":content,
        "time":datetime.now().isoformat()
    })
    if len(self.messages) > self.max_history:
        self.messages = self.message[-self.max_history:]
def get_context(self):
    out = ""
    for m in self.messages[-5]:
     out+=f"{m['role']}: {m['content']}\n"
    return out

#Intent Classification
class IntentAgent:
    def classify(self,message):
        text = message.lower()
        if "refund" in text:
            return"refund","high"
        if "cancle" in text :
            return "cancellation","high"
        if "invoice"in text or "bill"in text :
            return "billing", "medium"
        if "text" in text :
            return "general_help", "low"
            return "general", "low"

#Reply generation
class ReplyAgent:
    def create_reply(self,message,intent,urgency):
        if intent == "refund":
            return"We offer a clear,straightforward refund process because your satisfaction and confidence in product are our higest priority."
        if intent == "cancellation" :
            return"We are sorry to let you go!If you change your mind before [Date of next billing cycle end], you can easily reactive your subscription by visiting your setting > Billing page."
        if intent == "Billing":
            return"We understand billing can sometimes be confusing, and we are here to help you resolve your concern quickly.Please reply to this message or call us at [Phone number]."
        if intent == "general_help":
            return"We are standing by,ready to help with any question and we aim to solve your issue quickly."
            return"thank you for reaching out to us. How may i help you?"

# Coordinator ties with 3 features together
class Coordinator:
    def __init__(self, memory: Memory):
        self.intent_agent = IntentAgent()
        self.reply_agent = ReplyAgent()
        self.memory = Memory()
    def ask(self,message):
        self.memory.add("user", message)
        intent, urgency = self.intent_agent.classify(message)
        reply = self.reply_agent.create_reply(message, intent, urgency)

        final_output = {
            "intent": intent,
            "ungency": urgency,
            "reply": reply
        }
        self.memory.add("agent",reply)
        return final_output    
        
#Demo query
agent = Coordinator(memory=())
message = [ 
    "I am having trouble logging into my account."
    "Can you look up to my account balance?"
    "How do I reset my password"
    "Has my payment for the last invioce gone through?"
]
for msg in message :
    print("USER",msg)
    out = agent.ask(msg)
    print(json.dumps(out, indent=2))
    print("_" * 50)
    

