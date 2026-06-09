import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

class Tool:
    """Base class for agent tools - functions the AI can call."""
    
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
    
    def to_dict(self) -> Dict[str, Any]:
        """Format tool info for prompts."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.func.__code__.co_varnames[1:]
        }

class SmartAssistAgent:
    """Main AI agent that handles micro-tasks using simple rule-based logic."""
    
    def __init__(self):
        self.tools = self._setup_tools()
        self.memory = []   # Minimal conversation memory
        
    def _setup_tools(self) -> List[Tool]:
        """Initialize tools available to the agent."""
        
        # Tool 1: Email
        def send_email(recipient: str, subject: str, body: str) -> str:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"ğŸ“§ Email sent to {recipient}\nSubject: {subject}\nSent at: {timestamp}"
        
        # Tool 2: Meeting scheduler
        def schedule_meeting(date: str, time: str, attendees: str) -> str:
            attendees_list = [a.strip() for a in attendees.split(',')]
            return f"ğŸ“… Meeting scheduled:\nDate: {date}\nTime: {time}\nAttendees: {', '.join(attendees_list)}"
        
        # Tool 3: Reminder
        def set_reminder(task: str, minutes: int) -> str:
            eta = datetime.now().strftime("%H:%M") + f" (+{minutes}min)"
            return f"â�° Reminder set: '{task}' at {eta}"
        
        # Tool 4: Weather
        def weather_check(city: str) -> str:
            weather_data = {
                "Mumbai": "Sunny, 28Â°C",
                "Delhi": "Cloudy, 22Â°C",
                "Bangalore": "Rainy, 24Â°C"
            }
            return f"ğŸŒ¤ï¸� Weather in {city}: {weather_data.get(city, 'Unavailable')}"
        
        # â­� Tool 5: Alarm (NEW)
        def set_alarm(time: str) -> str:
            return f"â�° Alarm set for {time}. I'll notify you!"

        return [
            Tool("send_email", "Send an email", send_email),
            Tool("schedule_meeting", "Schedule a meeting", schedule_meeting),
            Tool("set_reminder", "Set a reminder", set_reminder),
            Tool("weather_check", "Check weather", weather_check),
            Tool("set_alarm", "Set an alarm for a specific time", set_alarm)  # NEW
        ]
    
    def parse_user_request(self, message: str) -> Dict[str, Any]:
        """Extract the userâ€™s intent and parameters."""
        msg = message.lower()
        
        # Email intent
        if any(w in msg for w in ["email", "send mail"]):
            person = re.search(r"to (\w+)", msg)
            recipient = person.group(1) if person else "unknown"
            return {"tool": "send_email", "params": {
                "recipient": recipient,
                "subject": "Quick update",
                "body": message
            }}
        
        # Meeting intent
        if any(w in msg for w in ["meeting", "schedule", "call"]):
            date = "tomorrow" if "tomorrow" in msg else "today"
            return {"tool": "schedule_meeting", "params": {
                "date": date,
                "time": "10 AM",
                "attendees": "team"
            }}
        
        # Reminder intent
        if "remind" in msg:
            task = re.search(r"remind me to (.*)", msg)
            return {"tool": "set_reminder", "params": {
                "task": task.group(1) if task else "task",
                "minutes": 30
            }}
        
        # Weather intent
        if "weather" in msg:
            city = re.search(r"in (\w+)", msg)
            return {"tool": "weather_check", "params": {
                "city": city.group(1) if city else "Mumbai"
            }}

        # â­� Alarm intent (NEW)
        if any(w in msg for w in ["alarm", "wake me", "wake up"]):
            time = re.search(r"(\d{1,2}\s*[:.]?\s*\d{0,2}\s*(am|pm)?)", msg)
            alarm_time = time.group(1).replace(".", ":") if time else "6 AM"
            return {"tool": "set_alarm", "params": {
                "time": alarm_time
            }}

        return {"tool": None, "params": {}}
    
    def execute_task(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Run the tool that matches the user intent."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.func(**params)
        return "â�Œ Unknown tool."
    
    def process(self, user_message: str) -> str:
        """Main agent loop."""
        self.memory.append({"user": user_message})
        
        intent = self.parse_user_request(user_message)
        
        if not intent["tool"]:
            return "ğŸ¤” Iâ€™m not sure what you want. Try:\n- send email to John\n- set alarm for 6 am\n- remind me to drink water"
        
        result = self.execute_task(intent["tool"], intent["params"])
        self.memory.append({"agent": result})
        
        return f"âœ… Task completed!\n\n{result}\n\nAnything else?"
        

# Demo program
def main():
    agent = SmartAssistAgent()
    
    print("ğŸš€ SmartAssist AI Agent is ready!")
    print("Try:")
    print("- send email to John")
    print("- schedule meeting tomorrow")
    print("- set alarm for 6 am")
    print("- remind me to call mom")
    print("- weather in Bangalore")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("ğŸ‘‹ Thanks for using SmartAssist!")
            break
        
        print("\nSmartAssist:", agent.process(user_input))

if __name__ == "__main__":
    main()


