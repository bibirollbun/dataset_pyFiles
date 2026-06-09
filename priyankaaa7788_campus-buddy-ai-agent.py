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


# OPTIONAL: run this if Kaggle is missing packages (usually not required)
!pip install sqlmodel sqlalchemy tinydb pytest



%%writefile utils.py
import uuid
import time
from dataclasses import dataclass

@dataclass
class Message:
    id: str
    sender: str
    text: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

def make_msg(sender: str, text: str) -> Message:
    return Message(id=str(uuid.uuid4()), sender=sender, text=text)

def pretty_print_conversation(conv):
    for m in conv:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp))
        print(f"[{t}] {m.sender.upper():8}: {m.text}")



from utils import Message, make_msg, pretty_print_conversation
print("import Message succesfully")



m1 = make_msg("user", "Hello!")
m2 = make_msg("agent", "Hi! How can I help?")
pretty_print_conversation([m1, m2])



from dataclasses import dataclass
import time, uuid

# ------------------------------
# 1. UTILS
# ------------------------------
@dataclass
class Message:
    id: str
    sender: str
    text: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

def make_msg(sender, text):
    return Message(id=str(uuid.uuid4()), sender=sender, text=text)

def pretty_print_conversation(conv):
    for m in conv:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.timestamp))
        print(f"[{t}] {m.sender.upper():8}: {m.text}")

# ------------------------------
# 2. MOCK CAMPUS API
# ------------------------------
COURSES = {
    "CS101": {
        "title": "Intro to Computer Science",
        "instructor": "Dr. Patel",
        "slots": ["Mon 10-12", "Wed 10-12"]
    }
}

ROOMS = [
    {"room_id": "R100", "capacity": 6},
    {"room_id": "R101", "capacity": 4},
]

BOOKINGS = []

def list_courses():
    return COURSES

def get_course(course_code):
    return COURSES.get(course_code, None)

def list_rooms():
    return ROOMS

def create_booking(user, room_id, start, end):
    booking = {
        "booking_id": str(uuid.uuid4()),
        "user": user,
        "room_id": room_id,
        "start_time": start,
        "end_time": end
    }
    BOOKINGS.append(booking)
    return booking

# ------------------------------
# 3. MEMORY
# ------------------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id):
        self.sessions[session_id] = []

    def append_message(self, session_id, message):
        self.sessions[session_id].append(message)

    def get_session(self, session_id):
        return self.sessions.get(session_id, [])

class MemoryBank:
    def __init__(self):
        self.store = {}

    def get_memory(self, user):
        return self.store.get(user, {})

    def set_memory(self, user, data):
        self.store[user] = data

    def update_memory_field(self, user, key, value):
        if user not in self.store:
            self.store[user] = {}
        self.store[user][key] = value

# ------------------------------
# 4. TOOLS
# ------------------------------
class CalendarTool:
    def __init__(self):
        self.events = {}

    def add_event(self, user, title, start, end):
        event = {
            "id": str(uuid.uuid4()),
            "user": user,
            "title": title,
            "start_time": start,
            "end_time": end
        }
        self.events.setdefault(user, []).append(event)
        return event

    def list_events(self, user):
        return self.events.get(user, [])

class BookingTool:
    def find_available_rooms(self):
        return ROOMS

    def book_room(self, user, room_id, start, end):
        return create_booking(user, room_id, start, end)

# ------------------------------
# 5. AGENTS
# ------------------------------
class BaseAgent:
    def __init__(self, name, memory, sessions, tools):
        self.name = name
        self.memory = memory
        self.sessions = sessions
        self.tools = tools

    def record(self, session_id, text):
        msg = make_msg(self.name, text)
        self.sessions.append_message(session_id, msg)

class InfoAgent(BaseAgent):
    def handle(self, user, query, session_id):
        if "cs101" in query.lower():
            course = get_course("CS101")
            response = f"{course['title']} by {course['instructor']}, slots: {course['slots']}"
        else:
            response = "I can provide course info. Try asking about CS101."

        self.record(session_id, response)
        return response

class ScheduleAgent(BaseAgent):
    def handle(self, user, query, session_id):
        events = self.tools["calendar"].list_events(user)
        response = f"You have {len(events)} events."
        self.record(session_id, response)
        return response

class BookingAgent(BaseAgent):
    def handle(self, user, query, session_id):
        booking = self.tools["booking"].book_room(user, "R100", "15:00", "16:00")
        response = f"Booked room R100 (15:00–16:00). ID: {booking['booking_id']}"
        self.record(session_id, response)
        return response

class ReminderAgent(BaseAgent):
    def handle(self, user, query, session_id):
        response = "Reminder set 10 minutes before."
        self.record(session_id, response)
        return response

# ------------------------------
# 6. ROUTER
# ------------------------------
class IntentRouter:
    def __init__(self, agents):
        self.agents = agents

    def route(self, user, query, session_id):
        q = query.lower()
        if "cs101" in q:
            return self.agents["info"].handle(user, query, session_id)
        if "class" in q or "schedule" in q:
            return self.agents["schedule"].handle(user, query, session_id)
        if "book" in q or "room" in q:
            return self.agents["booking"].handle(user, query, session_id)
        if "reminder" in q:
            return self.agents["reminder"].handle(user, query, session_id)

        # fallback
        return self.agents["info"].handle(user, query, session_id)

# ------------------------------
# 7. BUILD SYSTEM
# ------------------------------
def build_system():
    memory = MemoryBank()
    sessions = InMemorySessionService()
    tools = {
        "calendar": CalendarTool(),
        "booking": BookingTool()
    }

    agents = {
        "info": InfoAgent("InfoAgent", memory, sessions, tools),
        "schedule": ScheduleAgent("ScheduleAgent", memory, sessions, tools),
        "booking": BookingAgent("BookingAgent", memory, sessions, tools),
        "reminder": ReminderAgent("ReminderAgent", memory, sessions, tools)
    }

    router = IntentRouter(agents)
    return router, memory, sessions



router, memory, sessions = build_system()

user = "priyanka"
session_id = "s1"
sessions.create_session(session_id)

queries = [
    "Who teaches CS101?",
    "What classes do I have tomorrow?",
    "Please book a room",
    "Set a reminder for my study"
]

for q in queries:
    res = router.route(user, q, session_id)
    print("USER:", q)
    print("BOT :", res)
    print()

print("\nFULL CONVERSATION:\n")
pretty_print_conversation(sessions.get_session(session_id))





