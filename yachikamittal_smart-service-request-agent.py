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


# This cell imports any needed libraries. We're using only standard Python, so no extras.
# If you need more (e.g., for data visualization), you can add them here.
import random  # For simulating random staff assignment


# This class represents a staff member.
class Staff:
    def __init__(self, name, department):
        self.name = name
        self.department = department
        self.available = True  # Staff starts as available
    
    def assign_task(self):
        if self.available:
            self.available = False
            return True
        return False
    
    def complete_task(self):
        self.available = True


# This class represents a department with a list of staff.
class Department:
    def __init__(self, name, keywords):
        self.name = name
        self.keywords = keywords  # List of keywords to match requests (e.g., ["IT", "computer"])
        self.staff = []  # List of Staff objects
    
    def add_staff(self, staff):
        self.staff.append(staff)
    
    def get_available_staff(self):
        available = [s for s in self.staff if s.available]
        return random.choice(available) if available else None





# This class represents a service request with status tracking.
class ServiceRequest:
    def __init__(self, request_id, description):
        self.request_id = request_id
        self.description = description
        self.status = "Submitted"  # Initial status
        self.assigned_department = None
        self.assigned_staff = None
    
    def update_status(self, new_status):
        valid_statuses = ["Submitted", "Assigned", "In Progress", "Completed"]
        if new_status in valid_statuses:
            self.status = new_status
            print(f"Request {self.request_id} status updated to: {self.status}")
        else:
            print("Invalid status update.")
    
    def assign_department_and_staff(self, departments):
        # Simple keyword matching to identify department
        for dept in departments:
            if any(keyword.lower() in self.description.lower() for keyword in dept.keywords):
                self.assigned_department = dept
                staff = dept.get_available_staff()
                if staff:
                    staff.assign_task()
                    self.assigned_staff = staff
                    self.update_status("Assigned")
                    print(f"Assigned to department: {dept.name}, staff: {staff.name}")
                else:
                    print(f"No available staff in {dept.name}.")
                return
        print("No matching department found.")


# This is the main "agent" class that manages requests and departments.
class SmartServiceRequestAgent:
    def __init__(self):
        self.requests = []  # List of ServiceRequest objects
        self.departments = []  # List of Department objects
        self.next_request_id = 1
    
    def add_department(self, department):
        self.departments.append(department)
    
    def accept_service_request(self, description):
        request = ServiceRequest(self.next_request_id, description)
        self.requests.append(request)
        self.next_request_id += 1
        print(f"Service request accepted: ID {request.request_id} - {description}")
        # Automatically try to assign department and staff
        request.assign_department_and_staff(self.departments)
        return request
    
    def update_request_status(self, request_id, new_status):
        for req in self.requests:
            if req.request_id == request_id:
                req.update_status(new_status)
                if new_status == "Completed" and req.assigned_staff:
                    req.assigned_staff.complete_task()
                return
        print("Request not found.")
    
    def get_request_status(self, request_id):
        for req in self.requests:
            if req.request_id == request_id:
                return req.status
        return "Request not found."


# Create the agent
agent = SmartServiceRequestAgent()

# Create sample departments and staff
it_dept = Department("IT", ["IT", "computer", "software", "network"])
it_dept.add_staff(Staff("Alice", "IT"))
it_dept.add_staff(Staff("Bob", "IT"))

hr_dept = Department("HR", ["HR", "employee", "payroll", "recruitment"])
hr_dept.add_staff(Staff("Charlie", "HR"))
hr_dept.add_staff(Staff("Diana", "HR"))

# Add departments to the agent
agent.add_department(it_dept)
agent.add_department(hr_dept)

print("Agent initialized with departments: IT and HR.")


# Accept a new request (this will auto-assign based on keywords)
request1 = agent.accept_service_request("Fix my computer software issue.")

# Manually update status to In Progress
agent.update_request_status(1, "In Progress")

# Check status
print(f"Current status of request 1: {agent.get_request_status(1)}")

# Complete the request
agent.update_request_status(1, "Completed")


# Accept another request
request2 = agent.accept_service_request("Help with employee payroll.")

# Update to In Progress
agent.update_request_status(2, "In Progress")

# Check status
print(f"Current status of request 2: {agent.get_request_status(2)}")

# Complete it
agent.update_request_status(2, "Completed")


# Accept a request that doesn't match any department
request3 = agent.accept_service_request("Clean the office.")

# Check status (should remain Submitted)
print(f"Current status of request 3: {agent.get_request_status(3)}")


# Print a summary of all requests
for req in agent.requests:
    print(f"Request ID: {req.request_id}, Description: {req.description}, Status: {req.status}, Department: {req.assigned_department.name if req.assigned_department else 'None'}, Staff: {req.assigned_staff.name if req.assigned_staff else 'None'}")

