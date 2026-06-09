!pip install Gemini
!pip install adk


from adk import Agent, Memory
from Gemini import LLM 
import json

class AircraftMaintenanceAgent(Agent):
    def __init__(self, name, memory: Memory):
        super().__init__(name, memory)
        self.checklists = {
            "Boeing 737": ["Check engine oil levels", "Inspect landing gear", "Test flight control surfaces", "Verify fuel levels"],
            "Airbus A320": ["Inspect engine performance", "Test avionics systems", "Check tire pressure", "Verify hydraulic fluid levels"]
        }
        self.logs = []
        self.gemini = LLM(api_key="gemini_api_key")

    def generate_checklist(self, aircraft_type):
        """Generates a checklist for the selected aircraft type."""
        if aircraft_type in self.checklists:
            return self.checklists[aircraft_type]
        else:
            return "Aircraft type not found. Please enter a valid type."

    def get_task_explanation(self, task):
        """Fetches an explanation for a specific maintenance task using Gemini API."""
        prompt = f"Explain the importance of the following task in aircraft maintenance: {task}"
        explanation = self.gemini.query(prompt)
        return explanation

    def complete_checklist(self, aircraft_type):
        """Marks the checklist as complete and stores the log."""
        checklist = self.generate_checklist(aircraft_type)
        if isinstance(checklist, list):
            print(f"\nStarting checklist for {aircraft_type}:")
            for task in checklist:
                explanation = self.get_task_explanation(task)
                print(f"\n - {task}")
                print(f"   Explanation: {explanation}")
            
            print("\nMaintenance completed for:", aircraft_type)
            log_entry = {
                "aircraft_type": aircraft_type,
                "tasks_completed": checklist,
                "status": "Completed"
            }
            self.logs.append(log_entry)
        else:
            print(checklist)

    def view_logs(self):
        """Displays the maintenance logs."""
        if not self.logs:
            print("No logs found.")
        else:
            for log in self.logs:
                print(f"Aircraft: {log['aircraft_type']}, Tasks Completed: {', '.join(log['tasks_completed'])}, Status: {log['status']}")

memory = Memory()
agent = AircraftMaintenanceAgent(name="Aircraft Maintenance Agent", memory=memory)

aircraft_type = "Boeing 737"

agent.complete_checklist(aircraft_type)

print("\nMaintenance Logs:")
agent.view_logs()

