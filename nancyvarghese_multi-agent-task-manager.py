#This Python 3 environment comes with many helpful analytics libraries installed
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


#Install Dependencies
!pip install -q ipywidgets
print("âœ… Dependencies installed!")




# Import Libraries
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Google Generative AI
from google import genai
from google.genai import types

# Kaggle Secrets
from kaggle_secrets import UserSecretsClient

# Widgets for interactive interface
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

print("âœ… Imports successful!")


#Configuration
AGENT_CONFIG = {
    "user_name": "Kaggle User",
    "competition": "Image Forgery Detection",
    "current_model": "Baseline Model",
    "current_score": 0.0,
    "target_score": 1.0,
    "version": "2.0.0"
}

print("\n" + "="*70)
print("  ğŸ�¯ MULTI-AGENT TASK MANAGER")
print("="*70)
for key, value in AGENT_CONFIG.items():
    print(f"  â€¢ {key}: {value}")
print("="*70)



# Rule-Based AI Assistant (No API Required)
class RuleBasedAssistant:
    """
    Rule-based task analyzer (works without external APIs)
    Uses pattern matching and heuristics
    """
    
    def __init__(self):
        self.call_count = 0
        self.total_time = 0
        
        # Knowledge base for task analysis
        self.patterns = {
            "data": {
                "keywords": ["data", "dataset", "augmentation", "preprocessing", "cleaning"],
                "agent": "executor",
                "priority": "high",
                "base_hours": 4
            },
            "model": {
                "keywords": ["model", "architecture", "network", "efficientnet", "resnet", "unet"],
                "agent": "planner",
                "priority": "critical",
                "base_hours": 8
            },
            "training": {
                "keywords": ["train", "training", "optimizer", "learning rate", "loss"],
                "agent": "executor",
                "priority": "high",
                "base_hours": 6
            },
            "evaluation": {
                "keywords": ["evaluate", "metric", "score", "validation", "test"],
                "agent": "reviewer",
                "priority": "medium",
                "base_hours": 3
            },
            "debug": {
                "keywords": ["debug", "fix", "error", "bug", "issue"],
                "agent": "reviewer",
                "priority": "critical",
                "base_hours": 2
            },
            "analysis": {
                "keywords": ["analyze", "study", "research", "investigate", "explore"],
                "agent": "planner",
                "priority": "medium",
                "base_hours": 4
            }
        }
        
        print(f"âœ… Rule-based Assistant initialized with {len(self.patterns)} pattern categories")
    
    def analyze_task(self, title):
        """
        Analyze task using rule-based patterns
        """
        start_time = time.time()
        title_lower = title.lower()
        
        # Find matching pattern
        matched_category = "general"
        max_matches = 0
        
        for category, pattern in self.patterns.items():
            matches = sum(1 for keyword in pattern["keywords"] if keyword in title_lower)
            if matches > max_matches:
                max_matches = matches
                matched_category = category
        
        # Get pattern or use defaults
        if matched_category in self.patterns:
            pattern = self.patterns[matched_category]
            agent = pattern["agent"]
            priority = pattern["priority"]
            base_hours = pattern["base_hours"]
        else:
            agent = "executor"
            priority = "medium"
            base_hours = 3
        
        # Generate steps based on category
        steps = self._generate_steps(matched_category, title)
        
        # Add some variation to hours
        estimated_hours = base_hours + random.randint(-1, 2)
        estimated_hours = max(1, estimated_hours)
        
        # Update stats
        elapsed = time.time() - start_time
        self.call_count += 1
        self.total_time += elapsed
        
        return {
            "estimatedHours": estimated_hours,
            "priority": priority,
            "agent": agent,
            "steps": steps,
            "category": matched_category
        }
    
    def _generate_steps(self, category, title):
        """Generate action steps based on category"""
        steps_map = {
            "data": [
                "Review current data pipeline and identify bottlenecks",
                "Implement new augmentation techniques or preprocessing methods",
                "Validate data quality and consistency"
            ],
            "model": [
                "Research state-of-the-art architectures for this task",
                "Design or modify model architecture",
                "Test model on validation set and iterate"
            ],
            "training": [
                "Set up training pipeline with proper hyperparameters",
                "Monitor training metrics and adjust as needed",
                "Implement early stopping and checkpointing"
            ],
            "evaluation": [
                "Define evaluation metrics and baseline",
                "Run evaluation on test set",
                "Analyze results and identify improvement areas"
            ],
            "debug": [
                "Reproduce the issue in a controlled environment",
                "Identify root cause using debugging tools",
                "Implement fix and validate solution"
            ],
            "analysis": [
                "Gather relevant data and references",
                "Conduct thorough analysis and document findings",
                "Present insights and recommendations"
            ],
            "general": [
                "Break down task into smaller subtasks",
                "Execute each subtask systematically",
                "Review and validate results"
            ]
        }
        
        return steps_map.get(category, steps_map["general"])
    
    def get_stats(self):
        """Get usage statistics"""
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0
        return {
            "total_calls": self.call_count,
            "total_time": round(self.total_time, 2),
            "avg_time": round(avg_time, 3)
        }

# Initialize Assistant
assistant = RuleBasedAssistant()
print("âœ… Assistant ready to use!")



#Task Manager Class
class TaskManager:
    """
    Manages tasks with AI-powered analysis
    """
    
    def __init__(self, assistant_instance):
        self.assistant = assistant_instance
        self.tasks = []
        print("ğŸ“‚ Task Manager initialized")
    
    def add_task(self, title):
        """Add a new task"""
        task = {
            "id": len(self.tasks) + 1,
            "title": title,
            "status": "todo",
            "priority": "medium",
            "created_at": datetime.now().isoformat(),
            "estimated_hours": 0,
            "agent": None,
            "steps": [],
            "category": None
        }
        
        self.tasks.append(task)
        print(f"âœ… Task added: {title}")
        
        # Analyze with Assistant
        self.analyze_task(task)
        
        return task
    
    def analyze_task(self, task):
        """Analyze task with Assistant"""
        print(f"ğŸ¤– Analyzing task: {task['title']}...")
        
        try:
            analysis = self.assistant.analyze_task(task['title'])
            
            # Update task
            task['estimated_hours'] = analysis.get('estimatedHours', 0)
            task['priority'] = analysis.get('priority', 'medium')
            task['agent'] = analysis.get('agent', 'executor')
            task['steps'] = analysis.get('steps', [])
            task['category'] = analysis.get('category', 'general')
            
            print(f"  âœ“ Category: {task['category']}")
            print(f"  âœ“ Priority: {task['priority']}")
            print(f"  âœ“ Agent: {task['agent']}")
            print(f"  âœ“ Estimated: {task['estimated_hours']}h")
            
        except Exception as e:
            print(f"  âš ï¸� Analysis failed: {e}")
    
    def update_status(self, task_id, new_status):
        """Update task status"""
        for task in self.tasks:
            if task['id'] == task_id:
                old_status = task['status']
                task['status'] = new_status
                print(f"âœ… Task #{task_id} moved from '{old_status}' to '{new_status}'")
                return True
        print(f"â�Œ Task #{task_id} not found")
        return False
    
    def delete_task(self, task_id):
        """Delete a task"""
        for i, task in enumerate(self.tasks):
            if task['id'] == task_id:
                removed = self.tasks.pop(i)
                print(f"ğŸ—‘ï¸� Deleted task: {removed['title']}")
                return True
        print(f"â�Œ Task #{task_id} not found")
        return False
    
    def get_task_by_id(self, task_id):
        """Get task by ID"""
        for task in self.tasks:
            if task['id'] == task_id:
                return task
        return None
    
    def get_stats(self):
        """Get task statistics"""
        return {
            "total": len(self.tasks),
            "todo": len([t for t in self.tasks if t['status'] == 'todo']),
            "in_progress": len([t for t in self.tasks if t['status'] == 'in-progress']),
            "done": len([t for t in self.tasks if t['status'] == 'done']),
            "total_hours": sum(t.get('estimated_hours', 0) for t in self.tasks),
            "by_priority": {
                "critical": len([t for t in self.tasks if t.get('priority') == 'critical']),
                "high": len([t for t in self.tasks if t.get('priority') == 'high']),
                "medium": len([t for t in self.tasks if t.get('priority') == 'medium']),
                "low": len([t for t in self.tasks if t.get('priority') == 'low'])
            },
            "by_agent": {
                "planner": len([t for t in self.tasks if t.get('agent') == 'planner']),
                "executor": len([t for t in self.tasks if t.get('agent') == 'executor']),
                "reviewer": len([t for t in self.tasks if t.get('agent') == 'reviewer'])
            }
        }
    
    def display_board(self):
        """Display task board"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("ğŸ“Š TASK BOARD")
        print("="*70)
        print(f"Total: {stats['total']} | To Do: {stats['todo']} | In Progress: {stats['in_progress']} | Done: {stats['done']} | Hours: {stats['total_hours']}")
        print("="*70)
        
        # Group by status
        statuses = ["todo", "in-progress", "done"]
        status_labels = ["ğŸ“‹ TO DO", "âš¡ IN PROGRESS", "âœ… DONE"]
        
        for status, label in zip(statuses, status_labels):
            tasks_in_status = [t for t in self.tasks if t['status'] == status]
            print(f"\n{label} ({len(tasks_in_status)})")
            print("-" * 70)
            
            if not tasks_in_status:
                print("  (empty)")
            else:
                for task in tasks_in_status:
                    agent_icon = {"planner": "ğŸ§ ", "executor": "âš™ï¸�", "reviewer": "ğŸ”�"}.get(task.get('agent'), "ğŸ“Œ")
                    priority_icon = {"low": "ğŸŸ¢", "medium": "ğŸŸ¡", "high": "ğŸŸ ", "critical": "ğŸ”´"}.get(task.get('priority'), "âšª")
                    
                    print(f"  {agent_icon} #{task['id']} {task['title']}")
                    print(f"     {priority_icon} {task.get('priority', 'medium')} priority | {task.get('estimated_hours', 0)}h | {task.get('category', 'general')}")
                    
                    if task.get('steps'):
                        print(f"     ğŸ“� Action steps:")
                        for i, step in enumerate(task['steps'][:2], 1):  # Show first 2 steps
                            print(f"        {i}. {step}")
                        if len(task['steps']) > 2:
                            print(f"        ... and {len(task['steps']) - 2} more")
        
        print("\n" + "="*70)

# Initialize Task Manager
task_manager = TaskManager(assistant)
print("âœ… Task Manager ready!")


#Interactive Interface
def create_interactive_interface():
    """Create interactive task management interface"""
    
    # Output area
    output = widgets.Output()
    
    # Input widgets
    task_input = widgets.Textarea(
        placeholder='Enter task description (e.g., "Implement data augmentation pipeline")...',
        description='New Task:',
        layout=widgets.Layout(width='98%', height='60px')
    )
    
    add_button = widgets.Button(
        description='â�• Add Task',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    refresh_button = widgets.Button(
        description='ğŸ”„ Refresh',
        button_style='info',
        layout=widgets.Layout(width='150px')
    )
    
    stats_button = widgets.Button(
        description='ğŸ“Š Stats',
        button_style='success',
        layout=widgets.Layout(width='150px')
    )
    
    # Task management widgets
    task_id_input = widgets.IntText(
        description='Task ID:',
        value=1,
        layout=widgets.Layout(width='200px')
    )
    
    status_dropdown = widgets.Dropdown(
        options=['todo', 'in-progress', 'done'],
        description='Status:',
        layout=widgets.Layout(width='200px')
    )
    
    update_button = widgets.Button(
        description='âœ�ï¸� Update',
        button_style='warning',
        layout=widgets.Layout(width='120px')
    )
    
    delete_button = widgets.Button(
        description='ğŸ—‘ï¸� Delete',
        button_style='danger',
        layout=widgets.Layout(width='120px')
    )
    
    # Event handlers
    def on_add_task(b):
        with output:
            clear_output(wait=True)
            if task_input.value.strip():
                task_manager.add_task(task_input.value.strip())
                task_input.value = ''
                print("\n")
                task_manager.display_board()
            else:
                print("âš ï¸� Please enter a task description")
    
    def on_refresh(b):
        with output:
            clear_output(wait=True)
            task_manager.display_board()
    
    def on_stats(b):
        with output:
            clear_output(wait=True)
            stats = task_manager.get_stats()
            assistant_stats = assistant.get_stats()
            
            print("\n" + "="*70)
            print("ğŸ“Š DETAILED STATISTICS")
            print("="*70)
            
            print(f"\nğŸ“‹ Task Overview:")
            print(f"  â€¢ Total Tasks: {stats['total']}")
            print(f"  â€¢ To Do: {stats['todo']}")
            print(f"  â€¢ In Progress: {stats['in_progress']}")
            print(f"  â€¢ Done: {stats['done']}")
            print(f"  â€¢ Total Estimated Hours: {stats['total_hours']}")
            
            print(f"\nğŸ�¯ By Priority:")
            for priority, count in stats['by_priority'].items():
                if count > 0:
                    print(f"  â€¢ {priority.capitalize()}: {count}")
            
            print(f"\nğŸ¤– By Agent:")
            for agent, count in stats['by_agent'].items():
                if count > 0:
                    print(f"  â€¢ {agent.capitalize()}: {count}")
            
            print(f"\nâš¡ Assistant Performance:")
            print(f"  â€¢ Total Analyses: {assistant_stats['total_calls']}")
            print(f"  â€¢ Total Time: {assistant_stats['total_time']}s")
            print(f"  â€¢ Avg Time per Analysis: {assistant_stats['avg_time']}s")
            
            print("="*70)
    
    def on_update(b):
        with output:
            clear_output(wait=True)
            task_id = task_id_input.value
            new_status = status_dropdown.value
            task_manager.update_status(task_id, new_status)
            print("\n")
            task_manager.display_board()
    
    def on_delete(b):
        with output:
            clear_output(wait=True)
            task_id = task_id_input.value
            task_manager.delete_task(task_id)
            print("\n")
            task_manager.display_board()
    
    add_button.on_click(on_add_task)
    refresh_button.on_click(on_refresh)
    stats_button.on_click(on_stats)
    update_button.on_click(on_update)
    delete_button.on_click(on_delete)
    
    # Layout
    input_row = widgets.VBox([task_input, add_button])
    button_row = widgets.HBox([refresh_button, stats_button])
    management_row = widgets.HBox([task_id_input, status_dropdown, update_button, delete_button])
    
    # Display
    display(HTML("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h1 style='margin: 0; font-size: 32px;'>ğŸ¤– Multi-Agent Task Manager</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;'>
            AI-powered task management for Kaggle competitions | No external API required
        </p>
        <p style='margin: 5px 0 0 0; opacity: 0.7; font-size: 14px;'>
            âœ¨ Features: Auto-categorization â€¢ Priority assignment â€¢ Time estimation â€¢ Agent routing
        </p>
    </div>
    """))
    
    print("="*70)
    print("ğŸ’¡ QUICK START GUIDE")
    print("="*70)
    print("1. Enter your task in the text area above")
    print("2. Click 'â�• Add Task' to create it")
    print("3. AI will automatically analyze and categorize your task")
    print("4. Use Task ID + Status + Update to move tasks")
    print("5. Click 'ğŸ“Š Stats' to see detailed analytics")
    print("="*70)
    print()
    
    display(input_row)
    display(button_row)
    display(management_row)
    display(output)
    
    # Show initial board
    with output:
        task_manager.display_board()

# Create interface
create_interactive_interface()


#Example Usage & Pre-populated Demo
print("\n" + "="*70)
print("ğŸš€ DEMO: Adding Sample Tasks")
print("="*70)

# Add some demo tasks
demo_tasks = [
    "Implement advanced data augmentation with mixup and cutmix",
    "Train EfficientNet-B4 model with cosine annealing",
    "Debug RLE mask encoding errors in submission",
    "Analyze top leaderboard solutions",
    "Evaluate model performance on validation set"
]

print("\nAdding demo tasks...\n")
for task_title in demo_tasks:
    task_manager.add_task(task_title)
    print()

print("\n" + "="*70)
print("âœ… Demo complete! Scroll up to see the interactive interface.")
print("="*70)

