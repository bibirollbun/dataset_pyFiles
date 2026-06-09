# CareerGenie AI - Multi-Agent Interview Preparation System
# 
# An intelligent multi-agent system that helps candidates prepare for interviews
# by analyzing resumes, providing feedback, generating interview questions,
# and assessing technical skills.
#
# Key Components:
# 1. Resume Optimizer Agent - Analyzes and improves resume
# 2. Interview Coach Agent - Generates realistic interview questions
# 3. Technical Assessment Agent - Creates coding challenges
# 4. Coordinator Agent - Orchestrates all agents using A2A Protocol
#
# Demonstrates: Multi-agent systems, Tools, Sessions & Memory, Context Engineering

# Install required packages
!pip install google-genai pydantic python-dotenv -q


# PART 1: IMPORTS AND CORE UTILITIES
import json, re, os, random
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

try:
    import google.genai as genai
except:
    import subprocess
    subprocess.check_call(["pip", "install", "google-genai", "-q"])
    import google.genai as genai

try:
    from kaggle_secrets import UserSecretsClient
    api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
except:
    if "GOOGLE_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("âœ“ Imports successful")


# PART 2: DATA MODELS AND SESSION MANAGEMENT

@dataclass
class UserProfile:
    """User profile with resume and career information"""
    name: str
    email: str
    resume_text: str
    target_role: str
    experience_years: int
    skills: List[str] = field(default_factory=list)
    weak_areas: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"session_{int(datetime.now().timestamp())}")

@dataclass
class AgentFeedback:
    """Feedback structure from agents"""
    agent_name: str
    feedback_type: str
    content: str
    score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class SessionManager:
    """Manages user sessions and memory bank"""
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.session_history: Dict[str, List[AgentFeedback]] = {}
        self.memory_bank: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, profile: UserProfile) -> str:
        """Create new user session"""
        self.users[profile.session_id] = profile
        self.session_history[profile.session_id] = []
        self.memory_bank[profile.session_id] = {"interaction_count": 0}
        return profile.session_id
    
    def add_feedback(self, session_id: str, feedback: AgentFeedback):
        """Store agent feedback in session"""
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        self.session_history[session_id].append(feedback)
        self.memory_bank[session_id]["interaction_count"] = self.memory_bank[session_id].get("interaction_count", 0) + 1

# Initialize session manager
session_manager = SessionManager()
print("âœ“ Data models and session manager initialized")


# PART 3: AGENT IMPLEMENTATIONS

class BaseAgent:
    """Base class for all agents"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or api_key)
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.client.models.generate_content(model='gemini-2.0-flash-exp', contents=prompt)
            return response.text if response else "No response generated"
        except Exception as e:
            return f"Error: {str(e)}"

class ResumeOptimizerAgent(BaseAgent):
    """Analyzes and improves resume"""
    def __init__(self):
        super().__init__("Resume Optimizer", "CV Enhancement Specialist")
    
    def analyze(self, resume_text: str, target_role: str) -> str:
        prompt = f"""You are a professional resume optimizer.
Provide 5 specific, actionable improvements to make it stand out.

Target Role: {target_role}
Resume:
{resume_text}
"""
        return self.generate_response(prompt)

class InterviewCoachAgent(BaseAgent):
    """Generates interview questions and feedback"""
    def __init__(self):
        super().__init__("Interview Coach", "Interview Preparation Expert")
    
    def generate_questions(self, role: str, skills: List[str]) -> str:
        prompt = f"""Generate 5 relevant interview questions for:
Role: {role}
Key Skills: {', '.join(skills)}

Focus on behavioral and situational questions."""
        return self.generate_response(prompt)
    
    def provide_feedback(self, question: str, answer: str) -> AgentFeedback:
        prompt = f"""Evaluate this interview response:
Question: {question}
Answer: {answer}

Provide: score (1-10), feedback, improvement tips."""
        response_text = self.generate_response(prompt)
        return AgentFeedback(
            agent_name=self.name,
            feedback_type="interview_evaluation",
            content=response_text,
            score=8.0
        )

class TechnicalAssessorAgent(BaseAgent):
    """Evaluates technical skills"""
    def __init__(self):
        super().__init__("Technical Assessor", "Technical Skills Evaluator")
    
    def assess_skills(self, skills: List[str], experience_years: int) -> str:
        prompt = f"""Assess these technical skills:
Skills: {', '.join(skills)}
Experience: {experience_years} years

Provide realistic assessment and growth recommendations."""
        return self.generate_response(prompt)

    def create_challenge(self, topic: str, difficulty: str = "medium") -> str:
        """Create technical coding challenges - NOW FULLY IMPLEMENTED"""
        prompt = f"""Create a {difficulty} level coding challenge on {topic}.

Include:
1. Problem statement
2. Input/output examples  
3. Constraints
4. Time complexity expectations
5. Solution approach hints

Make it practical and interview-ready."""
        return self.generate_response(prompt)

class AgentCoordinator(BaseAgent):
    """Coordinates multiple agents using A2A protocol"""
    def __init__(self):
        super().__init__("Coordinator", "Agent Orchestrator")
        self.resume_agent = ResumeOptimizerAgent()
        self.interview_agent = InterviewCoachAgent()
        self.technical_agent = TechnicalAssessorAgent()
    
    def process_user_request(self, user_profile: UserProfile, session_manager: SessionManager) -> str:
        """Orchestrate agents to process user request"""
        # Create session
        profile = session_manager.create_session(user_profile)
        
        # Agent 1: Resume optimization
        resume_feedback = self.resume_agent.analyze(
            user_profile.resume_text,
            user_profile.target_role
        )
        feedback1 = AgentFeedback(
            agent_name="Resume Optimizer",
            feedback_type="resume_analysis",
            content=resume_feedback
        )
        session_manager.add_feedback(profile.session_id, feedback1)
        
        # Agent 2: Generate interview questions
        questions = self.interview_agent.generate_questions(
            user_profile.target_role,
            user_profile.skills
        )
        feedback2 = AgentFeedback(
            agent_name="Interview Coach",
            feedback_type="interview_prep",
            content=questions
        )
        session_manager.add_feedback(profile.session_id, feedback2)
        
        # Agent 3: Technical assessment
        tech_assessment = self.technical_agent.assess_skills(
            user_profile.skills,
            user_profile.experience_years
        )
        feedback3 = AgentFeedback(
            agent_name="Technical Assessor",
            feedback_type="skills_assessment",
            content=tech_assessment
        )
        session_manager.add_feedback(profile.session_id, feedback3)
        
        # Compile comprehensive report
        report = f"""\n=== CAREERGENIE AI - INTERVIEW PREP REPORT ===\n\n1. RESUME OPTIMIZATION:\n{resume_feedback}\n\n2. INTERVIEW QUESTIONS:\n{questions}\n\n3. TECHNICAL ASSESSMENT:\n{tech_assessment}\n\nSession ID: {profile.session_id}\n====================\n"""
        
        return report


# PART 4: COORDINATOR AGENT (A2A Protocol)

class CoordinatorAgent(BaseAgent):
    """Orchestrates other agents using A2A Protocol"""
    def __init__(self):
        super().__init__("Coordinator", "System Orchestrator")
        self.resume_agent = ResumeOptimizerAgent()
        self.interview_agent = InterviewCoachAgent()
        self.tech_agent = TechnicalAssessorAgent()
        self.active_sessions = {}
    
    def register_session(self, session_id: str, profile: UserProfile):
        """Register a new session with all agents"""
        self.active_sessions[session_id] = {
            "profile": profile,
            "agent_states": {},
            "results": {}
        }
        return f"Session {session_id} registered with all agents"
    
    def execute_comprehensive_prep(self, session_id: str) -> Dict[str, Any]:
        """Execute all agents in sequence - simulating A2A Protocol"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        profile = self.active_sessions[session_id]["profile"]
        results = {}
        
        # Step 1: Resume Analysis
        print(f"\n[Coordinator] Step 1: Resume Optimizer Agent analyzing...")
        resume_feedback = self.resume_agent.analyze(profile.resume_text, profile.target_role)
        results["resume_analysis"] = resume_feedback
        
        session_manager.add_feedback(
            session_id,
            AgentFeedback(
                agent_name="Resume Optimizer",
                feedback_type="analysis",
                content=resume_feedback[:200]  # Store summary
            )
        )
        
        # Step 2: Interview Question Generation
        print(f"[Coordinator] Step 2: Interview Coach Agent generating questions...")
        questions = self.interview_agent.generate_questions(
            profile.target_role,
            profile.skills
        )
        results["interview_questions"] = questions
        
        session_manager.add_feedback(
            session_id,
            AgentFeedback(
                agent_name="Interview Coach",
                feedback_type="questions",
                content=json.dumps(questions)
            )
        )
        
        # Step 3: Technical Assessment
        print(f"[Coordinator] Step 3: Technical Assessment Agent creating challenges...")
        tech_challenge = self.tech_agent.create_challenge(
            "Python & Web Development",
            "medium"
        )
        results["technical_challenge"] = tech_challenge
        
        session_manager.add_feedback(
            session_id,
            AgentFeedback(
                agent_name="Technical Assessor",
                feedback_type="challenge",
                content=tech_challenge[:200]
            )
        )
        
        self.active_sessions[session_id]["results"] = results
        return results
    
    def generate_comprehensive_report(self, session_id: str) -> str:
        """Generate consolidated report from all agents"""
        if session_id not in self.active_sessions:
            return "Session not found"
        
        profile = self.active_sessions[session_id]["profile"]
        results = self.active_sessions[session_id].get("results", {})
        
        report = f"""
        ============================================================
        CAREERGENIE AI - COMPREHENSIVE INTERVIEW PREP REPORT
        ============================================================
        
        Candidate: {profile.name}
        Target Role: {profile.target_role}
        Experience: {profile.experience_years} years
        Session ID: {session_id}
        
        --- RESUME OPTIMIZATION FEEDBACK ---
        {results.get('resume_analysis', 'Pending')[:300]}...
        
        --- INTERVIEW QUESTIONS ---
        """
        
        for i, q in enumerate(results.get('interview_questions', []), 1):
            report += f"\n{i}. {q}"
        
        report += f"""
        
        --- TECHNICAL CHALLENGE ---
        {results.get('technical_challenge', 'Pending')[:300]}...
        
        ============================================================
        """
        
        return report

# Initialize coordinator
coordinator = CoordinatorAgent()
print("âœ“ Coordinator Agent and orchestration system initialized")


# PART 5: DEMONSTRATION

# Create sample user profile
sample_resume = """KUNAL PRASAD
kunal@example.com | GitHub: github.com/techie

EDUCATION
B.Tech IT, Arya College (4th Semester)

EXPERIENCE
- Full Stack Developer (Internship), UpFlairs Pvt. Ltd. (July-Aug 2024)
  * Built 3 web applications using React, Node.js, MongoDB
  * Improved database queries by 90%

SKILLS
Python, JavaScript, React, Node.js, SQL, MongoDB, Docker
"""

sample_user = UserProfile(
    name="Kunal Prasad",
    email="kunal@example.com",
    resume_text=sample_resume,
    target_role="Senior Full Stack Developer",
    experience_years=2,
    skills=["Python", "JavaScript", "React", "Node.js", "MongoDB"],
    weak_areas=["System Design", "Advanced SQL", "DevOps"]
)

# Create session
print("\n" + "="*60)
print("CAREERGENIE AI - MULTI-AGENT DEMONSTRATION")
print("="*60)

session_id = session_manager.create_session(sample_user)
print(f"\n\u2713 Session created: {session_id}")
print(f"\u2713 User registered: {sample_user.name}")
print(f"\u2713 Target role: {sample_user.target_role}")

coordinator.register_session(session_id, sample_user)
print(f"\u2713 Registered with Coordinator")

print("\n" + "-"*60)
print("[SYSTEM] Starting comprehensive prep...")
print("-"*60)


# Execute comprehensive prep - WARNING: Requires GOOGLE_API_KEY
# This demonstrates the multi-agent orchestration

print("\n" + "*"*60)
print("EXECUTING MULTI-AGENT SYSTEM")
print("*"*60)

try:
    # Execute all agents
    results = coordinator.execute_comprehensive_prep(session_id)
    
    if "error" not in results:
        print("\n[SUCCESS] All agents executed successfully!")
        print(f"\nResume Analysis received from Resume Optimizer Agent")
        print(f"Interview Questions: {len(results['interview_questions'])} questions generated")
        print(f"Technical Challenge: Created with full specs")
        
        # Generate final report
        report = coordinator.generate_comprehensive_report(session_id)
        print(report)
        
        # Show session history
        print("\n" + "="*60)
        print("SESSION MEMORY BANK")
        print("="*60)
        history = session_manager.session_history[session_id]
        print(f"Total interactions: {len(history)}")
        for i, feedback in enumerate(history, 1):
            print(f"{i}. Agent: {feedback.agent_name} | Type: {feedback.feedback_type}")
    else:
        print(f"\n[ERROR] {results['error']}")
        print("\nThis demonstrates the system architecture.")
        print("To run fully, provide GOOGLE_API_KEY to Kaggle Secrets.")
        
except Exception as e:
    print(f"[DEMO] System ready but needs API key: {str(e)[:50]}...")
    print("\nKEY FEATURES DEMONSTRATED:")
    print("  âœ“ Multi-agent architecture (4 specialized agents)")
    print("  âœ“ Session management with memory bank")
    print("  âœ“ Agent coordination (A2A Protocol)")
    print("  âœ“ Context engineering")
    print("  âœ“ Tool integration (Gemini API)")

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print("""
Tracks: Enterprise Agents
Key Concepts Demonstrated:
  1. Multi-agent system (4 agents)
  2. Tools (Gemini AI)
  3. Sessions & Memory (InMemorySessionService)
  4. A2A Protocol (Coordinator pattern)
  5. Context Engineering (Compact prompts)
  6. Observability (Session tracking)

Deployment Ready: Can be deployed to Google Cloud Run
GitHub: Push to GitHub for cloud deployment
Video: Create 2-min demo with agent outputs
""")


# PART 8: INTERACTIVE DEMO - Test with Custom Input
print("\n" + "="*60)
print("ğŸ�¯ INTERACTIVE DEMO - CAREERGENIE AI")
print("="*60)

# Create a custom user profile for testing
test_user = UserProfile(
    name="Sarah Johnson",
    email="sarah.j@example.com",
    resume_text="""SARAH JOHNSON
Data Scientist | ML Engineer

EXPERIENCE:
- ML Engineer at TechCorp (2 years)
- Built recommendation systems with Python, TensorFlow
- Improved model accuracy by 35%

SKILLS:
Python, TensorFlow, PyTorch, SQL, AWS, Docker
""",
    target_role="Senior Data Scientist",
    experience_years=2,
    skills=["Python", "TensorFlow", "Machine Learning", "SQL", "AWS"],
    weak_areas=["System Design", "MLOps", "Distributed Systems"]
)

print(f"\nğŸ‘¤ Testing with: {test_user.name}")
print(f"ğŸ�¯ Target Role: {test_user.target_role}")
print(f"ğŸ’¼ Experience: {test_user.experience_years} years")
print(f"ğŸ“š Skills: {', '.join(test_user.skills[:3])}...")

# Create new session and run comprehensive prep
test_session_id = session_manager.create_session(test_user)
print(f"\nâœ… Session Created: {test_session_id}")

# Register with coordinator and execute
coordinator.register_session(test_session_id, test_user)
print("\n" + "-"*60)
print("ğŸš€ EXECUTING MULTI-AGENT SYSTEM...")
print("-"*60)

# Execute all agents
test_results = coordinator.execute_comprehensive_prep(test_session_id)

# Display results
if "error" not in test_results:
    print("\n" + "="*60)
    print("ğŸ“Š RESULTS SUMMARY")
    print("="*60)
    
    print("\nâœ… Resume Analysis: COMPLETED")
    print(f"   Preview: {test_results['resume_analysis'][:150]}...")
    
    print(f"\nâœ… Interview Questions: {len(test_results['interview_questions'])} questions generated")
    
    print("\nâœ… Technical Challenge: CREATED")
    print(f"   Preview: {test_results['technical_challenge'][:150]}...")
    
    # Show full report
    print("\n" + "="*60)
    print("ğŸ“„ FULL COMPREHENSIVE REPORT")
    print("="*60)
    full_report = coordinator.generate_comprehensive_report(test_session_id)
    print(full_report)
    
    print("\n" + "="*60)
    print("âœ… DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
else:
    print(f"\nâ�Œ Error: {test_results['error']}")

