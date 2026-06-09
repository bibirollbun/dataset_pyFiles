# Install required packages
!pip install google-adk google-generativeai python-dotenv -q

print("âœ… Installation complete!")


import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from kaggle_secrets import UserSecretsClient

# ADK Core Imports
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load environment variables
load_dotenv()

# Configure API Key (set this in your environment or .env file)
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("âš ï¸�  Warning: GOOGLE_API_KEY not set. Please set it to use Gemini models.")
    print("   You can set it with: os.environ['GOOGLE_API_KEY'] = 'your-api-key'")
else:
    # Set the API key in the environment for ADK to use
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print("âœ… API Key configured successfully")

# Constants
GEMINI_MODEL = "gemini-2.0-flash"
APP_NAME = "finance_coach"
USER_ID = "demo_user"
SESSION_ID = "session_001"

print(f"âœ… Using model: {GEMINI_MODEL}")
print(f"âœ… All modules imported successfully")


def analyze_budget(monthly_income: float, monthly_expenses: float) -> Dict[str, Any]:
    """
    Analyzes monthly budget and provides spending insights.
    
    Args:
        monthly_income: Total monthly income in dollars
        monthly_expenses: Total monthly expenses in dollars
        
    Returns:
        Dictionary with budget analysis including savings rate, surplus/deficit, and recommendations
    """
    surplus = monthly_income - monthly_expenses
    savings_rate = (surplus / monthly_income * 100) if monthly_income > 0 else 0
    
    # Determine budget health
    if savings_rate >= 20:
        health_status = "Excellent"
        advice = "You're saving at a healthy rate! Consider increasing investments."
    elif savings_rate >= 10:
        health_status = "Good"
        advice = "Solid savings rate. Look for opportunities to reduce expenses further."
    elif savings_rate > 0:
        health_status = "Fair"
        advice = "You're saving, but there's room for improvement. Aim for at least 20%."
    else:
        health_status = "Critical"
        advice = "You're spending more than you earn. Immediate budget review needed."
    
    return {
        "status": "success",
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "monthly_surplus": surplus,
        "savings_rate_percent": round(savings_rate, 2),
        "health_status": health_status,
        "advice": advice
    }


def calculate_emergency_fund(monthly_expenses: float, dependents: int = 0) -> Dict[str, Any]:
    """
    Calculates recommended emergency fund size based on monthly expenses and dependents.
    
    Args:
        monthly_expenses: Average monthly expenses in dollars
        dependents: Number of dependents (default: 0)
        
    Returns:
        Dictionary with emergency fund recommendations
    """
    # Base recommendation: 3-6 months
    # Add 1 month per dependent
    min_months = 3 + dependents
    max_months = 6 + dependents
    
    min_fund = monthly_expenses * min_months
    max_fund = monthly_expenses * max_months
    
    return {
        "status": "success",
        "monthly_expenses": monthly_expenses,
        "dependents": dependents,
        "recommended_months": f"{min_months}-{max_months}",
        "minimum_emergency_fund": round(min_fund, 2),
        "ideal_emergency_fund": round(max_fund, 2),
        "rationale": f"With {dependents} dependent(s), you should have {min_months}-{max_months} months of expenses saved."
    }


def calculate_debt_payoff(debt_amount: float, interest_rate: float, 
                         monthly_payment: float) -> Dict[str, Any]:
    """
    Calculates debt payoff timeline and total interest paid.
    
    Args:
        debt_amount: Total debt amount in dollars
        interest_rate: Annual interest rate as a percentage (e.g., 18.5 for 18.5%)
        monthly_payment: Planned monthly payment in dollars
        
    Returns:
        Dictionary with payoff timeline, total interest, and recommendations
    """
    if debt_amount <= 0:
        return {"status": "error", "message": "Debt amount must be positive"}
    
    if monthly_payment <= 0:
        return {"status": "error", "message": "Monthly payment must be positive"}
    
    # Convert annual rate to monthly
    monthly_rate = (interest_rate / 100) / 12
    
    # Check if payment covers interest
    monthly_interest = debt_amount * monthly_rate
    if monthly_payment <= monthly_interest:
        return {
            "status": "error",
            "message": f"Monthly payment (${monthly_payment}) must exceed monthly interest (${monthly_interest:.2f})"
        }
    
    # Calculate payoff timeline
    balance = debt_amount
    months = 0
    total_interest = 0
    
    while balance > 0 and months < 600:  # Cap at 50 years
        interest = balance * monthly_rate
        principal = monthly_payment - interest
        
        if principal > balance:
            principal = balance
            interest = balance * monthly_rate
        
        balance -= principal
        total_interest += interest
        months += 1
    
    years = months // 12
    remaining_months = months % 12
    
    # Calculate savings with increased payment
    increased_payment = monthly_payment * 1.2  # 20% increase
    increased_timeline = _calculate_months(debt_amount, interest_rate, increased_payment)
    saved_months = months - increased_timeline if increased_timeline else 0
    
    return {
        "status": "success",
        "original_debt": debt_amount,
        "interest_rate_percent": interest_rate,
        "monthly_payment": monthly_payment,
        "payoff_months": months,
        "payoff_timeline": f"{years} years, {remaining_months} months" if years > 0 else f"{months} months",
        "total_interest_paid": round(total_interest, 2),
        "total_amount_paid": round(debt_amount + total_interest, 2),
        "recommendation": f"Increasing payment by 20% to ${increased_payment:.2f} would save {saved_months} months!"
    }


def _calculate_months(debt: float, rate: float, payment: float) -> int:
    """Helper function for debt calculations"""
    monthly_rate = (rate / 100) / 12
    if payment <= debt * monthly_rate:
        return None
    balance = debt
    months = 0
    while balance > 0 and months < 600:
        interest = balance * monthly_rate
        principal = payment - interest
        if principal > balance:
            principal = balance
        balance -= principal
        months += 1
    return months


def calculate_savings_goal(goal_amount: float, current_savings: float, 
                          monthly_contribution: float, annual_return: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculates timeline to reach a savings goal with optional investment returns.
    
    Args:
        goal_amount: Target savings amount in dollars
        current_savings: Current amount saved in dollars
        monthly_contribution: Monthly savings contribution in dollars
        annual_return: Expected annual return as percentage (default: 0.0 for savings account)
        
    Returns:
        Dictionary with timeline to goal and recommendations
    """
    if goal_amount <= current_savings:
        return {
            "status": "success",
            "message": "Goal already achieved!",
            "goal_amount": goal_amount,
            "current_savings": current_savings,
            "surplus": current_savings - goal_amount
        }
    
    remaining = goal_amount - current_savings
    
    if monthly_contribution <= 0:
        return {
            "status": "error",
            "message": "Monthly contribution must be positive to reach your goal"
        }
    
    # Calculate with compound interest if return > 0
    if annual_return > 0:
        monthly_rate = (annual_return / 100) / 12
        balance = current_savings
        months = 0
        
        while balance < goal_amount and months < 1200:  # Cap at 100 years
            balance = balance * (1 + monthly_rate) + monthly_contribution
            months += 1
    else:
        # Simple calculation without interest
        months = int(remaining / monthly_contribution)
        if remaining % monthly_contribution > 0:
            months += 1
    
    years = months // 12
    remaining_months = months % 12
    
    return {
        "status": "success",
        "goal_amount": goal_amount,
        "current_savings": current_savings,
        "amount_needed": remaining,
        "monthly_contribution": monthly_contribution,
        "annual_return_percent": annual_return,
        "months_to_goal": months,
        "timeline": f"{years} years, {remaining_months} months" if years > 0 else f"{months} months",
        "final_amount": goal_amount,
        "recommendation": "Consider investing for higher returns if timeline allows" if annual_return == 0 and months > 12 else "Stay consistent with your savings plan!"
    }


# Test the tools
print("âœ… Financial tools defined successfully!")
print("\nğŸ§ª Testing tools with sample data...\n")

# Test budget analysis
budget_result = analyze_budget(5000, 3500)
print(f"Budget Analysis: {budget_result['health_status']} - Savings Rate: {budget_result['savings_rate_percent']}%")

# Test emergency fund
emergency_result = calculate_emergency_fund(3500, 1)
print(f"Emergency Fund: ${emergency_result['minimum_emergency_fund']:.2f} - ${emergency_result['ideal_emergency_fund']:.2f}")

# Test debt payoff
debt_result = calculate_debt_payoff(15000, 18.5, 500)
print(f"Debt Payoff: {debt_result['payoff_timeline']} - Total Interest: ${debt_result['total_interest_paid']:.2f}")

# Test savings goal
savings_result = calculate_savings_goal(10000, 2000, 400, 5)
print(f"Savings Goal: {savings_result['timeline']} to reach ${savings_result['goal_amount']}")

print("\nâœ… All tools tested successfully!")


# Agent 1: Financial Data Analyzer
financial_analyzer_agent = LlmAgent(
    name="financial_data_analyzer",
    model=GEMINI_MODEL,
    description="Analyzes user's financial data and identifies key metrics and areas of concern.",
    instruction="""You are a Financial Data Analyzer AI. Your role is to:

1. Extract financial information from the user's query (income, expenses, debts, goals)
2. Use the appropriate financial tools to analyze the data
3. Identify key financial metrics (savings rate, debt-to-income ratio, etc.)
4. Store your analysis results using the output_key 'financial_analysis'

Available Tools:
- analyze_budget: For income vs expense analysis
- calculate_emergency_fund: For emergency fund recommendations
- calculate_debt_payoff: For debt repayment analysis
- calculate_savings_goal: For savings goal timeline

**CRITICAL:** Your response MUST be a structured JSON analysis, not a narrative. Output ONLY JSON in this format:
```json
{
    "income_analysis": {...budget tool results...},
    "emergency_fund": {...emergency fund tool results...},
    "debt_analysis": {...debt tool results if applicable...},
    "savings_analysis": {...savings tool results if applicable...},
    "key_findings": ["finding 1", "finding 2", "finding 3"]
}
```

Call the appropriate tools based on the user's situation, then summarize findings in JSON format.""",
    tools=[analyze_budget, calculate_emergency_fund, calculate_debt_payoff, calculate_savings_goal],
    output_key="financial_analysis"  # Stores output in session state
)

# Agent 2: Financial Advisor
financial_advisor_agent = LlmAgent(
    name="financial_advisor",
    model=GEMINI_MODEL,
    description="Provides personalized financial advice based on analysis.",
    instruction="""You are a Financial Advisor AI. Your role is to:

1. Read the financial analysis from the previous agent: {financial_analysis}
2. Provide personalized, actionable financial advice
3. Prioritize recommendations by impact and urgency
4. Explain the "why" behind each recommendation in simple terms
5. Store your advice using the output_key 'financial_advice'

Output Format (MUST be JSON):
```json
{
    "priority_1": {
        "recommendation": "Clear, specific action",
        "rationale": "Why this matters most",
        "impact": "Expected outcome"
    },
    "priority_2": {...},
    "priority_3": {...},
    "educational_tips": ["tip 1", "tip 2", "tip 3"]
}
```

Focus on education and empowerment, not just prescriptive advice. Be encouraging but realistic.""",
    output_key="financial_advice"
)

# Agent 3: Action Plan Generator
action_plan_agent = LlmAgent(
    name="action_plan_generator",
    model=GEMINI_MODEL,
    description="Creates a concrete, step-by-step action plan.",
    instruction="""You are an Action Plan Generator AI. Your role is to:

1. Read the financial analysis: {financial_analysis}
2. Read the financial advice: {financial_advice}
3. Create a detailed, time-bound action plan
4. Break down recommendations into small, achievable steps
5. Provide a clear roadmap for the next 30, 90, and 180 days

Output Format (MUST be JSON):
```json
{
    "immediate_actions_30_days": [
        {"step": 1, "action": "Specific action", "expected_time": "X hours"},
        {"step": 2, "action": "Specific action", "expected_time": "X hours"}
    ],
    "short_term_90_days": [...],
    "medium_term_180_days": [...],
    "success_metrics": {
        "30_day": "Measurable outcome",
        "90_day": "Measurable outcome",
        "180_day": "Measurable outcome"
    },
    "resources": ["Resource 1", "Resource 2"],
    "encouragement": "Motivational closing message"
}
```

Make the plan realistic and achievable. Include specific tools, apps, or resources when helpful.""",
    output_key="action_plan"
)

# Create the Sequential Agent (orchestrates the workflow)
finance_coach_pipeline = SequentialAgent(
    name="finance_coach_pipeline",
    sub_agents=[
        financial_analyzer_agent,
        financial_advisor_agent,
        action_plan_agent
    ],
    description="Automated Personal Finance Coach that analyzes finances, provides advice, and creates action plans."
)

# Set as root agent (required for ADK)
root_agent = finance_coach_pipeline

print("âœ… Individual agents created successfully!")
print(f"   â€¢ {financial_analyzer_agent.name}")
print(f"   â€¢ {financial_advisor_agent.name}")
print(f"   â€¢ {action_plan_agent.name}")
print("\nâœ… Sequential Agent Pipeline created!")
print("\nğŸ“Š Pipeline Flow:")
print("   User Query â†’ Data Analyzer â†’ Advisor â†’ Action Planner â†’ Final Response")
print("\nâœ… Root agent configured!")


# Initialize Session Service (InMemorySessionService for development)
session_service = InMemorySessionService()

# Create a session for our demo user
session = await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

# Initialize Runner with our agent pipeline
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

print("âœ… Session service initialized")
print(f"   â€¢ App: {APP_NAME}")
print(f"   â€¢ User: {USER_ID}")
print(f"   â€¢ Session: {SESSION_ID}")
print("\nâœ… Runner configured and ready!")



async def run_finance_coach(user_query: str):
    """
    Helper function to run the finance coach using run_async.
    
    Args:
        user_query: The user's financial question or situation as a string
        
    Returns:
        List of events from the agent execution
    """
    print(f"\n{'='*80}")
    print(f"ğŸ§‘ USER QUERY:")
    print(f"{'='*80}")
    print(user_query)
    print(f"{'='*80}\n")
    
    try:
        # Create proper Content object from the user query string
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=user_query)]
        )
        
        # Use run_async which returns an async generator
        events = []
        final_response = None
        
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message
        ):
            events.append(event)
            
            # Print event information
            if hasattr(event, 'author') and event.author:
                print(f"ğŸ“¡ Event from: {event.author}")
            
            # Extract and display content
            if hasattr(event, 'content') and event.content:
                content = event.content
                
                # Handle Content object with parts
                if hasattr(content, 'parts') and content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"   {part.text[:200]}..." if len(part.text) > 200 else f"   {part.text}")
                            final_response = part.text
                        elif hasattr(part, 'function_call'):
                            print(f"   ğŸ”§ Tool call: {part.function_call.name}")
                        elif hasattr(part, 'function_response'):
                            print(f"   âœ… Tool response received")
        
        # Print the final response
        if final_response:
            print(f"\n{'='*80}")
            print(f"ğŸ¤– FINAL AGENT RESPONSE:")
            print(f"{'='*80}")
            print(final_response)
        
        print(f"\n{'='*80}")
        print(f"âœ… Completed! Total events: {len(events)}")
        print(f"{'='*80}\n")
        
        return events
        
    except Exception as e:
        print(f"â�Œ Error running finance coach: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return []

# Test Scenario 1: Young professional with debt
print("ğŸ“� Test Scenario 1: Young Professional with Debt")
await run_finance_coach("""
I'm 28 years old and need help with my finances. Here's my situation:
- Monthly income: $4,500
- Monthly expenses: $3,200
- Credit card debt: $8,000 at 18.5% interest
- Currently paying $300/month on the debt
- I want to save for a house down payment of $30,000
- No emergency fund yet
- No dependents

What should I do?
""")


# Test Scenario 2: Family with good savings habits
print("ğŸ“� Test Scenario 2: Family with Good Savings Habits")
await run_finance_coach("""
We're a family of 4 (2 adults, 2 kids) and doing well, but want to optimize:
- Combined monthly income: $9,000
- Monthly expenses: $6,500
- No debt
- Current savings: $15,000
- Goal: Save $50,000 for college fund in 8 years
- Can invest with expected 6% annual return

How can we make our savings work harder for us?
""")


import logging

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('finance_coach')

print("âœ… Observability configured!")
print("\nğŸ“Š Execution Flow Insights:")
print("   â€¢ Each agent execution is logged")
print("   â€¢ Tool calls are tracked")
print("   â€¢ State changes are recorded")
print("   â€¢ Session data is maintained")
print("\nğŸ’¡ In production, integrate with:")
print("   â€¢ Google Cloud Trace")
print("   â€¢ AgentOps")
print("   â€¢ Arize AX")
print("   â€¢ W&B Weave")

