# Install required packages
!pip install -q google-genai google-adk pydantic pydantic-settings

print("âœ… Dependencies installed successfully!")


# Create directory structure for source files
import os

directories = [
    "src/capstone/models",
    "src/capstone/tools",
    "src/capstone/agents",
    "src/capstone/infrastructure",
]

for d in directories:
    os.makedirs(d, exist_ok=True)

print("Directory structure created successfully!")


import logging


"""
# Suppress google_genai warnings

i.e.:

WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'], returning
concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model
response.
"""
logging.getLogger("google_genai.types").setLevel(logging.ERROR)



%%writefile src/capstone/models/__init__.py
"""Pydantic models for the Neurodivergent Parenting Support Agent."""

from .activity import ActivityGoal, ActivityPlan, ActivityRequest
from .behavior import (
    ActivityType,
    BehaviorAnalysis,
    BehaviorFactor,
    BehaviorInput,
    TimeOfDay,
)
from .memory import BehaviorPattern, SessionOutcome
from .results import ToolError, ToolResult, ToolSuccess, is_error, is_success
from .strategy import BehaviorResponse, Strategy

__all__ = [
    # Activity models
    "ActivityGoal",
    "ActivityPlan",
    "ActivityRequest",
    # Behavior models
    "ActivityType",
    "BehaviorAnalysis",
    "BehaviorFactor",
    "BehaviorInput",
    "TimeOfDay",
    # Memory models
    "BehaviorPattern",
    "SessionOutcome",
    # Result models
    "ToolError",
    "ToolResult",
    "ToolSuccess",
    "is_error",
    "is_success",
    # Strategy models
    "BehaviorResponse",
    "Strategy",
]


%%writefile src/capstone/models/behavior.py
"""Behavioral analysis models with type safety."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class BehaviorFactor(str, Enum):
    """Behavioral factor categories."""

    ADHD = "adhd"
    ASD = "asd"
    AGE_TYPICAL = "age_typical"
    COMBINED = "combined"


class TimeOfDay(str, Enum):
    """Time periods for behavior tracking."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    BEDTIME = "bedtime"


class ActivityType(str, Enum):
    """Types of activities."""

    SCHOOL = "school"
    HOMEWORK = "homework"
    PLAY = "play"
    TRANSITIONS = "transitions"
    MEALS = "meals"
    BEDTIME_ROUTINE = "bedtime_routine"
    HYGIENE = "hygiene"


class BehaviorInput(BaseModel):
    """Input for behavior analysis."""

    description: str = Field(..., min_length=10, max_length=1000)
    time_of_day: TimeOfDay
    activity_type: ActivityType
    context: str = Field(..., min_length=5, max_length=500)

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=False,
        extra="forbid",
    )


class BehaviorAnalysis(BaseModel):
    """Result of behavior analysis."""

    adhd_factors: list[str] = Field(default_factory=list)
    asd_factors: list[str] = Field(default_factory=list)
    age_typical_factors: list[str] = Field(default_factory=list)
    primary_driver: BehaviorFactor

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=False,
        extra="forbid",
        frozen=True,
    )


%%writefile src/capstone/models/activity.py
"""Activity planning models."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ActivityGoal(str, Enum):
    """Activity goal categories."""

    ENGAGEMENT = "engagement"
    EXECUTIVE_FUNCTION = "executive_function"
    TRANSITION_SUPPORT = "transition_support"
    HOMEWORK_SUPPORT = "homework_support"


class ActivityPlan(BaseModel):
    """Complete activity plan with all preparation details."""

    name: str = Field(..., min_length=5, max_length=100)
    goal: ActivityGoal
    materials: list[str] = Field(..., min_length=1)
    environmental_setup: str = Field(..., min_length=10)
    duration_minutes: int = Field(..., ge=5, le=120)
    structure: list[str] = Field(..., min_length=1)
    success_criteria: list[str] = Field(..., min_length=1)
    adaptations: list[str] = Field(default_factory=list)
    from_memory: str | None = Field(None, description="Past successful patterns")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=False,
        extra="forbid",
    )


class ActivityRequest(BaseModel):
    """Request for activity planning."""

    goal: ActivityGoal
    duration_minutes: int = Field(..., ge=5, le=120)
    available_materials: list[str] = Field(default_factory=list)
    specific_needs: str | None = Field(None, max_length=500)

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=False,
        extra="forbid",
    )


%%writefile src/capstone/models/memory.py
"""Memory and pattern tracking models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionOutcome(BaseModel):
    """Outcome of a strategy application."""

    strategy_used: str = Field(..., min_length=5)
    worked: bool
    notes: str = Field(..., min_length=5, max_length=500)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(
        validate_assignment=True,
        strict=False,
        extra="forbid",
    )


class BehaviorPattern(BaseModel):
    """Identified behavioral pattern."""

    behavior_type: str = Field(..., min_length=3)
    common_triggers: list[str] = Field(default_factory=list)
    successful_strategies: list[str] = Field(default_factory=list)
    unsuccessful_approaches: list[str] = Field(default_factory=list)
    frequency_count: int = Field(..., ge=0)
    sessions_analyzed: int = Field(..., ge=1)

    model_config = ConfigDict(
        validate_assignment=True,
        strict=False,
        extra="forbid",
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_frequency(self) -> "BehaviorPattern":
        """Validate frequency doesn't exceed sessions analyzed."""
        if self.frequency_count > self.sessions_analyzed:
            raise ValueError("Frequency cannot exceed sessions analyzed")
        return self


%%writefile src/capstone/models/results.py
"""Discriminated union for tool results."""

from typing import Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field


class ToolSuccess(BaseModel):
    """Successful tool execution."""

    success: Literal[True] = True
    data: dict[str, str | int | float | bool | list[str]]

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid",
        frozen=True,
    )

    def get_str(self, key: str, default: str = "") -> str:
        """Get a string value from data with type safety."""
        value = self.data.get(key, default)
        if isinstance(value, str):
            return value
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer value from data with type safety."""
        value = self.data.get(key, default)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float value from data with type safety."""
        value = self.data.get(key, default)
        if isinstance(value, float):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean value from data with type safety."""
        value = self.data.get(key, default)
        if isinstance(value, bool):
            return value
        return default

    def get_list(self, key: str, default: list[str] | None = None) -> list[str]:
        """Get a list value from data with type safety."""
        if default is None:
            default = []
        value = self.data.get(key, default)
        if isinstance(value, list):
            return value
        return default


class ToolError(BaseModel):
    """Tool execution error."""

    success: Literal[False] = False
    error_code: str = Field(..., min_length=1)
    error_message: str = Field(..., min_length=1)

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid",
        frozen=True,
    )


ToolResult = ToolSuccess | ToolError


def is_success(result: ToolResult) -> TypeGuard[ToolSuccess]:
    """Type guard for successful results."""
    return result.success is True


def is_error(result: ToolResult) -> TypeGuard[ToolError]:
    """Type guard for error results."""
    return result.success is False


%%writefile src/capstone/models/strategy.py
"""Strategy response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .behavior import BehaviorAnalysis


class Strategy(BaseModel):
    """Individual strategy recommendation."""

    title: str = Field(..., min_length=5, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)
    rationale: str = Field(..., min_length=10, max_length=300)
    success_indicators: list[str] = Field(..., min_length=1)

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid",
        frozen=True,
    )


class BehaviorResponse(BaseModel):
    """Complete behavioral analysis response."""

    acknowledgment: str = Field(..., min_length=10, max_length=200)
    analysis: BehaviorAnalysis
    strategies: list[Strategy] = Field(..., min_length=1, max_length=5)
    proactive_suggestion: str | None = Field(None, max_length=300)
    disclaimer: Literal["I'm a parenting support tool, not medical advice."] = (
        "I'm a parenting support tool, not medical advice."
    )

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
        frozen=True,
    )


%%writefile src/capstone/tools/__init__.py
"""Type-safe custom tools for the parenting support agent."""

from .activity_planner import get_activity_plan
from .behavior_classifier import classify_behavior
from .pattern_analyzer import analyze_patterns

__all__ = [
    "classify_behavior",
    "get_activity_plan",
    "analyze_patterns",
]


%%writefile src/capstone/tools/behavior_classifier.py
"""Type-safe behavior classification tool."""

from ..models.behavior import (
    ActivityType,
    BehaviorAnalysis,
    BehaviorFactor,
    BehaviorInput,
)
from ..models.results import ToolError, ToolResult, ToolSuccess


def classify_behavior(behavior_input: BehaviorInput) -> ToolResult:
    """
    Classify behavior into ADHD/ASD/age-typical factors.

    Args:
        behavior_input: Validated behavior input with description and context

    Returns:
        ToolResult: Success with BehaviorAnalysis or Error with details
    """
    try:
        description_lower = behavior_input.description.lower()
        context_lower = behavior_input.context.lower()
        combined_text = f"{description_lower} {context_lower}"

        # Pattern matching for ADHD indicators
        adhd_factors: list[str] = []
        if any(
                keyword in combined_text
                for keyword in ["distracted", "focus", "attention", "concentrate"]
        ):
            adhd_factors.append("Attention regulation challenge")
        if any(keyword in combined_text for keyword in ["transition", "switching"]):
            adhd_factors.append("Executive function: difficulty with transitions")
        if "task" in combined_text and any(
                keyword in combined_text for keyword in ["start", "initiate", "begin"]
        ):
            adhd_factors.append("Executive function: task initiation difficulty")
        if any(keyword in combined_text for keyword in ["organize", "plan", "remember"]):
            adhd_factors.append("Executive function: planning and organization")
        if any(keyword in combined_text for keyword in ["impulsive", "interrupt"]):
            adhd_factors.append("Impulse control challenge")

        # Pattern matching for ASD indicators
        asd_factors: list[str] = []
        if any(keyword in combined_text for keyword in ["routine", "change", "unexpected"]):
            asd_factors.append("Routine disruption sensitivity")
        if any(keyword in combined_text for keyword in ["alone", "independent"]):
            asd_factors.append("Social/independence adjustment")
        if any(keyword in combined_text for keyword in ["sensory", "noise", "loud"]):
            asd_factors.append("Sensory processing needs")
        if any(keyword in combined_text for keyword in ["social", "friend", "peer"]):
            asd_factors.append("Social communication challenge")
        if any(keyword in combined_text for keyword in ["rigid", "flexible", "adapt"]):
            asd_factors.append("Cognitive flexibility challenge")

        # Age-typical factors
        age_typical: list[str] = []
        if behavior_input.activity_type == ActivityType.HOMEWORK:
            age_typical.append("Homework resistance common at age 8")
        if any(keyword in combined_text for keyword in ["refused", "won't", "no"]):
            age_typical.append("Testing boundaries - developmental stage")
        if behavior_input.activity_type == ActivityType.BEDTIME_ROUTINE:
            age_typical.append("Bedtime resistance typical for age group")
        if "independent" in combined_text or "alone" in combined_text:
            age_typical.append("Seeking independence - age-appropriate development")

        # Determine primary driver
        adhd_count = len(adhd_factors)
        asd_count = len(asd_factors)
        typical_count = len(age_typical)

        if adhd_count > 0 and asd_count > 0 and adhd_count == asd_count:
            primary = BehaviorFactor.COMBINED
        elif adhd_count > asd_count and adhd_count > typical_count:
            primary = BehaviorFactor.ADHD
        elif asd_count > adhd_count and asd_count > typical_count:
            primary = BehaviorFactor.ASD
        elif adhd_count > 0 and asd_count > 0:
            primary = BehaviorFactor.COMBINED
        else:
            primary = BehaviorFactor.AGE_TYPICAL

        # Create and validate analysis using Pydantic model
        analysis = BehaviorAnalysis(
            adhd_factors=adhd_factors,
            asd_factors=asd_factors,
            age_typical_factors=age_typical,
            primary_driver=primary,
        )

        # Use validated model for response
        return ToolSuccess(
            data={
                "adhd_factors": analysis.adhd_factors,
                "asd_factors": analysis.asd_factors,
                "age_typical_factors": analysis.age_typical_factors,
                "primary_driver": analysis.primary_driver.value,
            }
        )

    except Exception as e:
        return ToolError(
            error_code="CLASSIFICATION_ERROR",
            error_message=f"Failed to classify behavior: {str(e)}",
        )


%%writefile src/capstone/tools/activity_planner.py
"""Type-safe activity planning tool."""

from ..models.activity import ActivityGoal, ActivityPlan, ActivityRequest
from ..models.results import ToolError, ToolResult, ToolSuccess

# Activity template database (type-safe)
ACTIVITY_TEMPLATES: dict[ActivityGoal, ActivityPlan] = {
    ActivityGoal.EXECUTIVE_FUNCTION: ActivityPlan(
        name="Morning Routine Mission",
        goal=ActivityGoal.EXECUTIVE_FUNCTION,
        materials=[
            "Visual checklist with pictures",
            "Timer (5 minutes per task)",
            "Reward stickers",
            "Distraction blocker sign",
        ],
        environmental_setup=(
            "Bathroom: Remove extra toys, toothbrush in clear view. "
            "Bedroom: Lay out clothes night before. Minimize choices."
        ),
        duration_minutes=15,
        structure=[
            "Brush teeth: 5 min (timer)",
            "Get dressed: 5 min (timer)",
            "Check-in: 5 min (review together)",
        ],
        success_criteria=[
            "Completes 1 task without redirection",
            "Uses visual checklist at least once",
            "Acknowledges distraction and refocuses",
        ],
        adaptations=[
            "Low energy: Reduce to 1 task only",
            "High distraction: Parent stays nearby",
            "Resistance: Offer choice of task order",
        ],
    ),
    ActivityGoal.TRANSITION_SUPPORT: ActivityPlan(
        name="Transition Timer Activity",
        goal=ActivityGoal.TRANSITION_SUPPORT,
        materials=[
            "Visual timer",
            "Transition warning cards",
            "Next activity picture",
        ],
        environmental_setup="Quiet space for transition preparation, minimal distractions",
        duration_minutes=10,
        structure=[
            "5-min warning with visual",
            "2-min warning with verbal cue",
            "Transition to next activity",
        ],
        success_criteria=[
            "Acknowledges warnings",
            "Transitions within 2 minutes of timer",
            "No major resistance",
        ],
        adaptations=[
            "Extra time: Add 10-min warning",
            "High resistance: Break into smaller steps",
        ],
    ),
    ActivityGoal.HOMEWORK_SUPPORT: ActivityPlan(
        name="Homework Break Down Strategy",
        goal=ActivityGoal.HOMEWORK_SUPPORT,
        materials=[
            "Homework divided into segments",
            "Timer for each segment",
            "Break activity (fidget, stretch)",
            "Reward system",
        ],
        environmental_setup=(
            "Quiet corner, minimal visual distractions, all materials ready before starting"
        ),
        duration_minutes=30,
        structure=[
            "Segment 1: 10 min work + 5 min break",
            "Segment 2: 10 min work + 5 min break",
        ],
        success_criteria=[
            "Completes 1 segment",
            "Takes break when timer rings",
            "Returns to work after break",
        ],
        adaptations=[
            "Overwhelm: Reduce to 1 segment",
            "High energy: Add movement breaks",
            "Low motivation: Increase reward frequency",
        ],
    ),
    ActivityGoal.ENGAGEMENT: ActivityPlan(
        name="Two-Person Board Game Practice",
        goal=ActivityGoal.ENGAGEMENT,
        materials=[
            "Simple board game (Uno, Connect 4, checkers)",
            "Visual timer",
            "Feelings chart",
        ],
        environmental_setup="Quiet corner, 2 chairs facing each other, table for game",
        duration_minutes=20,
        structure=[
            "Setup: 5 min",
            "Play: 10 min",
            "Reflection: 5 min",
        ],
        success_criteria=[
            "Takes 2-3 turns appropriately",
            "Uses 1-2 social phrases",
            "Expresses 1 feeling during game",
        ],
        adaptations=[
            "Frustration: Pause and use calming strategy",
            "Low engagement: Switch to more active game",
        ],
    ),
}


def get_activity_plan(request: ActivityRequest) -> ToolResult:
    """
    Get structured activity plan based on goal.

    Args:
        request: Validated activity request with goal and constraints

    Returns:
        ToolResult: Success with ActivityPlan or Error if goal not found
    """
    try:
        if request.goal not in ACTIVITY_TEMPLATES:
            return ToolError(
                error_code="INVALID_GOAL",
                error_message=f"Activity goal {request.goal.value} not supported",
            )

        plan = ACTIVITY_TEMPLATES[request.goal]

        # Adjust duration if requested differs from template
        if request.duration_minutes != plan.duration_minutes:
            # Create adjusted plan (immutable, so create new instance)
            plan = ActivityPlan(
                name=plan.name,
                goal=plan.goal,
                materials=plan.materials,
                environmental_setup=plan.environmental_setup,
                duration_minutes=request.duration_minutes,
                structure=plan.structure,
                success_criteria=plan.success_criteria,
                adaptations=plan.adaptations,
            )

        return ToolSuccess(
            data={
                "name": plan.name,
                "goal": plan.goal.value,
                "materials": plan.materials,
                "setup": plan.environmental_setup,
                "duration": plan.duration_minutes,
                "structure": plan.structure,
                "success_criteria": plan.success_criteria,
                "adaptations": plan.adaptations,
            }
        )

    except Exception as e:
        return ToolError(
            error_code="PLANNING_ERROR",
            error_message=f"Failed to generate activity plan: {str(e)}",
        )


%%writefile src/capstone/tools/pattern_analyzer.py
"""Type-safe pattern analysis tool."""

from ..models.memory import BehaviorPattern, SessionOutcome
from ..models.results import ToolError, ToolResult, ToolSuccess


def analyze_patterns(
    session_history: list[SessionOutcome],
    behavior_type: str,
) -> ToolResult:
    """
    Analyze behavioral patterns from session history.

    Args:
        session_history: List of validated session outcomes
        behavior_type: Type of behavior to analyze (e.g., "bedtime", "homework")

    Returns:
        ToolResult: Success with BehaviorPattern or Error if insufficient data
    """
    try:
        if len(session_history) < 1:
            return ToolError(
                error_code="INSUFFICIENT_DATA",
                error_message="Need at least 1 session for pattern analysis",
            )

        # Filter relevant sessions - look for behavior type keywords
        behavior_keywords = [behavior_type.lower()]
        # Add common variations for known behavior types
        if behavior_type.lower() == "bedtime":
            behavior_keywords.extend(["bed", "sleep", "night"])
        elif behavior_type.lower() == "homework":
            behavior_keywords.extend(["work", "study", "assignment"])
        elif behavior_type.lower() == "morning":
            behavior_keywords.extend(["wake", "routine", "breakfast"])

        relevant_sessions = [
            s
            for s in session_history
            if any(
                kw in s.strategy_used.lower() or kw in s.notes.lower()
                for kw in behavior_keywords
            )
        ]

        # If no sessions match keywords, analyze all sessions
        if len(relevant_sessions) == 0:
            relevant_sessions = session_history

        # Analyze successful strategies
        successful: list[str] = []
        unsuccessful: list[str] = []

        for session in relevant_sessions:
            if session.worked:
                successful.append(session.strategy_used)
            else:
                unsuccessful.append(session.strategy_used)

        # Extract common triggers from notes (simplified pattern matching)
        triggers: list[str] = []
        for session in relevant_sessions:
            notes_lower = session.notes.lower()
            if "screen time" in notes_lower:
                triggers.append("Screen time within 1 hour")
            if "skip" in notes_lower or "missed" in notes_lower:
                triggers.append("Skipped routine step")
            if "change" in notes_lower or "unexpected" in notes_lower:
                triggers.append("Schedule disruption earlier in day")

        # Count frequency (number of successful outcomes)
        frequency = sum(1 for s in relevant_sessions if s.worked)
        total_sessions = len(relevant_sessions)

        # Ensure frequency doesn't exceed total sessions (validation requirement)
        if frequency > total_sessions:
            frequency = total_sessions

        pattern = BehaviorPattern(
            behavior_type=behavior_type,
            common_triggers=list(set(triggers)),
            successful_strategies=list(set(successful)),
            unsuccessful_approaches=list(set(unsuccessful)),
            frequency_count=frequency,
            sessions_analyzed=total_sessions,
        )

        return ToolSuccess(
            data={
                "behavior_type": pattern.behavior_type,
                "common_triggers": pattern.common_triggers,
                "successful_strategies": pattern.successful_strategies,
                "unsuccessful_approaches": pattern.unsuccessful_approaches,
                "frequency": pattern.frequency_count,
                "sessions": pattern.sessions_analyzed,
            }
        )

    except Exception as e:
        return ToolError(
            error_code="ANALYSIS_ERROR",
            error_message=f"Failed to analyze patterns: {str(e)}",
        )



%%writefile src/capstone/agents/__init__.py
"""ADK specialist agents for neurodivergent parenting support."""

from .coordinator import create_coordinator
from .specialist_factory import (
    create_activity_planner_agent,
    create_adhd_expert,
    create_agent_by_type,
    create_asd_expert,
    create_developmental_expert,
    create_memory_agent,
    create_specialist_agent,
)

__all__ = [
    # Main coordinator
    "create_coordinator",
    # Specialist agents (backward compatible)
    "create_adhd_expert",
    "create_asd_expert",
    "create_developmental_expert",
    "create_memory_agent",
    "create_activity_planner_agent",
    # New factory functions
    "create_specialist_agent",
    "create_agent_by_type",
]


%%writefile src/capstone/agents/retry_config.py
"""Shared retry configuration for all agents following ADK best practices."""

from google.genai.types import GenerateContentConfig, HttpOptions, HttpRetryOptions


def get_retry_config() -> GenerateContentConfig:
    """Get standard retry configuration for handling 429 rate limit errors.

    Follows ADK best practices from:
    https://google.github.io/adk-docs/agents/models/#error-code-429-resource_exhausted

    Returns:
        GenerateContentConfig with retry options configured for 429 errors
    """
    return GenerateContentConfig(
        http_options=HttpOptions(
            retry_options=HttpRetryOptions(
                attempts=3,  # Maximum 3 retry attempts
                initial_delay=10.0,  # Start with 10 second delay
                max_delay=60.0,  # Cap at 60 seconds
                exp_base=2.0,  # Exponential backoff with base 2
                jitter=0.2,  # 20% jitter to prevent thundering herd
                http_status_codes=[429],  # Only retry on 429 RESOURCE_EXHAUSTED errors
            )
        )
    )


%%writefile src/capstone/agents/agent_configs.py
"""Agent configuration system - consolidates all specialist agent configs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    """Configuration for a specialist agent."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    model: str = Field(default="gemini-2.0-flash-exp", min_length=1)

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        strict=True,
        extra="forbid",
    )


# =============================================================================
# System Prompts
# =============================================================================

ADHD_EXPERT_PROMPT = """You are an ADHD specialist supporting parents of an 8-year-old child with ADHD.

**Your Expertise:**
- Executive function challenges (working memory, task initiation, organization)
- Attention regulation: sustained attention, selective attention, divided attention
- Impulse control and emotional dysregulation
- Hyperactivity and sensory-seeking behaviors
- Time blindness and planning difficulties

**Your Approach:**
1. **Identify ADHD-Specific Factors**: Analyze behavioral patterns through ADHD lens
2. **Draw on Evidence-Based Knowledge**: Reference best practices from:
   - CHADD (Children and Adults with ADHD)
   - Russell Barkley's research on executive function
   - ADDitude Magazine resources
   - Evidence-based behavioral interventions
3. **Provide Actionable Strategies**: Offer specific, implementable techniques that address:
   - Breaking tasks into smaller steps (chunking)
   - Visual supports and timers
   - Movement breaks and fidget tools
   - Positive reinforcement systems
   - Environmental modifications to reduce distractions

**Important Guidelines:**
- Distinguish between ADHD-driven behaviors and age-typical behaviors
- Acknowledge when a behavior may have multiple contributing factors
- Always include rationale for why a strategy works for ADHD brains
- Emphasize understanding over punishment
- You are a support tool, not a medical professional

**Response Format:**
1. Acknowledge the parent's concern with empathy
2. Identify ADHD-specific factors in the behavior
3. Provide 2-3 evidence-based strategies with clear rationale
4. Include success indicators for each strategy
"""

ASD_EXPERT_PROMPT = """You are an ASD (Autism Spectrum Disorder) specialist supporting parents of an 8-year-old child with ASD Level 1.

**Your Expertise:**
- Sensory processing differences: hyper-sensitivity and hypo-sensitivity across sensory modalities
- Communication patterns: literal thinking, processing time, social communication
- Need for predictability: routine disruptions, transitions, unexpected changes
- Special interests and focused attention patterns
- Information processing: visual thinking, pattern recognition, detail focus

**Your Approach:**
1. **Identify ASD-Specific Factors**: Analyze behavioral patterns through autism lens
2. **Draw on Evidence-Based Knowledge**: Reference best practices from:
   - Autism Speaks and Autism Self-Advocacy Network
   - Temple Grandin's work on sensory processing
   - CDC developmental milestones and autism resources
   - Social Stories and visual support strategies
3. **Provide Autism-Affirming Strategies**: Offer specific techniques that:
   - Respect sensory needs and preferences
   - Provide predictability through visual schedules
   - Support communication with processing time
   - Leverage strengths and special interests
   - Reduce sensory overwhelm in the environment

**Important Guidelines:**
- Distinguish between ASD-driven behaviors and co-occurring ADHD or age-typical behaviors
- Use neurodiversity-affirming language (avoid "fixing" language)
- Recognize behaviors as communication and sensory regulation
- Emphasize environmental modifications over behavior change
- You are a support tool, not a medical professional

**Response Format:**
1. Acknowledge the parent's concern with empathy
2. Identify ASD-specific factors in the behavior
3. Provide 2-3 evidence-based strategies with clear rationale
4. Include success indicators for each strategy
5. Suggest sensory or environmental modifications when relevant
"""

DEVELOPMENTAL_EXPERT_PROMPT = """You are a developmental specialist supporting parents of an 8-year-old child.

**Your Expertise:**
- Age-appropriate expectations for 8-year-olds (middle childhood/school-age)
- Developmental milestones across domains:
  * Cognitive: concrete operational thinking, emerging logical reasoning
  * Social: peer relationships, cooperation, understanding social rules
  * Emotional: emotional regulation development, self-awareness, empathy
  * Physical: fine and gross motor coordination, energy levels
- Individual developmental variations and ranges
- Distinguishing neurodivergent from age-typical behaviors

**Your Approach:**
1. **Identify Age-Typical Factors**: Recognize behaviors that are normal for 8-year-olds
2. **Draw on Developmental Knowledge**: Reference developmental norms from:
   - CDC developmental milestones
   - American Academy of Pediatrics guidelines
   - Child development research (Piaget, Erikson frameworks)
3. **Provide Developmental Context**: Help parents understand:
   - What's typical for this age
   - When neurodivergence intersects with typical development
   - Realistic expectations for an 8-year-old
   - How ADHD/ASD may amplify typical challenges

**Important Guidelines:**
- Normalize age-typical behaviors (testing boundaries, homework resistance, etc.)
- Acknowledge how neurodivergence can intensify typical developmental challenges
- Provide realistic expectations to reduce parent stress
- Distinguish between "won't" (choice) and "can't" (developmental/neurological)
- You are a support tool, not a medical professional

**Response Format:**
1. Acknowledge the parent's concern with empathy
2. Identify what's developmentally typical for an 8-year-old
3. Explain how ADHD/ASD may interact with typical development
4. Provide 1-2 strategies that honor both neurodivergence and developmental stage
5. Reassure parents about what's normal
"""

MEMORY_AGENT_PROMPT = """You are a memory and pattern learning specialist helping parents learn what works for their specific child.

**Your Expertise:**
- Analyzing session history to identify patterns
- Recognizing what strategies consistently work or don't work
- Identifying environmental triggers and conditions
- Personalizing recommendations based on family's unique experiences
- Tracking progress and celebrating improvements

**Your Approach:**
1. **Analyze Session History**: Examine past outcomes for patterns
2. **Identify Patterns**: Look for:
   - Strategies with consistent success
   - Common triggers that precede challenges
   - Time-of-day patterns
   - Environmental factors (screen time, schedule changes, etc.)
3. **Provide Personalized Insights**: Offer recommendations based on:
   - What has worked before for this family
   - What hasn't worked (to avoid repeating)
   - Patterns unique to this child
4. **Suggest Experiments**: Propose trying variations on successful strategies

**Important Guidelines:**
- Always base insights on the actual session data provided
- Acknowledge when there's insufficient data for strong conclusions
- Celebrate successes and recognize parent effort
- Frame "unsuccessful" strategies as learning opportunities
- Consider sample size (3 sessions vs. 10+ sessions)
- You are a support tool, not a medical professional

**Response Format:**
1. Summarize the pattern analysis results
2. Highlight key insights (successful strategies, triggers to avoid)
3. Provide 1-2 personalized recommendations based on the patterns
4. Suggest next steps for continued learning

**Session History Format:**
Session history should be provided as JSON:
[
  {
    "strategy_used": "Visual timer for bedtime",
    "worked": true,
    "notes": "Used 10-min visual timer, child responded well"
  },
  {
    "strategy_used": "Verbal reminders only for bedtime",
    "worked": false,
    "notes": "Got frustrated, escalated to meltdown"
  }
]
"""

ACTIVITY_PLANNER_PROMPT = """You are an activity planning specialist helping parents create structured activities for their 8-year-old child with ADHD and ASD Level 1.

**Your Expertise:**
- Creating clear, step-by-step activity structures
- Specifying all needed materials and environmental setup
- Setting realistic timeframes (5-120 minutes)
- Defining observable success criteria
- Adapting activities for executive function challenges and sensory needs

**Activity Goals You Support:**
1. **Executive Function**: Building planning, organization, working memory skills
2. **Transition Support**: Helping with difficult transitions between activities
3. **Homework Support**: Breaking down and structuring homework time
4. **Engagement**: Building skills through play and special interests

**Your Approach:**
1. **Understand the Goal**: Clarify what the parent wants to achieve
2. **Gather Context**: Ask about available materials, duration, and any past successes
3. **Create Structured Plan**: Design activities with:
   - Complete materials list
   - Environmental setup instructions
   - Step-by-step structure with timing
   - Observable success criteria
   - Adaptations for ADHD/ASD needs
4. **Incorporate Memory**: Use insights from past successful activities when provided

**Important Guidelines:**
- Be specific and concrete (not vague like "be prepared")
- Include visual supports and timers when appropriate
- Build in movement breaks for ADHD needs
- Consider sensory environment for ASD needs
- Set achievable success criteria (not perfection)
- You are a support tool, not a medical professional

**Response Format:**
1. Acknowledge the parent's activity goal
2. Create structured activity plan
3. Explain the rationale for key elements
4. Suggest how to introduce the activity to the child
5. Offer tips for troubleshooting common challenges
"""

COORDINATOR_SYSTEM_PROMPT = """You are a Parenting Coordinator supporting parents of an 8-year-old child with ADHD and ASD Level 1.

**Your Role:**
You orchestrate a team of 5 specialist agents to provide comprehensive, personalized support. You are the parent's primary interface - warm, empathetic, and practical.

**Your Team of Specialists:**
1. **ADHD Expert**: Executive function, attention, impulse control, hyperactivity
2. **ASD Expert**: Sensory processing, communication, routine needs, special interests
3. **Developmental Expert**: Age-typical behaviors, developmental milestones, realistic expectations
4. **Memory Agent**: Pattern learning, what works for THIS child, personalized insights
5. **Activity Planner**: Structured activity plans with materials, setup, and success criteria

**How to Use Your Team:**

**For Behavioral Challenges:**
1. **Always start with behavior analysis** - Consult ADHD Expert and/or ASD Expert to identify factors
2. **Add developmental context** - Ask Developmental Expert if behavior is age-typical
3. **Check past patterns** - If parent mentions history, ask Memory Agent about patterns
4. **Synthesize insights** - Combine perspectives to provide comprehensive understanding

**For Strategy Requests:**
1. **Understand the context** - Gather details about the specific situation
2. **Consult relevant specialists**:
   - ADHD Expert for executive function strategies
   - ASD Expert for sensory/communication strategies
   - Memory Agent for personalized recommendations based on past successes
3. **Get structured plans when needed** - Use Activity Planner for detailed activity structures

**For Proactive Planning:**
1. **Clarify the goal** - What does the parent want to achieve?
2. **Use Activity Planner** - Generate structured plan with materials, setup, timing
3. **Add specialist insights** - Get ADHD/ASD adaptations from relevant experts
4. **Incorporate memory** - Check Memory Agent for past successful approaches

**Important Coordination Guidelines:**
- **Be strategic about specialist calls**: Don't call all agents for simple questions
- **Synthesize, don't just pass along**: Combine insights into cohesive guidance
- **Maintain warm tone**: You are supportive, not clinical
- **Provide clear action steps**: Parents need implementable strategies
- **Acknowledge complexity**: ADHD + ASD means overlapping factors
- **Celebrate successes**: Recognize parent effort and child progress
- **Always include disclaimer**: "I'm a parenting support tool, not medical advice."

**Response Structure:**
1. **Acknowledge** the parent's concern with empathy
2. **Analyze** the situation (consult specialists as needed)
3. **Explain** what's happening (ADHD factors, ASD factors, age-typical factors)
4. **Recommend** 2-3 specific, actionable strategies with rationale
5. **Support** with encouragement and realistic expectations
6. **Disclaim** your role as a support tool, not medical professional

**Example Routing Logic:**
- "My son won't do homework" â†’ ADHD Expert (executive function) + Developmental Expert (homework resistance at age 8) + Memory Agent (past homework strategies)
- "He melts down during transitions" â†’ ASD Expert (routine needs) + ADHD Expert (attention shifts) + Activity Planner (transition support plan)
- "What worked before?" â†’ Memory Agent (pattern analysis)
- "I need a bedtime activity" â†’ Activity Planner + Memory Agent (past bedtime successes)
- "Is this normal for his age?" â†’ Developmental Expert

**Tone:** Empathetic, practical, evidence-informed, supportive, non-judgmental
"""

# =============================================================================
# Agent Configurations
# =============================================================================

AgentType = Literal[
    "adhd_expert",
    "asd_expert",
    "developmental_expert",
    "memory_agent",
    "activity_planner",
]

AGENT_CONFIGS: dict[AgentType, AgentConfig] = {
    "adhd_expert": AgentConfig(
        name="adhd_expert",
        description="Specialist in ADHD-related behaviors, executive function challenges, and evidence-based interventions for 8-year-old children",
        system_prompt=ADHD_EXPERT_PROMPT,
    ),
    "asd_expert": AgentConfig(
        name="asd_expert",
        description="Specialist in autism-related behaviors, sensory processing, communication patterns, and autism-affirming strategies for Level 1 ASD",
        system_prompt=ASD_EXPERT_PROMPT,
    ),
    "developmental_expert": AgentConfig(
        name="developmental_expert",
        description="Specialist in age-appropriate expectations, developmental milestones, and typical 8-year-old behaviors",
        system_prompt=DEVELOPMENTAL_EXPERT_PROMPT,
    ),
    "memory_agent": AgentConfig(
        name="memory_agent",
        description="Specialist in pattern learning, analyzing session history, and personalizing recommendations based on past experiences",
        system_prompt=MEMORY_AGENT_PROMPT,
    ),
    "activity_planner": AgentConfig(
        name="activity_planner",
        description="Specialist in creating structured activity plans with materials, setup, timing, and success criteria",
        system_prompt=ACTIVITY_PLANNER_PROMPT,
    ),
}


%%writefile src/capstone/agents/tool_formatters.py
"""Output formatters for tool results - type-safe formatting functions."""


def format_behavior_analysis(data: dict[str, str | int | float | bool | list[str]]) -> str:
    """
    Format behavior analysis result into human-readable string.

    Args:
        data: Tool result data containing primary_driver and factor lists.

    Returns:
        Formatted string describing the behavior analysis.
    """
    factors = []

    adhd_factors = data.get("adhd_factors")
    if adhd_factors and isinstance(adhd_factors, list):
        factors.append(f"ADHD: {', '.join(str(f) for f in adhd_factors)}")

    asd_factors = data.get("asd_factors")
    if asd_factors and isinstance(asd_factors, list):
        factors.append(f"ASD: {', '.join(str(f) for f in asd_factors)}")

    age_factors = data.get("age_typical_factors")
    if age_factors and isinstance(age_factors, list):
        factors.append(f"Age-typical: {', '.join(str(f) for f in age_factors)}")

    return f"Primary driver: {data['primary_driver']}\nFactors:\n" + "\n".join(factors)


def format_activity_plan(data: dict[str, str | int | float | bool | list[str]]) -> str:
    """
    Format activity plan result into human-readable string.

    Args:
        data: Tool result data containing activity plan details.

    Returns:
        Formatted string describing the activity plan.
    """
    output = [
        f"Activity: {data['name']}",
        f"Goal: {data['goal']}",
        f"Duration: {data['duration_minutes']} minutes",
    ]

    materials = data.get("materials")
    if materials and isinstance(materials, list):
        output.append(f"\nMaterials needed: {', '.join(str(m) for m in materials)}")

    output.append(f"\nEnvironmental setup: {data['environmental_setup']}")
    output.append("\nStructure:")

    structure = data.get("structure")
    if structure and isinstance(structure, list):
        for step in structure:
            output.append(f"  - {step}")

    output.append("\nSuccess criteria:")
    criteria = data.get("success_criteria")
    if criteria and isinstance(criteria, list):
        for criterion in criteria:
            output.append(f"  - {criterion}")

    adaptations = data.get("adaptations")
    if adaptations and isinstance(adaptations, list):
        output.append("\nAdaptations:")
        for adaptation in adaptations:
            output.append(f"  - {adaptation}")

    return "\n".join(output)


def format_pattern_analysis(data: dict[str, str | int | float | bool | list[str]]) -> str:
    """
    Format pattern analysis result into human-readable string.

    Args:
        data: Tool result data containing pattern analysis details.

    Returns:
        Formatted string describing the pattern analysis.
    """
    output = [
        f"Behavior: {data['behavior_type']}",
        f"Sessions analyzed: {data['sessions']}",
        f"Success rate: {data['frequency']}/{data['sessions']}",
    ]

    triggers = data.get("common_triggers")
    if triggers and isinstance(triggers, list):
        output.append(f"\nCommon triggers: {', '.join(str(t) for t in triggers)}")

    strategies = data.get("successful_strategies")
    if strategies and isinstance(strategies, list):
        output.append(f"\nSuccessful strategies: {', '.join(str(s) for s in strategies)}")

    unsuccessful = data.get("unsuccessful_approaches")
    if unsuccessful and isinstance(unsuccessful, list):
        output.append(f"\nUnsuccessful approaches: {', '.join(str(a) for a in unsuccessful)}")

    return "\n".join(output)


%%writefile src/capstone/agents/specialist_helpers.py
"""Helper functions for specialist agent consultation."""

from __future__ import annotations

from collections.abc import Callable

from google import genai
from google.adk.agents import LlmAgent


def consult_specialist(
        agent_getter: Callable[[], LlmAgent],
        question: str,
        specialist_name: str,
) -> str:
    """
    Generic specialist consultation with error handling.

    This function uses the genai.Client directly to consult specialist agents,
    passing the agent's instruction as the system prompt.

    Args:
        agent_getter: Callable that returns the specialist agent instance.
        question: Question to ask the specialist.
        specialist_name: Human-readable name for error messages.

    Returns:
        The specialist's response text, or an error message.
    """
    try:
        agent = agent_getter()

        # Use genai.Client directly with agent's instruction as system prompt
        client = genai.Client()
        response = client.models.generate_content(
            model=agent.model,
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=agent.instruction,
            ),
        )

        if response.text:
            return response.text
        return f"No response from {specialist_name}"
    except Exception as e:
        return f"Error consulting {specialist_name}: {str(e)}"



%%writefile src/capstone/agents/tool_wrappers.py
"""Tool wrapper functions for coordinator agent."""

import json
from collections.abc import Callable

from ..models import (
    ActivityGoal,
    ActivityRequest,
    ActivityType,
    BehaviorInput,
    SessionOutcome,
    TimeOfDay,
)
from ..models.results import ToolError, ToolSuccess
from ..tools.activity_planner import get_activity_plan
from ..tools.behavior_classifier import classify_behavior
from ..tools.pattern_analyzer import analyze_patterns
from .tool_formatters import (
    format_activity_plan,
    format_behavior_analysis,
    format_pattern_analysis,
)


def create_analyze_behavior_wrapper() -> Callable[..., str]:
    """Create wrapper for behavior classifier tool."""

    def _analyze_behavior_wrapper(
            description: str,
            time_of_day: TimeOfDay,
            activity_type: ActivityType,
            context: str,
    ) -> str:
        """Analyze behavior using the classifier tool."""
        behavior_input = BehaviorInput(
            description=description,
            time_of_day=time_of_day,
            activity_type=activity_type,
            context=context,
        )
        result = classify_behavior(behavior_input)

        if isinstance(result, ToolError):
            return f"Error analyzing behavior: {result.error_message}"

        # Type narrowing: result is ToolSuccess
        assert isinstance(result, ToolSuccess)
        return format_behavior_analysis(result.data)

    return _analyze_behavior_wrapper


def create_plan_activity_wrapper() -> Callable[..., str]:
    """Create wrapper for activity planner tool."""

    def _plan_activity_wrapper(
            goal: str,
            duration_minutes: int,
            materials: str,
    ) -> str:
        """Plan an activity using the planner tool."""
        try:
            # Parse materials from comma-separated string
            materials_list = [m.strip() for m in materials.split(",") if m.strip()]

            # Parse goal string to enum
            goal_enum = ActivityGoal(goal)

            # Create activity request
            request = ActivityRequest(
                goal=goal_enum,
                duration_minutes=duration_minutes,
                available_materials=materials_list,
            )

            # Plan activity
            result = get_activity_plan(request)

            if isinstance(result, ToolError):
                return f"Error planning activity: {result.error_message}"

            # Type narrowing: result is ToolSuccess
            assert isinstance(result, ToolSuccess)
            return format_activity_plan(result.data)

        except Exception as e:
            return f"Error: {str(e)}"

    return _plan_activity_wrapper


def create_analyze_patterns_wrapper() -> Callable[..., str]:
    """Create wrapper for pattern analyzer tool."""

    def _analyze_patterns_wrapper(
            session_history_json: str,
            behavior_type: str,
    ) -> str:
        """Analyze patterns from session history to find what works.

        Args:
            session_history_json: JSON array of sessions. Each session must have:
                - "strategy_used": string describing the strategy (e.g., "Visual timer 10 min")
                - "worked": boolean true/false
                - "notes": string with outcome details
                Example: [{"strategy_used": "Visual timer", "worked": true, "notes": "went to bed calmly"}]
            behavior_type: The behavior being analyzed (e.g., "bedtime", "homework")

        Returns:
            Analysis of successful vs unsuccessful strategies and patterns.
        """
        try:
            # Parse JSON session history
            session_data = json.loads(session_history_json)
            session_outcomes = []

            for session in session_data:
                outcome = SessionOutcome(
                    strategy_used=session["strategy_used"],
                    worked=session["worked"],
                    notes=session["notes"],
                )
                session_outcomes.append(outcome)

            # Analyze patterns
            result = analyze_patterns(session_outcomes, behavior_type)

            if isinstance(result, ToolError):
                return f"Error analyzing patterns: {result.error_message}"

            # Type narrowing: result is ToolSuccess
            assert isinstance(result, ToolSuccess)
            return format_pattern_analysis(result.data)

        except json.JSONDecodeError as e:
            return f"Error parsing session history JSON: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    return _analyze_patterns_wrapper



%%writefile src/capstone/agents/specialist_factory.py
"""Specialist agent factory - single factory function for all specialist agents."""

from collections.abc import Callable

from google.adk.agents import LlmAgent

from .agent_configs import AGENT_CONFIGS, AgentConfig, AgentType
from .retry_config import get_retry_config


def create_specialist_agent(
        config: AgentConfig,
        tools: list[Callable] | None = None,
) -> LlmAgent:
    """
    Create a specialist agent from configuration.

    This single factory function replaces 5 separate create_*() functions,
    reducing code duplication while maintaining type safety.

    Args:
        config: Agent configuration with name, description, and system prompt.
        tools: Optional list of tools for the agent. Defaults to empty list.

    Returns:
        Configured LlmAgent instance.
    """
    return LlmAgent(
        model=config.model,
        instruction=config.system_prompt,
        tools=tools or [],
        name=config.name,
        description=config.description,
        generate_content_config=get_retry_config(),
    )


def create_agent_by_type(
        agent_type: AgentType,
        tools: list[Callable] | None = None,
) -> LlmAgent:
    """
    Create a specialist agent by type name.

    Args:
        agent_type: Type of agent to create (e.g., "adhd_expert").
        tools: Optional list of tools for the agent.

    Returns:
        Configured LlmAgent instance.

    Raises:
        KeyError: If agent_type is not a valid agent type.
    """
    config = AGENT_CONFIGS[agent_type]
    return create_specialist_agent(config, tools)


# =============================================================================
# Convenience functions for backward compatibility
# =============================================================================


def create_adhd_expert() -> LlmAgent:
    """Create ADHD specialist agent."""
    return create_agent_by_type("adhd_expert")


def create_asd_expert() -> LlmAgent:
    """Create ASD specialist agent."""
    return create_agent_by_type("asd_expert")


def create_developmental_expert() -> LlmAgent:
    """Create developmental specialist agent."""
    return create_agent_by_type("developmental_expert")


def create_memory_agent(tools: list[Callable] | None = None) -> LlmAgent:
    """Create memory and pattern learning agent."""
    return create_agent_by_type("memory_agent", tools)


def create_activity_planner_agent(tools: list[Callable] | None = None) -> LlmAgent:
    """Create activity planning agent."""
    return create_agent_by_type("activity_planner", tools)


%%writefile src/capstone/agents/coordinator.py
"""Parenting coordinator agent that orchestrates specialist agents."""

from google.adk.agents import LlmAgent

from ..infrastructure.agent_factory import AgentFactory
from .agent_configs import COORDINATOR_SYSTEM_PROMPT
from .retry_config import get_retry_config
from .specialist_helpers import consult_specialist
from .tool_wrappers import (
    create_analyze_behavior_wrapper,
    create_analyze_patterns_wrapper,
    create_plan_activity_wrapper,
)


def create_coordinator(agent_factory: AgentFactory | None = None) -> LlmAgent:
    """
    Create parenting coordinator agent with dependency injection.

    This agent orchestrates 5 specialist agents:
    1. ADHD Expert - Executive function and attention challenges
    2. ASD Expert - Sensory processing and communication patterns
    3. Developmental Expert - Age-appropriate expectations
    4. Memory Agent - Pattern learning from past experiences
    5. Activity Planner - Structured activity recommendations

    The coordinator:
    - Routes questions to appropriate specialists
    - Synthesizes insights from multiple experts
    - Provides comprehensive, personalized support
    - Maintains empathetic, supportive communication

    Args:
        agent_factory: Optional AgentFactory instance for dependency injection.
                      If None, creates a new factory instance.

    Returns:
        Configured LlmAgent coordinator instance.
    """
    # Use injected factory or create new one (dependency injection)
    factory = agent_factory or AgentFactory()

    # Create specialist consultation functions using the factory
    def _consult_adhd_expert(question: str) -> str:
        """Consult ADHD specialist agent."""
        return consult_specialist(
            agent_getter=factory.get_adhd_expert,
            question=question,
            specialist_name="ADHD expert",
        )

    def _consult_asd_expert(question: str) -> str:
        """Consult ASD specialist agent."""
        return consult_specialist(
            agent_getter=factory.get_asd_expert,
            question=question,
            specialist_name="ASD expert",
        )

    def _consult_developmental_expert(question: str) -> str:
        """Consult developmental specialist agent."""
        return consult_specialist(
            agent_getter=factory.get_developmental_expert,
            question=question,
            specialist_name="developmental expert",
        )

    def _consult_memory_agent(question: str) -> str:
        """Consult memory and pattern learning agent."""
        return consult_specialist(
            agent_getter=lambda: factory.get_memory_agent(tools=[]),
            question=question,
            specialist_name="memory agent",
        )

    def _consult_activity_planner(question: str) -> str:
        """Consult activity planning specialist agent."""
        return consult_specialist(
            agent_getter=lambda: factory.get_activity_planner_agent(tools=[]),
            question=question,
            specialist_name="activity planner",
        )

    # Create tool wrappers
    _analyze_behavior_wrapper = create_analyze_behavior_wrapper()
    _plan_activity_wrapper = create_plan_activity_wrapper()
    _analyze_patterns_wrapper = create_analyze_patterns_wrapper()

    # Combine all tools - pass raw Python callables
    all_tools = [
        # Specialist agents as raw Python functions
        _consult_adhd_expert,
        _consult_asd_expert,
        _consult_developmental_expert,
        _consult_memory_agent,
        _consult_activity_planner,
        # Direct capability functions
        _analyze_behavior_wrapper,
        _plan_activity_wrapper,
        _analyze_patterns_wrapper,
    ]

    # Create coordinator agent with retry config (ADK best practice for 429 errors)
    coordinator = LlmAgent(
        model="gemini-2.0-flash-exp",
        instruction=COORDINATOR_SYSTEM_PROMPT,
        tools=all_tools,
        name="parenting_coordinator",
        description="Orchestrates specialist agents to provide comprehensive parenting support for neurodivergent children",
        generate_content_config=get_retry_config(),
    )

    return coordinator


%%writefile src/capstone/infrastructure/__init__.py
"""Infrastructure layer for external integrations."""


%%writefile src/capstone/infrastructure/agent_factory.py
"""Agent factory for creating and caching specialist agents."""

from collections.abc import Callable

from google.adk.agents import LlmAgent

from ..agents.agent_configs import AGENT_CONFIGS
from ..agents.specialist_factory import create_specialist_agent


class AgentFactory:
    """Factory for creating and caching specialist agents.

    Provides thread-safe agent creation with caching to avoid
    recreating agents on every request. Uses the centralized
    agent configuration system for consistency.

    Example:
        factory = AgentFactory()
        adhd_agent = factory.get_adhd_expert()  # Creates and caches
        adhd_agent2 = factory.get_adhd_expert()  # Returns cached instance
        assert adhd_agent is adhd_agent2  # Same instance
    """

    def __init__(self) -> None:
        """Initialize empty agent cache."""
        self._cache: dict[str, LlmAgent] = {}

    def get_adhd_expert(self) -> LlmAgent:
        """Get or create ADHD specialist agent."""
        if "adhd_expert" not in self._cache:
            config = AGENT_CONFIGS["adhd_expert"]
            self._cache["adhd_expert"] = create_specialist_agent(config)
        return self._cache["adhd_expert"]

    def get_asd_expert(self) -> LlmAgent:
        """Get or create ASD specialist agent."""
        if "asd_expert" not in self._cache:
            config = AGENT_CONFIGS["asd_expert"]
            self._cache["asd_expert"] = create_specialist_agent(config)
        return self._cache["asd_expert"]

    def get_developmental_expert(self) -> LlmAgent:
        """Get or create developmental specialist agent."""
        if "developmental_expert" not in self._cache:
            config = AGENT_CONFIGS["developmental_expert"]
            self._cache["developmental_expert"] = create_specialist_agent(config)
        return self._cache["developmental_expert"]

    def get_memory_agent(self, tools: list[Callable]) -> LlmAgent:
        """Get or create memory management agent with tools.

        Args:
            tools: List of memory management tools to provide to agent.

        Returns:
            Configured memory agent with provided tools.
        """
        if "memory_agent" not in self._cache:
            config = AGENT_CONFIGS["memory_agent"]
            self._cache["memory_agent"] = create_specialist_agent(config, tools)
        return self._cache["memory_agent"]

    def get_activity_planner_agent(self, tools: list[Callable]) -> LlmAgent:
        """Get or create activity planning agent with tools.

        Args:
            tools: List of activity planning tools to provide to agent.

        Returns:
            Configured activity planning agent with provided tools.
        """
        if "activity_planner_agent" not in self._cache:
            config = AGENT_CONFIGS["activity_planner"]
            self._cache["activity_planner_agent"] = create_specialist_agent(config, tools)
        return self._cache["activity_planner_agent"]

    def clear_cache(self) -> None:
        """Clear all cached agents.

        Useful for testing or when agents need to be recreated.
        """
        self._cache.clear()


%%writefile src/capstone/__init__.py
"""Neurodivergent Parenting Support Agent - ADK Capstone Project."""

__version__ = "0.1.0"


# Verify all files were created
import os


def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([f for f in files if f.endswith(".py")])
    return count


file_count = count_files("src/capstone")
print(f"Created {file_count} Python files in src/capstone/")
print("Source code setup complete!")


import os

from kaggle_secrets import UserSecretsClient

# Retrieve API key from Kaggle secrets
try:
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
    print("âœ… API key configured from Kaggle secrets")
except Exception:
    print("âš ï¸� Kaggle secrets not available. Checking environment variable...")
    if not os.getenv("GOOGLE_API_KEY"):
        print("â�Œ Error: GOOGLE_API_KEY not found!")
        print("Please add it as a Kaggle secret or set it manually:")
        print('  os.environ["GOOGLE_API_KEY"] = "your-api-key-here"')
    else:
        print("âœ… API key found in environment")


import sys
from pathlib import Path

# Add src directory to Python path
working_dir = Path.cwd()
src_dir = working_dir / "src"

if src_dir.exists():
    sys.path.insert(0, str(working_dir))
    print(f"âœ… Added {working_dir} to Python path")
    print(f"âœ… Source directory found: {src_dir}")
else:
    print(f"â�Œ Error: {src_dir} not found!")
    print("Please upload the 'src/' directory from the project.")


import asyncio
import re

from google import genai
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Import our multi-agent system
from src.capstone.agents import create_coordinator

print("âœ… All imports successful!")
print("âœ… Ready to initialize multi-agent system")


def print_separator(title: str) -> None:
    """Print formatted section separator."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def extract_retry_delay(error_message: str) -> float:
    """Extract retry delay in seconds from error message."""
    match = re.search(r"retry in (\d+\.?\d*)s", error_message)
    if match:
        return float(match.group(1))

    match = re.search(r"retryDelay.*?(\d+)s", error_message)
    if match:
        return float(match.group(1))

    return 10.0  # Default to 10 seconds


async def run_scenario_with_retry(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: genai.types.Content,
    max_retries: int = 3,
) -> None:
    """Run a scenario with automatic retry on 429 errors."""
    for attempt in range(max_retries):
        try:
            response_received = False
            final_response = ""
            event_count = 0

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message,
            ):
                event_count += 1

                # Check for agent turn events (shows which agent is working)
                if hasattr(event, "agent_turn_event"):
                    agent_event = event.agent_turn_event
                    if hasattr(agent_event, "agent_name") and agent_event.agent_name:
                        print(f"  ğŸ¤– {agent_event.agent_name} is analyzing...")

                # Check for final text response in event.content
                if hasattr(event, "content") and event.content:
                    content = event.content
                    if hasattr(content, "parts") and content.parts:
                        text_parts = []
                        for part in content.parts:
                            if hasattr(part, "text") and part.text:
                                text_parts.append(part.text)

                        if text_parts:
                            response_received = True
                            final_response = "\n".join(text_parts)

            # Print final response after all events processed
            if response_received and final_response:
                print(f"\nğŸ’¬ Response:\n{final_response}\n")
                return  # Success!

        except Exception as e:
            error_str = str(e)

            if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
                retry_delay = extract_retry_delay(error_str)

                if attempt < max_retries - 1:
                    print(
                        f"\nâ�³ Rate limit hit. Waiting {retry_delay:.1f}s before retry (attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(retry_delay)
                    print("ğŸ”„ Retrying...\n")
                else:
                    print(f"\nâ�Œ Rate limit exceeded after {max_retries} attempts.")
                    print("ğŸ’¡ Tips to resolve:")
                    print("   â€¢ Wait a few minutes before running again")
                    print("   â€¢ Check your API quota at: https://ai.dev/usage?tab=rate-limit")
                    return
            else:
                print(f"â�Œ Error: {error_str}")
                return


print("âœ… Helper functions defined")



print_separator("ğŸ¤– Neurodivergent Parenting Support Agent - ADK Demo")
print("Demonstrating multi-agent hierarchical system with:")
print("  â€¢ Parenting Coordinator (Manager)")
print("  â€¢ 5 Specialist Agents (ADHD, ASD, Developmental, Memory, Activity Planner)")
print("  â€¢ Custom tools (Behavior Classifier, Activity Planner, Pattern Analyzer)")
print("  â€¢ Google Search integration")
print(
    "\nğŸ’¡ Note: Retry handling follows ADK best practices with agent-level retries + application-level fallback."
)

# Create coordinator (which creates all specialists)
print("\nğŸ”§ Initializing multi-agent system...")
coordinator = create_coordinator()
print("âœ… All agents initialized!")

# Create runner with session service
session_service = InMemorySessionService()
runner = Runner(
    app_name="agents",
    agent=coordinator,
    session_service=session_service,
)
print("âœ… Runner configured with session management")
print("\nğŸ�¬ Ready to run demo scenarios!")


print_separator("ğŸ“‹ SCENARIO 1: Homework Refusal Analysis")

scenario_1 = """My 8-year-old son refuses to start his homework after school.
He just played video games for an hour and now says he's "too tired" for homework.
When I try to get him started, he gets frustrated and sometimes has a meltdown.
This happens most evenings around 5-6pm. Is this ADHD, autism, or just being 8?"""

print("Parent Question:")
print(f'"{scenario_1}"\n')
print("ğŸ¤– Coordinator orchestrating specialists...\n")

# Create session for scenario 1
await session_service.create_session(
    app_name="agents",
    user_id="demo_user",
    session_id="scenario_1",
)

# Run scenario with automatic retry
await run_scenario_with_retry(
    runner=runner,
    user_id="demo_user",
    session_id="scenario_1",
    message=genai.types.Content(parts=[genai.types.Part(text=scenario_1)]),
)

print("\nâœ… Scenario 1 complete!")


print_separator("ğŸ“‹ SCENARIO 2: Bedtime Routine Planning")

scenario_2 = """I need help creating a structured bedtime routine.
My son has trouble with the transition from play time to bedtime.
We have about 30 minutes, and I have a visual timer, some fidget toys, and his favorite books.
In the past, visual timers have worked well for him."""

print("Parent Question:")
print(f'"{scenario_2}"\n')
print("ğŸ¤– Coordinator orchestrating specialists...\n")

# Create session for scenario 2
await session_service.create_session(
    app_name="agents",
    user_id="demo_user",
    session_id="scenario_2",
)

# Run scenario with automatic retry
await run_scenario_with_retry(
    runner=runner,
    user_id="demo_user",
    session_id="scenario_2",
    message=genai.types.Content(parts=[genai.types.Part(text=scenario_2)]),
)

print("\nâœ… Scenario 2 complete!")


print_separator("ğŸ“‹ SCENARIO 3: Learning from Past Experiences")

scenario_3 = """Can you analyze what's been working for bedtime? Here's our history:

Session 1: Used visual timer (10 min), worked well, he went to bed without resistance
Session 2: Just verbal reminders, didn't work, he got upset and it took 45 minutes
Session 3: Visual timer + sensory break before bed, worked great, he was calm

What patterns do you see?"""

print("Parent Question:")
print(f'"{scenario_3}"\n')
print("ğŸ¤– Coordinator orchestrating specialists...\n")

# Create session for scenario 3
await session_service.create_session(
    app_name="agents",
    user_id="demo_user",
    session_id="scenario_311",
)

# Run scenario with automatic retry
await run_scenario_with_retry(
    runner=runner,
    user_id="demo_user",
    session_id="scenario_311",
    message=genai.types.Content(parts=[genai.types.Part(text=scenario_3)]),
)

print("\nâœ… Scenario 3 complete!")


from google.adk.agents import ParallelAgent
from google.adk.runners import Runner
from src.capstone.agents import create_adhd_expert, create_asd_expert, create_developmental_expert

print_separator("ğŸ“‹ SCENARIO 4: ParallelAgent - Concurrent Expert Consultation")

# Create ParallelAgent that consults all experts simultaneously
print("\nğŸ”§ Creating ParallelAgent expert panel...")
print("   - ADHD Expert (with GoogleSearch)")
print("   - ASD Expert (with GoogleSearch)")
print("   - Developmental Expert (with GoogleSearch)")

parallel_panel = ParallelAgent(
    name="parallel_expert_panel",
    description=(
        "Consults ADHD, ASD, and developmental experts in parallel "
        "to gather multiple perspectives simultaneously on child behavior."
    ),
    sub_agents=[create_adhd_expert(), create_asd_expert(), create_developmental_expert()],
)

print(f"\nâœ… ParallelAgent created: {parallel_panel.name}")
print(f"   Sub-agents: {len(parallel_panel.sub_agents)}")

# Create runner for parallel panel
parallel_runner = Runner(
    app_name="parallel_demo",
    agent=parallel_panel,
    session_service=session_service,
)

# Create session for parallel scenario
await session_service.create_session(
    app_name="parallel_demo",
    user_id="demo_user",
    session_id="scenario_4",
)

scenario_4 = """My 8-year-old gets overwhelmed during morning routines.
He struggles to get dressed, brush teeth, and eat breakfast without constant reminders.
What could be causing this and what strategies might help?"""

print("\nğŸ”„ Query (sent to all 3 experts in PARALLEL):")
print(f'"{scenario_4}"\n')
print("ğŸ¤– All three experts analyzing simultaneously...\n")

# Run the parallel query
await run_scenario_with_retry(
    runner=parallel_runner,
    user_id="demo_user",
    session_id="scenario_4",
    message=genai.types.Content(parts=[genai.types.Part(text=scenario_4)]),
)

print("\nğŸ’¡ ParallelAgent Benefit: All 3 experts responded concurrently!")


#%%
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from src.capstone.agents import create_adhd_expert, create_asd_expert, create_developmental_expert

print_separator("ğŸ“‹ SCENARIO 5: SequentialAgent - Research Pipeline")

# Create SequentialAgent for structured analysis pipeline
print("\nğŸ”§ Creating SequentialAgent research pipeline...")
print("   Stage 1: ADHD Expert (Research phase)")
print("   Stage 2: ASD Expert (Analysis phase)")
print("   Stage 3: Developmental Expert (Synthesis phase)")

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description=(
        "Sequential research pipeline: Stage 1 researches ADHD factors, "
        "Stage 2 analyzes sensory/social aspects, Stage 3 synthesizes "
        "age-appropriate recommendations. Each stage builds on the previous."
    ),
    sub_agents=[create_adhd_expert(), create_asd_expert(), create_developmental_expert()],
)

print(f"\nâœ… SequentialAgent created: {research_pipeline.name}")
print(f"   Stages: {len(research_pipeline.sub_agents)}")

# Create runner for sequential pipeline
sequential_runner = Runner(
    app_name="sequential_demo",
    agent=research_pipeline,
    session_service=session_service,
)

# Create session for sequential scenario
await session_service.create_session(
    app_name="sequential_demo",
    user_id="demo_user",
    session_id="scenario_5",
)

scenario_5 = """My child has intense meltdowns when plans change unexpectedly.
Yesterday we couldn't go to the park due to rain and he cried for an hour.
How can I help him handle unexpected changes better?"""

print("\nğŸ”„ Query (processed through 3 stages SEQUENTIALLY):")
print(f'"{scenario_5}"\n')
print("ğŸ¤– Processing: ADHD Expert â†’ ASD Expert â†’ Developmental Expert...\n")

# Run the sequential query
await run_scenario_with_retry(
    runner=sequential_runner,
    user_id="demo_user",
    session_id="scenario_5",
    message=genai.types.Content(parts=[genai.types.Part(text=scenario_5)]),
)

print("\nğŸ’¡ SequentialAgent Benefit: Each expert built on the previous analysis!")
print("   Flow: Research (ADHD) â†’ Analyze (ASD) â†’ Synthesize (Developmental)")

