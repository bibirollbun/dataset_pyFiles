# pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

print("âœ… ADK components imported successfully.")


# Reference data for CPT / ICD / HCPCS codes and prior authorization hints.
# This is a small illustrative subset for this notebook â€“
# in a real deployment, these would come from maintained code dictionaries.

VALID_CPT_CODES = {
    "99213", "99214", "99215",
    "99203", "99204",
    "83036",
    "93000", "93005",
    "71045", "71046",
    "70450", "70480", "70486",
    "72148", "72141",
    "73721",
}

VALID_ICD10_CODES = {
    "I10",
    "E11.9",
    "R07.9",
    "M54.5",
    "J06.9",
    "S72.001A",
    "G43.909",
    "N18.9",
    "M25.561",
}

CPT_RETIRED_MAPPING = {
    "99201": "99202",
    "99202": "99203",
}

VALID_HCPCS_CODES = {
    "G0438",
    "G0439",
    "J1885",
    "J1100",
}

PRIOR_AUTH_REQUIRED_CPTS = {
    "70450",
    "70480",
    "72148",
    "73721",
}

print("âœ… Code reference sets loaded.")



from typing import Any, Dict, List

Issue = Dict[str, Any]
ValidationResult = Dict[str, Any]


def _has_errors(issues: List[Issue]) -> bool:
    return any(issue.get("severity") == "error" for issue in issues)


def _summarize_issues(
    claim_id: str,
    standard_issues: List[Issue],
    plan_issues: List[Issue],
) -> str:
    if not standard_issues and not plan_issues:
        return (
            f"Claim {claim_id} looks clean based on current standard and "
            f"plan-specific checks."
        )

    parts: List[str] = [f"Claim {claim_id} has some potential issues:"]

    if standard_issues:
        parts.append(f"- Standard checks found {len(standard_issues)} issue(s).")
    if plan_issues:
        parts.append(f"- Plan-specific checks found {len(plan_issues)} issue(s).")

    combined = standard_issues + plan_issues
    top_n = combined[:3]

    for issue in top_n:
        src = issue.get("source", "unknown")
        msg = (issue.get("message") or "").rstrip(".")
        parts.append(f"  - [{src}] {msg}.")

    parts.append("Please review these before submitting to the payer.")
    return "\n".join(parts)


def _check_standard_rules(claim: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    if "patient" not in claim:
        issues.append({
            "severity": "error",
            "code": "MISSING_PATIENT",
            "field": "patient",
            "message": "Patient information is missing.",
            "suggested_fix": (
                "Ensure patient demographics (age, sex, and date of birth) are populated."
            ),
            "source": "standard",
        })

    if "payer" not in claim:
        issues.append({
            "severity": "error",
            "code": "MISSING_PAYER",
            "field": "payer",
            "message": "Payer information is missing.",
            "suggested_fix": "Include payer name and plan details.",
            "source": "standard",
        })

    claim_lines = claim.get("claim_lines") or []
    if not claim_lines:
        issues.append({
            "severity": "error",
            "code": "NO_LINES",
            "field": "claim_lines",
            "message": "Claim has no line items.",
            "suggested_fix": (
                "Add at least one claim line with CPT/HCPCS, quantity, and charge."
            ),
            "source": "standard",
        })
        return issues

    for line in claim_lines:
        ln = line.get("line_number", "UNKNOWN")

        cpt = line.get("cpt")
        if not cpt:
            issues.append({
                "severity": "error",
                "code": "MISSING_CPT",
                "field": f"claim_lines[{ln}].cpt",
                "message": f"Line {ln} is missing a CPT/HCPCS code.",
                "suggested_fix": (
                    "Fill in the appropriate CPT or HCPCS code for this service."
                ),
                "source": "standard",
            })
        else:
            cpt_str = str(cpt).strip()

            if cpt_str in CPT_RETIRED_MAPPING:
                new_code = CPT_RETIRED_MAPPING[cpt_str]
                issues.append({
                    "severity": "warning",
                    "code": "RETIRED_CPT_CODE",
                    "field": f"claim_lines[{ln}].cpt",
                    "message": (
                        f"CPT {cpt_str} appears to be retired and may no longer "
                        "be accepted by many payers."
                    ),
                    "suggested_fix": (
                        f"Consider using CPT {new_code} if clinically appropriate."
                    ),
                    "source": "standard",
                })
            else:
                is_hcpcs = cpt_str[0].isalpha()

                if is_hcpcs:
                    if VALID_HCPCS_CODES and cpt_str not in VALID_HCPCS_CODES:
                        issues.append({
                            "severity": "warning",
                            "code": "UNKNOWN_HCPCS_CODE",
                            "field": f"claim_lines[{ln}].cpt",
                            "message": (
                                f"HCPCS code {cpt_str} is not recognized in the "
                                "current demo reference set."
                            ),
                            "suggested_fix": (
                                "Verify that this is a valid HCPCS Level II code "
                                "for the date of service."
                            ),
                            "source": "standard",
                        })
                else:
                    if VALID_CPT_CODES and cpt_str not in VALID_CPT_CODES:
                        issues.append({
                            "severity": "warning",
                            "code": "UNKNOWN_CPT_CODE",
                            "field": f"claim_lines[{ln}].cpt",
                            "message": (
                                f"CPT code {cpt_str} is not recognized in the "
                                "current demo CPT reference set."
                            ),
                            "suggested_fix": (
                                "Confirm that this is a valid, active CPT code "
                                "for the billed service."
                            ),
                            "source": "standard",
                        })

            # prior auth hint
            if cpt:
                cpt_str = str(cpt).strip()
                if (
                    PRIOR_AUTH_REQUIRED_CPTS
                    and cpt_str in PRIOR_AUTH_REQUIRED_CPTS
                    and not claim.get("prior_auth_number")
                ):
                    issues.append({
                        "severity": "warning",
                        "code": "MISSING_PRIOR_AUTH",
                        "field": "prior_auth_number",
                        "message": (
                            f"CPT {cpt_str} often requires prior authorization for "
                            "many payers, but no prior_auth_number is present."
                        ),
                        "suggested_fix": (
                            "Confirm whether prior authorization is required for this "
                            "service under the patient's plan. If so, add the "
                            "authorization number or obtain authorization before "
                            "submission."
                        ),
                        "source": "standard",
                    })

        qty = line.get("quantity")
        if qty is None or qty <= 0:
            issues.append({
                "severity": "error",
                "code": "INVALID_QUANTITY",
                "field": f"claim_lines[{ln}].quantity",
                "message": f"Line {ln} has an invalid quantity: {qty}.",
                "suggested_fix": "Set quantity to a positive integer (e.g., 1).",
                "source": "standard",
            })

        charge = line.get("charge_amount")
        if charge is None or charge < 0:
            issues.append({
                "severity": "error",
                "code": "INVALID_CHARGE",
                "field": f"claim_lines[{ln}].charge_amount",
                "message": f"Line {ln} has an invalid charge amount: {charge}.",
                "suggested_fix": "Verify the billed charge and ensure it is non-negative.",
                "source": "standard",
            })

    dx_list = claim.get("diagnosis_codes") or []
    if not dx_list:
        issues.append({
            "severity": "error",
            "code": "NO_DIAGNOSIS",
            "field": "diagnosis_codes",
            "message": "No diagnosis codes were provided for this claim.",
            "suggested_fix": "Add at least one ICD-10 diagnosis code.",
            "source": "standard",
        })
    else:
        for dx in dx_list:
            if not dx:
                continue
            dx_str = str(dx).strip()

            if VALID_ICD10_CODES and dx_str not in VALID_ICD10_CODES:
                issues.append({
                    "severity": "warning",
                    "code": "UNKNOWN_DIAGNOSIS_CODE",
                    "field": "diagnosis_codes",
                    "message": (
                        f"Diagnosis code {dx_str} is not recognized in the "
                        "demo ICD-10 reference set."
                    ),
                    "suggested_fix": (
                        "Confirm that this is a valid ICD-10-CM diagnosis code."
                    ),
                    "source": "standard",
                })

    return issues


def _check_plan_specific_rules(claim: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []

    payer = claim.get("payer", {})
    plan_code = payer.get("plan_code")

    if plan_code == "EVG-HMO-101":
        if not claim.get("referral_id"):
            issues.append({
                "severity": "error",
                "code": "MISSING_REFERRAL",
                "field": "referral_id",
                "message": (
                    "Evergreen HMO typically requires a referral for specialist visits."
                ),
                "suggested_fix": (
                    "Ensure a valid referral ID is recorded for this HMO plan."
                ),
                "source": "plan_specific",
            })

        lines = claim.get("claim_lines", [])
        has_em = False
        em_line_needs_25 = None
        has_ekg = False

        for line in lines:
            cpt = (line.get("cpt") or "").strip()
            modifiers = line.get("modifiers") or []

            if cpt.startswith("9921") or cpt.startswith("9920"):
                has_em = True
                if "25" not in modifiers:
                    em_line_needs_25 = line

            if cpt == "93000":
                has_ekg = True

        if has_em and has_ekg and em_line_needs_25:
            ln = em_line_needs_25.get("line_number", "UNKNOWN")
            issues.append({
                "severity": "warning",
                "code": "MISSING_MOD_25",
                "field": f"claim_lines[{ln}].modifiers",
                "message": (
                    "For Evergreen HMO, same-day E/M + EKG is often denied "
                    "unless modifier -25 is applied to the E/M service."
                ),
                "suggested_fix": (
                    "If documentation supports a significant, separately "
                    "identifiable E/M service, add modifier -25."
                ),
                "source": "plan_specific",
            })

    return issues


def validate_claim(claim: Dict[str, Any]) -> ValidationResult:
    """
    Core rule engine: runs standard and plan-specific checks
    and returns a structured result plus a natural-language summary.
    """
    claim_id = claim.get("claim_id", "UNKNOWN")

    standard_issues = _check_standard_rules(claim)
    plan_issues = _check_plan_specific_rules(claim)

    qualified_standard = not _has_errors(standard_issues)
    qualified_plan = not _has_errors(plan_issues)

    if qualified_standard and qualified_plan:
        overall_status = "clean"
    elif not qualified_standard and not qualified_plan:
        overall_status = "high_risk_denial"
    else:
        overall_status = "needs_review"

    summary = _summarize_issues(
        claim_id,
        standard_issues,
        plan_issues,
    )

    return {
        "claim_id": claim_id,
        "qualified_standard": qualified_standard,
        "qualified_plan": qualified_plan,
        "overall_status": overall_status,
        "standard_issues": standard_issues,
        "plan_issues": plan_issues,
        "natural_language_summary": summary,
    }


print("âœ… validate_claim tool and helpers defined.")



# Example claims used for demos and evaluation.
# In production, these would be replaced by real claims from the billing system.

claim_example_clean = {
    "claim_id": "CLM-001",
    "patient": {"id": "PAT-001", "age": 67, "sex": "F", "dob": "1958-03-02"},
    "payer": {
        "name": "Acme Health",
        "plan_type": "Medicare Advantage",
        "plan_code": "ACME-MA-001",
    },
    "provider": {"npi": "1234567890", "name": "Downtown Internal Medicine Clinic"},
    "service_date": "2025-11-18",
    "place_of_service": "11",
    "diagnosis_codes": ["I10", "E11.9"],
    "prior_auth_number": None,
    "referral_id": None,
    "claim_lines": [
        {
            "line_number": 1,
            "cpt": "99213",
            "modifiers": ["25"],
            "quantity": 1,
            "charge_amount": 150.0,
            "diagnosis_pointers": [1, 2],
        },
        {
            "line_number": 2,
            "cpt": "83036",
            "modifiers": [],
            "quantity": 1,
            "charge_amount": 40.0,
            "diagnosis_pointers": [2],
        },
    ],
}

claim_example_with_issues = {
    "claim_id": "CLM-002",
    "patient": {"id": "PAT-002", "age": 45, "sex": "M", "dob": "1980-07-15"},
    "payer": {"name": "Evergreen HMO", "plan_type": "HMO", "plan_code": "EVG-HMO-101"},
    "provider": {"npi": "9876543210", "name": "Evergreen Cardiology Associates"},
    "service_date": "2025-11-18",
    "place_of_service": "11",
    "diagnosis_codes": ["E11.9"],
    "prior_auth_number": None,
    "referral_id": None,
    "claim_lines": [
        {
            "line_number": 1,
            "cpt": "99214",
            "modifiers": [],
            "quantity": 1,
            "charge_amount": 220.0,
            "diagnosis_pointers": [1],
        },
        {
            "line_number": 2,
            "cpt": "93000",
            "modifiers": [],
            "quantity": 1,
            "charge_amount": 90.0,
            "diagnosis_pointers": [1],
        },
    ],
}

print("âœ… Example claims loaded.")




import json

def pretty_print_validation(result: ValidationResult):
    print(f"Claim ID: {result['claim_id']}")
    print(f"Overall status: {result['overall_status']}")
    print(f"Qualified (standard): {result['qualified_standard']}")
    print(f"Qualified (plan):     {result['qualified_plan']}")
    print("\nStandard issues:")
    if not result["standard_issues"]:
        print("  - None")
    else:
        for issue in result["standard_issues"]:
            print(f"  - [{issue['severity']}] {issue['code']}: {issue['message']}")
    print("\nPlan-specific issues:")
    if not result["plan_issues"]:
        print("  - None")
    else:
        for issue in result["plan_issues"]:
            print(f"  - [{issue['severity']}] {issue['code']}: {issue['message']}")
    print("\nSummary:")
    print(result["natural_language_summary"])
    print("\n" + "="*80 + "\n")


print("ğŸ”� Running validator on clean claim...")
res_clean = validate_claim(claim_example_clean)
pretty_print_validation(res_clean)

print("ğŸ”� Running validator on Evergreen HMO claim with issues...")
res_bad = validate_claim(claim_example_with_issues)
pretty_print_validation(res_bad)



# Define the main LlmAgent for the Claims Pre-Screen use case.

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="claims_pre_screen_agent",
    description=(
        "An agent that pre-screens medical claims for potential denials. "
        "It uses a validation tool to run standard and plan-specific checks, "
        "then explains the issues in natural language."
    ),
    instruction=(
        "You are a Claims Pre-Screen Agent for a hospital billing team.\n"
        "\n"
        "When the user provides a claim in JSON format, you MUST:\n"
        "1. Call the `validate_claim` tool with the full claim JSON.\n"
        "2. Read the tool's structured output carefully.\n"
        "3. Explain the results in clear, concise natural language:\n"
        "   - Whether the claim looks clean or is high risk for denial.\n"
        "   - Separate issues from 'standard' checks vs 'plan_specific' checks.\n"
        "   - For each issue, briefly explain what is wrong and how to fix it.\n"
        "4. Give practical suggestions to increase approval chances, but do NOT "
        "invent fake coding rules or override the tool's structured result.\n"
        "\n"
        "Always include the tool's `overall_status` and summarize the most "
        "important issues first. If there are no issues, clearly state that "
        "the claim appears ready for submission based on the current rules."
    ),
    tools=[validate_claim],  # Function tool: ADK wraps this automatically
)

print("âœ… root_agent created and wired with validate_claim tool.")



import json
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- Configuration ---
APP_NAME = "claims_pre_screen_app"
USER_ID = "kaggle_user"
SESSION_ID = "session_001"

# --- Create session service ---
session_service = InMemorySessionService()

# Create session (Kaggle supports top-level await)
await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

print("ğŸŸ¢ Session created successfully.")

# --- Create Runner (same API as your VS Code version) ---
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[],   # We can add JsonFileLoggingPlugin later if needed
)

print("ğŸŸ¢ Runner initialized successfully.")


# --- Helper: run a claim through the agent ---
def run_claim_through_agent(claim: dict) -> str:
    """
    Sends a claim dictionary to the agent and returns its final response text.
    """
    user_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Here is a medical claim in JSON format. "
                    "Please analyze it using your tools and provide recommendations:\n\n"
                    + json.dumps(claim, indent=2)
                )
            )
        ],
    )

    final_text = None

    # NOTE: Runner.run REQUIRES session_id + user_id in this ADK version
    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content:
            final_text = "\n".join(
                part.text
                for part in event.content.parts
                if getattr(part, "text", None)
            )

    return final_text or "âš ï¸� No final response from agent."


# Simple JSONL logging plugin for observability (optional)

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

class JsonFileLoggingPlugin(BasePlugin):
    """
    Logs every ADK Event to a JSONL file: logs/<app_name>_<invocation_id>.jsonl
    This can be used for observability and debugging.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        super().__init__(name="json_file_logging")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Optional[Event]:
        invocation_id = getattr(invocation_context, "invocation_id", "unknown_invocation")
        app_name = getattr(invocation_context, "app_name", "app")

        log_file = self.log_dir / f"{app_name}_{invocation_id}.jsonl"

        try:
            if hasattr(event, "model_dump"):
                event_dict = event.model_dump()
            else:
                event_dict = {
                    "author": getattr(event, "author", None),
                    "content": str(getattr(event, "content", None)),
                    "timestamp": getattr(event, "timestamp", None),
                }

            event_dict["_logged_at_utc"] = datetime.utcnow().isoformat()

            with log_file.open("a", encoding="utf-8") as f:
                json.dump(event_dict, f, default=str)
                f.write("\n")
        except Exception as e:
            # Never let logging break the agent
            print(f"[JsonFileLoggingPlugin] Failed to log event: {e}")

        return None  # keep original event unchanged

print("âœ… JsonFileLoggingPlugin defined.")



# Re-create Runner with logging plugin enabled

logging_plugin = JsonFileLoggingPlugin(log_dir="logs")

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[logging_plugin],
)

print("âœ… Runner re-initialized with JsonFileLoggingPlugin.")



print("=== ğŸ§ª Agent run on CLEAN claim (CLM-001) ===\n")
response_clean = run_claim_through_agent(claim_example_clean)
print(response_clean)

print("\n" + "="*80 + "\n")

print("=== ğŸ§ª Agent run on EVERGREEN HMO claim with issues (CLM-002) ===\n")
response_bad = run_claim_through_agent(claim_example_with_issues)
print(response_bad)



# ğŸ”§ Try your own claim JSON here

# You can modify this dictionary or replace it with any claim JSON you want.
# The agent will run the same tool-based validation and produce a natural-language explanation.
# In practice, you would construct `your_claim` from your billing system's JSON payload.
# For example, by deserializing a message from your integration layer or data warehouse.

your_claim = {
    "claim_id": "CLM-TEST-001",
    "patient": {"id": "TEST-PATIENT", "age": 50, "sex": "F", "dob": "1975-09-30"},
    "payer": {"name": "Evergreen HMO", "plan_type": "HMO", "plan_code": "EVG-HMO-101"},
    "provider": {"npi": "1234509876", "name": "Sample Clinic"},
    "service_date": "2025-11-20",
    "place_of_service": "11",
    "diagnosis_codes": ["I10"],
    "prior_auth_number": None,
    "referral_id": None,
    "claim_lines": [
        {
            "line_number": 1,
            "cpt": "70450",
            "modifiers": [],
            "quantity": 1,
            "charge_amount": 400.0,
            "diagnosis_pointers": [1],
        }
    ],
}

print("=== ğŸ§ª Agent analysis of YOUR claim ===\n")
print(run_claim_through_agent(your_claim))



def eval_clean_claim_is_marked_clean():
    result = validate_claim(claim_example_clean)

    assert result["claim_id"] == "CLM-001"
    assert result["overall_status"] == "clean"
    assert result["qualified_standard"] is True
    assert result["qualified_plan"] is True
    assert result["standard_issues"] == []
    assert result["plan_issues"] == []

    print("âœ… eval_clean_claim_is_marked_clean passed.")


def eval_evergreen_claim_has_plan_specific_issues():
    result = validate_claim(claim_example_with_issues)

    assert result["claim_id"] == "CLM-002"
    assert result["overall_status"] in {"needs_review", "high_risk_denial"}

    assert result["qualified_standard"] is True
    assert result["qualified_plan"] is False

    plan_issues = result["plan_issues"]

    missing_referral = [
        issue for issue in plan_issues
        if issue.get("code") == "MISSING_REFERRAL"
    ]
    assert len(missing_referral) == 1
    assert missing_referral[0]["severity"] == "error"

    missing_mod25 = [
        issue for issue in plan_issues
        if issue.get("code") == "MISSING_MOD_25"
    ]
    assert len(missing_mod25) == 1
    assert missing_mod25[0]["severity"] == "warning"

    print("âœ… eval_evergreen_claim_has_plan_specific_issues passed.")


def eval_missing_patient_triggers_standard_error():
    bad_claim = dict(claim_example_clean)  
    bad_claim.pop("patient", None)

    result = validate_claim(bad_claim)

    assert result["qualified_standard"] is False
    assert result["overall_status"] in {"needs_review", "high_risk_denial"}

    standard_issues = result["standard_issues"]
    missing_patient = [
        issue for issue in standard_issues
        if issue.get("code") == "MISSING_PATIENT"
    ]
    assert len(missing_patient) == 1
    assert missing_patient[0]["severity"] == "error"

    print("âœ… eval_missing_patient_triggers_standard_error passed.")


# Run all evaluations
eval_clean_claim_is_marked_clean()
eval_evergreen_claim_has_plan_specific_issues()
eval_missing_patient_triggers_standard_error()

print("\nğŸ�‰ All evaluation checks passed.")


