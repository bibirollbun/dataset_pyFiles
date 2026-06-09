import os
import textwrap
from typing import Dict, List

import numpy as np
import yaml

from google.genai.types import Part, Content

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.adk.tools import load_memory  


import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key

print("✅ Loaded GOOGLE_API_KEY from Kaggle Secrets.")


MODEL_NAME = "gemini-2.0-flash"
APP_NAME = "kubewise-k8s-resource-agent"
USER_ID = "kubewise-user"
SESSION_ID = "kubewise-session-1"

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

if "GOOGLE_API_KEY" not in os.environ:
    print(
        "⚠️ GOOGLE_API_KEY is NOT set.\n"
        "   In Kaggle, add it as a secret named GOOGLE_API_KEY "
        "or export it locally before running the notebook."
    )
else:
    print("✅ GOOGLE_API_KEY found in environment.")


SYNTHETIC_METRICS: Dict[str, Dict[str, List[float]]] = {
    "prod/orders-service": {
        "cpu_mcores": [80, 120, 90, 140, 200, 160, 130, 150, 180, 210],
        "memory_mib": [220, 240, 230, 260, 280, 300, 310, 295, 305, 320],
    },
    "prod/payments-service": {
        "cpu_mcores": [40, 60, 50, 55, 70, 65, 75, 60, 80, 90],
        "memory_mib": [150, 160, 155, 170, 165, 175, 180, 190, 185, 200],
    },
    "nonprod/reporting-service": {
        "cpu_mcores": [20, 30, 25, 35, 40, 28, 32, 36, 38, 45],
        "memory_mib": [100, 110, 105, 120, 130, 115, 118, 122, 125, 135],
    },
}


def _metrics_key(namespace: str, workload: str) -> str:
    return f"{namespace}/{workload}"



def load_usage_samples(namespace: str, workload: str) -> Dict[str, object]:
    key = _metrics_key(namespace, workload)

    if key not in SYNTHETIC_METRICS:
        np.random.seed(abs(hash(key)) % (2**32))
        cpu = np.random.normal(loc=150, scale=40, size=60).clip(20)
        mem = np.random.normal(loc=300, scale=80, size=60).clip(64)
        SYNTHETIC_METRICS[key] = {
            "cpu_mcores": cpu.tolist(),
            "memory_mib": mem.tolist(),
        }

    data = SYNTHETIC_METRICS[key]
    return {
        "namespace": namespace,
        "workload": workload,
        "cpu_mcores": data["cpu_mcores"],
        "memory_mib": data["memory_mib"],
    }


def summarize_usage(namespace: str, workload: str) -> Dict[str, float]:
    samples = load_usage_samples(namespace, workload)
    cpu = np.array(samples["cpu_mcores"])
    mem = np.array(samples["memory_mib"])

    summary = {
        "namespace": namespace,
        "workload": workload,
        "cpu_avg_mcores": float(cpu.mean()),
        "cpu_p95_mcores": float(np.percentile(cpu, 95)),
        "cpu_max_mcores": float(cpu.max()),
        "memory_avg_mib": float(mem.mean()),
        "memory_p95_mib": float(np.percentile(mem, 95)),
        "memory_max_mib": float(mem.max()),
        "num_samples": len(cpu),
    }
    return summary


def generate_resource_patch(
    namespace: str,
    workload: str,
    current_yaml: str,
    target_utilization: float = 0.7,
) -> Dict[str, object]:

    metrics = summarize_usage(namespace, workload)

    obj = yaml.safe_load(current_yaml)

    container = obj["spec"]["template"]["spec"]["containers"][0]
    resources = container.setdefault("resources", {})
    requests = resources.setdefault("requests", {})
    limits = resources.setdefault("limits", {})

    cpu_p95 = metrics["cpu_p95_mcores"]
    mem_p95 = metrics["memory_p95_mib"]

    cpu_request = max(25, int(cpu_p95 / target_utilization))
    cpu_limit = int(cpu_request * 1.5)

    mem_request = max(64, int(mem_p95 / target_utilization))
    mem_limit = int(mem_request * 1.4)

    requests["cpu"] = f"{cpu_request}m"
    requests["memory"] = f"{mem_request}Mi"
    limits["cpu"] = f"{cpu_limit}m"
    limits["memory"] = f"{mem_limit}Mi"

    patched_yaml = yaml.safe_dump(obj, sort_keys=False)

    return {
        "namespace": namespace,
        "workload": workload,
        "target_utilization": target_utilization,
        "metrics_summary": metrics,
        "suggested_requests": {
            "cpu_mcores": cpu_request,
            "memory_mib": mem_request,
        },
        "suggested_limits": {
            "cpu_mcores": cpu_limit,
            "memory_mib": mem_limit,
        },
        "patched_yaml": patched_yaml,
    }



metrics_agent = LlmAgent(
    name="MetricsAnalystAgent",
    model=MODEL_NAME,
    description="Analyzes Kubernetes CPU & memory usage for a single workload.",
    instruction=textwrap.dedent("""
        You are a Kubernetes SRE and capacity planner.
        For the given namespace and workload:
        1. Call the `load_usage_samples` tool to retrieve CPU/memory samples.
        2. Optionally call `summarize_usage` to get summary statistics.
        3. Produce a compact summary that includes avg, p95 and max
           CPU (mcores) and memory (MiB), plus one paragraph of interpretation.
        Return your analysis as clear Markdown.
    """).strip(),
    tools=[load_usage_samples, summarize_usage],
    output_key="metrics_summary"  
)

planner_agent = LlmAgent(
    name="ResourcePlannerAgent",
    model=MODEL_NAME,
    description="Maps usage metrics to Kubernetes resource requests & limits.",
    instruction=textwrap.dedent("""
        You are a Kubernetes performance engineer.
        You receive:
        - `metrics_summary` in the session state with usage statistics.
        - A Deployment YAML snippet for the target workload.

        Use the `generate_resource_patch` tool to compute new CPU/memory
        requests and limits that target around 70% utilization at p95.
        Then briefly explain why these values are safe and cost-aware.
    """).strip(),
    tools=[generate_resource_patch],
    output_key="tuning_plan"
)

explainer_agent = LlmAgent(
    name="DevOpsExplainerAgent",
    model=MODEL_NAME,
    description="Explains tuning changes in friendly language for app owners.",
    instruction=textwrap.dedent("""
        You are a friendly DevOps teammate named KubeWise.
        You receive a `tuning_plan` from the previous agent.

        Tasks:
        1. First, optionally call the `load_memory` tool to see if there were
           previous KubeWise recommendations for this workload and highlight trends.
        2. Produce a clear explanation for an application team including:
           - Short summary of the issue.
           - A table-style text with OLD vs NEW CPU/memory requests/limits.
           - Expected impact on reliability and cloud cost.
           - A safe rollout plan (e.g., canary, progressive rollout, monitoring).

        Keep the tone professional but supportive.
    """).strip(),
    tools=[load_memory],
)

pipeline_agent = SequentialAgent(
    name="KubeWisePipeline",
    description="KubeWise pipeline: metrics → resource planning → explanation.",
    sub_agents=[metrics_agent, planner_agent, explainer_agent],
)

root_agent = pipeline_agent




session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

def ask_kubewise(question: str) -> str:
    user_msg = Content(
        role="user",
        parts=[Part.from_text(text=question)],
    )

    run_config = RunConfig(response_modalities=["TEXT"])

    final_text = ""
    events = runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_msg,
        run_config=run_config,
    )
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    return final_text.strip()


async def save_session_to_memory():
    completed_session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    await memory_service.add_session_to_memory(completed_session)
    print("✅ Session added to in-memory long-term storage.")



import warnings
warnings.filterwarnings("ignore", module="google.genai.types")

example_deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-service
  namespace: prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: orders-service
  template:
    metadata:
      labels:
        app: orders-service
    spec:
      containers:
      - name: orders
        image: mycorp/orders:1.0
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "300m"
            memory: "512Mi"
"""

user_prompt = (
    "You are KubeWise, a Kubernetes resource tuning assistant.\n\n"
    "Namespace: prod\n"
    "Workload: orders-service\n\n"
    "Here is the current Deployment YAML:\n\n"
    + example_deployment_yaml
    + "\n\nPlease:\n"
    "- Analyze recent CPU and memory usage for this workload.\n"
    "- Recommend new CPU/memory requests and limits based on the metrics.\n"
    "- Explain the change, the expected impact on reliability and cost,\n"
    "  and how to roll out the change safely.\n"
)

response = ask_kubewise(user_prompt)
print(response)



await save_session_to_memory()



follow_up_prompt = (
    "Earlier you generated a tuning plan for the `orders-service` workload "
    "in the `prod` namespace. Using any memory you have stored, briefly "
    "summarize what you previously recommended and why, and explain how an "
    "SRE could use that history to decide the next tuning iteration."
)

follow_up_response = ask_kubewise(follow_up_prompt)
print(follow_up_response)


