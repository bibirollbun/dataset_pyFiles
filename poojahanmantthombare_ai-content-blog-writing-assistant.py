# Cell 1: Imports & utilities
import json, time, random, os, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

def save_checkpoint(state: dict, filename="mcp_checkpoint.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"[checkpoint saved] -> {filename}")

def load_checkpoint(filename="mcp_checkpoint.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"[checkpoint loaded] <- {filename}")
        return state
    print("[no checkpoint found]")
    return None

def short_hash(s: str, n=6):
    return hashlib.sha1(s.encode()).hexdigest()[:n]





# Cell 3: Mock LLM agent (replace with real LLM calls if you have an API key)
class MockLLMAgent(Agent):
    def __init__(self, agent_id: str, creativity=0.7):
        super().__init__(agent_id)
        self.creativity = creativity

    def generate(self, prompt: str, role_hint: Optional[str]=None) -> str:
        base = f"[{self.agent_id} reply to: {prompt[:120]}]"
        variations = [
            "Here's a clear, helpful section that explains the concept.",
            "Short, punchy paragraph explaining main ideas with examples.",
            "A well-structured, SEO-friendly subheading and paragraph.",
            "An engaging intro with a hook and 2 bullets for readers."
        ]
        chosen = random.choice(variations)
        creativity_suffix = (" More depth." if random.random() < self.creativity else "")
        return f"{base} {chosen}{creativity_suffix}"

    def run(self, **kwargs) -> AgentResult:
        prompt = kwargs.get("prompt", "")
        role = kwargs.get("role", None)
        out = self.generate(prompt, role_hint=role)
        return AgentResult(agent_id=self.agent_id, success=True, output=out, meta={"prompt": prompt})



# Cell 5: Specialized agents (title, outline, draft, editor)
class TitleGeneratorAgent(MockLLMAgent):
    def run(self, **kwargs):
        topic = kwargs.get("topic", "Untitled Topic")
        prompts = [
            f"Give 6 catchy blog post titles for: {topic}",
            f"Write SEO-friendly titles for article about: {topic}"
        ]
        titles = []
        for p in prompts:
            res = self.generate(p)
            for i in range(3):
                titles.append(f"{topic} — {res.split(':')[-1].strip()} (angle {i+1})")
        return AgentResult(agent_id=self.agent_id, success=True, output=titles, meta={"topic": topic})

class OutlineAgent(MockLLMAgent):
    def run(self, **kwargs):
        title = kwargs.get("title", "Untitled")
        points = [
            "Introduction: hook + context",
            "What the problem/opportunity is",
            "How to approach it (steps & tools)",
            "Examples or case studies",
            "Common pitfalls and tips",
            "Conclusion + call to action"
        ]
        outline = {p: self.generate(f"Write a short paragraph for: {p} in article titled {title}") for p in points}
        return AgentResult(agent_id=self.agent_id, success=True, output=outline, meta={"title": title})

class DraftAgent(MockLLMAgent):
    def run(self, **kwargs):
        title = kwargs.get("title","Untitled")
        outline = kwargs.get("outline", {})
        draft_sections = []
        for h, p in outline.items():
            section_text = f"## {h}\n\n{p}\n\n" + self.generate(f"Expand section: {h} for article {title}")
            draft_sections.append(section_text)
        draft = f"# {title}\n\n" + "\n\n".join(draft_sections)
        return AgentResult(agent_id=self.agent_id, success=True, output=draft, meta={"word_count": len(draft.split())})

class EditorAgent(MockLLMAgent):
    def run(self, **kwargs):
        draft = kwargs.get("draft", "")
        edited = draft.replace(" More depth.", ".").replace("Here's", "Here is")
        if "call to action" not in edited.lower():
            edited += "\n\n**Call to action:** Try these steps and share your results."
        changelog = {"edits": "clarity improvements, CTA added, simplified sentences"}
        return AgentResult(agent_id=self.agent_id, success=True, output=edited, meta=changelog)



# Cell 6: MCP orchestrator with parallel/sequential and pause/resume
class MCP:
    def __init__(self, name="MCP-AI-Blog", checkpoint_file="mcp_checkpoint.json"):
        self.name = name
        self.checkpoint_file = checkpoint_file
        self.state = {"tasks": [], "history": []}

    def register_task(self, task_meta: dict):
        self.state["tasks"].append(task_meta)

    def run_parallel(self, agents: List[Agent], common_kwargs: dict, max_workers=4):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(agent.run, **common_kwargs): agent for agent in agents}
            for fut in as_completed(futures):
                agent = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = AgentResult(agent_id=agent.agent_id, success=False, output=str(e))
                results.append(res)
                self.state["history"].append({"agent": agent.agent_id, "result": (res.output if res.success else None)})
        return results

    def run_sequential(self, agents: List[Agent], initial_kwargs: dict):
        kwargs = initial_kwargs.copy()
        seq_results = []
        for agent in agents:
            res = agent.run(**kwargs)
            seq_results.append(res)
            if isinstance(res.output, dict):
                kwargs.update(res.output if isinstance(res.output, dict) else {})
            elif isinstance(res.output, list):
                if isinstance(agent, TitleGeneratorAgent):
                    kwargs["titles"] = res.output
                else:
                    kwargs["last_list"] = res.output
            else:
                kwargs["last_output"] = res.output
            self.state["history"].append({"agent": agent.agent_id, "result_sample": str(res.output)[:200]})
        return seq_results

    def pause(self):
        save_checkpoint(self.state, self.checkpoint_file)

    def resume(self):
        loaded = load_checkpoint(self.checkpoint_file)
        if loaded:
            self.state = loaded



# Cell 7: Example run (put into a single cell to run end-to-end)
topic = "AI Content & Blog Writing Assistant: Build a Multi-Agent System"
mcp = MCP()

# Parallel title generation
title_agents = [TitleGeneratorAgent(f"title_agent_{i}") for i in range(3)]
parallel_title_results = mcp.run_parallel(title_agents, {"topic": topic}, max_workers=3)

print("\n=== Parallel Title Generation Results ===")
all_titles = []
for r in parallel_title_results:
    print(f"\n[{r.agent_id}] produced {len(r.output)} titles (sample):")
    for t in r.output[:4]:
        print("  -", t)
    all_titles.extend(r.output)

# shortlist
seen = set(); shortlist = []
for t in all_titles:
    key = short_hash(t)
    if key not in seen:
        seen.add(key); shortlist.append(t)
shortlist = shortlist[:6]
print("\n[Shortlist Titles]")
for i,t in enumerate(shortlist,1):
    print(f"{i}. {t}")

# Sequential pipeline: outline -> draft -> edit
chosen_title = shortlist[0]
outline_agent = OutlineAgent("outline_agent_1")
draft_agent = DraftAgent("draft_agent_1")
editor_agent = EditorAgent("editor_agent_1")
seq_agents = [outline_agent, draft_agent, editor_agent]

print("\n=== Sequential Pipeline (Outline -> Draft -> Edit) ===")
seq_results = mcp.run_sequential(seq_agents, {"title": chosen_title})

outline_result = seq_results[0].output
draft_result = seq_results[1].output
edited_result = seq_results[2].output

print("\n[Outline sample keys]:", list(outline_result.keys())[:3])
print("\n[Draft word count sample]:", seq_results[1].meta.get("word_count"))
print("\n[Edited sample start]:")
print(edited_result[:700] + ("\n...[truncated]" if len(edited_result)>700 else ""))

# Tools analysis
keywords = extract_keywords(edited_result, top_k=8)
seo = seo_score(edited_result, keywords)
read = readability_score(edited_result)
summary = summarize(edited_result, max_sentences=4)
print("\n=== Tools Analysis ===")
print("Keywords:", keywords)
print(f"SEO score (mock): {seo:.1f}")
print(f"Readability score (approx): {read:.1f}")
print("Short summary:\n", summary)

# Loop agent refine until readability target
class LoopAgent:
    def __init__(self, edit_agent: EditorAgent, target_readability=55.0, max_iters=3):
        self.edit_agent = edit_agent
        self.target = target_readability
        self.max_iters = max_iters

    def refine(self, draft_text):
        state = {"iter": 0, "text": draft_text, "history": []}
        while state["iter"] < self.max_iters:
            score = readability_score(state["text"])
            state["history"].append({"iter": state["iter"], "readability": score})
            print(f"[LoopAgent] iteration {state['iter']} readability={score:.1f}")
            if score >= self.target:
                print("[LoopAgent] target reached.")
                break
            res = self.edit_agent.run(draft=state["text"])
            state["text"] = res.output
            state["iter"] += 1
        return state

loop_agent = LoopAgent(editor_agent, target_readability=55.0, max_iters=3)
loop_state = loop_agent.refine(edited_result)
final_text = loop_state["text"]
final_read = readability_score(final_text)
print(f"\n[Final readability after loop]: {final_read:.1f}")

# Save artifact and checkpoint
mcp.state["final_article"] = {"title": chosen_title, "content": final_text, "keywords": keywords, "seo": seo, "readability": final_read}
mcp.pause()
outfile = "final_article.md"
with open(outfile, "w", encoding="utf-8") as f:
    f.write(f"# {chosen_title}\n\n" + final_text)
print(f"\n[Saved article] -> {outfile} (size {os.path.getsize(outfile)} bytes)")

print("\n=== Compact Report ===")
report = {"topic": topic, "chosen_title": chosen_title, "keywords": keywords, "seo_score": seo, "readability": final_read, "artifact": outfile}
print(json.dumps(report, indent=2))


