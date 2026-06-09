# --- Base Tool Class ---
class Tool:
    def run(self, *args, **kwargs):
        raise NotImplementedError("Tool must implement run()")

# --- Agent Class ---
class Agent:
    def __init__(self, tools):
        self.tools = {tool.__class__.__name__: tool for tool in tools}

    def use(self, tool_name, **kwargs):
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in agent.")
        return tool.run(**kwargs)

# --- Tools Implementation ---

class ResearchTool(Tool):
    def run(self, topic):
        return f"Research findings about {topic}."

class OutlineTool(Tool):
    def run(self, research):
        return {
            "title": f"Blog About: {research}",
            "sections": [
                "Introduction",
                "Key Concepts",
                "Benefits",
                "Examples",
                "Conclusion"
            ]
        }

class DraftTool(Tool):
    def run(self, outline):
        draft = f"# {outline['title']}\n"
        for section in outline["sections"]:
            draft += f"\n## {section}\nThis section will cover {section.lower()}.\n"
        return draft

class EditTool(Tool):
    def run(self, draft):
        return draft.replace("This section will cover", "Let’s explore")

# --- Build the agent ---

agent = Agent([
    ResearchTool(),
    OutlineTool(),
    DraftTool(),
    EditTool()
])

# --- Test the agent ---
topic = "AI Agents in Business"
final_article = agent.use("EditTool", draft=agent.use(
    "DraftTool",
    outline=agent.use(
        "OutlineTool",
        research=agent.use("ResearchTool", topic=topic)
    )
))

print(final_article)


