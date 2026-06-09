!pip install --upgrade transformers --no-index --find-links="/kaggle/input/transformer-latest"


from transformers import pipeline
import torch

model_id = "/kaggle/input/gpt-oss-20b/transformers/default/1"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])


# Install required packages
!pip install --upgrade transformers --no-index --find-links="/kaggle/input/transformer-latest"
!pip install --no-index --find-links="/kaggle/input/wheels" langchain langchain_core langchain_community langchain_huggingface langgraph


from transformers import pipeline
import torch
import numpy as np
import ast
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.agent import AgentFinish
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult

print("All dependencies imported successfully!")


# Add all missing imports
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import List
import numpy as np

class SymmetryInput(BaseModel):
    matrix_data: List[float] = Field(..., description="List of numbers to be arranged into a square matrix for symmetry analysis")

@tool(args_schema=SymmetryInput)
def check_symmetry(matrix_data: List[float]) -> str:
    """
    Analyzes the symmetry of a matrix created from the input list.
    The function arranges the input numbers into a perfect square matrix and checks for:
    - Horizontal symmetry (top-bottom reflection)
    - Vertical symmetry (left-right reflection) 
    - Diagonal symmetry (main diagonal reflection)
    - Anti-diagonal symmetry (anti-diagonal reflection)
    
    Args:
        matrix_data: List of numbers to be arranged into a square matrix
        
    Returns:
        String describing the symmetry analysis results
    """
    try:
        # Import numpy inside the function to ensure it's available
        import numpy as np
        
        # Convert to numpy array
        data = np.array(matrix_data, dtype=float)
        n = len(data)
        
        # Check if we can form a perfect square matrix
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n != n:
            return f"Error: Cannot form a perfect square matrix from {n} elements. Need a perfect square number of elements."
        
        # Reshape into square matrix
        matrix = data.reshape(sqrt_n, sqrt_n)
        
        results = []
        results.append(f"Matrix ({sqrt_n}x{sqrt_n}):")
        results.append(str(matrix))
        results.append("")
        
        # Check horizontal symmetry (top-bottom)
        horizontal_symmetric = np.array_equal(matrix, np.flipud(matrix))
        results.append(f"Horizontal symmetry (top-bottom): {'YES' if horizontal_symmetric else 'NO'}")
        
        # Check vertical symmetry (left-right)
        vertical_symmetric = np.array_equal(matrix, np.fliplr(matrix))
        results.append(f"Vertical symmetry (left-right): {'YES' if vertical_symmetric else 'NO'}")
        
        # Check main diagonal symmetry
        diagonal_symmetric = np.array_equal(matrix, matrix.T)
        results.append(f"Main diagonal symmetry: {'YES' if diagonal_symmetric else 'NO'}")
        
        # Check anti-diagonal symmetry
        anti_diagonal_symmetric = np.array_equal(matrix, np.fliplr(np.flipud(matrix)).T)
        results.append(f"Anti-diagonal symmetry: {'YES' if anti_diagonal_symmetric else 'NO'}")
        
        # Overall symmetry assessment
        symmetric_types = []
        if horizontal_symmetric:
            symmetric_types.append("horizontal")
        if vertical_symmetric:
            symmetric_types.append("vertical")
        if diagonal_symmetric:
            symmetric_types.append("main diagonal")
        if anti_diagonal_symmetric:
            symmetric_types.append("anti-diagonal")
        
        if symmetric_types:
            results.append(f"Matrix is symmetric with respect to: {', '.join(symmetric_types)}")
        else:
            results.append("Matrix has no symmetry properties")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error analyzing symmetry: {str(e)}"

# Test the tool
print("Symmetry analysis tool created successfully!")
print("Tool name:", check_symmetry.name)
print("Tool description:", check_symmetry.description)


# Create tools list
tools = [check_symmetry]

# Convert tools to OpenAI function format
functions = [convert_to_openai_function(tool) for tool in tools]

# Create the function-calling compatible model
model = LocalChatModel(llm).bind_functions(functions)

print(f"Created {len(tools)} tools:")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")
print("Model with function binding created!")


from pydantic import Field

class LocalChatModel(BaseChatModel):
    """
    Debugging wrapper that shows exactly what the LLM is doing
    """
    pipeline: Any = Field(default=None)
    functions: List[Dict] = Field(default_factory=list)
    
    def __init__(self, huggingface_pipeline, **kwargs):
        super().__init__(pipeline=huggingface_pipeline, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "local_chat_model"
    
    def bind_functions(self, functions):
        """Bind functions to the model like ChatOpenAI does"""
        self.functions = functions
        return self
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from the model"""
        try:
            # Get the user message
            user_message = messages[-1].content if messages else ""
            
            print(f"DEBUG - User message: {user_message}")
            print(f"DEBUG - Available functions: {[f['name'] for f in self.functions]}")
            
            # Create a simple, clear prompt
            if self.functions:
                prompt = f"""You are an AI assistant with access to tools.

User question: {user_message}

Available tools:
{chr(10).join([f"- {func['name']}: {func['description']}" for func in self.functions])}

Instructions:
1. If you need to use a tool, respond with: "I will use [tool_name] with parameters [parameters]"
2. If you don't need a tool, respond normally

Your response:"""
            else:
                prompt = user_message
            
            print(f"DEBUG - Prompt sent to LLM: {prompt}")
            
            # Get response from your local model
            response = self.pipeline.invoke([HumanMessage(content=prompt)])
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            print(f"DEBUG - LLM raw response: {content}")
            
            # Let the LLM decide if it wants to use a tool
            if "I will use" in content and "with parameters" in content:
                print("DEBUG - LLM decided to use a tool")
                return self._parse_llm_decision(content)
            else:
                print("DEBUG - LLM decided not to use a tool")
                # Regular response
                message = AIMessage(content=content)
                return ChatResult(generations=[ChatGeneration(message=message)])
                
        except Exception as e:
            print(f"DEBUG - Error in LocalChatModel: {e}")
            message = AIMessage(content=f"Error: {str(e)}")
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _parse_llm_decision(self, content: str) -> ChatResult:
        """Parse the LLM's decision about tool usage"""
        try:
            print(f"DEBUG - Parsing LLM decision: {content}")
            
            # Extract tool name and parameters from LLM's response
            # The LLM should say: "I will use check_symmetry with parameters [1, 2, 3, 4, 5, 6, 7, 8, 9]"
            
            # Find the tool name
            tool_name = None
            for func in self.functions:
                if func['name'] in content:
                    tool_name = func['name']
                    break
            
            print(f"DEBUG - Extracted tool name: {tool_name}")
            
            if not tool_name:
                print("DEBUG - No valid tool name found, treating as regular response")
                message = AIMessage(content=content)
                return ChatResult(generations=[ChatGeneration(message=message)])
            
            # Extract parameters - let the LLM specify them
            # Look for "with parameters" and extract what comes after
            if "with parameters" in content:
                param_part = content.split("with parameters")[1].strip()
                print(f"DEBUG - Parameter part: {param_part}")
                
                # The LLM should provide the parameters in a clear format
                # We'll let the agent handle the parameter extraction
                parameters = {"llm_specified_params": param_part}
            else:
                parameters = {}
            
            print(f"DEBUG - Final parameters: {parameters}")
            
            # Create a function call message
            message = AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": tool_name,
                        "arguments": json.dumps(parameters)
                    }
                }
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
                
        except Exception as e:
            print(f"DEBUG - Error parsing LLM decision: {e}")
            message = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=message)])

print("Debugging LocalChatModel class created!")


def run_agent(user_input: str) -> str:
    """
    Run the agent with LLM-driven parameter extraction
    """
    try:
        print(f"DEBUG - Starting agent with input: {user_input}")
        
        result = chain.invoke({"input": user_input})
        
        print(f"DEBUG - Chain result type: {type(result)}")
        print(f"DEBUG - Chain result: {result}")
        
        if isinstance(result, AgentFinish):
            print("DEBUG - Agent finished without tool usage")
            return result.return_values['output']
        
        # Execute the tool
        tool_name = result.tool
        tool_input = result.tool_input
        
        print(f"DEBUG - Tool to execute: {tool_name}")
        print(f"DEBUG - Tool input: {tool_input}")
        
        # Handle LLM-specified parameters
        if "llm_specified_params" in tool_input:
            # The LLM specified the parameters, let's extract them intelligently
            param_spec = tool_input["llm_specified_params"]
            print(f"DEBUG - LLM specified parameters: {param_spec}")
            
            # Ask the LLM to extract the actual numbers
            extraction_prompt = f"""Extract the list of numbers from this text: {param_spec}

Return only the numbers in the format: [1, 2, 3, 4, 5, 6, 7, 8, 9]"""
            
            extraction_response = llm.invoke([HumanMessage(content=extraction_prompt)])
            extracted = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
            
            print(f"DEBUG - LLM extraction response: {extracted}")
            
            # Parse the extracted numbers
            try:
                # Find the list in the response
                start = extracted.find('[')
                end = extracted.find(']')
                if start != -1 and end != -1:
                    number_str = extracted[start+1:end]
                    numbers = [float(x.strip()) for x in number_str.split(',')]
                    tool_input = {"matrix_data": numbers}
                    print(f"DEBUG - Parsed numbers: {numbers}")
                else:
                    return "Could not extract numbers from LLM response"
            except Exception as e:
                print(f"DEBUG - Error parsing numbers: {e}")
                return f"Error parsing numbers: {str(e)}"
        
        # Find and execute the tool
        tool_to_run = None
        for tool in tools:
            if tool.name == tool_name:
                tool_to_run = tool
                break
        
        if tool_to_run is None:
            return f"Error: Tool '{tool_name}' not found"
        
        print(f"DEBUG - Executing tool: {tool_to_run.name}")
        
        # Run the tool
        observation = tool_to_run.run(tool_input)
        
        print(f"DEBUG - Tool result: {observation}")
        
        # Create final response
        final_response = f"Analysis complete:\n\n{observation}\n\nBased on this analysis, the matrix shows the symmetry properties listed above."
        return final_response
        
    except Exception as e:
        print(f"DEBUG - Error during agent execution: {e}")
        return f"Error during agent execution: {str(e)}"

print("LLM-driven agent implementation created!")


# Test with a symmetric matrix
test_input = "What is the symmetry of [1, 2, 3, 1, 2, 3, 1, 2, 3]?"
print("Testing agent:")
print("=" * 50)
result = run_agent(test_input)
print(result)


from pydantic import Field

class LocalChatModel(BaseChatModel):
    """
    Completely non-hardcoded wrapper - LLM makes all decisions
    """
    pipeline: Any = Field(default=None)
    functions: List[Dict] = Field(default_factory=list)
    
    def __init__(self, huggingface_pipeline, **kwargs):
        super().__init__(pipeline=huggingface_pipeline, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "local_chat_model"
    
    def bind_functions(self, functions):
        """Bind functions to the model like ChatOpenAI does"""
        self.functions = functions
        return self
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from the model"""
        try:
            # Get the user message
            user_message = messages[-1].content if messages else ""
            
            print(f"DEBUG - User message: {user_message}")
            print(f"DEBUG - Available functions: {[f['name'] for f in self.functions]}")
            
            # Create a completely open-ended prompt
            if self.functions:
                prompt = f"""You are an AI assistant with access to tools.

User question: {user_message}

Available tools:
{chr(10).join([f"- {func['name']}: {func['description']}" for func in self.functions])}

You can use these tools if needed. Respond naturally and let me know if you need to use any tools."""
            else:
                prompt = user_message
            
            print(f"DEBUG - Prompt sent to LLM: {prompt}")
            
            # Get response from your local model
            response = self.pipeline.invoke([HumanMessage(content=prompt)])
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            print(f"DEBUG - LLM raw response: {content}")
            
            # Let the LLM decide everything - no hardcoded checks
            return self._let_llm_decide(content)
                
        except Exception as e:
            print(f"DEBUG - Error in LocalChatModel: {e}")
            message = AIMessage(content=f"Error: {str(e)}")
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _let_llm_decide(self, content: str) -> ChatResult:
        """Let the LLM decide if it wants to use a tool - completely open-ended"""
        try:
            print(f"DEBUG - Letting LLM decide: {content}")
            
            # Ask the LLM to make the decision
            decision_prompt = f"""Based on your previous response: "{content}"

Do you want to use any of these tools?
{chr(10).join([f"- {func['name']}: {func['description']}" for func in self.functions])}

If yes, respond with:
TOOL: [tool_name]
PARAMETERS: [parameters in JSON format]

If no, respond with:
NO_TOOL: [your regular response]"""
            
            print(f"DEBUG - Decision prompt: {decision_prompt}")
            
            decision_response = self.pipeline.invoke([HumanMessage(content=decision_prompt)])
            decision_content = decision_response.content if hasattr(decision_response, 'content') else str(decision_response)
            
            print(f"DEBUG - LLM decision: {decision_content}")
            
            # Parse the LLM's decision
            if "TOOL:" in decision_content and "PARAMETERS:" in decision_content:
                return self._parse_llm_tool_decision(decision_content)
            else:
                # Regular response
                message = AIMessage(content=decision_content)
                return ChatResult(generations=[ChatGeneration(message=message)])
                
        except Exception as e:
            print(f"DEBUG - Error in LLM decision: {e}")
            message = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _parse_llm_tool_decision(self, content: str) -> ChatResult:
        """Parse the LLM's tool decision - completely LLM-driven"""
        try:
            print(f"DEBUG - Parsing LLM tool decision: {content}")
            
            # Extract tool name and parameters from LLM's response
            lines = content.split('\n')
            tool_name = None
            parameters = {}
            
            for line in lines:
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                elif line.startswith("PARAMETERS:"):
                    param_str = line.replace("PARAMETERS:", "").strip()
                    try:
                        parameters = json.loads(param_str)
                    except:
                        # Let the LLM fix the parameters
                        parameters = {"raw_params": param_str}
            
            print(f"DEBUG - LLM chose tool: {tool_name}")
            print(f"DEBUG - LLM chose parameters: {parameters}")
            
            if tool_name:
                # Create a function call message
                message = AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": tool_name,
                            "arguments": json.dumps(parameters)
                        }
                    }
                )
                return ChatResult(generations=[ChatGeneration(message=message)])
            else:
                # Regular response
                message = AIMessage(content=content)
                return ChatResult(generations=[ChatGeneration(message=message)])
                
        except Exception as e:
            print(f"DEBUG - Error parsing LLM tool decision: {e}")
            message = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=message)])

print("Completely non-hardcoded LocalChatModel class created!")


def run_agent(user_input: str) -> str:
    """
    Completely non-hardcoded agent - LLM makes all decisions
    """
    try:
        print(f"DEBUG - Starting agent with input: {user_input}")
        
        result = chain.invoke({"input": user_input})
        
        print(f"DEBUG - Chain result type: {type(result)}")
        print(f"DEBUG - Chain result: {result}")
        
        if isinstance(result, AgentFinish):
            print("DEBUG - Agent finished without tool usage")
            return result.return_values['output']
        
        # Execute the tool
        tool_name = result.tool
        tool_input = result.tool_input
        
        print(f"DEBUG - Tool to execute: {tool_name}")
        print(f"DEBUG - Tool input: {tool_input}")
        
        # Handle raw parameters from LLM
        if "raw_params" in tool_input:
            # Ask the LLM to convert raw parameters to proper format
            conversion_prompt = f"""Convert these parameters to proper JSON format: {tool_input['raw_params']}

Return only valid JSON."""
            
            conversion_response = llm.invoke([HumanMessage(content=conversion_prompt)])
            converted = conversion_response.content if hasattr(conversion_response, 'content') else str(conversion_response)
            
            print(f"DEBUG - LLM converted parameters: {converted}")
            
            try:
                tool_input = json.loads(converted)
            except:
                return "LLM could not provide valid parameters"
        
        # Find and execute the tool
        tool_to_run = None
        for tool in tools:
            if tool.name == tool_name:
                tool_to_run = tool
                break
        
        if tool_to_run is None:
            return f"Error: Tool '{tool_name}' not found"
        
        print(f"DEBUG - Executing tool: {tool_to_run.name}")
        
        # Run the tool
        observation = tool_to_run.run(tool_input)
        
        print(f"DEBUG - Tool result: {observation}")
        
        # Let the LLM create the final response
        final_prompt = f"""User asked: {user_input}

Tool result: {observation}

Provide a helpful response to the user."""
        
        final_response = llm.invoke([HumanMessage(content=final_prompt)])
        final_content = final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        print(f"DEBUG - LLM final response: {final_content}")
        
        return final_content
        
    except Exception as e:
        print(f"DEBUG - Error during agent execution: {e}")
        return f"Error during agent execution: {str(e)}"

print("Completely non-hardcoded agent implementation created!")


# Test with a symmetric matrix
test_input = "What is the symmetry of [1, 2, 3, 1, 2, 3, 1, 2, 3]?"
print("Testing agent:")
print("=" * 50)
result = run_agent(test_input)
print(result)




