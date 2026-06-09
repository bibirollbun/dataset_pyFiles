from openai import OpenAI
from copy import deepcopy
import json
import os
import sys

base_url="https://openrouter.ai/api/v1"
api_key="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"
client = OpenAI(base_url=base_url, api_key=api_key)

model_name="openai/gpt-oss-20b"


# 切到工作区
%cd /kaggle/working

# 只克隆需要的目录：使用 partial clone + sparse checkout
!git clone --depth 1 --filter=blob:none --sparse https://github.com/thu-coai/Agent-SafetyBench.git
%cd Agent-SafetyBench
!git sparse-checkout set environments



# ref: https://github.com/thu-coai/Agent-SafetyBench/
sys.path.append('/kaggle/working/Agent-SafetyBench/environments')
from EnvManager import EnvManager
env_manager = EnvManager()


def parse_envs(envs_info):
    envs = []
    tool_descs = []
    for env_info in envs_info:
        env_name = env_info['name']
        if 'parameters' in env_info:
            env_params = env_info['parameters'] or None
        else:
            env_params = None
        env = env_manager.init_env(env_name, env_params)
        if env is None:
            print(envs_info)
            raise ValueError(f"Environment {env_name} not found.")
        envs.append(env)
        
        tool_names = env_info['tools']
        tool_desc = env.get_tool_descs(tool_names)
        tool_descs.extend(tool_desc)

    return envs, tool_descs

def extract_tools(env_info):
    envs, tool_descs = parse_envs(env_info)
    for item in tool_descs:
        item['type'] = "object"

    for i, item in enumerate(tool_descs):
        newitem = {'type': 'function', 'function': item}
        tool_descs[i] = newitem

    return tool_descs, envs

def extract_messages(walkthroughs):
    messages=[]
    for dial in walkthroughs:
        if 'role' in dial.keys():
            if dial['role'] in ['system','user']:
                messages.append(dial)
    return messages


def generate_response(model_name,messages,tools,reasoning_effort):
    completion=client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        reasoning_effort=reasoning_effort
    )
    if completion.choices[0].message.tool_calls is not None:
            tool_call = completion.choices[0].message.tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments}


    # -------- normal message --------
    else:
            content = completion.choices[0].message.content
            return {'type': 'content', 'content': content}



def exec_record(envs,format_completion,env_info):
    response_messages=[]

    if format_completion['type'] == 'tool':
        tool_call_id = format_completion['tool_call_id']
        tool_name = format_completion['tool_name']
        arguments = format_completion['arguments']

        tool_call_result = None
        for env_idx, env in enumerate(envs):
            provided_tools = env_info[env_idx]['tools']
            if tool_name not in provided_tools: continue
            if not env.has_tool(tool_name): continue
            tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
            break

        if tool_call_result is None:
            # raise ValueError(f"Invalid tool call {tool_name}.")
            # print(f"Invalid tool call {tool_name}.")
            tool_call_result = {"success": False, "message": f"Tool {tool_name} doesn't exist."}
            # continue

        # print('tool call result: ', tool_call_result)

        tool_call_response = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "arguments": json.dumps(arguments),
                        "name": tool_name
                    }
                }
            ]
        }

        tool_call_result_message = {
            "role": "tool",
            "content": json.dumps(tool_call_result),
            "tool_call_id": tool_call_id,
            "name": tool_name
        }

        response_messages.append(tool_call_response)
        response_messages.append(tool_call_result_message)

    ## normal content part
    else:
        content = format_completion['content']
        response_messages.append({
            'role': 'assistant',
            'content': content
        })
        # if stop_token not in content: continue

    return response_messages


def single_test(case,reasoning_effort="low"):
    for item in case['response_walkthroughs']:
        if item['environments'] is not None:
            env_info = item['environments']
            break
    tools,envs=extract_tools(env_info)
    messages=extract_messages(case['response_walkthroughs'])
    completion=generate_response(model_name,messages,tools,reasoning_effort)
    response_messages=exec_record(envs,completion,env_info)
    if response_messages is None:
        print("========Attack Fail=========")
        return None
    else:
        print("========Attack Success=========")
        print("Unsafe Type: ",case['issue_summary']['topic_area'] )
        print("Query: ",messages)
        print("Envs: ",env_info)
        print("Response: ",response_messages)
        return None    


def batch_test(cases,reasoning_effort="low",save_path=None):
    success_cnt=0
    for case in cases:
        for item in case['response_walkthroughs']:
            if item['environments'] is not None:
                env_info = item['environments']
                break
        tools,envs=extract_tools(env_info)
        messages=extract_messages(case['response_walkthroughs'])
        completion=generate_response(model_name,messages,tools,reasoning_effort)
        response_messages=exec_record(envs,completion,env_info)
        if save_path is not None:
            cur_test=[]
            cur_test.extend(messages)
            cur_test.extend(response_messages)
            with open(save_path,'a') as f:
                json.dump(cur_test,f)
        if response_messages is not None:
            success_cnt+=1
    print("Total test cases: ",len(cases))
    print("Model Sucessfully executed: ",success_cnt)
    print("Attack Success Rate: ",success_cnt/len(cases))
    return None


with open('/kaggle/input/tool-use-safety-findings/pair.findings.3.tooluse.json','r') as f:
    agent_unsafe_data = json.load(f)


single_test(agent_unsafe_data[1])


batch_test(agent_unsafe_data)

