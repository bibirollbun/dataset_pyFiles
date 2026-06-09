import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install transformers==4.46.0


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import Dict, List, Tuple
import logging
import numpy as np
from scipy.interpolate import interp2d


device = "auto"
model_path = "ibm-granite/granite-3.1-2b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)


model.save_pretrained("modified_granite_model")
tokenizer.save_pretrained("modified_granite_model")


def load_model():
    device = "auto"
    model_path = "modified_granite_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()
    return model, tokenizer


def get_model_response(model, tokenizer, prompt, max_tokens=350):
    input_text = f"Question: {prompt}\n\nAnswer:"     #input question prompt
    input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)      #tokenize input
    """
    Output response generated using the model with specified parameters
    """
    output = model.generate(
        **input_tokens,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean up the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(input_text):].strip()
    return response


def model_inference():
    model, tokenizer = load_model() #load model and tokenizer
    """
    question prompts related to heat equation
    """
    cases = {
        "Case1Q1": """Analyze the steady-state heat equation T(x,y) = x^2+y^2 within a unit square. 
    Explain the mathematical behavior of the temperature, particularly at the corners and along specific axes. 
    Discuss the implications of boundary conditions on the temperature distribution in the system.""",
        "Case1Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        how does the temperature change with respect to the position along the x-axis at y = 0.5?""",
        "Case1Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        if we increase the coeeficient of pi in the force function what will happen?""",
        "Case2Q1": """Analyze the steady-state heat equation solution T(x, y) = x^2+y^2 within a unit square domain. 
Explain why the temperature is zero at both x=0 and x=1, considering the boundary conditions T(x, y) = 0 at these edges. 
Discuss the physical interpretation of this, particularly in terms of heat flow, and how the temperature distribution behaves near these boundaries.""",
        "Case2Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        at what coordinates does the maximum temperature occur, and what determines this location?""",
        "Case2Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        how does the temperature profile change along the vertical line x=0.5 compared to x=0.25?""",
        "Case3Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what is the temperature at the corner (0, 0) of the unit square mesh?""",
        "Case3Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?""",
        "Case3Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what does the boundary condition u(1,y)=y(1−y) represent physically in this heat diffusion problem?""",
        "Case4Q1": """Provide a detailed mathematical and physical interpretation of the steady-state heat equation T(x,y) = x² + y² in a unit square domain. Discuss boundary conditions and possible edge effects and provide 
        What does this decay rate tell you about the heat dissipation process within the system?""",
        "Case4Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time?""",
        "Case4Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        What is the effect of alpha on the decay rate of heat dissipation?"""
    }
    
    results = []   #store results
    
    for case, prompt in cases.items():      #iterate over each case
        print(f"\nTesting {case}...")
        try:
            response = get_model_response(model, tokenizer, prompt)    #get response
            results.append(response)
            print(f"\nResponse for {case}:")
            print(response)
        
        except Exception as e:
            print(f"Error in {case}: {str(e)}")
            results.append(f"Error: {str(e)}")
    
    return results   #return results


results = model_inference()    #  function call


df = pd.DataFrame({"Id" : list(range(1,13)), "Answer": results})    # Dataframe to store results
df.to_csv('submission.csv',index = False)    #save as csv file




