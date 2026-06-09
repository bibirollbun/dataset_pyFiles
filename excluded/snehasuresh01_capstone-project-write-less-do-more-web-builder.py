# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
from IPython.display import HTML, display



class RequirementAgent:
    """
    Converts user prompt into structured requirements.
    This simulates an AI understanding step.
    """
    def run(self, user_prompt):
        return {
            "page_type": "login" if "login" in user_prompt.lower() else "generic",
            "description": user_prompt,
            "need_css": True,
            "need_js": True if "validation" in user_prompt.lower() else False
        }


class HTMLAgent:
    """
    Generates HTML structure based on page type.
    """
    def run(self, req):
        if req["page_type"] == "login":
            return """
<!DOCTYPE html>
<html>
<head>
<title>Login</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<div class="container">
  <h2>Login</h2>
  <form id="loginForm">
    <input type="text" placeholder="Username" required>
    <input type="password" placeholder="Password" required>
    <button type="submit">Login</button>
  </form>
</div>
<script src="app.js"></script>
</body>
</html>
"""
        return "<h1>Generic Page</h1>"


class CSSAgent:
    """Generates CSS styling."""
    def run(self):
        return """
body {
  font-family: Arial;
  background: #f4f4f4;
}
.container {
  width: 300px;
  margin: 80px auto;
  padding: 20px;
  background: white;
  border-radius: 10px;
}
input, button {
  width: 100%;
  margin-bottom: 10px;
  padding: 10px;
}
"""


class JSAgent:
    """Generates JavaScript code for validation."""
    def run(self):
        return """
document.getElementById("loginForm").addEventListener("submit", function(e){
    e.preventDefault();
    alert("Login successful!");
});
"""


class DebugAgent:
    """
    Cleans, trims, and returns final code bundle.
    """
    def run(self, html, css, js):
        return {
            "html": html.strip(),
            "css": css.strip(),
            "js": js.strip()
        }



class WebsiteBuilder:
    """
    Connects all agents and generates website code.
    """
    def __init__(self):
        self.req_agent = RequirementAgent()
        self.html_agent = HTMLAgent()
        self.css_agent = CSSAgent()
        self.js_agent = JSAgent()
        self.debug_agent = DebugAgent()

    def build(self, user_prompt):
        req = self.req_agent.run(user_prompt)
        html = self.html_agent.run(req)
        css = self.css_agent.run()
        js = self.js_agent.run() if req["need_js"] else ""
        return self.debug_agent.run(html, css, js)



builder = WebsiteBuilder()

output = builder.build("Create a login page with css and validation")

print("===== HTML OUTPUT =====\n", output["html"])
print("\n===== CSS OUTPUT =====\n", output["css"])
print("\n===== JS OUTPUT =====\n", output["js"])



HTML(output["html"])



def test_html_structure():
    assert "<form" in output["html"], "HTML form missing!"

def test_css_applied():
    assert "background" in output["css"], "CSS missing!"

def test_js_validation():
    assert "Login successful" in output["js"], "JS validation missing!"

print("Running tests...")
test_html_structure()
test_css_applied()
test_js_validation()
print("All tests passed! ğŸ�‰")


