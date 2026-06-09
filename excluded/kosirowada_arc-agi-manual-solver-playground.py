!pip install streamlit streamlit-ace matplotlib


# # -*- coding: utf-8 -*-
# import os
# import json
# import streamlit as st
# from glob import glob
# from streamlit_ace import st_ace
# import io, contextlib
# import matplotlib.pyplot as plt

# # --- å¤šè¨€èª�å¯¾å¿œãƒ©ãƒ™ãƒ« ---
# LABELS = {
#     "en": {
#         "title": "ARC-AGI Code Tester",
#         "select_task": "Select a task",
#         "loaded": "Loaded",
#         "examples": "Examples",
#         "train_input": "Train Input",
#         "train_output": "Train Output",
#         "test_input": "Test Input",
#         "your_solution": "Your Solution",
#         "run_test": "â–¶ï¸� Run & Test",
#         "input": "Input",
#         "your_output": "Your Output",
#         "expected": "Expected",
#         "ok": "âœ… OK",
#         "ng": "â�Œ NG",
#         "all_passed": "ğŸ�‰ All examples passed! ğŸ�‰",
#         "error_func": "âš ï¸� Function `d` is not defined.",
#         "error_exec": "âš ï¸� Error during execution:"
#     },
#     "ja": {
#         "title": "ARC-AGI ã‚³ãƒ¼ãƒ‰ ãƒ†ã‚¹ã‚¿ãƒ¼",
#         "select_task": "ã‚¿ã‚¹ã‚¯ã‚’é�¸æŠ�",
#         "loaded": "ã‚’ãƒ­ãƒ¼ãƒ‰",
#         "examples": "ä¾‹",
#         "train_input": "Train å…¥åŠ›",
#         "train_output": "Train å‡ºåŠ›",
#         "test_input": "Test å…¥åŠ›",
#         "your_solution": "ã�‚ã�ªã�Ÿã�®è§£ç­”",
#         "run_test": "â–¶ï¸� å®Ÿè¡Œï¼†æ¤œè¨¼",
#         "input": "å…¥åŠ›",
#         "your_output": "ã�‚ã�ªã�Ÿã�®å‡ºåŠ›",
#         "expected": "æœŸå¾…ã�•ã‚Œã‚‹å‡ºåŠ›",
#         "ok": "âœ… OK",
#         "ng": "â�Œ NG",
#         "all_passed": "ğŸ�‰ å…¨ã�¦ã�®ä¾‹ã‚’ãƒ‘ã‚¹ã�—ã�¾ã�—ã�Ÿï¼� ğŸ�‰",
#         "error_func": "âš ï¸� é–¢æ•° `d` ã�Œå®šç¾©ã�•ã‚Œã�¦ã�„ã�¾ã�›ã‚“ã€‚",
#         "error_exec": "âš ï¸� å®Ÿè¡Œä¸­ã�«ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�—ã�¾ã�—ã�Ÿ:"
#     }
# }

# # --- è¨€èª�é�¸æŠ� ---
# lang = st.sidebar.selectbox("Language / è¨€èª�", options=["ja", "en"], index=0)
# l = LABELS[lang]

# st.set_page_config(layout="wide")
# st.title(l["title"])

# # --- ã‚µã‚¤ãƒ‰ãƒ�ãƒ¼: ã‚¿ã‚¹ã‚¯é�¸æŠ� ---
# st.sidebar.markdown(f"**{l['select_task']}**")

# task_paths = sorted(glob("google-code-golf-2025/task*.json"))
# task_names = [os.path.basename(p) for p in task_paths]
# selected_name = st.sidebar.selectbox("ã‚¿ã‚¹ã‚¯ã‚’é�¸æŠ�", task_names)
# # map the chosen filename back to its full path
# selected = task_paths[ task_names.index(selected_name) ]
# with open(selected) as f:
#     task = json.load(f)
# st.sidebar.markdown(f"**{selected_name} {l['loaded']}**")

# # --- ã‚°ãƒªãƒƒãƒ‰æ��ç”»é–¢æ•° ---
# def show_grid(grid, caption=None):
#     cmap = plt.get_cmap("tab10")
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     if caption:
#         ax.set_title(caption, fontsize=10)
#     st.pyplot(fig)
#     plt.close(fig)

# # --- ä¾‹ã�®è¡¨ç¤º ---
# st.subheader(l["examples"])
# train = task.get("train", [])
# test  = task.get("test", [])
# n = max(len(train), len(test))
# cols = st.columns(n)
# for i, col in enumerate(cols):
#     with col:
#         st.markdown(f"### Example {i+1}")
#         if i < len(train):
#             st.markdown(f"**{l['train_input']}**")
#             show_grid(train[i]["input"])
#             st.markdown(f"**{l['train_output']}**")
#             show_grid(train[i]["output"])
#         if i < len(test):
#             st.markdown(f"**{l['test_input']}**")
#             show_grid(test[i]["input"])

# # --- ã‚³ãƒ¼ãƒ‰å…¥åŠ›ã‚¨ãƒªã‚¢ ---
# st.subheader(l["your_solution"])
# def_code = """\
# # def p(g): ã‚’å®Ÿè£…ã�—ã�¦ã��ã� ã�•ã�„
# def p(g):
#     return g
# """
# user_code = st_ace(
#     value=def_code,
#     language="python",
#     theme="monokai",
#     key="ace",
#     font_size=14,
#     height=250,
# )

# # --- å®Ÿè¡Œï¼†æ¤œè¨¼ ---
# if st.button(l['run_test']):
#     buf = io.StringIO()
#     ns = {}
#     try:
#         with contextlib.redirect_stdout(buf):
#             exec(user_code, ns)
#         fn = ns.get("p")
#         if not callable(fn):
#             st.error(l['error_func'])
#         else:
#             results = {"train": [], "test": [], "arc-gen": []}
#             for split in ["train","test","arc-gen"]:
#                 for ex in task.get(split, []):
#                     out = fn(ex["input"])
#                     ok = (out == ex["output"])
#                     results[split].append({
#                         "input": ex["input"],
#                         "expected": ex["output"],
#                         "output": out,
#                         "ok": ok
#                     })
#             all_ok = all(item["ok"] for split in ["train","test"] for item in results[split])
#             stdout = buf.getvalue()
#             if stdout.strip():
#                 st.subheader("Stdout")
#                 st.code(stdout)
#             for split in ["train","test"]:
#                 if not results[split]: continue
#                 st.subheader(split.upper())
#                 cols = st.columns(len(results[split]))
#                 for col, res in zip(cols, results[split]):
#                     with col:
#                         st.markdown(f"**{l['input']}**")
#                         show_grid(res["input"])
#                         st.markdown(f"**{l['your_output']}**")
#                         show_grid(res["output"])
#                         st.markdown(f"**{l['expected']}**")
#                         show_grid(res["expected"])
#                         if res["ok"]:
#                             st.success(l['ok'])
#                         else:
#                             st.error(l['ng'])
#             if all_ok:
#                 st.success(l['all_passed'])
#     except Exception as e:
#         st.error(f"{l['error_exec']} {e}")


