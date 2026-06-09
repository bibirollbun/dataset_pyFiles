import os
import base64
from IPython.display import HTML

def show_gif(gif_path):
    with open(gif_path, "rb") as f:
        data_uri = base64.b64encode(f.read()).decode("utf-8")
    return f'<img src="data:image/gif;base64,{data_uri}" style="margin:5px;" />'

def animate_task(task):
    base_dir = "/kaggle/input/animationarc"
    all_html_parts = []

    for fname in sorted(os.listdir(base_dir)):
        if fname.startswith(task) and fname.endswith(".gif"):
            full_path = os.path.join(base_dir, fname)
            all_html_parts.append(show_gif(full_path))
    
    if not all_html_parts:
        return HTML(f"<p>No GIFs found for task: {task}</p>")
    
    full_html = "<div style='display:flex; flex-wrap:wrap'>" + "".join(all_html_parts) + "</div>"
    return HTML(full_html)


animate_task('2c181942')


animate_task('16b78196')


animate_task('62593bfd')


animate_task('b5ca7ac4')


animate_task('cbebaa4b')


animate_task('6e453dd6')


animate_task('da515329')


animate_task('53fb4810')


animate_task('64efde09')


animate_task('db0c5428')


animate_task('142ca369')


animate_task('b9e38dc0')


animate_task('35ab12c3')


animate_task('36a08778')


animate_task('3e6067c3')


animate_task('4c416de3')


animate_task('409aa875')


animate_task('b10624e5')


animate_task('97d7923e')


animate_task('4c3d4a41')


animate_task('e376de54')


animate_task('71e489b6')


animate_task('1ae2feb7')


animate_task('16de56c4')

