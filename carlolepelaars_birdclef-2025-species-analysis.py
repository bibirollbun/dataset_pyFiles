# Render Plotly as png
!pip install -Uqq kaleido


import numpy as np
import pandas as pd
import soundfile as sf
import plotly.express as px
from IPython.display import Audio

# Fix Plotly rendering in Jupyter forks.
# If you are running this notebook locally you can comment this out.
# This allows you to play with interactive Plotly plots.
import plotly.io as pio
pio.renderers.default = 'png'


BASE_PATH = "/kaggle/input/birdclef-2025/"
t = pd.read_csv(BASE_PATH + "taxonomy.csv")
t = t.drop(columns=["inat_taxon_id"])
t.tail(2)


print(f"There are '{len(t)}' different species in the dataset.")


def plot_value_counts(value_counts, title, x_label="Count", y_label="Category", top_n=None):
    sorted_counts = value_counts.sort_values()
    sorted_counts = sorted_counts.tail(top_n) if top_n is not None else sorted_counts
    fig = px.bar(
        x=sorted_counts.values,
        y=sorted_counts.index,
        orientation='h',
        labels={'x': x_label, 'y': y_label},
        title=title,
        text=sorted_counts.values
    )
    fig.update_layout(
        font=dict(size=14),
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor="white", font_size=14),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )
    return fig


fig = plot_value_counts(
    t['class_name'].value_counts(), 
    title="Distribution of Species by Class",
    y_label="Class Name"
)
fig.show()


def get_types(t):
    phrases = ['frog', 'otter', 'lion', 'fox', 'owl', 'flycatcher', 'hawk', 'parrot', 'toucan', 'raccoon', 'jaguar', 
            'squirrel', 'monkey', 'toad', 'duck', 'kingfisher', 'vulture', 'parakeet', 'dove', 'woodpecker', 
            'blackbird', 'caracara', 'cricket', 'macaw', 'ibis', 'oriol', 'warbler', 'swallow', 'kingbird', 
            'kestrel', 'seedeater', 'spinetail', 'peccary', 'sloth', 'spoonbill', 'hummingbird', 'stork', 
            'cicadas', 'antbird', 'heron', 'sandpiper', 'woodcreeper', 'cuckoo', 'wren', 'falcon', 'tyrant',
            'katydid', 'anhinga', 'bananaquit', 'donacobius', 'grassquit', 'antshrike', 'jay', 'tityra', 
            'curassow', 'motmot', 'tanager', 'saltator', 'grackle', 'becard', 'aracari', 'chachalaca', 
            'pauraque', 'potoo', 'bobwhite', 'guan', 'oropendola', 'manakin', 'ani', 'egret', 'kiskadee', 
            'tinamou', 'martin', 'tern', 'grebe', 'cormorant', 'screamer', 'piculet', 'hornero', 'pigeon', 
            'puffbird', 'kite', 'gallinule', 'jacamar', 'finch', 'tyrannulet', 'lapwing', 'euphonia', 
            'schiffornis', 'parula', 'jacana', 'trogon', 'elaenia', 'cacique']
    
    def get_names(t, phrase):
        return [i.lower() for i in t['common_name'] if phrase in i.lower()]

    result = {p: len(get_names(t, p)) for p in phrases}
    other_names = [name for name in t['common_name'] if not any(p in name.lower() for p in phrases)]
    result['other'] = len(other_names)
    return pd.Series(result)


all_types = get_types(t)
plot_value_counts(all_types, "All Species", top_n=20)


bird_types = get_types(t[t['class_name'] == 'Aves'])
plot_value_counts(bird_types, "Bird Species", top_n=15)


not_bird_types = get_types(t[t['class_name'] != 'Aves'])
plot_value_counts(not_bird_types, "Non-Bird Species", top_n=15)


def load_audio(path, sec=None, sample_rate=32_000) -> np.array:
    with sf.SoundFile(path) as f: audio = f.read(int(sec * sample_rate)) if sec else f.read()
    return audio

def show_audio(path, sec=None):
    return Audio(load_audio(path, sec=sec), rate=32_000)


show_audio(BASE_PATH + "train_audio/22973/XC882793.ogg", sec=5)


show_audio(BASE_PATH + "train_audio/42087/iNat860016.ogg")


show_audio(BASE_PATH + "train_audio/42087/iNat155127.ogg")


show_audio(BASE_PATH + "train_audio/50186/CSA35128.ogg", sec=30)

