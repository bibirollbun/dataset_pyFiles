from os import path, listdir
import pandas as pd
from ipywidgets import IntSlider, Layout, AppLayout, VBox
from matplotlib import pyplot as plt
%matplotlib widget

def ImageFiles(df, id, file_location='train'):
    try:
        assert any(df['tomo_id']==id), ['Invalid key name', id]
        train_dir = path.join(DATA_DIR, file_location, id)
        assert path.isdir(train_dir), ['The folder does not exist', train_dir]
        image_files = [path.join(train_dir, f) for f in listdir(train_dir) if f.endswith('.jpg')]
        assert image_files, ['The folder is empty', train_dir]
        print('\033[1;34mID:\033[0;30m{} \033[1;34mSize:\033[0;30m{} \033[1;34mPath:\033[0;30m{}\033[0;0m'.format(id, len(image_files), train_dir))
        return sorted(image_files)
    except AssertionError as e:
        print(f'\033[1;31m{e.args[0][0]}:\033[0;0m', e.args[0][1])
    except Exception as e:
        print(f'\033[1;31mError:\033[0;0m', e)

def SliceView(image_files, tomo_id):
    with plt.ioff():
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False   
        I = plt.imread(image_files[0]) 
        img = plt.imshow(I, cmap='gray')
    def update(change):
        I = plt.imread(image_files[change['new']-1]) 
        img.set_data(I)
        fig.canvas.draw_idle()    
    slider = IntSlider(value=1, min=1, max=len(image_files), description=tomo_id)
    slider.layout = Layout(width='30%')
    slider.style.handle_color='LimeGreen'
    slider.observe(update, names='value')    
    AppLayout(
        center=fig.canvas,
        footer=slider,
        pane_heights=[0, 6, 1]
    )
    vbox = VBox([slider, fig.canvas])    
    display(vbox)

def SampleStack(slices, rows=5, cols=5, start=1, step=1):
    slices = slices[start:rows*cols*step+start:step]
    stack = [[path.splitext(path.basename(f))[0], plt.imread(f)] for f in slices]
    with plt.ioff():
        sz = 8, 8*rows//cols+1
        fig, axes = plt.subplots(rows, cols, figsize=sz, constrained_layout = True)        
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')    
        for ax, (name, img) in zip(axes, stack):
            ax.set_title(name) 
            ax.imshow(img,cmap='gray')
    display(fig)



DATA_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
print(listdir(DATA_DIR))


filename = path.join(DATA_DIR, 'train_labels.csv')
train_labels = pd.read_csv(filename, index_col='row_id')
display(pd.concat([train_labels.head(3), train_labels.tail(3)]))


tomo_id = 'tomo_00e047'
image_files = ImageFiles(train_labels, tomo_id)


SampleStack(image_files, start=5, step=14)


SliceView(image_files, tomo_id)







