import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')



class Visualizer():
    """A class for visualizing single frames of mouse videos.

    From https://www.kaggle.com/code/ambrosm/mabe-eda-which-makes-sense
    """
    paws = ['forepaw_left', 'forepaw_right', 'hindpaw_left', 'hindpaw_right']
    head = ['ear_left', 'ear_right', 'nose', 'ear_left']

    def __init__(self, train):
        """Initialize a visualizer.
        
        Parameters:
        train: pandas DataFrame read from train.csv
        """
        self.train = train
    
    def load_video(self, train_idx):
        """Load the specified video into the visualizer"""
        self.train_idx = train_idx
        lab_id = self.train.iloc[train_idx].lab_id
        video_id = self.train.iloc[train_idx].video_id
        path = f"/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{lab_id}/{video_id}.parquet"
        self.video_name = path.split('/')[-1].split('.')[0]
        self.vid = pd.read_parquet(path)
        try:
            self.annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
        except FileNotFoundError:
            self.annot = None
        self.pvid = self.vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        self.bodyparts = set(self.pvid.loc[self.pvid.index[0], ('x', 1)].index)
        # print(self.bodyparts)
        self.n_mouses = len(np.unique(self.pvid.columns.get_level_values('mouse_id')))

    def __len__(self):
        """Frame count of video"""
        return len(self.pvid)

    def plot_frame(self, frame_indices):
        fig = make_subplots(
            len(frame_indices), 
            1, 
            shared_xaxes=True,
            subplot_titles=[f'frame: {frame_idx}' for frame_idx in frame_indices],
            vertical_spacing=0.02
        )
        
        for i_frame_idx, frame_idx in enumerate(frame_indices):
            """Plot the selected frame of the previously loaded video"""
            video_frame = self.pvid.index[frame_idx]
            if (self.pvid.loc[video_frame] == 0).all():
                print(f"{self.train_idx}.{frame_idx} is empty.")
                return
            
            # fig = go.Figure() # for one plot            
            colors = ['green', 'blue', 'orange', 'brown']

            for mouse, color in enumerate(colors[:self.n_mouses]):
                mouse_id = mouse + 1
                mx = self.pvid.loc[video_frame, ('x', mouse_id)].copy()
                my = self.pvid.loc[video_frame, ('y', mouse_id)].copy()

                # Plot the head
                # Every mouse has ear_left and ear_right
                if 'nose' in mx.index and mx['nose'] != 0:

                    # head fill 
                    # same as: plt.fill(mx[self.head], my[self.head], color=color, alpha=0.5)
                    head_x = mx[self.head].tolist()
                    head_y = my[self.head].tolist()
                    fig.add_trace(
                        go.Scatter(
                            x=head_x,
                            y=head_y,
                            fill='toself',
                            fillcolor=color,
                            opacity=0.5,
                            line=dict(color=color, width=0),
                            mode='lines',
                            showlegend=False,
                            hoverinfo='skip',
                        ),
                        row=i_frame_idx+1, col=1,
                    )

                    # noze point
                    # same as: plt.scatter([mx['nose']], [my['nose']], s=100, color=color)
                    fig.add_trace(
                        go.Scatter(
                            x=[mx['nose']],
                            y=[my['nose']],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            marker_symbol='diamond',
                            name=f'Mouse {mouse_id}',
                            showlegend=(i_frame_idx == 0), 
                            hovertemplate=f'<b>Nose</b><br>Mouse: {mouse_id}<br>X: {mx["nose"]}<br>Y: {my["nose"]}<extra></extra>'
                        ),
                        row=i_frame_idx+1, col=1,
                    )

                    # head parts (ear_left, ear_right) - invisible points for hoverinfo
                    for head_part in ['ear_left', 'ear_right']:
                        if head_part in mx.index and mx[head_part] != 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=[mx[head_part]],
                                    y=[my[head_part]],
                                    mode='markers',
                                    marker=dict(size=10, color=color, opacity=0),
                                    name=f'Mouse {mouse_id}',
                                    showlegend=False,
                                    hovertemplate=f'<b>{head_part}</b><br>Mouse: {mouse_id}<br>X: {mx[head_part]}<br>Y: {my[head_part]}<extra></extra>'
                                ),
                                row=i_frame_idx+1, col=1,
                            )
                    
                else:
                    # Ears line
                    # same as: plt.plot(mx[['ear_left', 'ear_right']], my[['ear_left', 'ear_right']], color=color)
                    ear_x = mx[['ear_left', 'ear_right']].tolist()
                    ear_y = my[['ear_left', 'ear_right']].tolist()
                    fig.add_trace(
                        go.Scatter(
                            x=ear_x,
                            y=ear_y,
                            mode='lines+markers',
                            line=dict(color=color, width=2),
                            marker=dict(size=5, color=color),
                            name=f'Mouse {mouse_id}',
                            showlegend=(i_frame_idx == 0),
                            hovertemplate=f'<b>Ear</b><br>Mouse: {mouse_id}<extra></extra>'
                        ),
                        row=i_frame_idx+1, col=1,
                    )
                if 'head' not in mx.index:
                    mx['head'] = mx[['ear_left', 'ear_right']].mean()
                    my['head'] = my[['ear_left', 'ear_right']].mean()

                # Plot the body and tail
                # Every mouse has tail_base, but it can be 0
                parts_list = ['head']
                if 'neck' in mx.index and mx['neck'] != 0:
                    parts_list.append('neck')
                if 'body_center' in mx.index and mx['body_center'] != 0:
                    parts_list.append('body_center')
                if mx['tail_base'] != 0:
                    parts_list.append('tail_base')
                if 'tail_tip' in mx.index and mx['tail_tip'] != 0:
                    parts_list.append('tail_tip')

                body_x = [mx[part] for part in parts_list if mx[part] != 0]
                body_y = [my[part] for part in parts_list if mx[part] != 0]
                body_customdata = [[mouse_id, part, mx[part], my[part]] for part in parts_list if mx[part] != 0]
                
                # same as: plt.plot(mx[parts_list], my[parts_list], color=color)
                fig.add_trace(
                    go.Scatter(
                        x=body_x,
                        y=body_y,
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=5, color=color),
                        showlegend=False,
                        customdata=body_customdata,
                        hovertemplate='<b>%{customdata[1]}</b><br>Mouse: %{customdata[0]}<br>X: %{customdata[2]}<br>Y: %{customdata[3]}<extra></extra>'
                    ),
                    row=i_frame_idx+1, col=1,
                )

                # Plot the width of the body
                if 'lateral_right' in mx.index:
                    # same as: plt.plot(mx[['lateral_right', 'lateral_left']], my[['lateral_right', 'lateral_left']], color=color)
                    fig.add_trace(
                        go.Scatter(
                            x=mx[['lateral_right', 'lateral_left']],
                            y=my[['lateral_right', 'lateral_left']],
                            mode='lines+markers',
                            line=dict(color=color, width=2),
                            marker=dict(size=5, color=color),
                            showlegend=False,
                            customdata=[[mouse_id, part, mx[part], my[part]] for part in ['lateral_right', 'lateral_left']],
                            hovertemplate='<b>%{customdata[1]}</b><br>Mouse: %{customdata[0]}<br>X: %{customdata[2]}<br>Y: %{customdata[3]}<extra></extra>'
                        ),
                        row=i_frame_idx+1, col=1,
                    )
                    

                # Plot the hip
                if 'hip_right' in mx.index:
                    # same as: plt.plot(mx[['hip_right', 'hip_left']], my[['hip_right', 'hip_left']], color=color)
                    fig.add_trace(
                        go.Scatter(
                            x=mx[['hip_right', 'hip_left']],
                            y=my[['hip_right', 'hip_left']],
                            mode='lines+markers',
                            line=dict(color=color, width=2),
                            marker=dict(size=5, color=color),
                            showlegend=False,
                            customdata=[[mouse_id, part, mx[part], my[part]] for part in ['hip_right', 'hip_left']],
                            hovertemplate='<b>%{customdata[1]}</b><br>Mouse: %{customdata[0]}<br>X: %{customdata[2]}<br>Y: %{customdata[3]}<extra></extra>'
                        ),
                        row=i_frame_idx+1, col=1,
                    )

                # Plot the paws
                if 'forepaw_left' in mx.index:
                    # same as: plt.scatter(mx[self.paws], my[self.paws], color=color)
                    fig.add_trace(
                        go.Scatter(
                            x=mx[self.paws],
                            y=my[self.paws],
                            mode='markers',
                            line=dict(color=color, width=2),
                            marker=dict(size=10, color=color),
                            showlegend=False,
                            customdata=[[mouse_id, part, mx[part], my[part]] for part in self.paws],
                            hovertemplate='<b>%{customdata[1]}</b><br>Mouse: %{customdata[0]}<br>X: %{customdata[2]}<br>Y: %{customdata[3]}<extra></extra>'
                        ),
                        row=i_frame_idx+1, col=1,
                    )

            if self.annot is not None:
                actions = set(self.annot.action[(self.annot.start_frame <= video_frame) & (video_frame <= self.annot.stop_frame)])
                if len(actions) == 0:
                    actions = ''
            else:
                actions = ''

            # same as: plt.title(f'{self.train_idx}.{frame_idx} {actions}')
            # same as: plt.gca().set_aspect('equal')
            fig.update_layout(
                title=f'{self.train_idx}.{frame_idx} {actions}',
                hovermode='closest',
                template='plotly_white',
                height=300*len(frame_indices),
                width=900
            )
        
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )


        # same as: plt.show()
        fig.show()
visualizer = Visualizer(train)


# HOW TO USE 
# Firstly, set the index of the training data
visualizer.load_video(5772)

# Secondly, put the list of frames that you want to check
visualizer.plot_frame([0, 500, 1000])

