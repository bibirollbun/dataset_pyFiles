import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from ipywidgets import interact, FloatSlider

# Function to draw the stick figures
def draw_stick_figures(ax, x_offset1=0, x_offset2=0.5):
    ax.clear()
    # First stick figure (Boy)
    # Head
    head1 = Circle((x_offset1, 1), 0.2, color='blue', fill=False)
    ax.add_patch(head1)
    # Body
    ax.plot([x_offset1, x_offset1], [0.8, 0.2], 'k-')
    # Arms
    ax.plot([x_offset1 - 0.3, x_offset1 + 0.3], [0.6, 0.6], 'k-')
    # Left leg
    ax.plot([x_offset1, x_offset1 - 0.2], [0.2, 0], 'k-')
    # Right leg
    ax.plot([x_offset1, x_offset1 + 0.2], [0.2, 0], 'k-')

    # Second stick figure (Girl)
    # Head
    head2 = Circle((x_offset2, 1), 0.2, color='pink', fill=False)
    ax.add_patch(head2)
    # Body
    ax.plot([x_offset2, x_offset2], [0.8, 0.2], 'k-')
    # Arms
    ax.plot([x_offset2 - 0.3, x_offset2 + 0.3], [0.6, 0.6], 'k-')
    # Dress (Triangle)
    dress = Polygon(
        [[x_offset2 - 0.2, 0.2], [x_offset2 + 0.2, 0.2], [x_offset2, 0.6]],
        closed=True,
        color='pink',
        alpha=0.5,
    )
    ax.add_patch(dress)
    # Left leg
    ax.plot([x_offset2, x_offset2 - 0.2], [0.2, 0], 'k-')
    # Right leg
    ax.plot([x_offset2, x_offset2 + 0.2], [0.2, 0], 'k-')

    # Set limits and aspect
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

# Function to update positions of stick figures
def move_stick_figures(position):
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_stick_figures(ax, x_offset1=position, x_offset2=position + 0.5)
    plt.show()

# Interactive slider
interact(
    move_stick_figures,
    position=FloatSlider(value=0, min=-0.5, max=0.5, step=0.05, description='Move'),
)



import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from ipywidgets import interact, FloatSlider

# Function to draw the stick figures
def draw_family(ax, x_offset_baby=0.25, y_offset_baby=0.2):
    ax.clear()
    
    # First stick figure (Boy - Parent 1)
    head1 = Circle((0, 1), 0.2, color='blue', fill=False)
    ax.add_patch(head1)
    ax.plot([0, 0], [0.8, 0.2], 'k-')  # Body
    ax.plot([-0.3, 0.3], [0.6, 0.6], 'k-')  # Arms
    ax.plot([0, -0.2], [0.2, 0], 'k-')  # Left leg
    ax.plot([0, 0.2], [0.2, 0], 'k-')  # Right leg

    # Second stick figure (Girl - Parent 2)
    head2 = Circle((0.5, 1), 0.2, color='pink', fill=False)
    ax.add_patch(head2)
    ax.plot([0.5, 0.5], [0.8, 0.2], 'k-')  # Body
    ax.plot([0.2, 0.8], [0.6, 0.6], 'k-')  # Arms
    dress = Polygon(
        [[0.3, 0.2], [0.7, 0.2], [0.5, 0.6]],
        closed=True,
        color='pink',
        alpha=0.5,
    )
    ax.add_patch(dress)
    ax.plot([0.5, 0.3], [0.2, 0], 'k-')  # Left leg
    ax.plot([0.5, 0.7], [0.2, 0], 'k-')  # Right leg

    # Third stick figure (Baby - Jumping)
    head_baby = Circle((x_offset_baby, y_offset_baby + 0.4), 0.1, color='green', fill=False)
    ax.add_patch(head_baby)
    ax.plot([x_offset_baby, x_offset_baby], [y_offset_baby + 0.3, y_offset_baby], 'k-')  # Body
    ax.plot([x_offset_baby - 0.15, x_offset_baby + 0.15], [y_offset_baby + 0.2, y_offset_baby + 0.2], 'k-')  # Arms
    ax.plot([x_offset_baby, x_offset_baby - 0.1], [y_offset_baby, y_offset_baby - 0.2], 'k-')  # Left leg
    ax.plot([x_offset_baby, x_offset_baby + 0.1], [y_offset_baby, y_offset_baby - 0.2], 'k-')  # Right leg

    # Set limits and aspect
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

# Function to update the baby’s position
def move_baby(jump):
    fig, ax = plt.subplots(figsize=(6, 4))
    y_offset_baby = 0.2 + 0.3 * abs(np.sin(jump * np.pi))  # Baby jumps up and down
    draw_family(ax, x_offset_baby=0.25, y_offset_baby=y_offset_baby)
    plt.show()

# Interactive slider
interact(
    move_baby,
    jump=FloatSlider(value=0, min=0, max=2, step=0.05, description='Jump'),
)


