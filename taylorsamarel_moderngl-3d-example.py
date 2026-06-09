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


# NFL FIELD WITH NUMBERS - FULLY WORKING VERSION
# ================================================================================
# STEP 1: INSTALL PACKAGES
# ================================================================================

!pip install -q moderngl==5.7.0
!pip install -q numpy>=1.24.0
!pip install -q Pillow>=10.0.0
!pip install -q imageio>=2.31.0
!pip install -q imageio-ffmpeg>=0.4.8
!pip install -q glcontext

# ================================================================================
# STEP 2: IMPORTS
# ================================================================================

import moderngl
import numpy as np
from PIL import Image
import math
from pathlib import Path
from IPython.display import Image as IPImage, display

print(f"âœ“ ModernGL version: {moderngl.__version__}")
print(f"âœ“ NumPy version: {np.__version__}")

# ================================================================================
# STEP 3: NFL FIELD CLASS
# ================================================================================

class NFLField:
    """NFL field with numbers drawn using lines"""
    
    VERTEX_SHADER = '''
    #version 330
    
    in vec3 in_position;
    in vec3 in_color;
    
    uniform mat4 mvp;
    
    out vec3 v_color;
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_color = in_color;
    }
    '''
    
    FRAGMENT_SHADER = '''
    #version 330
    
    in vec3 v_color;
    out vec4 f_color;
    
    void main() {
        f_color = vec4(v_color, 1.0);
    }
    '''
    
    def __init__(self):
        print("Building NFL field...")
        
        # Field dimensions
        self.field_width = 53.333
        self.half_width = self.field_width / 2
        self.sideline_width = 8
        
        # Create OpenGL context
        self.ctx = None
        for backend in ['egl', 'osmesa', None]:
            try:
                if backend:
                    self.ctx = moderngl.create_context(standalone=True, backend=backend)
                else:
                    self.ctx = moderngl.create_context(standalone=True)
                print(f"âœ“ OpenGL context ({backend or 'default'})")
                break
            except:
                continue
        
        if not self.ctx:
            raise RuntimeError("Cannot create OpenGL context")
        
        self.width, self.height = 1280, 720
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((self.width, self.height))
        )
        
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER
        )
        
        self.objects = []
        self._create_field()
        
        print("âœ“ Field complete")
    
    def _draw_digit(self, vertices, x, z, h, digit, size=1.5):
        """Draw a digit using line segments"""
        w = 0.15 * size
        
        segments = {
            '0': [
                [-1, 2, 1, 2],      # top
                [-1, -2, 1, -2],    # bottom
                [-1, 2, -1, -2],    # left
                [1, 2, 1, -2],      # right
            ],
            '1': [
                [0, 2, 0, -2],      # vertical center
            ],
            '2': [
                [-1, 2, 1, 2],      # top
                [1, 2, 1, 0],       # right top
                [-1, 0, 1, 0],      # middle
                [-1, 0, -1, -2],    # left bottom
                [-1, -2, 1, -2],    # bottom
            ],
            '3': [
                [-1, 2, 1, 2],      # top
                [1, 2, 1, -2],      # right
                [-1, 0, 1, 0],      # middle
                [-1, -2, 1, -2],    # bottom
            ],
            '4': [
                [-1, 2, -1, 0],     # left top
                [-1, 0, 1, 0],      # middle
                [1, 2, 1, -2],      # right
            ],
            '5': [
                [-1, 2, 1, 2],      # top
                [-1, 2, -1, 0],     # left top
                [-1, 0, 1, 0],      # middle
                [1, 0, 1, -2],      # right bottom
                [-1, -2, 1, -2],    # bottom
            ],
        }
        
        if digit in segments:
            for seg in segments[digit]:
                x1, z1, x2, z2 = seg
                x1 = x + x1 * size * 0.6
                x2 = x + x2 * size * 0.6
                z1 = z + z1 * size * 0.8
                z2 = z + z2 * size * 0.8
                
                if abs(x2 - x1) > abs(z2 - z1):  # Horizontal
                    vertices.extend([
                        [x1, h, z1 - w], [x2, h, z1 - w], [x2, h, z1 + w],
                        [x1, h, z1 - w], [x2, h, z1 + w], [x1, h, z1 + w],
                    ])
                else:  # Vertical
                    vertices.extend([
                        [x1 - w, h, z1], [x1 + w, h, z1], [x1 + w, h, z2],
                        [x1 - w, h, z1], [x1 + w, h, z2], [x1 - w, h, z2],
                    ])
    
    def _create_field(self):
        """Create all field components"""
        
        # DARK GREEN SIDELINE
        vertices = []
        colors = []
        h = -0.01
        
        sideline_color = [0, 0.25, 0]
        total_half = self.half_width + self.sideline_width
        
        vertices.extend([
            [-70, h, -total_half], [70, h, -total_half], [70, h, total_half],
            [-70, h, -total_half], [70, h, total_half], [-70, h, total_half],
        ])
        colors.extend([sideline_color] * 6)
        
        self._add_vao(vertices, colors)
        
        # GREEN FIELD
        vertices = []
        colors = []
        
        for i, x in enumerate(range(-50, 50, 5)):
            shade = [0, 0.38, 0] if i % 2 == 0 else [0, 0.42, 0]
            vertices.extend([
                [x, 0, -self.half_width], [x+5, 0, -self.half_width], [x+5, 0, self.half_width],
                [x, 0, -self.half_width], [x+5, 0, self.half_width], [x, 0, self.half_width],
            ])
            colors.extend([shade] * 6)
        
        # End zones
        endzone_color = [0, 0.33, 0]
        vertices.extend([
            [-60, 0, -self.half_width], [-50, 0, -self.half_width], [-50, 0, self.half_width],
            [-60, 0, -self.half_width], [-50, 0, self.half_width], [-60, 0, self.half_width],
            
            [50, 0, -self.half_width], [60, 0, -self.half_width], [60, 0, self.half_width],
            [50, 0, -self.half_width], [60, 0, self.half_width], [50, 0, self.half_width],
        ])
        colors.extend([endzone_color] * 12)
        
        self._add_vao(vertices, colors)
        
        # WHITE LINES
        vertices = []
        colors = []
        h = 0.01
        
        # Sidelines
        for side in [-self.half_width, self.half_width - 0.2]:
            vertices.extend([
                [-60, h, side], [60, h, side], [60, h, side + 0.2],
                [-60, h, side], [60, h, side + 0.2], [-60, h, side + 0.2],
            ])
            colors.extend([[1, 1, 1]] * 6)
        
        # End lines
        for x in [-60, 59.8]:
            vertices.extend([
                [x, h, -self.half_width], [x + 0.2, h, -self.half_width], 
                [x + 0.2, h, self.half_width], [x, h, -self.half_width], 
                [x + 0.2, h, self.half_width], [x, h, self.half_width],
            ])
            colors.extend([[1, 1, 1]] * 6)
        
        # Goal lines
        for x in [-50, 50]:
            vertices.extend([
                [x - 0.3, h, -self.half_width], [x + 0.3, h, -self.half_width],
                [x + 0.3, h, self.half_width], [x - 0.3, h, -self.half_width],
                [x + 0.3, h, self.half_width], [x - 0.3, h, self.half_width],
            ])
            colors.extend([[1, 1, 1]] * 6)
        
        # Yard lines
        for yard in range(-45, 50, 5):
            width = 0.15 if yard % 10 == 0 else 0.08
            vertices.extend([
                [yard - width, h, -self.half_width], [yard + width, h, -self.half_width],
                [yard + width, h, self.half_width], [yard - width, h, -self.half_width],
                [yard + width, h, self.half_width], [yard - width, h, self.half_width],
            ])
            colors.extend([[1, 1, 1]] * 6)
        
        self._add_vao(vertices, colors)
        
        # HASH MARKS
        vertices = []
        colors = []
        
        hash_pos = 11.79
        for yard in range(-50, 51):
            for z in [-hash_pos, hash_pos]:
                vertices.extend([
                    [yard - 0.05, h, z - 0.3], [yard + 0.05, h, z - 0.3],
                    [yard + 0.05, h, z + 0.3], [yard - 0.05, h, z - 0.3],
                    [yard + 0.05, h, z + 0.3], [yard - 0.05, h, z + 0.3],
                ])
                colors.extend([[1, 1, 1]] * 6)
        
        self._add_vao(vertices, colors)
        
        # NUMBERS
        vertices = []
        h = 0.005
        
        yard_nums = {
            -40: "10", -30: "20", -20: "30", -10: "40", 0: "50",
            10: "40", 20: "30", 30: "20", 40: "10"
        }
        
        for x_pos, num_str in yard_nums.items():
            # Top sideline
            z_top = -self.half_width - 4
            
            if num_str != "50":
                self._draw_digit(vertices, x_pos - 2, z_top, h, num_str[0])
                self._draw_digit(vertices, x_pos + 2, z_top, h, num_str[1])
            else:
                self._draw_digit(vertices, x_pos - 2, z_top, h, '5')
                self._draw_digit(vertices, x_pos + 2, z_top, h, '0')
            
            # Bottom sideline
            z_bottom = self.half_width + 4
            
            if num_str != "50":
                self._draw_digit(vertices, x_pos + 2, z_bottom, h, num_str[1])
                self._draw_digit(vertices, x_pos - 2, z_bottom, h, num_str[0])
            else:
                self._draw_digit(vertices, x_pos - 2, z_bottom, h, '0')
                self._draw_digit(vertices, x_pos + 2, z_bottom, h, '5')
        
        # FIX: Create colors array with correct size
        colors = [[1, 1, 1]] * len(vertices)  # One color per vertex
        self._add_vao(vertices, colors)
        
        # GOALPOSTS
        self._create_goalposts()
    
    def _create_goalposts(self):
        """Create goalposts"""
        vertices = []
        colors = []
        
        for x in [-60, 60]:
            # Base post
            for i in range(8):
                a1 = i * math.pi / 4
                a2 = (i + 1) * math.pi / 4
                
                for h in range(10):
                    x1 = x + 0.15 * math.cos(a1)
                    z1 = 0.15 * math.sin(a1)
                    x2 = x + 0.15 * math.cos(a2)
                    z2 = 0.15 * math.sin(a2)
                    
                    vertices.extend([
                        [x1, h, z1], [x2, h, z2], [x2, h+1, z2],
                        [x1, h, z1], [x2, h+1, z2], [x1, h+1, z1],
                    ])
                    colors.extend([[1, 0.9, 0]] * 6)
            
            # Crossbar
            crossbar_w = 6.167
            vertices.extend([
                [x - 0.15, 10, -crossbar_w], [x + 0.15, 10, -crossbar_w], 
                [x + 0.15, 10.4, -crossbar_w], [x - 0.15, 10, -crossbar_w], 
                [x + 0.15, 10.4, -crossbar_w], [x - 0.15, 10.4, -crossbar_w],
                
                [x - 0.15, 10, crossbar_w], [x + 0.15, 10, crossbar_w], 
                [x + 0.15, 10.4, crossbar_w], [x - 0.15, 10, crossbar_w], 
                [x + 0.15, 10.4, crossbar_w], [x - 0.15, 10.4, crossbar_w],
                
                [x - 0.15, 10, -crossbar_w], [x - 0.15, 10.4, -crossbar_w], 
                [x - 0.15, 10.4, crossbar_w], [x - 0.15, 10, -crossbar_w], 
                [x - 0.15, 10.4, crossbar_w], [x - 0.15, 10, crossbar_w],
                
                [x + 0.15, 10, -crossbar_w], [x + 0.15, 10.4, -crossbar_w], 
                [x + 0.15, 10.4, crossbar_w], [x + 0.15, 10, -crossbar_w], 
                [x + 0.15, 10.4, crossbar_w], [x + 0.15, 10, crossbar_w],
            ])
            colors.extend([[1, 0.9, 0]] * 24)
            
            # Uprights
            for z in [-crossbar_w, crossbar_w]:
                for h in range(10, 30):
                    vertices.extend([
                        [x - 0.1, h, z - 0.1], [x + 0.1, h, z - 0.1],
                        [x + 0.1, h+1, z - 0.1], [x - 0.1, h, z - 0.1],
                        [x + 0.1, h+1, z - 0.1], [x - 0.1, h+1, z - 0.1],
                        
                        [x - 0.1, h, z + 0.1], [x + 0.1, h, z + 0.1],
                        [x + 0.1, h+1, z + 0.1], [x - 0.1, h, z + 0.1],
                        [x + 0.1, h+1, z + 0.1], [x - 0.1, h+1, z + 0.1],
                    ])
                    colors.extend([[1, 0.9, 0]] * 12)
        
        self._add_vao(vertices, colors)
    
    def _add_vao(self, vertices, colors):
        """Create vertex array object"""
        if len(vertices) == 0:
            return
            
        vertices_np = np.array(vertices, dtype='f4')
        colors_np = np.array(colors, dtype='f4')
        
        vertex_data = np.hstack([vertices_np, colors_np])
        vbo = self.ctx.buffer(vertex_data.tobytes())
        
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '3f 3f', 'in_position', 'in_color')]
        )
        self.objects.append(vao)
    
    def render(self, angle, height=40):
        """Render the field"""
        self.fbo.use()
        self.ctx.clear(0.5, 0.7, 0.95, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        radius = 100
        cam_x = radius * math.cos(angle)
        cam_z = radius * math.sin(angle)
        
        eye = np.array([cam_x, height, cam_z], dtype='f4')
        target = np.array([0, 0, 0], dtype='f4')
        up = np.array([0, 1, 0], dtype='f4')
        
        z = eye - target
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        view = np.eye(4, dtype='f4')
        view[:3, 0] = x
        view[:3, 1] = y
        view[:3, 2] = z
        view[:3, 3] = eye
        view = np.linalg.inv(view)
        
        aspect = self.width / self.height
        fov = 60 * math.pi / 180
        near = 0.1
        far = 300.0
        f = 1 / math.tan(fov / 2)
        
        projection = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
            [0, 0, -1, 0]
        ], dtype='f4')
        
        mvp = (projection @ view).T
        self.program['mvp'].write(mvp.tobytes())
        
        for obj in self.objects:
            obj.render()
        
        pixels = self.fbo.read(components=3)
        img = Image.frombytes('RGB', (self.width, self.height), pixels)
        return img.transpose(Image.FLIP_TOP_BOTTOM)

# ================================================================================
# STEP 4: CREATE AND ANIMATE
# ================================================================================

print("\nCreating NFL field...")

try:
    field = NFLField()
    
    print("Rendering animation...")
    frames = []
    
    for i in range(120):
        angle = i * 0.015
        height = 45 + math.sin(i * 0.02) * 10
        frame = field.render(angle, height)
        frames.append(frame)
        
        if i % 40 == 0:
            print(f"  Frame {i}/120")
    
    print("âœ“ Complete")
    
    sample_path = Path("/kaggle/working/nfl_field.png")
    frames[40].save(sample_path)
    
    gif_path = Path("/kaggle/working/nfl_field.gif")
    frames[0].save(
        gif_path, 
        save_all=True, 
        append_images=frames[1::3],
        duration=100,
        loop=0
    )
    
    print("\nðŸ“¸ Sample:")
    display(IPImage(str(sample_path), width=900))
    
    print("\nðŸŽ¬ Animation:")
    display(IPImage(str(gif_path), width=900))
    
    print("\nâœ“ Files saved")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

