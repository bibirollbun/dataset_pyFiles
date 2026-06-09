import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from IPython.display import display, SVG as IPythonSVG


class SVGElement(ABC):
    """Abstract base class for all SVG elements."""
    
    @abstractmethod
    def to_xml(self) -> ET.Element:
        """Convert the element to an XML Element."""
        pass
    
    def add_common_attributes(self, element: ET.Element, attributes: Dict[str, str]) -> None:
        """Add common SVG attributes to an element if they exist in the attributes dictionary."""
        common_attrs = {
            'id', 'clip-path', 'clip-rule', 'color', 'color-interpolation',
            'color-interpolation-filters', 'color-rendering', 'display', 'fill',
            'fill-opacity', 'fill-rule', 'filter', 'flood-color', 'flood-opacity',
            'lighting-color', 'marker-end', 'marker-mid', 'marker-start', 'mask',
            'opacity', 'paint-order', 'stop-color', 'stop-opacity', 'stroke',
            'stroke-dasharray', 'stroke-dashoffset', 'stroke-linecap', 'stroke-linejoin',
            'stroke-miterlimit', 'stroke-opacity', 'stroke-width', 'transform'
        }
        
        for attr, value in attributes.items():
            if attr in common_attrs or attr in self.get_allowed_attributes():
                element.set(attr, str(value))
    
    @abstractmethod
    def get_allowed_attributes(self) -> set:
        """Return the set of allowed attributes for this element."""
        pass


class SVGShape(SVGElement):
    """Base class for SVG shape elements."""
    
    def __init__(self, **attributes):
        self.attributes = attributes
    
    def add_specific_attributes(self, element: ET.Element) -> None:
        """Add element-specific attributes."""
        allowed = self.get_allowed_attributes()
        for attr, value in self.attributes.items():
            if attr in allowed:
                element.set(attr, str(value))


class Circle(SVGShape):
    """Circle SVG element."""
    
    def __init__(self, cx: float, cy: float, r: float, **attributes):
        super().__init__(**attributes)
        self.cx = cx
        self.cy = cy
        self.r = r
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("circle")
        element.set("cx", str(self.cx))
        element.set("cy", str(self.cy))
        element.set("r", str(self.r))
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'cx', 'cy', 'r'}


class Rectangle(SVGShape):
    """Rectangle SVG element."""
    
    def __init__(self, x: float, y: float, width: float, height: float, rx: float = 0, ry: float = 0, **attributes):
        super().__init__(**attributes)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rx = rx
        self.ry = ry
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("rect")
        element.set("x", str(self.x))
        element.set("y", str(self.y))
        element.set("width", str(self.width))
        element.set("height", str(self.height))
        if self.rx > 0:
            element.set("rx", str(self.rx))
        if self.ry > 0:
            element.set("ry", str(self.ry))
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'x', 'y', 'width', 'height', 'rx', 'ry'}


class Ellipse(SVGShape):
    """Ellipse SVG element."""
    
    def __init__(self, cx: float, cy: float, rx: float, ry: float, **attributes):
        super().__init__(**attributes)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("ellipse")
        element.set("cx", str(self.cx))
        element.set("cy", str(self.cy))
        element.set("rx", str(self.rx))
        element.set("ry", str(self.ry))
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'cx', 'cy', 'rx', 'ry'}


class Line(SVGShape):
    """Line SVG element."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, **attributes):
        super().__init__(**attributes)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("line")
        element.set("x1", str(self.x1))
        element.set("y1", str(self.y1))
        element.set("x2", str(self.x2))
        element.set("y2", str(self.y2))
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'x1', 'y1', 'x2', 'y2'}


class Path(SVGShape):
    """Path SVG element."""
    
    def __init__(self, d: str, **attributes):
        super().__init__(**attributes)
        self.d = d
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("path")
        element.set("d", self.d)
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'d'}


class Polygon(SVGShape):
    """Polygon SVG element."""
    
    def __init__(self, points: List[tuple], **attributes):
        super().__init__(**attributes)
        self.points = points
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("polygon")
        points_str = " ".join(f"{x},{y}" for x, y in self.points)
        element.set("points", points_str)
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'points'}


class Polyline(SVGShape):
    """Polyline SVG element."""
    
    def __init__(self, points: List[tuple], **attributes):
        super().__init__(**attributes)
        self.points = points
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("polyline")
        points_str = " ".join(f"{x},{y}" for x, y in self.points)
        element.set("points", points_str)
        self.add_common_attributes(element, self.attributes)
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'points'}


# Helper factories for common SVG elements
class Group(SVGElement):
    """Group SVG element for organizing multiple elements."""
    
    def __init__(self, **attributes):
        self.attributes = attributes
        self.children = []
    
    def add(self, element: SVGElement) -> 'Group':
        """Add a child element to the group."""
        self.children.append(element)
        return self
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("g")
        self.add_common_attributes(element, self.attributes)
        
        for child in self.children:
            element.append(child.to_xml())
        
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'viewBox'}


class LinearGradient(SVGElement):
    """Linear Gradient SVG element."""
    
    def __init__(self, id: str, x1: float, y1: float, x2: float, y2: float, stops: List[Dict[str, Any]], **attributes):
        self.id = id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.stops = stops
        self.attributes = attributes
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("linearGradient")
        element.set("id", self.id)
        element.set("x1", str(self.x1))
        element.set("y1", str(self.y1))
        element.set("x2", str(self.x2))
        element.set("y2", str(self.y2))
        
        self.add_common_attributes(element, self.attributes)
        
        for stop_data in self.stops:
            stop = ET.SubElement(element, "stop")
            stop.set("offset", str(stop_data["offset"]))
            if "color" in stop_data:
                stop.set("stop-color", stop_data["color"])
            if "opacity" in stop_data:
                stop.set("stop-opacity", str(stop_data["opacity"]))
        
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'x1', 'x2', 'y1', 'y2', 'gradientUnits', 'gradientTransform', 'spreadMethod', 'href'}


class RadialGradient(SVGElement):
    """Radial Gradient SVG element."""
    
    def __init__(self, id: str, cx: float, cy: float, r: float, stops: List[Dict[str, Any]], fx: Optional[float] = None, fy: Optional[float] = None, **attributes):
        self.id = id
        self.cx = cx
        self.cy = cy
        self.r = r
        self.fx = fx
        self.fy = fy
        self.stops = stops
        self.attributes = attributes
    
    def to_xml(self) -> ET.Element:
        element = ET.Element("radialGradient")
        element.set("id", self.id)
        element.set("cx", str(self.cx))
        element.set("cy", str(self.cy))
        element.set("r", str(self.r))
        
        if self.fx is not None:
            element.set("fx", str(self.fx))
        if self.fy is not None:
            element.set("fy", str(self.fy))
        
        self.add_common_attributes(element, self.attributes)
        
        for stop_data in self.stops:
            stop = ET.SubElement(element, "stop")
            stop.set("offset", str(stop_data["offset"]))
            if "color" in stop_data:
                stop.set("stop-color", stop_data["color"])
            if "opacity" in stop_data:
                stop.set("stop-opacity", str(stop_data["opacity"]))
        
        return element
    
    def get_allowed_attributes(self) -> set:
        return {'cx', 'cy', 'r', 'fx', 'fy', 'fr', 'gradientUnits', 'gradientTransform', 'spreadMethod', 'href'}


class SVGContainer:
    """Main SVG container class."""
    
    def __init__(self, width: Union[int, str], height: Union[int, str], viewBox: Optional[str] = None):
        self.width = width
        self.height = height
        self.viewBox = viewBox
        self.elements = []
        self.defs = []
    
    def add(self, element: SVGElement) -> 'SVGContainer':
        """Add an element to the SVG."""
        self.elements.append(element)
        return self
    
    def add_def(self, element: SVGElement) -> 'SVGContainer':
        """Add a definition element to the SVG."""
        self.defs.append(element)
        return self
    
    def to_xml(self) -> ET.Element:
        """Convert the SVG to an XML Element."""
        root = ET.Element("svg")
        root.set("xmlns", "http://www.w3.org/2000/svg")
        root.set("width", str(self.width))
        root.set("height", str(self.height))
        
        if self.viewBox:
            root.set("viewBox", self.viewBox)
        
        # Add defs if present
        if self.defs:
            defs = ET.SubElement(root, "defs")
            for def_element in self.defs:
                defs.append(def_element.to_xml())
        
        # Add elements
        for element in self.elements:
            root.append(element.to_xml())
        
        return root
    
    def to_string(self, minimize: bool = True) -> str:
        """Convert the SVG to a string representation.
        
        Args:
            minimize: If True, minimize the SVG by removing unnecessary whitespace.
        """
        root = self.to_xml()
        
        if minimize:
            # Custom minimization to reduce size
            xml_str = ET.tostring(root, encoding='unicode')
            # Remove unnecessary whitespace
            xml_str = xml_str.replace('\n', '').replace('\t', '').replace('  ', '')
            return xml_str
        else:
            # Pretty-printed XML
            from xml.dom import minidom
            xml_str = ET.tostring(root, encoding='unicode')
            return minidom.parseString(xml_str).toprettyxml(indent="  ")


class SVGExamples:
    @staticmethod
    def create_simple_shapes_example():
        """Create an SVG with multiple simple shapes."""
        svg = SVGContainer(200, 200, "0 0 200 200")
        
        svg.add(Circle(50, 50, 40, fill="#6CA0DC"))  # Sky blue
        svg.add(Rectangle(100, 20, 80, 60, fill="#F88379"))  # Coral
        svg.add(Ellipse(150, 120, 40, 20, fill="#8A9A5B"))  # Sage green
        svg.add(Line(10, 100, 190, 180, stroke="#E6C200", stroke_width="3"))  # Soft gold
        
        return svg.to_string(minimize=True)
    
    @staticmethod
    def create_gradient_example():
        """Create an SVG with gradient fills."""
        svg = SVGContainer(200, 200, "0 0 200 200")
        linear_gradient = LinearGradient(
            id="gradient1", x1=0, y1=0, x2=1, y2=0,
            stops=[
                {"offset": "0%", "color": "#5B8FB9"},  # Soft blue
                {"offset": "100%", "color": "#B19CD9"}  # Soft purple
            ]
        )
        svg.add_def(linear_gradient)
        svg.add(Rectangle(20, 20, 160, 80, fill="url(#gradient1)"))
        radial_gradient = RadialGradient(
            id="gradient2", cx=0.5, cy=0.5, r=0.5,
            stops=[
                {"offset": "0%", "color": "#78C091"},  # Soft green
                {"offset": "100%", "color": "#F5B8D1"}  # Soft pink
            ]
        )
        svg.add_def(radial_gradient)
        svg.add(Circle(100, 150, 40, fill="url(#gradient2)"))
        return svg.to_string(minimize=True)
    
    @staticmethod
    def create_path_example():
        """Create an SVG with path elements."""
        svg = SVGContainer(200, 200, "0 0 200 200")
        svg.add(Path("M 10,10 L 100,10 L 50,90 Z", fill="#9B7CB9"))  # Lavender
        star_path = "M 100,10 L 113,50 L 156,50 L 122,76 L 135,120 L 100,95 L 65,120 L 78,76 L 44,50 L 87,50 Z"
        svg.add(Path(star_path, fill="#E1C16E", stroke="#8B7355", stroke_width="1"))  # Gold with tan stroke
        
        return svg.to_string(minimize=True)
    
    @staticmethod
    def create_group_example():
        """Create an SVG with grouped elements."""
        svg = SVGContainer(200, 200, "0 0 200 200")
        group = Group(transform="translate(100, 100) rotate(45)")
        group.add(Rectangle(-25, -25, 50, 50, fill="#7393B3"))  # Blue-gray
        group.add(Circle(0, 0, 30, fill="#E9967A", fill_opacity="0.5"))  # Salmon
        svg.add(group)
        
        return svg.to_string(minimize=True)
    
    @staticmethod
    def create_complex_svg_example():
        """Create a more complex SVG with multiple objects and optimizations."""
        svg = SVGContainer(400, 300, "0 0 400 300")
        svg.add(Rectangle(0, 0, 400, 300, fill="#f0f0f0"))  # Background
        gradient = LinearGradient(
            id="sunset", x1=0, y1=0, x2=0, y2=1,
            stops=[
                {"offset": "0%", "color": "#FFBE7D"},
                {"offset": "50%", "color": "#FF9AA2"},
                {"offset": "100%", "color": "#7F7EFF"}
            ]
        )
        svg.add_def(gradient)
        svg.add(Rectangle(0, 0, 400, 200, fill="url(#sunset)"))
        svg.add(Circle(320, 60, 30, fill="#FFECB3", stroke="#E8A87C", stroke_width="5"))
        svg.add(Path("M 0,200 L 150,100 L 200,150 L 300,90 L 400,200 Z", fill="#8D9DB6"))
        svg.add(Path("M 0,200 C 100,220 300,210 400,200 L 400,300 L 0,300 Z", fill="#A4D8E8", fill_opacity="0.7"))
        tree_group = Group()
        for x in [50, 80, 120, 270, 330]:
            tree_group.add(Rectangle(x-5, 180, 10, 30, fill="#8B7355"))
            tree_group.add(Path(f"M {x-20},180 L {x+20},180 L {x},150 Z", fill="#6B8E23"))
            tree_group.add(Path(f"M {x-15},160 L {x+15},160 L {x},130 Z", fill="#7BA05B"))
        
        svg.add(tree_group)
        return svg.to_string(minimize=True)


class SVGExampleTest:
    @staticmethod
    def validate_svg_size(svg_string, max_size=10000):
        """Validate that the SVG is within the size constraints."""
        size = len(svg_string.encode('utf-8'))
        print(size)
        if size > max_size:
            raise ValueError(f"SVG size ({size} bytes) exceeds maximum allowed size ({max_size} bytes)")
        return True

    @staticmethod
    def display_all_svgs():
        """Generate and display all SVG examples."""
        examples = [
            ("Simple Shapes Example", SVGExamples.create_simple_shapes_example()),
            ("Gradient Example", SVGExamples.create_gradient_example()),
            ("Path Example", SVGExamples.create_path_example()),
            ("Group Example", SVGExamples.create_group_example()),
            ("Complex SVG Example", SVGExamples.create_complex_svg_example()),
        ]
        
        for title, svg_string in examples:
            SVGExampleTest.display_svg_in_notebook(title, svg_string)
    
    @staticmethod
    def display_svg_in_notebook(title, svg_string):
        from IPython.display import display, HTML, SVG

        print(f"\n{title}")
        print("-" * len(title))
        
        # Display the SVG
        display(SVG(data=svg_string))
        
        # Optionally, display the size
        size = len(svg_string.encode('utf-8'))
        print(f"Size: {size} bytes")

# Run the examples
SVGExampleTest.display_all_svgs()



class ToolRegistry:
    """
    Registry managing tool definitions and handlers using a decorator pattern.
    """
    def __init__(self):
        self.tools_definitions = []
        self.handlers = {}
    
    def register(self, name: str, description: str, input_schema: Dict[str, Any]):
        """
        Decorator to register a function as a tool handler.
        """
        def decorator(handler_func):
            self.tools_definitions.append({
                "name": name,
                "description": description,
                "input_schema": input_schema
            })
            self.handlers[name] = handler_func
            return handler_func
        return decorator
    
    def get_tools(self):
        return self.tools_definitions
    
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any], tool_use_id: str):
        try:
            if tool_name in self.handlers:
                return self.handlers[tool_name](self, tool_input)
            else:
                error_msg = f"Unknown tool: {tool_name}"
                logging.error(error_msg)
                return [{"type": "text", "text": f"Error: {error_msg}"}]
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logging.error(error_msg)
            return [{"type": "text", "text": f"Error: {e}"}]


def setup_tool_registry() -> ToolRegistry:
    """Properly sets up the ToolRegistry with SVG creation tools."""
    registry = ToolRegistry()
    
    @registry.register(
        name="create_svg",
        description="Initialize a new SVG document with dimensions.",
        input_schema={
            "type": "object",
            "properties": {
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "viewBox": {"type": "string"},
                "attributes": {"type": "object"}
            },
            "required": ["width", "height"]
        }
    )
    def handle_create_svg(registry: ToolRegistry, tool_input: Dict[str, Any]):
        width, height = tool_input["width"], tool_input["height"]
        viewBox = tool_input.get("viewBox")
        attributes = tool_input.get("attributes", {})
        
        # SVGContainer Object Creation
        svg_obj = SVGContainer(width, height, viewBox)
        
        # SVG String Convert
        svg_str = svg_obj.to_string()
        
        return [
            {"type": "text", "text": f"Created SVG {width}x{height}"},
            {"type": "svg", "content": svg_str}
        ]
   
    @registry.register(
        name="add_rectangle",
        description="Add rectangle to existing SVG.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"},
                "attributes": {"type": "object"},
            },
            "required": ["svg", "x", "y", "width", "height"]
        }
    )
    def handle_add_rectangle(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        x, y = tool_input["x"], tool_input["y"]
        width, height = tool_input["width"], tool_input["height"]
        attributes = tool_input.get("attributes", {})
        root = ET.fromstring(svg_content)
        rect = Rectangle(x, y, width, height, **attributes).to_xml()
        root.append(rect)
        updated_svg = ET.tostring(root, encoding='unicode')
        return [
            {"type": "text", "text": "Added rectangle"},
            {"type": "svg", "content": updated_svg}
        ]
    
    @registry.register(
        name="add_circle",
        description="Add circle to existing SVG.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
                "cx": {"type": "number"},
                "cy": {"type": "number"},
                "r": {"type": "number"},
                "attributes": {"type": "object"},
            },
            "required": ["svg", "cx", "cy", "r"]
        }
    )
    def handle_add_circle(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        cx, cy, r = tool_input["cx"], tool_input["cy"], tool_input["r"]
        attributes = tool_input.get("attributes", {})
        root = ET.fromstring(svg_content)
        circle = Circle(cx, cy, r, **attributes).to_xml()
        root.append(circle)
        updated_svg = ET.tostring(root, encoding='unicode')
        return [
            {"type": "text", "text": "Added circle"},
            {"type": "svg", "content": updated_svg}
        ]
    
    @registry.register(
        name="add_path",
        description="Add path to existing SVG.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
                "d": {"type": "string"},
                "attributes": {"type": "object"},
            },
            "required": ["svg", "d"]
        }
    )
    def handle_add_path(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        d = tool_input["d"]
        attributes = tool_input.get("attributes", {})
        root = ET.fromstring(svg_content)
        path = Path(d, **attributes).to_xml()
        root.append(path)
        updated_svg = ET.tostring(root, encoding='unicode')
        return [
            {"type": "text", "text": "Added path"},
            {"type": "svg", "content": updated_svg}
        ]
    
    @registry.register(
        name="create_group",
        description="Create a group in existing SVG.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
                "attributes": {"type": "object"},
            },
            "required": ["svg"]
        }
    )
    def handle_create_group(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        attributes = tool_input.get("attributes", {})
        root = ET.fromstring(svg_content)
        group = Group(**attributes).to_xml()
        root.append(group)
        updated_svg = ET.tostring(root, encoding='unicode')
        return [
            {"type": "text", "text": "Created group"},
            {"type": "svg", "content": updated_svg}
        ]
    
    @registry.register(
        name="add_linear_gradient",
        description="Add a linear gradient to existing SVG.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
                "id": {"type": "string"},
                "x1": {"type": "number"},
                "y1": {"type": "number"},
                "x2": {"type": "number"},
                "y2": {"type": "number"},
                "stops": {"type": "array"},
                "attributes": {"type": "object"}
            },
            "required": ["svg", "id", "stops"]
        }
    )
    def handle_add_linear_gradient(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        grad_id = tool_input["id"]
        x1 = tool_input.get("x1", 0)
        y1 = tool_input.get("y1", 0)
        x2 = tool_input.get("x2", 1)
        y2 = tool_input.get("y2", 0)
        stops = tool_input["stops"]
        attributes = tool_input.get("attributes", {})
        
        root = ET.fromstring(svg_content)
        
        # Check if defs element exists, create if not
        defs_elem = root.find("defs")
        if defs_elem is None:
            defs_elem = ET.Element("defs")
            root.insert(0, defs_elem)
        
        # Create gradient element
        gradient = LinearGradient(grad_id, x1, y1, x2, y2, stops, **attributes).to_xml()
        defs_elem.append(gradient)
        
        updated_svg = ET.tostring(root, encoding='unicode')
        
        return [
            {"type": "text", "text": f"Added linear gradient with id '{grad_id}'"},
            {"type": "svg", "content": updated_svg}
        ]
    
    @registry.register(
        name="add_radial_gradient",
        description="Add a radial gradient to existing SVG.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
                "id": {"type": "string"},
                "cx": {"type": "number"},
                "cy": {"type": "number"},
                "r": {"type": "number"},
                "fx": {"type": "number"},
                "fy": {"type": "number"},
                "stops": {"type": "array"},
                "attributes": {"type": "object"}
            },
            "required": ["svg", "id", "cx", "cy", "r", "stops"]
        }
    )
    def handle_add_radial_gradient(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        grad_id = tool_input["id"]
        cx = tool_input["cx"]
        cy = tool_input["cy"]
        r = tool_input["r"]
        fx = tool_input.get("fx")
        fy = tool_input.get("fy")
        stops = tool_input["stops"]
        attributes = tool_input.get("attributes", {})
        
        root = ET.fromstring(svg_content)
        
        # Check if defs element exists, create if not
        defs_elem = root.find("defs")
        if defs_elem is None:
            defs_elem = ET.Element("defs")
            root.insert(0, defs_elem)
        
        # Create gradient element
        gradient = RadialGradient(grad_id, cx, cy, r, stops, fx, fy, **attributes).to_xml()
        defs_elem.append(gradient)
        
        updated_svg = ET.tostring(root, encoding='unicode')
        
        return [
            {"type": "text", "text": f"Added radial gradient with id '{grad_id}'"},
            {"type": "svg", "content": updated_svg}
        ]
    
    @registry.register(
        name="fix_svg_namespace",
        description="Fix namespace prefixes in SVG for cleaner output.",
        input_schema={
            "type": "object",
            "properties": {
                "svg": {"type": "string"}
            },
            "required": ["svg"]
        }
    )
    def handle_fix_svg_namespace(registry, tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        svg_content = tool_input["svg"]
        root = ET.fromstring(svg_content)
        
        # Fix namespaces by removing prefixes
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        
        # Ensure SVG has proper namespace
        root.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Remove any ns0 attributes
        for attr in list(root.attrib.keys()):
            if attr.startswith('xmlns:ns'):
                del root.attrib[attr]
        
        updated_svg = ET.tostring(root, encoding='unicode')
        
        return [
            {"type": "text", "text": "Fixed SVG namespace"},
            {"type": "svg", "content": updated_svg}
        ]
    
    return registry


class ToolExamples:
    @staticmethod
    def create_shapes_svg(registry):
        """Generate an SVG with simple shapes using the ToolRegistry."""
        # Debug output
        print("Creating simple shapes SVG...")
        
        create_input = {"width": 400, "height": 300}
        result = registry.execute_tool("create_svg", create_input, "simple_shapes")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        if not svg_content:
            print("Failed to extract SVG content from result:", result)
            raise ValueError("Failed to create base SVG")
        
        rect_input = {
            "svg": svg_content,
            "x": 50,
            "y": 50,
            "width": 100,
            "height": 80,
            "attributes": {"fill": "blue", "stroke": "black", "stroke-width": "2"}
        }
        result = registry.execute_tool("add_rectangle", rect_input, "blue_rect")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        circle_input = {
            "svg": svg_content,
            "cx": 250,
            "cy": 100,
            "r": 50,
            "attributes": {"fill": "red", "stroke": "black", "stroke-width": "2"}
        }
        result = registry.execute_tool("add_circle", circle_input, "red_circle")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)

        fix_input = {"svg": svg_content}
        result = registry.execute_tool("fix_svg_namespace", fix_input, "fix_namespace")
        return next((block["content"] for block in result if block["type"] == "svg"), None)

    @staticmethod
    def create_linear_gradient_svg(registry):
        """Generate an SVG with linear gradient fills using the ToolRegistry."""
        create_input = {"width": 400, "height": 300}
        result = registry.execute_tool("create_svg", create_input, "gradient_example")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        if not svg_content:
            raise ValueError("Failed to create base SVG")
        
        gradient_input = {
            "svg": svg_content,
            "id": "linearGrad",
            "x1": 0,
            "y1": 0,
            "x2": 1,
            "y2": 1,
            "stops": [
                {"offset": "0%", "color": "#ff0000", "opacity": 1},
                {"offset": "100%", "color": "#0000ff", "opacity": 1}
            ]
        }
        result = registry.execute_tool("add_linear_gradient", gradient_input, "linear_gradient")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        rect_input = {
            "svg": svg_content,
            "x": 50,
            "y": 50,
            "width": 300,
            "height": 200,
            "attributes": {"fill": "url(#linearGrad)", "stroke": "black", "stroke-width": "3"}
        }
        result = registry.execute_tool("add_rectangle", rect_input, "gradient_rect")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        fix_input = {"svg": svg_content}
        result = registry.execute_tool("fix_svg_namespace", fix_input, "fix_namespace")
        return next((block["content"] for block in result if block["type"] == "svg"), None)

    @staticmethod
    def create_radial_gradient_svg(registry):
        """Generate an SVG with radial gradient fill using the ToolRegistry."""
        create_input = {"width": 400, "height": 300}
        result = registry.execute_tool("create_svg", create_input, "radial_gradient_example")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        if not svg_content:
            raise ValueError("Failed to create base SVG")
        
        gradient_input = {
            "svg": svg_content,
            "id": "radialGrad",
            "cx": 0.5,
            "cy": 0.5,
            "r": 0.5,
            "fx": 0.25,
            "fy": 0.25,
            "stops": [
                {"offset": "0%", "color": "#ffff00", "opacity": 1},
                {"offset": "100%", "color": "#ff00ff", "opacity": 1}
            ]
        }
        result = registry.execute_tool("add_radial_gradient", gradient_input, "radial_gradient")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        circle_input = {
            "svg": svg_content,
            "cx": 200,
            "cy": 150,
            "r": 100,
            "attributes": {"fill": "url(#radialGrad)", "stroke": "black", "stroke-width": "3"}
        }
        result = registry.execute_tool("add_circle", circle_input, "gradient_circle")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        fix_input = {"svg": svg_content}
        result = registry.execute_tool("fix_svg_namespace", fix_input, "fix_namespace")
        return next((block["content"] for block in result if block["type"] == "svg"), None)
        
    @staticmethod
    def create_path_svg(registry):
        """Generate an SVG with a path element using the ToolRegistry."""
        # Create base SVG
        create_input = {"width": 400, "height": 300}
        result = registry.execute_tool("create_svg", create_input, "path_example")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        if not svg_content:
            raise ValueError("Failed to create base SVG")
        
        # Add a path drawing a heart shape
        path_input = {
            "svg": svg_content,
            "d": "M 200,100 C 150,60 100,100 100,150 C 100,200 150,250 200,280 C 250,250 300,200 300,150 C 300,100 250,60 200,100 Z",
            "attributes": {"fill": "pink", "stroke": "red", "stroke-width": "3"}
        }
        result = registry.execute_tool("add_path", path_input, "heart_path")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        # Fix namespace issues
        fix_input = {"svg": svg_content}
        result = registry.execute_tool("fix_svg_namespace", fix_input, "fix_namespace")
        return next((block["content"] for block in result if block["type"] == "svg"), None)

    @staticmethod
    def create_complex_svg(registry):
        """Generate a more complex SVG with various elements using the ToolRegistry."""
        # Create base SVG
        create_input = {"width": 500, "height": 400}
        result = registry.execute_tool("create_svg", create_input, "complex_example")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        if not svg_content:
            raise ValueError("Failed to create base SVG")
        
        # Add background
        rect_input = {
            "svg": svg_content,
            "x": 0,
            "y": 0,
            "width": 500,
            "height": 400,
            "attributes": {"fill": "#f0f0f0"}
        }
        result = registry.execute_tool("add_rectangle", rect_input, "background")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        # Add sun (circle)
        circle_input = {
            "svg": svg_content,
            "cx": 400,
            "cy": 80,
            "r": 40,
            "attributes": {"fill": "yellow", "stroke": "orange", "stroke-width": "2"}
        }
        result = registry.execute_tool("add_circle", circle_input, "sun")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        # Add mountains (path)
        path_input = {
            "svg": svg_content,
            "d": "M 0,400 L 0,250 L 100,150 L 200,250 L 300,200 L 400,300 L 500,200 L 500,400 Z",
            "attributes": {"fill": "#6b8e23"}
        }
        result = registry.execute_tool("add_path", path_input, "mountains")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        # Add water (path)
        path_input = {
            "svg": svg_content,
            "d": "M 0,350 C 100,320 200,380 300,340 C 400,300 500,360 500,350 L 500,400 L 0,400 Z",
            "attributes": {"fill": "#4682b4"}
        }
        result = registry.execute_tool("add_path", path_input, "water")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        # Create group for boat
        group_input = {
            "svg": svg_content,
            "attributes": {"transform": "translate(150,320) scale(0.5)"}
        }
        result = registry.execute_tool("create_group", group_input, "boat_group")
        svg_content = next((block["content"] for block in result if block["type"] == "svg"), None)
        
        # Fix namespace issues
        fix_input = {"svg": svg_content}
        result = registry.execute_tool("fix_svg_namespace", fix_input, "fix_namespace")
        return next((block["content"] for block in result if block["type"] == "svg"), None)


class ToolExamplesTest:
    @staticmethod
    def validate_svg_size(svg_string, max_size=10000):
        """Validate that the SVG is within the size constraints."""
        size = len(svg_string.encode('utf-8'))
        print(f"SVG size: {size} bytes")
        if size > max_size:
            raise ValueError(f"SVG size ({size} bytes) exceeds maximum allowed size ({max_size} bytes)")
        return True

    @staticmethod
    def display_svg_in_notebook(title, svg_string):
        """Display SVG in a Jupyter notebook with title and size information."""
        try:
            from IPython.display import display, SVG
            
            print(f"\n{title}")
            print("-" * len(title))
            
            # Display the SVG
            display(SVG(data=svg_string))
            
            # Display the size
            size = len(svg_string.encode('utf-8'))
            print(f"Size: {size} bytes")
            
            # Validate SVG is well-formed
            ET.fromstring(svg_string)
            print("SVG is valid XML.")
        
        except ImportError:
            print(f"\n{title}")
            print("-" * len(title))
            print("IPython not available. SVG can't be displayed in this environment.")
            print(f"SVG content (first 100 chars): {svg_string[:100]}...")
            print(f"Size: {len(svg_string.encode('utf-8'))} bytes")

    @staticmethod
    def run_svg_gallery():
        """Runs a gallery of SVG examples in a Jupyter notebook."""
        # Setup the registry
        print("Setting up tool registry...")
        registry = setup_tool_registry()
        
        # Debug: display registered tools
        print("Available tools:", [tool["name"] for tool in registry.get_tools()])
        
        # Simple shapes example using ToolRegistry
        simple_shapes_svg = ToolExamples.create_shapes_svg(registry)
        ToolExamplesTest.display_svg_in_notebook("Simple Shapes Example", simple_shapes_svg)
        
        # Linear gradient example
        linear_gradient_svg = ToolExamples.create_linear_gradient_svg(registry)
        ToolExamplesTest.display_svg_in_notebook("Linear Gradient Example", linear_gradient_svg)
        
        # Radial gradient example
        radial_gradient_svg = ToolExamples.create_radial_gradient_svg(registry)
        ToolExamplesTest.display_svg_in_notebook("Radial Gradient Example", radial_gradient_svg)
        
        # Path example
        path_svg = ToolExamples.create_path_svg(registry)
        ToolExamplesTest.display_svg_in_notebook("Path Example", path_svg)
        
        # Complex example
        complex_svg = ToolExamples.create_complex_svg(registry)
        ToolExamplesTest.display_svg_in_notebook("Complex SVG Example", complex_svg)


ToolExamplesTest.run_svg_gallery()


from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
from typing_extensions import TypedDict
import json
from pprint import pprint


user_secrets = UserSecretsClient()
genai.configure(api_key=user_secrets.get_secret("GEMINI_API_KEY"))


model = genai.GenerativeModel('models/gemini-2.0-flash-001')


class HelloWorldResponse(TypedDict):
    short_reason: str
    greeting: str

def get_hello_world(model, num_neighbors=5):
    prompt = f"""
    - Generate {num_neighbors} "Hello World" variations in JSON format
    - Make it a proper "Hello World" greeting
    - Each example should show how to print "Hello World" in a different language
    """
    
    response = model.generate_content([prompt], generation_config=
                          genai.GenerationConfig(response_mime_type='application/json',
                                                 response_schema=list[HelloWorldResponse]))

    return json.loads(response.text)


results = get_hello_world(model)
pprint(results)







