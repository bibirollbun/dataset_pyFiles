"""
Competition helper module for the Drawing With LLMs Kaggle Competition.

This module includes:
  - Helper functions to test your model locally using a dummy test set.
  - SVG validation logic to ensure that generated SVG code meets competition constraints.
  - A gateway class that simulates the inference loop used in Kaggle’s scoring system.

---------------------------------------------------------------
Package Notebook Instructions:
---------------------------------------------------------------
To participate in this competition, your Notebook must be a Kaggle Package Notebook.
Ensure the following:
1. **Model Class**: Your Notebook must define a class named `Model` with a method `predict(description: str) -> str`
   that takes a text description and returns valid SVG code.
   For example:
   
       class Model:
           def __init__(self):
               # Load models, initialize parameters, etc.
               pass
           
           def predict(self, description: str) -> str:
               # Your SVG generation logic here.
               svg_code = "<svg> ... </svg>"
               return svg_code

2. **Package Metadata**: Include proper metadata (e.g. in the Notebook settings) so that Kaggle
   recognizes your submission as a package. Fork one of the official starter notebooks to see an example.

3. **Dependencies**: Specify all necessary dependencies in your Notebook. Note that Internet access
   is disabled during scoring, so all dependencies must be included in the environment.

4. **Committing the Notebook**: 
   - Click the "Save Version" (or "Commit Version") button in the Kaggle Notebook editor.
   - Make sure the Notebook is set as a Package Notebook (this is usually pre-configured in the starter notebooks).
   - Write a descriptive commit message.
   - Once committed, the "Submit" button will become active.

5. **Submission**:
   - After committing your Notebook, click the "Submit" button.
   - Kaggle’s system will install your package in a container (with Internet disabled) and call your Model.predict()
     function on the hidden test set.
   - Your submission is then scored using the CLIP similarity metric (after converting your SVGs to PNGs).

Refer to the official starter notebooks ("Official Starter Notebook" and "Getting Started with Gemma 2 Notebook")
and Kaggle’s documentation for additional guidance.
---------------------------------------------------------------
"""

import pathlib
import sys
import inspect
from tqdm import tqdm  # progress bar library

# ---------------------------------------------------------------------------
# Path Handling and Proto Files Check
# ---------------------------------------------------------------------------
# Use __file__ if available; otherwise, fallback to the current working directory.
try:
    module_path = pathlib.Path(__file__).parent
except NameError:
    module_path = pathlib.Path.cwd()

gen_path = module_path / 'core' / 'generated'
if not (gen_path / 'kaggle_evaluation_pb2.py').exists():
    print('Warning: Missing required kaggle_evaluation proto / gRPC generated files. '
          'Bypassing check for local testing.')
else:
    sys.path.append(str(module_path))
    sys.path.append(str(gen_path))

# ---------------------------------------------------------------------------
# Add the directory containing svg.py to sys.path
# ---------------------------------------------------------------------------
# Adjust this path as necessary to point to the location of your svg.py file.
svg_directory = pathlib.Path('/kaggle/input/drawing-with-llms/kaggle_evaluation')
if str(svg_directory) not in sys.path:
    sys.path.insert(0, str(svg_directory))

# ---------------------------------------------------------------------------
# Import svg Module Functions
# ---------------------------------------------------------------------------
# When running as part of a package, use relative import; otherwise, try an absolute import.
if __package__:
    from .svg import test, _run_gateway, _run_inference_server  # noqa: F401
else:
    try:
        from svg import test, _run_gateway, _run_inference_server
    except ImportError as e:
        print("Warning: Could not import svg module. Ensure that '/kaggle/input/drawing-with-llms/kaggle_evaluation/svg.py' exists.")
        raise e

__all__ = ['test']
__version__ = '0.5.0'

# ---------------------------------------------------------------------------
# Competition Inference & Testing Functions
# ---------------------------------------------------------------------------
from kaggle_evaluation.core import relay, templates
from kaggle_evaluation.svg_gateway import SVGGateway

def test(model_cls: type, data_path: str | pathlib.Path | None = None) -> None:
    '''Tests this competition's inference loop over the given Model class.
    
    The provided Model class should have a `predict` function which accepts input(s)
    and returns output(s) with the shapes and types required by this competition.
    This function performs best-effort validation of this by running an inference
    loop with a dummy test set over Model.predict.
    By default the test set is taken from the `kaggle_evaluation` directory, but you
    may override to another directory with the same test file structure via the
    `data_path` arg.
    '''
    print('Creating Model instance...')
    model = model_cls()
    if not hasattr(model, 'predict') or not inspect.ismethod(model.predict):
        msg = f'Model does not have method predict.'
        raise ValueError(msg)

    print('Running inference tests...')
    server = relay.define_server(model.predict)
    server.start()
    try:
        gateway = SVGGateway(data_path)
        submission_path = gateway.run()
        print(f'Wrote test submission file to "{str(submission_path)}".')
    except Exception as err:
        raise err from None
    finally:
        server.stop(0)
    print('Success!')

def _run_gateway() -> None:
    '''Internal function for running the Gateway during a Kaggle scoring session.
    
    Starts a scoring session which assumes existence of an Inference Server to return
    inferences over the test set.
    '''
    gateway = SVGGateway()
    gateway.run()

def _run_inference_server(module: type) -> None:
    '''Internal function for running the Inference Server during a Kaggle scoring session.
    
    Takes the user's submitted, imported module and sets up the inference server exposing
    their required method(s).
    '''
    model = module.Model()
    server = templates.InferenceServer(model.predict)
    server.serve()

# ---------------------------------------------------------------------------
# SVG Constraints and Validation
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from defusedxml import ElementTree as etree

@dataclass(frozen=True)
class SVGConstraints:
    """Defines constraints for validating SVG documents.

    Attributes
    ----------
    max_svg_size : int, default=10000
        Maximum allowed size of an SVG file in bytes.
    allowed_elements : dict[str, set[str]]
        Mapping of the allowed elements to the allowed attributes of each element.
    """
    max_svg_size: int = 10000
    allowed_elements: dict[str, set[str]] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
            },
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feColorMatrix': {'result', 'in', 'type', 'values'},
            'feComposite': {
                'result',
                'style',
                'in',
                'in2',
                'operator',
                'k1',
                'k2',
                'k3',
                'k4',
            },
            'feFlood': {'result'},
            'feGaussianBlur': {
                'result',
                'in',
                'stdDeviation',
                'edgeMode',
            },
            'feMerge': {
                'result',
                'x',
                'y',
                'width',
                'height',
                'result',
            },
            'feMergeNode': {'result', 'in'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'feTurbulence': {
                'result',
                'baseFrequency',
                'numOctaves',
                'seed',
                'stitchTiles',
                'type',
            },
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
    )

    def validate_svg(self, svg_code: str) -> None:
        """Validates an SVG string against a set of predefined constraints.

        Parameters
        ----------
        svg_code : str
            The SVG string to validate.

        Raises
        ------
        ValueError
            If the SVG violates any of the defined constraints.
        """
        # Check file size
        if len(svg_code.encode('utf-8')) > self.max_svg_size:
            raise ValueError('SVG exceeds allowed size')

        # Parse XML securely
        tree = etree.fromstring(
            svg_code.encode('utf-8'),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )

        elements = set(self.allowed_elements.keys())

        # Check elements and attributes
        for element in tree.iter():
            # Extract tag name (ignoring XML namespaces)
            tag_name = element.tag.split('}')[-1]
            if tag_name not in elements:
                raise ValueError(f'Disallowed element: {tag_name}')

            # Check attributes
            for attr, attr_value in element.attrib.items():
                # Normalize attribute name
                attr_name = attr.split('}')[-1]
                if (
                    attr_name not in self.allowed_elements[tag_name]
                    and attr_name not in self.allowed_elements['common']
                ):
                    raise ValueError(f'Disallowed attribute: {attr_name}')

                # Disallow embedded data
                if 'data:' in attr_value.lower():
                    raise ValueError('Embedded data not allowed')
                if ';base64' in attr_value:
                    raise ValueError('Base64 encoded content not allowed')

                # Ensure href attributes reference internal IDs only
                if attr_name == 'href':
                    if not attr_value.startswith('#'):
                        raise ValueError(
                            f'Invalid href attribute in <{tag_name}>. Only internal references (starting with "#") are allowed. Found: "{attr_value}"'
                        )

# ---------------------------------------------------------------------------
# Gateway Notebook for SVG Image Generation with Progress Bar
# ---------------------------------------------------------------------------
import os
import tempfile
from typing import Any
import pandas as pd
import polars as pl

from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType, IS_RERUN
import kaggle_evaluation.core.templates
from kaggle_evaluation.svg_constraints import SVGConstraints

class SVGGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_path: str | pathlib.Path | None = None):
        super().__init__(target_column_name='svg')
        self.set_response_timeout_seconds(60 * 5)
        self.row_id_column_name = 'id'
        # When no data_path is provided, default to the directory of this file.
        self.data_path: pathlib.Path = pathlib.Path(data_path) if data_path else pathlib.Path(__file__).parent
        self.constraints: SVGConstraints = SVGConstraints()

    def generate_data_batches(self):
        # Read test data from the provided test.csv file.
        test = pl.read_csv(self.data_path / '/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv')
        # Convert group_by object to a list so we can use its length for progress tracking.
        groups = list(test.group_by('id'))
        for _, group in groups:
            yield group.item(0, 0), group.item(0, 1)  # id, description

    def get_all_predictions(self):
        row_ids, predictions = [], []
        # Build a list of all test batches for progress tracking.
        batches = list(self.generate_data_batches())
        for id, description in tqdm(batches, desc="Predicting SVGs", total=len(batches)):
            svg = self.predict(description)
            self.validate(svg)
            row_ids.append(id)
            predictions.append(svg)
        return predictions, row_ids

    def validate(self, svg: str):
        try:
            self.constraints.validate_svg(svg)
        except ValueError as err:
            msg = f'SVG failed validation: {str(err)}'
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, msg)

    def write_submission(self, predictions: list, row_ids: list) -> pathlib.Path:
        predictions_df = pl.DataFrame(
            data={
                self.row_id_column_name: row_ids,
                self.target_column_name: predictions,
            }
        )

        submission_path = pathlib.Path('/kaggle/working/submission.csv')
        if not IS_RERUN:
            with tempfile.NamedTemporaryFile(prefix='kaggle-evaluation-submission-', suffix='.csv', delete=False, mode='w+') as f:
                submission_path = pathlib.Path(f.name)
        predictions_df.write_csv(submission_path)
        return submission_path

# ---------------------------------------------------------------------------
# Main Block: SVG Validation Examples with Progress Bar
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    svg_validator = SVGConstraints()

    valid_svg = """
    <svg width="100" height="100">
      <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
    </svg>
    """

    invalid_size_svg = '<svg>' + ' ' * 10000 + '</svg>'  # Exceeds default size limit

    invalid_element_svg = """
    <svg>
      <script>alert('bad');</script>
    </svg>
    """

    invalid_attribute_svg = """
    <svg>
      <rect width="100" height="100" onclick="alert('bad')"/>
    </svg>
    """

    invalid_href_svg = """
    <svg>
      <use href="http://example.com/image.svg" />
    </svg>
    """

    invalid_embedded_image_element_svg = """
    <svg width="100" height="100">
      <image href="data:image/png;base64,iVBOAAAANSUhEUgAAAAUAAAAFCAYAAACN==" width="50" height="50"/>
    </svg>
    """

    invalid_data_uri_attribute_svg = """
    <svg width="100" height="100">
      <rect width="50" height="50" fill="url(data:image/png;base64,iVBOAAAANSUhEUgAAAAUAAAAFCAYAAACN==)" />
    </svg>
    """

    # Define test cases: (Label, SVG code, Expected Outcome: True if valid, False if should fail)
    tests = [
        ("Valid SVG example", valid_svg, True),
        ("SVG exceeding size limit", invalid_size_svg, False),
        ("SVG with disallowed element", invalid_element_svg, False),
        ("SVG with disallowed attribute", invalid_attribute_svg, False),
        ("SVG with invalid external href", invalid_href_svg, False),
        ("SVG with disallowed <image> element", invalid_embedded_image_element_svg, False),
        ("SVG with invalid data URI in attribute (fill)", invalid_data_uri_attribute_svg, False)
    ]

    print('Running SVG validation examples:')
    for label, svg_code, should_pass in tqdm(tests, desc="Validating SVGs", total=len(tests)):
        print(f'\n{label}:')
        try:
            svg_validator.validate_svg(svg_code)
            if should_pass:
                print('  Validation successful!')
            else:
                print('  Validation successful! (This should not happen)')
        except ValueError as e:
            if not should_pass:
                print(f'  Validation failed as expected with error: {e}')
            else:
                print(f'  Validation failed unexpectedly with error: {e}')














