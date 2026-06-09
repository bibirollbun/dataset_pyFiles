# Install dependencies
!pip uninstall -qqy jupyterlab
!pip install -U -q google-genai
!!apt-get update -qqy && apt-get install -qqy --no-install-recommends pandoc texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-xetex
!!apt-get install lmodern texlive-fonts-recommended


from __future__ import annotations

import io
import os
import re
import sys
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterable, List, Sequence
from IPython.display import IFrame


# API Key setup
from kaggle_secrets import UserSecretsClient

try:
    api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    print("Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


def resolve_repo_root() -> Path:
    """Best-effort resolution of the repository root when running in a notebook."""
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent
    return Path.cwd()


def ensure_pkg_on_path(repo_root: Path | None = None) -> Path:
    """Append the repository's `src` directory to sys.path if needed."""
    base = repo_root or resolve_repo_root()
    src_dir = base / "src"
    src_dir_str = str(src_dir)
    if src_dir.exists() and src_dir_str not in sys.path:
        sys.path.append(src_dir_str)
    return src_dir


def build_cli_argv(data_files: Iterable[Path], api_key: str | None, passthrough: List[str]) -> List[str]:
    """Construct the argument vector expected by `cli_data_explorer.cli.main`."""
    argv = ["cli-data-explorer", "workflow", "--data"]
    argv.extend(str(path) for path in data_files)
    if api_key:
        argv.extend(["--api-key", api_key])
    argv.extend(passthrough)
    return argv


# todo - fix path extraction
REPORT_PATH_PATTERN = re.compile(r"(workspace[\\/]run_\d+[\\/][^\s]+\.md)")
last_report_path: Path | None = None


def extract_report_path(output: str) -> Path | None:
    """Return the last markdown report path mentioned in CLI output."""
    matches = REPORT_PATH_PATTERN.findall(output)
    if matches:
        relative_string = matches[-1].strip()
        return Path(relative_string)
        
    return None


async def run_data_explorer(
    data: Sequence[str | Path] | str | Path,
    pkg_path: str = "/kaggle/input/cli-data-explorer-v2",
    api_key: str | None = None,
    vision_model: str | None = None,
    vision_disable: bool = False,
    extra_args: Sequence[str] | None = None,
) -> tuple[int, Path | None]:
    """Invoke cli-data-explorer using the same logic as data_explore.py."""
    global last_report_path

    ensure_pkg_on_path(Path(pkg_path))

    try:
        from cli_data_explorer import cli as cli_module
    except ImportError as exc:
        raise RuntimeError("Unable to import cli_data_explorer. Check your environment.") from exc

    if isinstance(data, (str, Path)):
        data_iterable: Sequence[str | Path] = [data]
    else:
        data_iterable = data

    data_paths = [Path(item).expanduser().resolve() for item in data_iterable]
    missing = [str(path) for path in data_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Data file(s) not found: {', '.join(missing)}")

    passthrough = list(extra_args or [])
    cli_argv = build_cli_argv(data_paths, api_key, passthrough)
    original_argv = sys.argv.copy()
    sys.argv = cli_argv

    buffer = io.StringIO()
    exit_code = 0
    with redirect_stdout(buffer), redirect_stderr(buffer):
        try:
            await cli_module.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        finally:
            sys.argv = original_argv

    output = buffer.getvalue()
    print(output, end="")

    report_path = extract_report_path(output)
    last_report_path = report_path

    return exit_code, report_path


data_files = ["/kaggle/input/youtube-analytics-data/youtube_recommendation_dataset -.csv"]
exit_code, report_path = await run_data_explorer(
    data_files,
    api_key=api_key, 
    extra_args=["--prompt", "Explore this dataset."],
)
print(f"cli-data-explorer exited with status {exit_code}")
print(f"Markdown report path: {report_path}")
last_report_path


cmd = (f"pandoc /kaggle/working/workspace/run_1764484238/final_report.md "
       f"--resource-path=/kaggle/working/workspace/run_1764484238 "
       "--pdf-engine=xelatex "
       "-o report.pdf "
       "-V geometry:margin=1in "          
      )

print(cmd)
os.system(cmd)


from IPython.display import IFrame
IFrame(
  src="./report.pdf",
  width=900,
  height=600,
)

