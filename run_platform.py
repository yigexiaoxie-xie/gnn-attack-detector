"""PyCharm-friendly launcher for the local Streamlit platform."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def build_streamlit_command(
    project_root: str | Path,
    *,
    port: int = 8501,
) -> list[str]:
    root = Path(project_root).resolve()
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "app.py"),
        "--global.developmentMode=false",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]


def main() -> int:
    root = Path(__file__).resolve().parent
    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed. Run scripts\\setup_windows_cpu.bat first.")
        return 1
    environment = os.environ.copy()
    port = int(environment.get("GNN_PLATFORM_PORT", "8501"))
    source_root = str(root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    print("Starting the graph attack detection platform...")
    print(f"Server browser: http://localhost:{port}")
    print(f"Another computer: http://SERVER_LAN_IP:{port}")
    return subprocess.call(
        build_streamlit_command(root, port=port),
        cwd=root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
