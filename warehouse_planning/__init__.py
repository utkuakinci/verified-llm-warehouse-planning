"""Verified LLM planning for warehouse robots."""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    # msvc-runtime installs the Visual C++ DLLs into the venv's Scripts folder, which Windows does
    # not search. torch needs them at import time and Fast Downward's downward.exe at launch, so
    # they go on PATH (inherited by subprocesses) before either is loaded.
    _scripts = Path(sys.prefix) / "Scripts"
    if (_scripts / "msvcp140.dll").exists():
        os.environ["PATH"] = f"{_scripts}{os.pathsep}{os.environ.get('PATH', '')}"
        os.add_dll_directory(str(_scripts))

ROOT = Path(__file__).resolve().parent.parent
PDDL_DIR = ROOT / "pddl"
DOMAIN_PATH = PDDL_DIR / "domain.pddl"
RESULTS_DIR = ROOT / "results"
