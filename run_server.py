#!/usr/bin/env python3
"""Start the MCP server from a checkout, without installing anything.

An MCP client is configured with a Python interpreter and a script:

    {
      "command": "<project>/.venv/Scripts/python.exe",
      "args": ["<project>/run_server.py"],
      "env": {"MCP_VIDEO_MAX_IMAGES": "20"}
    }

``setup_mcp.py`` generates exactly that, with the real paths substituted.

Why this file exists instead of pointing the client at
``src/mcp_video_frames/__main__.py``:

* the package lives under ``src/``, so the source tree is not importable
  until ``src`` is on ``sys.path``;
* doing that here means the client configuration does not need a
  ``PYTHONPATH`` entry — one less free-form value that can be wrong, and no
  dependence on how a particular client merges its own environment.

Diagnostics go to stderr; stdout stays clean for the JSON-RPC stream.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


def main() -> int:
    if not (SRC_DIR / "mcp_video_frames").is_dir():
        print(
            f"Package directory not found: {SRC_DIR / 'mcp_video_frames'}\n"
            f"run_server.py must sit next to src/.",
            file=sys.stderr,
        )
        return 1
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    # Delegating keeps one entry point and one dispatch rule for the whole
    # project: `python run_server.py` and `python -m mcp_video_frames` behave
    # identically, including the isatty help-vs-server split.
    from mcp_video_frames.__main__ import main as entry_main

    return entry_main()


if __name__ == "__main__":
    sys.exit(main())
