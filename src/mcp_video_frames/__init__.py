"""mcp-video-frames — give models without video input a way to *look* at video.

The package has one core library and two thin front ends:

    mcp_video_frames/            <- core library (all of the logic)
          ^              ^
      server.py        cli.py
      (MCP front end)  (CLI front end)

Everything that can be called must be callable without MCP, and without a
terminal.  The front ends only translate arguments and results.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
