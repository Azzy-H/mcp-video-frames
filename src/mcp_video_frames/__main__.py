"""Entry point dispatch: MCP server over stdio, or the CLI.

One console script, two behaviours::

    mcp-video-frames                 # client pipes stdio  -> MCP server
    mcp-video-frames doctor          # a human typed something -> CLI
    mcp-video-frames                 # a human, in a terminal -> help, exit 0

The ``isatty()`` check is the whole trick.  An MCP client always connects
over a pipe; a person always has a terminal.  Without it, typing the bare
command would give you a process that sits waiting for JSON-RPC and looks
hung.

``python -m mcp_video_frames`` behaves identically thanks to the
``__main__`` guard at the bottom — that is the fallback when the console
script is not on PATH.
"""

from __future__ import annotations

import sys

from .cli import main as cli_main
from .cli import print_help


def main() -> int:
    if len(sys.argv) > 1:  # subcommand -> CLI
        return cli_main(sys.argv[1:])
    if sys.stdin.isatty():  # no subcommand, at a terminal -> help, no hang
        print_help()
        return 0
    # Otherwise: stdio is a pipe, which means an MCP client is talking to us.
    from .server import serve_stdio

    return serve_stdio()


if __name__ == "__main__":  # makes `python -m mcp_video_frames` work too
    sys.exit(main())
