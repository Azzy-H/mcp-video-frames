"""CLI behaviour and the ``isatty()`` entry-point split.

The split matters more than it looks: an MCP client always connects over a
pipe, a person always has a terminal.  Getting it wrong means a bare
``mcp-video-frames`` typed by a human sits there waiting for JSON-RPC and
looks hung — a bug that only appears interactively, which is exactly the kind
that survives integration testing.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import pytest

from mcp_video_frames import cli


def _module_source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def _imported_names(source: str, module: str) -> set[str]:
    """Names imported from ``module`` anywhere in ``source``.

    Handles both absolute and relative imports, so ``from .core import Core``
    in a package module is recognised the same as
    ``from mcp_video_frames.core import Core``.
    """
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = (node.module or "").lstrip(".")
            if target == module or target.endswith("." + module):
                names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module or alias.name.endswith("." + module):
                    names.add(alias.name)
    return names


def _string_literals(source: str) -> list[str]:
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


# ----------------------------------------------------------------------
# entry-point dispatch
# ----------------------------------------------------------------------
@pytest.fixture()
def main_module(monkeypatch):
    import mcp_video_frames.__main__ as entry

    return entry


class FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty
        self.read_calls = 0

    def isatty(self) -> bool:
        return self._tty

    def read(self, *args):  # pragma: no cover - server would call this
        self.read_calls += 1
        return ""


class TestEntryDispatch:
    def test_arguments_go_to_the_cli(self, main_module, monkeypatch):
        seen: list[list[str]] = []
        monkeypatch.setattr(sys, "argv", ["mcp-video-frames", "doctor"])
        monkeypatch.setattr(main_module, "cli_main", lambda argv: seen.append(argv) or 0)
        monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True))
        assert main_module.main() == 0
        assert seen == [["doctor"]]

    def test_no_arguments_on_a_terminal_prints_help_and_returns(
        self, main_module, monkeypatch, capsys
    ):
        started: list[bool] = []
        monkeypatch.setattr(sys, "argv", ["mcp-video-frames"])
        monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True))
        monkeypatch.setattr(main_module, "print_help", lambda: started.append(True))
        assert main_module.main() == 0
        assert started == [True], "a terminal user must get help, not a hung server"

    def test_no_arguments_on_a_pipe_starts_the_server(self, main_module, monkeypatch):
        calls: list[bool] = []

        fake_server = types.ModuleType("mcp_video_frames.server")
        fake_server.serve_stdio = lambda: calls.append(True) or 0
        monkeypatch.setitem(sys.modules, "mcp_video_frames.server", fake_server)
        monkeypatch.setattr(sys, "argv", ["mcp-video-frames"])
        monkeypatch.setattr(sys, "stdin", FakeStdin(tty=False))

        assert main_module.main() == 0
        assert calls == [True]

    def test_isatty_check_is_the_only_thing_separating_the_two(
        self, main_module, monkeypatch
    ):
        help_calls: list[bool] = []
        server_calls: list[bool] = []
        fake_server = types.ModuleType("mcp_video_frames.server")
        fake_server.serve_stdio = lambda: server_calls.append(True) or 0
        monkeypatch.setitem(sys.modules, "mcp_video_frames.server", fake_server)
        monkeypatch.setattr(main_module, "print_help", lambda: help_calls.append(True))
        monkeypatch.setattr(sys, "argv", ["mcp-video-frames"])

        monkeypatch.setattr(sys, "stdin", FakeStdin(tty=False))
        main_module.main()
        monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True))
        main_module.main()

        assert help_calls == [True]
        assert server_calls == [True]


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
class TestParser:
    def test_subcommands_match_the_documented_set(self):
        parser = cli.build_parser()
        subparsers = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        assert len(subparsers) == 1
        assert set(subparsers[0].choices) == {
            "doctor",
            "info",
            "scan",
            "frames",
            "cache",
        }

    def test_scan_accepts_files_and_a_directory(self):
        args = cli.build_parser().parse_args(["scan", "a.mp4", "b.mp4", "--dir", "d"])
        assert args.videos == ["a.mp4", "b.mp4"]
        assert args.directory == "d"

    def test_frames_defaults(self):
        args = cli.build_parser().parse_args(["frames", "v.mp4"])
        assert args.start == 0.0
        assert args.end is None
        assert args.interval == 1.0
        assert args.max_edge == 768
        assert args.quality == 85
        assert args.image_format is None

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0
        assert "mcp-video-frames" in capsys.readouterr().out

    def test_unknown_subcommand_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["frobnicate"])
        assert excinfo.value.code == 2

    def test_no_arguments_prints_help_and_succeeds(self, capsys):
        assert cli.main([]) == 0
        out = capsys.readouterr().out
        for subcommand in ("doctor", "info", "scan", "frames", "cache"):
            assert subcommand in out
        assert "MCP server" in out


# ----------------------------------------------------------------------
# subcommand behaviour, with the core faked out
# ----------------------------------------------------------------------
class FakeCore:
    def __init__(self, **kwargs):
        self.config = types.SimpleNamespace(cache_max_bytes=1, frame_max_age_days=14)
        self._kwargs = kwargs
        self.scanned: list[tuple[str, bool]] = []
        self.pruned = 0
        self.frames_calls: list[dict] = []

    def doctor(self):
        return self._kwargs.get("doctor", {"ok": True})

    def video_basic_info(self, video, force=False):
        from mcp_video_frames.errors import InputError

        if video == "missing.mp4":
            raise InputError(
                f"Video file does not exist: {video}. Pass the absolute path "
                f"of a video file."
            )
        return {"path": video, "duration": 10.0, "width": 320, "height": 240}, object()

    def video_full_info(self, video, force=False):
        from mcp_video_frames.errors import FfmpegError

        self.scanned.append((video, force))
        if video == "bad.mp4":
            raise FfmpegError(
                "Cannot parse scene cuts.", detail="ffmpeg said nothing useful"
            )
        return {
            "detail": "full",
            "scene_cuts": [5.0],
            "silences": [{"start": 1.0, "end": 2.0, "duration": 1.0}],
            "loudness": {"integrated_lufs": -23.0, "true_peak_dbfs": -1.2, "lra": 5.0},
            "scan": {"cached": False},
        }

    def iter_videos(self, directory, recursive=True):
        return [Path(directory) / "a.mp4", Path(directory) / "b.mkv"]

    def cache_stats(self):
        return {"cache_root": "x", "videos": 2, "bytes": 100}

    def prune_cache(self, max_bytes=None):
        self.pruned += 1
        return {"removed_entries": 1, "cache_root": "x", "max_bytes": max_bytes}

    def view_frames(self, video, **kwargs):
        from mcp_video_frames.core import FramesResult
        from mcp_video_frames.budget import FramesPlan, FrameSpec

        self.frames_calls.append({"video": video, **kwargs})
        frames = [_FakeFrame(0, 0.0), _FakeFrame(1, 1.0)]
        plan = FramesPlan(
            start=0.0,
            end=1.0,
            interval=1.0,
            count=2,
            frames=tuple(FrameSpec(n=f.n, t=f.t) for f in frames),
            max_edge=kwargs.get("max_edge", 768),
            image_format=kwargs.get("image_format") or "jpeg",
            quality=85,
        )
        summary = {"video": video, "frames": 2, "cached": 0}
        return FramesResult(
            frames=frames,
            summary=summary,
            plan=plan,
            video_path=Path(video),
            identity="0" * 64,
        )


class _FakeFrame:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.data = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")
        self.size = len(self.data)
        self.mime_type = "image/jpeg"
        self.path = Path(f"t{t:09.3f}.jpg")
        self.width = 1
        self.height = 1


@pytest.fixture()
def fake_core(monkeypatch):
    def install(**kwargs):
        core = FakeCore(**kwargs)
        monkeypatch.setattr(cli, "Core", lambda *a, **k: core)
        return core

    return install


class TestDoctorCommand:
    def test_ok_report_exits_zero(self, fake_core, capsys):
        fake_core(doctor={"ok": True, "cache_root": "x"})
        assert cli.main(["doctor"]) == 0
        payload = json.loads(capsys.readouterr().out.split("\nSelf-check")[0])
        assert payload["ok"] is True

    def test_failing_report_exits_one_and_explains(self, fake_core, capsys):
        fake_core(
            doctor={
                "ok": False,
                "ffmpeg_error": "ffmpeg not found. Install ffmpeg.",
            }
        )
        assert cli.main(["doctor"]) == 1
        captured = capsys.readouterr()
        assert "ffmpeg not found" in captured.out
        assert "Self-check failed" in captured.err


class TestInfoCommand:
    def test_prints_basic_metadata(self, fake_core, capsys):
        fake_core()
        assert cli.main(["info", "v.mp4"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["duration"] == 10.0
        assert "scene_cuts" not in payload

    def test_missing_file_is_a_message_not_a_traceback(self, fake_core, capsys):
        fake_core()
        assert cli.main(["info", "missing.mp4"]) == 1
        captured = capsys.readouterr()
        assert "Video file does not exist" in captured.err
        assert "Traceback" not in captured.err


class TestScanCommand:
    def test_scans_named_files(self, fake_core, capsys):
        core = fake_core()
        assert cli.main(["scan", "a.mp4", "b.mp4"]) == 0
        assert [name for name, _force in core.scanned] == ["a.mp4", "b.mp4"]
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["scene_cuts"] == 1

    def test_directory_mode_expands_to_videos(self, fake_core, capsys):
        core = fake_core()
        assert cli.main(["scan", "--dir", "videos"]) == 0
        assert [name for name, _force in core.scanned] == [
            str(Path("videos") / "a.mp4"),
            str(Path("videos") / "b.mkv"),
        ]

    def test_one_bad_file_does_not_stop_the_batch(self, fake_core, capsys):
        core = fake_core()
        assert cli.main(["scan", "good.mp4", "bad.mp4", "also.mp4"]) == 1
        assert [name for name, _force in core.scanned] == [
            "good.mp4",
            "bad.mp4",
            "also.mp4",
        ]
        captured = capsys.readouterr()
        assert "Cannot parse scene cuts" in captured.err
        assert "ffmpeg said nothing useful" in captured.err

    def test_no_targets_is_an_actionable_error(self, fake_core, capsys):
        fake_core()
        assert cli.main(["scan"]) == 1
        assert "--dir" in capsys.readouterr().err

    def test_force_is_forwarded(self, fake_core):
        core = fake_core()
        cli.main(["scan", "a.mp4", "--force"])
        assert core.scanned == [("a.mp4", True)]


class TestFramesCommand:
    def test_emits_the_same_content_blocks_as_mcp(self, fake_core, capsys):
        fake_core()
        assert cli.main(["frames", "v.mp4"]) == 0
        blocks = json.loads(capsys.readouterr().out)
        assert [block["type"] for block in blocks] == ["text", "image", "text", "image", "text"]
        assert json.loads(blocks[0]["text"]) == {"n": 0, "t": 0.0}
        # Without --images-dir this is a real content-block sequence, so an
        # image block's `data` must decode as base64.
        import base64

        base64.b64decode(blocks[1]["data"], validate=True)

    def test_no_content_blocks_with_images_dir_still_writes_the_files(
        self, fake_core, tmp_path, capsys
    ):
        """The two flags together: files on disk, summary on stdout, no blocks."""
        fake_core()
        out_dir = tmp_path / "images"
        assert cli.main(
            ["frames", "v.mp4", "--images-dir", str(out_dir), "--no-content-blocks"]
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["blocks"] == 5
        assert "paths" not in payload
        assert payload["summary"]["frames"] == 2
        assert len(sorted(out_dir.iterdir())) == 2

    def test_images_dir_writes_files_instead_of_base64(self, fake_core, tmp_path, capsys):
        fake_core()
        out_dir = tmp_path / "images"
        assert cli.main(["frames", "v.mp4", "--images-dir", str(out_dir)]) == 0
        document = json.loads(capsys.readouterr().out)
        written = sorted(out_dir.iterdir())
        assert len(written) == 2

        # Wrapped, and the envelope says what `data` holds.  Emitting the bare
        # block sequence here would be indistinguishable from real MCP content
        # blocks, whose `data` field IS the base64 payload — a consumer would
        # decode a filename.
        assert document["paths"] is True
        blocks = document["blocks"]
        assert [block["type"] for block in blocks] == [
            "text",
            "image",
            "text",
            "image",
            "text",
        ]
        image = blocks[1]
        assert image["data"].startswith(cli.PATH_SCHEME)
        assert image["data"][len(cli.PATH_SCHEME) :] == str(written[0])

    def test_images_dir_does_not_invent_a_per_block_field(self, fake_core, tmp_path, capsys):
        """The signal belongs at the top, not repeated inside every block."""
        fake_core()
        assert cli.main(["frames", "v.mp4", "--images-dir", str(tmp_path / "i")]) == 0
        blocks = json.loads(capsys.readouterr().out)["blocks"]
        for block in blocks:
            expected = {"type", "text"} if block["type"] == "text" else {
                "type",
                "data",
                "mimeType",
            }
            assert set(block) == expected, block

    def test_images_dir_keeps_every_block(self, fake_core, tmp_path, capsys):
        """Same 2n+1 sequence as without the flag; only `data` differs."""
        fake_core()
        assert cli.main(["frames", "v.mp4", "--images-dir", str(tmp_path / "i")]) == 0
        with_paths = json.loads(capsys.readouterr().out)["blocks"]
        assert cli.main(["frames", "v.mp4"]) == 0
        plain = json.loads(capsys.readouterr().out)
        assert [b["type"] for b in with_paths] == [b["type"] for b in plain]
        # Text blocks are untouched; image blocks differ only in `data`.
        for sparse, normal in zip(with_paths, plain):
            if sparse["type"] == "text":
                assert sparse == normal
            else:
                assert sparse["mimeType"] == normal["mimeType"]
                assert sparse["data"] != normal["data"]

    def test_arguments_reach_the_core(self, fake_core):
        core = fake_core()
        cli.main(
            [
                "frames",
                "v.mp4",
                "--start",
                "2",
                "--end",
                "6",
                "--interval",
                "0.5",
                "--max-edge",
                "512",
                "--format",
                "png",
                "--quality",
                "70",
            ]
        )
        call = core.frames_calls[-1]
        assert call["start"] == 2.0
        assert call["end"] == 6.0
        assert call["interval"] == 0.5
        assert call["max_edge"] == 512
        assert call["image_format"] == "png"
        assert call["quality"] == 70

    def test_no_content_blocks_flag_prints_only_the_summary(self, fake_core, capsys):
        fake_core()
        assert cli.main(["frames", "v.mp4", "--no-content-blocks"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["blocks"] == 5
        assert payload["summary"]["frames"] == 2


class TestCacheCommand:
    def test_stats(self, fake_core, capsys):
        fake_core()
        assert cli.main(["cache", "--stats"]) == 0
        assert json.loads(capsys.readouterr().out)["videos"] == 2

    def test_prune(self, fake_core, capsys):
        core = fake_core()
        assert cli.main(["cache", "--prune", "--max-bytes", "123"]) == 0
        assert core.pruned == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["removed_entries"] == 1
        assert payload["max_bytes"] == 123

    def test_neither_flag_is_an_actionable_error(self, fake_core, capsys):
        fake_core()
        assert cli.main(["cache"]) == 1
        assert "--stats" in capsys.readouterr().err


class TestOutputEncoding:
    def test_stdio_is_forced_to_utf8(self, monkeypatch):
        """Messages are Chinese and stdout carries JSON.

        On a Windows console with a legacy code page, or when stdout is
        redirected (which defaults to UTF-16), the output would otherwise be
        mojibake or unparsable.
        """
        calls: list[tuple[str, str]] = []

        class FakeStream:
            def __init__(self, name):
                self.name = name

            def reconfigure(self, encoding=None, errors=None):
                calls.append((self.name, encoding))

        monkeypatch.setattr(sys, "stdout", FakeStream("out"))
        monkeypatch.setattr(sys, "stderr", FakeStream("err"))
        cli.ensure_utf8_stdio()
        assert calls == [("out", "utf-8"), ("err", "utf-8")]

    def test_streams_without_reconfigure_are_left_alone(self, monkeypatch):
        class Bare:
            pass

        monkeypatch.setattr(sys, "stdout", Bare())
        monkeypatch.setattr(sys, "stderr", Bare())
        cli.ensure_utf8_stdio()  # must not raise

    def test_a_refusing_stream_does_not_break_startup(self, monkeypatch):
        class Stubborn:
            def reconfigure(self, **kwargs):
                raise OSError("nope")

        monkeypatch.setattr(sys, "stdout", Stubborn())
        monkeypatch.setattr(sys, "stderr", Stubborn())
        cli.ensure_utf8_stdio()


class TestCliMatchesMcp:
    def test_info_and_video_info_basic_agree(self, fake_core, capsys):
        """Acceptance item 10: the two front ends must not drift."""
        fake_core()
        assert cli.main(["info", "v.mp4"]) == 0
        cli_payload = json.loads(capsys.readouterr().out)
        # The CLI calls Core.video_basic_info, which is exactly what the MCP
        # video_info(detail="basic") handler calls — no second implementation.
        assert cli_payload == {
            "path": "v.mp4",
            "duration": 10.0,
            "width": 320,
            "height": 240,
        }

    def test_cli_does_not_shell_out_to_ffmpeg(self):
        """The rule that keeps the two front ends from drifting.

        Checked structurally rather than by string search: a comment naming
        ffmpeg is fine, an invocation is not.
        """
        source = _module_source(cli)
        assert "subprocess" not in _imported_names(source, "subprocess")
        assert not [
            text for text in _string_literals(source) if text.strip() == "ffmpeg"
        ]

    def test_server_never_builds_an_ffmpeg_command(self, monkeypatch):
        from conftest import install_fake_mcp_sdk

        install_fake_mcp_sdk(monkeypatch)
        import importlib

        import mcp_video_frames.server as server

        server = importlib.reload(server)
        source = _module_source(server)
        assert "subprocess" not in _imported_names(source, "subprocess")
        for forbidden in ("extract_range", "extract_single", "run_ffmpeg", "ffprobe"):
            assert forbidden not in _imported_names(source, "mcp_video_frames")

    def test_cli_uses_the_core_library_functions(self):
        source = _module_source(cli)
        # The CLI must reach frames and scans through Core, never by
        # re-assembling an ffmpeg command of its own.
        assert "Core" in _imported_names(source, "core")
        assert "build_content_blocks" in _imported_names(source, "core")
        assert _imported_names(source, "frames") == set()
        assert _imported_names(source, "scan") == set()
