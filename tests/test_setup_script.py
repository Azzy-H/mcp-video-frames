"""The decision logic in setup_mcp.py: re-running must be idempotent and must
not forget a choice an earlier run already confirmed.

These tests build no virtual environment, install nothing and download
nothing — they check the *decisions* the script makes for a given state.  The
reason is practical: the script overwrites the config it generated itself, so
a re-run that wiped a configured ffmpeg would leave the user staring at
"ffmpeg not found" in their client, with the cause buried dozens of lines up.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import setup_mcp  # noqa: E402  (path set above)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    """Point the generated config at a temp directory, never the repo file."""
    target = tmp_path / "mcp-config.json"
    monkeypatch.setattr(setup_mcp, "CONFIG_PATH", target)
    return target


def write_existing_config(path: Path, env: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    setup_mcp.SERVER_NAME: {
                        "command": "python",
                        "args": ["run_server.py"],
                        "env": env,
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def args(**over) -> types.SimpleNamespace:
    base = {
        "ffmpeg": None,
        "ffmpeg_url": None,
        "no_install": False,
        "no_ffmpeg": False,
        "install_ffmpeg": False,
        "max_images": 20,
        "cache_dir": None,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture()
def no_path_ffmpeg(monkeypatch):
    """Pretend nothing is installed system-wide."""
    monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)


# ----------------------------------------------------------------------
# reading and comparing
# ----------------------------------------------------------------------
class TestReadExistingEnv:
    def test_missing_file_is_empty(self, config_path):
        assert setup_mcp.read_existing_env() == {}

    def test_roundtrip(self, config_path):
        write_existing_config(config_path, {"FFMPEG_PATH": "/x/ffmpeg"})
        assert setup_mcp.read_existing_env() == {"FFMPEG_PATH": "/x/ffmpeg"}

    def test_corrupt_file_is_empty_not_an_exception(self, config_path):
        config_path.write_text("{ not json", encoding="utf-8")
        assert setup_mcp.read_existing_env() == {}

    def test_unexpected_shape_is_empty(self, config_path):
        config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
        assert setup_mcp.read_existing_env() == {}


class TestDescribeChanges:
    """The change summary must cover the whole entry, not just env.

    Watching only env would stay silent when `command` / `args` / `cwd` change
    -- and those three are both the easiest to get wrong and the ones the user
    most needs to carry over into the client config.
    """

    @staticmethod
    def entry(**over) -> dict:
        base = {
            "command": "python",
            "args": ["run_server.py"],
            "cwd": "/project",
            "env": {"MCP_VIDEO_MAX_IMAGES": "20"},
        }
        base.update(over)
        return base

    def test_identical_entries_produce_nothing(self):
        assert setup_mcp.describe_changes(self.entry(), self.entry()) == []

    def test_reports_an_added_env_key(self):
        lines = setup_mcp.describe_changes(
            self.entry(), self.entry(env={"MCP_VIDEO_MAX_IMAGES": "20", "NEW": "3"})
        )
        assert lines == ["  + env.NEW = 3"], lines

    def test_reports_a_removed_env_key(self):
        lines = setup_mcp.describe_changes(
            self.entry(env={"A": "1", "B": "2"}), self.entry(env={"A": "1"})
        )
        assert any("env.B" in line and line.strip().startswith("-") for line in lines), lines

    def test_reports_a_changed_env_value(self):
        lines = setup_mcp.describe_changes(
            self.entry(env={"A": "1"}), self.entry(env={"A": "2"})
        )
        assert lines == ["  ~ env.A: 1 -> 2"], lines

    def test_reports_a_changed_command(self):
        """This one and the next two were added later: they used to be missed."""
        lines = setup_mcp.describe_changes(
            self.entry(), self.entry(command="/venv/bin/python")
        )
        assert any(line.startswith("  ~ command:") for line in lines), lines

    def test_reports_a_changed_cwd(self):
        lines = setup_mcp.describe_changes(self.entry(), self.entry(cwd="/elsewhere"))
        assert any(line.startswith("  ~ cwd:") for line in lines), lines

    def test_reports_changed_args(self):
        lines = setup_mcp.describe_changes(
            self.entry(), self.entry(args=["run_server.py", "--verbose"])
        )
        assert any(line.startswith("  ~ args:") for line in lines), lines

    def test_unchanged_keys_are_not_mentioned(self):
        lines = setup_mcp.describe_changes(
            self.entry(), self.entry(env={"MCP_VIDEO_MAX_IMAGES": "12"})
        )
        text = "\n".join(lines)
        assert "command" not in text
        assert "cwd" not in text
        assert "env.MCP_VIDEO_MAX_IMAGES" in text


# ----------------------------------------------------------------------
# the ffmpeg decision order
# ----------------------------------------------------------------------
class TestResolveFfmpegMemory:
    def test_remembers_the_path_an_earlier_run_used(self, no_path_ffmpeg, capsys):
        """A second run without --ffmpeg must reuse the path recorded before.

        Regression test: an earlier implementation let a third run silently
        drop FFMPEG_PATH.
        """
        previous = {
            "FFMPEG_PATH": sys.executable,
            "FFPROBE_PATH": "",
        }
        chosen = setup_mcp.resolve_ffmpeg(args(no_install=True), previous)
        assert chosen[0] == sys.executable
        assert "Reusing" in capsys.readouterr().out

    def test_forgets_a_path_that_no_longer_exists(self, no_path_ffmpeg, capsys):
        previous = {"FFMPEG_PATH": str(Path("/definitely/gone/ffmpeg"))}
        chosen = setup_mcp.resolve_ffmpeg(args(no_ffmpeg=True), previous)
        assert chosen == ("", "")
        assert "no longer exists" in capsys.readouterr().out

    def test_explicit_flag_beats_the_remembered_path(self, no_path_ffmpeg):
        previous = {"FFMPEG_PATH": "/old/ffmpeg"}
        chosen = setup_mcp.resolve_ffmpeg(
            args(ffmpeg=sys.executable, no_install=True), previous
        )
        assert chosen[0] == sys.executable

    def test_path_from_path_wins_when_nothing_remembered(self, monkeypatch):
        monkeypatch.setattr(
            setup_mcp.shutil, "which", lambda name: f"/usr/bin/{name}"
        )
        chosen = setup_mcp.resolve_ffmpeg(args(no_install=True), {})
        assert chosen == ("/usr/bin/ffmpeg", "/usr/bin/ffprobe")


class TestNoPromptCases:
    """With no ffmpeg anywhere, none of these cases should raise a menu."""

    def test_no_ffmpeg_flag_skips(self, no_path_ffmpeg, capsys):
        chosen = setup_mcp.resolve_ffmpeg(args(no_ffmpeg=True), {})
        assert chosen == ("", "")
        assert "skipping the prompt" in capsys.readouterr().out

    def test_no_install_skips(self, no_path_ffmpeg, capsys):
        chosen = setup_mcp.resolve_ffmpeg(args(no_install=True), {})
        assert chosen == ("", "")

    def test_non_interactive_skips_and_explains(self, no_path_ffmpeg, monkeypatch, capsys):
        monkeypatch.setattr(setup_mcp.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
        chosen = setup_mcp.resolve_ffmpeg(args(), {})
        assert chosen == ("", "")
        out = capsys.readouterr().out
        assert "--install-ffmpeg" in out
        assert "winget install" in out


# ----------------------------------------------------------------------
# what goes into the config
# ----------------------------------------------------------------------
class TestBuildConfig:
    def test_does_not_pin_a_path_that_is_already_on_path(self, monkeypatch):
        """A binary found on PATH must not be pinned by absolute path.

        Pinning it would turn a working setup into a future failure when the
        path changes.
        """
        monkeypatch.setattr(
            setup_mcp.shutil, "which", lambda name: {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}.get(name)
        )
        config = setup_mcp.build_config(
            max_images=12, cache_dir=None, ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe"
        )
        env = config["mcpServers"][setup_mcp.SERVER_NAME]["env"]
        assert "FFMPEG_PATH" not in env
        assert "FFPROBE_PATH" not in env

    def test_pins_a_path_that_is_not_on_path(self, monkeypatch):
        """An ffmpeg downloaded into .venv must be pinned, or the server misses it."""
        monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)
        config = setup_mcp.build_config(
            max_images=20, cache_dir=None, ffmpeg="/venv/bin/ffmpeg", ffprobe="/venv/bin/ffprobe"
        )
        env = config["mcpServers"][setup_mcp.SERVER_NAME]["env"]
        assert env["FFMPEG_PATH"] == "/venv/bin/ffmpeg"
        assert env["FFPROBE_PATH"] == "/venv/bin/ffprobe"

    def test_always_sets_the_image_limit(self, monkeypatch):
        monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)
        config = setup_mcp.build_config(
            max_images=7, cache_dir=None, ffmpeg="", ffprobe=""
        )
        env = config["mcpServers"][setup_mcp.SERVER_NAME]["env"]
        assert env["MCP_VIDEO_MAX_IMAGES"] == "7"

    def test_default_cache_is_project_local_with_a_pinned_cwd(self, monkeypatch):
        """The default cache lives in the project, and cwd must be pinned with it.

        A relative path resolves against the *client's* working directory.
        Without cwd, `.cache` lands wherever the client happened to start --
        worse than the temp-dir default, because it is unpredictable.
        """
        monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)
        entry = setup_mcp.build_config(
            max_images=20, cache_dir=None, ffmpeg="", ffprobe=""
        )["mcpServers"][setup_mcp.SERVER_NAME]
        assert entry["env"]["MCP_VIDEO_CACHE_DIR"] == ".cache"
        assert entry["cwd"] == str(setup_mcp.PROJECT_ROOT)

    def test_absolute_cache_dir_does_not_need_cwd(self, monkeypatch):
        """An absolute path does not need cwd, so do not add a field that can be wrong."""
        monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)
        # Must be absolute *for this platform*: on Windows a POSIX-style
        # "/var/cache" has no drive letter and Path.is_absolute() is False.
        absolute = "D:\\var\\cache\\vf" if setup_mcp.IS_WINDOWS else "/var/cache/vf"
        entry = setup_mcp.build_config(
            max_images=20, cache_dir=absolute, ffmpeg="", ffprobe=""
        )["mcpServers"][setup_mcp.SERVER_NAME]
        assert entry["env"]["MCP_VIDEO_CACHE_DIR"] == absolute
        assert "cwd" not in entry

    def test_relative_custom_cache_dir_also_pins_cwd(self, monkeypatch):
        monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)
        entry = setup_mcp.build_config(
            max_images=20, cache_dir="my-cache", ffmpeg="", ffprobe=""
        )["mcpServers"][setup_mcp.SERVER_NAME]
        assert entry["env"]["MCP_VIDEO_CACHE_DIR"] == "my-cache"
        assert entry["cwd"] == str(setup_mcp.PROJECT_ROOT)

    def test_cache_name_is_gitignored(self):
        """A project-local cache must be ignored, or it gets committed by mistake."""
        ignore = (setup_mcp.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert f".{setup_mcp.LOCAL_CACHE_DIRNAME}/" in ignore


class TestRerunIsIdempotent:
    """Pins down "a re-run changes nothing" as a whole."""

    def test_second_run_with_same_inputs_writes_the_same_env(self, config_path, monkeypatch):
        monkeypatch.setattr(setup_mcp.shutil, "which", lambda name: None)
        first = setup_mcp.build_config(
            max_images=20, cache_dir=None, ffmpeg=sys.executable, ffprobe=""
        )["mcpServers"][setup_mcp.SERVER_NAME]["env"]
        setup_mcp.write_config(
            {"mcpServers": {setup_mcp.SERVER_NAME: {"env": first}}}
        )

        previous = setup_mcp.read_existing_env()
        chosen = setup_mcp.resolve_ffmpeg(args(no_install=True), previous)
        second = setup_mcp.build_config(
            max_images=20, cache_dir=None, ffmpeg=chosen[0], ffprobe=chosen[1]
        )["mcpServers"][setup_mcp.SERVER_NAME]["env"]

        assert first == second
        assert setup_mcp.describe_changes(first, second) == []
