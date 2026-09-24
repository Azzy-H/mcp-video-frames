"""MCP protocol layer: content-block order and error semantics.

The order test is the most important test in the suite.  If the blocks came
back ``[image, text]`` pairs or with the summary first, nothing would raise:
the caller would simply receive images with the wrong timecodes attached, and
every conclusion drawn from them would be quietly wrong.

The SDK is stubbed here so the order guarantee is checked without needing the
real ``mcp`` package installed.  When it *is* installed the stub is skipped
and the real one is used.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from conftest import install_fake_mcp_sdk
from mcp_video_frames.config import Config
from mcp_video_frames.errors import InputError, LimitError


# ----------------------------------------------------------------------
# fake core
# ----------------------------------------------------------------------
def _png_1x1() -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


PNG_BYTES = _png_1x1()
JPEG_BYTES = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")


class FakeFrame:
    def __init__(self, n, t, data=PNG_BYTES, mime="image/png"):
        self.n = n
        self.t = t
        self.data = data
        self.size = len(data)
        self.mime_type = mime
        self.path = Path(f"t{t:09.3f}.png")
        self.width = 1
        self.height = 1


class FakePlan:
    def __init__(self, frames, image_format="png", interval=1.0, max_edge=768):
        self.frames = tuple(frames)
        self.count = len(frames)
        self.image_format = image_format
        self.interval = interval
        self.max_edge = max_edge
        self.clamped = False
        self.range = (frames[0].t, frames[-1].t) if frames else (0.0, 0.0)


class FakeResult:
    def __init__(self, frames, summary, image_format="png"):
        self.frames = list(frames)
        self.summary = summary
        self.plan = FakePlan(frames, image_format=image_format)


class FakeCore:
    """Duck-typed core: the protocol layer must not depend on its internals."""

    def __init__(self, config, frames=None, basic=None, full=None, raise_on=None):
        self.config = config
        self._frames = frames
        self._basic = basic or {"duration": 10.0, "width": 320, "height": 240}
        self._full = full
        self._raise_on = raise_on or {}
        self.calls: list[tuple[str, dict]] = []

    def view_frames(self, video, **kwargs):
        self.calls.append(("view_frames", {"video": video, **kwargs}))
        if "view_frames" in self._raise_on:
            raise self._raise_on["view_frames"]
        frames = self._frames
        if frames is None:
            frames = [FakeFrame(0, 0.0)]
        summary = {
            "video": video,
            "frames": len(frames),
            "range": [frames[0].t, frames[-1].t],
            "interval": kwargs.get("interval", 1.0),
            "format": "png",
            "max_edge": kwargs.get("max_edge", 768),
            "clamped": False,
            "cached": 0,
            # Mirrors what core.view_frames puts in the summary: the limits in
            # force for this call, which is the channel the agent reads.
            "limits": {
                "max_images_per_call": self.config.max_images_per_call,
                "max_image_bytes": self.config.max_image_bytes,
                "max_total_bytes": self.config.max_total_bytes,
            },
        }
        return FakeResult(frames, summary)

    def video_basic_info(self, video, force=False):
        self.calls.append(("video_basic_info", {"video": video, "force": force}))
        if "basic" in self._raise_on:
            raise self._raise_on["basic"]
        return dict(self._basic), object()

    def video_full_info(self, video, force=False):
        self.calls.append(("video_full_info", {"video": video, "force": force}))
        if "full" in self._raise_on:
            raise self._raise_on["full"]
        payload = dict(self._basic)
        payload.update(
            {"detail": "full", "scene_cuts": [5.0], "silences": [], "loudness": {}}
        )
        return payload

    def doctor(self):
        return {"ok": True}


@pytest.fixture()
def server_module(monkeypatch):
    install_fake_mcp_sdk(monkeypatch)
    import importlib

    import mcp_video_frames.server as server

    return importlib.reload(server)


@pytest.fixture()
def config(tmp_cache_dir):
    return Config.from_env(
        {
            "MCP_VIDEO_CACHE_DIR": str(tmp_cache_dir),
            "MCP_VIDEO_MAX_IMAGES": "7",
            "MCP_VIDEO_MAX_IMAGE_BYTES": "5242880",
            "MCP_VIDEO_MAX_TOTAL_BYTES": "20971520",
        }
    )


def make_server(server_module, config, **core_kwargs):
    core = FakeCore(config, **core_kwargs)
    return server_module.build_server(core), core


class TestConfigurationIsVisibleToTheAgent:
    """What the agent learns about this deployment's limits.

    A model sees only tool descriptions, parameter schemas and results — never
    the generated client config, the environment, or the startup log.  So a
    limit that is not in one of those three places is invisible: the caller can
    only discover it by being rejected, which costs a round trip and reads as a
    failure rather than a rule.
    """

    def test_view_frames_description_states_the_effective_image_limit(
        self, server_module, config
    ):
        server, _core = make_server(server_module, config)
        description = server.tools["view_frames"]["description"]
        assert str(config.max_images_per_call) in description, description

    def test_view_frames_description_states_the_byte_limits(
        self, server_module, config
    ):
        server, _core = make_server(server_module, config)
        description = server.tools["view_frames"]["description"]
        # Rendered in MB, so 5 MB and 20 MB respectively.
        assert "5 MB" in description, description
        assert "20 MB" in description, description

    def test_description_tracks_the_configuration_not_a_hardcoded_number(
        self, server_module
    ):
        """Change the limit, and the text the agent reads must change too."""
        other = Config.from_env({"MCP_VIDEO_MAX_IMAGES": "13"})
        server, _core = make_server(server_module, other)
        assert "13" in server.tools["view_frames"]["description"]

    def test_description_promises_no_silent_degradation(self, server_module, config):
        server, _core = make_server(server_module, config)
        description = server.tools["view_frames"]["description"].lower()
        assert "never silently dropped" in description
        assert "never truncates" in description

    def test_summary_reports_the_limits_in_force(self, server_module, config):
        """The result is the other channel the agent is guaranteed to see."""
        frames = [FakeFrame(0, 0.0)]
        server, _core = make_server(server_module, config, frames=frames)
        blocks = server.tools["view_frames"]["func"]("v.mp4")
        summary = json.loads(blocks[-1].text)
        assert summary["limits"] == {
            "max_images_per_call": config.max_images_per_call,
            "max_image_bytes": config.max_image_bytes,
            "max_total_bytes": config.max_total_bytes,
        }

    def test_video_info_description_warns_about_the_time_limit(
        self, server_module, config
    ):
        """The full scan is slow; the client's call timeout is invisible.

        The server cannot read the client's timeout — MCP does not carry it in
        `initialize` — so the description must warn without asserting a number.
        A concrete figure here would be wrong for any deployment that raised
        the limit, which is a normal thing to configure.
        """
        server, _core = make_server(server_module, config)
        description = server.tools["video_info"]["description"].lower()
        assert "time limit" in description
        for hardcoded in ("60 second", "60000", "60_000", "1 minute"):
            assert hardcoded not in description, description


class TestToolRegistration:
    def test_exactly_two_tools_are_exposed(self, server_module, config):
        server, _core = make_server(server_module, config)
        assert set(server.tools) == {"view_frames", "video_info"}

    def test_view_frames_parameters_match_the_spec(self, server_module, config):
        server, _core = make_server(server_module, config)
        params = server.tools["view_frames"]["schema"]
        assert set(params) == {
            "video",
            "start",
            "end",
            "interval",
            "max_frames",
            "max_edge",
            "format",
            "quality",
        }

    def test_video_info_parameters_match_the_spec(self, server_module, config):
        server, _core = make_server(server_module, config)
        params = server.tools["video_info"]["schema"]
        assert set(params) == {"video", "detail", "force"}

    def test_instructions_forbid_coverage_tracking(self, server_module, config):
        server, _core = make_server(server_module, config)
        # The module docstring is the guard against re-adding coverage
        # tracking; keep it honest.
        doc = server_module.__doc__ or ""
        assert "does NOT track which ranges" in doc


class TestArgumentBoundsReachTheSchema:
    """The bounds the server enforces must also be visible before the call.

    A parameter declared as a bare ``number`` tells the caller nothing, so the
    only way to learn that ``interval`` floors at 0.05 is to be rejected.  The
    obvious inference — "want more frames, shrink the interval" — walks
    straight into that rejection, which makes the omission expensive rather
    than cosmetic.

    The bounds are declared once, in :mod:`mcp_video_frames.budget`, and reused
    here; these tests fail if the two ever drift apart.
    """

    @staticmethod
    def _hint(server, tool: str, param: str):
        """The parameter's annotation with metadata, unwrapped from ``| None``.

        A default of ``None`` makes the annotation ``Optional[Annotated[...]]``,
        and ``Annotated`` metadata sits on the *inner* type, so the union has
        to be peeled off before the bounds are visible.
        """
        import typing

        hints = typing.get_type_hints(
            server.tools[tool]["func"], include_extras=True
        )
        annotation = hints[param]
        if getattr(annotation, "__metadata__", None) is None:
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                annotation = args[0]
        return annotation

    @classmethod
    def _bounds(cls, server, tool: str, param: str) -> dict[str, object]:
        """The ``ge``/``le`` values declared for one parameter.

        pydantic 2.13 stores them as ``Ge``/``Le`` constraint objects inside
        ``FieldInfo.metadata`` rather than as ``FieldInfo.ge`` attributes —
        which is exactly where its JSON Schema generator reads them, so this
        walks the same place.
        """
        annotation = cls._hint(server, tool, param)
        declared: dict[str, object] = {}
        for meta in getattr(annotation, "__metadata__", ()):
            sources = [meta, *getattr(meta, "metadata", ())]
            for source in sources:
                for attr in ("ge", "le"):
                    value = getattr(source, attr, None)
                    if value is not None:
                        declared[attr] = value
        return declared

    def test_interval_declares_the_validated_range(self, server_module, config):
        from mcp_video_frames import budget

        server, _core = make_server(server_module, config)
        declared = self._bounds(server, "view_frames", "interval")
        assert declared["ge"] == budget.MIN_INTERVAL
        assert declared["le"] == budget.MAX_INTERVAL

    def test_max_edge_declares_the_validated_range(self, server_module, config):
        from mcp_video_frames import budget

        server, _core = make_server(server_module, config)
        declared = self._bounds(server, "view_frames", "max_edge")
        assert declared["ge"] == budget.MIN_MAX_EDGE
        assert declared["le"] == budget.MAX_MAX_EDGE

    def test_quality_declares_the_validated_range(self, server_module, config):
        from mcp_video_frames import budget

        server, _core = make_server(server_module, config)
        declared = self._bounds(server, "view_frames", "quality")
        assert declared["ge"] == budget.MIN_QUALITY
        assert declared["le"] == budget.MAX_QUALITY

    def test_max_frames_declares_only_its_floor(self, server_module, config):
        """Its ceiling is the deployment limit, which varies, so it is omitted.

        Declaring a fixed ceiling here would be wrong for any deployment that
        raised ``MCP_VIDEO_MAX_IMAGES``.
        """
        from mcp_video_frames import budget

        server, _core = make_server(server_module, config)
        declared = self._bounds(server, "view_frames", "max_frames")
        assert declared == {"ge": budget.MIN_MAX_FRAMES}

    def test_format_is_an_enum_not_a_free_string(self, server_module, config):
        import typing

        from mcp_video_frames import budget

        server, _core = make_server(server_module, config)
        inner = self._hint(server, "view_frames", "format")
        assert set(typing.get_args(inner)) == set(budget.VALID_FORMATS)

    def test_video_info_detail_is_an_enum(self, server_module, config):
        import typing

        server, _core = make_server(server_module, config)
        hint = self._hint(server, "video_info", "detail")
        assert set(typing.get_args(hint)) == {"basic", "full"}

    def test_validated_bounds_are_not_duplicated_as_literals(self, config):
        """The bounds live in one place; nothing re-states them as a number.

        Guards the failure mode this whole class exists for: a schema that
        hardcodes 0.05 while the validator is changed to 0.1.
        """
        import inspect

        from mcp_video_frames import budget, server

        source = inspect.getsource(server)
        for value in (
            budget.MIN_INTERVAL,
            budget.MAX_INTERVAL,
            budget.MIN_MAX_EDGE,
            budget.MAX_MAX_EDGE,
            budget.MIN_QUALITY,
            budget.MAX_QUALITY,
        ):
            assert f"Field(ge={value}" not in source
            assert f"Field(le={value}" not in source


class TestViewFramesDescriptionStatesRequestRules:
    def test_max_frames_default_is_stated(self, server_module, config):
        """The schema says ``default: null``; the effective default is 12.

        Leaving that unsaid is the same class of bug as a missing bound: the
        caller plans against the wrong ceiling.
        """
        server, _core = make_server(server_module, config)
        description = server.tools["view_frames"]["description"]
        assert "max_frames defaults to 12" in description

    def test_the_caller_is_told_the_limits_are_not_theirs(self, server_module, config):
        server, _core = make_server(server_module, config)
        description = server.tools["view_frames"]["description"]
        assert "set by whoever deployed this server" in description


class TestContentBlockOrder:
    def test_five_frames_give_eleven_blocks_alternating(self, server_module, config):
        frames = [FakeFrame(n, float(n)) for n in range(5)]
        server, _core = make_server(server_module, config, frames=frames)
        blocks = server.tools["view_frames"]["func"]("v.mp4", start=0.0, end=4.0, interval=1.0)

        assert len(blocks) == 11
        kinds = [block.type for block in blocks]
        assert kinds == ["text", "image"] * 5 + ["text"]

    def test_each_timecode_precedes_its_own_image(self, server_module, config):
        frames = [FakeFrame(n, float(n)) for n in range(3)]
        server, _core = make_server(server_module, config, frames=frames)
        blocks = server.tools["view_frames"]["func"]("v.mp4")

        for index, frame in enumerate(frames):
            timecode = json.loads(blocks[2 * index].text)
            assert timecode == {"n": index, "t": float(index)}
            assert blocks[2 * index + 1].type == "image"
            assert blocks[2 * index + 1].data == base64.b64encode(frame.data).decode()

    def test_summary_is_last_and_carries_the_required_fields(
        self, server_module, config
    ):
        frames = [FakeFrame(n, float(n)) for n in range(2)]
        server, _core = make_server(server_module, config, frames=frames)
        blocks = server.tools["view_frames"]["func"]("v.mp4", interval=1.0)

        summary = json.loads(blocks[-1].text)
        assert blocks[-1].type == "text"
        for key in (
            "video",
            "frames",
            "range",
            "interval",
            "format",
            "max_edge",
            "clamped",
            "cached",
        ):
            assert key in summary, key
        assert summary["frames"] == 2
        assert summary["cached"] == 0

    def test_single_frame_gives_three_blocks(self, server_module, config):
        server, _core = make_server(
            server_module, config, frames=[FakeFrame(0, 3.0, data=JPEG_BYTES, mime="image/jpeg")]
        )
        blocks = server.tools["view_frames"]["func"]("v.mp4", start=3.0, end=3.0)
        assert [block.type for block in blocks] == ["text", "image", "text"]

    def test_image_mime_type_matches_the_payload(self, server_module, config):
        server, _core = make_server(
            server_module,
            config,
            frames=[FakeFrame(0, 0.0, data=JPEG_BYTES, mime="image/jpeg")],
        )
        blocks = server.tools["view_frames"]["func"]("v.mp4")
        assert blocks[1].mimeType == "image/jpeg"

    def test_blocks_are_sdk_content_objects_not_a_serialised_list(
        self, server_module, config
    ):
        """The list must pass through as blocks, not be JSON-encoded into one.

        A tool returning a plain list of dicts would be serialised into a
        single text block by most SDKs, which is exactly the failure this
        design cannot tolerate.
        """
        from conftest import FakeImageContent, FakeTextContent

        frames = [FakeFrame(n, float(n)) for n in range(2)]
        server, _core = make_server(server_module, config, frames=frames)
        blocks = server.tools["view_frames"]["func"]("v.mp4")
        assert all(
            isinstance(block, (FakeTextContent, FakeImageContent)) for block in blocks
        )
        assert not any(isinstance(block, (str, dict)) for block in blocks)

    def test_content_block_count_is_asserted(self, server_module, config, monkeypatch):
        """A miscount must raise, not ship a mismatched sequence."""
        from mcp_video_frames import core as core_module

        frames = [FakeFrame(0, 0.0)]
        server, _core = make_server(server_module, config, frames=frames)

        class Liar(list):
            def __iter__(self):
                return iter([])

        monkeypatch.setattr(
            core_module, "build_content_blocks", lambda result: Liar()
        )
        with pytest.raises(Exception) as excinfo:
            server.tools["view_frames"]["func"]("v.mp4")
        assert "content blocks" in str(excinfo.value)


class TestArgumentPassing:
    def test_arguments_reach_the_core_unchanged(self, server_module, config):
        server, core = make_server(server_module, config)
        server.tools["view_frames"]["func"](
            "v.mp4",
            start=2.0,
            end=6.0,
            interval=0.5,
            max_frames=5,
            max_edge=512,
            format="jpeg",
            quality=70,
        )
        _name, kwargs = core.calls[-1]
        assert kwargs == {
            "video": "v.mp4",
            "start": 2.0,
            "end": 6.0,
            "interval": 0.5,
            "max_frames": 5,
            "max_edge": 512,
            "image_format": "jpeg",
            "quality": 70,
        }

    def test_defaults_are_the_documented_ones(self, server_module, config):
        server, core = make_server(server_module, config)
        server.tools["view_frames"]["func"]("v.mp4")
        _name, kwargs = core.calls[-1]
        assert kwargs["start"] == 0.0
        assert kwargs["end"] is None
        assert kwargs["interval"] == 1.0
        assert kwargs["max_frames"] is None
        assert kwargs["max_edge"] == 768
        assert kwargs["image_format"] is None
        assert kwargs["quality"] == 85

    def test_public_parameter_names_match_the_spec(self, server_module, config):
        # The MCP parameter is named `format`; the core uses `image_format`
        # because `format` shadows a builtin.  That mapping must not leak.
        server, _core = make_server(server_module, config)
        signature = server.tools["view_frames"]["signature"]
        assert "format" in signature.parameters
        assert "image_format" not in signature.parameters
        assert signature.parameters["format"].default is None

    def test_format_default_is_left_to_the_core(self, server_module, config):
        # jpeg vs png depends on whether start == end, so the core decides.
        server, core = make_server(server_module, config)
        server.tools["view_frames"]["func"]("v.mp4", start=1.0, end=1.0)
        assert core.calls[-1][1]["image_format"] is None


class TestErrorSemantics:
    """What the client sees when a call fails.

    The raised *class* is deliberately not asserted everywhere here: it depends
    on which SDK generation is loaded (``ToolError`` when available,
    ``RuntimeError`` otherwise).  What must hold either way is that the text
    survives — that is the property callers depend on.
    """

    def test_errors_are_raised_as_the_sdk_tool_error(self, server_module, config):
        """The error text must survive to the client.

        This is the regression test for a real deployment bug: raising a plain
        exception made mcp 2.x treat it as a crash and replace the message with
        "Error executing tool view_frames", discarding the numbers and the
        suggested fix.  Only the SDK's own ToolError is passed through.
        """
        from conftest import FakeToolError

        server, _core = make_server(
            server_module,
            config,
            raise_on={
                "view_frames": InputError(
                    "interval=-1s is below the minimum of 0.05s."
                )
            },
        )
        with pytest.raises(FakeToolError) as excinfo:
            server.tools["view_frames"]["func"]("v.mp4")
        # Still our text, not a sanitised placeholder.
        assert "interval=-1" in str(excinfo.value)
        assert "Error executing tool" not in str(excinfo.value)

    def test_error_text_is_not_sanitised_away(self, server_module, config):
        """The full guidance — numbers plus next step — reaches the client."""
        from conftest import FakeToolError

        message = (
            "Request 0-600s at interval 0.25 yields 2401 frames, over the "
            "per-call limit of 12.\n"
            "Options:\n  - raise interval to >= 0.91"
        )
        server, _core = make_server(
            server_module, config, raise_on={"view_frames": LimitError(message)}
        )
        with pytest.raises(FakeToolError) as excinfo:
            server.tools["view_frames"]["func"]("v.mp4")
        text = str(excinfo.value)
        assert "2401 frames" in text
        assert "interval" in text

    def test_falls_back_when_the_sdk_has_no_tool_error(self, monkeypatch, config):
        """An SDK without ToolError must still start and still report."""
        install_fake_mcp_sdk(monkeypatch, with_tool_error=False)
        import importlib

        import mcp_video_frames.server as server

        server = importlib.reload(server)
        core = FakeCore(config, raise_on={"view_frames": InputError("broken")})
        built = server.build_server(core)
        with pytest.raises(Exception) as excinfo:
            built.tools["view_frames"]["func"]("v.mp4")
        assert "broken" in str(excinfo.value)

    def test_over_limit_raises_and_mentions_the_fix(self, server_module, config):
        server, _core = make_server(
            server_module,
            config,
            raise_on={
                "view_frames": LimitError(
                    "Request 0-60s at interval 1 yields 61 frames, over the "
                    "per-call limit of 12.\n"
                    "Options:\n  - raise interval to >= 5.46"
                )
            },
        )
        with pytest.raises(Exception) as excinfo:
            server.tools["view_frames"]["func"]("v.mp4", start=0.0, end=60.0)
        message = str(excinfo.value)
        assert "61 frames" in message
        assert "interval" in message

    def test_error_text_carries_no_traceback(self, server_module, config):
        server, _core = make_server(
            server_module,
            config,
            raise_on={
                "view_frames": InputError(
                    "interval=-1s is below the minimum of 0.05s."
                )
            },
        )
        with pytest.raises(Exception) as excinfo:
            server.tools["view_frames"]["func"]("v.mp4")
        message = str(excinfo.value)
        assert "Traceback" not in message
        assert "File \"" not in message

    def test_error_detail_is_appended(self, server_module, config):
        from mcp_video_frames.errors import FfmpegError

        server, _core = make_server(
            server_module,
            config,
            raise_on={
                "view_frames": FfmpegError(
                    "ffmpeg failed.", detail="No such file or directory"
                )
            },
        )
        with pytest.raises(Exception) as excinfo:
            server.tools["view_frames"]["func"]("v.mp4")
        assert "No such file or directory" in str(excinfo.value)

    def test_video_info_reports_errors_as_json_not_an_exception(
        self, server_module, config
    ):
        server, _core = make_server(
            server_module,
            config,
            raise_on={"basic": InputError("Video file does not exist: x.mp4.")},
        )
        payload = json.loads(server.tools["video_info"]["func"]("x.mp4"))
        assert "Video file does not exist" in payload["error"]
        assert payload["kind"] == "InputError"


class TestVideoInfoDetail:
    def test_basic_is_the_default(self, server_module, config):
        server, core = make_server(server_module, config)
        payload = json.loads(server.tools["video_info"]["func"]("v.mp4"))
        assert core.calls[-1][0] == "video_basic_info"
        assert "scene_cuts" not in payload

    def test_full_scans_whole_file(self, server_module, config):
        server, core = make_server(server_module, config)
        payload = json.loads(server.tools["video_info"]["func"]("v.mp4", detail="full"))
        assert core.calls[-1][0] == "video_full_info"
        assert payload["scene_cuts"] == [5.0]

    def test_force_is_forwarded(self, server_module, config):
        server, core = make_server(server_module, config)
        server.tools["video_info"]["func"]("v.mp4", detail="full", force=True)
        assert core.calls[-1][1]["force"] is True

    def test_unknown_detail_is_rejected_with_the_valid_values(
        self, server_module, config
    ):
        server, _core = make_server(server_module, config)
        payload = json.loads(server.tools["video_info"]["func"]("v.mp4", detail="deep"))
        assert "basic" in payload["error"] and "full" in payload["error"]

    def test_detail_is_case_insensitive(self, server_module, config):
        server, core = make_server(server_module, config)
        server.tools["video_info"]["func"]("v.mp4", detail="FULL")
        assert core.calls[-1][0] == "video_full_info"


class TestStartupLogging:
    def test_cache_path_is_printed_to_stderr(self, server_module, config, capsys):
        make_server(server_module, config)
        captured = capsys.readouterr()
        assert str(config.cache_root) in captured.err
        assert captured.out == ""
        assert "MCP_VIDEO_MAX_IMAGES" in captured.err

    def test_startup_self_check_reports_problems_without_dying(
        self, server_module, config, capsys
    ):
        """A broken environment must be diagnosable from inside the client.

        Exiting at startup leaves the client with "the command exited" and
        nothing else; starting and reporting gives the operator something to
        act on.
        """
        core = FakeCore(config)
        core.startup_check = lambda: ["ffmpeg not found. Install ffmpeg."]
        server = server_module.build_server(core)
        captured = capsys.readouterr()
        assert "ffmpeg not found" in captured.err
        assert set(server.tools) == {"view_frames", "video_info"}

    def test_self_check_runs_before_any_tool_call(self, server_module, config, capsys):
        calls: list[bool] = []
        core = FakeCore(config)
        core.startup_check = lambda: calls.append(True) or []
        server_module.build_server(core)
        assert calls == [True]
        assert core.calls == []


class _RecordingMCP:
    """Stand-in that records how ``run`` was called."""

    def __init__(self, accept_transport: bool):
        self.calls: list[dict] = []
        self._accept = accept_transport

    def run(self, **kwargs):
        if kwargs and not self._accept:
            raise TypeError("run() got an unexpected keyword argument 'transport'")
        self.calls.append(kwargs)


class TestServeStdio:
    def test_transport_is_named_explicitly(self, server_module, config, monkeypatch):
        recorder = _RecordingMCP(accept_transport=True)
        monkeypatch.setattr(server_module, "build_server", lambda core=None: recorder)
        assert server_module.serve_stdio() == 0
        assert recorder.calls == [{"transport": "stdio"}]

    def test_older_sdk_without_transport_still_serves(
        self, server_module, config, monkeypatch
    ):
        recorder = _RecordingMCP(accept_transport=False)
        monkeypatch.setattr(server_module, "build_server", lambda core=None: recorder)
        assert server_module.serve_stdio() == 0
        assert recorder.calls == [{}], "the fallback call must still run the server"
