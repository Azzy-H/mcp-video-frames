"""Parsers for ffmpeg scan output, tested against fixed samples.

These are the tests that keep "found nothing" distinguishable from "did not
understand the output".  Real ffmpeg stderr changes wording between versions,
so the samples below are pinned copies of what each filter actually prints —
including the progress lines that must be skipped.
"""

from __future__ import annotations

import pytest

from mcp_video_frames.errors import FfmpegError
from mcp_video_frames.scan import (
    parse_loudness,
    parse_scene_cuts,
    parse_silences,
)

# --- fixed samples ----------------------------------------------------
SCENE_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
  built with gcc 13.2.0
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
  Duration: 00:00:10.00, start: 0.000000, bitrate: 210 kb/s
    Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 320x240
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> wrapped_avframe (native))
Press [q] to stop, [?] for help
[Parsed_showinfo_1 @ 0x55d1] n:   0 pts:      0 pts_time:0       duration:1 ...
[Parsed_showinfo_1 @ 0x55d1] n:   1 pts:   1024 pts_time:0.0333333 duration:1 ...
[Parsed_showinfo_1 @ 0x55d1] n: 150 pts: 153600 pts_time:5       duration:1 ...
[Parsed_showinfo_1 @ 0x55d1] n: 151 pts: 154624 pts_time:5.03333 duration:1 ...
frame=  300 fps=120 q=-0.0 Lsize=N/A time=00:00:10.00 bitrate=N/A speed=4.02x
video:148kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
"""

SILENCE_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
  Duration: 00:00:10.00, start: 0.000000, bitrate: 210 kb/s
    Stream #0:0[0x1](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp
Output #0, null, to 'pipe:':
Stream mapping:
  Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
[silencedetect @ 0x55d1] silence_start: 1.998
[silencedetect @ 0x55d1] silence_end: 3.5 | silence_duration: 1.502
[silencedetect @ 0x55d1] silence_start: 6.2
[silencedetect @ 0x55d1] silence_end: 6.8 | silence_duration: 0.6
size=N/A time=00:00:10.01 bitrate=N/A speed= 583x
"""

SILENCE_NO_EVENTS_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
Stream mapping:
  Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))
size=N/A time=00:00:10.01 bitrate=N/A speed= 583x
"""

SILENCE_UNCLOSED_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
[silencedetect @ 0x55d1] silence_start: 7.25
size=N/A time=00:00:10.01 bitrate=N/A speed= 583x
"""

LOUDNESS_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
Stream mapping:
  Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))
[Parsed_ebur128_0 @ 0x55d1] t: 0.699979   M: -22.2 S: -120.7     I: -22.2 LUFS       LRA:   0.0 LU
[Parsed_ebur128_0 @ 0x55d1] t: 1.399979   M: -22.2 S: -22.2     I: -22.2 LUFS       LRA:   0.0 LU
[Parsed_ebur128_0 @ 0x55d1] t: 9.999979   M: -23.0 S: -22.9     I: -23.0 LUFS       LRA:   4.0 LU
size=N/A time=00:00:10.01 bitrate=N/A speed= 583x

Summary:

  Integrated loudness:
    I:         -23.0 LUFS
    Threshold: -33.0 LUFS

  Loudness range:
    LRA:         5.0 LU
    Threshold: -43.0 LUFS
    LRA low:   -25.5 LUFS
    LRA high:  -20.5 LUFS

  True peak:
    Peak:       -1.2 dBFS
"""

LOUDNESS_SILENT_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
[Parsed_ebur128_0 @ 0x55d1] t: 9.999979   M: -inf S: -inf     I: -inf LUFS       LRA:   0.0 LU
size=N/A time=00:00:10.01 bitrate=N/A speed= 583x

Summary:

  Integrated loudness:
    I:          -inf LUFS
"""

UNPARSABLE_STDERR = """\
ffmpeg version 7.0 Copyright (c) 2000-2024 the FFmpeg developers
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'test.mp4':
Something entirely unexpected happened and this line means nothing to us.
"""


class TestSceneCuts:
    def test_extracts_pts_time(self):
        assert parse_scene_cuts(SCENE_STDERR) == [0.0, 0.033, 5.0, 5.033]

    def test_ignores_showinfo_progress_noise(self):
        cuts = parse_scene_cuts(SCENE_STDERR)
        assert all(0.0 <= c <= 10.0 for c in cuts)

    def test_drops_events_beyond_the_known_duration(self):
        # A stray line from a longer file must not invent a timestamp.
        noisy = SCENE_STDERR + (
            "[Parsed_showinfo_1 @ 0x55d1] n: 900 pts: 921600 pts_time:30 \n"
        )
        assert parse_scene_cuts(noisy, duration=10.0) == [0.0, 0.033, 5.0, 5.033]

    def test_unparsable_output_raises_instead_of_returning_empty(self):
        with pytest.raises(FfmpegError) as excinfo:
            parse_scene_cuts(UNPARSABLE_STDERR)
        assert "showinfo" in str(excinfo.value)

    def test_empty_output_raises(self):
        with pytest.raises(FfmpegError):
            parse_scene_cuts("")

    def test_no_cuts_but_filter_ran_is_an_empty_list(self):
        # The filter ran (frame progress present) and found nothing: that is
        # a real answer, and it must differ from a parse failure.
        output = (
            "ffmpeg version 6.1.1\n"
            "frame=  300 fps=120 q=-0.0 Lsize=N/A time=00:00:10.00\n"
        )
        assert parse_scene_cuts(output) == []


class TestSilences:
    def test_extracts_intervals(self):
        assert parse_silences(SILENCE_STDERR) == [
            {"start": 1.998, "end": 3.5, "duration": 1.502},
            {"start": 6.2, "end": 6.8, "duration": 0.6},
        ]

    def test_none_found_is_an_empty_list(self):
        assert parse_silences(SILENCE_NO_EVENTS_STDERR) == []

    def test_unclosed_silence_is_reported_without_a_guessed_end(self):
        # The filter was flushed mid-silence; inventing an end would be a
        # silent fabrication.
        result = parse_silences(SILENCE_UNCLOSED_STDERR)
        assert result == [{"start": 7.25, "end": None, "duration": None}]

    def test_unparsable_output_raises(self):
        with pytest.raises(FfmpegError):
            parse_silences(UNPARSABLE_STDERR)

    def test_empty_output_raises(self):
        with pytest.raises(FfmpegError):
            parse_silences("")

    def test_negative_start_is_clamped_to_zero(self):
        output = "[silencedetect @ 0x1] silence_start: -0.02\n"
        output += "[silencedetect @ 0x1] silence_end: 1.0 | silence_duration: 1.02\n"
        assert parse_silences(output)[0]["start"] == 0.0

    def test_zero_length_silence_is_dropped(self):
        output = "[silencedetect @ 0x1] silence_start: 5.0\n"
        output += "[silencedetect @ 0x1] silence_end: 5.0 | silence_duration: 0.0\n"
        assert parse_silences(output) == []


class TestLoudness:
    def test_takes_the_summary_values(self):
        result = parse_loudness(LOUDNESS_STDERR)
        assert result["integrated_lufs"] == -23.0
        assert result["lra"] == 5.0
        assert result["true_peak_dbfs"] == -1.2

    def test_uses_the_last_occurrence_not_the_first(self):
        # Progress lines print partial figures; only the final summary is
        # the whole-file value.
        result = parse_loudness(LOUDNESS_STDERR)
        assert result["integrated_lufs"] != -22.2

    def test_digital_silence_is_recognised(self):
        result = parse_loudness(LOUDNESS_SILENT_STDERR)
        assert result["integrated_lufs"] == float("-inf")

    def test_unparsable_output_raises(self):
        with pytest.raises(FfmpegError) as excinfo:
            parse_loudness(UNPARSABLE_STDERR)
        assert "loudness" in str(excinfo.value)

    def test_empty_output_raises(self):
        with pytest.raises(FfmpegError):
            parse_loudness("")

    def test_no_audio_stream_is_an_empty_result_not_an_error(self):
        output = (
            "ffmpeg version 6.1.1\n"
            "Output file does not contain any stream\n"
        )
        assert parse_loudness(output) == {}


class TestParserContract:
    """The distinction the design hinges on: empty vs. unparsable."""

    @pytest.mark.parametrize(
        "parser",
        [parse_scene_cuts, parse_silences, parse_loudness],
    )
    def test_garbage_never_yields_an_empty_result(self, parser):
        with pytest.raises(FfmpegError):
            parser(UNPARSABLE_STDERR)

    @pytest.mark.parametrize(
        "parser",
        [parse_scene_cuts, parse_silences, parse_loudness],
    )
    def test_failure_carries_the_actual_output(self, parser):
        with pytest.raises(FfmpegError) as excinfo:
            parser(UNPARSABLE_STDERR)
        detail = excinfo.value.detail or ""
        assert "unexpected" in detail
