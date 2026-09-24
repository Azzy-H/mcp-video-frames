"""Frame-count, limit and byte-budget arithmetic (no ffmpeg needed)."""

from __future__ import annotations

import pytest

from mcp_video_frames.budget import (
    PlanRequest,
    check_total_budget,
    compute_frame_count,
    default_format,
    estimate_frame_bytes,
    estimate_payload_bytes,
    limit_error,
    plan_frames,
    qscale_for_quality,
    sampled_count,
)
from mcp_video_frames.errors import InputError, LimitError


class TestTimeoutBudgetIsConsistent:
    """A lock must outlast every legitimate hold of it.

    ``video_full_info`` scans the whole file while holding the per-video lock,
    so the lock timeout has to cover the whole scan.  These were once
    independent constants — 120 s for the lock against 12 h per scan pass —
    which meant a second client touching the same long video waited 120 s and
    then failed, contradicting the documented promise that clients share one
    cache.
    """

    def test_lock_timeout_covers_the_whole_scan(self):
        from mcp_video_frames.config import SCAN_PHASE_TIMEOUT, VIDEO_LOCK_TIMEOUT

        # Three sequential passes (scene, silence, loudness), plus slack for the
        # extraction around them.
        assert VIDEO_LOCK_TIMEOUT > 3 * SCAN_PHASE_TIMEOUT

    def test_scan_passes_share_one_constant(self):
        from mcp_video_frames import scan
        from mcp_video_frames.config import SCAN_PHASE_TIMEOUT

        for name in ("SCENE_TIMEOUT", "SILENCE_TIMEOUT", "LOUDNESS_TIMEOUT"):
            assert getattr(scan, name) == SCAN_PHASE_TIMEOUT

    def test_cache_lock_default_is_the_configured_one(self):
        from mcp_video_frames.cache import Cache
        from mcp_video_frames.config import VIDEO_LOCK_TIMEOUT

        import inspect

        default = inspect.signature(Cache.locked).parameters["timeout"].default
        assert default == VIDEO_LOCK_TIMEOUT

    def test_fallback_stale_lock_outlasts_a_scan(self):
        """The O_EXCL fallback must not reap a lock a live scan is holding."""
        from mcp_video_frames import _locks
        from mcp_video_frames.config import VIDEO_LOCK_TIMEOUT

        if _locks.HAVE_FILELOCK:
            pytest.skip("filelock is installed; the fallback class is not in use")
        assert _locks.FileLock.STALE_SECONDS > VIDEO_LOCK_TIMEOUT


class TestFrameCount:
    @pytest.mark.parametrize(
        ("start", "end", "interval", "expected"),
        [
            (0.0, 4.0, 1.0, 5),  # inclusive endpoints: 0,1,2,3,4
            (0.0, 0.0, 1.0, 1),  # single frame
            (5.0, 5.0, 1.0, 1),
            (0.0, 60.0, 1.0, 61),
            (0.0, 10.0, 2.0, 6),  # 0,2,4,6,8,10
            (0.0, 9.9, 2.0, 5),  # floor(9.9/2)=4 -> 5
            (1.0, 1.5, 1.0, 1),  # interval wider than the range
            (0.0, 0.5, 0.05, 11),  # 0.0..0.5 step 0.05
            (0.0, 0.04, 0.05, 1),
            (10.0, 10.0, 600.0, 1),
        ],
    )
    def test_counts(self, start, end, interval, expected):
        assert compute_frame_count(start, end, interval) == expected

    def test_reversed_range_is_one_frame_not_negative(self):
        # Negative ranges are rejected upstream; the arithmetic itself must
        # not invent frames.
        assert compute_frame_count(5.0, 1.0, 1.0) == 1

    def test_float_accumulation_does_not_lose_a_frame(self):
        # 0.1 * 30 has representation error; the count must not become 2.
        assert compute_frame_count(0.0, 0.3, 0.1) == 4


class TestSampledCount:
    """The plan is an upper bound; extraction truncates to what exists.

    A real ffmpeg build produces, for a 10.2 s clip at 1 s intervals:
    start 0 -> 10 frames, 3 -> 7, 5 -> 5, 7 -> 3, 9 -> 1.  Which of those the
    last grid point reaches depends on the final frame's duration, so the plan
    deliberately counts the grid point and the *verified* file count decides
    whether it is reported.  Over-counting loses a frame; under-counting would
    silently drop one the caller asked for.
    """

    @pytest.mark.parametrize(
        ("start", "end", "interval", "expected"),
        [
            (0.0, 10.2, 1.0, 11),
            (3.0, 10.2, 1.0, 8),
            (5.0, 10.2, 1.0, 6),
            (7.0, 10.2, 1.0, 4),
            (9.0, 10.2, 1.0, 2),
            (0.0, 10.0, 1.0, 11),
            (0.0, 9.0, 1.0, 10),
            (0.0, 4.0, 1.0, 5),
            (0.0, 20.0, 1.0, 21),
            (0.0, 5.0, 0.5, 11),
            (0.0, 1.0, 0.25, 5),
            (0.0, 0.5, 0.05, 11),
            (2.0, 2.0, 1.0, 1),  # single frame is always one frame
            (0.0, 0.04, 0.05, 1),
        ],
    )
    def test_counts(self, start, end, interval, expected):
        assert sampled_count(start, end, interval) == expected
        assert sampled_count(start, end, interval) == compute_frame_count(
            start, end, interval
        )

    def test_never_under_counts(self):
        # Every frame the caller asked for must be planned for.
        for end in (1.0, 4.0, 9.9, 10.0, 10.2):
            assert sampled_count(0.0, end, 1.0) >= compute_frame_count(0.0, end, 1.0)


class TestDefaultFormat:
    def test_range_defaults_to_jpeg(self):
        assert default_format(0.0, 10.0) == "jpeg"

    def test_single_frame_defaults_to_png(self):
        assert default_format(3.0, 3.0) == "png"


class TestLimits:
    def test_exactly_at_limit_is_allowed(self):
        plan = plan_frames(
            PlanRequest(start=0.0, end=19.0, interval=1.0, max_frames=20),
            max_images_per_call=20,
        )
        assert plan.count == 20

    def test_one_frame_over_limit_fails(self):
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=20.0, interval=1.0, max_frames=20),
                max_images_per_call=20,
            )
        assert "21 frames" in str(excinfo.value)
        assert "20" in str(excinfo.value)

    def test_default_max_frames_is_twelve(self):
        # The per-call default is narrower than the server ceiling.
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=30.0, interval=1.0),
                max_images_per_call=20,
            )
        assert "per-call limit of 12" in str(excinfo.value)

    def test_clamping_does_not_turn_a_feasible_call_into_an_error(self):
        # 2-30 s is 29 frames, but only 10 fit in a 12 s clip.
        plan = plan_frames(
            PlanRequest(start=2.0, end=30.0, interval=1.0),
            max_images_per_call=20,
            duration=12.0,
        )
        assert plan.clamped is True
        assert plan.count == 10

    def test_error_mentions_the_original_request_when_clamped(self):
        # Asking for 20 minutes of a 10 minute video: the headline quotes the
        # request, the suggestions describe the clamped range.
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=1200.0, interval=1.0),
                max_images_per_call=20,
                duration=600.0,
            )
        message = str(excinfo.value)
        assert "clamped" in message
        assert "1200" in message
        assert "1201 frames" in message

    def test_error_offers_an_interval_that_would_fit(self):
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=60.0, interval=1.0, max_frames=20),
                max_images_per_call=20,
            )
        message = str(excinfo.value)
        # The suggestion must actually work: re-plan with it and pass.
        assert "interval" in message
        suggested = _extract_suggested_interval(message)
        plan = plan_frames(
            PlanRequest(start=0.0, end=60.0, interval=suggested, max_frames=20),
            max_images_per_call=20,
        )
        assert plan.count <= 20

    def test_error_offers_a_workable_split(self):
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=60.0, interval=1.0, max_frames=20),
                max_images_per_call=20,
            )
        message = str(excinfo.value)
        # 61 frames / 20 per call = 4 calls, 16 frames each.
        assert "split into 4 calls" in message
        assert "0-15" in message
        assert "45-60" in message

    def test_split_suggestion_is_itself_valid(self):
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=60.0, interval=1.0, max_frames=20),
                max_images_per_call=20,
            )
        spans = _extract_split(str(excinfo.value))
        assert len(spans) >= 2
        for low, high in spans:
            plan = plan_frames(
                PlanRequest(start=low, end=high, interval=1.0, max_frames=20),
                max_images_per_call=20,
            )
            assert plan.count <= 20, (low, high, plan.count)

    def test_error_mentions_configuration(self):
        # limit_error is the message builder itself; it always raises.
        with pytest.raises(LimitError) as excinfo:
            limit_error(start=0.0, end=60.0, interval=1.0, count=61, limit=20)
        assert "MCP_VIDEO_MAX_IMAGES" in str(excinfo.value)
        assert "61 frames" in str(excinfo.value)
        assert "limit of 20" in str(excinfo.value)

    def test_far_over_limit_still_gives_concrete_numbers(self):
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=3600.0, interval=1.0),
                max_images_per_call=20,
            )
        assert "3601 frames" in str(excinfo.value)

    def test_max_frames_narrows_the_limit(self):
        with pytest.raises(LimitError) as excinfo:
            plan_frames(
                PlanRequest(start=0.0, end=10.0, interval=1.0, max_frames=5),
                max_images_per_call=20,
            )
        assert "per-call limit of 5" in str(excinfo.value)

    def test_max_frames_above_server_limit_is_a_parameter_error(self):
        with pytest.raises(InputError):
            plan_frames(
                PlanRequest(start=0.0, end=1.0, interval=1.0, max_frames=50),
                max_images_per_call=20,
            )

    def test_interval_is_never_widened_silently(self):
        # A passing plan must keep the requested interval to the digit.
        plan = plan_frames(
            PlanRequest(start=0.0, end=5.0, interval=0.5), max_images_per_call=20
        )
        assert plan.interval == 0.5
        assert [f.t for f in plan.frames] == [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
        ]

    def test_range_is_never_truncated_silently(self):
        with pytest.raises(LimitError):
            plan_frames(
                PlanRequest(start=0.0, end=60.0, interval=1.0, max_frames=20),
                max_images_per_call=20,
            )


class TestParameterValidation:
    @pytest.mark.parametrize("interval", [0.0499, -1.0, 601.0])
    def test_interval_out_of_range(self, interval):
        with pytest.raises(InputError):
            plan_frames(
                PlanRequest(start=0.0, end=1.0, interval=interval),
                max_images_per_call=20,
            )

    @pytest.mark.parametrize("interval", [0.05, 600.0])
    def test_interval_bounds_are_inclusive(self, interval):
        plan = plan_frames(
            PlanRequest(start=0.0, end=interval, interval=interval),
            max_images_per_call=20,
        )
        # Exactly one interval of span: the grid holds two points.
        assert plan.count == 2

    def test_negative_start(self):
        with pytest.raises(InputError):
            plan_frames(
                PlanRequest(start=-1.0, end=1.0, interval=1.0),
                max_images_per_call=20,
            )

    def test_end_before_start(self):
        with pytest.raises(InputError) as excinfo:
            plan_frames(
                PlanRequest(start=5.0, end=1.0, interval=1.0),
                max_images_per_call=20,
            )
        assert "single moment" in str(excinfo.value)

    @pytest.mark.parametrize("max_edge", [63, 4097, 0])
    def test_max_edge_range(self, max_edge):
        with pytest.raises(InputError):
            plan_frames(
                PlanRequest(start=0.0, end=1.0, interval=1.0, max_edge=max_edge),
                max_images_per_call=20,
            )

    @pytest.mark.parametrize("quality", [0, 101, -5])
    def test_quality_range(self, quality):
        with pytest.raises(InputError):
            plan_frames(
                PlanRequest(start=0.0, end=1.0, interval=1.0, quality=quality),
                max_images_per_call=20,
            )

    def test_unknown_format(self):
        with pytest.raises(InputError):
            plan_frames(
                PlanRequest(start=0.0, end=1.0, interval=1.0, image_format="webp"),
                max_images_per_call=20,
            )


class TestClamping:
    def test_end_beyond_duration_is_clamped_and_reported(self):
        plan = plan_frames(
            PlanRequest(start=0.0, end=100.0, interval=1.0),
            max_images_per_call=20,
            duration=10.0,
        )
        assert plan.clamped is True
        assert plan.end == pytest.approx(10.0, abs=1e-3)
        assert plan.requested_end == 100.0
        # A 10.0 s clip has no frame at 10.0: the grid holds 0..9.
        assert plan.count == 10

    def test_fully_outside_range_fails_after_clamping(self):
        with pytest.raises(InputError) as excinfo:
            plan_frames(
                PlanRequest(start=100.0, end=200.0, interval=1.0),
                max_images_per_call=20,
                duration=10.0,
            )
        assert "100" in str(excinfo.value)

    def test_single_moment_past_the_end_collapses_to_the_last_frame(self):
        # A point query is a legitimate question; answer it at the boundary.
        plan = plan_frames(
            PlanRequest(start=500.0, end=500.0, interval=1.0),
            max_images_per_call=20,
            duration=10.0,
        )
        assert plan.clamped is True
        assert plan.count == 1
        assert plan.start == 10.0

    def test_in_range_request_is_not_marked_clamped(self):
        plan = plan_frames(
            PlanRequest(start=0.0, end=10.0, interval=1.0),
            max_images_per_call=20,
            duration=10.0,
        )
        assert plan.clamped is False

    def test_clamped_request_within_duration_only_trims_the_end(self):
        plan = plan_frames(
            PlanRequest(start=2.0, end=30.0, interval=1.0, max_frames=20),
            max_images_per_call=20,
            duration=12.0,
        )
        assert plan.start == 2.0
        assert plan.end == pytest.approx(12.0, abs=1e-3)
        assert plan.count == 10  # t = 2..11; nothing starts at 12.0
        assert plan.clamped is True


class TestByteBudget:
    def test_jpeg_is_far_smaller_than_png(self):
        jpeg = estimate_frame_bytes(
            width=1920, height=1080, max_edge=768, image_format="jpeg"
        )
        png = estimate_frame_bytes(
            width=1920, height=1080, max_edge=768, image_format="png"
        )
        assert png > jpeg * 3

    def test_smaller_max_edge_means_fewer_bytes(self):
        big = estimate_frame_bytes(
            width=3840, height=2160, max_edge=1024, image_format="jpeg"
        )
        small = estimate_frame_bytes(
            width=3840, height=2160, max_edge=256, image_format="jpeg"
        )
        assert small < big

    def test_scaling_preserves_aspect_ratio(self):
        # Long edge maps to max_edge; the other edge shrinks proportionally.
        bytes_16_9 = estimate_frame_bytes(
            width=1920, height=1080, max_edge=768, image_format="jpeg"
        )
        bytes_9_16 = estimate_frame_bytes(
            width=1080, height=1920, max_edge=768, image_format="jpeg"
        )
        assert bytes_16_9 == bytes_9_16

    def test_frame_below_max_edge_is_not_upscaled(self):
        small = estimate_frame_bytes(
            width=320, height=240, max_edge=768, image_format="jpeg"
        )
        assert small < 320 * 240  # ~76k pixels worth, nowhere near upscaled

    def test_payload_includes_base64_overhead(self):
        assert estimate_payload_bytes(300, 10) == 4000

    def test_total_budget_fails_with_actionable_suggestions(self):
        with pytest.raises(LimitError) as excinfo:
            check_total_budget(
                frame_bytes=5 * 1024 * 1024,
                frame_count=20,
                max_total_bytes=40 * 1024 * 1024,
                max_edge=1024,
                image_format="png",
                interval=1.0,
                start=0.0,
                end=19.0,
            )
        message = str(excinfo.value)
        assert "max_edge" in message
        assert "jpeg" in message

    def test_plan_checks_the_total_budget_before_extracting(self):
        with pytest.raises(LimitError):
            plan_frames(
                PlanRequest(
                    start=0.0,
                    end=11.0,
                    interval=1.0,
                    max_edge=4096,
                    image_format="png",
                ),
                max_images_per_call=20,
                width=3840,
                height=2160,
                max_image_bytes=10 * 1024 * 1024,
                max_total_bytes=40 * 1024 * 1024,
            )


class TestQscale:
    def test_monotonic_and_bounded(self):
        values = [qscale_for_quality(q) for q in range(1, 101)]
        assert values == sorted(values, reverse=True)
        assert all(2 <= v <= 31 for v in values)

    def test_extremes(self):
        assert qscale_for_quality(100) == 2
        assert qscale_for_quality(1) == 31


def _extract_suggested_interval(message: str) -> float:
    import re

    match = re.search(r"raise interval to >= ([0-9.]+)", message)
    assert match, message
    return float(match.group(1))


def _extract_split(message: str) -> list[tuple[float, float]]:
    import re

    match = re.search(r"split into \d+ calls: (.+)", message)
    assert match, message
    spans = []
    for chunk in match.group(1).split(", "):
        low, _, high = chunk.partition("-")
        spans.append((float(low), float(high)))
    return spans
