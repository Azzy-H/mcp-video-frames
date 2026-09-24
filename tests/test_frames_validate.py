"""Pre-flight frame validation.

This is the check that stands between a corrupt file and a caller that would
believe whatever bytes arrived.  A model cannot tell a truncated JPEG from a
dark one, so "the bytes look wrong" has to be decided here, on the server,
before anything is returned.

No ffmpeg needed: the validator takes bytes.
"""

from __future__ import annotations

import struct

import pytest

from conftest import make_jpeg_bytes, make_png_bytes
from mcp_video_frames.errors import FfmpegError, LimitError
from mcp_video_frames.frames import (
    MIN_PLAUSIBLE_FRAME_BYTES,
    jpeg_dimensions,
    png_dimensions,
    sniff_format,
    validate_frame_file,
)

PNG = make_png_bytes(16, 9)
JPEG = make_jpeg_bytes(16, 9)


def write(tmp_path, data, name="frame.bin"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def validate(path, *, image_format, max_bytes=10 * 1024 * 1024, max_dim=8192):
    return validate_frame_file(
        path,
        image_format=image_format,
        max_image_bytes=max_bytes,
        max_image_dimension=max_dim,
    )


class TestSniffing:
    def test_recognises_real_headers(self):
        assert sniff_format(JPEG) == "jpeg"
        assert sniff_format(PNG) == "png"

    def test_rejects_other_content(self):
        assert sniff_format(b"GIF89a...") is None
        assert sniff_format(b"") is None
        assert sniff_format(b"<html>") is None


class TestDimensions:
    def test_png_ihdr(self):
        assert png_dimensions(PNG) == (16, 9)

    def test_jpeg_sof0(self):
        assert jpeg_dimensions(JPEG) == (16, 9)

    def test_jpeg_without_a_frame_header_is_not_measured(self):
        # Magic bytes plus filler: the classic truncated file.
        stub = b"\xff\xd8\xff" + b"\x00" * 64
        assert jpeg_dimensions(stub) is None

    def test_png_without_ihdr_is_not_measured(self):
        assert png_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) is None


class TestValidation:
    def test_accepts_a_valid_png(self, tmp_path):
        frame = validate(write(tmp_path, PNG, "a.png"), image_format="png")
        assert (frame.width, frame.height) == (16, 9)
        assert frame.mime_type == "image/png"
        assert frame.size == len(PNG)

    def test_accepts_a_valid_jpeg(self, tmp_path):
        frame = validate(write(tmp_path, JPEG, "a.jpg"), image_format="jpeg")
        assert (frame.width, frame.height) == (16, 9)
        assert frame.mime_type == "image/jpeg"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FfmpegError) as excinfo:
            validate(tmp_path / "absent.jpg", image_format="jpeg")
        assert "does not exist" in str(excinfo.value)

    def test_empty_file(self, tmp_path):
        with pytest.raises(FfmpegError) as excinfo:
            validate(write(tmp_path, b"", "e.jpg"), image_format="jpeg")
        assert "empty file" in str(excinfo.value)

    def test_jpeg_declared_but_png_content(self, tmp_path):
        with pytest.raises(FfmpegError) as excinfo:
            validate(write(tmp_path, PNG, "a.jpg"), image_format="jpeg")
        message = str(excinfo.value)
        assert "image/jpeg" in message
        assert "image/png" in message

    def test_png_declared_but_jpeg_content(self, tmp_path):
        with pytest.raises(FfmpegError):
            validate(write(tmp_path, JPEG, "a.png"), image_format="png")

    def test_unknown_content_reports_the_head_bytes(self, tmp_path):
        with pytest.raises(FfmpegError) as excinfo:
            validate(write(tmp_path, b"#!/bin/sh\necho hi\n", "x.jpg"), image_format="jpeg")
        assert "23 21 2F" in str(excinfo.value)

    def test_truncated_stub_is_rejected(self, tmp_path):
        # Right magic bytes, no image inside.
        stub = b"\xff\xd8\xff" + b"\x00" * 32
        with pytest.raises(FfmpegError) as excinfo:
            validate(write(tmp_path, stub, "s.jpg"), image_format="jpeg")
        assert "truncated" in str(excinfo.value)

    def test_header_without_dimensions_is_rejected(self, tmp_path):
        # Long enough to clear the size floor, still not a decodable image.
        stub = b"\xff\xd8\xff\xe0" + b"\x00" * 400
        with pytest.raises(FfmpegError) as excinfo:
            validate(write(tmp_path, stub, "h.jpg"), image_format="jpeg")
        assert "incomplete" in str(excinfo.value)

    def test_size_floor_is_above_a_bare_header(self):
        assert len(JPEG) >= MIN_PLAUSIBLE_FRAME_BYTES

    def test_oversized_frame_is_a_limit_error(self, tmp_path):
        path = write(tmp_path, JPEG, "big.jpg")
        with pytest.raises(LimitError) as excinfo:
            validate(path, image_format="jpeg", max_bytes=len(JPEG) - 1)
        message = str(excinfo.value)
        assert "per-frame limit" in message
        assert "max_edge" in message

    def test_dimension_limit_is_a_limit_error(self, tmp_path):
        big = make_jpeg_bytes(4000, 3000)
        path = write(tmp_path, big, "wide.jpg")
        with pytest.raises(LimitError) as excinfo:
            validate(path, image_format="jpeg", max_dim=1024)
        assert "4000" in str(excinfo.value)

    def test_zero_dimension_header_is_rejected(self, tmp_path):
        header = make_jpeg_bytes(1, 1)
        # Patch the SOF0 height/width to zero.
        index = header.index(b"\xff\xc0") + 5
        patched = header[:index] + struct.pack(">HH", 0, 0) + header[index + 4 :]
        with pytest.raises(FfmpegError) as excinfo:
            validate(write(tmp_path, patched, "zero.jpg"), image_format="jpeg")
        assert "untrustworthy" in str(excinfo.value)

    def test_error_messages_name_the_file(self, tmp_path):
        path = write(tmp_path, b"junk" * 40, "named.jpg")
        with pytest.raises(FfmpegError) as excinfo:
            validate(path, image_format="jpeg")
        assert "named.jpg" in str(excinfo.value)

    def test_timestamp_is_included_when_known(self, tmp_path):
        path = write(tmp_path, b"junk" * 40, "t.jpg")
        with pytest.raises(FfmpegError) as excinfo:
            validate_frame_file(
                path,
                image_format="jpeg",
                max_image_bytes=10**9,
                max_image_dimension=8192,
                timestamp=12.5,
            )
        assert "12.5" in str(excinfo.value)
