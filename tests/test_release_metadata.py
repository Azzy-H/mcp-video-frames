"""The MCP Registry metadata must agree with the release it points at.

These are text checks, not behaviour checks, and they exist because the failure
they prevent is expensive and lands on the wrong side of an irreversible step.

The registry does not take our word for owning the PyPI name: it fetches the
description of the *published* release and looks for a ``mcp-name:`` marker.  So
a marker that disagrees with ``server.json`` by one character, a ``version``
that is not the version being released, or a description one character over the
registry's 100-character limit, all produce a failure *after* the PyPI upload
has already happened — and PyPI versions cannot be reused, so the only fix is
another release.  0.1.1 exists because 0.1.0 was published without the marker at
all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The registry rejects a description longer than this, at publish time.
MAX_DESCRIPTION = 100

#: Server names are reverse-DNS with exactly one slash.
NAME_PATTERN = re.compile(r"[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+")

#: A version range is rejected; only an exact version is accepted.
VERSION_RANGE = re.compile(r"[\^*~<>=]|\.\*")


def _server_json() -> dict:
    return json.loads((PROJECT_ROOT / "server.json").read_text(encoding="utf-8"))


def _declared_version() -> str:
    """The version the package will actually report, read from the source."""
    source = (PROJECT_ROOT / "src/mcp_video_frames" / "__init__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'__version__ = "([^"]+)"', source)
    assert match, "src/mcp_video_frames/__init__.py declares no __version__"
    return match.group(1)


# ----------------------------------------------------------------------
# server.json: the registry validates these shapes server-side
# ----------------------------------------------------------------------
def test_server_json_satisfies_the_registry_schema():
    server = _server_json()

    assert set(server) >= {"name", "description", "version"}
    assert NAME_PATTERN.fullmatch(server["name"]), server["name"]
    assert len(server["description"]) <= MAX_DESCRIPTION, (
        f"description is {len(server['description'])} characters; "
        f"the registry caps it at {MAX_DESCRIPTION}"
    )
    assert len(server.get("title", "")) <= MAX_DESCRIPTION

    assert set(server["repository"]) >= {"url", "source"}

    assert server["packages"], "a server with no package cannot be installed"
    for package in server["packages"]:
        assert set(package) >= {"registryType", "identifier", "transport"}
        assert package["transport"] == {"type": "stdio"}
        version = package.get("version", "")
        assert not VERSION_RANGE.search(version), f"version range rejected: {version}"


def test_the_package_is_the_one_we_publish():
    """A typo here would point the registry at a package that is not ours."""
    package = _server_json()["packages"][0]

    assert package["registryType"] == "pypi"
    assert package["identifier"] == "mcp-video-frames"


# ----------------------------------------------------------------------
# The version has to be the version, in all three places
# ----------------------------------------------------------------------
def test_declared_versions_agree():
    server = _server_json()
    declared = _declared_version()

    assert server["version"] == declared, (
        "server.json and the package disagree; the registry would advertise a "
        "version that was never released"
    )
    assert server["packages"][0]["version"] == declared


# ----------------------------------------------------------------------
# Ownership: the marker is what proves the PyPI package is ours
# ----------------------------------------------------------------------
def test_readme_carries_the_marker_the_registry_looks_for():
    server = _server_json()
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    marker = f"mcp-name: {server['name']}"

    assert marker in readme, (
        "README.md is missing the ownership marker, so the registry cannot "
        "verify that the PyPI package belongs to this server"
    )
    # Hidden in a comment: invisible on PyPI, still present in the description
    # the registry fetches.
    assert re.search(r"<!--\s*" + re.escape(marker) + r"\s*-->", readme), (
        "the marker must sit inside an HTML comment so it stays out of the "
        "rendered README"
    )


# ----------------------------------------------------------------------
# The registry may only be claimed once PyPI actually has the release
# ----------------------------------------------------------------------
def test_the_registry_publish_runs_after_the_pypi_publish():
    workflow = (PROJECT_ROOT / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )

    assert re.search(r"^  registry:$", workflow, re.M), (
        "publish.yml has no registry job, so the registry listing would go stale"
    )
    # Ownership is verified by reading the description off PyPI, so racing the
    # upload publishes a claim the registry cannot yet confirm.
    assert re.search(r"^    needs: pypi$", workflow, re.M), (
        "the registry job must depend on the pypi job"
    )
