#!/usr/bin/env python3
"""One-time local setup: private virtual environment + MCP client config.

Run it with any Python 3.10+ interpreter::

    python setup_mcp.py

It does three things, and nothing else:

1. creates ``.venv/`` **inside this project** (never touches your system
   Python, never installs the project itself);
2. installs the two runtime dependencies from ``requirements.txt`` into that
   virtual environment;
3. writes ``mcp-config.json`` with the real paths filled in, ready to paste
   into your MCP client's configuration.

To uninstall, delete ``.venv/`` and the config entry. Nothing was written
outside this directory.

ffmpeg and ffprobe are not Python packages and cannot be installed here. The
script checks for them and tells you what to install if they are missing.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp_video_frames.config import Config  # noqa: E402  (sys.path set above)

VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
LAUNCHER = PROJECT_ROOT / "run_server.py"
CONFIG_PATH = PROJECT_ROOT / "mcp-config.json"
SERVER_NAME = "video-frames"
MIN_PYTHON = (3, 10)

#: Project-local cache directory name.  The package itself defaults to the
#: system temp directory (right for a published install, which must not write
#: gigabytes into site-packages); a local checkout uses this instead, so the
#: whole deployment stays in one deletable directory.
LOCAL_CACHE_DIRNAME = "cache"

IS_WINDOWS = platform.system() == "Windows"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def venv_paths() -> tuple[Path, Path]:
    """Return ``(python, pip)`` inside the project virtual environment."""
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe", VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "python", VENV_DIR / "bin" / "pip"


def say(message: str) -> None:
    print(message, flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def check_interpreter() -> None:
    if sys.version_info < MIN_PYTHON:
        die(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required; "
            f"this is {platform.python_version()}."
        )


def find_ffmpeg(explicit: str | None) -> tuple[str, str] | None:
    """Locate ffmpeg and ffprobe, in the order the server itself uses them.

    Returns ``(ffmpeg, ffprobe)`` or ``None`` when either is missing.
    ``ffprobe`` may legitimately be absent — the server falls back to parsing
    ffmpeg's own banner — but we treat a missing ffprobe as "needs attention"
    here, because it is worth telling the user rather than silently degrading.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            die(f"--ffmpeg points to a file that does not exist: {candidate}")
        # Assume ffprobe sits next to it; that is true of every distribution
        # layout (gyan.dev zip, chocolatey, brew, apt).
        sibling = candidate.with_name(
            "ffprobe" + (".exe" if candidate.suffix.lower() == ".exe" else "")
        )
        found_probe = str(sibling) if sibling.is_file() else ""
        return str(candidate), found_probe

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    # Half-present: still report it, because the server would degrade.
    if ffmpeg or ffprobe:
        return ffmpeg or "", ffprobe or ""
    return None


#: Where a downloaded ffmpeg lands, inside the project.  Deliberately next to
#: the venv's own executables so everything the deployment needs lives under
#: one directory that can be deleted in one go.
def tools_dir() -> Path:
    return VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")


#: Prebuilt Windows build.  This is the same distribution winget's
#: ``Gyan.FFmpeg`` installs, fetched directly so the project stays
#: self-contained and no system-wide installation happens.
WINDOWS_FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
WINDOWS_FFMPEG_APPROX_MB = 115


def download_ffmpeg_into_venv(url: str | None = None) -> tuple[str, str]:
    """Download a prebuilt ffmpeg and unpack it into the venv's bin directory.

    Nothing is installed system-wide: two executables land under ``.venv``,
    and the server is pointed at them through ``FFMPEG_PATH``/``FFPROBE_PATH``
    in the generated config.  Deleting ``.venv/`` removes them again.
    """
    import urllib.request
    import zipfile

    if url is None:
        if not IS_WINDOWS:
            die(
                "Automatic download currently supports Windows only.\n"
                "On macOS / Linux, install with your package manager first:\n"
                "    brew install ffmpeg          # macOS\n"
                "    sudo apt-get install ffmpeg  # Debian/Ubuntu\n"
                "or download a third-party static build and pass --ffmpeg <path>."
            )
        url = WINDOWS_FFMPEG_URL

    target_dir = tools_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    say(f"  Downloading {url}")
    say(f"  About {WINDOWS_FFMPEG_APPROX_MB} MB, please wait...")

    try:
        with urllib.request.urlopen(url, timeout=900) as response:
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - report verbatim
        die(
            f"Download failed: {type(exc).__name__}: {exc}\n"
            f"Configure ffmpeg another way, then re-run this script with "
            f"--ffmpeg <path>"
        )

    say(f"  Downloaded {len(payload) / 1048576:.1f} MB, extracting...")
    wanted = {"ffmpeg.exe", "ffprobe.exe"} if IS_WINDOWS else {"ffmpeg", "ffprobe"}
    extracted: dict[str, str] = {}
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name not in wanted:
                    continue
                # Guard against a hostile archive escaping the target dir.
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    continue
                with archive.open(member) as src:
                    data = src.read()
                destination = target_dir / name
                destination.write_bytes(data)
                if not IS_WINDOWS:
                    destination.chmod(0o755)
                extracted[name] = str(destination)
                say(f"    + {destination}")
    except zipfile.BadZipFile:
        die(
            "The downloaded file is not a valid zip.\n"
            "Download it manually and pass --ffmpeg <ffmpeg path>."
        )

    missing = sorted(wanted - set(extracted))
    if missing:
        die(f"Not found inside the archive: {', '.join(missing)}")
    return extracted["ffmpeg.exe" if IS_WINDOWS else "ffmpeg"], extracted[
        "ffprobe.exe" if IS_WINDOWS else "ffprobe"
    ]


def verify_ffmpeg(ffmpeg: str, ffprobe: str) -> bool:
    """Run the binaries once: being present on disk is not the same as working."""
    import subprocess

    for label, binary in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        if not binary:
            say(f"  {label}: not found (the server degrades: metadata falls back to parsing ffmpeg's own header output)")
            continue
        try:
            result = subprocess.run(
                [binary, "-version"], capture_output=True, text=True, timeout=60
            )
        except OSError as exc:
            say(f"  {label}: cannot execute ({exc})")
            return False
        first_line = (result.stdout or "").splitlines()[:1]
        say(f"  {label}: {first_line[0] if first_line else 'no output'}")
        if result.returncode != 0:
            say(f"  {label}: exit code {result.returncode}, possibly not a usable build")
            return False
    return bool(ffmpeg)


#: How the user chose to resolve a missing ffmpeg, for the generated config.
FfmpegChoice = tuple[str, str]


def read_existing_env() -> dict[str, str]:
    """Environment from a previously generated config, if there is one.

    Re-running setup must not silently forget a decision an earlier run made.
    Someone who pointed at their own ffmpeg, then re-runs out of habit, should
    not end up with a config that can no longer find it.
    """
    return _read_existing_entry().get("env") or {}


def _read_existing_entry() -> dict:
    try:
        document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        entry = document["mcpServers"][SERVER_NAME]
        return entry if isinstance(entry, dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def flatten_entry(entry: dict) -> dict[str, str]:
    """Flatten a client entry to ``key -> value`` for change reporting.

    The whole entry is flattened, not just ``env``: ``command``, ``args`` and
    ``cwd`` are the fields most likely to be wrong, and a summary that only
    watched the environment would stay silent when they change — which is
    exactly when the user needs to be told, because the client must be updated.
    """
    flat: dict[str, str] = {}
    for key, value in entry.items():
        if key == "env":
            for env_key, env_value in (value or {}).items():
                flat[f"env.{env_key}"] = str(env_value)
        elif isinstance(value, list):
            flat[key] = " ".join(str(item) for item in value)
        else:
            flat[key] = str(value)
    return flat


def describe_changes(before: dict, after: dict) -> list[str]:
    """Human-readable diff of two client entries, for the summary."""
    old = flatten_entry(before)
    new = flatten_entry(after)
    lines: list[str] = []
    for key in sorted(set(old) | set(new)):
        previous, current = old.get(key), new.get(key)
        if previous == current:
            continue
        if previous is None:
            lines.append(f"  + {key} = {current}")
        elif current is None:
            lines.append(f"  - {key} (was {previous})")
        else:
            lines.append(f"  ~ {key}: {previous} -> {current}")
    return lines


def resolve_ffmpeg(
    args: argparse.Namespace, previous_env: dict[str, str]
) -> tuple[str, str]:
    """Decide where ffmpeg comes from, in priority order.

    1. ``--ffmpeg`` given: it wins, and a bad path is a usage error.
    2. a choice recorded by an earlier run: reuse it, so re-running is a no-op.
    3. discovered on PATH: use it, and do not pin a path in the config.
    4. only ffmpeg found (no ffprobe): use it, note the degradation.
    5. otherwise: ask, unless there is nobody to ask.
    """
    if args.ffmpeg:
        resolved = find_ffmpeg(args.ffmpeg)
        if resolved is None or not resolved[0]:
            die(f"--ffmpeg is not usable: {args.ffmpeg}")
        if not resolved[1]:
            say(
                "  Note: no ffprobe next to it, so metadata degrades to "
                "parsing ffmpeg's own header output."
            )
        else:
            say(f"  Using the path from --ffmpeg: {resolved[0]}")
        return resolved

    remembered = previous_env.get("FFMPEG_PATH") or ""
    if remembered:
        if Path(remembered).is_file():
            ffprobe = previous_env.get("FFPROBE_PATH") or ""
            say(f"  Reusing the ffmpeg from the previous config: {remembered}")
            if ffprobe and not Path(ffprobe).is_file():
                say(f"  But the ffprobe recorded then no longer exists: {ffprobe}")
                ffprobe = ""
            return remembered, ffprobe if ffprobe else ""
        say(f"  The ffmpeg from the previous config no longer exists: {remembered} (searching again)")

    found = find_ffmpeg(None)
    if found and found[0] and found[1]:
        say(f"  Found ffmpeg (on PATH): {found[0]}")
        return found
    if found and found[0]:
        say(f"  Found ffmpeg (on PATH): {found[0]}")
        say("  No ffprobe in the same directory -- container metadata degrades to")
        say("  parsing ffmpeg's own header output. Frame extraction is unaffected.")
        return found

    say("")
    say("  No ffmpeg detected.")
    say("")

    if args.no_ffmpeg:
        say("  --no-ffmpeg was given, skipping the prompt.")
        say("  Once installed, re-run this script with --ffmpeg <path> to add it.")
        return ("", "")

    if args.no_install:
        say("  --no-install was given, skipping the prompt.")
        say(
            "  Once installed, re-run with --ffmpeg <path>, or add FFMPEG_PATH to "
            "the config's env by hand."
        )
        return ("", "")

    if not sys.stdin.isatty():
        # A script or CI run: never block on a prompt nobody can answer.
        say("  Not an interactive terminal, skipping the prompt. Options:")
        say("      python setup_mcp.py --ffmpeg <ffmpeg path>   # point at one")
        say("      python setup_mcp.py --install-ffmpeg         # download into .venv")
        say("  Or install it yourself:")
        say("      winget install Gyan.FFmpeg -e  /  brew install ffmpeg  /  apt-get install ffmpeg")
        return ("", "")

    say("  Choose how to proceed:")
    if IS_WINDOWS:
        say(f"    1) Download and unpack into this project's .venv (about "
            f"{WINDOWS_FFMPEG_APPROX_MB} MB; nothing system-wide, deleting .venv "
            f"uninstalls it)")
    else:
        say("    1) Download into this project's .venv (no prebuilt package for "
            "this platform; choosing this prints guidance)")
    say("    2) Point at an ffmpeg path (I installed it myself)")
    say("    3) Skip for now and write the config anyway (tool calls will error)")

    while True:
        try:
            answer = input("  Enter 1 / 2 / 3 (default 3): ").strip() or "3"
        except EOFError:
            answer = "3"
        if answer == "1":
            return download_ffmpeg_into_venv(args.ffmpeg_url)
        if answer == "2":
            entered = input("  Paste the full path to the ffmpeg executable: ").strip().strip('"')
            if not entered:
                say("  Nothing entered, choose again.")
                continue
            # Validate here rather than inside find_ffmpeg, which treats a bad
            # explicit path as a fatal command-line error.  An interactive
            # typo should send the user back to the menu, not kill the script.
            candidate = Path(entered).expanduser()
            if not candidate.is_file():
                say(f"  There is no file at that path: {candidate}")
                say("  Choose again (you can also drag the file into the terminal).")
                continue
            resolved = find_ffmpeg(str(candidate))
            if resolved is None or not resolved[0]:
                say(f"  {candidate} cannot be used as ffmpeg.")
                continue
            return resolved
        if answer == "3":
            say(
                "  Writing the config first. Without ffmpeg the server still "
                "starts, but tool calls will error."
            )
            return ("", "")
        say("  Enter 1, 2 or 3.")


# ----------------------------------------------------------------------
# virtual environment
# ----------------------------------------------------------------------
def create_venv(*, use_uv: bool) -> None:
    """Create the project's own virtual environment.

    Tried in order: ``uv venv`` (fast, and bootstraps reliably on interpreters
    that ship without ``ensurepip``), ``python -m venv``, then ``virtualenv``.
    Each attempt is a *user-local* action: only ``.venv/`` is written.
    """
    python_in_venv, pip_in_venv = venv_paths()

    if VENV_DIR.exists() and python_in_venv.is_file() and pip_in_venv.is_file():
        say(f"  A usable virtual environment already exists: {VENV_DIR}")
        return
    if VENV_DIR.exists():
        say(f"  The existing .venv is incomplete, rebuilding it: {VENV_DIR}")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    attempts: list[tuple[str, list[str]]] = []
    if use_uv:
        attempts.append(("uv venv", ["uv", "venv", str(VENV_DIR)]))
    attempts.append(("python -m venv", [sys.executable, "-m", "venv", str(VENV_DIR)]))
    attempts.append(("virtualenv", [sys.executable, "-m", "virtualenv", str(VENV_DIR)]))

    errors: list[str] = []
    for label, command in attempts:
        if shutil.which(command[0]) is None and not Path(command[0]).is_file():
            errors.append(f"{label}: {command[0]} not found")
            continue
        say(f"  Creating the virtual environment ({label})...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and python_in_venv.is_file():
            say(f"  Virtual environment created: {VENV_DIR}")
            return
        errors.append(f"{label}: {result.stderr.strip()[-300:]}")

    detail = "\n".join(f"    - {item}" for item in errors)
    die(
        f"Cannot create the virtual environment {VENV_DIR}. Tried:\n{detail}\n"
        f"Create it by hand and re-run this script:\n"
        f"    python -m venv {VENV_DIR}\n"
        f"or install uv (https://docs.astral.sh/uv/) and use uv venv."
    )


def install_requirements(*, use_uv: bool) -> None:
    """Install the runtime dependencies into the project venv only."""
    python_in_venv, pip_in_venv = venv_paths()
    if not REQUIREMENTS.is_file():
        die(f"{REQUIREMENTS} not found.")

    if use_uv:
        say("  Installing dependencies (uv pip install --python .venv)...")
        command = [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_in_venv),
            "-r",
            str(REQUIREMENTS),
        ]
    else:
        say("  Installing dependencies (pip inside .venv)...")
        command = [str(pip_in_venv), "install", "-r", str(REQUIREMENTS)]

    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        die(
            "Dependency installation failed. Common causes:\n"
            "    - network/proxy problem: pip cannot reach PyPI\n"
            "    - a corporate proxy needing a certificate: set HTTPS_PROXY, or use uv\n"
            f"Run the same command by hand to investigate:\n    {' '.join(command)}"
        )
    say("  Dependencies installed.")
    verify_imports(python_in_venv)


def verify_imports(python_in_venv: Path) -> None:
    """Prove the venv can import what the server needs, before writing config."""
    script = (
        "import sys;"
        "import mcp;"
        "from mcp.types import TextContent, ImageContent;"
        "sys.path.insert(0, r'{src}');"
        "import mcp_video_frames;"
        "from mcp_video_frames.server import _load_sdk;"
        "cls = _load_sdk()[1];"
        "print(mcp_video_frames.__version__, cls.__module__ + '.' + cls.__name__)"
    ).format(src=PROJECT_ROOT / "src")
    result = subprocess.run(
        [str(python_in_venv), "-c", script], capture_output=True, text=True
    )
    if result.returncode != 0:
        die(
            f"{python_in_venv} cannot import mcp and this project:\n"
            f"{result.stderr.strip()[-600:]}"
        )
    say(f"  Self-check passed: mcp_video_frames {result.stdout.strip()}")


# ----------------------------------------------------------------------
# client configuration
# ----------------------------------------------------------------------
def build_config(
    *,
    max_images: int,
    cache_dir: str | None,
    ffmpeg: str,
    ffprobe: str,
) -> dict:
    """Assemble the client entry.

    Three settings have non-obvious rules, all of them learned the hard way:

    * **cache directory** — defaults to the project's own ``.cache`` so a local
      deployment is self-contained and lands somewhere the OS will not silently
      reap.  ``--cache-dir`` overrides it.  A relative value is only emitted
      together with ``cwd``: relative paths resolve against the *client's*
      working directory, so without ``cwd`` the cache would land wherever the
      client happened to start, which is worse than the temp default.
    * **ffmpeg paths** — written only when they are actually needed.  When the
      binaries are on PATH the server finds them itself, and pinning an
      absolute path that later moves is a new failure mode.  A path inside
      ``.venv`` *is* pinned, because that is the point of downloading there.
    * **image limit** — always written, because it is the one setting that must
      match the client rather than this machine.
    """
    python_in_venv, _pip = venv_paths()
    defaults = Config.from_env({})

    # Order matters for a human reading the config: the two settings that must
    # be adjusted per deployment come first, then the discovered paths, then
    # the tuning knobs that most people never touch.
    env: dict[str, str] = {
        "MCP_VIDEO_MAX_IMAGES": str(max_images),
    }

    entry: dict[str, object] = {
        "command": str(python_in_venv),
        "args": [str(LAUNCHER)],
    }

    if cache_dir:
        env["MCP_VIDEO_CACHE_DIR"] = cache_dir
        if not Path(cache_dir).is_absolute():
            # Relative, so the working directory has to be pinned to match.
            entry["cwd"] = str(PROJECT_ROOT)
    else:
        env["MCP_VIDEO_CACHE_DIR"] = f".{LOCAL_CACHE_DIRNAME}"
        entry["cwd"] = str(PROJECT_ROOT)

    on_path = shutil.which("ffmpeg")
    if ffmpeg and (on_path is None or Path(ffmpeg).resolve() != Path(on_path).resolve()):
        env["FFMPEG_PATH"] = ffmpeg
    if ffprobe and shutil.which("ffprobe") is None:
        env["FFPROBE_PATH"] = ffprobe

    # Everything else is written out explicitly at its current default.
    # Omitting them would hide their existence: the config file is the only
    # place a user discovers that these knobs are tunable at all.  Re-running
    # the script refreshes them and reports any difference.
    for key, value in (
        ("MCP_VIDEO_MAX_IMAGE_BYTES", defaults.max_image_bytes),
        ("MCP_VIDEO_MAX_TOTAL_BYTES", defaults.max_total_bytes),
        ("MCP_VIDEO_MAX_IMAGE_DIM", defaults.max_image_dimension),
        ("MCP_VIDEO_CACHE_MAX_BYTES", defaults.cache_max_bytes),
        ("MCP_VIDEO_FRAME_MAX_AGE_DAYS", defaults.frame_max_age_days),
        ("MCP_VIDEO_SCENE_THRESHOLD", defaults.scene_threshold),
    ):
        env[key] = _format_number(value)

    entry["env"] = env

    return {"mcpServers": {SERVER_NAME: entry}}


def _format_number(value: float) -> str:
    """Render a config default the way an operator would type it.

    Integers stay integers (``14`` not ``14.0``); fractional values keep their
    digits (``0.3`` not ``0``), because the threshold is meaningful at that
    precision.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def write_config(config: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def client_config_hint() -> str:
    if IS_WINDOWS:
        return "%APPDATA%\\Claude\\claude_desktop_config.json"
    if platform.system() == "Darwin":
        return "~/Library/Application Support/Claude/claude_desktop_config.json"
    return "~/.config/Claude/claude_desktop_config.json"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="setup_mcp.py",
        description=(
            "Create .venv in this directory, install the runtime dependencies "
            "into it, and generate mcp-config.json. It does not install this "
            "project and does not touch the system Python."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="MCP_VIDEO_MAX_IMAGES written into the config (max images per call, default 20)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "MCP_VIDEO_CACHE_DIR written into the config. "
            f"Defaults to .{LOCAL_CACHE_DIRNAME} inside the project (which also "
            "writes cwd, because a relative path resolves against the working "
            "directory); pass an absolute path and cwd is omitted"
        ),
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="full path to the ffmpeg executable; use this when it is not on PATH",
    )
    parser.add_argument(
        "--install-ffmpeg",
        action="store_true",
        help="skip the prompt and download ffmpeg into .venv (prebuilt for Windows only)",
    )
    parser.add_argument(
        "--ffmpeg-url",
        default=None,
        help="download ffmpeg from this zip URL instead (mirror or offline copy)",
    )
    parser.add_argument(
        "--uv",
        action="store_true",
        help="use uv instead of pip to create the venv and install dependencies (faster)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="only write the config file; do not create a venv or install anything",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="print the config to stdout instead of writing a file",
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="never prompt about ffmpeg (for scripts); omits FFMPEG_PATH from the config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    check_interpreter()

    if args.print_only:
        found = find_ffmpeg(args.ffmpeg) or ("", "")
        print(
            json.dumps(
                build_config(
                    max_images=args.max_images,
                    cache_dir=args.cache_dir,
                    ffmpeg=found[0],
                    ffprobe=found[1],
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    previous_env = read_existing_env()
    rerun = bool(previous_env)

    say("mcp-video-frames local deployment")
    if rerun:
        say("existing config detected: this is a re-run, so confirmed choices are reused and not re-asked")
    say("=" * 40)

    if args.no_install:
        say("Skipping the virtual environment (--no-install).")
        python_in_venv, _pip = venv_paths()
        if not python_in_venv.is_file():
            say(f"  Note: {python_in_venv} does not exist, so the config will not work yet.")
    else:
        use_uv = args.uv or bool(shutil.which("uv"))
        if use_uv and not shutil.which("uv"):
            die("--uv was given, but uv is not on PATH.")
        create_venv(use_uv=use_uv)
        install_requirements(use_uv=use_uv)

    say("")
    say("Checking ffmpeg:")
    if args.install_ffmpeg or args.ffmpeg_url:
        ffmpeg, ffprobe = download_ffmpeg_into_venv(args.ffmpeg_url)
    else:
        ffmpeg, ffprobe = resolve_ffmpeg(args, previous_env)

    ready = verify_ffmpeg(ffmpeg, ffprobe)

    config = build_config(
        max_images=args.max_images,
        cache_dir=args.cache_dir,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    new_env = config["mcpServers"][SERVER_NAME]["env"]
    changes = describe_changes(previous_env, new_env)
    write_config(config)

    say("")
    say("=" * 40)
    say(f"Config written to: {CONFIG_PATH}")
    if rerun:
        if changes:
            say("")
            say("Changes since last time:")
            for line in changes:
                say(line)
        else:
            say("")
            say("Identical to last time; nothing changed.")
    else:
        say("")
        say(json.dumps(config, indent=2, ensure_ascii=False))
    say("")
    say(f"Merge the block above into your MCP client config: {client_config_hint()}")
    say("(if an mcpServers section already exists, just add the \"" + SERVER_NAME + "\" entry to it)")
    say("")
    if ready:
        say("Next: verify the server starts (no client needed)")
        say(f"    {venv_paths()[0]} test_server.py doctor")
    else:
        say("Note: ffmpeg is not ready. The server still starts, but every tool call will error.")
        say("Fix it and re-run this script; already-installed dependencies are not reinstalled.")
    say("")
    say("Re-running this script is safe: an existing virtual environment is reused,")
    say("reinstalling dependencies is idempotent, and a confirmed ffmpeg choice is kept.")
    say("Uninstall: delete .venv/ and remove this config block. Nothing else was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
