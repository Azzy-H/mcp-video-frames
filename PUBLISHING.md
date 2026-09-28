# Publishing to PyPI

The distribution name is `mcp-video-frames`. It was unclaimed on PyPI when this
was written — check before you rely on it:

```sh
python -m pip index versions mcp-video-frames   # "No matching distribution" == free
```

Two paths follow. **Path A** is Trusted Publishing from GitHub Actions: no secret
is ever written to disk, and it is what `.github/workflows/publish.yml` already
implements. **Path B** is a manual upload from your own machine.

Whichever you pick, do the pre-flight below first.

Rehearsing on TestPyPI is **optional**. It costs one extra account — TestPyPI is
a separate instance of the index with its own database, so your PyPI login does
not work there — and it buys exactly one thing: finding out that an upload fails
(a rejected metadata version, a broken README render, an entry point that will
not start) *before* you spend a version number on PyPI, where a used version is
gone forever. If you would rather not bother, skip every TestPyPI mention below
and go straight to PyPI.

## Pre-flight

```sh
python -m pip install --upgrade build "twine>=7"
python -m pytest -m "not integration"      # the release gate
rm -rf dist/
python -m build                            # writes dist/*.tar.gz and dist/*.whl
python -m twine check --strict dist/*
```

`twine check` must print `PASSED` for both files.

Four things bite people here:

- **The version lives in the package, not in `pyproject.toml`.**
  `[tool.hatch.version]` reads `__version__` from
  `src/mcp_video_frames/__init__.py`. Bump it there and add a `CHANGELOG.md`
  entry — editing a version string in `pyproject.toml` will not exist.
- **PyPI never lets you reuse a version, not even after deleting the release.**
  `0.1.0` is spent the moment the upload succeeds. Release `0.1.1` instead.
- **The metadata is core metadata 2.5** (PEP 639 puts the licence in an SPDX
  `License-Expression` field). Uploading it needs `twine>=7`; anything older
  rejects the file. `pypa/gh-action-pypi-publish@release/v1` has bundled twine 7
  since v1.14.
- **Never upload the same filename twice.** A rebuild of the same version
  produces different bytes and PyPI refuses it. Always `rm -rf dist/` first.

## Path A — Trusted Publishing (recommended)

1. **Create the account(s).** A [PyPI](https://pypi.org/account/register/)
   account is required, and 2FA is mandatory on it. Register separately on
   [TestPyPI](https://test.pypi.org/account/register/) only if you are
   rehearsing. TestPyPI's database is pruned periodically, so a TestPyPI account
   or project can disappear without warning; treat it purely as scratch space,
   never as somewhere to host a real release.

2. **Register the pending publisher.** On PyPI, go to
   <https://pypi.org/manage/account/publishing/> → *Add a new pending publisher*
   → GitHub, and fill in exactly:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `mcp-video-frames` |
   | Owner | `Azzy-H` |
   | Repository name | `mcp-video-frames` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   Then repeat the same form on TestPyPI at
   <https://test.pypi.org/manage/account/publishing/>, with environment name
   `testpypi` — or skip that half if you are not rehearsing. This entry is what
   lets the workflow upload without a token; it turns into a normal publisher
   entry after the first successful upload.

3. **Create the GitHub environments.** In the repository:
   *Settings → Environments → New environment*: `pypi`, plus `testpypi` if you
   are rehearsing. The names must match the pending publishers. Add required
   reviewers to `pypi` if you want a human click between "release published" and
   "uploaded".

4. **Rehearse (optional).** Push this repository, then
   *Actions → Publish → Run workflow* with target `testpypi`. Then confirm the
   install:

   ```sh
   python -m venv /tmp/rehearse
   /tmp/rehearse/bin/pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ mcp-video-frames
   /tmp/rehearse/bin/mcp-video-frames doctor
   ```

   (The extra index is needed because TestPyPI does not mirror `mcp` or
   `filelock`.)

5. **Release.** Bump `__version__`, update `CHANGELOG.md`, commit, then:

   ```sh
   git tag v0.1.1
   git push origin main --tags
   ```

   Publish a GitHub Release for that tag. The workflow builds, checks, and
   uploads to PyPI with a signed attestation.

## Path B — manual upload

Create an API token scoped to the project (PyPI: *Account settings → API
tokens*; for the very first upload, scope it to "Entire account", because the
project does not exist yet). Then:

```sh
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...

python -m twine upload --repository testpypi dist/*    # only if rehearsing
python -m twine upload dist/*                          # the real thing
```

`--repository testpypi` needs its own token from the TestPyPI account; a PyPI
token is not accepted there. Do not put either token in `~/.pypirc` if you can
avoid it — an exported variable does not linger on disk. Trusted Publishing
(Path A) avoids the secret entirely, which is why it is the recommended path.

## Releasing the next version

1. Bump `__version__` in `src/mcp_video_frames/__init__.py`.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading.
3. `python -m build && python -m twine check --strict dist/*`.
4. Commit, tag `vX.Y.Z`, push the tag.
5. Publish the GitHub Release.

## Optional: the official MCP registry

Publishing to PyPI is enough to make `pip install mcp-video-frames` work. To
also appear in the [MCP registry](https://registry.modelcontextprotocol.io/),
the registry has to be convinced you own the name. For a PyPI-backed server that
means adding a marker to `README.md`:

```html
<!-- mcp-name: io.github.Azzy-H/mcp-video-frames -->
```

(or shipping a `server.json`), then publishing with `mcp-publisher`. That is a
separate step from PyPI and can be done any time after the first release.
