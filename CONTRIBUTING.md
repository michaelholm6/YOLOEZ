# Contributing to YOLOEZ

Thank you for your interest in contributing to **YOLOEZ**!

YOLOEZ is an open-source, GUI-based application for labeling data, training models, and running inference with **Ultralytics YOLO11** models. It was developed at **Purdue University** and is published as open-source research software.

Contributions from users, researchers, and developers are welcome and encouraged.

---

## Who Should Contribute

YOLOEZ welcomes contributions from:

- **End users**
  - Bug reports
  - UX feedback
  - Usability suggestions

- **Researchers and developers**
  - Feature proposals
  - Code contributions
  - Test development
  - Documentation updates

In this project, researchers and developers are often the same people, and contributions are expected to meet research-quality software standards.

---

## What Contributions Are Accepted

At the current stage, we accept:

- Bug fixes
- Proposed new features (after discussion)
- GUI and UX improvements
- Documentation improvements
- Test coverage improvements

Large or architectural changes must be discussed in advance.

---

## Communication Workflow

### Issues

- **Bug reports** must be submitted as GitHub Issues.
- **Feature proposals** must be submitted as Issues *before* implementation.
- Large pull requests without prior discussion may be declined.

### Discussions

Use **GitHub Discussions** for:
- Usage questions
- Design or UX discussions
- Clarifying intended behavior
- Early-stage ideas

Please do not submit unsolicited large pull requests.

---

## Bug Reports

Bug reports should include:

- YOLOEZ version or commit hash
- Operating system (Windows or Linux)
- Python version
- CPU or GPU usage (and GPU model if applicable)
- Clear steps to reproduce
- Expected vs. actual behavior
- Screenshots or recordings for GUI issues
- Relevant logs or error messages

Well-documented bug reports help resolve issues faster.

---

## Development Workflow

### Branching Model

- `main` — stable, tested releases
- `development` — active development

External contributors should:
1. Fork the repository
2. Create feature branches from `development`
3. Submit pull requests back to `development`

---

## Pull Request Requirements

All pull requests must include:

- Tests covering new or modified functionality
- Documentation updates where applicable
- Screenshots or GIFs for GUI changes
- A reference to the related Issue (e.g. `Fixes #17`)
- Code formatted with `black`

Pull requests must pass all CI checks before being merged.

---

## Local Development Setup

### Python Version

YOLOEZ requires **Python 3.12 or newer**. Continuous integration tests against Python 3.12.

Check what you have — `python3 --version` on Linux and macOS, `py --version` on Windows. If it is older than 3.12, install a newer interpreter before continuing; installing into an older Python will fail, since the project declares `requires-python = ">=3.12"`.

Ways to obtain Python 3.12:

- **uv** (no administrator access needed) — downloads and manages its own interpreters:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  On Windows, install uv with `winget install astral-sh.uv` instead of the shell script. Note that on Windows uv is useful only for obtaining an interpreter; `uv sync` cannot install this project's dependencies there, for the reason given under Installation below.

- **Ubuntu / Debian** (requires `sudo`):

  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update && sudo apt install python3.12 python3.12-venv
  ```

- **pyenv** (no administrator access needed):

  ```bash
  pyenv install 3.12 && pyenv local 3.12
  ```

- **Windows** — the installer from [python.org](https://www.python.org/downloads/), or `winget install Python.Python.3.12`.

### Installation

Fork and clone the repository first, as described under [Development Workflow](#development-workflow) above, then run the commands below from the repository root.

#### Linux and macOS

The quickest route is uv, which creates `.venv`, fetches Python 3.12, and installs every dependency including PyQt5:

```bash
uv sync --extra dev --python 3.12
```

Run commands in that environment with `uv run`, for example `uv run pytest`.

Or set up a virtual environment manually:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Name the interpreter explicitly as `python3.12`. On distributions whose default `python3` is older, a plain `python3 -m venv` produces an environment YOLOEZ cannot be installed into.

#### Windows

Use a virtual environment and `pip`:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`uv sync` does **not** work on Windows for this project: PyQt5's Qt binaries (`pyqt5-qt5`) publish no Windows wheels at the version uv resolves to, so the sync fails with a "no source distribution or wheel for the current platform" error. `pip` selects a compatible older version. This is why PyQt5 is declared in the `[dev]` extra rather than among the core dependencies.

Either route installs:

* Runtime dependencies
* Development dependencies (`pytest`, `pytest-qt`, `black`, `pyinstaller`, and others)

---

## Testing

* Tests are written using **pytest**
* GUI tests use **pytest-qt**, and run headless — no visible display is required
* Tests must pass on both Windows and Linux

Run the full suite from the repository root:

```bash
pytest
```

If you set the project up with uv, prefix commands with `uv run`:

```bash
uv run pytest
```

On a headless Linux machine, Qt needs an offscreen platform plugin and a virtual display:

```bash
QT_QPA_PLATFORM=offscreen xvfb-run -a python -m pytest
```

Other useful commands:

```bash
# Run a single test
pytest tests/test_cases.py::test_labeling_workflow_runs_without_gui -v

# Run the suite with a coverage report
python -m pytest --cov=src --cov-report=xml

# Check formatting the way CI does
black --check .
```

New or changed functionality must be covered by tests, and the whole suite must pass before a pull request can be merged.

---

## Code Style

* **Formatting:** `black` is required
* No other linters are currently enforced

All code must be formatted before submission.

---

## Hardware and Model Support

* Contributions must function correctly on:

  * CPU-only systems
  * GPU-enabled systems (when available)
* Code should gracefully handle missing GPU support
* YOLOEZ currently targets **Ultralytics YOLO11**
* Changes affecting model backends must be discussed first

---

## GUI and UX Contributions

YOLOEZ is a GUI-first application.

For GUI and UX changes:

* Maintain consistency with existing workflows
* Prefer clarity over advanced configuration
* Assume users may not have ML or Python experience
* Include screenshots or screen recordings
* Keep tooltips and instructional popups accurate

UX regressions are treated as bugs.

---

## Licensing and Attribution

YOLOEZ is licensed under **AGPL-3.0-or-later**.

By contributing, you agree that:

* Your contributions are licensed under AGPL-3.0-or-later
* Existing copyright headers must be preserved
* New files must include appropriate headers
* You should add yourself to [AUTHORS.md](AUTHORS.md) after making a substantive contribution

If you are unsure about attribution practices, please ask.

---

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Acknowledgements

Thank you for helping improve YOLOEZ.

Community contributions are essential to making this tool reliable, usable, and impactful for researchers and practitioners.
