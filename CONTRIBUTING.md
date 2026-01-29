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

- **Minimum supported version:** Python **3.12**
- **CI-tested version:** Python **3.12**

Contributors are encouraged to use Python 3.12 when possible.

### Installation

```bash
pip install -e .[dev]
````

This installs:

* Runtime dependencies
* Development dependencies (`pytest`, `pytest-qt`, etc.)

---

## Testing

* Tests are written using **pytest**
* GUI tests use **pytest-qt**
* Tests must pass on:

  * Windows
  * Linux (headless via `xvfb`)

To run tests locally:

```bash
pytest
```

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
* You should add yourself to the `AUTHORS` file after making a substantive contribution

If you are unsure about attribution practices, please ask.

---

## Code of Conduct

A formal Code of Conduct has not yet been added.

Until then, contributors are expected to engage respectfully, professionally, and constructively.

Harassment or unprofessional behavior will not be tolerated.

---

## Acknowledgements

Thank you for helping improve YOLOEZ.

Community contributions are essential to making this tool reliable, usable, and impactful for researchers and practitioners.

```