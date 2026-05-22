# Contributing to snapAnalysis 

We welcome your feedback and contributions to `snapAnalysis`! 

## Bug reports, issues, and feedback
If you find a bug or other issue, want to request a new feature, or have other feedback, please submit an [Issue](https://github.com/hfoote/snapAnalysis/issues). When reporting bugs, please do your best to describe the problem in as much detail as possible, ideally including a minimal example to reproduce the bug. This helps us help you! 

## Contributing code

We encourage code contributions through pull requests! 

To install `snapAnalysis` for development, you'll need to install it from source using the [`uv` package manager](https://docs.astral.sh/uv/), which we use to develop `snapAnalysis`. If you need to install `uv`, see the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

1. Clone this repo
```bash
git clone https://github.com/hfoote/snapAnalysis.git
```

2. Install `snapAnalysis` and its dependencies (including its development dependencies) into a virtual environment with `uv`
```bash
cd snapAnalysis
uv sync --dev
```

When developing `snapAnalysis`, please continue to use `uv` for dependency and environment management according to their [documentation](https://docs.astral.sh/uv/). 

**Before submitting a PR**, please ensure that you have added tests for your code in the [tests folder](/tests/), and that the entire test suite and linter checks pass.
```bash
uv run pytest
uv tool run ruff check
```