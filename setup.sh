#!/bin/bash

# Check Python version
REQUIRED_PYTHON="3.10"
PYTHON_VERSION=$(python$REQUIRED_PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

if [ "$(printf '%s\n' "$REQUIRED_PYTHON" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_PYTHON" ]; then
    echo "Error: Python $REQUIRED_PYTHON or higher is required. Found Python $PYTHON_VERSION"
    echo "Please install Python $REQUIRED_PYTHON and try again."
    exit 1
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment with Python 3.9
uv venv --python=python$REQUIRED_PYTHON

# Activate virtual environment
source .venv/bin/activate

# Verify Python version in virtual environment
VENV_PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$VENV_PYTHON_VERSION" != "$REQUIRED_PYTHON" ]; then
    echo "Error: Virtual environment Python version mismatch. Expected $REQUIRED_PYTHON, got $VENV_PYTHON_VERSION"
    exit 1
fi

# Install the package in development mode
uv pip install -e .

# Install Playwright browsers (Chromium only)
# playwright install chromium

echo "Setup complete! Virtual environment is ready with Python $VENV_PYTHON_VERSION"
echo "Start: python main.py" 