#!/usr/bin/env python3
"""
Test runner script for the CML-WD-PyTorch project.
Run this script to execute the test suite.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the test suite using pytest."""
    project_root = Path(__file__).parent

    print("Installing test dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[test]"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Failed to install test dependencies:")
        print(result.stderr)
        return False

    print("Running tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"], cwd=project_root
    )

    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
