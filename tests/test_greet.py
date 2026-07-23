import subprocess
import sys

import pytest

from hello_sdd.greet import EmptyNameError, greet


def test_greet_formats_name() -> None:
    assert greet("Ada") == "Hello, Ada!"


def test_greet_strips_whitespace() -> None:
    assert greet("  Ada  ") == "Hello, Ada!"


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_greet_rejects_empty(name: str) -> None:
    with pytest.raises(EmptyNameError):
        greet(name)


def test_cli_success() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hello_sdd", "Ada"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "Hello, Ada!\n"


def test_cli_rejects_blank_name() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hello_sdd", "  "],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hello_sdd", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "name" in result.stdout
