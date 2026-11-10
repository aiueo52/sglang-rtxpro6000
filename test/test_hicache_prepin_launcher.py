# SPDX-License-Identifier: Apache-2.0
"""Check HiCache prepin settings through the launcher's print mode."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPIN = "SGLANG_HICACHE_PREPIN_BYTES"


def launch(**overrides):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(ROOT),
        "PYTHON": sys.executable,
        "TARGET_MODEL": str(ROOT),
        "TOKEN_MAP": "none",
        "CONC": "8",
        "CUDA_VISIBLE_DEVICES": "",
        "FLASH_NEXT_ADMIN_API_KEY": "cpu-test-placeholder",
    }
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(ROOT / "serve-fast.sh"), "wa", "--print"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert len(result.stdout.splitlines()) == 1
    return json.loads(result.stdout)


def test_default_prepin():
    assert launch()["env"][PREPIN] == "32200000000"


@pytest.mark.parametrize("size,expected", [
    ("16", "16200000000"),
    ("032", "32200000000"),
])
def test_prepin_size(size, expected):
    assert launch(FLASH_NEXT_HICACHE_GB=size)["env"][PREPIN] == expected


@pytest.mark.parametrize("overrides", [
    {"FLASH_NEXT_HICACHE_GB": "0"},
    {"CONC": "1"},
    {"FLASH_NEXT_EDITION": "hyperpro"},
])
def test_prepin_not_exported(overrides):
    assert PREPIN not in launch(**overrides)["env"]


@pytest.mark.parametrize("value", ["0", "12345", ""])
def test_explicit_prepin_preserved(value):
    assert launch(**{PREPIN: value})["env"][PREPIN] == value


def test_disabling_prepin_changes_only_its_value():
    default = launch()
    disabled = launch(**{PREPIN: "0"})
    assert default["argv"] == disabled["argv"]
    assert default["env"].pop(PREPIN) == "32200000000"
    assert disabled["env"].pop(PREPIN) == "0"
    assert default == disabled
