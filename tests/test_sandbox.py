"""Tests for Docker sandbox execution.

Requires a running Docker daemon. Skip with: pytest -m "not docker"
"""

import os
from pathlib import Path

import pytest

from tools.sandbox import SandboxManager

pytestmark = pytest.mark.docker

# Use project dir for temp files (snap Docker can't access /tmp)
_TEST_TMPDIR = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="module")
def sandbox():
    """Shared sandbox for the test module — avoids repeated container startup."""
    mgr = SandboxManager(idle_timeout=0)
    yield mgr
    mgr.stop()


def test_lifecycle():
    """Start, verify running, stop, verify stopped. Also checks idempotent start."""
    mgr = SandboxManager(idle_timeout=0)
    cid1 = mgr.start()
    assert mgr.is_running
    cid2 = mgr.start()
    assert cid1 == cid2
    mgr.stop()
    assert not mgr.is_running


def test_exec_and_state_persistence(sandbox):
    """Python exec works, errors are captured, and state persists across calls."""
    # Success
    result = sandbox.exec_code('print("hello sandbox")')
    assert result["exit_code"] == 0 and "hello sandbox" in result["stdout"]

    # Error
    result = sandbox.exec_code("raise ValueError('boom')")
    assert result["exit_code"] != 0 and "ValueError" in result["stderr"]

    # State persists
    sandbox.exec_code("open('/workspace/_flag.txt','w').write('yes')")
    result = sandbox.exec_code("print(open('/workspace/_flag.txt').read())")
    assert "yes" in result["stdout"]


def test_timeout(sandbox):
    result = sandbox.exec_code("import time; time.sleep(10)", timeout=2)
    assert result["exit_code"] == -1 and "timed out" in result["stderr"].lower()


def test_upload_hdf5_and_inspect(sandbox, test_h5_file):
    """Upload an HDF5 file, read it with h5py inside the sandbox."""
    sandbox.upload_file(test_h5_file, "/workspace/test.h5")
    result = sandbox.exec_code(
        "import h5py\n"
        "with h5py.File('/workspace/test.h5') as f:\n"
        "    print(list(f['test_group'].keys()))\n"
    )
    assert result["exit_code"] == 0 and "data" in result["stdout"]


def test_network_disabled(sandbox):
    result = sandbox.exec_code(
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://1.1.1.1', timeout=3)\n"
        "except Exception as e:\n"
        "    print(f'blocked: {e}')\n"
    )
    assert result["exit_code"] == 0 and "blocked" in result["stdout"].lower()
