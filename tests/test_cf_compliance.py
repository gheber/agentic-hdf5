"""Tests for the check_cf_compliance tool."""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

# Ensure tools/h5py bare imports resolve
_h5py_dir = str(Path(__file__).resolve().parent.parent / "tools" / "h5py")
if _h5py_dir not in sys.path:
    sys.path.insert(0, _h5py_dir)

from tools.h5py.check_cf_compliance import check_cf_compliance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Script templates are run in subprocesses so that netCDF4 initializes HDF5
# before h5py does — the same conflict that requires the tool's own subprocess.

_COMPLIANT_NC4_SCRIPT = """
import sys, numpy as np, netCDF4 as nc
path = sys.argv[1]
ds = nc.Dataset(path, "w", format="NETCDF4")
ds.Conventions = "CF-1.11"
ds.createDimension("time", 5)
ds.createDimension("lat", 3)
ds.createDimension("lon", 3)
t = ds.createVariable("time", "f8", ("time",))
t.standard_name = "time"
t.units = "seconds since 1970-01-01 00:00:00"
t.calendar = "standard"
t[:] = np.arange(5, dtype="f8")
lat = ds.createVariable("lat", "f4", ("lat",))
lat.standard_name = "latitude"
lat.units = "degrees_north"
lat[:] = np.array([-30.0, 0.0, 30.0], dtype="f4")
lon = ds.createVariable("lon", "f4", ("lon",))
lon.standard_name = "longitude"
lon.units = "degrees_east"
lon[:] = np.array([-60.0, 0.0, 60.0], dtype="f4")
sst = ds.createVariable("sst", "f4", ("time", "lat", "lon"), fill_value=-9999.0)
sst.standard_name = "sea_surface_temperature"
sst.units = "degrees_Celsius"
sst.long_name = "Sea Surface Temperature"
sst.coordinates = "lat lon"
sst[:] = np.random.rand(5, 3, 3).astype("f4") * 30
ds.close()
"""

_NONCOMPLIANT_NC4_SCRIPT = """
import sys, numpy as np, netCDF4 as nc
path = sys.argv[1]
ds = nc.Dataset(path, "w", format="NETCDF4")
# No Conventions attribute
ds.createDimension("time", 5)
t = ds.createVariable("time", "f8", ("time",))
# Malformed time units: trailing comment breaks UDUNITS (mirrors the ADCIRC/STOFS bug)
t.units = "seconds since 1970-01-01 00:00:00   ! model base date"
t[:] = np.arange(5, dtype="f8")
# Variable with no units or standard_name
raw = ds.createVariable("raw_signal", "f4", ("time",))
raw[:] = np.random.rand(5).astype("f4")
ds.close()
"""


@pytest.fixture
def plain_h5_file(tmp_path):
    """A plain HDF5 file written with h5py — not a NetCDF4 file."""
    path = str(tmp_path / "plain.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(10))
    return path


@pytest.fixture
def compliant_nc4_file(tmp_path):
    """A minimal, fully CF-1.11-compliant NetCDF4 file."""
    path = str(tmp_path / "compliant.nc")
    subprocess.run([sys.executable, "-c", _COMPLIANT_NC4_SCRIPT, path], check=True)
    return path


@pytest.fixture
def noncompliant_nc4_file(tmp_path):
    """A NetCDF4 file with intentional CF violations (malformed time units, missing Conventions)."""
    path = str(tmp_path / "noncompliant.nc")
    subprocess.run([sys.executable, "-c", _NONCOMPLIANT_NC4_SCRIPT, path], check=True)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plain_hdf5_returns_not_applicable(plain_h5_file):
    """Plain HDF5 files should be rejected; CF compliance does not apply."""
    result = check_cf_compliance(plain_h5_file)
    assert result["status"] == "not_applicable"
    assert "NetCDF4" in result["message"]


def test_nonexistent_file_returns_error():
    """A file path that doesn't exist should return a status=error dict."""
    result = check_cf_compliance("/tmp/does_not_exist_ahdf5.nc")
    assert result["status"] == "error"


def test_invalid_cf_version_returns_error(compliant_nc4_file):
    """An unrecognised CF version string should return a status=error dict."""
    result = check_cf_compliance(compliant_nc4_file, cf_version="99.0")
    assert result["status"] == "error"
    assert "available" in result["message"].lower()


def test_compliant_file_result_structure(compliant_nc4_file):
    """A valid NetCDF4 file should produce a well-formed result dict."""
    result = check_cf_compliance(compliant_nc4_file)
    assert result["status"] == "ok"
    assert result["cf_version_checked"] == "1.11"
    assert "score" in result
    assert "issues" in result
    assert "issue_counts" in result
    assert "passed" in result
    score = result["score"]
    assert score["possible"] > 0
    assert 0 <= score["percent"] <= 100


def test_compliant_file_has_no_high_severity_issues(compliant_nc4_file):
    """A well-formed CF-1.11 file should produce zero high-severity issues."""
    result = check_cf_compliance(compliant_nc4_file)
    assert result["status"] == "ok"
    assert result["issue_counts"]["high"] == 0


def test_noncompliant_file_detects_high_severity_issues(noncompliant_nc4_file):
    """A file with malformed time units should surface at least one high-severity issue."""
    result = check_cf_compliance(noncompliant_nc4_file)
    assert result["status"] == "ok"
    assert result["issue_counts"]["high"] > 0


def test_malformed_time_units_flagged(noncompliant_nc4_file):
    """The specific malformed time units defect should appear in issue messages."""
    result = check_cf_compliance(noncompliant_nc4_file)
    assert result["status"] == "ok"
    all_messages = [m for issue in result["issues"] for m in issue["messages"]]
    time_unit_flagged = any("time" in m.lower() and "units" in m.lower() for m in all_messages)
    assert time_unit_flagged, f"Expected a time-units issue but got: {all_messages}"


def test_issues_have_required_fields(noncompliant_nc4_file):
    """Every issue in the result must carry severity, check name, score, and messages."""
    result = check_cf_compliance(noncompliant_nc4_file)
    assert result["status"] == "ok"
    for issue in result["issues"]:
        assert "severity" in issue
        assert issue["severity"] in ("high", "medium", "low")
        assert "check" in issue
        assert "score" in issue
        assert "possible" in issue
        assert "messages" in issue
        assert isinstance(issue["messages"], list)
        assert len(issue["messages"]) > 0
