"""Tests for the check_cf_compliance tool."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Ensure tools/h5py bare imports resolve
_h5py_dir = str(Path(__file__).resolve().parent.parent / "tools" / "h5py")
if _h5py_dir not in sys.path:
    sys.path.insert(0, _h5py_dir)

from tools.h5py.check_cf_compliance import check_cf_compliance


# Script templates run in subprocesses so netCDF4 initializes HDF5 before h5py does.

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
ds.createDimension("time", 5)
t = ds.createVariable("time", "f8", ("time",))
# Malformed time units: trailing comment breaks UDUNITS (mirrors the ADCIRC/STOFS bug)
t.units = "seconds since 1970-01-01 00:00:00   ! model base date"
t[:] = np.arange(5, dtype="f8")
ds.close()
"""


@pytest.fixture
def compliant_nc4_file(tmp_path):
    path = str(tmp_path / "compliant.nc")
    subprocess.run([sys.executable, "-c", _COMPLIANT_NC4_SCRIPT, path], check=True)
    return path


@pytest.fixture
def noncompliant_nc4_file(tmp_path):
    path = str(tmp_path / "noncompliant.nc")
    subprocess.run([sys.executable, "-c", _NONCOMPLIANT_NC4_SCRIPT, path], check=True)
    return path


def test_non_netcdf4_inputs_rejected(tmp_path):
    """Plain HDF5 and nonexistent files should return non-ok statuses."""
    plain_h5 = str(tmp_path / "plain.h5")
    with h5py.File(plain_h5, "w") as f:
        f.create_dataset("data", data=np.arange(10))

    assert check_cf_compliance(plain_h5)["status"] == "not_applicable"
    assert check_cf_compliance("/tmp/does_not_exist_ahdf5.nc")["status"] == "error"


def test_compliant_file(compliant_nc4_file):
    """A well-formed CF-1.11 file should score clean with a valid result structure."""
    result = check_cf_compliance(compliant_nc4_file)
    assert result["status"] == "ok"
    assert result["cf_version_checked"] == "1.11"
    assert result["issue_counts"]["high"] == 0
    assert result["score"]["possible"] > 0
    assert 0 <= result["score"]["percent"] <= 100


def test_noncompliant_file(noncompliant_nc4_file):
    """A file with malformed time units should surface high-severity issues with correct structure."""
    result = check_cf_compliance(noncompliant_nc4_file)
    assert result["status"] == "ok"
    assert result["issue_counts"]["high"] > 0
    all_messages = [m for issue in result["issues"] for m in issue["messages"]]
    assert any("time" in m.lower() and "units" in m.lower() for m in all_messages)
    for issue in result["issues"]:
        assert issue["severity"] in ("high", "medium", "low")
        assert isinstance(issue["messages"], list) and len(issue["messages"]) > 0
