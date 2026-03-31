import json
import os
import subprocess
import sys
import tempfile

try:
    from tools.h5py.registry import hdf5_tool
except ImportError:
    def hdf5_tool(**_kw):
        return lambda f: f

# Subprocess script that runs compliance-checker in a fresh process.
# Must import netCDF4 before any h5py to avoid HDF5 library state conflicts.
_CHECKER_SCRIPT = """
import json, os, sys, tempfile

filepath = sys.argv[1]
cf_version = sys.argv[2]
checker_name = f"cf:{cf_version}"

# Import netCDF4 first — before any h5py — to avoid HDF5 library init conflicts.
try:
    import netCDF4 as nc_lib
    ds = nc_lib.Dataset(filepath)
    file_format = ds.file_format
    conventions_attr = getattr(ds, "Conventions", None) or getattr(ds, "convention", None)
    ds.close()
except Exception as e:
    print(json.dumps({"status": "not_applicable", "message": str(e)}))
    sys.exit(0)

try:
    from compliance_checker.runner import ComplianceChecker, CheckSuite
except ImportError:
    print(json.dumps({"status": "error", "message": "compliance-checker not installed. Install with: pip install compliance-checker"}))
    sys.exit(0)

CheckSuite.load_all_available_checkers()
if checker_name not in CheckSuite.checkers:
    available = sorted(k for k in CheckSuite.checkers if k.startswith("cf:"))
    print(json.dumps({"status": "error", "message": f"CF version not available. Available: {available}"}))
    sys.exit(0)

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
    tmp_path = tf.name

try:
    return_value, _ = ComplianceChecker.run_checker(
        ds_loc=filepath,
        checker_names=[checker_name],
        verbose=False,
        criteria="normal",
        output_filename=tmp_path,
        output_format="json",
    )
    with open(tmp_path) as f:
        raw = json.load(f)
finally:
    os.unlink(tmp_path)

result = raw.get(checker_name, {})
issues = []
for priority_key, severity in [
    ("high_priorities", "high"),
    ("medium_priorities", "medium"),
    ("low_priorities", "low"),
]:
    for check in result.get(priority_key, []):
        if check["msgs"]:
            score, total = check["value"]
            issues.append({
                "severity": severity,
                "check": check["name"],
                "score": score,
                "possible": total,
                "messages": check["msgs"],
            })

scored = result.get("scored_points", 0)
possible = result.get("possible_points", 0)
print(json.dumps({
    "status": "ok",
    "file_format": file_format,
    "cf_version_declared": str(conventions_attr) if conventions_attr else None,
    "score": {
        "scored": scored,
        "possible": possible,
        "percent": round(100 * scored / possible, 1) if possible else 0,
    },
    "issue_counts": {
        "high": result.get("high_count", 0),
        "medium": result.get("medium_count", 0),
        "low": result.get("low_count", 0),
    },
    "issues": issues,
    "passed": return_value == 0,
}))
"""


@hdf5_tool(
    category="inspection",
    keywords=["cf", "compliance", "conventions", "netcdf", "standards", "units", "metadata", "check", "validate"],
    use_cases=[
        "Checking whether a NetCDF4 file conforms to CF conventions before publishing",
        "Identifying missing or invalid metadata for THREDDS, ERDDAP, or OPeNDAP services",
        "Assessing data interoperability and reusability",
        "Preparing files for scientific data repositories (e.g., Zenodo, Dataverse)",
    ],
)
def check_cf_compliance(filepath: str, cf_version: str = "1.11") -> dict:
    """
    Check a NetCDF4/HDF5 file for CF convention compliance using the IOOS compliance-checker.

    Returns a structured report of compliance issues organized by severity, with CF spec
    section references and scores. Only applicable to NetCDF4-format files; plain HDF5 files
    written directly with h5py are not subject to CF conventions.

    Runs the checker in a subprocess to avoid HDF5 library state conflicts between h5py
    and the netCDF4 library.

    Args:
        filepath: Path to the NetCDF4 or HDF5 file
        cf_version: CF conventions version to check against (e.g., "1.6", "1.7", "1.8", "1.9",
                    "1.10", "1.11"). Defaults to latest (1.11).

    Returns:
        Dictionary with compliance score, issues by severity, and file format information.
        Issues include CF spec section reference, severity, partial score, and specific messages.
    """
    # Use h5py (already loaded) to detect NetCDF4 files before spinning up the subprocess.
    # _NCProperties is written by the netCDF4 library and is the definitive marker.
    import h5py
    try:
        with h5py.File(filepath, "r") as f:
            is_netcdf4 = "_NCProperties" in f.attrs
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not open file: {e}",
            "filepath": filepath,
        }

    if not is_netcdf4:
        return {
            "status": "not_applicable",
            "message": (
                "File does not appear to be a NetCDF4 file (no _NCProperties marker found). "
                "CF compliance applies to NetCDF4-format files. Plain HDF5 files written "
                "directly with h5py are not subject to CF conventions."
            ),
            "filepath": filepath,
        }

    # Run compliance-checker in a subprocess so netCDF4 initializes HDF5 before h5py does.
    filepath = os.path.abspath(filepath)
    proc = subprocess.run(
        [sys.executable, "-c", _CHECKER_SCRIPT, filepath, cf_version],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "status": "error",
            "message": f"Checker subprocess failed: {proc.stderr.strip() or 'no output'}",
            "filepath": filepath,
        }

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": f"Could not parse checker output: {proc.stdout[:200]}",
            "filepath": filepath,
        }

    result["filepath"] = filepath
    result["cf_version_checked"] = cf_version
    return result
