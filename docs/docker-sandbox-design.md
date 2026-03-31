# Docker Sandbox Design

## Goal

Add sandboxed script execution to agentic-hdf5 so that generated scripts (Python, shell) run inside a Docker container rather than on the host. The same container persists for the duration of a session, avoiding cold-start overhead on every call.

## Architecture

```
Claude Code / MCP Client
        │
        ▼
   MCP Server (mcp_server.py)
        │
        ▼
   SandboxManager (sandbox.py)     ← new module
        │
        ├─ start()     → docker run -d ... → container_id
        ├─ exec(code)  → docker exec ... → stdout/stderr
        ├─ upload(src) → docker cp ... → copies file into container
        └─ stop()      → docker rm -f ...
```

**Container lifecycle**: One container per `SandboxManager` instance. The MCP server holds a single instance, started lazily on first sandbox tool call, reused for all subsequent calls, stopped on server shutdown.

## Steps

### 1. Dockerfile
- Base: `python:3.12-slim`
- Install: `h5py numpy matplotlib xarray` (matches host deps)
- Workdir: `/workspace`
- No entrypoint (container runs as a long-lived sleep process)

### 2. `SandboxManager` class (`tools/sandbox.py`)
- `start()` — build image if needed, `docker run -d` with security flags (`--cap-drop ALL`, `--no-new-privileges`, `--network none`, mem/cpu limits)
- `exec_code(code: str, language: str) -> dict` — write code to temp file, `docker cp` it in, `docker exec` to run it, return stdout/stderr/exit_code
- `upload_file(host_path, container_path)` — `docker cp` for getting HDF5 files into sandbox
- `download_file(container_path, host_path)` — `docker cp` out (for results)
- `stop()` — kill and remove container
- Idle timeout: auto-stop after configurable period (default 30 min)

### 3. MCP tools (`mcp_server.py`)
Two new tools:
- `sandbox_exec(code: str, language: str = "python", files: list[str] = [])` — copies listed files into container, executes code, returns output
- `sandbox_reset()` — tears down and restarts the container (clean slate)

### 4. Tests
- Unit tests for `SandboxManager` (requires Docker daemon)
- Pytest marker `@pytest.mark.docker` so CI can skip if Docker unavailable
- Test: start → exec Python → verify output
- Test: file upload → exec that reads it → verify
- Test: stop/restart cycle
- Test: security (verify `--network none` blocks outbound)
