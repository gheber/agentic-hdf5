"""
Docker-based sandbox for executing scripts in an isolated container.

The SandboxManager maintains a single persistent container per session,
avoiding cold-start overhead on repeated executions.
"""

import atexit
import logging
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IMAGE_NAME = "ahdf5-sandbox"
DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile.sandbox"
CONTAINER_PREFIX = "ahdf5-sandbox-"

# Defaults
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = "1.0"
DEFAULT_IDLE_TIMEOUT = 1800  # 30 minutes


class SandboxManager:
    """Manages a persistent Docker container for sandboxed code execution."""

    def __init__(
        self,
        *,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        cpu_limit: str = DEFAULT_CPU_LIMIT,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        network: bool = False,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.idle_timeout = idle_timeout
        self.network = network
        self.container_id: Optional[str] = None
        self._lock = threading.Lock()
        self._last_activity: float = 0
        self._idle_timer: Optional[threading.Timer] = None
        atexit.register(self.stop)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Start the sandbox container (builds image if needed). Returns container ID."""
        with self._lock:
            if self.container_id and self._container_running():
                return self.container_id

            self._ensure_docker()
            self._ensure_image()

            name = f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:8]}"
            cmd = [
                "docker", "run", "-d",
                "--name", name,
                "--memory", self.memory_limit,
                "--cpus", self.cpu_limit,
                "--pids-limit", "64",
            ]
            if not self.network:
                cmd += ["--network", "none"]

            cmd.append(IMAGE_NAME)

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.container_id = result.stdout.strip()[:12]
            self._touch()
            logger.info("Sandbox container started: %s", self.container_id)
            return self.container_id

    def stop(self) -> None:
        """Stop and remove the sandbox container."""
        with self._lock:
            self._cancel_idle_timer()
            if self.container_id:
                subprocess.run(
                    ["docker", "rm", "-f", self.container_id],
                    capture_output=True, text=True,
                )
                logger.info("Sandbox container stopped: %s", self.container_id)
                self.container_id = None

    def exec_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> dict:
        """Execute code in the sandbox. Returns {stdout, stderr, exit_code}."""
        self.start()  # lazy start

        interpreter = "python" if language == "python" else "bash"

        try:
            result = subprocess.run(
                ["docker", "exec", "-i", self.container_id, interpreter],
                input=code,
                capture_output=True, text=True, timeout=timeout,
            )

            self._touch()
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "exit_code": -1,
            }

    def upload_file(self, host_path: str, container_path: str = "/workspace/") -> dict:
        """Copy a file from host into the sandbox."""
        self.start()
        host_path = str(Path(host_path).resolve())
        try:
            subprocess.run(
                ["docker", "cp", host_path, f"{self.container_id}:{container_path}"],
                capture_output=True, text=True, check=True,
            )
            self._touch()
            return {"status": "ok", "container_path": container_path}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": e.stderr}

    def download_file(self, container_path: str, host_path: str) -> dict:
        """Copy a file from the sandbox to the host."""
        self.start()
        host_path = str(Path(host_path).resolve())
        try:
            subprocess.run(
                ["docker", "cp", f"{self.container_id}:{container_path}", host_path],
                capture_output=True, text=True, check=True,
            )
            self._touch()
            return {"status": "ok", "host_path": host_path}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": e.stderr}

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self.container_id is not None and self._container_running()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_docker(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker not found on PATH. Install Docker to use sandbox execution."
            )

    def _ensure_image(self) -> None:
        """Build the sandbox image if it doesn't exist."""
        result = subprocess.run(
            ["docker", "images", "-q", IMAGE_NAME],
            capture_output=True, text=True,
        )
        if not result.stdout.strip():
            logger.info("Building sandbox image %s ...", IMAGE_NAME)
            subprocess.run(
                ["docker", "build", "-t", IMAGE_NAME, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)],
                capture_output=True, text=True, check=True,
            )

    def _container_running(self) -> bool:
        if not self.container_id:
            return False
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id],
            capture_output=True, text=True,
        )
        return result.stdout.strip() == "true"

    def _touch(self) -> None:
        """Reset idle timeout."""
        self._last_activity = time.time()
        self._cancel_idle_timer()
        if self.idle_timeout > 0:
            self._idle_timer = threading.Timer(self.idle_timeout, self._idle_stop)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _idle_stop(self) -> None:
        logger.info("Sandbox idle timeout reached, stopping container")
        self.stop()


# ------------------------------------------------------------------
# Module-level API (matches pattern of other tools in tools/h5py/)
# ------------------------------------------------------------------

_default_sandbox = SandboxManager()


def sandbox_exec(
    code: str,
    language: str = "python",
    files: Optional[list[str]] = None,
    timeout: int = 30,
) -> dict:
    """Execute code in an isolated Docker sandbox.

    State persists across calls within a session. Optionally upload files first.
    """
    if files:
        for fpath in files:
            result = _default_sandbox.upload_file(fpath)
            if result.get("status") == "error":
                return result
    return _default_sandbox.exec_code(code, language=language, timeout=timeout)


def sandbox_reset() -> dict:
    """Tear down and restart the sandbox container for a clean environment."""
    _default_sandbox.stop()
    _default_sandbox.start()
    return {"status": "ok", "message": "Sandbox reset"}
