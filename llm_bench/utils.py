"""
Benchmark utilities: generate config permutations from search_space.json
and parse server args from generated config YAMLs.
"""

from __future__ import annotations

import itertools
import json
import shlex
from pathlib import Path
from typing import Any
import subprocess
import yaml
import os
import time
from typing import Optional
import requests
import psutil
import threading
import signal
import sys

def load_search_space(path: str | Path) -> dict[str, Any]:
    """Load search space from a JSON file."""
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def _product_dict(**kwargs: list[Any]) -> list[dict[str, Any]]:
    """Cartesian product of named value lists; returns list of dicts."""
    keys = list(kwargs.keys())
    value_lists = [kwargs[k] for k in keys]
    result = []
    for combo in itertools.product(*value_lists):
        result.append(dict(zip(keys, combo)))
    return result


def generate_config_permutations(
    search_space_path: str | Path,
    output_dir: str | Path,
    *,
    name_prefix: str = "config",
    name_zero_pad: int = 0,
) -> list[Path]:
    """
    Read search_space.json and generate one config.yaml per permutation.

    Each config has:
      - base_command
      - environment_variables (key-value for that permutation)
      - server_args (key-value for that permutation)

    Returns paths to created YAML files.
    """
    search_space_path = Path(search_space_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_search_space(search_space_path)
    base_command = data["base_command"]
    server_args_space = data.get("server_args", {})
    env_space = data.get("environment_variables", {})

    server_combos = _product_dict(**server_args_space) if server_args_space else [{}]
    env_combos = _product_dict(**env_space) if env_space else [{}]

    total = len(server_combos) * len(env_combos)
    pad = name_zero_pad or len(str(max(1, total - 1)))
    created: list[Path] = []
    idx = 0
    for server_combo, env_combo in itertools.product(server_combos, env_combos):
        config = {
            "base_command": base_command,
            "environment_variables": env_combo,
            "server_args": server_combo,
        }
        out_path = output_dir / f"{name_prefix}_{str(idx).zfill(pad)}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        created.append(out_path)
        idx += 1
    return created


def get_server_args(config_path: str | Path) -> list[str]:
    """
    Read a config YAML and return server args as a list of strings suitable
    to append to base_command.

    Rule: keys with value null or false are omitted. All other values are
    emitted as --key=value (or --key for boolean true if desired; we use
    --key=true for consistency).
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return get_server_args_from_config(config)


def get_server_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    From a config dict (with 'server_args' and optionally 'base_command'),
    return server args as a list of strings. Keys with value null or false
    are omitted.
    """
    server_args = config.get("server_args") or {}
    out: list[str] = []
    for k, v in server_args.items():
        if v is None or v is False:
            continue
        arg_key = k
        if v is True:
            out.append(f"--{arg_key}")
        else:
            out.append(f"--{arg_key}={v}")
    return out

def get_server_args_from_config_for_mlflow(config_path: str | Path) -> dict[str, str]:
    """
    From a config dict (with 'server_args' and optionally 'base_command'),
    return server args as a dict of strings for MLflow. Keys with value null or false
    are omitted.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    server_args = config.get("server_args") or {}
    out: dict[str, str] = {}
    for k, v in server_args.items():
        if v is None or v is False:
            continue
        arg_key = k
        if v is True:
            out[f"server_args::{arg_key}"] = "true"
        else:
            out[f"server_args::{arg_key}"] = str(v)
    return out


def get_full_command(config_path: str | Path) -> list[str]:
    """
    Return base_command + server args (excluding null/false) as a single
    list of strings for launching the server.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    base = config.get("base_command") or []
    args = get_server_args_from_config(config)
    return list(base) + args


def _env_assignments(env: dict[str, Any]) -> list[str]:
    """Env vars as KEY=value, omitting null/false; values shell-quoted."""
    out = []
    for k, v in env.items():
        if v is None or v is False:
            continue
        out.append(f"{k}={shlex.quote(str(v))}")
    return out


def get_full_command_shell(config_path: str | Path) -> str:
    """
    Return the full command as a single string for use with sh -c:
    environment_variables (KEY=value, null/false omitted) followed by
    base_command + server_args. Safe for subprocess.run(["sh", "-c", result]).
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return get_full_command_shell_from_config(config)


def get_full_command_shell_from_config(config_path: str | Path) -> str:
    """
    Same as get_full_command_shell but takes a config dict. Builds
    ENV1=val1 ENV2=val2 base_cmd arg1 arg2 ... with proper quoting.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    env = config.get("environment_variables") or {}
    base = config.get("base_command") or []
    args = get_server_args_from_config(config)
    parts = _env_assignments(env) + [shlex.quote(str(p)) for p in base + args]
    return " ".join(parts)

def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    Supports leading KEY=VALUE env vars (e.g. "VAR=1 python script.py") so that
    notebook/CI commands work without requiring shell=True.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    env = os.environ.copy()
    i = 0
    while i < len(parts):
        part = parts[i]
        if "=" in part and not part.startswith("-") and not part.startswith("/"):
            key, _, value = part.partition("=")
            if key and value is not None and key.replace("_", "").isalnum():
                env[key] = value
                i += 1
                continue
        break
    parts = parts[i:]
    if not parts:
        raise ValueError(
            "Command contains only environment variable assignments, no executable"
        )
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = 8000):
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    
    lock_socket = None

    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)

    return process

def _raise_if_process_exited(process: Optional[Any]) -> None:
    if process is None:
        return

    if hasattr(process, "poll"):
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited with code {return_code}")
        return

    if hasattr(process, "is_alive") and not process.is_alive():
        return_code = getattr(process, "exitcode", None)
        if return_code is None:
            raise RuntimeError("Server process exited")
        raise RuntimeError(f"Server process exited with code {return_code}")


def wait_for_http_ready(
    url: str,
    timeout: Optional[int] = None,
    process: Optional[Any] = None,
    headers: Optional[dict] = None,
    request_timeout: int = 5,
) -> None:
    """Wait for an HTTP endpoint to return status 200."""
    start_time = time.perf_counter()
    while True:
        _raise_if_process_exited(process)
        try:
            response = requests.get(url, headers=headers, timeout=request_timeout)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            _raise_if_process_exited(process)

        if _is_wait_timeout(start_time, timeout):
            raise TimeoutError(
                f"Endpoint {url} did not become ready within timeout period"
            )
        time.sleep(1)

def _is_wait_timeout(start_time: float, timeout: Optional[int]) -> bool:
    if timeout is None:
        return False
    return time.perf_counter() - start_time > timeout

def wait_for_server(
    base_url: str,
    timeout: int = None,
    process: Optional[subprocess.Popen] = None,
) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server.
        timeout: Maximum time to wait in seconds. None means wait forever.
        process: Optional server process used for early-exit checks.
    """
    wait_for_http_ready(
        url=f"{base_url}/models",
        timeout=timeout,
        process=process,
        headers={"Authorization": "Bearer None"},
    )
    time.sleep(5)


def kill_process(process: subprocess.Popen) -> None:
    """Kill a process."""
    process.terminate()
    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

    return process

def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass

def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """
    kill_process_tree(process.pid)

def ensure_model_downloaded(model_name: str):

    from huggingface_hub import snapshot_download
    snapshot_download(model_name)