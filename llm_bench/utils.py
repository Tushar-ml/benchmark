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

def run_command(command: str):
    """Run a command in the background. Returns the Popen process (e.g. for kill_process)."""
    import subprocess
    import sys

    process = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    return process

def kill_process(process: subprocess.Popen) -> None:
    """Kill a process."""
    process.terminate()
    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

    return process

def wait_for_health():
    """Wait for the health endpoint to be ready."""
    import requests
    import time

    max_retries = 20
    for _ in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    return False

def ensure_model_downloaded(model_name: str):

    from huggingface_hub import snapshot_download
    snapshot_download(model_name)