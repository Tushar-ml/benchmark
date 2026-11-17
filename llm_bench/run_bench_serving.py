#!/usr/bin/env python3
"""
Benchmark runner with optional MLflow logging.

When MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME are set, results and model
params are logged to MLflow.  Otherwise, results are saved to a local CSV via
bench_serving.py's --summary-file flag (original behaviour).

Optional env vars for MLflow mode (put them in llm_bench/.env, see .env.example):
    MLFLOW_TRACKING_URI     – MLflow server URL
    MLFLOW_TRACKING_TOKEN   – bearer token (or MLFLOW_TRACKING_USERNAME/PASSWORD)

Default mode benchmarks an ALREADY RUNNING OpenAI-compatible server with real
coding prompts (SWE-bench Lite, see coding_dataset.py). Pass --launch-server for
the original behaviour (launch vLLM from ../configs search space, synthetic prompts).
"""

import asyncio
import os
import subprocess
from argparse import Namespace
from itertools import product
import time
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

# accept MLFLOW_USERNAME/MLFLOW_PASSWORD as aliases for the names the MLflow client reads
for _a, _b in (("MLFLOW_USERNAME", "MLFLOW_TRACKING_USERNAME"), ("MLFLOW_PASSWORD", "MLFLOW_TRACKING_PASSWORD")):
    if os.environ.get(_a) and not os.environ.get(_b):
        os.environ[_b] = os.environ[_a]

from bench_serving import run_benchmark

# TODO: server launch command and readiness check
# ── Benchmark matrix ─────────────────────────────────────────────────────

MODEL_NAME = os.getenv("BENCH_MODEL", "zai-org/GLM-4.6")
PORT = int(os.getenv("BENCH_PORT", "8000"))
CONCURRENCIES = [int(x) for x in os.getenv("BENCH_CONCURRENCIES", "1,10,20,50,100").split(",")]
INPUT_TOKS = [int(x) for x in os.getenv("BENCH_INPUT_TOKS", "100,1000,10000").split(",")]
OUTPUT_TOKS = [int(x) for x in os.getenv("BENCH_OUTPUT_TOKS", "200").split(",")]
PCMLS = [0]
NUM_REQUESTS = int(os.getenv("BENCH_NUM_REQUESTS", "0"))
SUMMARY_FILE = f"{MODEL_NAME.split('/')[-1]}.csv"
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", MODEL_NAME)
# Real-dataset mode (default): HF dataset id in SWE-bench format, or jsonl:/path/file.jsonl
CODING_DATASET = os.getenv("CODING_DATASET", "princeton-nlp/SWE-bench_Lite")
from coding_dataset import CODE_CORPUS
do_warmup = True
dry_run = False


MODEL_PARAMS = {}
REQUEST_PARAMS = {}

BASE_URL = os.getenv("BENCH_BASE_URL", f"http://localhost:{PORT}/v1")
API_KEY = os.getenv("BENCH_API_KEY", "")
# comma-separated "Name:value" pairs, e.g. BENCH_HEADERS=x-session-id:bench1
EXTRA_HEADERS = [h for h in os.getenv("BENCH_HEADERS", "").split(",") if h]

# ── MLflow helpers ───────────────────────────────────────────────────────

METRIC_KEYS = {
    "wall_time_s", "prompt_tokens_avg", "output_tokens_avg",
    "total_output_tokens", "total_prompt_tokens",
    "throughput_req_per_s", "throughput_output_tok_per_s", "throughput_total_tok_per_s",
    "avg_latency_ms", "median_latency_ms",
    "p90_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "min_latency_ms", "max_latency_ms",
    "avg_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "avg_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "avg_per_req_tok_per_s", "median_per_req_tok_per_s",
}


def use_mlflow() -> bool:
    return bool(
        os.environ.get("MLFLOW_TRACKING_URI")
    )

def get_device_info() -> dict:
    """Query nvidia-smi for GPU device information."""
    info = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,count",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            fields = ["device_info::gpu_name", "device_info::driver_version", "device_info::gpu_count"]
            for gpu_idx, line in enumerate(lines):
                values = [v.strip() for v in line.split(",")]
                for field, val in zip(fields, values):
                    info[field] = val

        smi_header = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        if smi_header.returncode == 0:
            for line in smi_header.stdout.splitlines():
                if "CUDA Version" in line:
                    info["device_info::cuda_version"] = line.split("CUDA Version:")[1].strip().split()[0]
                    break

    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu_error"] = "nvidia-smi not available"
    return info


def log_to_mlflow(mlflow, entries: dict, bench_params: dict):
    all_params = {**MODEL_PARAMS, **REQUEST_PARAMS, **bench_params}
    mlflow.log_params(all_params)

    metrics = {}
    for k, v in entries.items():
        if k in METRIC_KEYS:
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                pass
    mlflow.log_metrics(metrics)


# ── Server metadata for a running server (same key structure as the config-driven flow) ──

def _docker(container: str, *cmd: str) -> str:
    r = subprocess.run(["docker", "exec", container, *cmd], capture_output=True, text=True, timeout=30)
    return r.stdout if r.returncode == 0 else ""


def _parse_cmdline(tokens: list[str]) -> dict[str, str]:
    """`--flag value` / `--flag=value` / bare `--flag` -> {flag: value}; last occurrence wins (argparse)."""
    out, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            k, eq, v = t[2:].partition("=")
            if not eq:
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    v = tokens[i + 1]
                    i += 1
                else:
                    v = "true"
            # a later --no-x cancels an earlier --x and vice versa (argparse BooleanOptionalAction)
            out.pop(k[3:] if k.startswith("no-") else "no-" + k, None)
            out[k] = v
        i += 1
    return out


def get_running_server_metadata(containers: list[str]) -> dict:
    """Read real server args/versions from the live containers (not from a config file).

    Args identical across roles -> server_args::<arg>; differing ones (e.g. prefill vs decode)
    -> server_args::<role>::<arg>. Role comes from --disaggregation-mode.
    """
    per_role, versions = {}, {}
    for c in containers:
        # containers may share the host PID namespace (pid: host), so list every worker process
        # visible from this container and key them by --disaggregation-mode, not by container.
        pids = _docker(c, "sh", "-c", "pgrep -f '^python -m dynamo.vllm'").split()
        for pid in pids:
            raw = _docker(c, "cat", f"/proc/{pid}/cmdline")
            args = _parse_cmdline([t for t in raw.split("\0") if t])
            if args:
                per_role[args.get("disaggregation-mode", c)] = args
        pip = _docker(c, "pip", "list", "--format=freeze")
        for line in pip.splitlines():
            name, _, ver = line.partition("==")
            if name in ("vllm", "ai-dynamo", "nixl", "flashinfer-python", "torch"):
                versions[name] = ver
    if not per_role:
        print(f"WARNING: no dynamo.vllm process found via containers {containers}")

    params: dict = {}
    roles = list(per_role)
    for k in sorted({k for a in per_role.values() for k in a}):
        vals = {r: per_role[r].get(k) for r in roles}
        if len({v for v in vals.values()}) == 1:
            params[f"server_args::{k}"] = str(vals[roles[0]])[:500]
        else:
            for r, v in vals.items():
                if v is not None:
                    params[f"server_args::{r}::{k}"] = str(v)[:500]
    if versions.get("vllm"):
        params["version"] = versions["vllm"]
    for name in ("ai-dynamo", "nixl", "flashinfer-python", "torch"):
        if versions.get(name):
            params[f"deployment::{name}_version"] = versions[name]
    params["deployment::topology"] = os.getenv("BENCH_TOPOLOGY", "+".join(roles))
    return params


def get_quantization(model: str) -> dict:
    """quantization / quantization_config from the model's HF config.json (offline cache)."""
    import glob
    hub = os.path.join(os.getenv("HF_HOME_HOST", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots", "*", "config.json"))
    if not hits:
        return {}
    qc = json.load(open(hits[0])).get("quantization_config") or {}
    out = {"quantization": qc.get("quant_method", "unknown")}
    if qc:
        out["quantization::config"] = json.dumps(qc, sort_keys=True)[:500]
    return out


def get_gpu_type() -> str:
    name = get_device_info().get("device_info::gpu_name", "")
    import re
    m = re.search(r"\b([BHAL]\d{2,3}|GB\d{2,3})\b", name)
    return (m.group(1) if m else name).lower()


# ── Main ─────────────────────────────────────────────────────────────────

def warmup():
    import requests
    for _ in range(5):
        model_name = requests.get(BASE_URL + "/models").json()["data"][0]["id"]
        response = requests.post(BASE_URL + "/chat/completions", json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 100,
            "temperature": 0.0,
        })
        print(response.json())

    
def main(dry_run_json=None, dataset_prompts=None):
    mlflow_enabled = use_mlflow()

    if mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(experiment_name)
        print("MLflow logging enabled")
    else:
        mlflow = None
        print(f"MLflow not configured – results will be saved to {SUMMARY_FILE}")

    input_toks = INPUT_TOKS  # dataset_prompts: {input_tokens: [prompts]} (real issue + real code) or None
    for concurrency, input_tok, output_tok, pcml_frac in product(
        CONCURRENCIES, input_toks, OUTPUT_TOKS, PCMLS,
    ):
        pcml = 0 if dataset_prompts else int(pcml_frac * input_tok)
        prompts_for_run = dataset_prompts[input_tok] if dataset_prompts else None
        import bench_serving, itertools
        bench_serving._dataset_counter = itertools.count()  # every run walks its prompt list from the start

        args = Namespace(
            base_url=BASE_URL,
            num_requests=NUM_REQUESTS,
            concurrency=concurrency,
            request_rate=REQUEST_PARAMS["request_rate"],
            model=None,
            chat=REQUEST_PARAMS["chat"],
            stream=REQUEST_PARAMS["stream"],
            prompt_tokens=input_tok,
            dataset_prompts=prompts_for_run,
            ignore_eos=not dataset_prompts,  # real prompts: let the model stop naturally
            prompt_cache_max_len=pcml,
            prompt_cache_percentage=pcml_frac,
            prompt_randomize=True,
            prompt_text=None,
            prompt_chars=None,
            max_tokens_range=0.0,
            max_tokens=output_tok,
            temperature=REQUEST_PARAMS["temperature"],
            api_key=API_KEY,
            header=EXTRA_HEADERS,
            timeout=600,
            summary_file=None if mlflow_enabled else SUMMARY_FILE,
            show_response=False,
        )

        if mlflow_enabled:
            bench_params = {
                "num_requests": NUM_REQUESTS,
                "concurrency": concurrency,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "prompt_cache_fraction": pcml_frac,
                "prompt_cache_max_len": pcml,
                "prompt_randomize": not dataset_prompts,
            }
            if dataset_prompts:
                bench_params["dataset"] = CODING_DATASET + " + " + CODE_CORPUS
                bench_params["dataset_num_prompts"] = len(prompts_for_run)
                bench_params["prompt_source"] = "real issue + real python source, fixed length"
            gpu_info = get_device_info()
            bench_params.update(gpu_info)

            run_name = os.environ.get(
                "MLFLOW_RUN_NAME",
                f"{MODEL_NAME.split('/')[-1]}_c{concurrency}_i{input_tok}_o{output_tok}_{time.strftime('%Y-%m-%d-%H-%M-%S')}",
            )

            if not dry_run_json:
                with mlflow.start_run(run_name=run_name):
                    exit_code, entries = asyncio.run(run_benchmark(args))

                    if entries:
                        log_to_mlflow(mlflow, entries, bench_params)
                        print(f"Logged run '{run_name}' to MLflow ({len(entries)} fields)")
                    else:
                        mlflow.set_tag("status", "all_requests_failed")
                        print(f"Run '{run_name}' failed – tagged in MLflow")
            else:
                dry_run_json["run_name"] = run_name
                dry_run_json["params"].update(bench_params)

                with open(f"dry_runs/dry_run_{run_name}.json", "w") as f:
                    json.dump(dry_run_json, f)

        else:
            exit_code, _entries = asyncio.run(run_benchmark(args))

    if mlflow_enabled:
        print("\nAll benchmark runs logged to MLflow.")
    else:
        print(f"\nAll benchmark runs saved to {SUMMARY_FILE}.")


def run_attached():
    """Benchmark an already running server with real coding prompts."""
    import requests
    from coding_dataset import build_fixed_length_prompts

    global MODEL_PARAMS, REQUEST_PARAMS
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    served = requests.get(BASE_URL + "/models", headers=headers, timeout=30).json()["data"][0]["id"]
    print(f"Server model: {served}")

    n_prompts = max(NUM_REQUESTS, max(CONCURRENCIES) * 5, 10)
    prompts = {t: build_fixed_length_prompts(t, n_prompts) for t in INPUT_TOKS}
    print(f"Built {n_prompts} real coding prompts for each input length {INPUT_TOKS}")

    containers = [c for c in os.getenv("BENCH_SERVER_CONTAINERS", "").split(",") if c]
    MODEL_PARAMS = {
        "server": os.getenv("BENCH_SERVER", "vllm"),
        "quantization": "unknown",
        "gpu::type": os.getenv("BENCH_GPU_TYPE") or get_gpu_type(),
        "gpu::count": int(os.getenv("BENCH_GPU_COUNT", "8")),
        "server_args::model": served,
    }
    MODEL_PARAMS.update(get_quantization(os.getenv("BENCH_QUANT_MODEL", served.rsplit("-", 1)[0] if served.split("-")[-1].isdigit() else served)))
    if containers:
        MODEL_PARAMS.update(get_running_server_metadata(containers))
    else:
        print("WARNING: BENCH_SERVER_CONTAINERS not set - server_args/version will be missing from MLflow")
    print("Params to log:", json.dumps({k: str(v)[:80] for k, v in MODEL_PARAMS.items()}, indent=1))
    REQUEST_PARAMS = {"request_rate": None, "stream": True, "chat": True, "temperature": 0.0}

    if do_warmup:
        for _ in range(3):
            requests.post(BASE_URL + "/chat/completions", headers=headers, timeout=300, json={
                "model": served, "max_tokens": 32, "temperature": 0.0,
                "messages": [{"role": "user", "content": "Write a Python hello world."}]})
    main(dataset_prompts=prompts)


def run_launch_server():
    """Original flow: launch vLLM per config permutation, synthetic prompts."""
    from utils import generate_config_permutations, launch_server_cmd, kill_process_tree, wait_for_server, get_server_args_from_config_for_mlflow, get_full_command_shell_from_config, ensure_model_downloaded
    global MODEL_PARAMS, REQUEST_PARAMS


    base_config_dir = "../configs/glm-4p6/h200/vllm"
    search_space_path = os.path.join(base_config_dir, "search_space.json")
    config_paths = generate_config_permutations(search_space_path, base_config_dir)
    print(f"Generated {len(config_paths)} configs")


    for config_path in config_paths:

        server_args = get_server_args_from_config_for_mlflow(config_path)
        full_command = get_full_command_shell_from_config(config_path)
        model_name = server_args["server_args::model"]
        process = None
        try:

            MODEL_PARAMS = {
                "server": "vllm",
                "version": "0.16.0",
                "quantization": "fp16", #fp16,fp8,mxfp4,nvfp4
                "gpu::type": "h200",
                "gpu::count": 2,
                **server_args,
            }

            REQUEST_PARAMS = {
                "request_rate": None,
                "stream": True,
                "chat": True,
                "temperature": 0.0,
            }

            dry_run_json = None
            if dry_run:
                os.makedirs("dry_runs", exist_ok=True)
                dry_run_json = {
                    "params": MODEL_PARAMS
                }
                dry_run_json["params"].update(REQUEST_PARAMS)

            else:
                ensure_model_downloaded(model_name)
                process = launch_server_cmd(full_command , PORT)
                wait_for_server(BASE_URL, process=process)
                print("Server is ready")

                if do_warmup:
                    warmup()

            main(dry_run_json)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if process:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    import sys
    if "--launch-server" in sys.argv:
        run_launch_server()
    else:
        run_attached()
