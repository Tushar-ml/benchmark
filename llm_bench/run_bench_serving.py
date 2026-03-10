#!/usr/bin/env python3
"""
Benchmark runner with optional MLflow logging.

When MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME are set, results and model
params are logged to MLflow.  Otherwise, results are saved to a local CSV via
bench_serving.py's --summary-file flag (original behaviour).

Optional env vars for MLflow mode:
    MLFLOW_TRACKING_URI     – MLflow server URL
    MLFLOW_EXPERIMENT_NAME  – Experiment name
    MLFLOW_RUN_NAME         – Custom run name (auto-generated if omitted)
"""

import asyncio
import os
import subprocess
from argparse import Namespace
from itertools import product
import time
from utils import generate_config_permutations, run_command, kill_process, wait_for_health, get_server_args_from_config_for_mlflow, get_full_command_shell_from_config, ensure_model_downloaded

from bench_serving import run_benchmark

# TODO: server launch command and readiness check
# ── Benchmark matrix ─────────────────────────────────────────────────────

MODEL_NAME = "test"
CONCURRENCIES = [2,4,8,16,32,64,128,256,512]
INPUT_TOKS = [50, 100, 256, 512,1024,2048,4096,8192,10000]
OUTPUT_TOKS = [500]
PCMLS = [0, 0.5, 0.8, 0.95]
NUM_REQUESTS = 0
SUMMARY_FILE = f"{MODEL_NAME.split('/')[-1]}.csv"
experiment_name = MODEL_NAME
warmup = False


MODEL_PARAMS = {}
REQUEST_PARAMS = {}

BASE_URL = "http://localhost:8000/v1"
API_KEY = os.getenv("BENCH_API_KEY", "")
EXTRA_HEADERS = ["id:f49b2e20-fef3-4441-9358-897f946b8ae2"]

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
    ) and not warmup

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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    mlflow_enabled = use_mlflow()

    if mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(experiment_name)
        print("MLflow logging enabled")
    else:
        mlflow = None
        print(f"MLflow not configured – results will be saved to {SUMMARY_FILE}")

    for concurrency, input_tok, output_tok, pcml_frac in product(
        CONCURRENCIES, INPUT_TOKS, OUTPUT_TOKS, PCMLS,
    ):
        pcml = int(pcml_frac * input_tok)

        args = Namespace(
            base_url=BASE_URL,
            num_requests=NUM_REQUESTS,
            concurrency=concurrency,
            request_rate=REQUEST_PARAMS["request_rate"],
            model=MODEL_NAME,
            chat=REQUEST_PARAMS["chat"],
            stream=REQUEST_PARAMS["stream"],
            prompt_tokens=input_tok,
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
                "prompt_cache_max_len": pcml,
                "prompt_randomize": True
            }
            gpu_info = get_device_info()
            bench_params.update(gpu_info)

            run_name = os.environ.get(
                "MLFLOW_RUN_NAME",
                f"{MODEL_NAME.split('/')[-1]}_c{concurrency}_i{input_tok}_o{output_tok}_{time.strftime('%Y-%m-%d-%H-%M-%S')}",
            )

            with mlflow.start_run(run_name=run_name):
                exit_code, entries = asyncio.run(run_benchmark(args))

                if entries:
                    log_to_mlflow(mlflow, entries, bench_params)
                    print(f"Logged run '{run_name}' to MLflow ({len(entries)} fields)")
                else:
                    mlflow.set_tag("status", "all_requests_failed")
                    print(f"Run '{run_name}' failed – tagged in MLflow")
        else:
            exit_code, _entries = asyncio.run(run_benchmark(args))

    if mlflow_enabled:
        print("\nAll benchmark runs logged to MLflow.")
    else:
        print(f"\nAll benchmark runs saved to {SUMMARY_FILE}.")


if __name__ == "__main__":

    import os
    os.environ["MLFLOW_TRACKING_URI"] = "http://admin:********@a8e6c4207413949b898c70462c6f63c6-705429131.us-west-2.elb.amazonaws.com:5000/"

    base_config_dir = "/home/ubuntu/benchmark/configs/glm-4p6/h200/sglang"
    search_space_path = os.path.join(base_config_dir, "search_space.json")
    config_paths = generate_config_permutations(search_space_path, base_config_dir)
    print(f"Generated {len(config_paths)} configs")


    for config_path in config_paths:

        server_args = get_server_args_from_config_for_mlflow(config_path)
        full_command = get_full_command_shell_from_config(config_path)
        model_name = server_args["server_args::model"]
        process = None
        try:
            ensure_model_downloaded(model_name)
            process = run_command(full_command)
            if not wait_for_health():
                print("Health endpoint not ready after 20 seconds, killing process")
                kill_process(process)
                raise RuntimeError("Health endpoint not ready after 20 seconds")

            MODEL_PARAMS = {
                "server": "sglang",
                "version": "0.5.9",
                **server_args,
            }

            REQUEST_PARAMS = {
                "request_rate": None,
                "stream": True,
                "chat": True,
                "temperature": 0.0,
            }


            main()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if process:
                kill_process(process)
