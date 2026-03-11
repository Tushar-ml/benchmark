#!/usr/bin/env python3
"""
Request-count based LLM benchmark with controlled concurrency.

Unlike the locust-based load_test.py which is time-bound, this script runs
exactly N requests with a configurable concurrency limit (asyncio.Semaphore)
and waits for all of them to finish — no matter how long each one takes.

Usage examples:
    # 10 requests, 2 at a time
    python bench_serving.py --base-url http://localhost:8000/v1 --num-requests 10 --concurrency 2

    # With streaming, chat mode, custom model
    python bench_serving.py --base-url http://localhost:8000/v1 \
        --num-requests 50 --concurrency 8 --stream --chat \
        --model meta-llama/Llama-3.2-1B-Instruct --prompt-tokens 512 --max-tokens 64

    # Save results to CSV
    python bench_serving.py --base-url http://localhost:8000/v1 \
        --num-requests 20 --concurrency 4 --summary-file results.csv
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import tiktoken

PROMPTS = [
    "Describe how reinforcement learning differs from supervised learning, and provide real-world applications of both.",
    "Explain the process of building a machine learning model from data collection to model deployment. Include the tools and techniques commonly used at each stage.",
    "Compare and contrast the major cloud service providers (AWS, GCP, and Azure). Highlight their strengths and weaknesses in terms of scalability, pricing, and service offerings.",
    "Discuss the current cybersecurity landscape and the most prevalent types of attacks. What are the best practices organizations can implement to protect their networks?",
    "How does quantum computing differ from classical computing? Explain how quantum bits (qubits) function and describe a potential use case for quantum supremacy.",
    "Examine the role of smart contracts in blockchain technology. How do they work, and what are the most common challenges in implementing them?",
    "Describe the concept of black holes in astrophysics. What theories exist to explain their formation, and what mysteries remain unsolved?",
    "Outline the periodic table's organization and how it reflects the electronic structure of elements. Explain trends such as electronegativity and atomic radius.",
    "How does CRISPR-Cas9 technology work, and what are its ethical implications in genetic engineering?",
    "Analyze the long-term impact of the Industrial Revolution on society, politics, and economics in Europe.",
    "Compare the philosophies of existentialism and utilitarianism. How do they approach the question of what constitutes a 'good life'?",
    "Discuss the psychological concept of cognitive dissonance and how it influences human behavior and decision-making.",
    "How do social media platforms impact modern social interactions, relationships, and individual self-esteem?",
    "Explain the concept of supply and demand in a market economy. How do government policies, such as subsidies or tariffs, affect these forces?",
    "Evaluate the advantages and disadvantages of a federal system of government compared to a unitary system, using specific countries as examples.",
    "Analyze the themes of isolation and identity in Mary Shelley's Frankenstein. How do these themes reflect the broader social concerns of the time?",
    "Discuss the major environmental challenges posed by climate change. What are the most viable solutions to mitigate its impact?",
    "What are the latest advancements in personalized medicine? Discuss how genomic data is used to tailor treatments to individual patients.",
    "Compare the structural properties of steel and carbon fiber. In which applications would you prefer one over the other, and why?",
    "Explain the concept of non-Euclidean geometry and its implications in fields like physics and computer science. How does it differ from traditional Euclidean geometry?",
]

PROMPT_PREFIX_TOKEN = "Pad "

tokenizer = tiktoken.encoding_for_model("gpt-4")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: int
    success: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    output_text: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Prompt generation (mirrors load_test.py logic)
# ---------------------------------------------------------------------------

def build_prompt(args) -> str:
    suffix = random.choice(PROMPTS)
    suffix_tokens = len(tokenizer.encode(suffix))

    if args.prompt_text:
        if args.prompt_text.startswith("@"):
            with open(args.prompt_text[1:]) as f:
                return f.read()
        return args.prompt_text

    if args.prompt_chars:
        raw = PROMPT_PREFIX_TOKEN * (args.prompt_chars // len(PROMPT_PREFIX_TOKEN) + 1) + suffix
        return raw[: args.prompt_chars]

    pad_count = max(0, args.prompt_tokens - suffix_tokens)
    if args.prompt_randomize:
        cache_tokens = args.prompt_cache_max_len
        random_tokens = pad_count - cache_tokens
        return (
            PROMPT_PREFIX_TOKEN * cache_tokens
            + " ".join(chr(ord("a") + random.randint(0, 25)) for _ in range(random_tokens))
            + " " + suffix
        )
    return PROMPT_PREFIX_TOKEN * pad_count + suffix


# ---------------------------------------------------------------------------
# Payload formatting
# ---------------------------------------------------------------------------

def format_openai_payload(args, prompt: str, max_tokens: int) -> dict:
    data = {
        "model": args.model,
        "max_tokens": max_tokens,
        "stream": args.stream,
        "temperature": args.temperature,
        "ignore_eos": True,
    }
    if args.chat:
        data["messages"] = [{"role": "user", "content": prompt}]
    else:
        data["prompt"] = prompt
    return data


def get_endpoint(args) -> str:
    if args.chat:
        return "/chat/completions"
    return "/completions"


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def parse_chunk(args, data: dict) -> tuple[str, Optional[int], Optional[int]]:
    """Returns (text, completion_tokens_from_usage, prompt_tokens_from_usage)."""
    usage = data.get("usage")
    choice = data["choices"][0]
    if args.chat:
        if args.stream:
            text = choice.get("delta", {}).get("content", "") or choice.get("delta", {}).get("reasoning_content", "")
        else:
            text = choice.get("message", {}).get("content", "") or choice.get("message", {}).get("reasoning_content", "")
    else:
        text = choice.get("text", "")
    comp_tokens = usage["completion_tokens"] if usage else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    return text, comp_tokens, prompt_tokens


# ---------------------------------------------------------------------------
# Single request coroutine
# ---------------------------------------------------------------------------

async def send_request(
    request_id: int,
    args,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    prompt = build_prompt(args)
    prompt_tok_count = len(tokenizer.encode(prompt))

    max_tokens = args.max_tokens
    if args.max_tokens_range > 0:
        lo = max(1, int(max_tokens * (1 - args.max_tokens_range)))
        hi = int(max_tokens * (1 + args.max_tokens_range))
        max_tokens = random.randint(lo, hi)

    payload = format_openai_payload(args, prompt, max_tokens)
    url = args.base_url.rstrip("/") + get_endpoint(args)

    async with semaphore:
        t_start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return RequestResult(
                        request_id=request_id, success=False,
                        error=f"HTTP {resp.status}: {body[:500]}",
                        total_latency_ms=(time.perf_counter() - t_start) * 1000,
                    )

                combined_text = ""
                t_first_token = None
                usage_comp_tokens = None
                usage_prompt_tokens = None

                if args.stream:
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        line = line[len("data:"):].strip()
                        if line == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text, ct, pt = parse_chunk(args, chunk_data)
                        if ct is not None:
                            usage_comp_tokens = (usage_comp_tokens or 0) + ct
                        if pt is not None:
                            usage_prompt_tokens = pt
                        if text:
                            combined_text += text
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                else:
                    body = await resp.json()
                    text, ct, pt = parse_chunk(args, body)
                    combined_text = text or ""
                    usage_comp_tokens = ct
                    usage_prompt_tokens = pt
                    t_first_token = time.perf_counter()

                t_end = time.perf_counter()
                if t_first_token is None:
                    t_first_token = t_end

                output_tok_count = usage_comp_tokens or len(tokenizer.encode(combined_text, allowed_special="all"))

                return RequestResult(
                    request_id=request_id,
                    success=True,
                    prompt_tokens=usage_prompt_tokens or prompt_tok_count,
                    output_tokens=output_tok_count,
                    ttft_ms=(t_first_token - t_start) * 1000,
                    total_latency_ms=(t_end - t_start) * 1000,
                    generation_latency_ms=(t_end - t_first_token) * 1000,
                    output_text=combined_text,
                )
        except Exception as e:
            return RequestResult(
                request_id=request_id, success=False,
                error=repr(e),
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
            )


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

async def progress_printer(results: list[RequestResult], total: int, stop_event: asyncio.Event):
    while not stop_event.is_set():
        done = len(results)
        ok = sum(1 for r in results if r.success)
        fail = done - ok
        print(f"\r  Progress: {done}/{total} completed ({ok} ok, {fail} failed)", end="", flush=True)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    print(f"\r  Progress: {len(results)}/{total} completed" + " " * 20)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    if args.num_requests <= 0:
        args.num_requests = max(args.concurrency * 5, 10)

    semaphore = asyncio.Semaphore(args.concurrency)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    if args.header:
        for h in args.header:
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    # Auto-detect model if not specified
    if not args.model:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(args.base_url.rstrip("/") + "/models") as resp:
                data = await resp.json()
                args.model = data["data"][0]["id"]
                print(f"Auto-detected model: {args.model}")

    print(f"{'=' * 70}")
    print(f"  Benchmark: {args.num_requests} requests, concurrency {args.concurrency}")
    print(f"  Model:     {args.model}")
    print(f"  Endpoint:  {args.base_url}{get_endpoint(args)}")
    print(f"  Prompt:    ~{args.prompt_tokens} tokens | Max output: {args.max_tokens} tokens")
    print(f"  Stream:    {args.stream} | Chat: {args.chat}")
    print(f"{'=' * 70}")

    connector = aiohttp.TCPConnector(limit=args.concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    results: list[RequestResult] = []
    stop_event = asyncio.Event()

    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        progress_task = asyncio.create_task(progress_printer(results, args.num_requests, stop_event))

        bench_start = time.perf_counter()
        tasks = []
        for i in range(args.num_requests):
            task = asyncio.create_task(send_request(i, args, session, semaphore))
            tasks.append(task)
            if args.request_rate is not None and 0 < args.request_rate < float("inf"):
                await asyncio.sleep(1.0 / args.request_rate)

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if args.show_response and result.success:
                print(f"\n--- Request {result.request_id} ---")
                print(result.output_text[:500])
            elif not result.success:
                print(f"\n  [FAIL] Request {result.request_id}: {result.error[:200]}")

        bench_end = time.perf_counter()
        stop_event.set()
        await progress_task

    wall_time = bench_end - bench_start

    # Compute stats
    ok_results = [r for r in results if r.success]
    fail_count = len(results) - len(ok_results)

    if not ok_results:
        print("\nAll requests failed!")
        return 1, {}

    latencies = sorted(r.total_latency_ms for r in ok_results)
    ttfts = sorted(r.ttft_ms for r in ok_results) if args.stream else []
    output_toks = [r.output_tokens for r in ok_results]
    prompt_toks = [r.prompt_tokens for r in ok_results]
    throughputs = [
        r.output_tokens / (r.total_latency_ms / 1000) if r.total_latency_ms > 0 else 0
        for r in ok_results
    ]
    tpots = [
        r.generation_latency_ms / r.output_tokens if r.output_tokens > 0 else 0
        for r in ok_results
    ]

    total_output_tokens = sum(output_toks)
    total_prompt_tokens = sum(prompt_toks)

    entries = {
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "successful": len(ok_results),
        "failed": fail_count,
        "wall_time_s": f"{wall_time:.2f}",
        "prompt_tokens_avg": f"{sum(prompt_toks) / len(prompt_toks):.0f}",
        "output_tokens_avg": f"{sum(output_toks) / len(output_toks):.0f}",
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "throughput_req_per_s": f"{len(ok_results) / wall_time:.2f}",
        "throughput_output_tok_per_s": f"{total_output_tokens / wall_time:.2f}",
        "throughput_total_tok_per_s": f"{(total_output_tokens + total_prompt_tokens) / wall_time:.2f}",
        "avg_latency_ms": f"{sum(latencies) / len(latencies):.2f}",
        "median_latency_ms": f"{percentile(latencies, 50):.2f}",
        "p90_latency_ms": f"{percentile(latencies, 90):.2f}",
        "p95_latency_ms": f"{percentile(latencies, 95):.2f}",
        "p99_latency_ms": f"{percentile(latencies, 99):.2f}",
        "min_latency_ms": f"{latencies[0]:.2f}",
        "max_latency_ms": f"{latencies[-1]:.2f}",
    }

    if args.stream and ttfts:
        entries.update({
            "avg_ttft_ms": f"{sum(ttfts) / len(ttfts):.2f}",
            "median_ttft_ms": f"{percentile(ttfts, 50):.2f}",
            "p90_ttft_ms": f"{percentile(ttfts, 90):.2f}",
            "p95_ttft_ms": f"{percentile(ttfts, 95):.2f}",
            "p99_ttft_ms": f"{percentile(ttfts, 99):.2f}",
        })

    sorted_tpots = sorted(tpots)
    entries.update({
        "avg_tpot_ms": f"{sum(tpots) / len(tpots):.2f}",
        "median_tpot_ms": f"{percentile(sorted_tpots, 50):.2f}",
        "p90_tpot_ms": f"{percentile(sorted_tpots, 90):.2f}",
        "p95_tpot_ms": f"{percentile(sorted_tpots, 95):.2f}",
        "p99_tpot_ms": f"{percentile(sorted_tpots, 99):.2f}",
    })

    sorted_throughputs = sorted(throughputs)
    entries.update({
        "avg_per_req_tok_per_s": f"{sum(throughputs) / len(throughputs):.2f}",
        "median_per_req_tok_per_s": f"{percentile(sorted_throughputs, 50):.2f}",
    })

    max_key = max(len(k) for k in entries)
    print(f"\n{'=' * 70}")
    print(f"{'Summary':^70}")
    print(f"{'=' * 70}")
    for k, v in entries.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<{max_key + 5}}: {v}")
    print(f"{'=' * 70}")

    if args.summary_file:
        file_exists = os.path.exists(args.summary_file)
        with open(args.summary_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entries.keys())
            if not file_exists or os.path.getsize(args.summary_file) == 0:
                writer.writeheader()
            writer.writerow(entries)
        print(f"Results appended to {args.summary_file}")

    exit_code = 1 if fail_count > 0 else 0
    return exit_code, entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Request-count based LLM benchmark with controlled concurrency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-url", required=True,
        help="Base URL of the API (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--num-requests", type=int, default=0,
        help="Total number of requests to send. "
             "If 0 or omitted, defaults to max(concurrency * 5, 10)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Maximum number of concurrent in-flight requests (default: 1)",
    )
    parser.add_argument(
        "--request-rate", type=float, default=None,
        help="Rate limit for launching requests (requests/sec). "
             "If not set, requests are launched as fast as the semaphore allows",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help="Model name. Auto-detected from /v1/models if not specified",
    )
    parser.add_argument(
        "--chat", action="store_true", default=False,
        help="Use /v1/chat/completions instead of /v1/completions",
    )
    parser.add_argument(
        "--stream", action="store_true", default=False,
        help="Use streaming responses",
    )
    parser.add_argument(
        "-p", "--prompt-tokens", type=int, default=512,
        help="Approximate prompt length in tokens (default: 512)",
    )
    parser.add_argument(
        "--prompt-chars", type=int, default=None,
        help="Prompt length in characters (overrides --prompt-tokens)",
    )
    parser.add_argument(
        "--prompt-text", type=str, default=None,
        help="Literal prompt text or @filename to load from file",
    )
    parser.add_argument(
        "--prompt-randomize", action="store_true", default=False,
        help="Randomize part of the prompt to defeat caching",
    )
    parser.add_argument(
        "--prompt-cache-max-len", type=int, default=0,
        help="Fixed prefix length for cache simulation (default: 0)",
    )
    parser.add_argument(
        "-o", "--max-tokens", type=int, default=64,
        help="Max output tokens per request (default: 64)",
    )
    parser.add_argument(
        "--max-tokens-range", type=float, default=0.0,
        help="Randomize max-tokens by +/- this fraction (default: 0, no randomization)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "-k", "--api-key", type=str, default=None,
        help="Bearer token for API auth",
    )
    parser.add_argument(
        "--header", action="append", default=[],
        help="Extra headers as key:value (repeatable)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600,
        help="Per-request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--summary-file", type=str, default=None,
        help="Append summary row to this CSV file",
    )
    parser.add_argument(
        "--show-response", action="store_true", default=False,
        help="Print each response body",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    exit_code, _entries = asyncio.run(run_benchmark(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()