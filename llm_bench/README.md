# LLM Load Test

This directory contains two complementary benchmark tools and a plotting utility for evaluating LLM inference endpoints.

| Tool | Best for | Stopping condition |
|------|----------|--------------------|
| `load_test.py` (Locust) | Sustained-load / QPS testing | Time-based (`-t`) and/or request-count (`--max-requests`) |
| `bench_serving.py` (async) | Exact request-count runs with controlled concurrency | Always request-count (`--num-requests`) |
| `plot_bench.py` | Visualising results from either tool | N/A |

## Installation

```bash
pip install -r requirements.txt
# bench_serving.py and plot_bench.py additionally need:
pip install aiohttp matplotlib pandas
```

---

## Tool 1 — `load_test.py` (Locust-based)

The original load test. It relies on [Locust](https://locust.io/) and runs from `locust.conf` automatically.

### Target

- `-H`: target endpoint URL (preceding `/v1/...`). E.g. `-H http://localhost` or `-H https://api.fireworks.ai/inference`. Defaults to `localhost:80`.
- (optional) `-m`: model to send requests to. Can be omitted for a local test if the server has a single model loaded only.
- (optional) `--provider`: provider name like `fireworks` or `openai`. APIs have slight differences that the script accounts for. If omitted the script tries to guess based on URI and API return information. Must be specified for non-OpenAI-compatible providers like Triton.
- `-k`: API key to be passed as `Authorization: Bearer ...`.

### Rate of requests

There are several primary modes the script can be used:

1. **Fixed concurrency**. N workers are created. Each sends a request, waits for the response and then sends the next request. Thus as concurrency increases, the server will get more loaded and latency will grow. Usually increasing concurrency beyond some point doesn't increase throughput and just leads to growing latency.
   - `-u`: the number of concurrent workers to spawn (standard Locust argument)
   - `-r`: the rate per second of spawning concurrent workers. If processing workload takes a while (more than several seconds), it makes sense to set this value to something lower than `-u` for a gradual ramp-up to avoid request bursts.
   - (optionally) `--burst <period in seconds>`: synchronizes all N workers to issue requests in one go with the specified interval. The maximum latency should be less than the period, otherwise some workers may fall behind.

2. **Fixed QPS**. The script ensures that input requests are issued at specific times to average out at the specified rate per second. If the target QPS is too high and the server is overloaded it will likely drop additional requests or stall.
   - `--qps`: the desired rate of requests per second. Can be a fractional number, e.g. `0.1`.
   - `-u <high number> -r <high number>`: needs to be set to a sufficiently high value to allow generating the target QPS. The script will complain if it's too low. Passing something like `-u 100 -r 100` is a good choice.
   - (optional) `--qps-distribution`: specify how to space out requests. Default is `constant` meaning evenly spaced out. `exponential` is an option simulating [Poisson distribution](https://en.wikipedia.org/wiki/Traffic_generation_model#Poisson_traffic_model).

### Stopping condition

- `-t`: duration (e.g. `5min`) for which to run the test (standard Locust option).
- `--max-requests`: stop the test after exactly this many total requests complete across all users, regardless of elapsed time. This is useful when individual requests may take longer than the time limit (e.g. long generations, cold starts). Can be used alone or combined with `-t` — whichever fires first wins.

```bash
# Request-count only (no time limit — runs until 20 requests finish)
locust --max-requests 20 -u 4 -r 4 -H http://localhost:8000/v1 --chat --stream -p 512 -o 128

# Both (stop after 10 requests OR 2 minutes, whichever comes first)
locust --max-requests 10 -t 2min -u 4 -r 4 -H http://localhost:8000/v1 --chat --stream -p 512 -o 128
```

### Workload

The tool currently supports only a fixed prompt specified as one of:
- `-p`: prompt length in tokens. The script will generate some prompt of this length.
- `--prompt-text`: use the specified text as a prompt instead of generating one. It can be a file reference starting with an ampersand, e.g. `@prompt.txt`.

The number of tokens to generate is sampled on every request from a given distribution:
- `-o`/`--max-tokens`: maximum number of tokens to generate. If --max-tokens-distribution is non-constant this is going to be the mean of the distribution.
- `--max-tokens-distribution`: specifies probability distribution to use.
- `--max-tokens-range`: specifies "the width" of the distribution (e.g. stddev for "normal" distribution). Specified value `alpha` is relative to `max-tokens`. Default is 0.3 so most of the range falls in "3 sigma" region.
- `--max-tokens-cap`: specify upper bound to "truncate" the probability distribution. The lower bound is always 1 token. This allows to sample from "truncated normal" or "truncated exponential" distributions.

Based on the above settings the following distributions are supported:
- `constant`: use `--max_tokens` value on every request
- `uniform`: sample from the range `[max_tokens - max_tokens * alpha, max_tokens + max_tokens * alpha]`
- `normal`: sample from gaussian distribution `N(max_tokens, max_tokens * alpha)`
- `exponential`: sample from exponential distribution with the mean `max_tokens`. `alpha` is ignored

The benchmark makes the best effort to ensure the desired `max_tokens` number is respected:
- for providers that support it, it passes `ignore_eos` or `min_tokens` parameter to avoid early stopping
- the default prompt is a lengthy code generation request that usually doesn't stop early
- it verifies the number of tokens actually generated and prints warnings on mismatch. Different providers use varying mechanisms of returning generated number of tokens. For some of them `--logprobs` might be needed in the streaming mode.
- optionally, `--tokenizer` can be passed specifying Huggingface tokenizer to be used to count the output tokens on client side.

Generation options:
- `--chat`: specify to call chat API instead of raw completions
- `--stream`: stream the result back. Enabling this gives "time to first token" and "time per token" metrics
- (optional) `--logprobs`: corresponds to `logprobs` API parameter. For some providers, it's needed for output token counting in streaming mode.

### Writing results

- `--summary-file`: Append the line with the summary to the specified CSV file. Useful for generating a spreadsheet with perf sweep results. If the file doesn't exist, it writes out the header first.

### Custom prompts

Sometimes it's necessary to replay exact prompts, for example in the case of embedding images.
`--prompt-text` option can be used in this case to specify a file with .jsonl extension (starting with an ampersand, e.g. `@prompt.jsonl`.).
jsonl files will be read line-by-line and will be randomly chosen for each request. Each line has to have a valid JSON object with 'prompt' and optional 'images' keys. For example:
```
{"prompt": "<image>What color is the cat?", images: ["data:image/jpeg;base64,BASE_64_DATA]}
{"prompt": "<image>What color is the dog?", images: ["data:image/jpeg;base64,BASE_64_DATA]}
```

### Locust examples

Maintain fixed 8 requests concurrency against local deployment:

```bash
locust -u 8 -r 2 -p 512 -o 128
```

Call streaming chat API locally with the request issued every 2 seconds. Run for 1 minute and save results to `results.csv`:

```bash
locust -t 1min -u 100 -r 100 -p 512 -o 128 --stream --chat --qps 0.5 --summary-file results.csv
```

Benchmark Fireworks public deployment with 1 request only:

```bash
locust -u 1 -H https://api.fireworks.ai/inference -p 128 -o 200 --api-key $FIREWORKS_API_KEY --model=accounts/fireworks/models/llama-v3-8b
```

Benchmark OpenAI deployment with 1 request only:

```bash
locust -u 1 -H https://api.openai.com -p 128 -o 200 --api-key $OPENAI_API_KEY --model=gpt-3.5-turbo --chat
```

Benchmark local Triton deployment with a given prompt at 1 QPS:

```bash
locust -u 100 -r 100 --prompt-text "$PROMPT" -o 100 --provider triton-infer -H http://localhost:8000 --tokenizer /path/to/my/hf/tokenizer --qps 1
```

### Batch sweep (`run_load_test.py`)

`run_load_test.py` iterates over concurrency levels, input/output token sizes and cache prefix ratios, calling `locust` for each combination and appending results to a single CSV.

```bash
python run_load_test.py
```

Edit the variables at the top of the file to customise the sweep.

---

## Tool 2 — `bench_serving.py` (async, request-count based)

A standalone asyncio benchmark that runs exactly N requests with a semaphore-based concurrency limit. There is **no time limit** — every request runs to completion no matter how long it takes. This is ideal when:

- Individual requests may exceed any reasonable time window (long generations, cold starts).
- You want deterministic "run exactly N requests" semantics.
- You need precise control over in-flight concurrency without Locust's worker model.

### Key arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--base-url` | API base URL (e.g. `http://localhost:8000/v1`) | *required* |
| `--num-requests` | Total number of requests to send | *required* |
| `--concurrency` | Max in-flight requests (semaphore) | 1 |
| `--request-rate` | Rate-limit request launches (req/s) | unlimited |
| `-m` / `--model` | Model name (auto-detected if omitted) | auto |
| `--chat` | Use `/v1/chat/completions` | off |
| `--stream` | Stream responses (enables TTFT metrics) | off |
| `-p` / `--prompt-tokens` | Approximate prompt length in tokens | 512 |
| `-o` / `--max-tokens` | Max output tokens per request | 64 |
| `--max-tokens-range` | Randomise max-tokens by +/- this fraction | 0 |
| `--prompt-randomize` | Randomise part of the prompt to defeat caching | off |
| `--prompt-cache-max-len` | Fixed prefix length for cache simulation | 0 |
| `-k` / `--api-key` | Bearer token for auth | none |
| `--header` | Extra headers as `key:value` (repeatable) | none |
| `--timeout` | Per-request timeout in seconds | 600 |
| `--summary-file` | Append summary row to this CSV | none |
| `--show-response` | Print each response body | off |

### Examples

```bash
# 10 requests, 2 at a time, streaming chat
python bench_serving.py --base-url http://localhost:8000/v1 \
    --num-requests 10 --concurrency 2 --chat --stream \
    --model meta-llama/Llama-3.2-1B-Instruct -p 1000 -o 64

# With rate limiting (5 req/s launch rate, max 4 concurrent)
python bench_serving.py --base-url http://localhost:8000/v1 \
    --num-requests 50 --concurrency 4 --request-rate 5 --chat \
    --model my-model --summary-file results.csv

# Single cold-start request (no time pressure)
python bench_serving.py --base-url http://localhost:8000/v1 \
    --num-requests 1 --concurrency 1 --chat -o 512 --timeout 1200
```

### Batch sweep (`run_bench_serving.py`)

`run_bench_serving.py` mirrors `run_load_test.py` but uses `bench_serving.py` instead of Locust. It sweeps concurrency, input/output tokens and cache prefix ratios, appending results to a single CSV.

```bash
python run_bench_serving.py
```

### Output

A summary with percentiles (P50/P90/P95/P99) is printed at the end of every run. Metrics include:

- End-to-end latency
- Time to first token (TTFT, streaming only)
- Time per output token (TPOT)
- Throughput (req/s and output tok/s)

When `--summary-file` is specified, each run appends a row to the CSV with all metrics.

---

## Plotting — `plot_bench.py`

Reads CSV files produced by **either** `bench_serving.py` or the Locust `load_test.py` and generates performance plots. Multiple CSV files can be overlaid for A/B comparison.

### Generated plots

| File | Description |
|------|-------------|
| `avg_latency_vs_concurrency.png` | Mean E2E latency vs concurrency (shaded P50–P90 band) |
| `90_latency_vs_concurrency.png` | P90 latency vs concurrency |
| `throughput_vs_concurrency.png` | Request throughput (req/s) vs concurrency |
| `token_throughput_vs_concurrency.png` | Output tokens/s vs concurrency |
| `ttft_vs_concurrency.png` | Time to first token vs concurrency (streaming only) |
| `tpot_vs_concurrency.png` | Time per output token vs concurrency |
| `latency_vs_throughput.png` | Pareto curve (concurrency labels on each point) |
| `latency_percentiles.png` | P50/P90/P95/P99 bar chart at max concurrency |

### Examples

```bash
# Plot a single results file
python plot_bench.py --csv bench-serving-results.csv

# A/B comparison of two runs
python plot_bench.py --csv run-a.csv --csv run-b.csv --labels "vLLM" "SGLang"

# Filter to a specific output-token count, custom output directory
python plot_bench.py --csv results.csv --filter-col generation_tokens --filter-val 64 -o ./my_plots

# Compare locust and bench_serving results side by side
python plot_bench.py --csv test-vllm-v4.csv --csv bench-serving-results.csv \
    --labels "Locust" "Async" -o comparison_plots
```

Column names are normalised internally, so CSVs from both tools (and even mixed sources) work together seamlessly.

---

## Locust UI mode

Instead of relying on textual data, it's also possible to plot the Locust results in Grafana.

```bash
pip install locust locust-plugins
locust-compose up
```

This starts your local Postgre and Grafana. Grafana is available at http://127.0.0.1:3000 (sometimes logs don't print out).

Then run the test as specified above with an additional argument:

```bash
locust --config locust-grafana.conf ...
```

This starts the load test locally and pushes results into Grafana in real-time. Besides the actual requests, we push additional metrics (e.g. time per token) as separate fake requests to get stats aggregation. Make sure to remove them from aggregation when viewing the graphs.

Other settings for Locust are in `./locust.conf`. You may start Locust in non-headless mode, but its UI is very basic and misses advanced stats aggregation capabilities.

---

## File overview

| File | Purpose |
|------|---------|
| `load_test.py` | Locust-based load test (time and/or request-count bounded) |
| `bench_serving.py` | Async request-count benchmark with semaphore concurrency |
| `plot_bench.py` | Plot CSV results from either tool |
| `run_load_test.py` | Batch sweep runner for `load_test.py` |
| `run_bench_serving.py` | Batch sweep runner for `bench_serving.py` |
| `locust.conf` | Default Locust settings |
| `locust-grafana.conf` | Locust settings for Grafana mode |
| `requirements.txt` | Python dependencies |
