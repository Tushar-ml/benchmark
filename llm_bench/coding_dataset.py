"""Real coding prompts for bench_serving.py.

Default: SWE-bench Lite (princeton-nlp/SWE-bench_Lite, 300 real GitHub issues
from 12 Python repos, public, no auth). Each prompt asks the model to resolve
one issue, built from the issue text (+ maintainer hints when present).

Also accepts a local JSONL file ({"prompt": "..."} per line) via
CODING_DATASET=jsonl:/path/to/file.jsonl
"""

import json
import os
import random
import uuid

SWE_BENCH_LITE = "princeton-nlp/SWE-bench_Lite"

TEMPLATE = (
    "You are an expert software engineer. Resolve the following GitHub issue "
    "in the repository `{repo}`.\n\n"
    "## Issue\n{problem}\n"
    "{hints}"
    "\nExplain the root cause briefly, then give the fix as a unified diff."
)


def _swe_bench_prompts(name: str = SWE_BENCH_LITE, split: str = "test") -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    prompts = []
    for row in ds:
        hints = (row.get("hints_text") or "").strip()
        prompts.append(
            TEMPLATE.format(
                repo=row["repo"],
                problem=row["problem_statement"].strip(),
                hints=f"\n## Hints from the discussion\n{hints}\n" if hints else "",
            )
        )
    return prompts


def _jsonl_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [json.loads(line)["prompt"] for line in f if line.strip()]


def load_prompts(spec: str = SWE_BENCH_LITE, seed: int = 42, max_chars: int = 0) -> list[str]:
    """spec: an HF dataset id (SWE-bench format) or 'jsonl:/path'. Shuffled with a fixed seed."""
    prompts = _jsonl_prompts(spec[len("jsonl:"):]) if spec.startswith("jsonl:") else _swe_bench_prompts(spec)
    if max_chars:
        prompts = [p for p in prompts if len(p) <= max_chars]
    random.Random(seed).shuffle(prompts)
    if not prompts:
        raise RuntimeError(f"No prompts loaded from {spec!r}")
    return prompts


# ── Fixed-length prompts: real issue + real source code as filler ────────────────────────────
# Used to get the same 100 / 1000 / 10000 input-token points as the synthetic benchmark, but with
# real coding content: a SWE-bench issue first, then real Python files (codeparrot-clean-valid)
# until the token budget is reached. A unique request id leads every prompt, so no two prompts share
# a prefix and server-side prefix caching cannot inflate results (same intent as prompt_randomize).

CODE_CORPUS = "codeparrot/codeparrot-clean-valid"
CODE_CACHE = os.path.expanduser("~/.cache/llm_bench/code_corpus.jsonl")
_enc = None


def _tok():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.encoding_for_model("gpt-4")  # same tokenizer bench_serving.py counts with
    return _enc


def load_code_files(n: int = 400, min_chars: int = 2000, max_chars: int = 40000) -> list[dict]:
    """Real Python source files ({path, content}); cached locally after the first download."""
    if os.path.exists(CODE_CACHE):
        with open(CODE_CACHE) as f:
            return [json.loads(l) for l in f if l.strip()]
    from datasets import load_dataset

    ds = load_dataset(CODE_CORPUS, split="train", streaming=True).shuffle(seed=42, buffer_size=2000)
    files = []
    for row in ds:
        if min_chars <= len(row["content"]) <= max_chars:
            files.append({"path": f"{row['repo_name']}/{row['path']}", "content": row["content"]})
        if len(files) >= n:
            break
    os.makedirs(os.path.dirname(CODE_CACHE), exist_ok=True)
    with open(CODE_CACHE, "w") as f:
        f.write("\n".join(json.dumps(x) for x in files))
    return files


def _issues(seed: int):
    from datasets import load_dataset

    ds = load_dataset(SWE_BENCH_LITE, split="test")
    rows = [(r["repo"], r["problem_statement"].strip()) for r in ds]
    random.Random(seed).shuffle(rows)
    return rows


def build_fixed_length_prompts(target_tokens: int, n: int, seed: int = 42) -> list[str]:
    """n prompts of ~target_tokens (tiktoken gpt-4 count), each: real issue + real source files."""
    enc = _tok()
    rng = random.Random(seed + target_tokens)
    issues, files = _issues(seed + target_tokens), load_code_files()
    tail = "\nExplain the root cause briefly, then give the fix as a unified diff."
    tail_n = len(enc.encode(tail))
    prompts = []
    for i in range(n):
        repo, problem = issues[i % len(issues)]
        head = f"[request {uuid.uuid4().hex[:12]}]\nYou are an expert software engineer. Resolve the following GitHub issue in `{repo}`.\n\n## Issue\n"
        head_n = len(enc.encode(head))
        budget = max(1, target_tokens - head_n - tail_n)
        # issue gets at most 40% of the budget (all of it for short ones); code fills the rest
        issue_ids = enc.encode(problem)[: max(1, int(budget * 0.4)) if target_tokens > 300 else budget]
        issue = enc.decode(issue_ids)
        remaining = budget - len(issue_ids)
        code = ""
        if remaining > 20:
            code = "\n\n## Relevant source files\n"
            remaining -= len(enc.encode(code))
            order = rng.sample(range(len(files)), len(files))
            for j in order:
                if remaining <= 20:
                    break
                chunk = f"\n### {files[j]['path']}\n```python\n{files[j]['content']}\n```\n"
                ids = enc.encode(chunk)
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    chunk = enc.decode(ids)
                code += chunk
                remaining -= len(ids)
        prompts.append(head + issue + code + tail)
    return prompts
