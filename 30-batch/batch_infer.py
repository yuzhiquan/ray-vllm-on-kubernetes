"""Stage 3: offline batch inference / evaluation (Ray Data + vLLM on RayJob).

Uses the weights exported by stage 2 to generate over the holdout set, writes
parquet, and prints one coarse-grained metric. Swap the preprocess function and
the same code becomes data synthesis (using a large model to create samples for
the next training round). That is exactly why "data processing" and "inference"
share one set of operators on Ray.

The difference from online serving: there is no HTTP layer and no SLO here, the
goal is throughput. So batch_size can be large, and concurrency decides how many
vLLM engine actors are brought up at once.

Verified version: Ray 2.58.0. From 2.57 on the exported name is build_processor,
earlier versions called it build_llm_processor; the code below handles both.
"""

import json
import os
import time
from pathlib import Path

import ray
from ray.data.llm import vLLMEngineProcessorConfig

try:
    from ray.data.llm import build_processor
except ImportError:  # Ray < 2.57
    from ray.data.llm import build_llm_processor as build_processor

SHARED_DIR = os.environ.get("SHARED_DIR", "/mnt/cluster_storage")
# Both defaults are symlinks; main() resolves them to immutable versions before
# use — otherwise, if someone reruns an upstream stage midway, the result file
# ends up mixing output from two versions with no way to tell them apart.
MODEL_PATH = os.environ.get("MODEL_PATH", f"{SHARED_DIR}/models/sft-current")
EVAL_PATH = os.environ.get(
    "EVAL_PATH", f"{SHARED_DIR}/data/tokenized/current/eval.jsonl"
)
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", f"{SHARED_DIR}/outputs/batch")
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
# Acceptance check: sample this many rows, 0 means check everything. If the
# fraction of empty output exceeds the threshold, fail the job.
ACCEPT_SAMPLE = int(os.environ.get("ACCEPT_SAMPLE", "0"))
MAX_EMPTY_FRACTION = float(os.environ.get("MAX_EMPTY_FRACTION", "0.0"))

CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))


def build(model_source: str) -> "object":
    config = vLLMEngineProcessorConfig(
        # A local directory, an HF id, or s3:// all work. What is passed here is
        # an already-resolved immutable version directory, so "what gets
        # evaluated is the exact set of weights recorded in the report" — even
        # if the symlink is repointed midway.
        model_source=model_source,
        engine_kwargs={
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "gpu_memory_utilization": 0.85,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        # An int means autoscaling from 1..n; a tuple (n, n) is a fixed pool.
        concurrency=CONCURRENCY,
        batch_size=BATCH_SIZE,
    )

    return build_processor(
        config,
        preprocess=lambda row: dict(
            messages=[{"role": "user", "content": row["instruction"]}],
            # detokenize=False because the detokenize stage is on by default,
            # so the pipeline rather than the engine does the decoding; switch
            # this to True when that stage is turned off.
            sampling_params=dict(temperature=0.0, max_tokens=MAX_TOKENS, detokenize=False),
        ),
        postprocess=lambda row: dict(
            instruction=row["instruction"],
            reference=row.get("response", ""),
            generated=row["generated_text"],
        ),
    )


def is_usable(value) -> bool:
    """An explicit acceptance criterion: must be a non-empty string.

    It cannot be written as ``str(value).strip()`` — ``str(None)`` is
    ``"None"``, which would count "the engine returned nothing at all" as
    acceptable output.
    """
    return isinstance(value, str) and bool(value.strip())


def main() -> None:
    model_path = Path(MODEL_PATH)
    eval_path = Path(EVAL_PATH)
    if not model_path.exists():
        raise SystemExit(f"model directory {MODEL_PATH} not found, run stage 2 first")
    if not eval_path.exists():
        raise SystemExit(f"eval set {EVAL_PATH} not found, run stage 1 first")

    # Resolve the symlinks into immutable versions and use the resolved paths
    # from here on.
    model_version = model_path.resolve(strict=True)
    eval_version = eval_path.resolve(strict=True)
    print(f"[batch] model version {model_version}")
    print(f"[batch] eval version {eval_version}")

    ray.init(address="auto")

    processor = build(str(model_version))
    dataset = processor(ray.data.read_json(str(eval_version)))

    output_dir = Path(OUTPUT_ROOT) / f"v-{RUN_ID}"
    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists, pick a different RUN_ID instead of "
            "overwriting existing results"
        )
    output_dir.mkdir(parents=True)
    dataset.write_parquet(str(output_dir))
    print(f"[batch] generated results written to: {output_dir}")

    # Pipeline acceptance check (not a model quality evaluation): confirm the
    # engine really produced usable text. This step will fail the job —
    # otherwise the "acceptance check" is just a log line and downstream stages
    # start up all the same.
    written = ray.data.read_parquet(str(output_dir))
    total = written.count()
    rows = written.take_all() if ACCEPT_SAMPLE == 0 else written.take(ACCEPT_SAMPLE)
    checked = len(rows)
    if total == 0 or checked == 0:
        raise SystemExit("[batch] acceptance check failed: no output rows at all")

    bad = [row for row in rows if not is_usable(row.get("generated"))]
    empty_fraction = len(bad) / checked
    print(
        f"[batch] acceptance check: {total} rows total, {checked} rows checked, "
        f"{len(bad)} rows unusable ({empty_fraction:.1%})"
    )

    for row in rows[:2]:
        print(json.dumps(
            {"q": row["instruction"], "a": str(row.get("generated"))[:200]},
            ensure_ascii=False,
            indent=2,
        ))

    report = {
        "run_id": RUN_ID,
        "model_version": str(model_version),
        "eval_version": str(eval_version),
        "rows_total": total,
        "rows_checked": checked,
        "rows_unusable": len(bad),
        "empty_fraction": empty_fraction,
        "max_empty_fraction": MAX_EMPTY_FRACTION,
        "passed": empty_fraction <= MAX_EMPTY_FRACTION,
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not report["passed"]:
        raise SystemExit(
            f"[batch] acceptance check failed: unusable output {empty_fraction:.1%} "
            f"exceeds threshold {MAX_EMPTY_FRACTION:.1%}"
        )
    print("[batch] acceptance check passed")


if __name__ == "__main__":
    main()