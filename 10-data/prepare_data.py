"""Stage 1: data processing (Ray Data on RayJob).

Read raw jsonl -> filter -> exact deduplication by content hash -> carve out a
disjoint holdout -> apply the chat template and tokenize -> fixed-length padding
-> write parquet.

Four design constraints are worth noting:

1. Every row of the output parquet is a fixed-length array (input_ids /
   attention_mask / labels), so stage 2's iter_torch_batches can stack them
   straight into rectangular tensors with no custom collate.
2. In labels, both the prompt segment and the padding segment are set to -100,
   so loss is computed on the answer only.
3. Over-length samples get an explicit policy: the prompt is never truncated
   (that would break the template), the answer may be truncated but keeps its
   eos, and a sample is rejected outright if the supervision budget is too
   small. Without this, a prompt that fills max_len yields labels that are all
   -100, and a whole batch with zero supervision turns cross-entropy into nan.
4. The holdout is **disjoint** from the training set, and both are written into
   the same version directory, which avoids the mismatch where "the training
   data is the old version but the eval data has already been overwritten by
   the new one."

Artifacts are written to tokenized/v-<RUN_ID>/ (parquet, eval.jsonl,
manifest.json). The tokenized/current symlink is updated only after everything
succeeds. Version directories are created exclusively; existing versions are
never overwritten.

Verified against Ray 2.58.0.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import ray

SHARED_DIR = os.environ.get("SHARED_DIR", "/mnt/cluster_storage")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RAW_DIR = os.environ.get("RAW_DIR", f"{SHARED_DIR}/data/raw")
TOKENIZED_ROOT = os.environ.get("TOKENIZED_ROOT", f"{SHARED_DIR}/data/tokenized")
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
EVAL_ROWS = int(os.environ.get("EVAL_ROWS", "32"))
# The eval set may take at most this fraction of the dataset, so that a small
# dataset does not get consumed entirely by EVAL_ROWS.
EVAL_MAX_FRACTION = float(os.environ.get("EVAL_MAX_FRACTION", "0.25"))
# Minimum supervision budget for over-length samples: the answer must keep at
# least this many tokens (excluding eos), otherwise the whole sample is
# rejected. See the comments in Tokenize._encode.
MIN_ANSWER_TOKENS = int(os.environ.get("MIN_ANSWER_TOKENS", "16"))
TOKENIZE_CONCURRENCY = int(os.environ.get("TOKENIZE_CONCURRENCY", "2"))

# This exists only so the whole pipeline can run without an external dataset.
# In a real setting, point RAW_DIR at your own jsonl / parquet; just align the
# field names to instruction / response.
DEMO_ROWS = [
    ("In Kubernetes, does a Pod reaching Running mean it can take traffic?",
     "No. Running only says the container has started; whether it can take "
     "traffic is decided by the readiness probe. After reaching Running, a "
     "model server still has to download weights, initialize the engine and "
     "allocate KV cache, and readiness must keep the Pod out of Endpoints for "
     "that whole window."),
    ("What problem does vLLM's continuous batching solve?",
     "It solves short requests being stuck behind long ones under static "
     "batching. The scheduler re-forms the batch at every decode step, so "
     "finished sequences leave immediately and queued sequences join "
     "immediately, which raises GPU utilization and throughput."),
    ("Why is CPU utilization a poor metric for HPA on an inference service?",
     "Because the bottleneck is the GPU and GPU memory, not the CPU. During "
     "decode the CPU is often idle while the GPU is already saturated. You "
     "should use metrics that reflect queueing, such as queue depth, "
     "concurrent request count, or TTFT/TPOT."),
    ("Why does the KV cache turn an inference replica into a stateful service?",
     "Only a prefix hit in the cache lets you skip redundant prefill. The "
     "cache lives in the GPU memory of one specific replica, so a request "
     "routed elsewhere has to recompute it. That means routing has to be aware "
     "of where the cache is."),
    ("What extra requirements does a rolling update impose on streaming "
     "responses?",
     "A long-lived SSE connection is cut when the Pod terminates, and the "
     "client has already received part of the tokens. You need a long enough "
     "terminationGracePeriodSeconds, preStop draining, and clearly defined "
     "retry and deduplication semantics after an interruption."),
    ("What is the trade-off between tensor parallelism and pipeline "
     "parallelism?",
     "Tensor parallelism splits within a layer and communicates frequently, so "
     "it suits a single node with high-bandwidth interconnect. Pipeline "
     "parallelism splits by layer, communicates less but has bubbles, so it "
     "suits multi-node. If the model fits on one machine, do not turn TP on."),
    ("How does Ray scheduling relate to Kubernetes scheduling?",
     "Kubernetes schedules Pods onto nodes; Ray schedules tasks and actors "
     "inside the cluster formed by those existing Pods. KubeRay connects the "
     "two layers, but Ray's logical GPU quota is not GPU memory isolation."),
    ("Why must LLM training always handle checkpoint recovery?",
     "Multi-GPU, multi-node runs last a long time, and the probability of a "
     "single-point failure grows with scale. Without periodic checkpoints and "
     "automatic recovery, one preemption means starting over from scratch."),
]


def seed_demo_data(raw_dir: str) -> None:
    """Write demo data when RAW_DIR has no jsonl, so the pipeline can run."""
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    target = Path(raw_dir) / "demo.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        # The 16 repeats exist only to make 8 samples form several batches;
        # this is not a claim about a reasonable training set size.
        for repeat in range(16):
            for instruction, response in DEMO_ROWS:
                fh.write(json.dumps(
                    {"instruction": instruction, "response": response, "repeat": repeat},
                    ensure_ascii=False,
                ) + "\n")
    print(f"[data] no jsonl in RAW_DIR, wrote demo data: {target}")


def has_input(raw_dir: str) -> bool:
    root = Path(raw_dir)
    return root.is_dir() and any(root.glob("*.jsonl"))


def content_hash(row: Dict[str, Any]) -> Dict[str, Any]:
    digest = hashlib.sha256(
        f"{row['instruction']}\x00{row['response']}".encode("utf-8")
    ).hexdigest()
    return {**row, "content_hash": digest}


def take_first(group):
    """Keep one row per hash, giving exact deduplication. batch_format is pinned to pandas."""
    return group.head(1)


class Tokenize:
    """Stateful UDF: the tokenizer is loaded once per actor.

    The over-length policy is explicit, not "concatenate first, then truncate
    uniformly": the latter makes labels all -100 (zero supervision) when the
    prompt fills max_len, and a whole batch of zero-supervision samples turns
    cross-entropy into 0/0 = nan. Even when part of the answer survives,
    truncation can cut off the eos, so the model never learns to stop.
    """

    def __init__(self, base_model: str, max_len: int, min_answer_tokens: int):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len
        self.min_answer_tokens = min_answer_tokens

    def _encode(self, instruction: str, response: str):
        """Return (input_ids, attention_mask, labels); return None for unusable samples."""
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id

        # Strategy 1: never truncate the prompt. Its tail is the template's assistant
        # start marker; left-truncation breaks format consistency, right-truncation
        # misaligns the supervision boundary — better to drop it than to build a bad sample.
        budget = self.max_len - len(prompt_ids)
        if budget < self.min_answer_tokens + 1:  # +1 leaves room for eos
            return None

        # Strategy 2: the answer may be truncated, but eos must be kept — otherwise the model never learns to stop.
        keep = min(len(answer_ids), budget - 1)
        supervised = answer_ids[:keep] + [eos]

        input_ids = prompt_ids + supervised
        labels = [-100] * len(prompt_ids) + supervised
        attention_mask = [1] * len(input_ids)

        pad = self.max_len - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad
            labels = labels + [-100] * pad
            attention_mask = attention_mask + [0] * pad
        return input_ids, attention_mask, labels

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        encoded: List[List[int]] = []
        masks: List[List[int]] = []
        label_rows: List[List[int]] = []
        rejected = 0
        for instruction, response in zip(batch["instruction"], batch["response"]):
            result = self._encode(str(instruction), str(response))
            if result is None:
                rejected += 1
                continue
            input_ids, attention_mask, labels = result
            encoded.append(input_ids)
            masks.append(attention_mask)
            label_rows.append(labels)
        if rejected:
            # Each actor prints its own count; main() reports the total from the
            # row-count difference.
            print(f"[data] rejected {rejected} over-length samples in this batch (the prompt squeezed out the supervision budget)")
        shape = (len(encoded), self.max_len)
        return {
            "input_ids": np.asarray(encoded, dtype=np.int64).reshape(shape),
            "attention_mask": np.asarray(masks, dtype=np.int64).reshape(shape),
            "labels": np.asarray(label_rows, dtype=np.int64).reshape(shape),
        }


def pick_eval_hashes(ds, total: int) -> set:
    """Deterministically pick the set of content_hash values for the holdout.

    Sort by content_hash and take the first N, so the same input always carves
    out the same eval set. The cap is EVAL_MAX_FRACTION of the dataset;
    otherwise a small dataset would be consumed entirely by EVAL_ROWS and the
    training set would end up empty.
    """
    cap = max(1, int(total * EVAL_MAX_FRACTION))
    rows = min(EVAL_ROWS, cap)
    selected = ds.sort("content_hash").take(rows)
    return {row["content_hash"] for row in selected}


def dump_eval_set(ds, path: Path) -> int:
    """Write the holdout as plain-text jsonl inside the data version directory.

    It goes inside the version directory rather than under data/, so the eval
    set and its corresponding training set are the same version. That avoids
    the mismatch where "the training data is the old version but the eval data
    has already been overwritten by the new one."
    """
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in ds.take_all():
            fh.write(json.dumps(
                {"instruction": row["instruction"], "response": row["response"]},
                ensure_ascii=False,
            ) + "\n")
            count += 1
    return count


def publish(version_dir: Path, root: Path) -> None:
    """Atomically point current at this run's artifacts; old versions are kept for rollback."""
    link = root / "current"
    staging = root / f".current-{RUN_ID}"
    staging.symlink_to(version_dir.name)
    os.replace(staging, link)
    print(f"[data] {link} -> {version_dir.name}")


def main() -> None:
    if not has_input(RAW_DIR):
        seed_demo_data(RAW_DIR)

    ray.init(address="auto")

    ds = ray.data.read_json(RAW_DIR)
    raw_count = ds.count()

    ds = ds.filter(
        lambda row: bool(str(row.get("instruction") or "").strip())
        and bool(str(row.get("response") or "").strip())
    )

    # Exact deduplication: group by hash, then take one row per group.
    # Approximate deduplication (MinHash / SimHash) is cheaper, but unnecessary
    # at this scale, and it would make the result non-reproducible.
    deduped = (
        ds.map(content_hash)
        .groupby("content_hash")
        .map_groups(take_first, batch_format="pandas")
        .materialize()
    )
    total = deduped.count()
    print(f"[data] raw {raw_count} rows -> {total} rows after filtering and deduplication")

    # Carve out the holdout first, then exclude it from the training set — the
    # two must be disjoint for stage 3's evaluation to mean anything.
    eval_hashes = pick_eval_hashes(deduped, total)
    eval_ds = deduped.filter(lambda row: row["content_hash"] in eval_hashes)
    train_ds = deduped.filter(lambda row: row["content_hash"] not in eval_hashes).materialize()
    train_count = train_ds.count()
    if train_count == 0:
        raise SystemExit("training set is empty after excluding the holdout; lower EVAL_ROWS or add more data")

    root = Path(TOKENIZED_ROOT)
    version_dir = root / f"v-{RUN_ID}"
    # Exclusive creation: re-running with the same RUN_ID must explicitly change
    # the ID; it must not overwrite a version that may be being read right now.
    try:
        version_dir.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f"{version_dir} already exists; pick a different RUN_ID instead of overwriting an existing data version")

    tokenized = train_ds.map_batches(
        Tokenize,
        fn_constructor_kwargs={
            "base_model": BASE_MODEL,
            "max_len": MAX_LEN,
            "min_answer_tokens": MIN_ANSWER_TOKENS,
        },
        batch_format="numpy",
        batch_size=64,
        concurrency=TOKENIZE_CONCURRENCY,
    ).materialize()

    kept = tokenized.count()
    if kept == 0:
        raise SystemExit(
            f"all {train_count} samples were rejected as over-length; "
            f"raise MAX_LEN (currently {MAX_LEN}) or shorten the prompts"
        )
    tokenized.write_parquet(str(version_dir))

    # The eval set and the manifest are written only after tokenization
    # succeeds. In the other order, a failed preparation run would leave a new
    # eval set paired with the old training set.
    eval_count = dump_eval_set(eval_ds, version_dir / "eval.jsonl")

    manifest = {
        "run_id": RUN_ID,
        "base_model": BASE_MODEL,
        "raw_rows": raw_count,
        "deduped_rows": total,
        "train_rows_before_tokenize": train_count,
        "train_rows": kept,
        "rejected_overlength": train_count - kept,
        "eval_rows": eval_count,
        "max_len": MAX_LEN,
        "min_answer_tokens": MIN_ANSWER_TOKENS,
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[data] train {kept} rows (rejected {train_count - kept} over-length), "
        f"eval {eval_count} rows, the two are disjoint"
    )

    publish(version_dir, root)
    print(f"[data] data version published: {version_dir}")


if __name__ == "__main__":
    main()