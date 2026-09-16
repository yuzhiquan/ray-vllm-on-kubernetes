# Stage 3: Offline Batch Inference — Treating vLLM as a Data Operator

> This article corresponds to [`batch_infer.py`](batch_infer.py) and [`rayjob-batch-infer.yaml`](rayjob-batch-infer.yaml).
> Previous article: [Stage 2 SFT Training](../20-train/sft-training.en.md). Baseline: Ray 2.58.0, KubeRay v1.7.0.

## 1. Where this stage sits in the LLM lifecycle

In LLM engineering, "inference" is really **two completely different workloads** that share the same engine but have opposite goals:

| | Offline batch inference (this stage) | Online serving (stage 4) |
| --- | --- | --- |
| Goal | **Throughput**: how many rows per unit time | **Latency**: how fast the first token comes back for a single request |
| Input | A known dataset | An unpredictable request stream |
| Who is waiting | Nobody; you look at the results when it finishes | A user is waiting |
| SLO | Only "finish by tonight" | p99 of TTFT / TPOT |
| Batch strategy | Bigger is better | Constrained by latency |
| GPU memory strategy | Push to the limit | Leave headroom for bursts |
| Failure handling | Just rerun | Must pull traffic immediately |
| Scaling | Computed once from the data volume | Driven by real-time load feedback |

**Confusing the two is the most common architectural mistake in AI infra.** The classic symptom is using online serving to run a batch job (spin up a RayService, then write a script that hammers it with concurrent HTTP calls). The result: an extra layer of HTTP serialization overhead, rate limiting by the online service's `max_ongoing_requests`, retry logic you have to write yourself, and a service that keeps holding the GPU after the run is done.

Offline batch inference has three typical uses in the lifecycle:

```
training artifact ──┬─→ ① evaluation: compute metrics, decide whether to ship   ← what this directory does
                    ├─→ ② data synthesis: create samples for the next round     ← same code, different preprocess
                    └─→ ③ business batch jobs: offline labeling, summarization, translation  ← same code, different data source
```

The code skeleton is identical for all three, and that is no coincidence — they are all "call the model row by row over a dataset." That is precisely the value of turning vLLM into a data operator.

**This stage also carries a special responsibility: it is the real acceptance check for stage 2.** At the end of stage 2 all you know is that the loss went down and the files exist; only by loading the weights here and generating fluent text can you prove the training artifact is actually usable.

**What upstream provides**: the `models/sft-current` symlink (stage 2) and `data/eval.jsonl` (stage 1).
**What downstream needs**: a basis for judging "should this model be shipped."

## 2. What this stage must accomplish

| # | Task | Consequence of skipping it |
| --- | --- | --- |
| 1 | Load **the exact weights about to be shipped** | What you evaluate is not what you ship, so evaluation is meaningless |
| 2 | Batch-generate over the holdout set | — |
| 3 | Results persisted and traceable | No way to review "what the model actually output at the time" |
| 4 | Produce a conclusion you can act on | The ship decision rests on a human eyeballing two samples |

Item 1 is implemented with symlinks (section 3.3 below); item 4 is where this directory **explicitly falls short** (see section 7).

## 3. Why the code is written this way

### 3.1 Why ray.data.llm instead of starting vLLM yourself

The most intuitive way to write this is:

```python
# naive version
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL_PATH)
prompts = [row["instruction"] for row in read_jsonl(EVAL_PATH)]
outputs = llm.generate(prompts, SamplingParams(temperature=0))
```

On a few dozen rows this is perfectly fine. Scale up and you hit four problems in order, each of which you have to solve with your own code:

1. **The data does not fit in memory.** `read_jsonl` reads everything in; a few million prompts blow it up. It has to become streaming.
2. **Only one GPU is used.** To use multiple GPUs you have to spawn processes, shard the data, and merge results yourself.
3. **No backpressure.** Reading data is far faster than generating, so intermediate results pile up.
4. **One failure means starting over.** Six hours in, it dies on row 8,000,000, and there is no checkpoint.

`ray.data.llm` takes care of all four:

```python
config = vLLMEngineProcessorConfig(model_source=MODEL_PATH, ..., concurrency=CONCURRENCY, batch_size=BATCH_SIZE)
processor = build_processor(config, preprocess=..., postprocess=...)
dataset = processor(ray.data.read_json(EVAL_PATH))
dataset.write_parquet(str(output_dir))
```

- Streaming read/write: `read_json` and `write_parquet` are both chunked, so memory usage is independent of data volume.
- Multi-GPU parallelism: `concurrency` decides how many engine actors start, and Ray schedules them onto different GPUs.
- Backpressure: Ray Data's streaming executor automatically throttles upstream based on downstream consumption speed.
- Fault tolerance: if an actor dies, Ray rebuilds it and recomputes that batch (`should_continue_on_error=True` even gives you row-level fault tolerance).

**And it is the same set of operators as stage 1.** The `processor` is just a transform applied to a Dataset; it composes freely with `map` / `filter`, and multiple processors can be chained. "Data processing" and "batch inference" are not two systems on Ray, which is one of the core reasons this directory chose Ray.

### 3.2 What is going on with that compatibility import

```python
try:
    from ray.data.llm import build_processor
except ImportError:  # Ray < 2.57
    from ray.data.llm import build_llm_processor as build_processor
```

Ray 2.57 renamed this builder from `build_llm_processor` to `build_processor`. In the 2.58 source, `build_llm_processor` is no longer in the `__all__` of `ray/data/llm.py`.

Examples online and in plenty of docs still use the old name. The upside of the try/except is that this code runs on both 2.49 and 2.58; the downside is that it hides the version difference — which is why the comment spells out where the line is drawn.

This kind of rename is not an accident but the normal cost of `ray.data.llm` still being in beta (the source marks it `@PublicAPI(stability="beta")`). **If you use a beta API, you accept this and pin the version hard.**

### 3.3 Resolve the symlink into an immutable version before handing it to the engine

```python
MODEL_PATH = os.environ.get("MODEL_PATH", f"{SHARED_DIR}/models/sft-current")
EVAL_PATH = os.environ.get("EVAL_PATH", f"{SHARED_DIR}/data/tokenized/current/eval.jsonl")

# Resolve the symlinks into immutable versions and use the resolved paths from here on
model_version = Path(MODEL_PATH).resolve(strict=True)
eval_version = Path(EVAL_PATH).resolve(strict=True)
print(f"[batch] model version {model_version}")
print(f"[batch] eval version {eval_version}")

processor = build(str(model_version))          # pass the resolved absolute path
dataset = processor(ray.data.read_json(str(eval_version)))
```

`model_source` accepts an HF model id, a local directory, or an `s3://` path. The default points at the `sft-current` symlink for convenience, **but you must not hand the symlink straight to the engine**:

If someone launches a new training run halfway through the evaluation, `sft-current` will point at the new version. Engine actors that already loaded weights are still using the old version, while an actor newly brought up by Ray for fault tolerance loads the new one — **the result file ends up mixing output from two models, with nothing recorded that lets you tell which row came from which**. The same problem applies to the eval set.

`Path.resolve(strict=True)` converts it once, at the start of the job, into a definite path like `models/sft-<RUN_ID>`, and from then on every engine actor uses the same bytes. `strict=True` makes a dangling symlink fail immediately instead of surfacing as a cryptic error when the engine tries to load.

The resolved paths are written into `acceptance.json` in the output directory, so "which model version does this evaluation result correspond to" is on the record rather than in someone's memory.

This has a downstream implication worth noting: **stage 4's `model_source` must not be a symlink either**, but for an entirely different reason — not concurrency, but "changing a symlink does not reload weights already loaded into GPU memory." See [serving article, section 3.2](../40-serve/online-serving.en.md).

### 3.4 engine_kwargs: why offline can be more aggressive than online

```python
engine_kwargs={
    "max_model_len": MAX_MODEL_LEN,          # 2048
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "gpu_memory_utilization": 0.85,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
},
```

Item by item:

**`max_model_len`** — the maximum sequence length the engine supports (prompt + output). It directly determines the per-sequence KV cache size, and therefore how many sequences the engine can hold at once. Setting it larger than you actually need is pure waste; setting it smaller than your actual input means over-long requests get rejected outright. This value has to line up with stage 1's `MAX_LEN` plus `MAX_TOKENS`: `1024 + 256 < 2048`, with headroom.

**`gpu_memory_utilization: 0.85`** — vLLM pre-allocates this fraction of GPU memory for the weights plus the KV cache pool. Offline you can go to 0.85 or even 0.9, because **there is no burst traffic**: the input is a known dataset and a batch of extremely long requests will not suddenly arrive. Stage 4's online serving needs to be more careful with the same 0.85, because the distribution of long requests online is out of your control (stage 4 in this directory also uses 0.85, on the premise of "one model owning the whole GPU"; when multiple models share a GPU it must be lowered).

What is left of that fraction after the weights is the KV cache pool, and together with `max_model_len` it determines how many
sequences the engine can hold at once. Offline you do not need to compute this precisely — if it does not fit, it queues, and nobody is waiting anyway. **Online you must compute it precisely**;
the formula and measured numbers for the two Qwen2.5 sizes are in [serving article, section 3.3](../40-serve/online-serving.en.md).

**`enable_prefix_caching: True`** — caches the KV of identical prefixes and skips prefill on a hit. Every sample in the eval set shares the same chat template prefix (the `<|im_start|>user` part), so it always hits; the speedup is essentially free. But the template header is only a few dozen tokens, so the payoff is limited; what actually produces order-of-magnitude gains is long system prompts and multi-turn conversations — see serving article, section 3.3.

**`enable_chunked_prefill: True`** — splits the prefill of a long prompt into chunks and interleaves it with the decode of other requests. Online you turn it on to stop one long prompt's prefill from blocking every decode request (a direct improvement to TTFT p99); offline you turn it on to raise GPU utilization. Different reasons in the two scenarios, but the same conclusion.

**`tensor_parallel_size`** — only needed above 1 when the model does not fit on a single GPU. It slices every layer horizontally across GPUs, and intra-layer communication is extremely frequent, so it is **only suitable for single-node high-bandwidth interconnect (NVLink)**. Turning on TP across nodes gets dragged to death by the network. Both 0.5B and 7B use 1 on an 80GB card.

### 3.5 What concurrency and batch_size each control

```python
concurrency=CONCURRENCY,   # how many vLLM engine actors to start
batch_size=BATCH_SIZE,     # how many rows to hand the engine per batch
```

The two parameters govern different levels and are easy to mix up:

- **`concurrency`** = number of engine instances = number of GPUs occupied (each engine takes at least one GPU, or TP-many with TP enabled). Written as an integer `n` it means autoscaling from 1..n; written as a tuple `(n, n)` it is a fixed pool. **A fixed pool suits offline work better** — the data volume is known, you work out how many GPUs you need once, and there is no need to probe as you go. This directory uses an integer so it can still run when only 1 GPU is available.
- **`batch_size`** = how many rows are handed to the engine at a time. Note this is **not** vLLM's internal batch — vLLM has its own continuous batching and dynamically recomposes at every decode step. The `batch_size` here is the Ray Data-level batch size; it affects task granularity and memory usage.

There is also `max_concurrent_batches`, defaulting to 8, which controls how many batches are in flight inside a single engine actor. The default is usually fine; before tuning it, confirm the bottleneck is really here.

**How to pick these numbers**: first run 100 rows with `concurrency=1` to measure single-GPU throughput, then `total data volume / single-GPU throughput / target duration` gives you the number of GPUs you need. That is far more reliable than tuning by feel.

### 3.6 The preprocess / postprocess contract

```python
processor = build_processor(
    config,
    preprocess=lambda row: dict(
        messages=[{"role": "user", "content": row["instruction"]}],
        sampling_params=dict(temperature=0.0, max_tokens=MAX_TOKENS, detokenize=False),
    ),
    postprocess=lambda row: dict(
        instruction=row["instruction"],
        reference=row.get("response", ""),
        generated=row["generated_text"],
    ),
)
```

**`messages`, not `prompt`.** Passing messages lets the engine apply the chat template itself — the same template from the same tokenizer that stage 1 used. This is the second safeguard for train/inference format consistency: stage 1 uses `apply_chat_template` to generate the training data, and here the engine uses that same template to build the inference input. If you switched to passing a raw `prompt`, you would have to assemble the template yourself and be back to the risk of inconsistency.

(You only use the `prompt` field for embedding or classification tasks, because those tasks have no conversational structure.)

**`temperature=0.0`.** Evaluation must be reproducible. Sampling with temperature above 0 gives different results every run, making it impossible to tell "did the metric change because the model changed or because sampling jittered." For data synthesis you want the opposite: raise the temperature and pair it with `top_p`, because there you need diversity — same code, opposite parameters for a different purpose.

**`detokenize=False`, that counterintuitive parameter.** Ray Data's processor enables a separate detokenize stage by default, so the pipeline rather than the engine is responsible for turning token ids back into text. So you have to tell the engine "don't convert" to avoid converting twice. The rule is:

| detokenize stage | in `sampling_params` |
| --- | --- |
| enabled (default) | `detokenize=False` |
| disabled | `detokenize=True` |

Getting it backwards will not necessarily raise an error, but you will either decode once for nothing or get unexpected fields.

**postprocess carries `reference` back.** The output file then has three columns — question, reference answer, and model output — so computing metrics later or spot-checking by hand does not require joining back to the raw data. The cost of storing one extra column is far lower than the cost of not being able to line things up afterwards.

### 3.7 Why the file checks come before ray.init

```python
def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"model directory {MODEL_PATH} not found, run stage 2 first")
    if not Path(EVAL_PATH).exists():
        raise SystemExit(f"eval set {EVAL_PATH} not found, run stage 1 first")

    ray.init(address="auto")
```

The order is deliberate: **fast failure must happen before expensive operations.**

If you `ray.init` first and only then discover the path does not exist, you have already paid for: KubeRay creating the cluster, pulling images, the GPU Pod being scheduled successfully, and the Ray cluster forming — several minutes and one GPU held, all to die on a `FileNotFoundError`. And the error would be buried in Ray's call stack, far less direct than the plain message these two lines print.

Sequential dependency between stages is inherent to this chain (data → training → evaluation → serving); you cannot apply all the manifests in one go. These two checks are the cheapest way to make that constraint explicit.

### 3.8 The acceptance check must be able to fail the job

First, look at a version that **appears to be doing an acceptance check but actually verifies nothing**:

```python
# ✗ only prints, never raises
rows = ray.data.read_parquet(str(output_dir)).take(5)
empty = sum(1 for row in rows if not str(row["generated"]).strip())
print(f"[batch] sampled {len(rows)} rows, {empty} empty outputs")
```

Two problems:

1. **It can never fail.** All five sampled rows can be empty strings and the job is still SUCCEEDED, `make batch` still passes, and stage 4 can still start. The so-called "acceptance check" is just a log line.
2. **`str(None)` is `"None"`, which counts as non-empty.** When the engine returned nothing at all, this check declares it "acceptable" — exactly the case that most needs to be caught is precisely the one that slips through.

Turned into a real gate:

```python
def is_usable(value) -> bool:
    """Explicit criterion: must be a non-empty string. Cannot use str(value).strip()."""
    return isinstance(value, str) and bool(value.strip())

written = ray.data.read_parquet(str(output_dir))
total = written.count()
rows = written.take_all() if ACCEPT_SAMPLE == 0 else written.take(ACCEPT_SAMPLE)
if total == 0 or len(rows) == 0:
    raise SystemExit("[batch] acceptance check failed: no output rows at all")

bad = [r for r in rows if not is_usable(r.get("generated"))]
empty_fraction = len(bad) / len(rows)
...
if empty_fraction > MAX_EMPTY_FRACTION:
    raise SystemExit(f"[batch] acceptance check failed: unusable output {empty_fraction:.1%} exceeds threshold")
```

Each of the three changes has its reason:

- **Check everything by default** (`ACCEPT_SAMPLE=0`) rather than sampling 5 rows. This step just reads strings once; the cost is far below generation itself. Sampling is only needed when the data volume is large.
- **`isinstance(value, str)`** rules out `None`, numbers, and missing columns.
- **The threshold is configurable but defaults to 0** (`MAX_EMPTY_FRACTION=0.0`), meaning "not a single empty row is allowed." Real scenarios may need to tolerate a few failures, in which case you raise it explicitly — rather than having no threshold at all.

The result, together with the model/eval versions, is written into `acceptance.json`:

```json
{"model_version": ".../models/sft-20260914-2130",
 "eval_version": ".../data/tokenized/v-20260914-2100/eval.jsonl",
 "rows_total": 2, "rows_unusable": 0, "empty_fraction": 0.0, "passed": true}
```

**Note that this is still only a pipeline self-check, not a model quality evaluation.** There is exactly one question it can answer: the model loaded successfully and produced non-empty text. That happens to be precisely the acceptance check stage 2 needs most — as said earlier, when a falling loss, existing files, and GPU utilization are all true, the model can still have learned nothing; whereas "it can generate fluent text" cannot be faked.

Questions it cannot answer: is the model better or worse than the previous version, is it overfitting, should it be shipped. A real evaluation has to add:

- **General capability**: standard suites like lm-eval-harness;
- **Business metrics**: your own discriminator, rule checks, or LLM-as-judge;
- **Regression gate**: compare against the previous version and refuse to release if a metric drops beyond a threshold.

The third is the biggest gap in this directory; see section 7.

## 4. Why the YAML is written this way

The stage 3 manifest is very similar to stage 2's (both are GPU RayJobs), so this section only covers what is **different**. For the shared parts (the head's `num-gpus: "0"`, `/dev/shm`, Guaranteed QoS, ConfigMap distribution, the `share` emptyDir) see [training article, section 4](../20-train/sft-training.en.md).

### 4.1 Why only 1 GPU worker

```yaml
workerGroupSpecs:
  - groupName: gpu-group
    replicas: 1
    minReplicas: 1
    maxReplicas: 1
```

The fundamental difference from training: **batch inference can be partitioned horizontally, training cannot.**

- Training's 2 workers have to all-reduce gradients at every step; they are an indivisible group — one missing and work cannot start, which is why the previous article covered gang scheduling.
- Every sample in batch inference is independent of the others. One GPU for 8 hours and 8 GPUs for 1 hour give exactly the same result.

So the GPU count here is purely a choice about "how long until it finishes"; there is no correctness requirement, and therefore no need for gang scheduling. The demo data is only 32 rows, so 1 GPU is plenty.

When scaling up, adjust `replicas` and `CONCURRENCY` together (keeping `CONCURRENCY ≤ total GPU count`). Note that when `CONCURRENCY` exceeds the available GPU count, the extra engine actors wait for resources indefinitely — the symptom is similar to "stuck at Initializing" in the training article, section 3.3.

### 4.2 Why /dev/shm is equally mandatory here

```yaml
- name: shared-memory
  emptyDir:
    medium: Memory
    sizeLimit: 8Gi
```

In the training article this was for NCCL and the DataLoader. The purpose here is different but equally mandatory: **vLLM's multiprocessing execution backend passes tensors through shared memory.** With `tensor_parallel_size > 1` it becomes a hard requirement, because each TP rank is a separate process.

Even with `tensor_parallel_size=1`, vLLM may still start a separate engine process. The behavior under the default 64MB is just as baffling as in training: a hang, or an error that has nothing to do with shared memory. So this piece of config should be written into every manifest that runs vLLM.

### 4.3 Why activeDeadlineSeconds is 7200 and not 14400

```yaml
ttlSecondsAfterFinished: 300
activeDeadlineSeconds: 7200
```

The duration of batch inference is **estimable** (row count × average tokens per row / throughput), unlike training with its convergence uncertainty. So the ceiling can be set much closer to reality; 2 hours is enormous headroom for 32 rows.

How to set this value: measure throughput at small scale first, compute the expected duration, and multiply by 2-3x as the ceiling. Set it too loose and it is no protection at all (a hung job still holds the GPU); set it too tight and it kills healthy jobs.

### 4.4 Why the environment variables all live on the head

```yaml
headGroupSpec:
  template:
    spec:
      containers:
        - name: ray-head
          env:
            - name: MODEL_PATH
              value: /mnt/cluster_storage/models/sft-current
            - name: CONCURRENCY
              value: "1"
            ...
```

A RayJob's `entrypoint` runs on the **head Pod** (under K8sJobMode a submitter Job submits it via `ray job submit`, and the code runs on the head). Every config value `batch_infer.py` reads takes effect on the driver side — `vLLMEngineProcessorConfig` is constructed on the driver and then distributed to the actors, so only the head needs these environment variables.

The workers keep only `SHARED_DIR`, because the engine actors need to reach the model directory on the shared volume.

**This also explains why the PVC must be mounted on both head and workers**: the driver does the path-existence checks on the head, and the engine actors actually read the weight files on the workers. Mount only one side and the symptom is "the check passes but the engine fails to load," or the reverse.

## 5. How to verify it

```bash
make batch
```

```bash
# 1. The job succeeded. Because a failed acceptance check raises, SUCCEEDED now really includes "the output is usable"
kubectl -n llm-pipeline get rayjob llm-batch-infer -o jsonpath='{.status.jobStatus}{"\n"}'

# 2. Check the versions and the acceptance verdict
kubectl -n llm-pipeline logs -l job-name=llm-batch-infer-<suffix> | grep -A 6 '\[batch\]'
# Expect to see:
#   [batch] model version /mnt/cluster_storage/models/sft-<RUN_ID>
#   [batch] eval version /mnt/cluster_storage/data/tokenized/v-<RUN_ID>/eval.jsonl
#   [batch] acceptance check: N rows total, N rows checked, 0 rows unusable (0.0%)
#   [batch] acceptance check passed

# 3. Results and the acceptance report
#    (from a temporary Pod with the PVC mounted)
ls /mnt/cluster_storage/outputs/batch/v-*/
cat /mnt/cluster_storage/outputs/batch/v-*/acceptance.json
```

Criteria:

1. `jobStatus: SUCCEEDED` — this one now carries more weight than before: a failed acceptance check exits non-zero;
2. `passed: true` and `rows_unusable: 0` in `acceptance.json`;
3. **`model_version` points at exactly the version stage 2 just exported** (matching the `run_id` in `pipeline_provenance.json`);
4. In the two printed `{"q": ..., "a": ...}` entries, `a` is fluent English relevant to the question, and **ends naturally** rather than being cut off by `max_tokens`.

Item 4 needs a human to look at it, but it carries the most information; it verifies three things at once:

| Observed behavior | What it means |
| --- | --- |
| Output is fluent and relevant | Stage 2 training worked, and stage 1's template and labels are both correct |
| Output restates the question | Stage 1's label masking is wrong (the prompt was not masked with -100) |
| Output never stops and gets cut off | Stage 1 is missing the eos, or the eos was masked out |
| Output is gibberish | Stage 2's state_dict key prefix was not cleaned up, so the weights loaded misaligned |
| Output format is completely different from the training data | The chat template differs between training and inference |

**In other words, these three lines of stage 3 output are the only checkup point for every silent failure in stages 1 and 2.** That is also why it should not be skipped — even if you do no formal evaluation.

## 6. The mistakes that bite hardest

| Symptom | Actual cause |
| --- | --- |
| `ImportError: cannot import name 'build_llm_processor'` | Ray ≥ 2.57 renamed it to `build_processor` (section 3.2) |
| OOM when the engine starts | `gpu_memory_utilization` too high, or sharing the GPU with another process |
| Stuck at engine initialization | `/dev/shm` too small (section 4.2) |
| Generated results are empty strings | The `detokenize` parameter is set backwards relative to the detokenize stage (section 3.6) |
| Engine actors wait for resources forever | `CONCURRENCY` > available GPU count (section 4.1) |
| Output is truncated | `max_tokens` too small, or stage 1 is missing the eos |
| Requests rejected (too long) | `max_model_len` < prompt + max_tokens |
| Different results on every run | `temperature` not set to 0 (section 3.6) |

## 7. What this stage deliberately omits

This is **the stage with the biggest gap** of the four in this directory, because it is supposed to be the gatekeeper for shipping:

| Not done | Cost | Priority |
| --- | --- | --- |
| **Quantitative metrics + regression gate** | Only "there is non-empty output" is verified, not "the output is better than the previous version" | Highest |
| Per-file hash verification | The model version path is recorded, but there is no proof the bytes in that directory were not modified | Medium |
| Writing the acceptance verdict back into the model directory | Stage 4 cannot check "has this version been evaluated" before loading | Medium |
| Multiple sampling runs to test stability | No way to distinguish "the model is bad" from "this sampling run is bad" | Low (unnecessary when evaluating with temperature=0) |

Already added (absent in earlier versions): **resolving symlinks into immutable versions and recording them**,
**an acceptance gate that fails the job**, **an `acceptance.json` report persisted to disk**, and **an eval set disjoint from the training set** (stage 1).

The minimum viable way to fill the first gap does not require bringing in an evaluation framework:

1. Compute a **per-token negative log-likelihood (NLL) on the validation set** — a continuous, stable number aligned with the training objective, far more reliable than human judgment of generation quality;
2. Compare against the previous version and **exit with failure** if the regression exceeds a threshold;
3. On pass, write a `_READY.json` marker into the model directory; stage 4 checks that marker before loading and refuses to start without it.

That way the "gate" is not a sentence in a process document but a `raise` in the code. Two conventions go with it: per-file hash verification of the artifacts, and "refuse to overwrite an already-released version" — stage 2 in this directory already implements the latter with `export_dir.exists()`.

One trade-off worth mentioning along the way: that implementation computes NLL with transformers directly (not vLLM), because computing NLL only needs a single forward pass and no autoregressive generation, so vLLM's scheduling optimizations are of no use. **So the ideal stage 3 is really two jobs**: one that computes NLL with transformers as the gate, and one that batch-generates with vLLM for human spot checks and data synthesis. This directory only does the latter.

## 8. What changes when you scale up

| Data volume | Configuration |
| --- | --- |
| Tens of rows (this directory) | `concurrency=1`, `batch_size=32`, 1 GPU |
| A hundred thousand rows | `concurrency=(4,4)` fixed pool, `batch_size=64`, 4 GPUs; mind `min_rows_per_file` when `write_parquet` writes results |
| Tens of millions of rows | Turn on `should_continue_on_error=True` for row-level fault tolerance; switch to object storage; consider bucketing and sorting by input length to raise in-batch utilization; `activeDeadlineSeconds` has to be recomputed |

One counterintuitive optimization: **sorting by prompt length before feeding the engine** significantly raises throughput, because when lengths within a batch are close, less is wasted on padding and KV cache allocation is more compact. Offline this is pure gain (nobody is waiting on an individual result); online it is completely unworkable (it would postpone short requests indefinitely). This is one more example of the offline/online trade-off.

---

**Next article**: [Stage 4 — Online Serving](../40-serve/online-serving.en.md), covering how RayService turns vLLM into an OpenAI-compatible endpoint, why the probes are the most dangerous configuration in this stage, and what additional requirements streaming responses impose on rolling releases.