# Stage 1: Data processing — turning instruction/response data into fixed-length tensors the trainer can eat directly

> This article corresponds to [`prepare_data.py`](prepare_data.py) and [`rayjob-data-prep.yaml`](rayjob-data-prep.yaml).
> For the series overview see the [parent README](../README.md). Baseline: Ray 2.58.0, KubeRay v1.7.0.

## 1. Where this stage sits in the LLM lifecycle

**This stage prepares SFT data.** SFT (supervised fine-tuning) belongs to **post-training**, and it
is a different job from processing a pretraining corpus. If you came here looking for pretraining
corpus cleaning, the code skeleton of this stage is reusable, but the template logic and the
deduplication algorithm both have to be replaced — see the end of this section.

```
pretrain ──► post-training ──► inference deployment ──► continuous iteration
             │
             ├─ SFT (supervised fine-tuning)   ◄── this is what the four stages here do
             ├─ preference alignment DPO / RLHF
             └─ distillation / domain adaptation
```

Three characteristics tell you "this is SFT data, not a pretraining corpus," and all three are
visible directly in `prepare_data.py`:

1. The input is **paired** `instruction` / `response`, not an unlabeled plain-text stream.
2. The prompt span in `labels` is masked with `-100`, so **loss is computed on the answer only**,
   not on every token.
3. It applies `apply_chat_template`, whereas pretraining has no notion of a chat template.

The difference between the two determines almost every technical choice in this stage:

| | SFT data (this stage) | Pretraining corpus (not this stage) |
| --- | --- | --- |
| Scale | thousands ~ millions of rows | TB ~ PB |
| Structure | paired instruction / response | plain text stream, no labels |
| Core difficulty | **where the supervision signal goes**, template consistency | deduplication, quality scoring, mixture ratios, throughput |
| loss coverage | the answer portion only | all tokens |
| Deduplication strategy | exact deduplication is usually enough | approximate deduplication is mandatory (MinHash / SimHash) |
| Dominant cost | negligible | I/O and CPU |

If what you are building is a pretraining corpus, the code skeleton of this stage (Ray Data operators + versioned publishing) still applies, but the template logic inside `Tokenize` has to be replaced wholesale, and deduplication has to switch to an approximate algorithm.

**What upstream gives you**: a pile of `{"instruction": ..., "response": ...}` jsonl.
**What downstream wants**: stage 2's `iter_torch_batches` wants `input_ids` / `attention_mask` / `labels` that can be stacked directly into rectangular tensors, without having to write a `collate_fn`.

The entire gap between those two sentences is the work of this stage.

## 2. What this stage must accomplish

| # | Work | Consequence of skipping it |
| --- | --- | --- |
| 1 | Structural filtering: drop empty instruction or empty response | An empty answer contributes a pure-padding sample, polluting the loss denominator |
| 2 | Deduplication | Duplicate samples silently up-weight part of the data, and frequently repeated ones get memorized |
| 3 | **Apply the chat template** | The input format differs between training and inference, and the model answers off-topic once deployed |
| 4 | tokenize | — |
| 5 | **Build labels, mask the prompt** | The model learns to restate the question instead of answering it |
| 6 | Fixed-length padding | Downstream has to write a collate_fn and handle length mismatches across workers |
| 7 | Versioned publishing + eval split | No rollback, and no way to answer "which data was this model trained on" |

Items 3 and 5 are where all the technical substance of SFT data preparation lives, and they are also the two most prone to silent failure — get them wrong and training still converges, the loss curve still looks fine, and you only discover the model behaves incorrectly after it goes live.

## 3. Why the code is written this way

### 3.1 Why Ray Data instead of pandas or Spark

```python
import ray
ds = ray.data.read_json(RAW_DIR)
```

Three reasons, in order of importance:

1. **It shares a runtime with downstream.** Stage 2 uses Ray Train, stage 3 uses Ray Data + vLLM. Using Ray Data for the data stage too means all four stages share the same image, the same serialization, the same logging and dashboard. Going cross-stack (Spark produces data → PyTorch reads it) means maintaining an extra format contract and an extra dependency set.
2. **Streaming execution.** Ray Data is chunked and streaming, so it does not need to read the whole dataset into memory. pandas simply fails once the data exceeds single-machine memory, and the failure usually hits after you have already been running for 40 minutes.
3. **The same operators can feed the trainer.** In stage 2, `TorchTrainer(datasets={"train": ds})` consumes a Ray Dataset directly, and Ray takes care of sharding per worker. With pandas you would have to write the sharding logic yourself, and also guarantee that each rank gets non-overlapping data.

Conversely, if your data is only a few tens of thousands of rows and your team already has a mature Spark data pipeline, then having Spark produce parquet and Ray only read it is a perfectly reasonable choice. This stage uses Ray Data to keep the whole pipeline on one runtime, not because Ray Data is better than Spark at data processing.

### 3.2 Why filter before map

```python
ds = ds.filter(lambda row: bool(...instruction...) and bool(...response...))
deduped = ds.map(content_hash).groupby("content_hash").map_groups(...)
# tokenize comes last
```

The ordering is "cheap operations first, expensive ones later." `filter` is pure string checking, while `Tokenize` has to load a tokenizer and do actual encoding — a two-to-three orders of magnitude difference in per-row cost. Filter first, then tokenize, and the discarded samples never pay the encoding cost.

Deduplication also comes before tokenization — it deduplicates the **original text** rather than the token sequence. The results are almost identical, but a hash of the original text does not depend on the tokenizer version, so the deduplication result stays the same when you swap the base model, which is more reproducible.

### 3.3 Why exact deduplication by content hash instead of MinHash

```python
def content_hash(row):
    digest = hashlib.sha256(
        f"{row['instruction']}\x00{row['response']}".encode("utf-8")
    ).hexdigest()
    return {**row, "content_hash": digest}
```

Two details are worth calling out:

- **The `\x00` separator.** Concatenating without a separator makes `("ab", "c")` and `("a", "bc")` produce the same hash. `\x00` never appears in normal text, which makes it the least-effort separator. This bug will never show up on small data.
- **Exact, not approximate.** SFT data volume is small, so the cost of exact deduplication is negligible; approximate deduplication requires tuning a threshold, and the result changes whenever the threshold does, making it non-reproducible. **Only at pretraining-corpus scale, where the insufficient coverage of exact deduplication (change one punctuation mark and you bypass it) becomes a real problem, is MinHash's complexity worth introducing.**

```python
deduped = (
    ds.map(content_hash)
    .groupby("content_hash")
    .map_groups(take_first, batch_format="pandas")
    .materialize()
)
```

`batch_format="pandas"` is specified explicitly, because the default for `map_groups` is `"default"` — and whether that default hands you a numpy dict or a pandas DataFrame depends on the column types of the data, which is not something you should be guessing about while writing code. Pinned to pandas, `take_first` is a single line of `group.head(1)` with no semantic ambiguity whatsoever.

`.materialize()` lands the result as a concrete dataset. Without it, the later `take()` in `dump_eval_set` and the full pass in `map_batches` would each re-run the upstream filter + groupby. `groupby` is a shuffle operation, and running it twice is genuine waste.

### 3.4 Why Tokenize is a class rather than a function

```python
class Tokenize:
    def __init__(self, base_model, max_len):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
```

As a function, Ray Data calls it statelessly, so every batch would have to `from_pretrained` the tokenizer again. As a class, Ray instantiates it as an actor and `__init__` runs once per actor. For an object like a tokenizer — expensive to load, cheap to call — this is mandatory.

The price is that `concurrency` becomes a required argument:

```python
.map_batches(
    Tokenize,
    fn_constructor_kwargs={"base_model": BASE_MODEL, "max_len": MAX_LEN},
    batch_format="numpy",
    batch_size=64,
    concurrency=TOKENIZE_CONCURRENCY,
)
```

Ray Data needs to know how many actors to start; for a stateless function it can decide on its own, for a stateful class you must tell it.

`import transformers` sits inside `__init__` rather than at module top level so the driver process does not have to import transformers — the driver only orchestrates. This is necessary when the driver and worker images differ; the four stages in this directory share one image, so here it is just good hygiene.

### 3.5 Why apply_chat_template is mandatory

```python
prompt = self.tokenizer.apply_chat_template(
    [{"role": "user", "content": instruction}],
    tokenize=False,
    add_generation_prompt=True,
)
```

**This is the single most important line in the whole stage.**

The input to a chat model is not bare text but a conversation structure with special markers. Qwen2.5's template looks like this (illustrative):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
```

The crux is this: **when stage 4's vLLM handles a `/v1/chat/completions` request, it renders messages into a prompt using the same template from the same tokenizer.** If at training time you assembled your own `f"Q: {q}\nA: "`, then the trigger format the model learned and the format it actually receives at inference are not the same thing. The consequences are:

- loss drops normally, and the eval set (if you also use your own assembled format) looks good;
- the moment it goes live it answers off-topic, never stops, or restates the system prompt.

This is the most classic silent failure in SFT. Using `apply_chat_template` makes the template have a single source of truth — decided by the model card, read from the same tokenizer by both training and inference.

`add_generation_prompt=True` means appending `<|im_start|>assistant\n` at the end, the marker for "it's the model's turn to speak." That way the position where the prompt ends is exactly the position where the answer begins, so the labels computed in the next step line up.

### 3.6 Why labels are built this way

```python
prompt_ids  = tokenizer(prompt,   add_special_tokens=False)["input_ids"]
answer_ids  = tokenizer(response, add_special_tokens=False)["input_ids"]
answer_ids  = answer_ids + [tokenizer.eos_token_id]

input_ids      = (prompt_ids + answer_ids)[:max_len]
labels         = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
attention_mask = [1] * len(input_ids)
```

Four decisions:

**① `add_special_tokens=False`.** The template has already put in the special markers it needs; letting the tokenizer add them again automatically would yield duplicated BOS and the like. Moreover, the prompt and the answer are encoded separately and then concatenated, so if each encoding adds special tokens automatically, extra content appears at the junction, and `len(prompt_ids)` no longer matches the actual boundary position — which means the masking range of labels is wrong too.

**② `-100` masks the prompt.** `-100` is the default `ignore_index` of PyTorch's `CrossEntropyLoss`, and HuggingFace models use that convention directly when computing loss internally. With the prompt masked, the model is supervised only on the conditional distribution "given the question, generate the answer," and does not waste capacity learning to restate the question.

**③ eos is appended at the end of the answer and participates in loss.** It sits in the supervised segment rather than the masked segment, so the model learns "emit eos when the answer is done." Miss this step and the deployed model keeps generating until `max_tokens`, which shows up as "it finished talking and is still rambling."

**④ Length and alignment must be strictly consistent.** `input_ids` and `labels` are built in lockstep by the same code, guaranteeing that `labels[i]` always corresponds to `input_ids[i]`. If the two were truncated by their own separate logic, an off-by-one misalignment would be easy to introduce — and labels off by one would teach the model to "predict the current token" rather than "predict the next token," so training looks normal while the model learns nothing at all.

### 3.6.1 Over-length samples need an explicit policy, not "concatenate first, then truncate uniformly"

The most intuitive way to write it is this, and it has one fatal flaw:

```python
# ✗ Wrong: looks symmetric and elegant, but silently produces zero-supervision samples on long prompts
input_ids = (prompt_ids + answer_ids)[:max_len]
labels    = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
```

**If the prompt itself is long enough to fill `max_len`, `labels` will be entirely `-100`.** I measured this code with a synthetic tokenizer: with the prompt encoding to 10 tokens and `max_len=8`, the output labels were eight `-100` values — **zero supervised tokens**.

The consequences come in two layers:

1. this sample contributes nothing to training and just burns compute;
2. worse, **if a whole batch of samples is like this, both the numerator and denominator of cross-entropy are 0 and loss becomes `nan`**. Once nan passes through backward into the parameters it never comes back out, every subsequent step is nan, and the weights are ruined. And all of this happens with no error message at all.

That same truncation can also cut off the eos — so the claim in item ③ above, that "the model learns to emit eos," does not hold for long samples.

So there has to be an **explicitly stated policy**:

```python
eos = self.tokenizer.eos_token_id

# Strategy 1: never truncate the prompt. Its tail is the template's assistant
# start marker; left-truncation breaks format consistency, right-truncation
# misaligns the supervision boundary — better to drop it than to build a bad sample.
budget = self.max_len - len(prompt_ids)
if budget < self.min_answer_tokens + 1:      # +1 leaves room for eos
    return None                               # reject

# Strategy 2: the answer may be truncated, but eos must be kept — otherwise the model never learns to stop.
keep = min(len(answer_ids), budget - 1)
supervised = answer_ids[:keep] + [eos]

input_ids = prompt_ids + supervised
labels    = [-100] * len(prompt_ids) + supervised
```

The two strategies lean in different directions, each for a reason:

| | Strategy | Why |
| --- | --- | --- |
| prompt | **never truncate; reject the whole sample if it does not fit** | The template structure must not be broken. Left-truncation drops `<\|im_start\|>user`, right-truncation drops the assistant start marker — either way the training input and the inference input are no longer the same format |
| answer | **may be truncated, but eos is forcibly kept** | A truncated answer is still a valid supervision signal (learning the beginning has value too), whereas eos is the only source of stopping behavior and must be preserved |

Measured confirmation (prompt=10 tokens, `min_answer_tokens=4`):

```
max_len=14 answer= 6t -> rejected (budget=4 < 5)
max_len=15 answer= 6t -> supervised= 5  eos at last position=True  answer truncated=True
max_len=20 answer=40t -> supervised=10  eos at last position=True  answer truncated=True
max_len=64 answer=40t -> supervised=41  eos at last position=True  answer truncated=False
```

**As long as a sample is accepted, eos is guaranteed to be at the last position of the supervised segment** — whether or not the answer was truncated. Only then does the claim in item ③ hold unconditionally.

The number of rejected samples is recorded in `manifest.json` as `rejected_overlength`, and the job fails outright when everything is rejected:

```python
if kept == 0:
    raise SystemExit(f"all {train_count} samples were rejected as over-length; raise MAX_LEN (currently {MAX_LEN})")
```

"Silently discard all the data and then exit successfully" is the worst possible outcome — downstream gets an empty dataset, and the training job fails somewhere completely different.

### 3.7 Why fixed-length padding

```python
pad = self.max_len - len(input_ids)
if pad > 0:
    input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad
    labels         = labels + [-100] * pad
    attention_mask = attention_mask + [0] * pad
```

The three arrays are each padded with something different, each for its own reason:

- `input_ids` is padded with `pad_token_id`: a placeholder, which must be a valid token id.
- `labels` is padded with `-100`: padding positions do not count toward loss.
- `attention_mask` is padded with `0`: padding positions are not attended to.

**All three must be padded together; missing any one of them is a bug**: miss the `-100` in labels and the model works hard at learning to "predict the pad token," with loss diluted by a mass of meaningless terms; miss the `0` in attention_mask and real tokens will attend to padding, polluting the representations.

So why fixed length rather than dynamic padding (padding each batch to that batch's longest)?

Because downstream is `iter_torch_batches`. It stacks the parquet columns directly into tensors, which requires every row in a batch to have the same length. Once lengths are fixed, stage 2's training loop is clean enough to be a single for:

```python
for batch in shard.iter_torch_batches(batch_size=...):
    output = model(input_ids=batch["input_ids"], ...)
```

No `collate_fn` needed, no cross-worker length mismatch to handle. **The price is wasted compute** — the padding tokens you added still have to go through the GPU. When the real length distribution has a heavy tail, the waste can exceed half. At that point the right move is not to switch to dynamic padding but to adopt **sequence packing** (splicing several short samples into one `max_len` sequence, separated by position_ids and the attention mask), which can push utilization close to 100%. This directory does not do packing, because it would make label construction an order of magnitude more complex, which is not a good fit for a first version.

One incidental detail:

```python
if self.tokenizer.pad_token_id is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
```

Many base models (the Qwen and Llama families) have no dedicated pad token. Reusing eos is the common practice, and its safety is doubly guaranteed by `attention_mask=0` and `labels=-100` — those positions are neither attended to nor counted toward loss, so which id fills them really does not matter.

### 3.8 Why artifacts are versioned with an atomic symlink switch

```python
def publish(version_dir: Path, root: Path) -> None:
    link = root / "current"
    staging = root / f".current-{RUN_ID}"
    staging.symlink_to(version_dir.name)
    os.replace(staging, link)
```

Never overwrite, only add a directory, then atomically swap the pointer. Three benefits:

1. **A re-run cannot break a training job in flight.** If you wrote over `tokenized/` in place, a training job that is reading the data would see half-new, half-old parquet.
2. **Rollback works.** The previous version directory is still there; just point back at it.
3. **Atomicity.** `symlink_to` creates a temporary link first, then `os.replace` swaps it atomically. `rename` is atomic within the same filesystem, so a reader always sees either the complete old link or the complete new link, never an intermediate state.

Note that `symlink_to(version_dir.name)` uses a **relative name** rather than an absolute path. That way the symlink stays valid when the whole PVC gets mounted at a different path (say, from `/mnt/cluster_storage` to `/shared`).

The version directory is created **exclusively**, not with `exist_ok=True`:

```python
try:
    version_dir.mkdir(parents=True)          # no exist_ok
except FileExistsError:
    raise SystemExit(f"{version_dir} already exists; pick a different RUN_ID instead of overwriting an existing data version")
```

`RUN_ID` has only second-level precision, and it can also be specified explicitly via an environment variable. With `exist_ok=True`, two launches within the same second, or a manually passed duplicate `RUN_ID`, would append into the same directory — at which point the promise of "never overwrite an old version" is void, and void very quietly (parquet filenames carry a random suffix, so files from both runs get mixed together and the row count inexplicably grows). **For "never overwrite" to hold, it has to be a raise.**

There are two limitations of this mechanism you must know about:

**① It depends on POSIX semantics.** Move to S3 and there is no atomic rename and no symlink, so the publishing protocol has to be redesigned (usually a pointer file plus readers tolerating brief inconsistency). This is also the main reason this directory chose an RWX PVC over object storage.

**② An atomic switch is not a snapshot.** Atomically replacing a directory entry only guarantees that a reader sees "the complete old link" or "the complete new link"; it **does not guarantee that a series of subsequent open calls belong to the same version**. A consumer could perfectly well resolve one file before the switch and another one after. So downstream must **resolve the symlink into an immutable path at the start and use that path throughout** — which is exactly what stage 2's `resolve_data_version()` and stage 3's `Path.resolve()` do. The symlink is just a convenient entry point for humans, not a concurrency-safe data contract.

### 3.9 The eval set: deterministic, disjoint, and the same version as the training data

The eval set is stored as plain-text jsonl rather than tokens, because what stage 3's vLLM needs is text (it applies the template and tokenizes itself). Three constraints matter more than "carve out a slice" itself:

```python
def pick_eval_hashes(ds, total: int) -> set:
    cap = max(1, int(total * EVAL_MAX_FRACTION))     # at most 25%
    rows = min(EVAL_ROWS, cap)
    selected = ds.sort("content_hash").take(rows)     # sorted by hash, deterministic
    return {row["content_hash"] for row in selected}

eval_hashes = pick_eval_hashes(deduped, total)
eval_ds  = deduped.filter(lambda r: r["content_hash"] in eval_hashes)
train_ds = deduped.filter(lambda r: r["content_hash"] not in eval_hashes)
```

**① Deterministic.** Sort by `content_hash` and take the first N, so the same input always carves out the same eval set. Using `take(N)` to grab "the first N rows" directly does not work — row order depends on file read order and chunking, so the result changes as soon as you change the concurrency, and the eval numbers from two experiments are not comparable.

**② Disjoint.** Use the hash set to **exclude** the eval portion from the training set. An early version merely "carved out a slice" without excluding it, so the model had seen the eval samples during training and stage 3's numbers were optimistically biased — with no way to estimate by how much. This costs one line of `filter`; there is no reason not to do it.

**③ There is a cap.** `EVAL_MAX_FRACTION` prevents a small dataset from being consumed entirely by `EVAL_ROWS`. The demo data has only 8 rows after deduplication, and `EVAL_ROWS=32` would put all of it into eval, leaving the training set empty. With the 25% cap it becomes 2 eval rows and 6 training rows. When the training set is empty after exclusion, fail outright:

```python
if train_count == 0:
    raise SystemExit("training set is empty after excluding the holdout; lower EVAL_ROWS or add more data")
```

**④ Put it in the same version directory as the training data, and write it only after tokenization succeeds.**

```python
tokenized.write_parquet(str(version_dir))                    # training data first
eval_count = dump_eval_set(eval_ds, version_dir / "eval.jsonl")   # then eval
```

Getting the order backwards produces an insidious failure: an early version wrote the eval set to `data/eval.jsonl` (unversioned, and written **before** tokenization). A failed data-preparation run would leave **a new eval set paired with the old training data** — because `tokenized/current` still points at the old version while eval.jsonl has already been overwritten. So stage 3 would evaluate a model trained on old data using a new holdout, the result is meaningless, and nothing indicates anything went wrong.

Once it lives in the version directory, `tokenized/current/eval.jsonl` and `tokenized/current/*.parquet` necessarily come from the same source. Stage 3's default `EVAL_PATH` was changed to `data/tokenized/current/eval.jsonl` accordingly.

## 4. Why the YAML is written this way

### 4.1 Why RayJob instead of RayCluster

KubeRay has three CRDs with different responsibilities:

| CRD | Used for | In this directory |
| --- | --- | --- |
| `RayCluster` | A long-lived general-purpose cluster, tasks submitted manually | Not used |
| `RayJob` | A one-shot job, can manage the cluster lifecycle along with it | Stages 1, 2, 3 |
| `RayService` | A resident service + Serve application | Stage 4 |

Data processing is a textbook one-shot job: when it finishes, the resources should go back. Using RayJob lets KubeRay handle "create cluster → submit job → wait for completion → destroy cluster," so you do not write the orchestration yourself.

```yaml
spec:
  entrypoint: python /home/ray/scripts/prepare_data.py
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 300
  activeDeadlineSeconds: 3600
```

- `shutdownAfterJobFinishes: true` — **the default is false**. Without turning it on explicitly, the RayCluster stays around after the job finishes, continuing to hold CPU and memory. This is the most common resource leak among newcomers.
- `ttlSecondsAfterFinished: 300` — only takes effect when the previous item is true. The 300 seconds leave a window for log collection: the moment the cluster is destroyed, Pod logs disappear with it, and if your logging pipeline has any lag, `ttl: 0` throws away the crime scene of a failed job.
- `activeDeadlineSeconds: 3600` — a timeout after which KubeRay actively terminates the job. Data processing hanging on some network download is a common failure; without this ceiling a job can hang all night.

### 4.2 Why the script is delivered via ConfigMap

```yaml
volumes:
  - name: scripts
    configMap:
      name: llm-e2e-scripts
```

The accompanying Makefile:

```makefile
scripts:
	kubectl -n $(NAMESPACE) create configmap llm-e2e-scripts \
		--from-file=prepare_data.py=10-data/prepare_data.py \
		... \
		--dry-run=client -o yaml | kubectl apply -f -
```

This is an **explicit trade-off**: changing one line of Python needs no image rebuild or push, which makes experimental iteration far faster. The price is traceability — you cannot work backwards from a Pod to which version of the code it actually ran.

`--dry-run=client -o yaml | kubectl apply -f -` turns `create configmap` into an idempotent upsert, so repeated execution does not fail with AlreadyExists.

**In production you should do the opposite**: `COPY` the scripts into the image and reference it by digest rather than tag, so that "image digest" uniquely determines the code version. The ConfigMap approach suits this directory's purpose (getting the pipeline running end to end), not an environment that needs auditing.

### 4.3 What the two volumes each do

```yaml
volumeMounts:
  - name: share
    mountPath: /shared
  - name: cluster-storage
    mountPath: /mnt/cluster_storage
  - name: scripts
    mountPath: /home/ray/scripts
volumes:
  - name: share
    emptyDir: {}
  - name: cluster-storage
    persistentVolumeClaim:
      claimName: llm-shared
```

- `share` (emptyDir, mounted at `/shared`): scratch space inside the Pod, preserved across container restarts and gone when the Pod is deleted. Used for exchanging files between containers in the same Pod, or staging logs somewhere that survives a container restart.
- `cluster-storage` (PVC, mounted at `/mnt/cluster_storage`): the **cross-stage handoff surface**. All four stages mount the same PVC; this is the only path along which data and models flow.

The two are not interchangeable: emptyDir does not span Pods, and a PVC is not a good place for high-frequency temporary files.

### 4.4 Why autoscaling is turned off

```yaml
workerGroupSpecs:
  - groupName: cpu-group
    replicas: 2
    minReplicas: 2
    maxReplicas: 2
```

Three equal values amount to turning off the Ray autoscaler. A batch job's resource demand is predictable (the data volume is known, the concurrency is configured), and enabling autoscaling only introduces jitter: right after the job starts, Ray has not submitted any tasks yet, so the autoscaler may scale down first and then have to scale back up, wasting another round of Pod startup.

An online service is the opposite scenario — the load is unpredictable, which is why stage 4's `minReplicas: 1 / maxReplicas: 4` is enabled.

### 4.5 Why the resources are set this way

```yaml
resources:
  requests:
    cpu: "4"
    memory: 12Gi
  limits:
    cpu: "8"
    memory: 16Gi
```

CPU request and limit differ, and memory request and limit differ too, so the Pod lands in **Burstable** QoS. This is a deliberate choice for batch work:

- **CPU is a compressible resource.** Exceeding the request only gets you throttled, not killed. So `limit > request` lets the job run faster while the node is idle and fall back to the request as a floor when the node is busy.
- **Memory is an incompressible resource.** Exceeding the limit means an immediate OOMKill. `request < limit` means you are betting that "most of the time the limit is not reached," and losing the bet means getting killed.

For batch work that bet pays off: if you get killed, just re-run, and meanwhile cluster bin-packing improves. **For a resident service it does not pay off** — which is why stage 4's GPU workers use `request == limit` to get Guaranteed QoS and avoid being OOM'd at peak traffic.

### 4.6 Why no probes are written

The data-prep container has no `livenessProbe` / `readinessProbe`. This is not an omission:

KubeRay injects them automatically when the user declares none. For Ray ≥ 2.53, what gets injected is the unified health endpoint `httpGet :52365/api/healthz`; earlier versions use an exec probe with wget against raylet health. **Writing them yourself is more likely to get them wrong** — the port, the path, and the "how long counts as not started yet" values have all been tuned by KubeRay against Ray's actual behavior.

Stage 4 is the only place that needs explicit probes; the reason is in that article — there is a "writing one makes things worse" trap there.

### 4.7 Why the HF token is optional

```yaml
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token
      key: hf_token
      optional: true
```

`optional: true` means the Pod starts normally when the Secret does not exist; the environment variable simply is not set. The Qwen2.5 family is public and needs no token; only pulling gated models like Llama does. Without `optional`, on a cluster where the Secret was never created the Pod gets stuck in `CreateContainerConfigError`, and the error message will not tell you "you actually don't need this."

## 5. How to verify it

```bash
make data
```

It first deletes the old RayJob of the same name (re-applying a completed RayJob does not re-run it, and that is exactly the semantics of "re-run"), then applies, then calls `wait_rayjob.sh` to poll `status.jobStatus`.

**Note that you cannot use `kubectl wait --for=condition=Complete rayjob/...`** — KubeRay v1.7.0's `RayJobStatus` has no `Conditions` field, so that condition never appears, and `kubectl wait` waits until it times out, making even a successful job "fail the wait." See [section 5 of the training article](../20-train/sft-training.en.md).

Confirm layer by layer:

```bash
# 1. The job succeeded
kubectl -n llm-pipeline get rayjob llm-data-prep \
  -o jsonpath='{.status.jobStatus}{"\n"}'   # expect SUCCEEDED

# 2. Look at the key log lines
kubectl -n llm-pipeline logs -l job-name=llm-data-prep-<suffix> | grep '\[data\]'
# expect to see:
#   raw N rows -> M rows after filtering and deduplication     with M < N
#   train K rows (rejected J over-length), eval E rows, the two are disjoint

# 3. Artifacts and manifest
kubectl -n llm-pipeline run artifact-check --rm -it --restart=Never \
  --image=busybox:1.36 --overrides='
{"spec":{"containers":[{"name":"c","image":"busybox:1.36",
 "command":["sh","-c","ls -l /mnt/cluster_storage/data/tokenized/ && ls /mnt/cluster_storage/data/tokenized/current/ && cat /mnt/cluster_storage/data/tokenized/current/manifest.json"],
 "volumeMounts":[{"name":"s","mountPath":"/mnt/cluster_storage"}]}],
 "volumes":[{"name":"s","persistentVolumeClaim":{"claimName":"llm-shared"}}]}}'
```

Criteria:
- `tokenized/current -> v-<RUN_ID>`, and the directory contains `*.parquet`, `eval.jsonl`, `manifest.json`;
- the row count after deduplication is **strictly less** than the raw row count (the demo data repeats 16 times, so it should be 8 rows after deduplication);
- in `manifest.json`, `train_rows + eval_rows == deduped_rows`, and `rejected_overlength` is explainable (it should be 0 for the demo data).

The second one is the most informative check: if the row count is the same before and after deduplication, then `content_hash` or `map_groups` did not take effect. The equation in the third one verifies both "disjoint" and "no sample got dropped inexplicably" at the same time.

## 6. The mistakes that bite hardest

1. **Template inconsistency** (section 3.5). Training assembles its own strings while inference goes through the chat template. Symptom: training is perfect, production talks nonsense.
2. **labels off by one**. Truncating them separately, or shifting manually while the model shifts internally as well. Symptom: loss does not drop, or drops in inexplicable ways.
3. **Forgetting eos, or having truncation cut it off** (section 3.6.1). Symptom: the model never stops, generating all the way to max_tokens.
4. **Long prompts producing zero-supervision samples** (section 3.6.1). Symptom: a whole batch's loss becomes nan and the weights are ruined afterwards; or it merely burns compute for nothing.
5. **Only padding one or two of the padding trio** (section 3.7). Symptom: abnormally low loss (diluted by padding outside of -100) or polluted representations.
6. **Hashing without a separator** (section 3.3). Symptom: extremely low-probability false deduplication, never exposed on small data.
7. **Forgetting `materialize()`**. Symptom: the job completes but takes twice as long, because the shuffle ran twice.

The first five all fall under "training converges anyway" silent errors, and the only way to catch them is to check in this stage. The least-effort check is decoding one sample back into text:

```python
# For debugging: confirm the masking boundary is correct
ids = row["input_ids"]; labels = row["labels"]
print("full input:", tokenizer.decode(ids))
print("supervised part:", tokenizer.decode([i for i, l in zip(ids, labels) if l != -100]))
```

The second line **must print only the answer content** (including eos), with no question and no template markers. That single check catches most of the cases in 1, 2, 3 and 4 above.

## 7. What this stage deliberately omits

Listed honestly, because all of them are needed in real settings:

| Omitted | Cost | When it becomes mandatory |
| --- | --- | --- |
| Per-file sha256 | `manifest.json` records row counts and configuration, but cannot prove "the file contents were not modified" | When you need auditing |
| Quality scoring / length distribution checks | Garbage samples make it into training anyway | When the data source is not under your control |
| PII / sensitive content filtering | Compliance risk | When using real user data |
| sequence packing | With long-tail data, more than half the compute may be wasted | When training cost becomes the bottleneck |
| Approximate deduplication | Change one punctuation mark and you bypass deduplication | At pretraining-corpus scale |
| Multi-turn conversation support | Only single-turn instruction-response is possible | When the target is a conversational assistant |
| Pinning the HF tokenizer version | `from_pretrained(BASE_MODEL)` does not pin a revision, so an upstream tokenizer update makes old and new data versions incomparable | When you need long-term reproducibility |

Already added (absent in early versions): **a disjoint holdout**, **deterministic splitting**, **exclusive creation of the version directory**,
**`manifest.json` provenance records**, **an explicit over-length policy with a rejection count**.

Of what remains, the two most worth adding first are **per-file sha256** and **pinning the tokenizer revision** — both are cheap, and together they answer "can this data be rebuilt." The approach is straightforward: when writing the manifest, compute a sha256 for every parquet file and record it alongside; recompute and compare before reading downstream. Pin the tokenizer with `from_pretrained(..., revision="<commit sha>")`.

## 8. What changes when you scale up

| Data scale | What needs to change |
| --- | --- |
| Thousands ~ tens of thousands of rows (this directory) | Nothing |
| Hundreds of thousands ~ millions | Raise `TOKENIZE_CONCURRENCY` close to the CPU core count; raise worker `replicas`; use `write_parquet(min_rows_per_file=...)` to avoid fragmented files |
| Tens of millions and up | Switch to object storage (giving up symlink semantics); switch deduplication to MinHash; consider packing; the `groupby` shuffle becomes the bottleneck and the partitioning strategy needs evaluation |

The effect of `MAX_LEN` is quadratic (attention complexity); going from 1024 to 4096 is not merely 4x the GPU memory. It also determines stage 2's GPU memory footprint and the lower bound of stage 4's `max_model_len` (`MAX_LEN + MAX_TOKENS ≤ max_model_len`), and `max_model_len` in turn determines the concurrency ceiling of the online service — changing it means doing the arithmetic across all four stages at once, and the formula is in [section 3.3 of the serving article](../40-serve/online-serving.en.md).

---

**Next up**: [Stage 2 — SFT training](../20-train/sft-training.en.md), covering how Ray Train organizes workers, why FSDP's state_dict must be gathered collectively, and why exporting in the HF directory format gives the inference side zero conversion work.