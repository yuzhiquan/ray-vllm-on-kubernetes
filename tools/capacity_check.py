#!/usr/bin/env python3
"""Read the engine parameters out of a manifest, redo the capacity accounting, and
flag configurations that contradict themselves.

No GPU, no Ray, no model weights required — it only reads YAML plus a few model
structure constants.

Usage:
    python3 tools/capacity_check.py --manifest 40-serve/rayservice-llm.yaml \\
        --model qwen3-8b --card-gb 80
    python3 tools/capacity_check.py --manifest 40-serve/rayservice-llm.yaml \\
        --model qwen2.5-0.5b --card-gb 24

Model constants come from each model's config.json on HuggingFace. The weight byte
count for Qwen3-8B is the `weight_resident_bytes` reported by the calculator in
https://github.com/bojieli/ai-infra-book (`calc.py forward --model qwen3-8b`); the
other two are parameter-count estimates and are labelled as such in the output.
"""

import argparse
import json
import sys

GIB = 1024 ** 3

MODELS = {
    "qwen3-8b": dict(
        layers=36, kv_heads=8, head_dim=128, weight_bytes=16_381_470_720,
        source="calc.py forward --model qwen3-8b (weight_resident_bytes)"),
    "qwen2.5-7b": dict(
        layers=28, kv_heads=4, head_dim=128, weight_bytes=15_231_233_024,
        source="HF config.json + approx. 7.62B params x 2 bytes"),
    "qwen2.5-0.5b": dict(
        layers=24, kv_heads=2, head_dim=64, weight_bytes=988_000_000,
        source="HF config.json + approx. 494M params x 2 bytes"),
}


def kv_bytes_per_token(m, dtype_bytes=2):
    """2 (K and V) x layers x KV heads x head_dim x bytes per element.

    Note this uses the KV head count (`num_key_value_heads`), NOT the attention
    head count. Modern models use GQA, where several query heads share one group
    of KV heads; substituting the attention head count overestimates KV memory by
    the grouping factor.
    """
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * dtype_bytes


def read_engine_config(path):
    """Pull the engine parameters out of the RayService manifest's embedded
    serveConfigV2."""
    import yaml

    doc = yaml.safe_load(open(path, encoding="utf-8"))
    inner = yaml.safe_load(doc["spec"]["serveConfigV2"])
    cfg = inner["applications"][0]["args"]["llm_configs"][0]
    engine = cfg["engine_kwargs"]
    deploy = cfg.get("deployment_config", {})
    auto = deploy.get("autoscaling_config", {})
    return dict(
        max_model_len=engine["max_model_len"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        tensor_parallel_size=engine.get("tensor_parallel_size", 1),
        target_ongoing_requests=auto.get("target_ongoing_requests"),
        max_ongoing_requests=deploy.get("max_ongoing_requests"),
        min_replicas=auto.get("min_replicas"),
        max_replicas=auto.get("max_replicas"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--card-gb", type=float, required=True,
                    help="nominal memory of one card, decimal GB as vendors quote it "
                         "(80 means 80GB)")
    ap.add_argument("--avg-len", type=int, nargs="*", default=[512, 1024, 2048, 4096],
                    help="candidate average sequence lengths (prompt + output)")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args()

    m = MODELS[args.model]
    cfg = read_engine_config(args.manifest)

    per_token = kv_bytes_per_token(m)
    cards = cfg["tensor_parallel_size"]
    # Tensor parallelism splits weights and KV across cards, so total memory scales
    total_card_bytes = args.card_gb * 10 ** 9 * cards
    reserved = total_card_bytes * cfg["gpu_memory_utilization"]
    kv_pool = reserved - m["weight_bytes"]
    token_budget = kv_pool / per_token if kv_pool > 0 else 0
    worst_case_seq_bytes = per_token * cfg["max_model_len"]

    findings = []
    if kv_pool <= 0:
        findings.append(
            f"weights ({m['weight_bytes']/GIB:.1f} GiB) already exceed the reserved "
            f"memory ({reserved/GIB:.1f} GiB); the engine will not start. Add cards "
            f"(tensor_parallel_size) or use a smaller model.")
    target = cfg["target_ongoing_requests"]
    if target and token_budget:
        # Worst case: assume every in-flight request fills max_model_len
        worst_case_concurrency = token_budget / cfg["max_model_len"]
        if target > worst_case_concurrency:
            findings.append(
                f"target_ongoing_requests={target} exceeds the worst-case concurrency "
                f"ceiling of {worst_case_concurrency:.1f} (every request filling "
                f"max_model_len) — over-subscribed by about "
                f"{target/worst_case_concurrency:.1f}x. A burst of long requests will "
                f"exhaust the KV pool, which shows up as queueing and latency "
                f"degradation, with no error.")
        elif worst_case_concurrency / target > 20:
            findings.append(
                f"target_ongoing_requests={target} is very conservative relative to "
                f"capacity: even in the worst case {worst_case_concurrency:.0f} "
                f"sequences fit, {worst_case_concurrency/target:.0f}x headroom. "
                f"Autoscaling will trigger while the KV pool is still nearly empty.")
    if cfg["max_ongoing_requests"] and target and cfg["max_ongoing_requests"] <= target:
        findings.append(
            f"max_ongoing_requests={cfg['max_ongoing_requests']} is not greater than "
            f"target_ongoing_requests={target}, leaving no buffer while scaling out.")

    out = dict(
        model=args.model, model_source=m["source"],
        layers=m["layers"], kv_heads=m["kv_heads"], head_dim=m["head_dim"],
        cards=cards, kv_bytes_per_token=per_token,
        weight_gib=round(m["weight_bytes"] / GIB, 2),
        reserved_gib=round(reserved / GIB, 2),
        kv_pool_gib=round(kv_pool / GIB, 2),
        token_budget=int(token_budget),
        max_model_len=cfg["max_model_len"],
        worst_case_seq_mib=round(worst_case_seq_bytes / 1024 ** 2, 1),
        concurrency_by_avg_len={n: int(token_budget / n) for n in args.avg_len},
        config=cfg, findings=findings,
    )

    if args.format == "json":
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1 if findings else 0

    print(f"model {args.model}  ({m['layers']} layers, {m['kv_heads']} KV heads, "
          f"head_dim {m['head_dim']})")
    print(f"  structure and weights from: {m['source']}")
    print(f"  KV per token = 2 x {m['layers']} x {m['kv_heads']} x {m['head_dim']} x 2B "
          f"= {per_token:,} B = {per_token/1024:.0f} KiB")
    print()
    print(f"manifest {args.manifest}")
    print(f"  max_model_len          {cfg['max_model_len']:,}")
    print(f"  gpu_memory_utilization {cfg['gpu_memory_utilization']}")
    print(f"  tensor_parallel_size   {cards}")
    print(f"  target/max_ongoing     {target} / {cfg['max_ongoing_requests']}")
    print(f"  replicas               {cfg['min_replicas']} .. {cfg['max_replicas']}")
    print()
    print(f"capacity ({args.card_gb:g}GB x {cards} card(s))")
    print(f"  reserved memory  {reserved/GIB:8.1f} GiB")
    print(f"  minus weights    {m['weight_bytes']/GIB:8.1f} GiB")
    print(f"  KV pool          {kv_pool/GIB:8.1f} GiB   (activations and CUDA context "
          f"still to be subtracted)")
    print(f"  token budget     {token_budget/1000:8.0f} K tokens")
    print(f"  per-seq ceiling  {worst_case_seq_bytes/1024**2:8.1f} MiB  (a sequence "
          f"filling max_model_len)")
    print()
    print("  average sequence length -> concurrent sequences")
    for n in args.avg_len:
        print(f"    {n:>5} tokens  ->  {int(token_budget/n):>6}")
    print()
    if findings:
        print("findings:")
        for f in findings:
            print(f"  ! {f}")
    else:
        print("findings: none")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())