#!/usr/bin/env bash
#
# Stage 4 acceptance check: four checks against the OpenAI-compatible API exposed by RayService
#   1. /v1/models contains model_id exactly
#   2. the non-streaming response has real content and a finish_reason
#   3. the streaming response is complete (content deltas present, [DONE] present, no
#      error events, transfer did not fail)
#   4. the stream arrives **incrementally** (judged by arrival time, not by counting lines)
#
# Why check 4 measures time: counting SSE lines cannot distinguish "arriving one by one"
# from "buffered by a proxy and delivered all at once". The line count is identical in both
# cases, but in the latter the TTFT equals the whole generation time.
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-pipeline}"
RAYSERVICE="${RAYSERVICE:-llm-serve}"
SERVICE="${SERVICE:-${RAYSERVICE}-serve-svc}"
MODEL_ID="${MODEL_ID:-sft-qwen}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
PROMPT="${PROMPT:-Does a Pod entering Running mean it can take traffic?}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
MAX_TOKENS="${MAX_TOKENS:-128}"
# Criterion for incremental delivery: the lower bound (in seconds) on the time span from
# the first content chunk to the last one.
# In a fully buffered response every line arrives at once, so the span is close to 0.
MIN_STREAM_SPAN="${MIN_STREAM_SPAN:-0.15}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RESET=$'\033[0m'

log_info() { printf '%s[INFO]%s %s\n' "$GREEN" "$RESET" "$*"; }
log_warning() { printf '%s[WARN]%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
log_error() { printf '%s[ERROR]%s %s\n' "$RED" "$RESET" "$*" >&2; }

usage() {
  cat <<'EOF'
Usage: smoke_test.sh [-h]

Environment variables:
  NAMESPACE        namespace                        (default llm-pipeline)
  RAYSERVICE       RayService name                  (default llm-serve)
  SERVICE          Serve Service name               (default <RAYSERVICE>-serve-svc)
  MODEL_ID         model_id from LLMConfig          (default sft-qwen)
  LOCAL_PORT       local port-forward port          (default 8000)
  PROMPT           question used for the test
  WAIT_TIMEOUT     seconds to wait for readiness    (default 900)
  MIN_STREAM_SPAN  min time span for incremental delivery, seconds (default 0.15)

Prerequisites: kubectl points at the target cluster, and 40-serve/rayservice-llm.yaml has been applied.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for tool in kubectl curl python3; do
  command -v "$tool" >/dev/null || { log_error "Missing dependency: $tool"; exit 1; }
done

WORKDIR="$(mktemp -d)"
PF_PID=""
cleanup() {
  if [[ -n "$PF_PID" ]] && kill -0 "$PF_PID" 2>/dev/null; then
    kill "$PF_PID" 2>/dev/null || true
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

# The request body is built with json.dumps rather than string interpolation —
# interpolation produces invalid JSON as soon as PROMPT contains a quote, a backslash or
# a newline.
make_body() {
  local stream="$1"
  MODEL_ID="$MODEL_ID" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" STREAM="$stream" \
    python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL_ID"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
    "stream": os.environ["STREAM"] == "true",
}, ensure_ascii=False))'
}

log_info "Waiting for RayService ${RAYSERVICE} to become ready (up to ${WAIT_TIMEOUT}s)"
# RayService really does have a Ready condition (rayservice_types.go: RayServiceReady = "Ready"),
# so kubectl wait works here — unlike the RayJob case.
if ! kubectl -n "$NAMESPACE" wait --for=condition=Ready \
  "rayservice/${RAYSERVICE}" --timeout="${WAIT_TIMEOUT}s"; then
  log_error "RayService did not become ready"
  kubectl -n "$NAMESPACE" get rayservice "$RAYSERVICE" \
    -o jsonpath='{.status.activeServiceStatus.applicationStatuses}' || true
  exit 1
fi

log_info "Setting up port-forward localhost:${LOCAL_PORT} -> ${SERVICE}:8000"
kubectl -n "$NAMESPACE" port-forward "svc/${SERVICE}" "${LOCAL_PORT}:8000" \
  >"$WORKDIR/pf.log" 2>&1 &
PF_PID=$!

BASE="http://localhost:${LOCAL_PORT}"
ready=0
for _ in $(seq 1 30); do
  # The port-forward process must be alive; otherwise some other local service may be
  # answering on this port.
  if ! kill -0 "$PF_PID" 2>/dev/null; then
    log_error "The port-forward process has exited, see the log below"
    cat "$WORKDIR/pf.log" >&2
    exit 1
  fi
  if curl -fsS --max-time 5 "${BASE}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { log_error "/v1/models is still unreachable after the port-forward was established"; exit 1; }

# ---------- Check 1: /v1/models ----------
log_info "Check 1/4: /v1/models contains ${MODEL_ID} exactly"
if ! curl -fsS --max-time 15 "${BASE}/v1/models" -o "$WORKDIR/models.json"; then
  log_error "The /v1/models request failed"
  exit 1
fi
cat >"$WORKDIR/check_models.py" <<'PYEOF'
import json, os, sys

data = json.load(open(os.environ["WORKDIR"] + "/models.json"))
ids = [m.get("id") for m in data.get("data", [])]
if os.environ["MODEL_ID"] not in ids:
    print(f"Registered models: {ids}", file=sys.stderr)
    sys.exit(1)
PYEOF

if ! MODEL_ID="$MODEL_ID" WORKDIR="$WORKDIR" python3 "$WORKDIR/check_models.py"; then
  log_error "/v1/models does not contain ${MODEL_ID} exactly"
  exit 1
fi
log_info "model_id ${MODEL_ID} is registered"

# ---------- Check 2: non-streaming ----------
log_info "Check 2/4: the non-streaming response has real content"
if ! curl -fsS --max-time "$CURL_MAX_TIME" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(make_body false)" -o "$WORKDIR/nonstream.json"; then
  log_error "The non-streaming request failed (HTTP error or timeout)"
  exit 1
fi
cat >"$WORKDIR/check_nonstream.py" <<'PYEOF'
import json, os, sys

data = json.load(open(os.environ["WORKDIR"] + "/nonstream.json"))
if "error" in data:
    sys.exit("An error was returned: " + json.dumps(data["error"], ensure_ascii=False))
choices = data.get("choices") or []
if not choices:
    sys.exit("No choices: " + json.dumps(data, ensure_ascii=False)[:300])
content = (choices[0].get("message") or {}).get("content")
if not isinstance(content, str) or not content.strip():
    sys.exit("content is empty or not a string")
finish = choices[0].get("finish_reason")
if not finish:
    sys.exit("finish_reason is missing")
print(f"  content {len(content)} chars, finish_reason={finish}")
print("  preview:", content[:120].replace("\n", " "))
PYEOF

if ! WORKDIR="$WORKDIR" python3 "$WORKDIR/check_nonstream.py"; then
  log_error "The non-streaming response is not acceptable"
  exit 1
fi

# ---------- Checks 3 and 4: streaming ----------
log_info "Checks 3/4 and 4/4: stream completeness and incremental delivery"

# Use **one** python process that timestamps as it reads. We must not fork a python per
# line to get the time: interpreter startup is a few tens of milliseconds, which inflates
# every inter-line gap and makes buffering detection useless.
cat >"$WORKDIR/read_stream.py" <<'PYEOF'
import json, os, sys, time

span_min = float(os.environ["MIN_STREAM_SPAN"])
events, done, protocol_frames = [], False, 0

for line in sys.stdin:
    now = time.monotonic()
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[len("data: "):].strip()
    if payload == "[DONE]":
        done = True
        continue
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        sys.exit(f"Unparseable SSE payload: {payload[:200]}")
    if "error" in obj:
        sys.exit(f"An error event appeared in the stream: {obj['error']}")
    got_content = False
    for choice in obj.get("choices", []):
        piece = (choice.get("delta") or {}).get("content")
        if isinstance(piece, str) and piece:
            events.append((now, piece))
            got_content = True
    if not got_content:
        protocol_frames += 1

if not done:
    sys.exit("The stream ended without [DONE]; the response is incomplete (it may have been cut off midway)")
if not events:
    sys.exit(f"No content delta at all, only {protocol_frames} protocol frames")

text = "".join(piece for _, piece in events)
print(f"  {len(events)} content chunks, {len(text)} chars total, {protocol_frames} protocol frames")
print("  preview:", text[:120].replace("\n", " "))

if len(events) < 2:
    print("  Only one content chunk, skipping the incremental judgement (normal for a short answer)")
    sys.exit(0)
span = events[-1][0] - events[0][0]
print(f"  span from first to last chunk {span:.3f}s (lower bound {span_min}s)")
if span < span_min:
    sys.exit(
        f"The content chunks arrived almost simultaneously ({span:.3f}s); the response was very "
        "likely buffered by an intermediate layer; check the gateway's proxy_buffering / buffer filter"
    )
PYEOF

set +e
curl -sS -N --no-buffer --max-time "$CURL_MAX_TIME" "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(make_body true)" 2>"$WORKDIR/stream.err" \
  | MIN_STREAM_SPAN="$MIN_STREAM_SPAN" python3 "$WORKDIR/read_stream.py"
rc=("${PIPESTATUS[@]}")
set -e
curl_rc="${rc[0]}"
parse_rc="${rc[1]}"

# curl's exit code must be checked separately. The earlier version used
# `| grep -c ... || true`, which swallowed transfer failures (such as exit 28 on timeout)
# entirely.
if [[ "$curl_rc" -ne 0 ]]; then
  log_error "The streaming transfer failed, curl exit code ${curl_rc}"
  head -5 "$WORKDIR/stream.err" >&2 || true
  exit 1
fi
if [[ "$parse_rc" -ne 0 ]]; then
  log_error "The streaming response is not acceptable"
  exit 1
fi

log_info "All four checks passed"