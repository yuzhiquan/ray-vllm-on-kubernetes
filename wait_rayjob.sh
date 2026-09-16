#!/usr/bin/env bash
#
# Wait until a RayJob has genuinely finished.
#
# Why not kubectl wait --for=condition=Complete:
# KubeRay v1.7.0's RayJobStatus has **no Conditions field** (only jobStatus and
# jobDeploymentStatus), so that condition never appears and kubectl wait just
# blocks until the timeout. And jobDeploymentStatus=Complete does not mean
# success either — IsJobDeploymentTerminal returns true for both Complete and
# Failed. The real success signal is status.jobStatus == SUCCEEDED.
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-pipeline}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
RESET=$'\033[0m'

log_info() { printf '%s[INFO]%s %s\n' "$GREEN" "$RESET" "$*"; }
log_warning() { printf '%s[WARN]%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
log_error() { printf '%s[ERROR]%s %s\n' "$RED" "$RESET" "$*" >&2; }

usage() {
  cat <<'EOF'
Usage: wait_rayjob.sh <rayjob-name> <timeout-seconds>

Environment variables:
  NAMESPACE      namespace            (default llm-pipeline)
  POLL_INTERVAL  poll interval in sec (default 10)

Exit codes: 0=SUCCEEDED, 1=FAILED/STOPPED/timeout
EOF
}

if [[ $# -ne 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 1
fi

NAME="$1"
TIMEOUT="$2"

field() {
  kubectl -n "$NAMESPACE" get rayjob "$NAME" -o "jsonpath={.status.$1}" 2>/dev/null || true
}

log_info "Waiting for RayJob ${NAME} (up to ${TIMEOUT}s, polling every ${POLL_INTERVAL}s)"
deadline=$((SECONDS + TIMEOUT))
last=""

while true; do
  job_status="$(field jobStatus)"
  deploy_status="$(field jobDeploymentStatus)"
  current="${deploy_status:-?}/${job_status:-?}"

  if [[ "$current" != "$last" ]]; then
    log_info "status: jobDeploymentStatus=${deploy_status:-<empty>} jobStatus=${job_status:-<empty>}"
    last="$current"
  fi

  case "$job_status" in
    SUCCEEDED)
      log_info "RayJob ${NAME} succeeded"
      exit 0
      ;;
    FAILED | STOPPED)
      log_error "RayJob ${NAME} ended with ${job_status}"
      log_error "reason : $(field reason)"
      log_error "message: $(field message)"
      exit 1
      ;;
  esac

  # Failures that happen before the job is even submitted (image cannot be
  # pulled, admission rejected, cluster never comes up) show up only in
  # jobDeploymentStatus; jobStatus stays empty.
  if [[ "$deploy_status" == "Failed" ]]; then
    log_error "RayJob ${NAME} failed to deploy (no jobStatus produced yet)"
    log_error "reason : $(field reason)"
    log_error "message: $(field message)"
    exit 1
  fi

  if ((SECONDS > deadline)); then
    log_error "Timed out waiting for RayJob ${NAME} (${TIMEOUT}s)"
    log_error "last status: jobDeploymentStatus=${deploy_status:-<empty>} jobStatus=${job_status:-<empty>}"
    kubectl -n "$NAMESPACE" get rayjob "$NAME" -o wide || true
    exit 1
  fi

  sleep "$POLL_INTERVAL"
done