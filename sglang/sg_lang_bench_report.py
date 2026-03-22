#!/usr/bin/env python3
"""
Benchmark run summarizer + plots for AIPerf artifacts produced by benchmark.sh.

Directory layout expected:
  benchmarks/<MODEL>_<TIMESTAMP>/<suite>_c<C>_req<R>_in<I>_out<O>/
    aiperf_artifacts/profile_export_aiperf.json   (required)
    aiperf_artifacts/profile_export_aiperf.csv
    aiperf_artifacts/profile_export.jsonl         (optional, --per-request)
    aiperf_artifacts/server_metrics_export.csv    (optional, SGLang or vLLM)

Supports both SGLang (sglang:*) and vLLM (vllm:*) Prometheus metric namespaces.
Backend is auto-detected from the server_metrics_export.csv file.

SGLang metrics extracted (beyond the original set):
  sglang:gen_throughput          — server-side generation throughput gauge (tok/s)
  sglang:new_token_ratio         — ratio of new tokens vs cached in each batch
  sglang:decode_sum_seq_lens     — total decode tokens in-flight (queue depth proxy)
  sglang:num_retracted_reqs      — requests retracted due to over-scheduling
  sglang:spec_accept_rate        — speculative decoding acceptance rate
  sglang:spec_accept_length      — speculative decoding average accepted length
  sglang:evicted_tokens          — KV cache eviction pressure (counter total)
  sglang:cached_tokens           — prefix cache tokens saved (counter total)
  sglang:swa_token_usage         — SWA layer token usage (for SWA/hybrid models)

No hard dependency on pandas/matplotlib:
  - table + CSV export always work (stdlib only)
  - plotting requires matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_NAME_RE = re.compile(
    r"^(?P<suite>baseline|concurrency|longctx|stress)"
    r"_c(?P<c>\d+)_req(?P<req>\d+)_in(?P<in>\d+)_out(?P<out>\d+)$"
)

# Direction guidance for every numeric column (used in legend and report)
METRIC_DIRECTIONS: Dict[str, str] = {
    "ttft_avg_ms":           "lower is better",
    "ttft_p95_ms":           "lower is better",
    "itl_avg_ms":            "lower is better",
    "itl_p95_ms":            "lower is better",
    "req_lat_avg_ms":        "lower is better",
    "req_lat_p95_ms":        "lower is better",
    "e2e_p95_ms":            "lower is better",
    "prefill_p95_ms":        "lower is better",
    "decode_p95_ms":         "lower is better",
    "queue_p95_ms":          "lower is better",
    "rps":                   "higher is better",
    "out_tps":               "higher is better",
    "total_tps":             "higher is better",
    "prompt_tps":            "higher is better",
    "gen_tps":               "higher is better",
    "gen_tps_gauge":         "higher is better",
    "kv_p95":                "lower is safer (more headroom)",
    "kv_avg":                "lower is safer (more headroom)",
    "swa_usage_avg":         "lower is safer",
    "waiting_avg":           "lower is better",
    "retracted_avg":         "lower is better (0 = no over-scheduling)",
    "new_token_ratio_avg":   "higher = less caching overhead",
    "decode_inflight_avg":   "informational (total tokens in decode)",
    "pcache_hit":            "higher is better (workload dependent)",
    "cached_tokens_total":   "higher is better",
    "evicted_tokens_total":  "lower is better",
    "spec_accept_rate":      "higher is better (speculative decoding efficiency)",
    "spec_accept_length":    "higher is better (avg accepted draft tokens/step)",
    "pr_ttft_p99":           "lower is better",
    "pr_ttft_max":           "lower is better",
    "pr_req_p99":            "lower is better",
    "pr_req_max":            "lower is better",
    "pr_wait_p99":           "lower is better",
    "pr_wait_max":           "lower is better",
    "pr_osl_mismatch":       "lower is better",
    "little_L":              "higher = more in-system work (often queueing)",
    "q_pressure":            "higher = more queueing pressure",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    """Inclusive linear-interpolation percentile on a pre-sorted list."""
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    r = (p / 100.0) * (n - 1)
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    if lo == hi:
        return float(sorted_vals[lo])
    w = r - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "" or s.lower() in {"na", "n/a", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def _p(v: Optional[float], unit: str = "", digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{digits}f}{unit}"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-request JSONL parsing
# ---------------------------------------------------------------------------

@dataclass
class PerRequestSummary:
    n: int
    ttft_p95_ms: Optional[float]
    ttft_p99_ms: Optional[float]
    ttft_max_ms: Optional[float]
    req_lat_p95_ms: Optional[float]
    req_lat_p99_ms: Optional[float]
    req_lat_max_ms: Optional[float]
    http_wait_p95_ms: Optional[float]
    http_wait_p99_ms: Optional[float]
    http_wait_max_ms: Optional[float]
    osl_mismatch_rate: Optional[float]


def _read_profile_export_jsonl(
    path: Path, osl_mismatch_threshold_pct: float
) -> Optional[PerRequestSummary]:
    if not path.exists():
        return None

    ttft: List[float] = []
    req_lat: List[float] = []
    http_wait: List[float] = []
    mismatch = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            metrics = obj.get("metrics", {})
            if not isinstance(metrics, dict):
                continue

            def _get_ms(name: str) -> Optional[float]:
                m = metrics.get(name)
                if not isinstance(m, dict):
                    return None
                if (m.get("unit") or "") != "ms":
                    return None
                return _safe_float(m.get("value"))

            v_ttft = _get_ms("time_to_first_token")
            v_req  = _get_ms("request_latency")
            v_wait = _get_ms("http_req_waiting")

            if v_ttft is not None:
                ttft.append(v_ttft)
            if v_req is not None:
                req_lat.append(v_req)
            if v_wait is not None:
                http_wait.append(v_wait)

            m_osl = metrics.get("osl_mismatch_diff_pct")
            if isinstance(m_osl, dict) and (m_osl.get("unit") or "") == "%":
                diff = _safe_float(m_osl.get("value"))
                if diff is not None and abs(diff) >= osl_mismatch_threshold_pct:
                    mismatch += 1

            total += 1

    ttft.sort()
    req_lat.sort()
    http_wait.sort()

    def _pct(vals: List[float], p: float) -> Optional[float]:
        return _percentile(vals, p)

    return PerRequestSummary(
        n=total,
        ttft_p95_ms=_pct(ttft, 95),
        ttft_p99_ms=_pct(ttft, 99),
        ttft_max_ms=(ttft[-1] if ttft else None),
        req_lat_p95_ms=_pct(req_lat, 95),
        req_lat_p99_ms=_pct(req_lat, 99),
        req_lat_max_ms=(req_lat[-1] if req_lat else None),
        http_wait_p95_ms=_pct(http_wait, 95),
        http_wait_p99_ms=_pct(http_wait, 99),
        http_wait_max_ms=(http_wait[-1] if http_wait else None),
        osl_mismatch_rate=(mismatch / total) if total > 0 else None,
    )


# ---------------------------------------------------------------------------
# Server metrics CSV parsing (multi-section AIPerf format)
# ---------------------------------------------------------------------------

def _read_server_metrics_csv(path: Path) -> List[Dict[str, str]]:
    """
    AIPerf server_metrics_export.csv has multiple CSV sections separated by blank
    lines, each with its own header row.  Returns a flat list of all rows.
    """
    all_rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    def _blank(ln: str) -> bool:
        return (not ln.strip()) or ln.lstrip().startswith("#")

    i = 0
    while i < len(lines):
        while i < len(lines) and _blank(lines[i]):
            i += 1
        if i >= len(lines):
            break

        header = next(csv.reader([lines[i]]))
        i += 1

        data: List[str] = []
        while i < len(lines):
            ln = lines[i]
            if _blank(ln):
                j = i
                while j < len(lines) and _blank(lines[j]):
                    j += 1
                if j < len(lines) and lines[j].startswith("Endpoint,"):
                    break
                i += 1
                continue
            if ln.startswith("Endpoint,") and data:
                break
            data.append(ln)
            i += 1

        if data:
            for r in csv.DictReader(data, fieldnames=header):
                all_rows.append({k: (v or "") for k, v in r.items()})

    return all_rows


def _detect_backend(rows: List[Dict[str, str]]) -> str:
    for r in rows:
        m = r.get("Metric", "")
        if m.startswith("sglang:"):
            return "sglang"
        if m.startswith("vllm:"):
            return "vllm"
    return "unknown"


# ---------------------------------------------------------------------------
# Row finders
# ---------------------------------------------------------------------------

def _find_gauge(
    rows: List[Dict[str, str]],
    metric: str,
    model: Optional[str],
    extra: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "gauge":
            continue
        if r.get("Metric") != metric:
            continue
        if model and r.get("model_name") and r.get("model_name") != model:
            continue
        if extra and any(r.get(k, "") != v for k, v in extra.items()):
            continue
        return r
    return None


def _find_counter(
    rows: List[Dict[str, str]],
    metric: str,
    model: Optional[str],
    extra: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "counter":
            continue
        if r.get("Metric") != metric:
            continue
        if model and r.get("model_name") and r.get("model_name") != model:
            continue
        if extra and any(r.get(k, "") != v for k, v in extra.items()):
            continue
        return r
    return None


def _find_histogram(
    rows: List[Dict[str, str]],
    metric: str,
    model: Optional[str],
    extra: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "histogram":
            continue
        if r.get("Metric") != metric:
            continue
        if model and r.get("model_name") and r.get("model_name") != model:
            continue
        if extra and any(r.get(k, "") != v for k, v in extra.items()):
            continue
        return r
    return None


def _gauge_stats(
    rows: List[Dict[str, str]],
    metric: str,
    model: Optional[str],
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[float]]:
    row = _find_gauge(rows, metric, model, extra)
    if row is None:
        return {"avg": None, "p50": None, "p95": None, "p99": None}
    return {
        "avg": _safe_float(row.get("avg")),
        "p50": _safe_float(row.get("p50")),
        "p95": _safe_float(row.get("p95")),
        "p99": _safe_float(row.get("p99")),
    }


def _counter_total_rate(
    row: Optional[Dict[str, str]],
) -> Tuple[Optional[float], Optional[float]]:
    if not row:
        return None, None
    return _safe_float(row.get("total")), _safe_float(row.get("rate"))


def _hist_p50_p95_ms(
    row: Optional[Dict[str, str]],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (p50_ms, p95_ms) from a histogram row, converting seconds → ms."""
    if not row:
        return None, None
    p50 = _safe_float(row.get("p50_estimate") or row.get("p50"))
    p95 = _safe_float(row.get("p95_estimate") or row.get("p95"))
    unit = (row.get("Unit") or "").strip().lower()
    if unit in ("seconds", "s"):
        if p50 is not None:
            p50 *= 1000.0
        if p95 is not None:
            p95 *= 1000.0
    return p50, p95


# ---------------------------------------------------------------------------
# Backend-specific extraction
# ---------------------------------------------------------------------------

def _extract_sglang_metrics(
    rows: List[Dict[str, str]], model: Optional[str]
) -> Dict[str, Any]:
    """
    Extract server-side metrics from SGLang sglang:* Prometheus metrics.

    Metric mapping:
      kv_cache_usage_perc    → sglang:token_usage            (ratio ×100 → %)
      num_requests_running   → sglang:num_running_reqs
      num_requests_waiting   → sglang:num_queue_reqs
      prefix_cache           → sglang:cache_hit_rate          (ratio gauge, avg)
      prompt tokens/s        → sglang:prompt_tokens           (counter rate)
      gen tokens/s           → sglang:generation_tokens       (counter rate)
      gen throughput gauge   → sglang:gen_throughput          (gauge tok/s)
      e2e latency            → sglang:e2e_request_latency_seconds (histogram)
      ttft                   → sglang:time_to_first_token_seconds (histogram)
      itl                    → sglang:inter_token_latency_seconds (histogram)
      prefill latency        → sglang:per_stage_req_latency_seconds stage=prefill_waiting
      decode latency         → proxy via ITL p95 (no per-request decode histogram)
      queue latency          → sglang:queue_time_seconds      (histogram)
      new_token_ratio        → sglang:new_token_ratio         (gauge)
      decode_inflight        → sglang:decode_sum_seq_lens     (gauge, total tokens in decode)
      retracted_avg          → sglang:num_retracted_reqs      (gauge)
      spec_accept_rate       → sglang:spec_accept_rate        (gauge)
      spec_accept_length     → sglang:spec_accept_length      (gauge)
      evicted_tokens_total   → sglang:evicted_tokens          (counter total)
      cached_tokens_total    → sglang:cached_tokens           (counter total)
      swa_usage_avg          → sglang:swa_token_usage         (gauge)
    """
    out: Dict[str, Any] = {}

    # --- KV / token usage (ratio 0-1 → ×100 for %) ---
    kv = _gauge_stats(rows, "sglang:token_usage", model)
    out["kv_usage_avg"] = (kv["avg"] * 100.0) if kv["avg"] is not None else None
    out["kv_usage_p95"] = (kv["p95"] * 100.0) if kv["p95"] is not None else None

    # --- SWA layer token usage (hybrid/SWA models) ---
    swa = _gauge_stats(rows, "sglang:swa_token_usage", model)
    out["swa_usage_avg"] = (swa["avg"] * 100.0) if swa["avg"] is not None else None

    # --- Running / waiting / retracted requests ---
    out["running_avg"]  = _gauge_stats(rows, "sglang:num_running_reqs",  model)["avg"]
    out["waiting_avg"]  = _gauge_stats(rows, "sglang:num_queue_reqs",    model)["avg"]
    retracted           = _gauge_stats(rows, "sglang:num_retracted_reqs", model)
    out["retracted_avg"] = retracted["avg"]

    # --- Prefix cache hit rate ---
    out["prefix_cache_hit_ratio"] = _gauge_stats(rows, "sglang:cache_hit_rate", model)["avg"]

    # --- Cached tokens saved (counter total) ---
    cached_total, _ = _counter_total_rate(_find_counter(rows, "sglang:cached_tokens", model))
    out["cached_tokens_total"] = cached_total

    # --- KV eviction pressure (counter total) ---
    evicted_total, _ = _counter_total_rate(_find_counter(rows, "sglang:evicted_tokens", model))
    out["evicted_tokens_total"] = evicted_total

    # --- Token rates from counters ---
    _, prompt_rate = _counter_total_rate(_find_counter(rows, "sglang:prompt_tokens",     model))
    _, gen_rate    = _counter_total_rate(_find_counter(rows, "sglang:generation_tokens", model))
    out["prompt_tps"] = prompt_rate
    out["gen_tps"]    = gen_rate

    # --- Server-side generation throughput gauge (tok/s) ---
    out["gen_tps_gauge"] = _gauge_stats(rows, "sglang:gen_throughput", model)["avg"]

    # --- New token ratio (1.0 = all new tokens, <1 = cache reuse helping) ---
    out["new_token_ratio_avg"] = _gauge_stats(rows, "sglang:new_token_ratio", model)["avg"]

    # --- Decode in-flight token depth (sum of all sequence lengths in decode) ---
    out["decode_inflight_avg"] = _gauge_stats(rows, "sglang:decode_sum_seq_lens", model)["avg"]

    # --- Speculative decoding (zero when not used) ---
    out["spec_accept_rate"]   = _gauge_stats(rows, "sglang:spec_accept_rate",   model)["avg"]
    out["spec_accept_length"] = _gauge_stats(rows, "sglang:spec_accept_length", model)["avg"]

    # --- E2E latency histogram ---
    _, e2e_p95 = _hist_p50_p95_ms(_find_histogram(rows, "sglang:e2e_request_latency_seconds", model))
    out["e2e_p95_ms"] = e2e_p95

    # --- TTFT histogram (fallback for non-streaming runs) ---
    _, ttft_p95 = _hist_p50_p95_ms(_find_histogram(rows, "sglang:time_to_first_token_seconds", model))
    out["ttft_p95_ms_server"] = ttft_p95

    # --- ITL histogram (p50 ≈ typical ITL, p95 = tail) ---
    itl_p50, itl_p95 = _hist_p50_p95_ms(_find_histogram(rows, "sglang:inter_token_latency_seconds", model))
    out["itl_avg_ms_server"] = itl_p50
    out["itl_p95_ms_server"] = itl_p95

    # --- Prefill waiting latency (tp_rank=0 avoids double-counting multi-rank) ---
    pfrow = _find_histogram(
        rows, "sglang:per_stage_req_latency_seconds", model,
        extra={"stage": "prefill_waiting", "tp_rank": "0"},
    )
    _, prefill_p95 = _hist_p50_p95_ms(pfrow)
    out["prefill_p95_ms"] = prefill_p95

    # --- No separate decode histogram in SGLang → use ITL p95 as proxy ---
    out["decode_p95_ms"] = itl_p95

    # --- Queue time histogram (tp_rank=0) ---
    qrow = _find_histogram(rows, "sglang:queue_time_seconds", model, extra={"tp_rank": "0"})
    _, queue_p95 = _hist_p50_p95_ms(qrow)
    out["queue_p95_ms"] = queue_p95

    return out


def _extract_vllm_metrics(
    rows: List[Dict[str, str]], model: Optional[str]
) -> Dict[str, Any]:
    """Extract server-side metrics from vLLM vllm:* Prometheus metrics."""
    out: Dict[str, Any] = {}

    kv = _gauge_stats(rows, "vllm:kv_cache_usage_perc", model)
    out["kv_usage_avg"] = kv["avg"]
    out["kv_usage_p95"] = kv["p95"]
    out["swa_usage_avg"]      = None
    out["retracted_avg"]      = None
    out["new_token_ratio_avg"] = None
    out["decode_inflight_avg"] = None
    out["spec_accept_rate"]    = None
    out["spec_accept_length"]  = None
    out["cached_tokens_total"] = None
    out["evicted_tokens_total"] = None
    out["gen_tps_gauge"]       = None

    out["running_avg"] = _gauge_stats(rows, "vllm:num_requests_running", model)["avg"]
    out["waiting_avg"] = _gauge_stats(rows, "vllm:num_requests_waiting", model)["avg"]

    hits_total, _ = _counter_total_rate(_find_counter(rows, "vllm:prefix_cache_hits",    model))
    qry_total,  _ = _counter_total_rate(_find_counter(rows, "vllm:prefix_cache_queries", model))
    out["prefix_cache_hit_ratio"] = (
        hits_total / qry_total if hits_total is not None and qry_total else None
    )
    out["cached_tokens_total"]  = hits_total  # reuse hits as "cached tokens" proxy
    out["evicted_tokens_total"] = None

    _, prompt_rate = _counter_total_rate(_find_counter(rows, "vllm:prompt_tokens",     model))
    _, gen_rate    = _counter_total_rate(_find_counter(rows, "vllm:generation_tokens", model))
    out["prompt_tps"] = prompt_rate
    out["gen_tps"]    = gen_rate

    _, e2e_p95    = _hist_p50_p95_ms(_find_histogram(rows, "vllm:e2e_request_latency_seconds",   model))
    _, prefill_p95 = _hist_p50_p95_ms(_find_histogram(rows, "vllm:request_prefill_time_seconds", model))
    _, decode_p95  = _hist_p50_p95_ms(_find_histogram(rows, "vllm:request_decode_time_seconds",  model))
    _, queue_p95   = _hist_p50_p95_ms(_find_histogram(rows, "vllm:request_queue_time_seconds",   model))

    out["e2e_p95_ms"]     = e2e_p95
    out["prefill_p95_ms"] = prefill_p95
    out["decode_p95_ms"]  = decode_p95
    out["queue_p95_ms"]   = queue_p95
    out["ttft_p95_ms_server"] = None
    out["itl_avg_ms_server"]  = None
    out["itl_p95_ms_server"]  = None

    return out


# ---------------------------------------------------------------------------
# Run data model
# ---------------------------------------------------------------------------

def _metric(profile: Dict[str, Any], key: str, field: str = "avg") -> Optional[float]:
    obj = profile.get(key)
    if not isinstance(obj, dict):
        return None
    return _safe_float(obj.get(field))


@dataclass(frozen=True)
class RunId:
    model_timestamp: str
    run_name: str

    @property
    def suite(self) -> str:
        m = RUN_NAME_RE.match(self.run_name)
        return m.group("suite") if m else "unknown"

    @property
    def concurrency(self) -> Optional[int]:
        m = RUN_NAME_RE.match(self.run_name)
        return int(m.group("c")) if m else None

    @property
    def request_count(self) -> Optional[int]:
        m = RUN_NAME_RE.match(self.run_name)
        return int(m.group("req")) if m else None

    @property
    def input_tokens(self) -> Optional[int]:
        m = RUN_NAME_RE.match(self.run_name)
        return int(m.group("in")) if m else None

    @property
    def output_tokens(self) -> Optional[int]:
        m = RUN_NAME_RE.match(self.run_name)
        return int(m.group("out")) if m else None

    def short(self) -> str:
        return f"{self.model_timestamp}/{self.run_name}"


@dataclass
class RunSummary:
    rid: RunId
    path: Path
    url: Optional[str]
    model: Optional[str]
    backend: str                          # "sglang" | "vllm" | "unknown"
    streaming: Optional[bool]

    # AIPerf profile metrics (always present)
    req_throughput_rps: Optional[float]   # requests/sec
    out_tput_tps:       Optional[float]   # output tokens/sec
    total_tput_tps:     Optional[float]   # total tokens/sec (prompt+output)
    ttft_avg_ms:        Optional[float]
    ttft_p95_ms:        Optional[float]
    itl_avg_ms:         Optional[float]
    itl_p95_ms:         Optional[float]
    req_lat_avg_ms:     Optional[float]
    req_lat_p95_ms:     Optional[float]

    # Server-side metrics (from server_metrics_export.csv)
    kv_usage_avg:           Optional[float]   # % of KV pool used (avg)
    kv_usage_p95:           Optional[float]   # % of KV pool used (p95)
    swa_usage_avg:          Optional[float]   # SWA layer token usage % (SGLang)
    running_avg:            Optional[float]   # avg running requests
    waiting_avg:            Optional[float]   # avg queued requests
    retracted_avg:          Optional[float]   # avg retracted requests (SGLang)
    new_token_ratio_avg:    Optional[float]   # ratio new tokens in batch (SGLang)
    decode_inflight_avg:    Optional[float]   # avg total tokens in decode (SGLang)
    prefix_cache_hit_ratio: Optional[float]   # prefix cache hit ratio (0-1)
    cached_tokens_total:    Optional[float]   # total cached tokens (SGLang counter)
    evicted_tokens_total:   Optional[float]   # total evicted tokens (SGLang counter)
    prompt_tps:             Optional[float]   # prompt tokens/sec (counter rate)
    gen_tps:                Optional[float]   # generation tokens/sec (counter rate)
    gen_tps_gauge:          Optional[float]   # gen throughput gauge (SGLang)
    spec_accept_rate:       Optional[float]   # speculative decoding accept rate (SGLang)
    spec_accept_length:     Optional[float]   # speculative decoding avg accept length (SGLang)
    e2e_p95_ms:             Optional[float]   # server-side E2E latency p95
    prefill_p95_ms:         Optional[float]   # prefill phase latency p95
    decode_p95_ms:          Optional[float]   # decode latency p95 (or ITL proxy)
    queue_p95_ms:           Optional[float]   # queue wait p95

    # Per-request tails (populated with --per-request)
    pr_ttft_p99_ms:       Optional[float]
    pr_ttft_max_ms:       Optional[float]
    pr_req_lat_p99_ms:    Optional[float]
    pr_req_lat_max_ms:    Optional[float]
    pr_http_wait_p99_ms:  Optional[float]
    pr_http_wait_max_ms:  Optional[float]
    pr_osl_mismatch_rate: Optional[float]

    # Derived
    little_L:   Optional[float]   # L = rps × (req_lat_avg_sec)  [Little's Law]
    q_pressure: Optional[float]   # waiting_avg / concurrency


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def load_run(
    run_dir: Path,
    per_request: bool,
    osl_mismatch_threshold_pct: float,
) -> Optional[RunSummary]:
    arts         = run_dir / "aiperf_artifacts"
    profile_json = arts / "profile_export_aiperf.json"
    server_csv   = arts / "server_metrics_export.csv"
    jsonl        = arts / "profile_export.jsonl"

    if not profile_json.exists():
        return None

    profile = _read_json(profile_json)

    # Pull config metadata
    ic  = profile.get("input_config", {}) or {}
    ep  = ic.get("endpoint", {}) or {}
    urls  = ep.get("urls",        []) or []
    names = ep.get("model_names", []) or []
    url      = urls[0]  if urls  else None
    model    = names[0] if names else None
    streaming = ep.get("streaming") if isinstance(ep.get("streaming"), bool) else None

    rid = RunId(model_timestamp=run_dir.parent.name, run_name=run_dir.name)

    # AIPerf profile scalars
    req_throughput_rps = _metric(profile, "request_throughput",    "avg")
    out_tput_tps       = _metric(profile, "output_token_throughput","avg")
    total_tput_tps     = _metric(profile, "total_token_throughput", "avg")
    ttft_avg_ms        = _metric(profile, "time_to_first_token",    "avg")
    ttft_p95_ms        = _metric(profile, "time_to_first_token",    "p95")
    itl_avg_ms         = _metric(profile, "inter_token_latency",    "avg")
    itl_p95_ms         = _metric(profile, "inter_token_latency",    "p95")
    req_lat_avg_ms     = _metric(profile, "request_latency",        "avg")
    req_lat_p95_ms     = _metric(profile, "request_latency",        "p95")

    # Server metrics
    backend = "unknown"
    sm: Dict[str, Any] = {}

    if server_csv.exists():
        rows    = _read_server_metrics_csv(server_csv)
        backend = _detect_backend(rows)
        sm      = (_extract_sglang_metrics(rows, model)
                   if backend == "sglang"
                   else _extract_vllm_metrics(rows, model))

        # Prefer server histogram TTFT/ITL for SGLang non-streaming runs
        if backend == "sglang":
            if ttft_p95_ms is None:
                ttft_p95_ms = sm.get("ttft_p95_ms_server")
            if itl_avg_ms is None:
                itl_avg_ms = sm.get("itl_avg_ms_server")
            if itl_p95_ms is None:
                itl_p95_ms = sm.get("itl_p95_ms_server")

    # Per-request tails
    pr = None
    if per_request and jsonl.exists():
        pr = _read_profile_export_jsonl(jsonl, osl_mismatch_threshold_pct)

    # Derived metrics
    little_L = (
        req_throughput_rps * (req_lat_avg_ms / 1000.0)
        if req_throughput_rps is not None and req_lat_avg_ms is not None
        else None
    )
    q_pressure = (
        sm.get("waiting_avg") / float(rid.concurrency)
        if rid.concurrency and sm.get("waiting_avg") is not None
        else None
    )

    return RunSummary(
        rid=rid, path=run_dir, url=url, model=model,
        backend=backend, streaming=streaming,
        req_throughput_rps=req_throughput_rps,
        out_tput_tps=out_tput_tps,
        total_tput_tps=total_tput_tps,
        ttft_avg_ms=ttft_avg_ms,
        ttft_p95_ms=ttft_p95_ms,
        itl_avg_ms=itl_avg_ms,
        itl_p95_ms=itl_p95_ms,
        req_lat_avg_ms=req_lat_avg_ms,
        req_lat_p95_ms=req_lat_p95_ms,
        kv_usage_avg=sm.get("kv_usage_avg"),
        kv_usage_p95=sm.get("kv_usage_p95"),
        swa_usage_avg=sm.get("swa_usage_avg"),
        running_avg=sm.get("running_avg"),
        waiting_avg=sm.get("waiting_avg"),
        retracted_avg=sm.get("retracted_avg"),
        new_token_ratio_avg=sm.get("new_token_ratio_avg"),
        decode_inflight_avg=sm.get("decode_inflight_avg"),
        prefix_cache_hit_ratio=sm.get("prefix_cache_hit_ratio"),
        cached_tokens_total=sm.get("cached_tokens_total"),
        evicted_tokens_total=sm.get("evicted_tokens_total"),
        prompt_tps=sm.get("prompt_tps"),
        gen_tps=sm.get("gen_tps"),
        gen_tps_gauge=sm.get("gen_tps_gauge"),
        spec_accept_rate=sm.get("spec_accept_rate"),
        spec_accept_length=sm.get("spec_accept_length"),
        e2e_p95_ms=sm.get("e2e_p95_ms"),
        prefill_p95_ms=sm.get("prefill_p95_ms"),
        decode_p95_ms=sm.get("decode_p95_ms"),
        queue_p95_ms=sm.get("queue_p95_ms"),
        pr_ttft_p99_ms=(pr.ttft_p99_ms       if pr else None),
        pr_ttft_max_ms=(pr.ttft_max_ms        if pr else None),
        pr_req_lat_p99_ms=(pr.req_lat_p99_ms  if pr else None),
        pr_req_lat_max_ms=(pr.req_lat_max_ms  if pr else None),
        pr_http_wait_p99_ms=(pr.http_wait_p99_ms if pr else None),
        pr_http_wait_max_ms=(pr.http_wait_max_ms if pr else None),
        pr_osl_mismatch_rate=(pr.osl_mismatch_rate if pr else None),
        little_L=little_L,
        q_pressure=q_pressure,
    )


# ---------------------------------------------------------------------------
# Discovery & filtering
# ---------------------------------------------------------------------------

def iter_runs(benchmarks_dir: Path) -> Iterable[Path]:
    if not benchmarks_dir.exists():
        return
    for model_ts in sorted(benchmarks_dir.iterdir()):
        if not model_ts.is_dir():
            continue
        for run in sorted(model_ts.iterdir()):
            if not run.is_dir():
                continue
            if (run / "aiperf_artifacts" / "profile_export_aiperf.json").exists():
                yield run


def filter_runs(
    runs: List[RunSummary],
    model_ts: Optional[str],
    suite: Optional[str],
) -> List[RunSummary]:
    out = runs
    if model_ts:
        out = [r for r in out if r.rid.model_timestamp == model_ts]
    if suite:
        out = [r for r in out if r.rid.suite == suite]
    return out


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

# Column definitions: (header, getter, digits)
_COLUMNS = [
    ("run",              lambda r: r.rid.short(),                      None),
    ("suite",            lambda r: r.rid.suite,                        None),
    ("c",                lambda r: str(r.rid.concurrency or ""),       None),
    ("req",              lambda r: str(r.rid.request_count or ""),     None),
    ("in",               lambda r: str(r.rid.input_tokens or ""),      None),
    ("out",              lambda r: str(r.rid.output_tokens or ""),     None),
    ("backend",          lambda r: r.backend,                          None),
    ("rps",              lambda r: r.req_throughput_rps,               3),
    ("out_tps",          lambda r: r.out_tput_tps,                     2),
    ("ttft_avg_ms",      lambda r: r.ttft_avg_ms,                      2),
    ("ttft_p95_ms",      lambda r: r.ttft_p95_ms,                      2),
    ("itl_avg_ms",       lambda r: r.itl_avg_ms,                       3),
    ("itl_p95_ms",       lambda r: r.itl_p95_ms,                       3),
    ("req_lat_avg_ms",   lambda r: r.req_lat_avg_ms,                   2),
    ("req_lat_p95_ms",   lambda r: r.req_lat_p95_ms,                   2),
    ("kv_avg%",          lambda r: r.kv_usage_avg,                     2),
    ("kv_p95%",          lambda r: r.kv_usage_p95,                     2),
    ("swa_avg%",         lambda r: r.swa_usage_avg,                    2),
    ("running_avg",      lambda r: r.running_avg,                      2),
    ("waiting_avg",      lambda r: r.waiting_avg,                      2),
    ("retracted_avg",    lambda r: r.retracted_avg,                    2),
    ("new_tok_ratio",    lambda r: r.new_token_ratio_avg,              3),
    ("decode_inflight",  lambda r: r.decode_inflight_avg,              1),
    ("prefill_p95_ms",   lambda r: r.prefill_p95_ms,                   2),
    ("decode_p95_ms",    lambda r: r.decode_p95_ms,                    2),
    ("e2e_p95_ms",       lambda r: r.e2e_p95_ms,                       2),
    ("queue_p95_ms",     lambda r: r.queue_p95_ms,                     2),
    ("prompt_tps",       lambda r: r.prompt_tps,                       2),
    ("gen_tps",          lambda r: r.gen_tps,                          2),
    ("gen_tps_gauge",    lambda r: r.gen_tps_gauge,                    2),
    ("pcache_hit",       lambda r: r.prefix_cache_hit_ratio,           4),
    ("cached_tok",       lambda r: r.cached_tokens_total,              0),
    ("evicted_tok",      lambda r: r.evicted_tokens_total,             0),
    ("spec_accept_rate", lambda r: r.spec_accept_rate,                 4),
    ("spec_accept_len",  lambda r: r.spec_accept_length,               2),
    ("pr_ttft_p99",      lambda r: r.pr_ttft_p99_ms,                   2),
    ("pr_ttft_max",      lambda r: r.pr_ttft_max_ms,                   2),
    ("pr_req_p99",       lambda r: r.pr_req_lat_p99_ms,                2),
    ("pr_req_max",       lambda r: r.pr_req_lat_max_ms,                2),
    ("pr_wait_p99",      lambda r: r.pr_http_wait_p99_ms,              2),
    ("pr_wait_max",      lambda r: r.pr_http_wait_max_ms,              2),
    ("pr_osl_mismatch",  lambda r: r.pr_osl_mismatch_rate,             4),
    ("little_L",         lambda r: r.little_L,                         2),
    ("q_pressure",       lambda r: r.q_pressure,                       3),
]


def _cell(getter, r: RunSummary, digits: Optional[int]) -> str:
    if digits is None:
        return getter(r) or ""
    v = getter(r)
    return _p(v, "", digits)


def print_table(runs: List[RunSummary], limit: Optional[int]) -> None:
    rows = runs[:limit] if limit else runs
    headers = [h for h, _, _ in _COLUMNS]
    table   = [[_cell(g, r, d) for _, g, d in _COLUMNS] for r in rows]

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "  "

    def fmt(vals: List[str]) -> str:
        return sep.join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in table:
        print(fmt(row))

    backends = sorted({r.backend for r in rows})
    print(f"\nDetected backend(s): {', '.join(backends)}")
    print("\nLegend:")
    print("  Latencies (ms)   : lower is better")
    print("  Throughputs      : higher is better")
    print("  kv_%             : lower = more KV headroom  (SGLang: token_usage×100, vLLM: kv_cache_usage_perc)")
    print("  decode_p95_ms    : SGLang proxy via ITL p95 (no per-request decode histogram)")
    print("  gen_tps_gauge    : SGLang sglang:gen_throughput gauge (tok/s, complement to gen_tps counter rate)")
    print("  new_tok_ratio    : fraction of tokens in each batch that are new (vs cached); SGLang only")
    print("  decode_inflight  : sum of all decode sequence lengths in-flight; SGLang only")
    print("  retracted_avg    : avg requests retracted due to over-scheduling; SGLang only")
    print("  spec_*           : speculative decoding metrics; zero/N/A when spec decoding is off")
    print("  cached_tok       : total prefix-cache token hits (SGLang counter / vLLM hits proxy)")
    print("  evicted_tok      : total KV cache evictions; SGLang only")
    if any(r.pr_osl_mismatch_rate is not None for r in rows):
        print("  pr_osl_mismatch  : fraction of requests with |osl_diff| >= threshold (--per-request only)")


def write_csv(runs: List[RunSummary], out_path: Path) -> None:
    headers = [h for h, _, _ in _COLUMNS]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in runs:
            w.writerow([_cell(g, r, d) for _, g, d in _COLUMNS])
    print(f"Wrote CSV: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_runs(runs: List[RunSummary], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"NOTE: matplotlib needed for plots ({e}). pip install matplotlib", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    runs2 = sorted(
        [r for r in runs if r.rid.concurrency is not None],
        key=lambda r: (r.rid.suite, r.rid.concurrency or 0, r.rid.input_tokens or 0),
    )
    backends = {r.backend for r in runs2} - {"unknown"}
    be_note  = f"  [{', '.join(sorted(backends))}]" if backends else ""

    def _scatter(
        getter,
        ylabel: str,
        fname: str,
        note: str = "",
        secondary_getter=None,
        secondary_label: str = "",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        suites = sorted({r.rid.suite for r in runs2})
        plotted = False
        for s in suites:
            xs, ys, labels = [], [], []
            for r in runs2:
                if r.rid.suite != s:
                    continue
                y = getter(r)
                if y is None or r.rid.concurrency is None:
                    continue
                xs.append(r.rid.concurrency)
                ys.append(y)
                labels.append(f"c={r.rid.concurrency}")
            if xs:
                plotted = True
                line, = ax.plot(xs, ys, marker="o", linestyle="-", label=s)
                for x, yv, lbl in zip(xs, ys, labels):
                    ax.annotate(
                        lbl, (x, yv),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=line.get_color(), alpha=0.8,
                    )
        # Optional secondary series (e.g. gauge vs counter-rate)
        if secondary_getter is not None:
            for s in suites:
                xs2, ys2 = [], []
                for r in runs2:
                    if r.rid.suite != s:
                        continue
                    y = secondary_getter(r)
                    if y is None or r.rid.concurrency is None:
                        continue
                    xs2.append(r.rid.concurrency)
                    ys2.append(y)
                if xs2:
                    plotted = True
                    ax.plot(xs2, ys2, marker="s", linestyle="--",
                            label=f"{s} ({secondary_label})", alpha=0.6)

        title = ylabel + be_note
        if note:
            title += f"\n{note}"
        ax.set_xlabel("Concurrency (aiperf in-flight requests)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No numeric data available for this metric\nin the selected runs.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)

        path = out_dir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

    # --- Latency ---
    _scatter(lambda r: r.ttft_p95_ms,
             "TTFT p95 (ms)", "ttft_p95_vs_concurrency.png",
             note="(AIPerf JSON or SGLang server histogram)")
    _scatter(lambda r: r.ttft_avg_ms,
             "TTFT avg (ms)", "ttft_avg_vs_concurrency.png")
    _scatter(lambda r: r.itl_p95_ms,
             "ITL p95 (ms)", "itl_p95_vs_concurrency.png",
             note="(AIPerf JSON or SGLang server histogram)")
    _scatter(lambda r: r.itl_avg_ms,
             "ITL avg/p50 (ms)", "itl_avg_vs_concurrency.png")
    _scatter(lambda r: r.req_lat_p95_ms,
             "Request latency p95 (ms) [aiperf]", "req_lat_p95_vs_concurrency.png")
    _scatter(lambda r: r.req_lat_avg_ms,
             "Request latency avg (ms) [aiperf]", "req_lat_avg_vs_concurrency.png")
    _scatter(lambda r: r.e2e_p95_ms,
             "E2E request latency p95 (ms) [server]", "e2e_p95_vs_concurrency.png")
    _scatter(lambda r: r.prefill_p95_ms,
             "Prefill/prefill-wait p95 (ms)", "prefill_p95_vs_concurrency.png",
             note="(SGLang: per_stage_req_latency prefill_waiting | vLLM: request_prefill_time)")
    _scatter(lambda r: r.decode_p95_ms,
             "Decode p95 (ms)", "decode_p95_vs_concurrency.png",
             note="(SGLang: proxy via ITL p95 | vLLM: request_decode_time)")
    _scatter(lambda r: r.queue_p95_ms,
             "Queue time p95 (ms) [server]", "queue_p95_vs_concurrency.png")

    # --- Throughput ---
    _scatter(lambda r: r.out_tput_tps,
             "Output token throughput (tok/s) [aiperf]", "out_tps_vs_concurrency.png")
    _scatter(lambda r: r.gen_tps,
             "Generation token rate (tok/s) [counter rate]", "gen_tps_vs_concurrency.png",
             secondary_getter=lambda r: r.gen_tps_gauge,
             secondary_label="gauge")
    _scatter(lambda r: r.prompt_tps,
             "Prompt token rate (tok/s) [counter rate]", "prompt_tps_vs_concurrency.png")

    # --- KV / memory ---
    _scatter(lambda r: r.kv_usage_p95,
             "KV/token usage p95 (%)", "kv_usage_p95_vs_concurrency.png",
             note="(SGLang: token_usage×100 | vLLM: kv_cache_usage_perc)")
    _scatter(lambda r: r.kv_usage_avg,
             "KV/token usage avg (%)", "kv_usage_avg_vs_concurrency.png")
    _scatter(lambda r: r.swa_usage_avg,
             "SWA token usage avg (%) [SGLang]", "swa_usage_avg_vs_concurrency.png")
    _scatter(lambda r: r.evicted_tokens_total,
             "Total KV cache evictions [SGLang]", "evicted_tokens_vs_concurrency.png",
             note="(sglang:evicted_tokens counter total during benchmark)")

    # --- Queue / scheduling ---
    _scatter(lambda r: r.waiting_avg,
             "Avg waiting requests", "waiting_avg_vs_concurrency.png",
             note="(SGLang: num_queue_reqs | vLLM: num_requests_waiting)")
    _scatter(lambda r: r.retracted_avg,
             "Avg retracted requests [SGLang]", "retracted_avg_vs_concurrency.png",
             note="(sglang:num_retracted_reqs — increase --schedule-conservativeness if > 0)")
    _scatter(lambda r: r.decode_inflight_avg,
             "Avg total decode tokens in-flight [SGLang]", "decode_inflight_vs_concurrency.png",
             note="(sglang:decode_sum_seq_lens)")
    _scatter(lambda r: r.new_token_ratio_avg,
             "New token ratio avg [SGLang]", "new_token_ratio_vs_concurrency.png",
             note="(sglang:new_token_ratio — fraction of batch tokens that are new vs cached)")

    # --- Cache ---
    _scatter(lambda r: r.prefix_cache_hit_ratio,
             "Prefix cache hit ratio", "prefix_cache_hit_ratio_vs_concurrency.png",
             note="(SGLang: cache_hit_rate gauge | vLLM: hits/queries)")
    _scatter(lambda r: r.cached_tokens_total,
             "Total cached tokens [prefix cache]", "cached_tokens_vs_concurrency.png")

    # --- Speculative decoding ---
    _scatter(lambda r: r.spec_accept_rate,
             "Spec decoding accept rate [SGLang]", "spec_accept_rate_vs_concurrency.png",
             note="(sglang:spec_accept_rate — zero/N/A when spec decoding is disabled)")
    _scatter(lambda r: r.spec_accept_length,
             "Spec decoding avg accept length [SGLang]", "spec_accept_length_vs_concurrency.png")

    # --- Derived ---
    _scatter(lambda r: r.little_L,
             "Little's Law  L = rps × lat_avg_sec", "littles_law_vs_concurrency.png")
    _scatter(lambda r: r.q_pressure,
             "Queue pressure  waiting_avg / concurrency", "q_pressure_vs_concurrency.png")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _default_plot_out(benchmarks_dir: str, model_ts: Optional[str]) -> Path:
    b = Path(benchmarks_dir)
    return (b / model_ts / "_plots") if model_ts else (b / "_plots")


def write_markdown_report(
    runs: List[RunSummary],
    out_path: Path,
    title: str,
    include_per_request: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_c = sorted([r for r in runs if r.rid.concurrency is not None],
                  key=lambda r: r.rid.concurrency or 0)

    knee: Optional[RunSummary] = None
    for i in range(1, len(by_c)):
        prev, cur = by_c[i - 1], by_c[i]
        if prev.ttft_p95_ms and cur.ttft_p95_ms and prev.ttft_p95_ms > 0:
            if (cur.ttft_p95_ms / prev.ttft_p95_ms) >= 2.0:
                knee = cur
                break

    best_tput = max(
        (r for r in runs if r.out_tput_tps is not None),
        key=lambda r: r.out_tput_tps or -1, default=None,
    )
    best_lat = min(
        (r for r in runs if r.ttft_p95_ms is not None),
        key=lambda r: r.ttft_p95_ms or float("inf"), default=None,
    )
    most_evictions = max(
        (r for r in runs if r.evicted_tokens_total is not None),
        key=lambda r: r.evicted_tokens_total or 0, default=None,
    )

    backends = sorted({r.backend for r in runs})

    lines: List[str] = [
        f"## {title}",
        "",
        f"**Detected backend(s):** {', '.join(backends)}",
        "",
        "### Metric directions",
        "",
        "- **Lower is better**: TTFT, ITL, request latency, prefill/decode/e2e/queue p95, waiting/retracted/evicted.",
        "- **Higher is better**: throughput (rps, tokens/sec), cache hit ratio, spec accept rate/length.",
        "- **KV/token usage %**: lower = more headroom. SGLang: `token_usage×100`. vLLM: `kv_cache_usage_perc`.",
        "- **decode_p95_ms (SGLang)**: proxy via ITL p95 — SGLang has no per-request decode histogram.",
        "- **gen_tps vs gen_tps_gauge**: counter rate (`generation_tokens`) vs real-time gauge (`gen_throughput`). Both should track; divergence indicates idle periods.",
        "- **new_token_ratio**: closer to 1.0 = all new tokens (no prefix reuse); lower = heavy cache reuse.",
        "- **retracted_avg**: requests retracted due to over-scheduling. Non-zero → increase `--schedule-conservativeness`.",
        "- **spec_*** metrics: only populated when speculative decoding (`--speculative-algorithm`) is enabled.",
        "",
        "### Highlights",
        "",
    ]

    if best_lat:
        lines.append(f"- **Best TTFT p95**: `{best_lat.rid.run_name}` (c={best_lat.rid.concurrency}) → {_p(best_lat.ttft_p95_ms, ' ms', 2)}")
    if best_tput:
        lines.append(f"- **Best output throughput**: `{best_tput.rid.run_name}` (c={best_tput.rid.concurrency}) → {_p(best_tput.out_tput_tps, ' tok/s', 2)}")
    if knee:
        lines.append(f"- **Saturation knee (heuristic)**: around c={knee.rid.concurrency} (`{knee.rid.run_name}`) where TTFT p95 jumps ≥2× vs previous concurrency.")
    if most_evictions and (most_evictions.evicted_tokens_total or 0) > 0:
        lines.append(f"- **Most KV evictions**: `{most_evictions.rid.run_name}` → {_p(most_evictions.evicted_tokens_total, ' tokens', 0)} evicted (KV pressure warning).")

    lines += [
        "",
        "### Plots generated",
        "",
    ]

    all_plots = [
        ("ttft_p95_vs_concurrency.png",        "TTFT p95"),
        ("ttft_avg_vs_concurrency.png",         "TTFT avg"),
        ("itl_p95_vs_concurrency.png",          "ITL p95"),
        ("itl_avg_vs_concurrency.png",          "ITL avg/p50"),
        ("req_lat_p95_vs_concurrency.png",      "Request latency p95 [aiperf]"),
        ("e2e_p95_vs_concurrency.png",          "E2E latency p95 [server]"),
        ("prefill_p95_vs_concurrency.png",      "Prefill latency p95"),
        ("decode_p95_vs_concurrency.png",       "Decode latency p95"),
        ("queue_p95_vs_concurrency.png",        "Queue time p95"),
        ("out_tps_vs_concurrency.png",          "Output throughput [aiperf]"),
        ("gen_tps_vs_concurrency.png",          "Gen token rate (counter + gauge)"),
        ("prompt_tps_vs_concurrency.png",       "Prompt token rate"),
        ("kv_usage_p95_vs_concurrency.png",     "KV usage p95 %"),
        ("kv_usage_avg_vs_concurrency.png",     "KV usage avg %"),
        ("swa_usage_avg_vs_concurrency.png",    "SWA usage avg % [SGLang]"),
        ("evicted_tokens_vs_concurrency.png",   "KV evictions [SGLang]"),
        ("waiting_avg_vs_concurrency.png",      "Avg waiting requests"),
        ("retracted_avg_vs_concurrency.png",    "Avg retracted requests [SGLang]"),
        ("decode_inflight_vs_concurrency.png",  "Decode tokens in-flight [SGLang]"),
        ("new_token_ratio_vs_concurrency.png",  "New token ratio [SGLang]"),
        ("prefix_cache_hit_ratio_vs_concurrency.png", "Prefix cache hit ratio"),
        ("cached_tokens_vs_concurrency.png",    "Total cached tokens"),
        ("spec_accept_rate_vs_concurrency.png", "Spec accept rate [SGLang]"),
        ("spec_accept_length_vs_concurrency.png","Spec accept length [SGLang]"),
        ("littles_law_vs_concurrency.png",      "Little's Law L"),
        ("q_pressure_vs_concurrency.png",       "Queue pressure"),
    ]
    for fname, label in all_plots:
        lines.append(f"- `{fname}` — {label}")

    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def _sort_key(r: RunSummary, sort_keys: List[str]) -> tuple:
    key: List[Any] = []
    for k in sort_keys:
        if k == "suite":
            key.append(r.rid.suite)
        elif k in {"c", "concurrency"}:
            key.append(r.rid.concurrency if r.rid.concurrency is not None else 10**9)
        elif k == "in":
            key.append(r.rid.input_tokens  if r.rid.input_tokens  is not None else 10**9)
        elif k == "out":
            key.append(r.rid.output_tokens if r.rid.output_tokens is not None else 10**9)
        elif k == "req":
            key.append(r.rid.request_count if r.rid.request_count is not None else 10**9)
        elif k == "ttft_p95":
            key.append(r.ttft_p95_ms if r.ttft_p95_ms is not None else float("inf"))
        elif k == "out_tps":
            key.append(-(r.out_tput_tps if r.out_tput_tps is not None else -float("inf")))
        elif k == "rps":
            key.append(-(r.req_throughput_rps if r.req_throughput_rps is not None else -float("inf")))
        elif k == "kv_p95":
            key.append(r.kv_usage_p95 if r.kv_usage_p95 is not None else float("inf"))
        elif k == "e2e_p95":
            key.append(r.e2e_p95_ms if r.e2e_p95_ms is not None else float("inf"))
        else:
            key.append("")
    key.append(r.rid.short())
    return tuple(key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize and plot AIPerf benchmark runs.\n"
            "Supports SGLang (sglang:*) and vLLM (vllm:*) — backend auto-detected."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--benchmarks-dir", default="benchmarks",
                    help="Root directory containing <MODEL>_<TIMESTAMP>/ subdirectories.")
    ap.add_argument("--model-ts", default=None,
                    help="Filter to one model+timestamp dir, e.g. gpt-oss-20b_20260322_080523.")
    ap.add_argument("--suite", default=None,
                    choices=[None, "baseline", "concurrency", "longctx", "stress"],
                    help="Filter to one benchmark suite.")
    ap.add_argument("--limit", type=int, default=50,
                    help="Max rows to print in the summary table.")
    ap.add_argument("--sort", default="suite,c",
                    help="Comma-separated sort keys: suite, c, in, out, req, ttft_p95, out_tps, rps, kv_p95, e2e_p95.")
    ap.add_argument("--per-request", action="store_true",
                    help="Parse profile_export.jsonl for per-request p99/max tails and OSL mismatch rate (slower).")
    ap.add_argument("--osl-mismatch-threshold-pct", type=float, default=5.0,
                    help="Abs %% threshold for OSL mismatch counting (--per-request only).")
    ap.add_argument("--plot", action="store_true",
                    help="Generate matplotlib scatter plots (requires matplotlib).")
    ap.add_argument("--by-suite", action="store_true",
                    help="Generate per-suite plots under <plot-out>/<suite>/ in addition to all/.")
    ap.add_argument("--plot-out", default=None,
                    help="Plot output dir. Default: <benchmarks-dir>/<model-ts>/_plots.")
    ap.add_argument("--report", action="store_true",
                    help="Write a Markdown summary report into the plot directory.")
    ap.add_argument("--report-name", default="report.md",
                    help="Filename for the Markdown report.")
    ap.add_argument("--csv-out", default=None,
                    help="Write the summary table to this CSV file path.")

    args = ap.parse_args(argv)
    bdir = Path(args.benchmarks_dir)

    all_runs: List[RunSummary] = []
    for run_dir in iter_runs(bdir):
        s = load_run(
            run_dir,
            per_request=bool(args.per_request),
            osl_mismatch_threshold_pct=float(args.osl_mismatch_threshold_pct),
        )
        if s:
            all_runs.append(s)

    runs = filter_runs(all_runs, args.model_ts, args.suite)

    sort_keys = [k.strip() for k in str(args.sort).split(",") if k.strip()]
    runs.sort(key=lambda r: _sort_key(r, sort_keys))

    if not runs:
        print(f"No runs found under {bdir.resolve()}")
        return 2

    print_table(runs, args.limit)

    if args.csv_out:
        write_csv(runs, Path(args.csv_out))

    if args.plot:
        plot_out = Path(args.plot_out) if args.plot_out else _default_plot_out(args.benchmarks_dir, args.model_ts)
        if args.by_suite:
            plot_runs(runs, plot_out / "all")
            for s in sorted({r.rid.suite for r in runs if r.rid.suite != "unknown"}):
                plot_runs([r for r in runs if r.rid.suite == s], plot_out / s)
        else:
            plot_runs(runs, plot_out)

    if args.report:
        out_dir = Path(args.plot_out) if args.plot_out else _default_plot_out(args.benchmarks_dir, args.model_ts)
        if args.by_suite:
            write_markdown_report(
                runs, out_dir / "all" / args.report_name,
                title=f"Benchmark report ({args.model_ts or 'all'} / all)",
                include_per_request=bool(args.per_request),
            )
            for s in sorted({r.rid.suite for r in runs if r.rid.suite != "unknown"}):
                write_markdown_report(
                    [r for r in runs if r.rid.suite == s],
                    out_dir / s / args.report_name,
                    title=f"Benchmark report ({args.model_ts or 'all'} / {s})",
                    include_per_request=bool(args.per_request),
                )
        else:
            write_markdown_report(
                runs, out_dir / args.report_name,
                title=f"Benchmark report ({args.model_ts or 'all runs'})",
                include_per_request=bool(args.per_request),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())