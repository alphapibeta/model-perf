#!/usr/bin/env python3
"""
Benchmark run summarizer + simple plots for AIPerf artifacts produced by benchmark.sh.

Works with directories like:
  benchmarks/<MODEL>_<TIMESTAMP>/<suite>_c<...>_req<...>_in<...>_out<...>/
    aiperf_artifacts/profile_export_aiperf.json
    aiperf_artifacts/profile_export_aiperf.csv
    aiperf_artifacts/server_metrics_export.csv

Supports both vLLM (vllm:*) and SGLang (sglang:*) server metrics.
Auto-detects backend from the metrics CSV.

No hard dependency on pandas/matplotlib:
  - summaries always work (stdlib only)
  - plotting requires matplotlib (and optionally pandas)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_NAME_RE = re.compile(
    r"^(?P<suite>baseline|concurrency|longctx|stress)_c(?P<c>\d+)_req(?P<req>\d+)_in(?P<in>\d+)_out(?P<out>\d+)$"
)

METRIC_DIRECTIONS: Dict[str, str] = {
    "ttft_avg_ms": "lower is better",
    "ttft_p95_ms": "lower is better",
    "itl_avg_ms": "lower is better",
    "itl_p95_ms": "lower is better",
    "req_lat_avg_ms": "lower is better",
    "req_lat_p95_ms": "lower is better",
    "e2e_p95_ms": "lower is better",
    "prefill_p95_ms": "lower is better",
    "decode_p95_ms": "lower is better",
    "queue_p95_ms": "lower is better",
    "rps": "higher is better",
    "out_tps": "higher is better",
    "total_tps": "higher is better",
    "prompt_tps": "higher is better",
    "gen_tps": "higher is better",
    "kv_p95": "lower is safer (more headroom)",
    "waiting_avg": "lower is better",
    "pcache_hit": "higher is better (workload dependent)",
    "pr_ttft_p99": "lower is better",
    "pr_ttft_max": "lower is better",
    "pr_req_p99": "lower is better",
    "pr_req_max": "lower is better",
    "pr_wait_p99": "lower is better",
    "pr_wait_max": "lower is better",
    "pr_osl_mismatch": "lower is better",
    "little_L": "higher means more in-system work (often queueing)",
    "q_pressure": "higher means more queueing pressure",
}


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
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


def _read_profile_export_jsonl(path: Path, osl_mismatch_threshold_pct: float) -> Optional[PerRequestSummary]:
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

            def get_ms(name: str) -> Optional[float]:
                m = metrics.get(name)
                if not isinstance(m, dict):
                    return None
                if (m.get("unit") or "") != "ms":
                    return None
                return _safe_float(m.get("value"))

            v_ttft = get_ms("time_to_first_token")
            v_req = get_ms("request_latency")
            v_wait = get_ms("http_req_waiting")

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

    def pctl(vals: List[float], p: float) -> Optional[float]:
        return _percentile(vals, p)

    return PerRequestSummary(
        n=total,
        ttft_p95_ms=pctl(ttft, 95),
        ttft_p99_ms=pctl(ttft, 99),
        ttft_max_ms=(ttft[-1] if ttft else None),
        req_lat_p95_ms=pctl(req_lat, 95),
        req_lat_p99_ms=pctl(req_lat, 99),
        req_lat_max_ms=(req_lat[-1] if req_lat else None),
        http_wait_p95_ms=pctl(http_wait, 95),
        http_wait_p99_ms=pctl(http_wait, 99),
        http_wait_max_ms=(http_wait[-1] if http_wait else None),
        osl_mismatch_rate=(mismatch / total) if total > 0 else None,
    )


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
    fmt = f"{{:.{digits}f}}"
    out = fmt.format(v)
    return f"{out}{unit}"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_server_metrics_csv(path: Path) -> List[Dict[str, str]]:
    """
    AIPerf server_metrics_export.csv contains multiple CSV sections separated by blank
    lines, each with its own header row. Reads all sections into a flat list of rows.
    """
    all_rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    def is_comment_or_blank(ln: str) -> bool:
        return (not ln.strip()) or ln.lstrip().startswith("#")

    i = 0
    while i < len(lines):
        while i < len(lines) and is_comment_or_blank(lines[i]):
            i += 1
        if i >= len(lines):
            break

        header_line = lines[i]
        header = next(csv.reader([header_line]))
        i += 1

        data_lines: List[str] = []
        while i < len(lines):
            ln = lines[i]
            if is_comment_or_blank(ln):
                j = i
                while j < len(lines) and is_comment_or_blank(lines[j]):
                    j += 1
                if j < len(lines) and lines[j].startswith("Endpoint,"):
                    break
                i += 1
                continue
            if ln.startswith("Endpoint,") and data_lines:
                break
            data_lines.append(ln)
            i += 1

        if data_lines:
            reader = csv.DictReader(data_lines, fieldnames=header)
            for r in reader:
                all_rows.append({k: (v if v is not None else "") for k, v in r.items()})

    return all_rows


def _detect_backend(rows: List[Dict[str, str]]) -> str:
    """Auto-detect whether server metrics are from vLLM or SGLang."""
    for r in rows:
        metric = r.get("Metric", "")
        if metric.startswith("sglang:"):
            return "sglang"
        if metric.startswith("vllm:"):
            return "vllm"
    return "unknown"


# ---------------------------------------------------------------------------
# Generic row finders (work for both vllm: and sglang: prefixes)
# ---------------------------------------------------------------------------

def _find_gauge_row(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str],
    extra_filters: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "gauge":
            continue
        if r.get("Metric") != metric_name:
            continue
        if prefer_model and r.get("model_name") and r.get("model_name") != prefer_model:
            continue
        if extra_filters:
            if any(r.get(k, "") != v for k, v in extra_filters.items()):
                continue
        return r
    return None


def _find_counter_row(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str],
    extra_filters: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "counter":
            continue
        if r.get("Metric") != metric_name:
            continue
        if prefer_model and r.get("model_name") and r.get("model_name") != prefer_model:
            continue
        if extra_filters:
            if any(r.get(k, "") != v for k, v in extra_filters.items()):
                continue
        return r
    return None


def _find_histogram_row(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str],
    extra_filters: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "histogram":
            continue
        if r.get("Metric") != metric_name:
            continue
        if prefer_model and r.get("model_name") and r.get("model_name") != prefer_model:
            continue
        if extra_filters:
            if any(r.get(k, "") != v for k, v in extra_filters.items()):
                continue
        return r
    return None


def _counter_total_rate(row: Optional[Dict[str, str]]) -> Tuple[Optional[float], Optional[float]]:
    if not row:
        return None, None
    return _safe_float(row.get("total")), _safe_float(row.get("rate"))


def _hist_p50_p95_ms(row: Optional[Dict[str, str]]) -> Tuple[Optional[float], Optional[float]]:
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


def _gauge_summary(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str],
    extra_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[float]]:
    best = _find_gauge_row(rows, metric_name, prefer_model, extra_filters)
    if best is None:
        return {"avg": None, "p50": None, "p95": None, "p99": None}
    return {
        "avg": _safe_float(best.get("avg")),
        "p50": _safe_float(best.get("p50")),
        "p95": _safe_float(best.get("p95")),
        "p99": _safe_float(best.get("p99")),
    }


# ---------------------------------------------------------------------------
# Backend-specific metric extraction
# ---------------------------------------------------------------------------

def _extract_sglang_metrics(
    rows: List[Dict[str, str]], model: Optional[str], profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract server-side metrics from SGLang's sglang:* Prometheus metrics.

    SGLang metric mapping vs vLLM:
      kv_cache_usage_perc  → sglang:token_usage (ratio 0-1, multiply by 100 for %)
      num_requests_running → sglang:num_running_reqs
      num_requests_waiting → sglang:num_queue_reqs
      prefix_cache         → sglang:cache_hit_rate (ratio)
      prompt tokens/s      → from counter sglang:prompt_tokens rate
      gen tokens/s         → from counter sglang:generation_tokens rate
      e2e latency          → sglang:e2e_request_latency_seconds (histogram)
      ttft                 → sglang:time_to_first_token_seconds (histogram)
      itl                  → sglang:inter_token_latency_seconds (histogram)
      prefill latency      → sglang:per_stage_req_latency_seconds stage=prefill_waiting
      decode latency       → not directly exposed per-request; use itl as proxy
      queue latency        → sglang:queue_time_seconds (histogram)
    """
    out: Dict[str, Any] = {}

    # KV / token usage  (ratio 0–1 → multiply by 100 for %)
    kv = _gauge_summary(rows, "sglang:token_usage", prefer_model=model)
    kv_avg = kv.get("avg")
    kv_p95 = kv.get("p95")
    out["kv_usage_avg"] = (kv_avg * 100.0) if kv_avg is not None else None
    out["kv_usage_p95"] = (kv_p95 * 100.0) if kv_p95 is not None else None

    # Running / waiting requests
    running = _gauge_summary(rows, "sglang:num_running_reqs", prefer_model=model)
    waiting = _gauge_summary(rows, "sglang:num_queue_reqs", prefer_model=model)
    out["running_avg"] = running.get("avg")
    out["waiting_avg"] = waiting.get("avg")

    # Prefix cache hit rate (SGLang exposes it directly as a ratio gauge)
    cache_gauge = _gauge_summary(rows, "sglang:cache_hit_rate", prefer_model=model)
    out["prefix_cache_hit_ratio"] = cache_gauge.get("avg")

    # Token rates from counters
    _, prompt_rate = _counter_total_rate(_find_counter_row(rows, "sglang:prompt_tokens", prefer_model=model))
    _, gen_rate    = _counter_total_rate(_find_counter_row(rows, "sglang:generation_tokens", prefer_model=model))
    out["prompt_tps"] = prompt_rate
    out["gen_tps"]    = gen_rate

    # E2E latency histogram
    _, e2e_p95 = _hist_p50_p95_ms(_find_histogram_row(rows, "sglang:e2e_request_latency_seconds", prefer_model=model))
    out["e2e_p95_ms"] = e2e_p95

    # TTFT histogram  (prefer server histogram; fallback to AIPerf profile JSON)
    _, ttft_p95 = _hist_p50_p95_ms(_find_histogram_row(rows, "sglang:time_to_first_token_seconds", prefer_model=model))
    out["ttft_p95_ms_server"] = ttft_p95

    # ITL histogram
    itl_p50, itl_p95 = _hist_p50_p95_ms(_find_histogram_row(rows, "sglang:inter_token_latency_seconds", prefer_model=model))
    out["itl_avg_ms_server"] = itl_p50   # p50 ≈ median ≈ typical ITL
    out["itl_p95_ms_server"] = itl_p95

    # Prefill waiting latency (stage histogram, tp_rank=0 to avoid double-count)
    prefill_row = _find_histogram_row(
        rows, "sglang:per_stage_req_latency_seconds", prefer_model=model,
        extra_filters={"stage": "prefill_waiting", "tp_rank": "0"},
    )
    _, prefill_p95 = _hist_p50_p95_ms(prefill_row)
    out["prefill_p95_ms"] = prefill_p95

    # No separate decode histogram in SGLang; use ITL p95 as decode proxy
    out["decode_p95_ms"] = itl_p95

    # Queue time histogram
    queue_row = _find_histogram_row(
        rows, "sglang:queue_time_seconds", prefer_model=model,
        extra_filters={"tp_rank": "0"},
    )
    _, queue_p95 = _hist_p50_p95_ms(queue_row)
    out["queue_p95_ms"] = queue_p95

    return out


def _extract_vllm_metrics(
    rows: List[Dict[str, str]], model: Optional[str]
) -> Dict[str, Any]:
    """Extract server-side metrics from vLLM's vllm:* Prometheus metrics."""
    out: Dict[str, Any] = {}

    kv = _gauge_summary(rows, "vllm:kv_cache_usage_perc", prefer_model=model)
    out["kv_usage_avg"] = kv.get("avg")
    out["kv_usage_p95"] = kv.get("p95")

    running = _gauge_summary(rows, "vllm:num_requests_running", prefer_model=model)
    waiting = _gauge_summary(rows, "vllm:num_requests_waiting", prefer_model=model)
    out["running_avg"] = running.get("avg")
    out["waiting_avg"] = waiting.get("avg")

    hits_total, _   = _counter_total_rate(_find_counter_row(rows, "vllm:prefix_cache_hits",    prefer_model=model))
    queries_total, _ = _counter_total_rate(_find_counter_row(rows, "vllm:prefix_cache_queries", prefer_model=model))
    if hits_total is not None and queries_total and queries_total > 0:
        out["prefix_cache_hit_ratio"] = hits_total / queries_total
    else:
        out["prefix_cache_hit_ratio"] = None

    _, prompt_rate = _counter_total_rate(_find_counter_row(rows, "vllm:prompt_tokens",     prefer_model=model))
    _, gen_rate    = _counter_total_rate(_find_counter_row(rows, "vllm:generation_tokens", prefer_model=model))
    out["prompt_tps"] = prompt_rate
    out["gen_tps"]    = gen_rate

    _, e2e_p95    = _hist_p50_p95_ms(_find_histogram_row(rows, "vllm:e2e_request_latency_seconds", prefer_model=model))
    _, prefill_p95 = _hist_p50_p95_ms(_find_histogram_row(rows, "vllm:request_prefill_time_seconds", prefer_model=model))
    _, decode_p95  = _hist_p50_p95_ms(_find_histogram_row(rows, "vllm:request_decode_time_seconds",  prefer_model=model))
    _, queue_p95   = _hist_p50_p95_ms(_find_histogram_row(rows, "vllm:request_queue_time_seconds",   prefer_model=model))
    out["e2e_p95_ms"]    = e2e_p95
    out["prefill_p95_ms"] = prefill_p95
    out["decode_p95_ms"]  = decode_p95
    out["queue_p95_ms"]   = queue_p95
    out["ttft_p95_ms_server"] = None
    out["itl_avg_ms_server"]  = None
    out["itl_p95_ms_server"]  = None

    return out


def _extract_metric_obj(profile: Dict[str, Any], key: str) -> Dict[str, Any]:
    obj = profile.get(key, {})
    return obj if isinstance(obj, dict) else {}


def _metric(profile: Dict[str, Any], key: str, field: str = "avg") -> Optional[float]:
    return _safe_float(_extract_metric_obj(profile, key).get(field))


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
    backend: str  # "vllm" | "sglang" | "unknown"
    streaming: Optional[bool]

    req_throughput_rps: Optional[float]
    out_tput_tps: Optional[float]
    total_tput_tps: Optional[float]
    ttft_avg_ms: Optional[float]
    ttft_p95_ms: Optional[float]
    itl_avg_ms: Optional[float]
    itl_p95_ms: Optional[float]
    req_lat_avg_ms: Optional[float]
    req_lat_p95_ms: Optional[float]

    kv_usage_avg: Optional[float]
    kv_usage_p95: Optional[float]
    running_avg: Optional[float]
    waiting_avg: Optional[float]

    prefix_cache_hit_ratio: Optional[float]
    prompt_tps: Optional[float]
    gen_tps: Optional[float]
    e2e_p95_ms: Optional[float]
    prefill_p95_ms: Optional[float]
    decode_p95_ms: Optional[float]
    queue_p95_ms: Optional[float]

    pr_ttft_p99_ms: Optional[float]
    pr_ttft_max_ms: Optional[float]
    pr_req_lat_p99_ms: Optional[float]
    pr_req_lat_max_ms: Optional[float]
    pr_http_wait_p99_ms: Optional[float]
    pr_http_wait_max_ms: Optional[float]
    pr_osl_mismatch_rate: Optional[float]

    little_L: Optional[float]
    q_pressure: Optional[float]


def load_run(run_dir: Path, per_request: bool, osl_mismatch_threshold_pct: float) -> Optional[RunSummary]:
    artifacts     = run_dir / "aiperf_artifacts"
    profile_json  = artifacts / "profile_export_aiperf.json"
    server_csv    = artifacts / "server_metrics_export.csv"
    profile_jsonl = artifacts / "profile_export.jsonl"
    if not profile_json.exists():
        return None

    profile = _read_json(profile_json)
    input_cfg    = profile.get("input_config", {}) if isinstance(profile.get("input_config"), dict) else {}
    endpoint_cfg = input_cfg.get("endpoint", {})   if isinstance(input_cfg.get("endpoint"),   dict) else {}
    urls         = endpoint_cfg.get("urls")         if isinstance(endpoint_cfg.get("urls"),         list) else []
    model_names  = endpoint_cfg.get("model_names")  if isinstance(endpoint_cfg.get("model_names"),  list) else []
    url      = urls[0]         if urls         else None
    model    = model_names[0]  if model_names  else None
    streaming = endpoint_cfg.get("streaming")
    if not isinstance(streaming, bool):
        streaming = None

    rid = RunId(model_timestamp=run_dir.parent.name, run_name=run_dir.name)

    # AIPerf profile metrics (always available regardless of backend)
    req_throughput_rps = _metric(profile, "request_throughput", "avg")
    out_tput_tps       = _metric(profile, "output_token_throughput", "avg")
    total_tput_tps     = _metric(profile, "total_token_throughput", "avg")

    # TTFT / ITL from AIPerf profile JSON (present when server streams tokens)
    ttft_avg_ms  = _metric(profile, "time_to_first_token", "avg")
    ttft_p95_ms  = _metric(profile, "time_to_first_token", "p95")
    itl_avg_ms   = _metric(profile, "inter_token_latency", "avg")
    itl_p95_ms   = _metric(profile, "inter_token_latency", "p95")
    req_lat_avg_ms = _metric(profile, "request_latency", "avg")
    req_lat_p95_ms = _metric(profile, "request_latency", "p95")

    # Server metrics
    backend = "unknown"
    kv_usage_avg = kv_usage_p95 = running_avg = waiting_avg = None
    prefix_cache_hit_ratio = prompt_tps = gen_tps = None
    e2e_p95_ms = prefill_p95_ms = decode_p95_ms = queue_p95_ms = None

    if server_csv.exists():
        rows    = _read_server_metrics_csv(server_csv)
        backend = _detect_backend(rows)

        if backend == "sglang":
            sm = _extract_sglang_metrics(rows, model, profile)
        else:
            sm = _extract_vllm_metrics(rows, model)

        kv_usage_avg           = sm.get("kv_usage_avg")
        kv_usage_p95           = sm.get("kv_usage_p95")
        running_avg            = sm.get("running_avg")
        waiting_avg            = sm.get("waiting_avg")
        prefix_cache_hit_ratio = sm.get("prefix_cache_hit_ratio")
        prompt_tps             = sm.get("prompt_tps")
        gen_tps                = sm.get("gen_tps")
        e2e_p95_ms             = sm.get("e2e_p95_ms")
        prefill_p95_ms         = sm.get("prefill_p95_ms")
        decode_p95_ms          = sm.get("decode_p95_ms")
        queue_p95_ms           = sm.get("queue_p95_ms")

        # Prefer server histogram TTFT/ITL for SGLang (more reliable when non-streaming)
        if backend == "sglang":
            if ttft_p95_ms is None and sm.get("ttft_p95_ms_server") is not None:
                ttft_p95_ms = sm["ttft_p95_ms_server"]
            if itl_avg_ms is None and sm.get("itl_avg_ms_server") is not None:
                itl_avg_ms = sm["itl_avg_ms_server"]
            if itl_p95_ms is None and sm.get("itl_p95_ms_server") is not None:
                itl_p95_ms = sm["itl_p95_ms_server"]

    pr = None
    if per_request and profile_jsonl.exists():
        pr = _read_profile_export_jsonl(profile_jsonl, osl_mismatch_threshold_pct=osl_mismatch_threshold_pct)

    # Little's Law & queue pressure
    little_L = None
    if req_throughput_rps is not None and req_lat_avg_ms is not None:
        little_L = req_throughput_rps * (req_lat_avg_ms / 1000.0)

    q_pressure = None
    if rid.concurrency is not None and rid.concurrency > 0 and waiting_avg is not None:
        q_pressure = waiting_avg / float(rid.concurrency)

    return RunSummary(
        rid=rid,
        path=run_dir,
        url=url,
        model=model,
        backend=backend,
        streaming=streaming,
        req_throughput_rps=req_throughput_rps,
        out_tput_tps=out_tput_tps,
        total_tput_tps=total_tput_tps,
        ttft_avg_ms=ttft_avg_ms,
        ttft_p95_ms=ttft_p95_ms,
        itl_avg_ms=itl_avg_ms,
        itl_p95_ms=itl_p95_ms,
        req_lat_avg_ms=req_lat_avg_ms,
        req_lat_p95_ms=req_lat_p95_ms,
        kv_usage_avg=kv_usage_avg,
        kv_usage_p95=kv_usage_p95,
        running_avg=running_avg,
        waiting_avg=waiting_avg,
        prefix_cache_hit_ratio=prefix_cache_hit_ratio,
        prompt_tps=prompt_tps,
        gen_tps=gen_tps,
        e2e_p95_ms=e2e_p95_ms,
        prefill_p95_ms=prefill_p95_ms,
        decode_p95_ms=decode_p95_ms,
        queue_p95_ms=queue_p95_ms,
        pr_ttft_p99_ms=(pr.ttft_p99_ms if pr else None),
        pr_ttft_max_ms=(pr.ttft_max_ms if pr else None),
        pr_req_lat_p99_ms=(pr.req_lat_p99_ms if pr else None),
        pr_req_lat_max_ms=(pr.req_lat_max_ms if pr else None),
        pr_http_wait_p99_ms=(pr.http_wait_p99_ms if pr else None),
        pr_http_wait_max_ms=(pr.http_wait_max_ms if pr else None),
        pr_osl_mismatch_rate=(pr.osl_mismatch_rate if pr else None),
        little_L=little_L,
        q_pressure=q_pressure,
    )


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


def filter_runs(runs: List[RunSummary], model_ts: Optional[str], suite: Optional[str]) -> List[RunSummary]:
    out = runs
    if model_ts:
        out = [r for r in out if r.rid.model_timestamp == model_ts]
    if suite:
        out = [r for r in out if r.rid.suite == suite]
    return out


def print_table(runs: List[RunSummary], limit: Optional[int]) -> None:
    rows = runs[:limit] if limit else runs
    # Show backend column so it's obvious what was detected
    headers = [
        "run", "suite", "c", "req", "in", "out", "backend",
        "rps", "out_tps",
        "ttft_avg_ms", "ttft_p95_ms",
        "itl_avg_ms", "itl_p95_ms",
        "req_lat_avg_ms", "req_lat_p95_ms",
        "kv_avg%", "kv_p95%",
        "running_avg", "waiting_avg",
        "prefill_p95_ms", "decode_p95_ms",
        "e2e_p95_ms", "queue_p95_ms",
        "prompt_tps", "gen_tps",
        "pcache_hit",
        "pr_ttft_p99", "pr_ttft_max",
        "pr_req_p99", "pr_req_max",
        "pr_wait_p99", "pr_wait_max",
        "pr_osl_mismatch",
        "little_L", "q_pressure",
    ]

    table: List[List[str]] = []
    for r in rows:
        kv_avg_pct = r.kv_usage_avg   # already *100 for sglang, raw% for vllm
        kv_p95_pct = r.kv_usage_p95
        table.append([
            r.rid.short(), r.rid.suite,
            str(r.rid.concurrency or ""), str(r.rid.request_count or ""),
            str(r.rid.input_tokens or ""), str(r.rid.output_tokens or ""),
            r.backend,
            _p(r.req_throughput_rps, "", 3), _p(r.out_tput_tps, "", 2),
            _p(r.ttft_avg_ms, "", 2), _p(r.ttft_p95_ms, "", 2),
            _p(r.itl_avg_ms, "", 3), _p(r.itl_p95_ms, "", 3),
            _p(r.req_lat_avg_ms, "", 2), _p(r.req_lat_p95_ms, "", 2),
            _p(kv_avg_pct, "", 2), _p(kv_p95_pct, "", 2),
            _p(r.running_avg, "", 2), _p(r.waiting_avg, "", 2),
            _p(r.prefill_p95_ms, "", 2), _p(r.decode_p95_ms, "", 2),
            _p(r.e2e_p95_ms, "", 2), _p(r.queue_p95_ms, "", 2),
            _p(r.prompt_tps, "", 2), _p(r.gen_tps, "", 2),
            _p(r.prefix_cache_hit_ratio, "", 4),
            _p(r.pr_ttft_p99_ms, "", 2), _p(r.pr_ttft_max_ms, "", 2),
            _p(r.pr_req_lat_p99_ms, "", 2), _p(r.pr_req_lat_max_ms, "", 2),
            _p(r.pr_http_wait_p99_ms, "", 2), _p(r.pr_http_wait_max_ms, "", 2),
            _p(r.pr_osl_mismatch_rate, "", 4),
            _p(r.little_L, "", 2), _p(r.q_pressure, "", 3),
        ])

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: List[str]) -> str:
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for row in table:
        print(fmt_row(row))

    backends = set(r.backend for r in rows)
    print(f"\nDetected backend(s): {', '.join(sorted(backends))}")
    print("\nLegend (direction):")
    print("  - Latencies (ms): lower is better")
    print("  - Throughputs (rps / tokens/sec): higher is better")
    print("  - kv_%: lower is safer (more KV headroom); SGLang token_usage×100, vLLM kv_cache_usage_perc")
    print("  - waiting_avg, q_pressure: lower is better (less queueing)")
    print("  - pcache_hit: higher is better (workload dependent)")
    print("  - decode_p95_ms (SGLang): proxy via inter-token-latency p95")
    if any(r.pr_osl_mismatch_rate is not None for r in rows):
        print("  - pr_osl_mismatch: lower is better (more consistent output length)")


def plot_runs(runs: List[RunSummary], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"NOTE: matplotlib is required for plotting ({e}).", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    runs2 = [r for r in runs if r.rid.concurrency is not None]
    runs2.sort(key=lambda r: (r.rid.suite, r.rid.concurrency or 0, r.rid.input_tokens or 0, r.rid.output_tokens or 0))

    # Detect what backends are present and annotate titles accordingly
    backends = set(r.backend for r in runs2)
    backend_note = f"  [{', '.join(sorted(backends))}]" if backends - {"unknown"} else ""

    def _scatter(metric_getter, ylabel: str, fname: str, note: str = "") -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        suites = sorted(set(r.rid.suite for r in runs2))
        plotted_any = False

        for s in suites:
            xs: List[int] = []
            ys: List[float] = []
            labels: List[str] = []
            for r in runs2:
                if r.rid.suite != s:
                    continue
                y = metric_getter(r)
                if y is None:
                    continue
                if r.rid.concurrency is None:
                    continue
                xs.append(r.rid.concurrency)
                ys.append(y)
                labels.append(f"c={r.rid.concurrency}")

            if xs:
                plotted_any = True
                line, = ax.plot(xs, ys, marker="o", linestyle="-", label=s)
                # Annotate each point with its concurrency value
                for x, y_val, lbl in zip(xs, ys, labels):
                    ax.annotate(
                        lbl, (x, y_val),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=line.get_color(), alpha=0.8,
                    )

        full_title = ylabel + backend_note
        if note:
            full_title += f"\n{note}"
        ax.set_xlabel("Concurrency (aiperf in-flight requests)")
        ax.set_ylabel(ylabel)
        ax.set_title(full_title)
        ax.grid(True, alpha=0.3)

        if plotted_any:
            ax.legend()
        else:
            ax.text(
                0.5, 0.5,
                "No numeric data available for this metric\nin the selected runs.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11,
            )

        path = out_dir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Wrote {path}")

    _scatter(lambda r: r.ttft_p95_ms,
             "TTFT p95 (ms)", "ttft_p95_vs_concurrency.png",
             note="(from profile JSON or SGLang server histogram)")
    _scatter(lambda r: r.itl_p95_ms,
             "Inter-token latency p95 (ms)", "itl_p95_vs_concurrency.png",
             note="(from profile JSON or SGLang server histogram)")
    _scatter(lambda r: r.itl_avg_ms,
             "Inter-token latency avg/p50 (ms)", "itl_avg_vs_concurrency.png",
             note="(from profile JSON or SGLang server histogram)")
    _scatter(lambda r: r.out_tput_tps,
             "Output token throughput (tokens/sec)", "out_tps_vs_concurrency.png")
    _scatter(lambda r: r.kv_usage_p95,
             "KV/token usage p95 (%)", "kv_usage_p95_vs_concurrency.png",
             note="(SGLang: token_usage×100; vLLM: kv_cache_usage_perc)")
    _scatter(lambda r: r.kv_usage_avg,
             "KV/token usage avg (%)", "kv_usage_avg_vs_concurrency.png",
             note="(SGLang: token_usage×100; vLLM: kv_cache_usage_perc)")
    _scatter(lambda r: r.prefill_p95_ms,
             "Prefill/prefill-wait p95 (ms)", "prefill_p95_vs_concurrency.png",
             note="(SGLang: per_stage_req_latency prefill_waiting)")
    _scatter(lambda r: r.decode_p95_ms,
             "Decode p95 (ms)", "decode_p95_vs_concurrency.png",
             note="(SGLang: proxy via ITL p95)")
    _scatter(lambda r: r.e2e_p95_ms,
             "E2E request latency p95 (ms) [server]", "e2e_p95_vs_concurrency.png")
    _scatter(lambda r: r.queue_p95_ms,
             "Queue time p95 (ms) [server]", "queue_p95_vs_concurrency.png")
    _scatter(lambda r: r.req_lat_p95_ms,
             "Request latency p95 (ms) [aiperf]", "req_lat_p95_vs_concurrency.png")
    _scatter(lambda r: r.prefix_cache_hit_ratio,
             "Prefix cache hit ratio", "prefix_cache_hit_ratio_vs_concurrency.png",
             note="(SGLang: cache_hit_rate gauge; vLLM: hits/queries)")
    _scatter(lambda r: r.waiting_avg,
             "Avg waiting requests", "waiting_avg_vs_concurrency.png",
             note="(SGLang: num_queue_reqs; vLLM: num_requests_waiting)")
    _scatter(lambda r: r.gen_tps,
             "Generation token rate (tokens/sec)", "gen_tps_vs_concurrency.png")
    _scatter(lambda r: r.little_L,
             "Little's Law L = rps × lat_avg_sec", "littles_law_vs_concurrency.png")


def _default_plot_out(benchmarks_dir: str, model_ts: Optional[str]) -> Path:
    b = Path(benchmarks_dir)
    if model_ts:
        return b / model_ts / "_plots"
    return b / "_plots"


def write_markdown_report(
    runs: List[RunSummary],
    out_path: Path,
    title: str,
    include_per_request: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_c = [r for r in runs if r.rid.concurrency is not None]
    by_c.sort(key=lambda r: r.rid.concurrency or 0)

    knee: Optional[RunSummary] = None
    for i in range(1, len(by_c)):
        prev, cur = by_c[i - 1], by_c[i]
        if prev.ttft_p95_ms and cur.ttft_p95_ms and prev.ttft_p95_ms > 0 and (cur.ttft_p95_ms / prev.ttft_p95_ms) >= 2.0:
            knee = cur
            break

    best_tput = max((r for r in runs if r.out_tput_tps is not None), key=lambda r: r.out_tput_tps or -1, default=None)
    best_lat  = min((r for r in runs if r.ttft_p95_ms is not None), key=lambda r: r.ttft_p95_ms or float("inf"), default=None)

    backends = sorted({r.backend for r in runs})

    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"**Detected backend(s):** {', '.join(backends)}")
    lines.append("")
    lines.append("### Metric directions")
    lines.append("")
    lines.append("- **Lower is better**: TTFT, request latency, ITL, prefill/decode/e2e/queue p95, waiting.")
    lines.append("- **Higher is better**: throughput (rps, tokens/sec), prompt/gen rates, cache hit ratio.")
    lines.append("- **KV/token usage %**: lower = more headroom. SGLang uses `token_usage×100`; vLLM uses `kv_cache_usage_perc`.")
    lines.append("- **decode_p95_ms (SGLang)**: proxy via inter-token-latency p95 (no per-request decode histogram in SGLang).")
    lines.append("")
    lines.append("### Highlights")
    lines.append("")
    if best_lat:
        lines.append(f"- **Best TTFT p95**: `{best_lat.rid.run_name}` (c={best_lat.rid.concurrency}) → {_p(best_lat.ttft_p95_ms, ' ms', 2)}")
    if best_tput:
        lines.append(f"- **Best output throughput**: `{best_tput.rid.run_name}` (c={best_tput.rid.concurrency}) → {_p(best_tput.out_tput_tps, ' tok/s', 2)}")
    if knee:
        lines.append(f"- **Saturation knee (heuristic)**: around c={knee.rid.concurrency} where TTFT p95 jumps ≥2× vs previous concurrency.")
    lines.append("")
    lines.append("### Plots")
    lines.append("")
    for png in [
        "ttft_p95_vs_concurrency.png", "itl_p95_vs_concurrency.png", "itl_avg_vs_concurrency.png",
        "out_tps_vs_concurrency.png", "req_lat_p95_vs_concurrency.png",
        "kv_usage_p95_vs_concurrency.png", "kv_usage_avg_vs_concurrency.png",
        "prefill_p95_vs_concurrency.png", "decode_p95_vs_concurrency.png",
        "e2e_p95_vs_concurrency.png", "queue_p95_vs_concurrency.png",
        "waiting_avg_vs_concurrency.png", "gen_tps_vs_concurrency.png",
        "prefix_cache_hit_ratio_vs_concurrency.png", "littles_law_vs_concurrency.png",
    ]:
        lines.append(f"- `{png}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize and plot AIPerf benchmark runs (supports vLLM and SGLang backends).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarks-dir", default="benchmarks", help="Benchmarks root directory")
    p.add_argument("--model-ts", default=None, help="Filter to a specific model+timestamp directory")
    p.add_argument("--suite", default=None, choices=[None, "baseline", "concurrency", "longctx", "stress"], help="Filter by suite")
    p.add_argument("--limit", type=int, default=50, help="Max rows to print")
    p.add_argument("--sort", default="suite,c", help="Sort keys: suite,c,in,out,req,ttft_p95,out_tps,rps")
    p.add_argument("--per-request", action="store_true",
                   help="Also parse profile_export.jsonl for per-request p99/max and OSL mismatch rate.")
    p.add_argument("--osl-mismatch-threshold-pct", type=float, default=5.0)
    p.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    p.add_argument("--by-suite", action="store_true",
                   help="Also generate per-suite plots under <plot-out>/<suite>/")
    p.add_argument("--plot-out", default=None)
    p.add_argument("--report", action="store_true", help="Write a Markdown summary report")
    p.add_argument("--report-name", default="report.md")

    args = p.parse_args(argv)
    bdir = Path(args.benchmarks_dir)
    all_runs: List[RunSummary] = []
    for run_dir in iter_runs(bdir):
        s = load_run(run_dir, per_request=bool(args.per_request),
                     osl_mismatch_threshold_pct=float(args.osl_mismatch_threshold_pct))
        if s:
            all_runs.append(s)

    runs = filter_runs(all_runs, args.model_ts, args.suite)

    sort_keys = [k.strip() for k in str(args.sort).split(",") if k.strip()]

    def sort_key(r: RunSummary) -> Tuple:
        key: List[Any] = []
        for k in sort_keys:
            if k == "suite":
                key.append(r.rid.suite)
            elif k in {"c", "concurrency"}:
                key.append(r.rid.concurrency if r.rid.concurrency is not None else 10**9)
            elif k == "in":
                key.append(r.rid.input_tokens if r.rid.input_tokens is not None else 10**9)
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
            else:
                key.append("")
        key.append(r.rid.short())
        return tuple(key)

    runs.sort(key=sort_key)

    if not runs:
        print(f"No runs found under {bdir.resolve()}")
        return 2

    print_table(runs, args.limit)

    if args.plot:
        plot_out = args.plot_out
        if plot_out is None:
            plot_out = str(_default_plot_out(args.benchmarks_dir, args.model_ts))
        base_out = Path(plot_out)
        if args.by_suite:
            plot_runs(runs, base_out / "all")
            for s in sorted({r.rid.suite for r in runs if r.rid.suite != "unknown"}):
                plot_runs([r for r in runs if r.rid.suite == s], base_out / s)
        else:
            plot_runs(runs, base_out)

    if args.report:
        out_dir = Path(args.plot_out) if args.plot_out else _default_plot_out(args.benchmarks_dir, args.model_ts)
        if args.by_suite:
            write_markdown_report(runs, out_dir / "all" / str(args.report_name),
                                   title=f"Benchmark report ({args.model_ts or 'all'} / all)",
                                   include_per_request=bool(args.per_request))
            for s in sorted({r.rid.suite for r in runs if r.rid.suite != "unknown"}):
                write_markdown_report([r for r in runs if r.rid.suite == s],
                                       out_dir / s / str(args.report_name),
                                       title=f"Benchmark report ({args.model_ts or 'all'} / {s})",
                                       include_per_request=bool(args.per_request))
        else:
            write_markdown_report(runs, out_dir / str(args.report_name),
                                   title=f"Benchmark report ({args.model_ts or 'all runs'})",
                                   include_per_request=bool(args.per_request))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())