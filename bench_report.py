#!/usr/bin/env python3
"""
Benchmark run summarizer + simple plots for AIPerf artifacts produced by benchmark.sh.

Works with directories like:
  benchmarks/<MODEL>_<TIMESTAMP>/<suite>_c<...>_req<...>_in<...>_out<...>/
    aiperf_artifacts/profile_export_aiperf.json
    aiperf_artifacts/profile_export_aiperf.csv
    aiperf_artifacts/server_metrics_export.csv

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
    # latency
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
    # throughput
    "rps": "higher is better",
    "out_tps": "higher is better",
    "total_tps": "higher is better",
    "prompt_tps": "higher is better",
    "gen_tps": "higher is better",
    # utilization / pressure
    "kv_p95": "lower is safer (more headroom)",
    "waiting_avg": "lower is better",
    # caches
    "pcache_hit": "higher is better (workload dependent)",
    # per-request tails
    "pr_ttft_p99": "lower is better",
    "pr_ttft_max": "lower is better",
    "pr_req_p99": "lower is better",
    "pr_req_max": "lower is better",
    "pr_wait_p99": "lower is better",
    "pr_wait_max": "lower is better",
    "pr_osl_mismatch": "lower is better",
    # derived
    "little_L": "higher means more in-system work (often queueing)",
    "q_pressure": "higher means more queueing pressure",
}

def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    """
    Inclusive percentile with linear interpolation.
    Expects sorted_vals already sorted ascending.
    p in [0, 100]
    """
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    # rank in [0, n-1]
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
    osl_mismatch_rate: Optional[float]  # fraction with abs(diff_pct) >= threshold


def _read_profile_export_jsonl(path: Path, osl_mismatch_threshold_pct: float) -> Optional[PerRequestSummary]:
    """
    Parse AIPerf per-request records export (JSONL).
    We focus on a few high-signal latency metrics and OSL mismatch rate.
    """
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
                    # Some metrics are arrays or other units; only keep ms scalars here.
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

            # OSL mismatch diff pct (unit: %)
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
    AIPerf server_metrics_export.csv contains multiple CSV "sections" separated by blank
    lines, each with its own header row. This parser reads all sections and returns a
    flat list of rows.
    """
    all_rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    def is_comment_or_blank(ln: str) -> bool:
        return (not ln.strip()) or ln.lstrip().startswith("#")

    i = 0
    while i < len(lines):
        # Seek to next header
        while i < len(lines) and is_comment_or_blank(lines[i]):
            i += 1
        if i >= len(lines):
            break

        header_line = lines[i]
        header = next(csv.reader([header_line]))
        i += 1

        # Collect data lines until next blank/comment + next header OR EOF.
        data_lines: List[str] = []
        while i < len(lines):
            ln = lines[i]
            if is_comment_or_blank(ln):
                # Look ahead: if next non-blank/comment line is a new header, stop this section.
                j = i
                while j < len(lines) and is_comment_or_blank(lines[j]):
                    j += 1
                if j < len(lines) and lines[j].startswith("Endpoint,"):
                    break
                i += 1
                continue
            if ln.startswith("Endpoint,") and data_lines:
                # Defensive: a new header appeared without blank separation.
                break
            data_lines.append(ln)
            i += 1

        if data_lines:
            reader = csv.DictReader(data_lines, fieldnames=header)
            for r in reader:
                all_rows.append({k: (v if v is not None else "") for k, v in r.items()})

    return all_rows


def _server_counter_row(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str]
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "counter":
            continue
        if r.get("Metric") != metric_name:
            continue
        if prefer_model and r.get("model_name") and r.get("model_name") != prefer_model:
            continue
        return r
    return None


def _server_histogram_row(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str]
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("Type") != "histogram":
            continue
        if r.get("Metric") != metric_name:
            continue
        if prefer_model and r.get("model_name") and r.get("model_name") != prefer_model:
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
    # AIPerf histogram exports use *_estimate fields and are in the metric's unit.
    p50 = _safe_float(row.get("p50_estimate"))
    p95 = _safe_float(row.get("p95_estimate"))
    unit = (row.get("Unit") or "").strip()
    if unit == "seconds":
        if p50 is not None:
            p50 *= 1000.0
        if p95 is not None:
            p95 *= 1000.0
    return p50, p95


def _extract_metric_obj(profile: Dict[str, Any], key: str) -> Dict[str, Any]:
    obj = profile.get(key, {})
    return obj if isinstance(obj, dict) else {}


def _metric(profile: Dict[str, Any], key: str, field: str = "avg") -> Optional[float]:
    return _safe_float(_extract_metric_obj(profile, key).get(field))


def _server_metric_summary(
    rows: List[Dict[str, str]], metric_name: str, prefer_model: Optional[str]
) -> Dict[str, Optional[float]]:
    """
    For gauge metrics exported as summary rows with columns:
      Endpoint,Type,Metric,Unit,avg,min,max,std,p1,...,p99,engine,model_name,...
    Return a dict with avg/p50/p95/p99 where present.
    """
    best: Optional[Dict[str, str]] = None
    for r in rows:
        if r.get("Metric") != metric_name:
            continue
        if r.get("Type") != "gauge":
            continue
        if prefer_model and r.get("model_name") and r.get("model_name") != prefer_model:
            continue
        # Pick the first match; if multiple, prefer the one with sleep_state empty (for kv_cache_usage_perc it's empty)
        if best is None:
            best = r
        elif best.get("sleep_state") and not r.get("sleep_state"):
            best = r
    if best is None:
        return {"avg": None, "p50": None, "p95": None, "p99": None}
    return {
        "avg": _safe_float(best.get("avg")),
        "p50": _safe_float(best.get("p50")),
        "p95": _safe_float(best.get("p95")),
        "p99": _safe_float(best.get("p99")),
    }


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
    streaming: Optional[bool]

    # core AIPerf metrics (from profile_export_aiperf.json)
    req_throughput_rps: Optional[float]
    out_tput_tps: Optional[float]
    total_tput_tps: Optional[float]
    ttft_avg_ms: Optional[float]
    ttft_p95_ms: Optional[float]
    itl_avg_ms: Optional[float]
    itl_p95_ms: Optional[float]
    req_lat_avg_ms: Optional[float]
    req_lat_p95_ms: Optional[float]

    # vLLM server metrics (from server_metrics_export.csv), best-effort
    kv_usage_avg: Optional[float]
    kv_usage_p95: Optional[float]
    running_avg: Optional[float]
    waiting_avg: Optional[float]

    # additional vLLM metrics (from server_metrics_export.csv)
    prefix_cache_hit_ratio: Optional[float]  # hits / queries (total)
    prompt_tps: Optional[float]  # prompt token rate
    gen_tps: Optional[float]  # generation token rate
    e2e_p95_ms: Optional[float]
    prefill_p95_ms: Optional[float]
    decode_p95_ms: Optional[float]
    queue_p95_ms: Optional[float]

    # optional per-request metrics (from profile_export.jsonl)
    pr_ttft_p99_ms: Optional[float]
    pr_ttft_max_ms: Optional[float]
    pr_req_lat_p99_ms: Optional[float]
    pr_req_lat_max_ms: Optional[float]
    pr_http_wait_p99_ms: Optional[float]
    pr_http_wait_max_ms: Optional[float]
    pr_osl_mismatch_rate: Optional[float]

    # derived (helps explain "what is up" at high concurrency)
    little_L: Optional[float]  # Little's Law estimate L ≈ rps * (req_lat_avg_sec)
    q_pressure: Optional[float]  # waiting_avg / concurrency (rough queue pressure)


def load_run(run_dir: Path, per_request: bool, osl_mismatch_threshold_pct: float) -> Optional[RunSummary]:
    artifacts = run_dir / "aiperf_artifacts"
    profile_json = artifacts / "profile_export_aiperf.json"
    server_csv = artifacts / "server_metrics_export.csv"
    profile_jsonl = artifacts / "profile_export.jsonl"
    if not profile_json.exists():
        return None

    profile = _read_json(profile_json)
    input_cfg = profile.get("input_config", {}) if isinstance(profile.get("input_config"), dict) else {}
    endpoint_cfg = input_cfg.get("endpoint", {}) if isinstance(input_cfg.get("endpoint"), dict) else {}
    urls = endpoint_cfg.get("urls") if isinstance(endpoint_cfg.get("urls"), list) else []
    url = urls[0] if urls else None
    model_names = endpoint_cfg.get("model_names") if isinstance(endpoint_cfg.get("model_names"), list) else []
    model = model_names[0] if model_names else None
    streaming = endpoint_cfg.get("streaming")
    if not isinstance(streaming, bool):
        streaming = None

    rid = RunId(model_timestamp=run_dir.parent.name, run_name=run_dir.name)

    # AIPerf profile metrics
    req_throughput_rps = _metric(profile, "request_throughput", "avg")
    out_tput_tps = _metric(profile, "output_token_throughput", "avg")
    total_tput_tps = _metric(profile, "total_token_throughput", "avg")

    ttft_avg_ms = _metric(profile, "time_to_first_token", "avg")
    ttft_p95_ms = _metric(profile, "time_to_first_token", "p95")
    itl_avg_ms = _metric(profile, "inter_token_latency", "avg")
    itl_p95_ms = _metric(profile, "inter_token_latency", "p95")
    req_lat_avg_ms = _metric(profile, "request_latency", "avg")
    req_lat_p95_ms = _metric(profile, "request_latency", "p95")

    kv_usage_avg = kv_usage_p95 = running_avg = waiting_avg = None
    prefix_cache_hit_ratio = prompt_tps = gen_tps = None
    e2e_p95_ms = prefill_p95_ms = decode_p95_ms = queue_p95_ms = None
    if server_csv.exists():
        rows = _read_server_metrics_csv(server_csv)
        kv = _server_metric_summary(rows, "vllm:kv_cache_usage_perc", prefer_model=model)
        kv_usage_avg, kv_usage_p95 = kv.get("avg"), kv.get("p95")
        running = _server_metric_summary(rows, "vllm:num_requests_running", prefer_model=model)
        waiting = _server_metric_summary(rows, "vllm:num_requests_waiting", prefer_model=model)
        running_avg, waiting_avg = running.get("avg"), waiting.get("avg")

        # Counters
        hits_total, _ = _counter_total_rate(_server_counter_row(rows, "vllm:prefix_cache_hits", prefer_model=model))
        queries_total, _ = _counter_total_rate(_server_counter_row(rows, "vllm:prefix_cache_queries", prefer_model=model))
        if hits_total is not None and queries_total and queries_total > 0:
            prefix_cache_hit_ratio = hits_total / queries_total

        _, prompt_rate = _counter_total_rate(_server_counter_row(rows, "vllm:prompt_tokens", prefer_model=model))
        _, gen_rate = _counter_total_rate(_server_counter_row(rows, "vllm:generation_tokens", prefer_model=model))
        prompt_tps, gen_tps = prompt_rate, gen_rate

        # Histograms (p95 focus)
        _, e2e_p95_ms = _hist_p50_p95_ms(_server_histogram_row(rows, "vllm:e2e_request_latency_seconds", prefer_model=model))
        _, prefill_p95_ms = _hist_p50_p95_ms(_server_histogram_row(rows, "vllm:request_prefill_time_seconds", prefer_model=model))
        _, decode_p95_ms = _hist_p50_p95_ms(_server_histogram_row(rows, "vllm:request_decode_time_seconds", prefer_model=model))
        _, queue_p95_ms = _hist_p50_p95_ms(_server_histogram_row(rows, "vllm:request_queue_time_seconds", prefer_model=model))

    pr = None
    if per_request and profile_jsonl.exists():
        pr = _read_profile_export_jsonl(profile_jsonl, osl_mismatch_threshold_pct=osl_mismatch_threshold_pct)

    # Derived metrics
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
    headers = [
        "run",
        "suite",
        "c",
        "req",
        "in",
        "out",
        "rps",
        "out_tps",
        "ttft_avg_ms",
        "ttft_p95_ms",
        "itl_avg_ms",
        "req_lat_p95_ms",
        "kv_avg",
        "kv_p95",
        "prefill_p95_ms",
        "decode_p95_ms",
        "e2e_p95_ms",
        "prompt_tps",
        "gen_tps",
        "pcache_hit",
        "waiting_avg",
        "pr_ttft_p99",
        "pr_ttft_max",
        "pr_req_p99",
        "pr_req_max",
        "pr_wait_p99",
        "pr_wait_max",
        "pr_osl_mismatch",
        "little_L",
        "q_pressure",
    ]

    table: List[List[str]] = []
    for r in rows:
        table.append(
            [
                r.rid.short(),
                r.rid.suite,
                str(r.rid.concurrency or ""),
                str(r.rid.request_count or ""),
                str(r.rid.input_tokens or ""),
                str(r.rid.output_tokens or ""),
                _p(r.req_throughput_rps, "", 3),
                _p(r.out_tput_tps, "", 2),
                _p(r.ttft_avg_ms, "", 2),
                _p(r.ttft_p95_ms, "", 2),
                _p(r.itl_avg_ms, "", 2),
                _p(r.req_lat_p95_ms, "", 2),
                _p(r.kv_usage_avg, "", 4),
                _p(r.kv_usage_p95, "", 4),
                _p(r.prefill_p95_ms, "", 2),
                _p(r.decode_p95_ms, "", 2),
                _p(r.e2e_p95_ms, "", 2),
                _p(r.prompt_tps, "", 2),
                _p(r.gen_tps, "", 2),
                _p(r.prefix_cache_hit_ratio, "", 4),
                _p(r.waiting_avg, "", 2),
                _p(r.pr_ttft_p99_ms, "", 2),
                _p(r.pr_ttft_max_ms, "", 2),
                _p(r.pr_req_lat_p99_ms, "", 2),
                _p(r.pr_req_lat_max_ms, "", 2),
                _p(r.pr_http_wait_p99_ms, "", 2),
                _p(r.pr_http_wait_max_ms, "", 2),
                _p(r.pr_osl_mismatch_rate, "", 4),
                _p(r.little_L, "", 2),
                _p(r.q_pressure, "", 3),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: List[str]) -> str:
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for row in table:
        print(fmt_row(row))

    print("\nLegend (direction):")
    print("  - Latencies (ms): lower is better")
    print("  - Throughputs (rps / tokens/sec): higher is better")
    print("  - kv_p95: lower is safer (more KV headroom)")
    print("  - waiting_avg, q_pressure: lower is better (less queueing)")
    print("  - pcache_hit: higher is better (workload dependent)")
    if any(r.pr_osl_mismatch_rate is not None for r in rows):
        print("  - pr_osl_mismatch: lower is better (more consistent output length)")


def plot_runs(runs: List[RunSummary], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"NOTE: matplotlib is required for plotting ({e}).", file=sys.stderr)
        print("Install it with: python3 -m pip install matplotlib", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Only plot runs that have a numeric concurrency
    runs2 = [r for r in runs if r.rid.concurrency is not None]
    runs2.sort(key=lambda r: (r.rid.suite, r.rid.concurrency or 0, r.rid.input_tokens or 0, r.rid.output_tokens or 0))

    def _scatter(metric_getter, ylabel: str, fname: str) -> None:
        plt.figure(figsize=(10, 6))
        suites = sorted(set(r.rid.suite for r in runs2))
        for s in suites:
            xs, ys, labels = [], [], []
            for r in runs2:
                if r.rid.suite != s:
                    continue
                y = metric_getter(r)
                if y is None:
                    continue
                xs.append(r.rid.concurrency)
                ys.append(y)
                labels.append(r.rid.run_name)
            if xs:
                plt.plot(xs, ys, marker="o", linestyle="-", label=s)
        plt.xlabel("Concurrency (aiperf in-flight requests)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs concurrency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = out_dir / fname
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Wrote {path}")

    _scatter(lambda r: r.ttft_p95_ms, "TTFT p95 (ms)", "ttft_p95_vs_concurrency.png")
    _scatter(lambda r: r.itl_p95_ms, "Inter-token latency p95 (ms)", "itl_p95_vs_concurrency.png")
    _scatter(lambda r: r.out_tput_tps, "Output token throughput (tokens/sec)", "out_tps_vs_concurrency.png")
    _scatter(lambda r: r.kv_usage_p95, "KV cache usage p95 (ratio)", "kv_usage_p95_vs_concurrency.png")
    _scatter(lambda r: r.prefill_p95_ms, "Prefill p95 (ms) [server]", "prefill_p95_vs_concurrency.png")
    _scatter(lambda r: r.decode_p95_ms, "Decode p95 (ms) [server]", "decode_p95_vs_concurrency.png")
    _scatter(lambda r: r.e2e_p95_ms, "E2E request latency p95 (ms) [server]", "e2e_p95_vs_concurrency.png")
    _scatter(lambda r: r.prefix_cache_hit_ratio, "Prefix cache hit ratio (hits/queries)", "prefix_cache_hit_ratio_vs_concurrency.png")


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
    best_lat = min((r for r in runs if r.ttft_p95_ms is not None), key=lambda r: r.ttft_p95_ms or float("inf"), default=None)

    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append("### Metric directions (how to read ‘better’)")
    lines.append("")
    lines.append("- **Lower is better**: TTFT, request latency, inter-token latency, server-side prefill/decode/e2e p95, waiting/queueing.")
    lines.append("- **Higher is better**: throughput (requests/sec, tokens/sec), prompt/gen token rates, cache hit ratio (workload dependent).")
    lines.append("- **KV cache usage**: lower means more headroom; high values can limit concurrency or risk OOM.")
    lines.append("")
    lines.append("### Highlights")
    lines.append("")
    if best_lat:
        lines.append(f"- **Best latency (TTFT p95)**: `{best_lat.rid.run_name}` (c={best_lat.rid.concurrency}) → {_p(best_lat.ttft_p95_ms, ' ms', 2)}")
    if best_tput:
        lines.append(f"- **Best throughput (out_tps)**: `{best_tput.rid.run_name}` (c={best_tput.rid.concurrency}) → {_p(best_tput.out_tput_tps, ' tok/s', 2)}")
    if knee:
        lines.append(f"- **Saturation knee (heuristic)**: around c={knee.rid.concurrency} (`{knee.rid.run_name}`) where TTFT p95 jumps ≥2× vs previous concurrency.")
    lines.append("")
    lines.append("### Queueing interpretation")
    lines.append("")
    lines.append("- If **throughput plateaus** but **TTFT/latency rises**, you’re saturated; extra concurrency becomes queueing.")
    lines.append("- `little_L ≈ rps * (req_lat_avg_sec)` is a Little’s Law sanity check for in-system work.")
    lines.append("- `q_pressure = waiting_avg / concurrency` is a rough queue-pressure indicator.")
    lines.append("")
    if include_per_request:
        lines.append("### Per-request tails (from `profile_export.jsonl`)")
        lines.append("")
        lines.append("- Prefer these when debugging spikes: `pr_ttft_p99/max`, `pr_wait_p99/max`, `pr_req_p99/max`.")
        lines.append("- `pr_osl_mismatch` is the fraction of requests with `abs(osl_mismatch_diff_pct) >= threshold`.")
        lines.append("")
    lines.append("### Plots (if generated)")
    lines.append("")
    lines.append("- `ttft_p95_vs_concurrency.png`")
    lines.append("- `itl_p95_vs_concurrency.png`")
    lines.append("- `out_tps_vs_concurrency.png`")
    lines.append("- `kv_usage_p95_vs_concurrency.png`")
    lines.append("- `prefill_p95_vs_concurrency.png`, `decode_p95_vs_concurrency.png`, `e2e_p95_vs_concurrency.png`")
    lines.append("- `prefix_cache_hit_ratio_vs_concurrency.png`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize and plot AIPerf benchmark runs under ./benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarks-dir", default="benchmarks", help="Benchmarks root directory")
    p.add_argument("--model-ts", default=None, help="Filter to a specific model+timestamp directory, e.g. gpt-oss-20b_20260318_054504")
    p.add_argument("--suite", default=None, choices=[None, "baseline", "concurrency", "longctx", "stress"], help="Filter by suite")
    p.add_argument("--limit", type=int, default=50, help="Max rows to print")
    p.add_argument("--sort", default="suite,c", help="Sort keys: suite,c,in,out,req,ttft_p95,out_tps,rps")
    p.add_argument(
        "--per-request",
        action="store_true",
        help="Also parse aiperf_artifacts/profile_export.jsonl for per-request p99/max and OSL mismatch rate (slower).",
    )
    p.add_argument(
        "--osl-mismatch-threshold-pct",
        type=float,
        default=5.0,
        help="Threshold (absolute %) used to count OSL mismatch rate when --per-request is enabled.",
    )
    p.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    p.add_argument(
        "--plot-out",
        default=None,
        help=(
            "Directory to write plots into. "
            "If omitted and --model-ts is set, defaults to <benchmarks-dir>/<model-ts>/_plots. "
            "Otherwise defaults to <benchmarks-dir>/_plots."
        ),
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Write a Markdown summary report into the plot output directory.",
    )
    p.add_argument(
        "--report-name",
        default="report.md",
        help="Filename for the Markdown report (written under the plot output directory).",
    )

    args = p.parse_args(argv)
    bdir = Path(args.benchmarks_dir)
    all_runs: List[RunSummary] = []
    for run_dir in iter_runs(bdir):
        s = load_run(run_dir, per_request=bool(args.per_request), osl_mismatch_threshold_pct=float(args.osl_mismatch_threshold_pct))
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
                # higher is better; sort descending by negating
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
            if args.model_ts:
                plot_out = str(Path(args.benchmarks_dir) / args.model_ts / "_plots")
            else:
                plot_out = str(Path(args.benchmarks_dir) / "_plots")
        plot_runs(runs, Path(plot_out))

    if args.report:
        out_dir = Path(args.plot_out) if args.plot_out else _default_plot_out(args.benchmarks_dir, args.model_ts)
        report_path = out_dir / str(args.report_name)
        title = f"Benchmark report ({args.model_ts or 'all runs'})"
        write_markdown_report(runs, report_path, title=title, include_per_request=bool(args.per_request))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

