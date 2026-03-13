#!/usr/bin/env python3
"""
EKG-centric export at (optionally) 200 Hz with annotations inserted per-sample.

What this script produces (in --outdir):
  1) ekg_channel_info.csv
  2) ekg_data.csv
  3) annotations_aligned_to_ekg.csv
  4) ekg_data_with_annotations.csv   <-- NEW (EKG-centric, per-sample at sfreq or --target-sfreq)

The "EKG-centric" CSV contains one row per EKG sample with:
  - sample_index, time_sec, ekg_uV
  - epoch + stage (forward-filled from "Stage - ..." annotations)
  - *_active columns at sample rate for duration-based events (Arousal / Respiratory Event / Desaturation / LM)
  - annotation_onset_types + annotation_onset_desc at the exact sample where an annotation starts (point events too)

Notes / assumptions about your annotation CSV:
  - 3 columns, no header: epoch, clock_time (HH:MM:SS.xx), description
  - "epoch" is typically 30-second PSG epoch numbering (1-based)
  - "Stage - ..." rows are treated as stage changes and forward-filled across epochs
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Optional environment workaround:
# Some environments have an incompatibility between `numba` and `coverage`
# (coverage>=7) which can break importing MNE. This patch is harmless if not needed.
# ---------------------------------------------------------------------
def _safe_patch_coverage_for_numba() -> None:
    try:
        import typing as _typing
        import coverage.types as cov_types  # type: ignore

        if not hasattr(cov_types, "Tracer"):
            cov_types.Tracer = object  # type: ignore[attr-defined]
        # Add commonly referenced typing aliases if missing
        for name in (
            "TShouldTraceFn",
            "TShouldStartContextFn",
            "TTraceData",
            "TWarnFn",
            "TTraceFn",
            "TFileDisposition",
        ):
            if not hasattr(cov_types, name):
                setattr(cov_types, name, _typing.Any)  # type: ignore[attr-defined]
    except Exception:
        return


_safe_patch_coverage_for_numba()

import mne  # noqa: E402


def find_ekg_channel(ch_names: Sequence[str], pattern: str = r"(EKG|ECG)") -> str:
    """Return the first channel name matching EKG/ECG regex (prefer exact 'EKG'/'ECG')."""
    regex = re.compile(pattern, flags=re.IGNORECASE)
    matches = [ch for ch in ch_names if regex.search(ch)]
    if not matches:
        raise ValueError(
            f"No EKG/ECG-like channel found using pattern={pattern!r}.\n"
            f"Available channels: {list(ch_names)}"
        )
    for pref in ("EKG", "ECG"):
        for ch in matches:
            if ch.upper() == pref:
                return ch
    return matches[0]


def parse_hhmmss_to_seconds(t: str) -> float:
    """Parse a time string like '22:57:07.00' into seconds-of-day."""
    t = str(t).strip()
    parts = t.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported time format: {t!r} (expected HH:MM:SS.xx)")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600.0 + m * 60.0 + s


def add_relative_seconds_with_rollover(times: pd.Series, *, start_time_str: Optional[str] = None) -> pd.Series:
    """
    Convert HH:MM:SS.xx strings into seconds relative to the start time.
    Handles midnight rollover by detecting time-of-day decreases.
    """
    tod = times.astype(str).map(parse_hhmmss_to_seconds).to_numpy()
    start = tod[0] if start_time_str is None else parse_hhmmss_to_seconds(start_time_str)

    rel = np.empty_like(tod, dtype=float)
    roll = 0.0
    prev = tod[0]
    for i, v in enumerate(tod):
        if v < prev - 1.0:  # rollover
            roll += 24.0 * 3600.0
        rel[i] = v + roll - start
        prev = v
    return pd.Series(rel, index=times.index)


_DURATION_SEC_RE = re.compile(r"Dur:\s*([0-9]*\.?[0-9]+)\s*sec", flags=re.IGNORECASE)
_DURATION_MIN_RE = re.compile(r"Dur:\s*([0-9]*\.?[0-9]+)\s*min", flags=re.IGNORECASE)


def parse_duration_seconds(desc: str) -> float:
    """Extract duration in seconds from an annotation description. Returns NaN if absent."""
    if desc is None:
        return float("nan")
    s = str(desc)
    m = _DURATION_SEC_RE.search(s)
    if m:
        return float(m.group(1))
    m = _DURATION_MIN_RE.search(s)
    if m:
        return float(m.group(1)) * 60.0
    return float("nan")


def annotation_event_type(desc: str) -> str:
    """Coarse event type (prefix before first ' - '), with special handling for Stage."""
    s = str(desc).strip()
    if s.lower().startswith("stage"):
        return "Stage"
    if " - " in s:
        return s.split(" - ")[0].strip()
    return s


def parse_stage_label(desc: str) -> str:
    """Extract stage label from 'Stage - X'."""
    s = str(desc).strip()
    m = re.match(r"(?i)stage\s*-\s*(.*)$", s)
    return (m.group(1).strip() if m else s)


def infer_epoch_length_seconds(stage_df: pd.DataFrame, default: float = 30.0) -> float:
    """
    Infer epoch length using median(delta_t / delta_epoch) from Stage rows.
    Falls back to default if insufficient info.
    """
    if len(stage_df) < 2:
        return float(default)

    df = stage_df.sort_values("epoch")
    d_epoch = df["epoch"].diff().to_numpy()
    d_t = df["t_sec"].diff().to_numpy()

    mask = np.isfinite(d_epoch) & np.isfinite(d_t) & (d_epoch > 0)
    if not np.any(mask):
        return float(default)

    ratios = d_t[mask] / d_epoch[mask]
    # Robust median, then round to milliseconds to avoid weird floats like 29.999999
    est = float(np.median(ratios))
    if est <= 0 or not np.isfinite(est):
        return float(default)
    return round(est, 3)


def build_stage_by_epoch(
    ann: pd.DataFrame,
    *,
    total_samples: int,
    sfreq: float,
    epoch_len_sec: Optional[float] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    Build a 1-based array stage_by_epoch[epoch] = stage_label.
    Stage is forward-filled from "Stage - ..." annotations.
    Returns (stage_by_epoch, epoch_len_sec, samples_per_epoch).
    """
    stage_rows = ann[ann["etype"] == "Stage"].copy()
    if stage_rows.empty:
        # no stage info; return empty stage array
        epoch_len = float(epoch_len_sec) if epoch_len_sec is not None else 30.0
        samples_per_epoch = max(1, int(round(epoch_len * sfreq)))
        max_epoch = max(int(ann["epoch"].max()), int(math.ceil(total_samples / samples_per_epoch)))
        stage_by_epoch = np.full(max_epoch + 2, "", dtype=object)
        return stage_by_epoch, epoch_len, samples_per_epoch

    stage_rows["stage"] = stage_rows["description"].map(parse_stage_label)
    epoch_len = float(epoch_len_sec) if epoch_len_sec is not None else infer_epoch_length_seconds(stage_rows, default=30.0)
    samples_per_epoch = max(1, int(round(epoch_len * sfreq)))

    max_epoch = max(int(ann["epoch"].max()), int(math.ceil(total_samples / samples_per_epoch)))
    stage_by_epoch = np.full(max_epoch + 2, "", dtype=object)  # index 0 unused

    # Sort stage change points by epoch
    stage_rows = stage_rows[["epoch", "stage"]].drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    epochs = stage_rows["epoch"].to_numpy(dtype=int)
    stages = stage_rows["stage"].to_numpy(dtype=object)

    # Fill from epoch 1 to first stage epoch with the first stage (common: "No Stage")
    first_epoch = int(epochs[0])
    stage_by_epoch[1:first_epoch] = stages[0]

    for i in range(len(epochs)):
        start_e = int(epochs[i])
        end_e = int(epochs[i + 1]) if i + 1 < len(epochs) else (max_epoch + 1)
        stage_by_epoch[start_e:end_e] = stages[i]

    # Forward-fill any remaining blanks (just in case)
    last = ""
    for e in range(1, len(stage_by_epoch)):
        if stage_by_epoch[e] == "":
            stage_by_epoch[e] = last
        else:
            last = stage_by_epoch[e]

    return stage_by_epoch, epoch_len, samples_per_epoch


def export_ekg_channel_info(raw: mne.io.BaseRaw, ekg_ch_name: str, out_csv: str) -> None:
    ekg_idx = raw.ch_names.index(ekg_ch_name)
    sfreq = float(raw.info["sfreq"])
    duration_sec = float(raw.n_times) / sfreq

    try:
        mne_ch_type = raw.get_channel_types(picks=[ekg_idx])[0]
    except Exception:
        mne_ch_type = "unknown"

    extras = raw._raw_extras[0] if hasattr(raw, "_raw_extras") and raw._raw_extras else {}

    row = {
        "channel_name": ekg_ch_name,
        "channel_index": ekg_idx,
        "mne_channel_type": mne_ch_type,
        "sfreq_hz": sfreq,
        "n_samples": int(raw.n_times),
        "duration_sec": duration_sec,
        "meas_date": raw.info.get("meas_date"),
        "orig_unit_from_edf": getattr(raw, "_orig_units", {}).get(ekg_ch_name),
    }

    if isinstance(extras, dict):
        for k in ("physical_max", "digital_max", "units", "cal", "offsets", "n_samps"):
            if k in extras:
                v = extras[k]
                try:
                    row[f"edf_{k}"] = v[ekg_idx]
                except Exception:
                    row[f"edf_{k}"] = v

    pd.DataFrame([row]).to_csv(out_csv, index=False)


def read_annotation_csv(annotation_csv: str) -> pd.DataFrame:
    ann = pd.read_csv(annotation_csv, header=None, names=["epoch", "time", "description"])
    ann["epoch"] = ann["epoch"].astype(int)
    ann["time"] = ann["time"].astype(str).str.strip()
    ann["description"] = ann["description"].astype(str).str.strip()
    return ann


def align_annotations(
    ann: pd.DataFrame,
    *,
    sfreq: float,
) -> pd.DataFrame:
    ann = ann.copy()
    ann["t_sec"] = add_relative_seconds_with_rollover(ann["time"])
    ann["duration_sec"] = ann["description"].map(parse_duration_seconds)
    ann["etype"] = ann["description"].map(annotation_event_type)

    ann["start_sample"] = np.round(ann["t_sec"].to_numpy() * sfreq).astype(int)
    dur_samples = np.round(ann["duration_sec"].fillna(0).to_numpy() * sfreq).astype(int)
    ann["duration_samples"] = dur_samples

    ann["end_sample_exclusive"] = ann["start_sample"] + ann["duration_samples"]
    ann["end_sample"] = np.maximum(ann["start_sample"], ann["end_sample_exclusive"] - 1)
    return ann


def sanitize_colname(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def export_ekg_data_csv(
    raw: mne.io.BaseRaw,
    ekg_ch_name: str,
    *,
    out_csv: str,
    chunk_seconds: float,
    total_samples: int,
) -> None:
    """Write basic EKG data CSV: sample_index,time_sec,ekg_uV"""
    ekg_idx = raw.ch_names.index(ekg_ch_name)
    sfreq = float(raw.info["sfreq"])
    chunk_size = max(1, int(round(chunk_seconds * sfreq)))

    first = True
    for i0 in range(0, total_samples, chunk_size):
        i1 = min(total_samples, i0 + chunk_size)
        ekg_uV = raw.get_data(picks=[ekg_idx], start=i0, stop=i1)[0] * 1e6

        sample_idx = np.arange(i0, i1, dtype=np.int64)
        time_sec = sample_idx / sfreq

        df_chunk = pd.DataFrame(
            {"sample_index": sample_idx, "time_sec": time_sec, "ekg_uV": ekg_uV}
        )
        df_chunk.to_csv(
            out_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
            float_format="%.3f",
        )
        first = False


def export_ekg_with_annotations(
    raw: mne.io.BaseRaw,
    ekg_ch_name: str,
    ann_aligned: pd.DataFrame,
    *,
    out_csv: str,
    chunk_seconds: float,
    total_samples: int,
    epoch_len_sec: Optional[float] = None,
) -> None:
    """
    EKG-centric output: one row per sample with stage + duration-event actives + onset annotations.
    """
    ekg_idx = raw.ch_names.index(ekg_ch_name)
    sfreq = float(raw.info["sfreq"])
    chunk_size = max(1, int(round(chunk_seconds * sfreq)))

    # Build stage forward-fill by epoch
    stage_by_epoch, epoch_len_used, samples_per_epoch = build_stage_by_epoch(
        ann_aligned, total_samples=total_samples, sfreq=sfreq, epoch_len_sec=epoch_len_sec
    )

    # Duration-based event types in THIS annotation file (commonly: these 4)
    duration_types = (
        ann_aligned.loc[
            ann_aligned["duration_sec"].notna() & (ann_aligned["duration_sec"] > 0) & (ann_aligned["etype"] != "Stage"),
            "etype",
        ]
        .value_counts()
        .index.tolist()
    )

    # In your example file, this ends up being: Arousal / Respiratory Event / Desaturation / LM
    intervals_by_type: Dict[str, List[Tuple[int, int]]] = {t: [] for t in duration_types}

    # Onset annotation maps (includes point events AND duration events)
    onset_types: Dict[int, List[str]] = defaultdict(list)
    onset_desc: Dict[int, List[str]] = defaultdict(list)

    # Populate structures
    for row in ann_aligned.itertuples(index=False):
        s = int(row.start_sample)
        if s < 0:
            continue
        et = str(row.etype)
        onset_types[s].append(et)
        onset_desc[s].append(str(row.description))

        dur = row.duration_sec
        if dur is not None and np.isfinite(dur) and float(dur) > 0 and et in intervals_by_type:
            e_excl = int(row.end_sample_exclusive)
            if e_excl > s:
                intervals_by_type[et].append((s, e_excl))

    # Sort intervals per type for slightly faster overlap checks
    for t in list(intervals_by_type.keys()):
        intervals_by_type[t].sort(key=lambda x: x[0])

    # Prepare sorted onset sample keys for streaming fill
    onset_keys_sorted = np.array(sorted(onset_types.keys()), dtype=np.int64)
    onset_ptr = 0

    first = True
    for i0 in range(0, total_samples, chunk_size):
        i1 = min(total_samples, i0 + chunk_size)
        n = i1 - i0

        # EKG chunk in microvolts
        ekg_uV = raw.get_data(picks=[ekg_idx], start=i0, stop=i1)[0] * 1e6

        sample_idx = np.arange(i0, i1, dtype=np.int64)
        time_sec = sample_idx / sfreq

        # Epoch index (1-based) and stage
        epoch_idx = (sample_idx // samples_per_epoch) + 1
        epoch_idx = np.clip(epoch_idx, 1, len(stage_by_epoch) - 1)
        stage = stage_by_epoch[epoch_idx]

        # Duration-based active columns
        active_cols: Dict[str, np.ndarray] = {}
        for t, intervals in intervals_by_type.items():
            arr = np.zeros(n, dtype=np.uint8)
            # Mark overlaps with this chunk
            for (s, e) in intervals:
                if e <= i0:
                    continue
                if s >= i1:
                    break  # intervals sorted by start
                a = max(s, i0) - i0
                b = min(e, i1) - i0
                if b > a:
                    arr[a:b] = 1
            active_cols[f"{sanitize_colname(t)}_active"] = arr

        # Onset columns (sparse)
        onset_types_col = np.full(n, "", dtype=object)
        onset_desc_col = np.full(n, "", dtype=object)

        # Advance pointer to first onset >= i0
        while onset_ptr < len(onset_keys_sorted) and onset_keys_sorted[onset_ptr] < i0:
            onset_ptr += 1
        p = onset_ptr
        while p < len(onset_keys_sorted):
            s = int(onset_keys_sorted[p])
            if s >= i1:
                break
            idx = s - i0
            # types: unique + stable order
            types_list = onset_types[s]
            seen = set()
            types_unique = []
            for x in types_list:
                if x not in seen:
                    seen.add(x)
                    types_unique.append(x)
            onset_types_col[idx] = ";".join(types_unique)
            onset_desc_col[idx] = ";".join(onset_desc[s])
            p += 1
        onset_ptr = p

        df = pd.DataFrame(
            {
                "sample_index": sample_idx,
                "time_sec": time_sec,
                "ekg_uV": ekg_uV,
                "epoch": epoch_idx,
                "stage": stage,
                "annotation_onset_types": onset_types_col,
                "annotation_onset_desc": onset_desc_col,
                **active_cols,
            }
        )

        df.to_csv(
            out_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
            float_format="%.3f",
        )
        first = False

    # Write a small sidecar describing inferred epoch length
    sidecar = os.path.splitext(out_csv)[0] + "_meta.txt"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(f"sfreq_hz={sfreq}\n")
        f.write(f"epoch_len_sec={epoch_len_used}\n")
        f.write(f"samples_per_epoch={samples_per_epoch}\n")
        f.write(f"duration_event_types={duration_types}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True, help="EDF path")
    ap.add_argument("--ann", required=True, help="Annotation CSV path")
    ap.add_argument("--outdir", required=True, help="Output directory")

    ap.add_argument("--ekg-pattern", default=r"(EKG|ECG)", help="Regex to find EKG/ECG channel")
    ap.add_argument("--chunk-seconds", type=float, default=300.0, help="Chunk size for streaming CSV writes")
    ap.add_argument("--max-seconds", type=float, default=None, help="Process only first N seconds (for testing)")

    ap.add_argument(
        "--target-sfreq",
        type=float,
        default=200.0,
        help="Resample EKG to this sampling rate before export (default: 200 Hz). "
             "If EDF is already 200 Hz, no resampling occurs.",
    )

    ap.add_argument(
        "--epoch-len-sec",
        type=float,
        default=None,
        help="Override epoch length in seconds. If omitted, inferred from Stage annotations (usually ~30s).",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Read EDF
    raw = mne.io.read_raw_edf(args.edf, preload=False, verbose="ERROR")
    ekg_ch = find_ekg_channel(raw.ch_names, pattern=args.ekg_pattern)

    # Keep only EKG channel to reduce memory + speed up
    raw.pick_channels([ekg_ch])

    # Resample to target (default: 200 Hz) if needed
    if args.target_sfreq is not None:
        cur = float(raw.info["sfreq"])
        if abs(cur - float(args.target_sfreq)) > 1e-9:
            # Load data for resampling (only 1 channel now)
            raw.load_data()
            raw.resample(float(args.target_sfreq))

    sfreq = float(raw.info["sfreq"])
    total_samples = int(raw.n_times)
    if args.max_seconds is not None:
        total_samples = min(total_samples, int(round(float(args.max_seconds) * sfreq)))

    # 1) EKG channel info
    export_ekg_channel_info(raw, ekg_ch, os.path.join(args.outdir, "ekg_channel_info.csv"))

    # 2) Read + align annotations to this (possibly resampled) sfreq
    ann = read_annotation_csv(args.ann)
    ann_aligned = align_annotations(ann, sfreq=sfreq)
    ann_aligned.to_csv(os.path.join(args.outdir, "annotations_aligned_to_ekg.csv"), index=False, float_format="%.3f")

    # 3) Export raw EKG data (ekg_data.csv)
    export_ekg_data_csv(
        raw,
        ekg_ch,
        out_csv=os.path.join(args.outdir, "ekg_data.csv"),
        chunk_seconds=args.chunk_seconds,
        total_samples=total_samples,
    )

    # 4) Export EKG-centric per-sample annotations
    export_ekg_with_annotations(
        raw,
        ekg_ch,
        ann_aligned,
        out_csv=os.path.join(args.outdir, "ekg_data_with_annotations.csv"),
        chunk_seconds=args.chunk_seconds,
        total_samples=total_samples,
        epoch_len_sec=args.epoch_len_sec,
    )

    print("Done.")
    print(f"EKG channel: {ekg_ch}")
    print(f"Sampling rate used: {sfreq} Hz")
    print(f"Total samples exported: {total_samples}")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
