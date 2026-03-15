from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


OPENAI_PRICING_PER_MTOKENS = {
    # Source as of 2026-03-14:
    # https://developers.openai.com/docs/pricing
    "gpt-5.4": {
        "input": 2.50,
        "output": 15.00,
        "notes": "Assumes each request stayed at or below 272k input tokens.",
    },
}

GEMINI_PRICING_PER_MTOKENS = {
    # Source as of 2026-03-14:
    # https://ai.google.dev/gemini-api/docs/pricing
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "notes": "Uses <=200k prompt tier and excludes any hidden thinking tokens not surfaced in usage logs.",
    },
}

KNOWN_TABLE_OMISSIONS = {
    "NREM_sup_min",
    "NREM_lat_min",
    "Arousal_PLM_no",
    "Arousal_LM_no",
}

KNOWN_MINOR_TEXT_FIELDS = {
    "Diagnosis_etc",
    "PSG_M_05_Complaint",
}


@dataclass
class RunData:
    label: str
    run_dir: Path
    final_row: Dict[str, str]
    provenance: Dict[str, Any]
    conflicts: Dict[str, Any]
    resolution: Dict[str, Any]
    plan: Dict[str, Any]
    summary: Dict[str, Any]
    usage: Dict[str, Any]
    ocr_texts: Dict[str, str]
    map_meta: Dict[str, Dict[str, Any]]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_single_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def load_run(label: str, run_dir: Path) -> RunData:
    ocr_dir = run_dir / "ocr_pages"
    map_dir = run_dir / "map_pages"
    ocr_texts = {}
    if ocr_dir.exists():
        ocr_texts = {p.name: p.read_text(encoding="utf-8") for p in sorted(ocr_dir.glob("*.txt"))}
    map_meta = {}
    if map_dir.exists():
        for p in sorted(map_dir.glob("*.meta.json")):
            map_meta[p.name] = load_json(p)
    return RunData(
        label=label,
        run_dir=run_dir,
        final_row=read_single_csv(run_dir / "Patient_10.csv"),
        provenance=load_json(run_dir / "provenance" / "Patient_10_provenance.json"),
        conflicts=load_json(run_dir / "conflicts" / "Patient_10_conflicts.json"),
        resolution=load_json(run_dir / "conflict_resolution" / "Patient_10_resolution.json"),
        plan=load_json(run_dir / "unified_plan.json"),
        summary=load_json(run_dir / "ocr_map_summary.json"),
        usage=load_json(run_dir / "openai_usage_summary.json"),
        ocr_texts=ocr_texts,
        map_meta=map_meta,
    )


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_loose(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[\s#.,:;()/%\-\[\]<>`'\"]+", "", text)


def is_minor_text_mismatch(field: str, actual: str, expected: str) -> bool:
    if field not in KNOWN_MINOR_TEXT_FIELDS:
        return False
    if strip_loose(actual) == strip_loose(expected):
        return True
    return SequenceMatcher(None, normalize_text(actual), normalize_text(expected)).ratio() >= 0.93


def parse_run_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --run value: {raw!r}. Expected LABEL=PATH")
    label, path = raw.split("=", 1)
    return label, Path(path).resolve()


def read_reference_row(example_csv: Path, patient_index: int) -> Dict[str, str]:
    with example_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if patient_index < 1 or patient_index > len(rows):
        raise ValueError(f"patient_index {patient_index} is out of range for {example_csv}")
    return rows[patient_index - 1]


def extract_page_id(name: str) -> str:
    match = re.search(r"_(\d{4})\.(?:txt|jpg|json)$", name)
    return match.group(1) if match else ""


def bundle_name_for_page(page_id: str) -> str:
    return f"bundle_{page_id}__S20260210213237804797_JUYOUNG.KIM1102_email_{page_id}.meta.json"


def ocr_file_name_for_page(page_id: str) -> str:
    return f"S20260210213237804797_JUYOUNG.KIM1102_email_{page_id}.txt"


def get_entries(provenance: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    entries = provenance.get(field)
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def candidate_values(entries: Iterable[Dict[str, Any]]) -> List[str]:
    values = []
    for entry in entries:
        if "value" in entry:
            values.append(normalize_text(entry.get("value")))
    return values


def page_evidence(run: RunData, other: RunData, page_id: str) -> Dict[str, Any]:
    ocr_name = ocr_file_name_for_page(page_id)
    meta_name = bundle_name_for_page(page_id)
    run_text = run.ocr_texts.get(ocr_name, "")
    other_text = other.ocr_texts.get(ocr_name, "")
    ratio = None
    if run_text and other_text:
        ratio = round(SequenceMatcher(None, run_text, other_text).ratio(), 4)
    return {
        "page_id": page_id,
        "run_ocr_chars": len(run_text),
        "other_ocr_chars": len(other_text),
        "ocr_similarity_vs_other": ratio,
        "run_route": run.map_meta.get(meta_name, {}).get("map_route"),
        "other_route": other.map_meta.get(meta_name, {}).get("map_route"),
        "run_valid_keys": run.map_meta.get(meta_name, {}).get("valid_keys"),
        "other_valid_keys": other.map_meta.get(meta_name, {}).get("valid_keys"),
    }


def usage_by_model(usage: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    # The pipeline writes deduped usage into the `ocr` and `route` sections.
    # `map` and `resolver` currently mirror the same non-OCR aggregate and would
    # triple-count cost if we summed every model-bearing section.
    for section_name in ("ocr", "route"):
        value = usage.get(section_name)
        if not isinstance(value, dict) or "model" not in value:
            continue
        model = str(value.get("model") or "")
        out[model]["requests"] += float(value.get("request_count") or 0)
        out[model]["input_tokens"] += float(value.get("input_tokens") or 0)
        out[model]["output_tokens"] += float(value.get("output_tokens") or 0)
        out[model]["total_tokens"] += float(value.get("total_tokens") or 0)
    return dict(out)


def estimate_cost(usage: Dict[str, Dict[str, float]]) -> Tuple[float | None, List[str]]:
    total_cost = 0.0
    notes: List[str] = []
    priced_any = False
    for model, stats in usage.items():
        pricing = OPENAI_PRICING_PER_MTOKENS.get(model) or GEMINI_PRICING_PER_MTOKENS.get(model)
        if not pricing:
            notes.append(f"No official pricing configured for {model}.")
            continue
        input_cost = stats["input_tokens"] / 1_000_000 * pricing["input"]
        output_cost = stats["output_tokens"] / 1_000_000 * pricing["output"]
        total_cost += input_cost + output_cost
        priced_any = True
        model_note = pricing.get("notes")
        if model_note:
            notes.append(f"{model}: {model_note}")
    return (round(total_cost, 4) if priced_any else None), notes


def classify_cause(
    field: str,
    actual: str,
    expected: str,
    run: RunData,
    other: RunData,
) -> Dict[str, Any]:
    entries = get_entries(run.provenance, field)
    other_entries = get_entries(other.provenance, field)
    resolution = run.resolution.get(field)
    conflict = run.conflicts.get(field)
    other_actual = normalize_text(other.final_row.get(field, ""))
    actual = normalize_text(actual)
    expected = normalize_text(expected)

    source_names = [entry.get("image", "") for entry in entries] or [entry.get("image", "") for entry in other_entries]
    page_ids = sorted({extract_page_id(name) for name in source_names if extract_page_id(name)})
    evidence = [page_evidence(run, other, page_id) for page_id in page_ids]

    if field == "Database_ID":
        return {
            "stage": "output",
            "category": "metadata_assembly_gap",
            "cause": "No provenance exists for this field; GPT populated it while the Gemini run left the metadata field blank.",
            "evidence": evidence,
        }

    if is_minor_text_mismatch(field, actual, expected):
        return {
            "stage": "output",
            "category": "formatting_or_minor_text_normalization",
            "cause": "The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant.",
            "evidence": evidence,
        }

    if field == "N_Sleepattack" and actual == "0" and expected == "":
        return {
            "stage": "map",
            "category": "placeholder_overinterpreted_as_zero",
            "cause": "The source form shows '-' for sleep attack, and the mapper converted that placeholder into 0 instead of leaving the field blank.",
            "evidence": evidence,
        }

    if field.startswith("BQ_"):
        return {
            "stage": "ocr",
            "category": "checkbox_alignment_error",
            "cause": "The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping.",
            "evidence": evidence,
        }

    if field in KNOWN_TABLE_OMISSIONS and actual == "":
        return {
            "stage": "map",
            "category": "table_subfield_omission",
            "cause": "The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely.",
            "evidence": evidence,
        }

    if field.startswith("PSQI_") and actual == "" and other_actual == expected:
        return {
            "stage": "map",
            "category": "questionnaire_split_omission",
            "cause": "The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields.",
            "evidence": evidence,
        }

    if field in {"PSQI_02_Latency_HH", "PSQI_02_Latency_MM"} and other_actual == expected and actual != expected:
        return {
            "stage": "map",
            "category": "time_field_split_error",
            "cause": "The model extracted the sleep-latency question but decomposed the HH/MM fields incorrectly.",
            "evidence": evidence,
        }

    if resolution or len(entries) > 1 or conflict:
        entry_values = candidate_values(entries)
        if any(strip_loose(expected) == strip_loose(value) for value in entry_values):
            return {
                "stage": "resolve",
                "category": "wrong_conflict_resolution",
                "cause": "Multiple candidate values were extracted, and the resolver chose a different source than the one that matches the reference row.",
                "evidence": evidence,
            }
        return {
            "stage": "resolve",
            "category": "document_conflict_chosen_differently",
            "cause": "The document contains competing candidates for this field, and the resolver selected a different value than the reference row.",
            "evidence": evidence,
        }

    if actual == "" and other_actual == expected:
        def severe_drop(item: Dict[str, Any]) -> bool:
            return (
                item.get("run_ocr_chars", 0) < max(200, int((item.get("other_ocr_chars", 0) or 0) * 0.4))
                or (item.get("ocr_similarity_vs_other") is not None and item.get("ocr_similarity_vs_other", 1.0) < 0.6)
            )

        severe_ocr_drop = any(severe_drop(item) for item in evidence)
        good_ocr_but_low_map = any(
            item.get("run_ocr_chars", 0) >= max(400, int((item.get("other_ocr_chars", 0) or 0) * 0.8))
            and (item.get("other_valid_keys") or 0) >= (item.get("run_valid_keys") or 0) + 10
            for item in evidence
        )
        route_drift = any(item.get("run_route") != item.get("other_route") for item in evidence)
        if good_ocr_but_low_map:
            return {
                "stage": "map",
                "category": "map_omission",
                "cause": "The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping.",
                "evidence": evidence,
            }
        if severe_ocr_drop:
            return {
                "stage": "ocr",
                "category": "upstream_ocr_loss",
                "cause": "The corresponding page OCR is much shorter or much less similar than the other run, so the downstream mapper never saw the needed content.",
                "evidence": evidence,
            }
        if route_drift:
            return {
                "stage": "route",
                "category": "route_scope_loss",
                "cause": "The page was routed differently than the correct run, shrinking the CDM scope and suppressing extraction of the needed field.",
                "evidence": evidence,
            }
        return {
            "stage": "map",
            "category": "map_omission",
            "cause": "The source page is present and routed, but this field never made it into the mapped output.",
            "evidence": evidence,
        }

    if actual != "" and other_actual == expected:
        severe_ocr_drop = any(
            item.get("run_ocr_chars", 0) and item.get("other_ocr_chars", 0) and item.get("ocr_similarity_vs_other", 1.0) < 0.8
            for item in evidence
        )
        if field == "PSG_M_04_WakeNo":
            return {
                "stage": "ocr",
                "category": "numeric_ocr_error",
                "cause": "The morning questionnaire count was OCRed as a different integer, and the mapper faithfully carried that wrong number through.",
                "evidence": evidence,
            }
        if field in {"Neckcir_cm", "BMI"} and severe_ocr_drop:
            return {
                "stage": "ocr",
                "category": "numeric_ocr_error",
                "cause": "A numeric field was misread in OCR, producing the wrong candidate value before mapping or resolution.",
                "evidence": evidence,
            }
        return {
            "stage": "map",
            "category": "extraction_or_parsing_error",
            "cause": "The field was extracted from a source page, but the selected value does not match the reference even though another run did.",
            "evidence": evidence,
        }

    return {
        "stage": "map",
        "category": "extraction_gap",
        "cause": "The field differs from the reference, but the available provenance does not isolate a narrower failure mode.",
        "evidence": evidence,
    }


def compare_run_to_reference(run: RunData, other: RunData, reference: Dict[str, str]) -> Dict[str, Any]:
    all_fields = sorted(set(reference) | set(run.final_row))
    reference_non_null = sum(1 for value in reference.values() if normalize_text(value) != "")
    all_matches = 0
    filled_matches = 0
    mismatches: List[Dict[str, Any]] = []
    for field in all_fields:
        actual = normalize_text(run.final_row.get(field, ""))
        expected = normalize_text(reference.get(field, ""))
        if actual == expected:
            all_matches += 1
            if expected != "":
                filled_matches += 1
            continue
        if expected != "" and actual == expected:
            filled_matches += 1
        cause = classify_cause(field, actual, expected, run, other)
        mismatches.append(
            {
                "field": field,
                "actual": actual,
                "expected": expected,
                "other_run_actual": normalize_text(other.final_row.get(field, "")),
                "analysis": cause,
            }
        )
    for field, expected in reference.items():
        actual = normalize_text(run.final_row.get(field, ""))
        expected = normalize_text(expected)
        if expected != "" and actual == expected:
            filled_matches += 0

    usage_models = usage_by_model(run.usage)
    cost_estimate, cost_notes = estimate_cost(usage_models)
    total_tokens = run.summary.get("openai_usage_total_tokens")
    input_tokens = run.summary.get("openai_usage_input_tokens")
    output_tokens = run.summary.get("openai_usage_output_tokens")
    if isinstance(total_tokens, (int, float)) and isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        extra_tokens = total_tokens - input_tokens - output_tokens
        if extra_tokens > 0:
            cost_notes.append(
                f"Usage log has {int(extra_tokens)} extra tokens beyond input+output; cost estimate is a lower bound unless those tokens are broken out."
            )
    stage_counts = Counter(item["analysis"]["stage"] for item in mismatches)
    category_counts = Counter(item["analysis"]["category"] for item in mismatches)
    return {
        "run_label": run.label,
        "run_dir": str(run.run_dir),
        "total_fields": len(all_fields),
        "reference_non_null_fields": reference_non_null,
        "all_field_exact_matches": all_matches,
        "all_field_exact_accuracy": round(all_matches / len(all_fields), 4) if all_fields else None,
        "reference_non_null_exact_matches": reference_non_null - sum(1 for item in mismatches if item["expected"] != ""),
        "reference_non_null_exact_accuracy": round((reference_non_null - sum(1 for item in mismatches if item["expected"] != "")) / reference_non_null, 4) if reference_non_null else None,
        "mismatch_count": len(mismatches),
        "non_null_output_fields": run.summary.get("output_row_non_null_keys"),
        "total_elapsed_seconds": run.summary.get("total_elapsed_seconds"),
        "usage_by_model": usage_models,
        "request_count": run.summary.get("openai_usage_request_count"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_estimate_usd": cost_estimate,
        "cost_notes": cost_notes,
        "mismatch_stage_counts": dict(stage_counts),
        "mismatch_category_counts": dict(category_counts),
        "mismatches": mismatches,
    }


def build_markdown(reference: Dict[str, str], run_reports: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Patient_10 Example Comparison")
    lines.append("")
    lines.append(f"- Reference patient: `{normalize_text(reference.get('Name', ''))}`")
    lines.append(f"- Reference non-null fields: `{sum(1 for v in reference.values() if normalize_text(v) != '')}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Run | Ref-filled accuracy | All-field exact | Mismatches | Time (s) | Requests | Input tokens | Output tokens | Total tokens | Cost est. (USD) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for report in run_reports:
        cost = "" if report["cost_estimate_usd"] is None else f"{report['cost_estimate_usd']:.4f}"
        lines.append(
            "| {run} | {filled:.2%} | {all_acc:.2%} | {mismatch} | {sec:.1f} | {req} | {inp} | {out} | {tot} | {cost} |".format(
                run=report["run_label"],
                filled=report["reference_non_null_exact_accuracy"] or 0.0,
                all_acc=report["all_field_exact_accuracy"] or 0.0,
                mismatch=report["mismatch_count"],
                sec=float(report["total_elapsed_seconds"] or 0.0),
                req=int(report["request_count"] or 0),
                inp=int(report["input_tokens"] or 0),
                out=int(report["output_tokens"] or 0),
                tot=int(report["total_tokens"] or 0),
                cost=cost,
            )
        )
    lines.append("")
    for report in run_reports:
        lines.append(f"## {report['run_label']}")
        lines.append("")
        lines.append(f"- Run dir: `{report['run_dir']}`")
        lines.append(f"- Stage counts: `{json.dumps(report['mismatch_stage_counts'], ensure_ascii=False)}`")
        lines.append(f"- Category counts: `{json.dumps(report['mismatch_category_counts'], ensure_ascii=False)}`")
        if report["cost_notes"]:
            lines.append(f"- Cost notes: `{' | '.join(report['cost_notes'])}`")
        lines.append("")
        lines.append("| Field | Actual | Expected | Likely stage | Cause |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in report["mismatches"]:
            lines.append(
                "| {field} | {actual} | {expected} | {stage} | {cause} |".format(
                    field=item["field"],
                    actual=(item["actual"] or "[blank]").replace("|", "\\|"),
                    expected=(item["expected"] or "[blank]").replace("|", "\\|"),
                    stage=item["analysis"]["stage"],
                    cause=item["analysis"]["cause"].replace("|", "\\|"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example_csv", required=True)
    ap.add_argument("--patient_index", type=int, required=True)
    ap.add_argument("--run", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    parsed_runs = [parse_run_arg(item) for item in args.run]
    if len(parsed_runs) != 2:
        raise ValueError("This analyzer currently expects exactly two --run values.")

    reference = read_reference_row(Path(args.example_csv).resolve(), args.patient_index)
    first = load_run(*parsed_runs[0])
    second = load_run(*parsed_runs[1])

    reports = [
        compare_run_to_reference(first, second, reference),
        compare_run_to_reference(second, first, reference),
    ]
    report = {
        "reference": {
            "patient_index": args.patient_index,
            "name": normalize_text(reference.get("Name", "")),
            "non_null_fields": sum(1 for value in reference.values() if normalize_text(value) != ""),
        },
        "runs": reports,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.output_md:
        Path(args.output_md).write_text(build_markdown(reference, reports), encoding="utf-8")


if __name__ == "__main__":
    main()
