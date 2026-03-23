from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PSG_RDI_SPO2_KEYS = ["RDI_no", "RDI_idx", "Lowest_SpO2"]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_csv_row_by_patient(path: Path) -> dict[str, dict[str, str]]:
    rows = load_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        patient = row.get("patient") or row.get("Patient_no") or row.get("Patient")
        if patient:
            out[patient] = row
    return out


def load_single_row_csv(path: Path) -> dict[str, str]:
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def parse_threshold_percent(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)%", text or "")
    if not m:
        raise ValueError(f"Could not parse threshold percent from: {text}")
    return float(m.group(1)) / 100.0


def discover_full_cpap_keys(example_csv: Path) -> list[str]:
    with example_csv.open(newline="", encoding="utf-8-sig") as f:
        headers = csv.DictReader(f).fieldnames or []
    keys = []
    for key in headers:
        if re.fullmatch(r"Pressure_(0[5-9]|1\d|2\d)", key):
            keys.append(key)
            continue
        if re.fullmatch(r"Pr(0[5-9]|1\d|2\d)_.+", key):
            keys.append(key)
    return sorted(set(keys))


def build_cdm_groups(cdm_csv: Path, example_csv: Path) -> dict[str, list[str]]:
    rows = load_csv_rows(cdm_csv)
    groups: dict[str, list[str]] = {
        "Basic": [],
        "Demographics": [],
        "CoreQ": [],
        "SubQ": [],
        "PSG": [],
    }
    for row in rows:
        cdm_type = (row.get("cdm type") or "").strip()
        key = (row.get("csv key") or "").strip()
        if not key:
            continue
        if cdm_type in groups:
            groups[cdm_type].append(key)

    groups["PSG-RDI/SpO2"] = list(PSG_RDI_SPO2_KEYS)
    groups["PSG"] = [k for k in groups["PSG"] if k not in PSG_RDI_SPO2_KEYS]
    groups["CPAP"] = discover_full_cpap_keys(example_csv)
    return groups


def load_example_row(example_csv: Path, patient_num: int) -> dict[str, str]:
    with example_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    idx = patient_num - 1
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"No example.csv row for patient index {patient_num}")
    return rows[idx]


def normalize(v: str | None) -> str:
    return (v or "").strip()


def count_correct(keys: list[str], actual: dict[str, str], expected: dict[str, str]) -> int:
    return sum(1 for key in keys if normalize(actual.get(key)) == normalize(expected.get(key)))


def conflict_keys_for_patient(patient_dir: Path) -> set[str]:
    conflict_path = patient_dir / "conflicts" / f"{patient_dir.name}_conflicts.json"
    if not conflict_path.exists():
        return set()
    data = json.loads(conflict_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return set(data.keys())
    return set()


def format_acc(correct: int | None, total: int | None, status: str | None = None) -> str:
    if status:
        return status
    assert correct is not None and total is not None
    pct = (correct / total * 100.0) if total else 0.0
    return f"{correct}/{total} ({pct:.2f}%)"


def format_stage_status(initial_pass: bool | None, conflict_pass: bool | None, status: str | None = None) -> str:
    if status:
        return status
    return f"{'PASS' if initial_pass else 'FAIL'} / {'PASS' if conflict_pass else 'FAIL'}"


def write_table(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in fieldnames) + " |\n")


def summarize_status(patient_dir: Path) -> tuple[str, str]:
    summary_path = patient_dir / "ocr_map_summary.json"
    if summary_path.exists():
        return "COMPLETE", ""
    log_path = patient_dir.parent / f"{patient_dir.name}.run.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if "insufficient_quota" in text:
            return "RUN_INCOMPLETE_QUOTA", "OpenAI OCR quota exhausted during fresh rerun"
        return "RUN_INCOMPLETE", "Fresh rerun did not finish"
    return "MISSING", "Patient output directory not completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cdm-csv", required=True)
    parser.add_argument("--example-csv", required=True)
    parser.add_argument("--stages-csv", required=True)
    parser.add_argument("--patients", nargs="+", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    cdm_csv = Path(args.cdm_csv)
    example_csv = Path(args.example_csv)
    stages_csv = Path(args.stages_csv)
    patients = args.patients

    groups = build_cdm_groups(cdm_csv, example_csv)
    stages = load_csv_rows(stages_csv)
    thresholds = {
        int(row["단계"]): {
            "cdm_types": [x.strip() for x in row["cdm type"].split(",")],
            "initial_pct": parse_threshold_percent(row["1차 성능"]),
            "conflict_pct": parse_threshold_percent(row["Conflict 포함 성능"]),
        }
        for row in stages
    }

    status_rows: list[dict[str, str]] = []
    cdm_rows: list[dict[str, str]] = []
    stage_rows: list[dict[str, str]] = []
    totals = {
        stage: {"initial_correct": 0, "conflict_correct": 0, "total": 0, "completed_patients": 0}
        for stage in thresholds
    }

    for patient in patients:
        patient_dir = run_root / patient
        status, detail = summarize_status(patient_dir)
        status_rows.append({"Patient_no": patient, "run_status": status, "detail": detail})

        if status != "COMPLETE":
            incomplete = "RUN_INCOMPLETE_QUOTA" if status == "RUN_INCOMPLETE_QUOTA" else "RUN_INCOMPLETE"
            cdm_row = {"Patient_no": patient}
            for name in ["Basic", "Demographics", "CoreQ", "SubQ", "PSG-RDI/SpO2", "PSG", "CPAP"]:
                cdm_row[name] = incomplete
            cdm_rows.append(cdm_row)

            stage_row = {"Patient_no": patient}
            for stage in [1, 2, 3, 4]:
                stage_row[f"S{stage}_initial"] = incomplete
                stage_row[f"S{stage}_conflict_enhanced"] = incomplete
                stage_row[f"S{stage}_pass_fail"] = incomplete
            stage_row["highest_stage_initial"] = incomplete
            stage_row["highest_stage_conflict"] = incomplete
            stage_rows.append(stage_row)
            continue

        patient_num = int(patient.split("_")[1])
        expected = load_example_row(example_csv, patient_num)
        actual_path = patient_dir / f"{patient}.csv"
        actual = load_single_row_csv(actual_path)
        conflicts = conflict_keys_for_patient(patient_dir)

        cdm_row = {"Patient_no": patient}
        for name in ["Basic", "Demographics", "CoreQ", "SubQ", "PSG-RDI/SpO2", "PSG", "CPAP"]:
            keys = groups[name]
            correct = count_correct(keys, actual, expected)
            cdm_row[name] = format_acc(correct, len(keys))
        cdm_rows.append(cdm_row)

        stage_row = {"Patient_no": patient}
        highest_initial = 0
        highest_conflict = 0
        for stage in [1, 2, 3, 4]:
            stage_keys: list[str] = []
            for cdm_type in thresholds[stage]["cdm_types"]:
                stage_keys.extend(groups[cdm_type])
            stage_keys = list(dict.fromkeys(stage_keys))
            total = len(stage_keys)
            initial_correct = count_correct(stage_keys, actual, expected)
            conflict_correct = initial_correct + sum(
                1
                for key in stage_keys
                if normalize(actual.get(key)) != normalize(expected.get(key)) and key in conflicts
            )
            initial_pass = (initial_correct / total) >= thresholds[stage]["initial_pct"]
            conflict_pass = (conflict_correct / total) >= thresholds[stage]["conflict_pct"]
            if initial_pass:
                highest_initial = stage
            if conflict_pass:
                highest_conflict = stage
            stage_row[f"S{stage}_initial"] = format_acc(initial_correct, total)
            stage_row[f"S{stage}_conflict_enhanced"] = format_acc(conflict_correct, total)
            stage_row[f"S{stage}_pass_fail"] = format_stage_status(initial_pass, conflict_pass)

            totals[stage]["initial_correct"] += initial_correct
            totals[stage]["conflict_correct"] += conflict_correct
            totals[stage]["total"] += total
            totals[stage]["completed_patients"] += 1

        stage_row["highest_stage_initial"] = str(highest_initial)
        stage_row["highest_stage_conflict"] = str(highest_conflict)
        stage_rows.append(stage_row)

    total_rows: list[dict[str, str]] = []
    for stage in [1, 2, 3, 4]:
        total = totals[stage]["total"]
        init_correct = totals[stage]["initial_correct"]
        conf_correct = totals[stage]["conflict_correct"]
        init_pass = (init_correct / total) >= thresholds[stage]["initial_pct"] if total else False
        conf_pass = (conf_correct / total) >= thresholds[stage]["conflict_pct"] if total else False
        total_rows.append(
            {
                "Stage": str(stage),
                "cdm_types": ", ".join(thresholds[stage]["cdm_types"]),
                "completed_patients": str(totals[stage]["completed_patients"]),
                "current_schema_total_per_patient": str(total // max(totals[stage]["completed_patients"], 1)),
                "total_initial_acc": format_acc(init_correct, total),
                "total_conflict_enhanced_acc": format_acc(conf_correct, total),
                "pass_fail": format_stage_status(init_pass, conf_pass),
            }
        )

    manifest = {
        "run_root": str(run_root),
        "patients_requested": patients,
        "patients_completed": [r["Patient_no"] for r in status_rows if r["run_status"] == "COMPLETE"],
        "patients_incomplete": [r["Patient_no"] for r in status_rows if r["run_status"] != "COMPLETE"],
        "schema": "current_schema_full_cpap",
        "psg_rdi_spo2_keys": PSG_RDI_SPO2_KEYS,
        "cpap_key_count": len(groups["CPAP"]),
        "note": "Fresh rerun only. Incomplete patients are marked RUN_INCOMPLETE* and excluded from stage totals.",
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    outputs = [
        ("run_status_subset8_freshrerun.csv", ["Patient_no", "run_status", "detail"], status_rows),
        (
            "patient_cdmtype_accuracy_table_subset8_freshrerun_current_schema_full_cpap.csv",
            ["Patient_no", "Basic", "Demographics", "CoreQ", "SubQ", "PSG-RDI/SpO2", "PSG", "CPAP"],
            cdm_rows,
        ),
        (
            "patient_stage_accuracy_table_subset8_freshrerun_current_schema_full_cpap.csv",
            [
                "Patient_no",
                "S1_initial",
                "S1_conflict_enhanced",
                "S1_pass_fail",
                "S2_initial",
                "S2_conflict_enhanced",
                "S2_pass_fail",
                "S3_initial",
                "S3_conflict_enhanced",
                "S3_pass_fail",
                "S4_initial",
                "S4_conflict_enhanced",
                "S4_pass_fail",
                "highest_stage_initial",
                "highest_stage_conflict",
            ],
            stage_rows,
        ),
        (
            "stage_total_accuracy_table_completed7_freshrerun_current_schema_full_cpap.csv",
            [
                "Stage",
                "cdm_types",
                "completed_patients",
                "current_schema_total_per_patient",
                "total_initial_acc",
                "total_conflict_enhanced_acc",
                "pass_fail",
            ],
            total_rows,
        ),
    ]
    for filename, fieldnames, rows in outputs:
        csv_path = run_root / filename
        md_path = csv_path.with_suffix(".md")
        write_table(csv_path, fieldnames, rows)
        write_markdown(md_path, fieldnames, rows)


if __name__ == "__main__":
    main()
