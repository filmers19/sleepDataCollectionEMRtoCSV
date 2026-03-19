from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE_RUN = Path(
    "out_patient01to10_liveocr_gpt54_route_gpt54_map_gpt51multiroute_resolve_gpt54_resumable_20260317"
)
OUT_CSV = Path("official_questionnaire_labels_psqi_wrong_patients_20260318.csv")
TARGET_PATIENTS = [
    "Patient_01",
    "Patient_04",
    "Patient_05",
    "Patient_06",
    "Patient_07",
    "Patient_08",
    "Patient_10",
]


def bundle_id(meta_stem: str) -> str:
    return meta_stem.replace(".meta", "")


MANUAL_FAMILY_BY_BUNDLE = {
    "Patient_01": {
        "bundle_0010__S20260210204833505232_JUYOUNG.KIM1102_email_0010": "RLS",
        "bundle_0011__S20260210204833505232_JUYOUNG.KIM1102_email_0011": "PSQI",
        "bundle_0012__S20260210204833505232_JUYOUNG.KIM1102_email_0012": "PSQI",
        "bundle_0018__S20260210204833505232_JUYOUNG.KIM1102_email_0018": "BQ",
        "bundle_0019__S20260210204833505232_JUYOUNG.KIM1102_email_0019": "SSS",
        "bundle_0020__S20260210204833505232_JUYOUNG.KIM1102_email_0020": "BDI",
        "bundle_0021__S20260210204833505232_JUYOUNG.KIM1102_email_0021": "BDI",
        "bundle_0022__S20260210204833505232_JUYOUNG.KIM1102_email_0022": "BDI",
        "bundle_0023__S20260210204833505232_JUYOUNG.KIM1102_email_0023": "MQ",
    },
    "Patient_04": {
        "bundle_0010__S20260210204833505232_JUYOUNG.KIM1102_email_0045": "PSQI",
        "bundle_0011__S20260210204833505232_JUYOUNG.KIM1102_email_0046": "MQ",
        "bundle_0012__S20260210204833505232_JUYOUNG.KIM1102_email_0047": "SSS",
        "bundle_0013__S20260210204833505232_JUYOUNG.KIM1102_email_0048": "BQ",
        "bundle_0014__S20260210204833505232_JUYOUNG.KIM1102_email_0049": "ISI",
        "bundle_0015__S20260210204833505232_JUYOUNG.KIM1102_email_0050": "BDI",
        "bundle_0016__S20260210204833505232_JUYOUNG.KIM1102_email_0051": "BDI",
        "bundle_0017__S20260210204833505232_JUYOUNG.KIM1102_email_0052": "BDI",
        "bundle_0018__S20260210204833505232_JUYOUNG.KIM1102_email_0053": "BDI",
    },
    "Patient_05": {
        "bundle_0010__S20260210204833505232_JUYOUNG.KIM1102_email_0065": "RLS",
        "bundle_0016__S20260210204833505232_JUYOUNG.KIM1102_email_0071": "BQ",
        "bundle_0017__S20260210204833505232_JUYOUNG.KIM1102_email_0072": "BDI",
        "bundle_0018__S20260210204833505232_JUYOUNG.KIM1102_email_0073": "BDI",
        "bundle_0019__S20260210204833505232_JUYOUNG.KIM1102_email_0074": "BDI",
        "bundle_0020__S20260210204833505232_JUYOUNG.KIM1102_email_0075": "PSQI",
        "bundle_0021__S20260210204833505232_JUYOUNG.KIM1102_email_0076": "PSQI",
        "bundle_0022__S20260210204833505232_JUYOUNG.KIM1102_email_0077": "SSS",
        "bundle_0023__S20260210204833505232_JUYOUNG.KIM1102_email_0078": "MQ",
    },
    "Patient_06": {
        "bundle_0005__S20260210201938294685_JUYOUNG.KIM1102_email_0005": "SSS",
        "bundle_0006__S20260210201938294685_JUYOUNG.KIM1102_email_0007": "ISI",
        "bundle_0007__S20260210201938294685_JUYOUNG.KIM1102_email_0008": "IRLS",
        "bundle_0008__S20260210201938294685_JUYOUNG.KIM1102_email_0009": "PSQI",
        "bundle_0011__S20260210201938294685_JUYOUNG.KIM1102_email_0012": "MQ",
        "bundle_0018__S20260210201938294685_JUYOUNG.KIM1102_email_0019": "QOL",
        "bundle_0019__S20260210201938294685_JUYOUNG.KIM1102_email_0020": "QOL",
        "bundle_0020__S20260210201938294685_JUYOUNG.KIM1102_email_0021": "QOL",
        "bundle_0021__S20260210201938294685_JUYOUNG.KIM1102_email_0022": "BDI",
        "bundle_0022__S20260210201938294685_JUYOUNG.KIM1102_email_0023": "BDI",
        "bundle_0023__S20260210201938294685_JUYOUNG.KIM1102_email_0024": "BDI",
    },
    "Patient_07": {
        "bundle_0005__S20260210201938294685_JUYOUNG.KIM1102_email_0029": "BQ",
        "bundle_0008__S20260210201938294685_JUYOUNG.KIM1102_email_0032": "ISI",
        "bundle_0009__S20260210201938294685_JUYOUNG.KIM1102_email_0033": "IRLS",
        "bundle_0010__S20260210201938294685_JUYOUNG.KIM1102_email_0034": "SSS",
        "bundle_0011__S20260210201938294685_JUYOUNG.KIM1102_email_0035": "PSQI",
        "bundle_0012__S20260210201938294685_JUYOUNG.KIM1102_email_0036": "BDI",
        "bundle_0013__S20260210201938294685_JUYOUNG.KIM1102_email_0037": "BDI",
        "bundle_0014__S20260210201938294685_JUYOUNG.KIM1102_email_0038": "BDI",
        "bundle_0015__S20260210201938294685_JUYOUNG.KIM1102_email_0039": "QOL",
        "bundle_0016__S20260210201938294685_JUYOUNG.KIM1102_email_0040": "QOL",
        "bundle_0017__S20260210201938294685_JUYOUNG.KIM1102_email_0041": "QOL",
        "bundle_0021__S20260210201938294685_JUYOUNG.KIM1102_email_0045": "MQ",
    },
    "Patient_08": {
        "bundle_0005__S20260210201938294685_JUYOUNG.KIM1102_email_0053": "SSS",
        "bundle_0006__S20260210201938294685_JUYOUNG.KIM1102_email_0054": "MQ",
        "bundle_0015__S20260210201938294685_JUYOUNG.KIM1102_email_0063": "SSS",
        "bundle_0016__S20260210201938294685_JUYOUNG.KIM1102_email_0064": "BQ",
        "bundle_0017__S20260210201938294685_JUYOUNG.KIM1102_email_0065": "ISI",
        "bundle_0018__S20260210201938294685_JUYOUNG.KIM1102_email_0066": "IRLS",
        "bundle_0019__S20260210201938294685_JUYOUNG.KIM1102_email_0067": "PSQI",
        "bundle_0020__S20260210201938294685_JUYOUNG.KIM1102_email_0068": "BDI",
        "bundle_0021__S20260210201938294685_JUYOUNG.KIM1102_email_0069": "BDI",
        "bundle_0022__S20260210201938294685_JUYOUNG.KIM1102_email_0070": "BDI",
        "bundle_0023__S20260210201938294685_JUYOUNG.KIM1102_email_0071": "QOL",
        "bundle_0024__S20260210201938294685_JUYOUNG.KIM1102_email_0072": "QOL",
        "bundle_0025__S20260210201938294685_JUYOUNG.KIM1102_email_0073": "QOL",
    },
    "Patient_10": {
        "bundle_0008__S20260210213237804797_JUYOUNG.KIM1102_email_0008": "MQ",
        "bundle_0014__S20260210213237804797_JUYOUNG.KIM1102_email_0014": "SSS",
        "bundle_0015__S20260210213237804797_JUYOUNG.KIM1102_email_0015": "BQ",
        "bundle_0016__S20260210213237804797_JUYOUNG.KIM1102_email_0016": "ISI",
        "bundle_0017__S20260210213237804797_JUYOUNG.KIM1102_email_0017": "IRLS",
        "bundle_0018__S20260210213237804797_JUYOUNG.KIM1102_email_0018": "PSQI",
        "bundle_0020__S20260210213237804797_JUYOUNG.KIM1102_email_0020": "BDI",
        "bundle_0021__S20260210213237804797_JUYOUNG.KIM1102_email_0021": "BDI",
        "bundle_0022__S20260210213237804797_JUYOUNG.KIM1102_email_0022": "BDI",
        "bundle_0023__S20260210213237804797_JUYOUNG.KIM1102_email_0023": "QOL",
        "bundle_0024__S20260210213237804797_JUYOUNG.KIM1102_email_0024": "QOL",
        "bundle_0025__S20260210213237804797_JUYOUNG.KIM1102_email_0025": "QOL",
    },
}


def main() -> None:
    rows = []
    for patient in TARGET_PATIENTS:
        for meta_path in sorted((SOURCE_RUN / patient / "map_pages").glob("*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            route = meta.get("map_route")
            if route not in {"map_route_night_questionnaire", "map_route_morning_questionnaire"}:
                continue
            src_images = meta.get("source_images") or []
            src = src_images[0] if src_images else ""
            ocr_name = Path(src).stem + ".txt" if src else ""
            ocr_path = SOURCE_RUN / patient / "ocr_pages" / ocr_name
            text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            snippet = " | ".join(lines[:10])[:1000]
            b = bundle_id(meta_path.stem)
            family = MANUAL_FAMILY_BY_BUNDLE.get(patient, {}).get(b, "NON")
            rows.append(
                {
                    "patient": patient,
                    "bundle": b,
                    "route": route,
                    "source_image": src,
                    "ocr_txt": ocr_name,
                    "manual_official": "1" if family != "NON" else "0",
                    "manual_family": family,
                    "snippet": snippet,
                }
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_CSV)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
