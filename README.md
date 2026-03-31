# sleepDataCollectionEMRtoCSV

Utilities for acquiring structured sleep-study data from:
- questionnaire and paper form images
- PSG/CPAP paper reports
- EMR exports
- EDF-derived signal workflows

The main goal of this repository is to convert heterogeneous sleep-clinic source data into a consistent CDM-style CSV output.

## Repository Scope

This repository currently keeps the core acquisition pipeline only:
- [codebase_A_emr_to_csv](/home/yhlee/human_workspace/0_research/dataCollection/codebase_A_emr_to_csv): EMR-to-CSV scripts
- [codebase_B_paper_to_csv](/home/yhlee/human_workspace/0_research/dataCollection/codebase_B_paper_to_csv): OCR, split, map, resolve pipeline for paper questionnaires and PSG reports
- [101_edf_to_ekg_annotated_200Hz_2nd.py](/home/yhlee/human_workspace/0_research/dataCollection/101_edf_to_ekg_annotated_200Hz_2nd.py): EDF-derived signal processing utility
- [cropAndPrepareOCR.py](/home/yhlee/human_workspace/0_research/dataCollection/cropAndPrepareOCR.py): image cropping / OCR preparation utility
- [cdm_new.csv](/home/yhlee/human_workspace/0_research/dataCollection/cdm_new.csv): CDM key specification used by the paper pipeline

Patient datasets, labels, temporary outputs, and internal review files are intentionally not tracked.

## Main Paper Pipeline

Primary entry point:
- [111_unified_ocr_map_pipeline.py](/home/yhlee/human_workspace/0_research/dataCollection/codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py)

Core mapping logic:
- [103_paper_to_cdm_SA.py](/home/yhlee/human_workspace/0_research/dataCollection/codebase_B_paper_to_csv/103_paper_to_cdm_SA.py)

High-level flow:
1. OCR page images or reuse existing OCR text.
2. Merge OCR by patient.
3. Split OCR text into route/map categories such as `basic`, `phx_habit`, `mq`, `psg`, `cpap`, `psqi`, `ess`, `rbd`, and others.
4. Map each category into CDM key-value pairs with LLM-based mappers.
5. Validate and normalize mapped values against the CDM schema.
6. Merge category outputs into a final patient row.
7. Resolve multi-value conflicts deterministically.
8. Write final patient CSV and review reports.

## Output Schema

The pipeline no longer requires `example.csv` for normal unlabeled runs.

CSV column order is now loaded from:
- [output_schema_columns.json](/home/yhlee/human_workspace/0_research/dataCollection/codebase_B_paper_to_csv/output_schema_columns.json)

`example.csv` is only needed when you want labeled evaluation against a reference table.

## Typical Usage

### 1. Live OCR -> split -> map -> resolve

```bash
export OPENAI_API_KEY=YOUR_KEY

python codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py \
  --output_dir out_live_run \
  --input_root path/to/patient_images \
  --patient_name Patient_01 \
  --cdm_csv cdm_new.csv \
  --pipeline_mode ocr_map_resolve \
  --ocr_model_id gpt-5.4 \
  --map_model_id gpt-5.4 \
  --save_intermediate
```

### 2. Reuse OCR and rerun split/map/resolve

```bash
export OPENAI_API_KEY=YOUR_KEY

python codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py \
  --output_dir out_reuse_ocr_run \
  --input_root path/to/patient_images \
  --patient_name Patient_01 \
  --reuse_ocr_dir path/to/previous_run/Patient_01/ocr_pages \
  --cdm_csv cdm_new.csv \
  --pipeline_mode ocr_map_resolve \
  --map_model_id gpt-5.4 \
  --save_intermediate
```

### 3. Evaluation against labels

If you have a private label file, provide it explicitly:

```bash
python codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py \
  --output_dir out_eval_run \
  --input_root path/to/patient_images \
  --patient_name Patient_01 \
  --cdm_csv cdm_new.csv \
  --example_csv path/to/private_labels.csv \
  --eval_reference_index 1
```

## Review Outputs

A patient run can generate:
- final patient CSV
- final markdown review report
- conflict report if conflicting mapped values remain
- intermediate OCR / split / map artifacts when `--save_intermediate` is enabled

## Notes

- This repository assumes closed-model API access for the active paper pipeline.
- The current default OpenAI models in the unified runner are `gpt-5.4` for OCR and mapping.
- Some older helper scripts remain in `codebase_B_paper_to_csv`, but the active maintained workflow is centered on `103_paper_to_cdm_SA.py` and `111_unified_ocr_map_pipeline.py`.
