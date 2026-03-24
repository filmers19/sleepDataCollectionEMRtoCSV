# Category Pipeline Verification

## Structural Verification

- Syntax check passed:
  - `python -m py_compile codebase_B_paper_to_csv/103_paper_to_cdm_SA.py codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py`
- Unified runner CLI parses successfully:
  - `python codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py --help`

## Retrieval Verification

`CDMRetriever.category_rows(...)` now returns the shared `basic` slice plus the category-specific slice.

Observed row counts:

- `basic_questionnaire`: `48`
- `psg`: `359`
- `psqi`: `58`
- `sss`: `13`
- `ess`: `20`
- `fss`: `21`
- `berlin`: `22`
- `isi`: `19`
- `rls`: `29`
- `rbd`: `28`
- `phq`: `21`
- `bdi`: `33`
- `qol`: `38`

`basic_questionnaire` without shared `basic` rows:

- `36`

## Heuristic Fallback Verification

Known OCR samples:

- PSG sample -> `['psg']`
- ESS sample -> `['ess']`
- Mixed questionnaire page `0038` -> `['sss', 'ess', 'fss']`
- PSQI page `0042` -> `['psqi']`

## Current Limitation

Live end-to-end verification has not been run yet because the current shell does not have API credentials:

- `OPENAI_API_KEY_SET = False`
- `GOOGLE_API_KEY_SET = False`

That means the redesign has been verified structurally and locally, but not yet with a full live patient run.
