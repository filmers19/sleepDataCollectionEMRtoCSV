# Category Pipeline Migration Plan

## Goal

Replace the route-based per-page mapping workflow with a patient-level category-based workflow:

1. OCR all pages for one patient.
2. Merge OCR text with source-page delimiters.
3. Categorize pages into:
   - `basic_questionnaire`
   - `psg`
   - `psqi`
   - `sss`
   - `ess`
   - `fss`
   - `berlin`
   - `isi`
   - `rls`
   - `rbd`
   - `phq`
   - `bdi`
   - `qol`
4. Merge non-adjacent OCR text blocks that share the same category.
5. Run one category-specific map pass per populated category with:
   - the shared `basic` CDM slice
   - the category-specific CDM slice from `cdm_revised.csv`
6. Keep merge + resolve downstream behavior.

## Review

The redesign is good for this codebase because:

- `cdm_revised.csv` already has a `map category` column, so the schema supports category-specific retrieval natively.
- A patient-level category split avoids the current route/family/type branching and should reduce prompt ambiguity.
- Deterministic category text merging is safer than asking a model to rewrite OCR text by category.

Main engineering risks:

- A category splitter that rewrites text would introduce OCR drift.
  - Mitigation: split agent outputs only page-to-category assignments; merged category text is rebuilt deterministically from saved OCR.
- PSG reports can mention questionnaire scores without being questionnaire pages.
  - Mitigation: heuristic fallback treats PSG as PSG and only uses questionnaire categories on explicit questionnaire pages.
- Provenance can become less page-specific after category merging.
  - Mitigation: category meta stores `source_images`, and input contexts carry category labels.

## Implementation Steps

1. Extend CDM retrieval with `map category` awareness.
2. Add patient-level category split prompts, normalization, and heuristic fallback.
3. Add category-specific map prompts and category-specific candidate selection.
4. Switch unified pipeline orchestration from:
   - OCR -> route -> family/type -> route-map
   to:
   - OCR -> category split -> category merge -> category-map
5. Keep conflict resolver and evaluation unchanged.

## Verification Steps

1. Syntax verification
   - `python -m py_compile codebase_B_paper_to_csv/103_paper_to_cdm_SA.py codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py`
2. Category retrieval verification
   - confirm each category retrieves `basic` + category-specific CDM rows
3. Heuristic fallback verification on known OCR pages
   - PSG sample -> `psg`
   - ESS sample -> `ess`
   - mixed SSS/ESS/FSS page -> `sss`, `ess`, `fss`
   - PSQI page -> `psqi`
4. Live end-to-end verification
   - run patient subset with reused OCR or live OCR
   - compare accuracy against previous route-based baseline

## Cleanup Status

- Active unified pipeline now uses the category-based design.
- Active map prompts no longer depend on route/family/type tags.
- Route/family helper code still exists in the core module for backward compatibility and can be removed in a later cleanup pass after live validation.
