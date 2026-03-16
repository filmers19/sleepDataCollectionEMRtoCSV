# CPAP PSG Support Plan

## Goal

Add CPAP titration polysomnography support without bloating `cdm_revised.csv`, while keeping the current OCR -> route -> map -> resolve pipeline architecture intact.

## Design Decisions

### 1. New route types

Add two CPAP-specific route types:

- `map_route_cpap_psg_report_general`
- `map_route_cpap_psg_report_extensive`

Meaning:

- `map_route_cpap_psg_report_general`
  - doctor/staff-authored CPAP titration PSG report page
  - contains standard PSG summary metrics and CPAP pressure-step metrics
- `map_route_cpap_psg_report_extensive`
  - same as the CPAP general route
  - additionally identified by the keyword `FULL NIGHT CPAP POLYSOMNOGRAPHY REPORT`
  - uses the same split-map behavior as the existing extensive PSG route

### 2. Route-to-CDM distribution

CPAP PSG pages share:

- `CORE_ALWAYS_KEYS`
- common PSG report metrics
- CPAP-only pressure-step metrics

The CPAP routes must therefore use:

- `psg_report`
- plus `cpap_pressure`

This avoids duplicating the normal PSG family while still giving CPAP pages their pressure-step fields.

### 3. Long-term schema design

Do not manually store every CPAP step row in `cdm_revised.csv`.

Instead:

- keep the `Pressure_05` / `Pr05_*` rows as the canonical prototype block
- expand them in code to `06..29`
- if explicit rows already exist for some generated keys, do not overwrite them

This preserves:

- explicit candidate keys for the map agent
- strict validation against known `CDMRow`s
- flat CSV output compatibility with `example.csv`

## Engineering Plan

### Routing

Update:

- route constants
- route descriptions
- route normalization aliases
- heuristic CPAP scoring
- router prompt options

CPAP-specific heuristic signals:

- `CPAP polysomnography`
- `CPAP titration`
- `PAP titration`
- `nasal CPAP titration`
- `cmH2O`
- pressure-step lines such as `Pressure 5 cmH2O`
- `FULL NIGHT CPAP POLYSOMNOGRAPHY REPORT` for the extensive route

### CDM distribution

Add one document label family:

- `cpap_pressure`

This family matches:

- `Pressure_XX`
- `PrXX_*`

Keep `psg_report` focused on the ordinary PSG report fields only.

### Split behavior

`map_route_cpap_psg_report_extensive` should split OCR text exactly like `map_route_psg_report_extensive`.

### Map prompt

No map prompt change in this implementation.

Reason:

- the CPAP routes now deliver the correct candidate key set
- prompt changes can be evaluated later if mapping recall on CPAP pages is still insufficient

## Implemented

Implemented in `codebase_B_paper_to_csv/103_paper_to_cdm_SA.py`:

- CPAP general/extensive route constants and descriptions
- route normalization aliases for CPAP route names
- heuristic CPAP route detection
- router prompt updated from 4 routes to 6 routes
- CPAP extensive split behavior
- new `cpap_pressure` label family
- CPAP route distribution = `psg_report + cpap_pressure`
- virtual CPAP pressure-step row expansion from prototype `05` rows to `06..29`

## Validation Performed

- `py_compile` passed for:
  - `codebase_B_paper_to_csv/103_paper_to_cdm_SA.py`
  - `codebase_B_paper_to_csv/111_unified_ocr_map_pipeline.py`
- runtime sanity check confirmed:
  - generated keys include `Pressure_29` and `Pr29_arousal_spont_idx`
  - CPAP route includes core identity keys
  - CPAP route includes ordinary PSG metrics
  - CPAP route includes CPAP-only pressure-step metrics
  - CPAP extensive route uses split behavior
  - heuristic routing classifies `FULL NIGHT CPAP POLYSOMNOGRAPHY REPORT` as `map_route_cpap_psg_report_extensive`

## Deferred / Follow-up

- If CPAP mapping recall is weak, consider a targeted map prompt refinement later.
- If you later shorten `cdm_revised.csv`, the current virtual-expansion code is already compatible with that cleanup.
