# Patient_10 Example Comparison

- Reference patient: `최보영`
- Reference non-null fields: `247`

## Metrics

| Run | Ref-filled accuracy | All-field exact | Mismatches | Time (s) | Requests | Input tokens | Output tokens | Total tokens | Cost est. (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeFirst | 93.52% | 97.69% | 16 | 499.1 | 56 | 358421 | 39582 | 398003 | 1.4898 |
| LLMAll | 94.74% | 97.98% | 14 | 479.5 | 54 | 382088 | 41694 | 423782 | 1.5806 |

## CodeFirst

- Run dir: `/home/yhlee/human_workspace/0_research/dataCollection/out_patient10_frozenlatestocr_route_gpt54_map_gpt54_resolve_codefirst_20260314`
- Stage counts: `{"map": 8, "resolve": 5, "output": 3}`
- Category counts: `{"table_subfield_omission": 4, "document_conflict_chosen_differently": 3, "metadata_assembly_gap": 1, "formatting_or_minor_text_normalization": 2, "wrong_conflict_resolution": 2, "map_omission": 1, "extraction_gap": 3}`
- Cost notes: `gpt-5.4: Assumes each request stayed at or below 272k input tokens.`

| Field | Actual | Expected | Likely stage | Cause |
| --- | --- | --- | --- | --- |
| Arousal_LM_no | [blank] | 5 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| Arousal_PLM_idx_re | 9.3 | 8.8 | resolve | The document contains competing candidates for this field, and the resolver selected a different value than the reference row. |
| Arousal_PLM_no | [blank] | 54 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| BMI | 18.8 | 28.8 | resolve | The document contains competing candidates for this field, and the resolver selected a different value than the reference row. |
| Database_ID | [blank] | 001_20140612_137_P | output | No provenance exists for this field; GPT populated it while the Gemini run left the metadata field blank. |
| Diagnosis_etc | # Severe PLMS with moderate RLS # Moderate OSA with moderate snoring # Altered sleep structure with sleep maintenance difficulty # Mild depression | # Severe PLMS with moderate RLS # Moderate OSA with moderate snoring # Altered sleep structure with sleep maintanance difficulty # Mild depression | output | The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant. |
| HI_REM | 1.3 | 0 | resolve | Multiple candidate values were extracted, and the resolver chose a different source than the one that matches the reference row. |
| NREM_lat_min | [blank] | 252 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| NREM_sup_min | [blank] | 70 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| Neckcir_cm | 31 | 32 | resolve | Multiple candidate values were extracted, and the resolver chose a different source than the one that matches the reference row. |
| PSG_M_02_SubSL_HH | [blank] | 0 | map | The source page is present and routed, but this field never made it into the mapped output. |
| PSG_M_04_WakeNo | 11 | 4 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| PSG_M_05_Complaint | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함. | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함 | output | The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant. |
| PSQI_02_Latency_HH | [blank] | 0 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| PSQI_02_Latency_MM | 20 | 30 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| SQ_Wakefreq | 4.5 | 3 | resolve | The document contains competing candidates for this field, and the resolver selected a different value than the reference row. |

## LLMAll

- Run dir: `/home/yhlee/human_workspace/0_research/dataCollection/out_patient10_frozenlatestocr_route_gpt54_map_gpt54_resolve_keyrun_20260314`
- Stage counts: `{"map": 9, "resolve": 3, "output": 2}`
- Category counts: `{"table_subfield_omission": 4, "extraction_gap": 4, "document_conflict_chosen_differently": 2, "formatting_or_minor_text_normalization": 2, "wrong_conflict_resolution": 1, "placeholder_overinterpreted_as_zero": 1}`
- Cost notes: `gpt-5.4: Assumes each request stayed at or below 272k input tokens.`

| Field | Actual | Expected | Likely stage | Cause |
| --- | --- | --- | --- | --- |
| Arousal_LM_no | [blank] | 5 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| Arousal_PLM_idx_re | 9.3 | 8.8 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| Arousal_PLM_no | [blank] | 54 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| BMI | 18.8 | 28.8 | resolve | The document contains competing candidates for this field, and the resolver selected a different value than the reference row. |
| Diagnosis_etc | # Severe PLMS with moderate RLS # Moderate OSA with moderate snoring # Altered sleep structure with sleep maintenance difficulty # Mild depression | # Severe PLMS with moderate RLS # Moderate OSA with moderate snoring # Altered sleep structure with sleep maintanance difficulty # Mild depression | output | The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant. |
| HI_REM | 1.3 | 0 | resolve | Multiple candidate values were extracted, and the resolver chose a different source than the one that matches the reference row. |
| NREM_lat_min | [blank] | 252 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| NREM_sup_min | [blank] | 70 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| N_Sleepattack | 0 | [blank] | map | The source form shows '-' for sleep attack, and the mapper converted that placeholder into 0 instead of leaving the field blank. |
| PSG_M_04_WakeNo | 11 | 4 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| PSG_M_05_Complaint | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함. | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함 | output | The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant. |
| PSQI_02_Latency_HH | [blank] | 0 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| PSQI_02_Latency_MM | 20 | 30 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| SQ_Wakefreq | 7 | 3 | resolve | The document contains competing candidates for this field, and the resolver selected a different value than the reference row. |
