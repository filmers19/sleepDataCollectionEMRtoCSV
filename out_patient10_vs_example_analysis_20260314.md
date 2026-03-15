# Patient_10 Example Comparison

- Reference patient: `최보영`
- Reference non-null fields: `247`

## Metrics

| Run | Ref-filled accuracy | All-field exact | Mismatches | Time (s) | Requests | Input tokens | Output tokens | Total tokens | Cost est. (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT54 | 94.74% | 97.98% | 14 | 479.5 | 54 | 382088 | 41694 | 423782 | 1.5806 |
| Gemini31ProPreview | 65.59% | 87.72% | 85 | 1597.1 | 81 | 409938 | 39501 | 584993 | 1.2939 |

## GPT54

- Run dir: `/home/yhlee/human_workspace/0_research/dataCollection/out_patient10_frozenlatestocr_route_gpt54_map_gpt54_resolve_keyrun_20260314`
- Stage counts: `{"map": 8, "resolve": 3, "output": 2, "ocr": 1}`
- Category counts: `{"table_subfield_omission": 4, "extraction_gap": 1, "document_conflict_chosen_differently": 2, "formatting_or_minor_text_normalization": 2, "wrong_conflict_resolution": 1, "placeholder_overinterpreted_as_zero": 1, "numeric_ocr_error": 1, "questionnaire_split_omission": 1, "time_field_split_error": 1}`
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
| PSG_M_04_WakeNo | 11 | 4 | ocr | The morning questionnaire count was OCRed as a different integer, and the mapper faithfully carried that wrong number through. |
| PSG_M_05_Complaint | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함. | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함 | output | The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant. |
| PSQI_02_Latency_HH | [blank] | 0 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_02_Latency_MM | 20 | 30 | map | The model extracted the sleep-latency question but decomposed the HH/MM fields incorrectly. |
| SQ_Wakefreq | 7 | 3 | resolve | The document contains competing candidates for this field, and the resolver selected a different value than the reference row. |

## Gemini31ProPreview

- Run dir: `/home/yhlee/human_workspace/0_research/dataCollection/out_patient10_live_gemini31propreview_all_20260314`
- Stage counts: `{"map": 71, "ocr": 12, "output": 2}`
- Category counts: `{"map_omission": 44, "upstream_ocr_loss": 2, "table_subfield_omission": 4, "extraction_gap": 3, "checkbox_alignment_error": 9, "metadata_assembly_gap": 1, "numeric_ocr_error": 1, "formatting_or_minor_text_normalization": 1, "questionnaire_split_omission": 20}`
- Cost notes: `gemini-3.1-pro-preview: Uses <=200k prompt tier and excludes any hidden thinking tokens not surfaced in usage logs. | Usage log has 135554 extra tokens beyond input+output; cost estimate is a lower bound unless those tokens are broken out.`

| Field | Actual | Expected | Likely stage | Cause |
| --- | --- | --- | --- | --- |
| AHI_lat | [blank] | 1.2 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_lat_N1 | [blank] | 0.7 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_lat_N2 | [blank] | 1.5 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_lat_N3 | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_lat_NREM | [blank] | 1.2 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_lat_REM | [blank] | 1.3 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_sup | [blank] | 3.4 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_sup_N1 | [blank] | 5.3 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_sup_N2 | [blank] | 1.7 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_sup_N3 | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_sup_NREM | [blank] | 3.4 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_sup_REM | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AHI_total | [blank] | 1.6 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_cent | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_cent_NREM | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_cent_REM | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_mix | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_mix_NREM | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_mix_REM | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_obs | [blank] | 0.7 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_obs_NREM | [blank] | 0.6 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| AI_obs_REM | [blank] | 1.3 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| Arousal_LM_idx | [blank] | 0.8 | ocr | The corresponding page OCR is much shorter or much less similar than the other run, so the downstream mapper never saw the needed content. |
| Arousal_LM_no | [blank] | 5 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| Arousal_PLM_idx | [blank] | 9.3 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| Arousal_PLM_idx_re | [blank] | 8.8 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| Arousal_PLM_no | [blank] | 54 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| Arousal_idx | [blank] | 28 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| Arousal_resp_idx | [blank] | 0.8 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| Arousal_snoring_idx | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| Arousal_spont_idx | [blank] | 16.6 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| BQ_01 | 2 | 3 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_02 | 2 | 3 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_03 | 1 | 2 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_04 | 3 | 4 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_05 | 1 | 2 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_07 | 1 | 2 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_08 | 1 | 2 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_09 | 1 | 2 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| BQ_10 | 1 | 2 | ocr | The Berlin Questionnaire OCR preserved the page, but checkbox positions drifted enough to shift selected options before mapping. |
| Database_ID | [blank] | 001_20140612_137_P | output | No provenance exists for this field; GPT populated it while the Gemini run left the metadata field blank. |
| Diagnosis_etc | [blank] | # Severe PLMS with moderate RLS # Moderate OSA with moderate snoring # Altered sleep structure with sleep maintanance difficulty # Mild depression | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| FSS_07 | [blank] | 5 | map | The source page is present and routed, but this field never made it into the mapped output. |
| FSS_08 | [blank] | 6 | map | The source page is present and routed, but this field never made it into the mapped output. |
| FSS_09 | [blank] | 6 | map | The source page is present and routed, but this field never made it into the mapped output. |
| HI | [blank] | 1 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| HI_NREM | [blank] | 1.1 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| HI_REM | [blank] | 0 | map | The field differs from the reference, but the available provenance does not isolate a narrower failure mode. |
| LM_idx | [blank] | 2.9 | ocr | The corresponding page OCR is much shorter or much less similar than the other run, so the downstream mapper never saw the needed content. |
| N1_pct | [blank] | 33.3 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| N2_pct | [blank] | 54.2 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| N3_pct | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| NREM_lat_min | [blank] | 252 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| NREM_sup_min | [blank] | 70 | map | The value exists in the PSG tables/leg-movement section, but this subfield was never emitted into provenance, so the mapper missed it entirely. |
| Nap_Freq | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| Neckcir_cm | 37 | 32 | ocr | A numeric field was misread in OCR, producing the wrong candidate value before mapping or resolution. |
| PLM_idx | [blank] | 52.5 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| PSG_M_05_Complaint | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함. | 잠에서 깨어났지만 머리가 맑지 않고 자기 전보다 더 피곤함 | output | The extracted content is substantively the same, but exact-string comparison fails because of whitespace, punctuation, or a minor spelling variant. |
| PSQI_01_BedIn_HH_free | [blank] | 23 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_01_BedIn_MM_free | [blank] | 0 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_03_BedOut_HH_free | [blank] | 7 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_03_BedOut_HH_week | [blank] | 6 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_03_BedOut_MM_free | [blank] | 30 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_03_BedOut_MM_week | [blank] | 30 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_04_SD_HH_free | [blank] | 7 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_04_SD_HH_week | [blank] | 6 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_04_SD_MM_free | [blank] | 0 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_04_SD_MM_week | [blank] | 30 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_05_d | [blank] | 3 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_05_e | [blank] | 1 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_05_f | [blank] | 0 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_05_g | [blank] | 0 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_05_h | [blank] | 2 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_05_i | [blank] | 1 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_06 | [blank] | 2 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_07 | [blank] | 0 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_08 | [blank] | 1 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| PSQI_09 | [blank] | 2 | map | The questionnaire OCR page is present, but the mapper only emitted part of the PSQI structure and dropped the rest of the split fields. |
| RDI_idx | [blank] | 1.8 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| RDI_no | [blank] | 11 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| REM_lat_min | [blank] | 46 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| REM_pct | [blank] | 12.5 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| REM_sup_min | [blank] | 0 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| SQ_Satisfaction | [blank] | 1 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
| WASO_pct | [blank] | 12.3 | map | The OCR page is present and comparable, but this run mapped far fewer keys from it than the other run, so the field was dropped during mapping. |
