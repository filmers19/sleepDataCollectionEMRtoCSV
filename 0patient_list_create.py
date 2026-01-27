import pandas as pd
import os
from datetime import timedelta

# ==========================================
# --- CONFIGURATION SECTION ---
# ==========================================
NUM_FILES = 1          # Number of SheetX.csv files to read
TARGET_YEAR = "2024"   # Configurable Year 
START_SEQ_NUM = 1      # Starting number for sequences
# ==========================================

# 1. Define the Master DataFrame Structure
columns_str = """Hospital_ID    Name    Lab_ID  Device_Type PSG_Date    PSG_No  PSG_Type    Database_ID Previous_Data   SEX AGE Height_cm   Weight_kg   BMI Neckcir_cm  Occupation_cat  Occupation  Shiftwork   PHx_CVA PHx_Parkinson   PHx_PNS PHx_Epilepsy    PHx_Dementia    PHx_Alcoholism  PHx_Cancer  PHx_Renal   PHx_Pulmonary   PHx_HTN PHx_Thyroid PHx_Liver   PHx_DM  PHx_Cardiovascular  PHx_NasalFx PHx_Sinusitis   PHx_GERD_ulcer  PHx_Psy Habit_Caffein   Habit_Alcohol   Habit_Smoking   Habit_Workout   Diagnosis_I Diagnosis_II    Diagnosis_III   Diagnosis_IV    Diagnosis_V Diagnosis_etc   PSQI_01_BedIn_HH_week   PSQI_01_BedIn_MM_week   PSQI_01_BedIn_HH_free   PSQI_01_BedIn_MM_free   PSQI_02_Latency_HH_week PSQI_02_Latency_MM_week PSQI_02_Latency_HH_free PSQI_02_Latency_MM_free PSQI_Latency_avg    PSQI_Latency    PSQI_03_BedOut_HH_week  PSQI_03_BedOut_MM_week  PSQI_03_BedOut_HH_free  PSQI_03_BedOut_MM_free  PSQI_TIB_week   PSQI_TIB_free   PSQI_TIB_avg    PSQI_MST_week   PSQI_MST_free   PSQI_MSFsc  PSQI_SJL    PSQI_TIB    PSQI_MST    PSQI_04_SD_HH_week  PSQI_04_SD_MM_week  PSQI_04_SD_HH_free  PSQI_04_SD_MM_free  PSQI_SD_week    PSQI_SD_free    PSQI_SD_avg PSQI_SD PSQI_HSE    PSQI_05_a   PSQI_05_b   PSQI_05_c   PSQI_05_d   PSQI_05_e   PSQI_05_f   PSQI_05_g   PSQI_05_h   PSQI_05_i   PSQI_05_j   PSQI_05_j_text  PSQI_06 PSQI_07 PSQI_08 PSQI_09 PSQI_10 PSQI_10_a   PSQI_10_b   PSQI_10_c   PSQI_10_d   PSQI_10_e   PSQI_10_e_text  PSQI_comp_02_raw    PSQI_comp_02    PSQI_comp_03    PSQI_comp_04    PSQI_comp_05    PSQI_comp_07    PSQI_Total  SQ_Satisfaction SQ_Wakefreq Nap_Freq    Nap_HH_week Nap_MM_week Nap_week_min    Nap_HH_free Nap_MM_free Nap_free_min    Nap_Satisfaction    N_Sleepattack   N_Cataplexy N_Hallucination FN_Paralysis    SSS ESS_01_book ESS_02_tv   ESS_03_sitting  ESS_04_transport    ESS_05_rest ESS_06_talk ESS_07_meal ESS_08_driving  ESS_Total   FSS_01  FSS_02  FSS_03  FSS_04  FSS_05  FSS_06  FSS_07  FSS_08  FSS_09  FSS_Total   BQ_01   BQ_02   BQ_03   BQ_04   BQ_05   BQ_06   BQ_07   BQ_08   BQ_09   BQ_10   BQ_Cat01    BQ_Cat02    BQ_Cat03    BQ_Risk STOP_Total  STOP_Risk   STOPBANG_Total  STOPBANG_Risk   ISI_01_a    ISI_01_b    ISI_01_c    ISI_02  ISI_03  ISI_04  ISI_05  ISI_Total   ISI_Category    RLS_01_urge RLS_02_rest RLS_03_move RLS_04_night    RLS_Category    RLS_05_frequency    RLS_06_family   RLS_07_observed IRLS_01 IRLS_02 IRLS_03 IRLS_04 IRLS_05 IRLS_06 IRLS_07 IRLS_08 IRLS_09 IRLS_10 IRLS_Total  IRLS_Category   RBD_FreqDream   RBD_Observed    RBDSQ_01    RBDSQ_02    RBDSQ_03    RBDSQ_04    RBDSQ_05    RBDSQ_06_01 RBDSQ_06_02 RBDSQ_06_03 RBDSQ_06_04 RBDSQ_07    RBDSQ_08    RBDSQ_09    RBDSQ_10    RBDSQ_10_Y  RBDSQ_Total PHQ_01  PHQ_02  PHQ_03  PHQ_04  PHQ_05  PHQ_06  PHQ_07  PHQ_08  PHQ_09  PHQ_Total   PHQ_Category    PSG_M_01_Hypnotics  PSG_M_02_SubSL_HH   PSG_M_02_SubSL_MM   PSG_M_02_SubSL_min  PSG_M_02_SubSL_Home PSG_M_03_SubSD_HH   PSG_M_03_SubSD_MM   PSG_M_03_SubSD_hr   PSG_M_03_SubSD_Home PSG_M_04_WakeNo PSG_M_05_Alertness  PSG_M_05_Complaint  PSG_M_06_SQ_a   PSG_M_06_SQ_b   PSG_M_06_SQ_c   PSG_M_06_SQ_d   PSG_M_06_SQ_e   PSG_M_07_Dream  PSG_M_07_Dream_text PSG_M_08_Wake   TST_min SL_min  REM_SL_min  Sleep_Eff   Arousal_no  Arousal_idx Arousal_resp_idx    Arousal_snoring_idx Arousal_PLM_idx Arousal_spont_idx   REM_pct N1_pct  N2_pct  N3_pct  WASO_pct    AI_obs  AI_obs_REM  AI_obs_NREM AI_cent AI_cent_REM AI_cent_NREM    AI_mix  AI_mix_REM  AI_mix_NREM HI  HI_REM  HI_NREM AHI_total   AHI_sup AHI_lat RDI_no  RDI_idx Lowest_SpO2 REM_sup_min REM_lat_min NREM_sup_min    NREM_lat_min    AHI_sup_REM AHI_lat_REM AHI_sup_N1  AHI_lat_N1  AHI_sup_N2  AHI_lat_N2  AHI_sup_N3  AHI_lat_N3  AHI_sup_NREM    AHI_lat_NREM    AHI_REM AHI_NREM    PLM_idx LM_idx  Arousal_PLM_no  Arousal_PLM_idx_re  Arousal_LM_no   Arousal_LM_idx  Pressure_05 Pr05_time_min   Pr05_position   Pr05_stage  Pr05_AHI    Pr05_snoring    Pr05_lowest_SpO2    Pr05_PLM_idx    Pr05_arousal_PLM_idx    Pr05_arousal_resp_idx   Pr05_arousal_spont_idx  Pressure_06 Pr06_time_min   Pr06_position   Pr06_stage  Pr06_AHI    Pr06_snoring    Pr06_lowest_SpO2    Pr06_PLM_idx    Pr06_arousal_PLM_idx    Pr06_arousal_resp_idx   Pr06_arousal_spont_idx  Pressure_07 Pr07_time_min   Pr07_position   Pr07_stage  Pr07_AHI    Pr07_snoring    Pr07_lowest_SpO2    Pr07_PLM_idx    Pr07_arousal_PLM_idx    Pr07_arousal_resp_idx   Pr07_arousal_spont_idx  Pressure_08 Pr08_time_min   Pr08_position   Pr08_stage  Pr08_AHI    Pr08_snoring    Pr08_lowest_SpO2    Pr08_PLM_idx    Pr08_arousal_PLM_idx    Pr08_arousal_resp_idx   Pr08_arousal_spont_idx  Pressure_09 Pr09_time_min   Pr09_position   Pr09_stage  Pr09_AHI    Pr09_snoring    Pr09_lowest_SpO2    Pr09_PLM_idx    Pr09_arousal_PLM_idx    Pr09_arousal_resp_idx   Pr09_arousal_spont_idx  Pressure_10 Pr10_time_min   Pr10_position   Pr10_stage  Pr10_AHI    Pr10_snoring    Pr10_lowest_SpO2    Pr10_PLM_idx    Pr10_arousal_PLM_idx    Pr10_arousal_resp_idx   Pr10_arousal_spont_idx  Pressure_11 Pr11_time_min   Pr11_position   Pr11_stage  Pr11_AHI    Pr11_snoring    Pr11_lowest_SpO2    Pr11_PLM_idx    Pr11_arousal_PLM_idx    Pr11_arousal_resp_idx   Pr11_arousal_spont_idx  Pressure_12 Pr12_time_min   Pr12_position   Pr12_stage  Pr12_AHI    Pr12_snoring    Pr12_lowest_SpO2    Pr12_PLM_idx    Pr12_arousal_PLM_idx    Pr12_arousal_resp_idx   Pr12_arousal_spont_idx  Pressure_13 Pr13_time_min   Pr13_position   Pr13_stage  Pr13_AHI    Pr13_snoring    Pr13_lowest_SpO2    Pr13_PLM_idx    Pr13_arousal_PLM_idx    Pr13_arousal_resp_idx   Pr13_arousal_spont_idx  Pressure_14 Pr14_time_min   Pr14_position   Pr14_stage  Pr14_AHI    Pr14_snoring    Pr14_lowest_SpO2    Pr14_PLM_idx    Pr14_arousal_PLM_idx    Pr14_arousal_resp_idx   Pr14_arousal_spont_idx  Pressure_15 Pr15_time_min   Pr15_position   Pr15_stage  Pr15_AHI    Pr15_snoring    Pr15_lowest_SpO2    Pr15_PLM_idx    Pr15_arousal_PLM_idx    Pr15_arousal_resp_idx   Pr15_arousal_spont_idx  Pressure_16 Pr16_time_min   Pr16_position   Pr16_stage  Pr16_AHI    Pr16_snoring    Pr16_lowest_SpO2    Pr16_PLM_idx    Pr16_arousal_PLM_idx    Pr16_arousal_resp_idx   Pr16_arousal_spont_idx  Pressure_17 Pr17_time_min   Pr17_position   Pr17_stage  Pr17_AHI    Pr17_snoring    Pr17_lowest_SpO2    Pr17_PLM_idx    Pr17_arousal_PLM_idx    Pr17_arousal_resp_idx   Pr17_arousal_spont_idx  Pressure_18 Pr18_time_min   Pr18_position   Pr18_stage  Pr18_AHI    Pr18_snoring    Pr18_lowest_SpO2    Pr18_PLM_idx    Pr18_arousal_PLM_idx    Pr18_arousal_resp_idx   Pr18_arousal_spont_idx  Pressure_19 Pr19_time_min   Pr19_position   Pr19_stage  Pr19_AHI    Pr19_snoring    Pr19_lowest_SpO2    Pr19_PLM_idx    Pr19_arousal_PLM_idx    Pr19_arousal_resp_idx   Pr19_arousal_spont_idx  Pressure_20 Pr20_time_min   Pr20_position   Pr20_stage  Pr20_AHI    Pr20_snoring    Pr20_lowest_SpO2    Pr20_PLM_idx    Pr20_arousal_PLM_idx    Pr20_arousal_resp_idx   Pr20_arousal_spont_idx  Pressure_21 Pr21_time_min   Pr21_position   Pr21_stage  Pr21_AHI    Pr21_snoring    Pr21_lowest_SpO2    Pr21_PLM_idx    Pr21_arousal_PLM_idx    Pr21_arousal_resp_idx   Pr21_arousal_spont_idx  Pressure_22 Pr22_time_min   Pr22_position   Pr22_stage  Pr22_AHI    Pr22_snoring    Pr22_lowest_SpO2    Pr22_PLM_idx    Pr22_arousal_PLM_idx    Pr22_arousal_resp_idx   Pr22_arousal_spont_idx  Pressure_23 Pr23_time_min   Pr23_position   Pr23_stage  Pr23_AHI    Pr23_snoring    Pr23_lowest_SpO2    Pr23_PLM_idx    Pr23_arousal_PLM_idx    Pr23_arousal_resp_idx   Pr23_arousal_spont_idx  Pressure_24 Pr24_time_min   Pr24_position   Pr24_stage  Pr24_AHI    Pr24_snoring    Pr24_lowest_SpO2    Pr24_PLM_idx    Pr24_arousal_PLM_idx    Pr24_arousal_resp_idx   Pr24_arousal_spont_idx  Pressure_25 Pr25_time_min   Pr25_position   Pr25_stage  Pr25_AHI    Pr25_snoring    Pr25_lowest_SpO2    Pr25_PLM_idx    Pr25_arousal_PLM_idx    Pr25_arousal_resp_idx   Pr25_arousal_spont_idx  Pressure_26 Pr26_time_min   Pr26_position   Pr26_stage  Pr26_AHI    Pr26_snoring    Pr26_lowest_SpO2    Pr26_PLM_idx    Pr26_arousal_PLM_idx    Pr26_arousal_resp_idx   Pr26_arousal_spont_idx  Pressure_27 Pr27_time_min   Pr27_position   Pr27_stage  Pr27_AHI    Pr27_snoring    Pr27_lowest_SpO2    Pr27_PLM_idx    Pr27_arousal_PLM_idx    Pr27_arousal_resp_idx   Pr27_arousal_spont_idx  Pressure_28 Pr28_time_min   Pr28_position   Pr28_stage  Pr28_AHI    Pr28_snoring    Pr28_lowest_SpO2    Pr28_PLM_idx    Pr28_arousal_PLM_idx    Pr28_arousal_resp_idx   Pr28_arousal_spont_idx  Pressure_29 Pr29_time_min   Pr29_position   Pr29_stage  Pr29_AHI    Pr29_snoring    Pr29_lowest_SpO2    Pr29_PLM_idx    Pr29_arousal_PLM_idx    Pr29_arousal_resp_idx   Pr29_arousal_spont_idx  pre_SSS MSLT_01_start   MSLT_01_end MSLT_01_Sleep_onset MSLT_01_REM_onset   MSLT_01_SL_min  MSLT_01_REM_SL_min  MSLT_01_N1_idx  MSLT_01_N2_idx  MSLT_01_N3_idx  MSLT_01_REM_idx MSLT_01_SSS MSLT_01_dream_idx   MSLT_02_start   MSLT_02_end MSLT_02_Sleep_onset MSLT_02_REM_onset   MSLT_02_SL_min  MSLT_02_REM_SL_min  MSLT_02_N1_idx  MSLT_02_N2_idx  MSLT_02_N3_idx  MSLT_02_REM_idx MSLT_02_SSS MSLT_02_dream_idx   MSLT_03_start   MSLT_03_end MSLT_03_Sleep_onset MSLT_03_REM_onset   MSLT_03_SL_min  MSLT_03_REM_SL_min  MSLT_03_N1_idx  MSLT_03_N2_idx  MSLT_03_N3_idx  MSLT_03_REM_idx MSLT_03_SSS MSLT_03_dream_idx   MSLT_04_start   MSLT_04_end MSLT_04_Sleep_onset MSLT_04_REM_onset   MSLT_04_SL_min  MSLT_04_REM_SL_min  MSLT_04_N1_idx  MSLT_04_N2_idx  MSLT_04_N3_idx  MSLT_04_REM_idx MSLT_04_SSS MSLT_04_dream_idx   MSLT_05_start   MSLT_05_end MSLT_05_Sleep_onset MSLT_05_REM_onset   MSLT_05_SL_min  MSLT_05_REM_SL_min  MSLT_05_N1_idx  MSLT_05_N2_idx  MSLT_05_N3_idx  MSLT_05_REM_idx MSLT_05_SSS MSLT_05_dream_idx   MSLT_mean_SL"""

columns = columns_str.split()
# Note: We do NOT add PSG_Type2 here manually. We will insert it programmatically.
master_df = pd.DataFrame(columns=columns)

col_mapping = {
    '등록번호': 'Hospital_ID',
    '환자명': 'Name',
    '검사명': 'PSG_Type',
    '검사일시': 'PSG_Date',
    '나이': 'AGE',       # Added
    '성별': 'SEX'        # Added
}

temp_dfs = []
for i in range(1, NUM_FILES + 1):
    filename = f"Sheet{i}.csv"
    if os.path.exists(filename):
        print(f"Reading {filename}...")
        sheet_df = None
        for encoding in ['utf-8', 'cp949', 'euc-kr']:
            try:
                sheet_df = pd.read_csv(filename, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if sheet_df is not None:
            sheet_df.columns = sheet_df.columns.str.strip()
            
            # Clean Name
            if '환자명' in sheet_df.columns:
                sheet_df['환자명'] = sheet_df['환자명'].astype(str).str.replace('♣', '', regex=False)
                sheet_df['환자명'] = sheet_df['환자명'].replace('nan', '')

            temp_df = pd.DataFrame(columns=columns, index=range(len(sheet_df)))
            
            for csv_col, master_col in col_mapping.items():
                if csv_col in sheet_df.columns:
                    temp_df[master_col] = sheet_df[csv_col]
            
            # --- TRANSFORMATIONS ---
            if 'SEX' in temp_df.columns:
                # Map M->0, F->1
                temp_df['SEX'] = temp_df['SEX'].map({'M': 0, 'F': 1})
            
            temp_dfs.append(temp_df)
    else:
        print(f"File {filename} not found.")

if temp_dfs:
    master_df = pd.concat([master_df] + temp_dfs, ignore_index=True)
    
    # 1. Parsing the dates
    master_df['Temp_Date'] = pd.to_datetime(
        master_df['PSG_Date'], format='%m-%d %H:%M', errors='coerce'
    ).fillna(
        pd.to_datetime(master_df['PSG_Date'], errors='coerce')
    )
    
    # Sort
    master_df = master_df.sort_values(by='Temp_Date', ascending=True)

    print("Classifying PSG Types...")
    
    # Pre-calculate MSLT events
    mslt_events = []
    for idx, row in master_df.iterrows():
        p_type = str(row['PSG_Type']).lower()
        if 'mslt' in p_type and pd.notna(row['Temp_Date']):
            mslt_events.append((row['Hospital_ID'], row['Temp_Date']))
    
    def get_new_psg_type(row):
        original_type = str(row['PSG_Type']).lower()
        
        if 'mslt' in original_type:
             return 'N'

        current_id = row['Hospital_ID']
        current_date = row['Temp_Date']
        
        has_poly = 'polysomnography' in original_type
        has_cpap = 'cpap' in original_type
        has_split = 'split night' in original_type

        if has_poly:
            if has_cpap:
                return 'C'
            elif has_split:
                return 'SP'
            else:
                found_mslt = False
                if pd.notna(current_date):
                    for mslt_id, mslt_date in mslt_events:
                        if mslt_id == current_id:
                            diff = abs((current_date - mslt_date).days)
                            if diff <= 7:
                                found_mslt = True
                                break
                if found_mslt:
                    return 'M'
                else:
                    return 'P'
        else:
            return 'N'

    # Calculate the new types
    new_types = master_df.apply(get_new_psg_type, axis=1)

    # Insert 'PSG_Type2' immediately to the right of 'PSG_Type'
    if 'PSG_Type' in master_df.columns:
        loc_index = master_df.columns.get_loc('PSG_Type') + 1
        master_df.insert(loc_index, 'PSG_Type2', new_types)
    else:
        master_df['PSG_Type2'] = new_types

    # -----------------------------------------------
    # FINAL LOGIC: Filter, Rename, and Assignment
    # -----------------------------------------------
    
    # 1. Remove rows where PSG_Type2 is 'N'
    print(f"Rows before filtering: {len(master_df)}")
    master_df = master_df[master_df['PSG_Type2'] != 'N'].copy()
    print(f"Rows after removing 'N': {len(master_df)}")

    # 2. Swap columns: Drop old 'PSG_Type', rename 'PSG_Type2' -> 'PSG_Type'
    master_df = master_df.drop(columns=['PSG_Type'])
    master_df = master_df.rename(columns={'PSG_Type2': 'PSG_Type'})

    # 3. Assign Database_ID, PSG_Date, and PSG_No
    # ID Format: 001_{YEAR}{MM}{DD}_{number}_{PSG_Type}
    # Date Format: {YEAR}{MM}{DD}
    # PSG_No Format: P{YEAR}-{number}
    
    current_seq = START_SEQ_NUM
    
    # Iterate through the DataFrame using index to safely update
    for idx, row in master_df.iterrows():
        # A. Calculate MM/DD part
        if pd.notna(row['Temp_Date']):
            mmdd = row['Temp_Date'].strftime('%m%d')
        else:
            mmdd = "0000"
        
        p_type = row['PSG_Type']
        
        # B. Generate Database_ID
        db_id = f"001_{TARGET_YEAR}{mmdd}_{current_seq:03d}_{p_type}"
        master_df.at[idx, 'Database_ID'] = db_id
        
        # C. Update PSG_Date to YYYYMMDD
        clean_date = f"{TARGET_YEAR}{mmdd}"
        master_df.at[idx, 'PSG_Date'] = clean_date
        
        # D. Assign PSG_No (P{YEAR}-{number})
        psg_no_val = f"P{TARGET_YEAR}-{current_seq:03d}"
        master_df.at[idx, 'PSG_No'] = psg_no_val
        
        current_seq += 1

    # Cleanup
    del master_df['Temp_Date']

# Save
output_filename = 'filled_sleep_study.csv'
master_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Data saved to '{output_filename}'")