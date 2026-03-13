import pandas as pd
import re
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- 1. Schema Definitions ---
VARS_ANTHRO = ['Height_cm', 'Weight_kg', 'BMI', 'Neckcir_cm', 'Occupation', 'Shiftwork']
VARS_PHX = [
    'PHx_CVA', 'PHx_Parkinson', 'PHx_PNS', 'PHx_Epilepsy', 'PHx_Dementia', 'PHx_Alcoholism', 
    'PHx_Cancer', 'PHx_Renal', 'PHx_Pulmonary', 'PHx_HTN', 'PHx_Thyroid', 'PHx_Liver', 
    'PHx_DM', 'PHx_Cardiovascular', 'PHx_NasalFx', 'PHx_Sinusitis', 'PHx_GERD_ulcer', 'PHx_Psy'
]
VARS_HABITS = ['Habit_Caffein', 'Habit_Alcohol', 'Habit_Smoking', 'Habit_Workout']
VARS_PSQI = [
    'PSQI_01_BedIn_HH_week', 'PSQI_01_BedIn_MM_week', 'PSQI_01_BedIn_HH_free', 'PSQI_01_BedIn_MM_free', 
    'PSQI_02_Latency_HH_week', 'PSQI_02_Latency_MM_week', 'PSQI_02_Latency_HH_free', 'PSQI_02_Latency_MM_free', 
    'PSQI_03_BedOut_HH_week', 'PSQI_03_BedOut_MM_week', 'PSQI_03_BedOut_HH_free', 'PSQI_03_BedOut_MM_free', 
    'PSQI_04_SD_HH_week', 'PSQI_04_SD_MM_week', 'PSQI_04_SD_HH_free', 'PSQI_04_SD_MM_free'
]
VARS_SCALES = [
    'SQ_Satisfaction', 'SQ_Wakefreq', 'Nap_Freq', 'Nap_HH_week', 'Nap_MM_week', 'Nap_HH_free', 'Nap_MM_free', 
    'Nap_Satisfaction', 'N_Sleepattack', 'N_Cataplexy', 'N_Hallucination', 'N_Paralysis', 'SSS', 
    'ESS_01_book', 'ESS_02_tv', 'ESS_03_sitting', 'ESS_04_transport', 'ESS_05_rest', 'ESS_06_talk', 'ESS_07_meal', 'ESS_08_driving', 
    'FSS_01', 'FSS_02', 'FSS_03', 'FSS_04', 'FSS_05', 'FSS_06', 'FSS_07', 'FSS_08', 'FSS_09', 
    'BQ_01', 'BQ_02', 'BQ_03', 'BQ_04', 'BQ_05', 'BQ_06', 'BQ_07', 'BQ_08', 'BQ_09', 'BQ_10', 
    'RLS_01_urge', 'RLS_02_rest', 'RLS_03_move', 'RLS_04_night', 'RLS_05_frequency', 'RLS_06_family', 'RLS_07_observed', 
    'IRLS_01', 'IRLS_02', 'IRLS_03', 'IRLS_04', 'IRLS_05', 'IRLS_06', 'IRLS_07', 'IRLS_08', 'IRLS_09', 'IRLS_10', 
    'RBD_FreqDream', 'RBD_Observed', 
    'RBDSQ_01', 'RBDSQ_02', 'RBDSQ_03', 'RBDSQ_04', 'RBDSQ_05', 'RBDSQ_06_01', 'RBDSQ_06_02', 'RBDSQ_06_03', 'RBDSQ_06_04', 
    'RBDSQ_07', 'RBDSQ_08', 'RBDSQ_09', 'RBDSQ_10', 'RBDSQ_10_Y',
    'PHQ_01', 'PHQ_02', 'PHQ_03', 'PHQ_04', 'PHQ_05', 'PHQ_06', 'PHQ_07', 'PHQ_08', 'PHQ_09'
]
ALL_KEYS = VARS_ANTHRO + VARS_PHX + VARS_HABITS + VARS_PSQI + VARS_SCALES

# --- 2. Logic Helpers ---
def parse_range(val_str, mode='mid'):
    """Calculates midpoint or max, returns float."""
    if not isinstance(val_str, str): return val_str
    # Handles "10-12" or "10~12"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", val_str)
    if m:
        v1, v2 = float(m.group(1)), float(m.group(2))
        return (v1 + v2) / 2 if mode == 'mid' else max(v1, v2)
    # Handles "10"
    m = re.search(r"(\d+(?:\.\d+)?)", val_str)
    return float(m.group(1)) if m else None

def parse_time(ampm, hh_str, mm_str, context='duration'):
    """Standardizes either a clock time (bedin/bedout) or a duration into (HH, MM).

    - Supports ranges (e.g., '10-12', '10~12') and decimals (e.g., '1.5').
    - For bedin/bedout: applies AM/PM conversion to 24-hour time.
    - For duration: no AM/PM conversion is applied.
    """
    hh = parse_range(hh_str, 'mid') if hh_str else 0.0
    mm = parse_range(mm_str, 'mid') if mm_str else 0.0

    # If provided strings were non-empty but unparsable
    if hh_str and hh is None:
        return None, None
    if mm_str and mm is None:
        mm = 0.0

    # Only apply AM/PM logic if this is a clock time (bedin/bedout)
    if context in ['bedin', 'bedout']:
        # Infer AM/PM if missing
        if not ampm or str(ampm).strip() == "":
            if context == 'bedin':
                # bedtime: 6~12 usually PM, but in this dataset "12시" is overwhelmingly midnight
                if hh is not None and 12 <= hh < 13:
                    ampm = '오전'
                else:
                    ampm = '오후' if (hh is not None and 6 <= hh <= 12) else '오전'
            elif context == 'bedout':
                ampm = '오전' if (hh is not None and 1 <= hh <= 11) else '오후'

        # Special-case: for bedtime, "12시" should be treated as midnight (0시) even when AM/PM is noisy
        if context == 'bedin' and hh is not None and 12 <= hh < 13:
            hh = hh - 12  # 12.x -> 0.x
            ampm = '오전'  # prevent PM conversion

        # Convert to 24-hour clock
        if ampm == '오전' and hh == 12:
            hh = 0
        elif ampm == '오후' and hh is not None and hh < 12:
            hh += 12

    if hh is None:
        return None, None

    # Overflow fractional hours into minutes (e.g., 1.5 hours -> 1h 30m)
    total_mm = mm + (hh % 1) * 60
    final_hh = int(hh) + int(total_mm // 60)
    final_mm = int(round(total_mm % 60))

    # Normalize if rounding bumped to 60
    if final_mm == 60:
        final_hh += 1
        final_mm = 0

    return final_hh, final_mm

# --- 3. Core Parsing Engine ---
def parse_nq_text(text):
    if not isinstance(text, str) or not text or text == "매치없음": return {}
    text = text.replace('\xa0', ' ').strip()
    data = {}

    # --- 1. Anthropometrics ---
    for k, p in [('Height_cm', r"신장\s*([\d.~-]+)"), ('Weight_kg', r"체중\s*([\d.~-]+)"), 
                 ('BMI', r"BMI\s*([\d.~-]+)"), ('Neckcir_cm', r"목둘레\s*([\d.~-]+)")]:
        m = re.search(p, text)
        if m: 
            val = parse_range(m.group(1))
            if val is not None: data[k] = int(val+0.5) # Rounding per requirements

    # --- 2. Occupation & Habits ---
    m_occ = re.search(r"직업\s+(.+)", text)
    if m_occ: data['Occupation'] = m_occ.group(1).split('\n')[0].strip()
    
        # Habits: if the corresponding question section is absent, leave as NaN (do not force 0)
    habit_q_map = {
        'Habit_Caffein': (r"카페인\s*섭취.*?질문", r"카페인.*?마신다"),
        'Habit_Alcohol': (r"음주\s*에\s*대한\s*질문", r"음주.*?마신다"),
        'Habit_Workout': (r"운동\s*에\s*대한\s*질문", r"운동.*?운동한다"),
    }
    for k, (q_pat, yes_pat) in habit_q_map.items():
        if re.search(q_pat, text, re.S):
            data[k] = 1 if re.search(yes_pat, text, re.S) else 0
        else:
            data[k] = None

    # Smoking: keep original 0/1/2 mapping, but only if the smoking-question section exists
    if re.search(r"흡연\s*에\s*대한\s*질문", text, re.S):
        if re.search(r"흡연.*?금연했다면", text, re.S):
            data['Habit_Smoking'] = 1
        elif re.search(r"흡연.*?피운다", text, re.S):
            data['Habit_Smoking'] = 2
        else:
            data['Habit_Smoking'] = 0
    else:
        data['Habit_Smoking'] = None

    # Shiftwork: set only if the question exists, otherwise leave NaN
    m_shift = re.search(r"교대\s*근무를\s*합니까\?\s*(예|아니오)", text)
    if m_shift:
        data['Shiftwork'] = 1 if m_shift.group(1) == '예' else 0
    else:
        data['Shiftwork'] = None
# --- 3. Medical History (Updated Map) ---
    phx_map = {
       'PHx_CVA': r'뇌졸중', 'PHx_Parkinson': r'파킨슨씨\s*병', 'PHx_PNS': r'말초\s*신경질환',
       'PHx_Epilepsy': r'경련성\s*질환', 'PHx_Dementia': r'치매|노망', 'PHx_Alcoholism': r'알코올\s*중독증',
       'PHx_Cancer': r'암', 'PHx_Renal': r'신부전\s*\(신장질환\)', 'PHx_Pulmonary': r'만성\s*폐질환',
       'PHx_HTN': r'고혈압', 'PHx_Thyroid': r'갑상선질환', 'PHx_Liver': r'간염', 'PHx_DM': r'당뇨',
       'PHx_Cardiovascular': r'심장질환', 'PHx_NasalFx': r'코뼈가\s*부러진\s*적',
       'PHx_Sinusitis': r'축농증|알러지성\s*비염', 'PHx_GERD_ulcer': r'위궤양|위식도\s*역류', 
       'PHx_Psy': r'정신과적 질환|우울증|불안증|공황장애'
    }
    m_phx = re.search(r"1\.\s*현재 다음과 같은 질환.*?\n(.*?)\n2\.", text, re.S)
    phx_text = m_phx.group(1) if m_phx else ""
    for k, p in phx_map.items():
        data[k] = 1 if re.search(p, phx_text) else 0

        # --- 4. PSQI (User Specified Logic) ---
    psqi_sec_match = re.search(r"수면의\s*질\s*지수\s*\(PSQI\)(.*?)(?:주간\s*졸림도|$)", text, re.S)
    if psqi_sec_match:
        psqi_sec = psqi_sec_match.group(1)

        # Helper: isolate each question block to avoid cross-matching (fixes PSQI_02 leaking into Q3/Q4)
        def _psqi_qblock(q_num, next_q_num):
            m = re.search(rf"{q_num}\.\s*.*?(?=\s*{next_q_num}\.|$)", psqi_sec, re.S)
            return m.group(0) if m else ""

        q1_block = _psqi_qblock(1, 2)
        q2_block = _psqi_qblock(2, 3)
        q3_block = _psqi_qblock(3, 4)
        q4_block = _psqi_qblock(4, 5)

        # Pattern for clock times (AM/PM HH:MM)
        time_p = r"(?:(오전|오후)\s*)?(\d+(?:[-~]\d+)?)\s*시(?:\s*(\d+(?:[-~]\d+)?)\s*분)?"
        # Pattern for durations (HH and/or MM)
        dur_p = r"(?:(\d+(?:\.\d+)?(?:[-~]\d+(?:\.\d+)?)?)\s*(?:시간|시))?(?:\s*(\d+(?:\.\d+)?(?:[-~]\d+(?:\.\d+)?)?)\s*분)?"

        for sfx, lbl in [('week', '주중'), ('free', '주말')]:
            # Q1 Bed-In (clock time)
            m1 = re.search(rf"{lbl}\s*[:\s]\s*" + time_p, q1_block, re.S)
            if m1:
                h, m_ = parse_time(m1.group(1), m1.group(2), m1.group(3), 'bedin')
                data[f'PSQI_01_BedIn_HH_{sfx}'], data[f'PSQI_01_BedIn_MM_{sfx}'] = h, m_

            # Q2 Latency (duration) - FIX: parse both hours and minutes, and do not leak into Q3/Q4
            m2 = re.search(rf"{lbl}\s*[:\s]\s*" + dur_p, q2_block, re.S)
            if m2 and (m2.group(1) or m2.group(2)):
                h, m_ = parse_time(None, m2.group(1), m2.group(2), 'duration')
                data[f'PSQI_02_Latency_HH_{sfx}'], data[f'PSQI_02_Latency_MM_{sfx}'] = h, m_

            # Q3 Bed-Out (clock time)
            m3 = re.search(rf"{lbl}\s*[:\s]\s*" + time_p, q3_block, re.S)
            if m3:
                h, m_ = parse_time(m3.group(1), m3.group(2), m3.group(3), 'bedout')
                data[f'PSQI_03_BedOut_HH_{sfx}'], data[f'PSQI_03_BedOut_MM_{sfx}'] = h, m_

            # Q4 Sleep Duration (duration)
            m4 = re.search(rf"{lbl}\s*[:\s]\s*" + dur_p, q4_block, re.S)
            if m4 and (m4.group(1) or m4.group(2)):
                h, m_ = parse_time(None, m4.group(1), m4.group(2), 'duration')
                data[f'PSQI_04_SD_HH_{sfx}'], data[f'PSQI_04_SD_MM_{sfx}'] = h, m_
# --- 5. SQ, Nap, Narcolepsy ---
    # SQ
    m_sq1 = re.search(r"밤에 수면시간이 충분하다고 느끼십니까\?\s*(예|아니오)", text)
    if m_sq1: data['SQ_Satisfaction'] = 1 if m_sq1.group(1) == '예' else 0

# 1. Wake Frequency (Handles "2-3 번")
    m_sq2 = re.search(r"밤에\s*몇\s*번\s*깨십니까\?.*?([\d\-\s~]+)번", text)
    if m_sq2:
        val = parse_range(m_sq2.group(1), 'max')
        if val is not None:
            data['SQ_Wakefreq'] = int(val)
    
    # 2. Nap Frequency (Handles "1 번")
    m_nf = re.search(r"낮잠을\s*잡니까\?.*?([\d\-\s~]+)번", text)
    if m_nf:
        val = parse_range(m_nf.group(1), 'max')
        if val is not None:
            data['Nap_Freq'] = int(val)
    
        # 3. Nap Times (weekday/weekend) - FIX:
    #   - Parse ranges/decimals (e.g., 1.5시간 -> 1h 30m)
    #   - Do NOT copy weekday values into weekend (or vice versa) when one side is missing
    #   - If the nap-time line is missing, leave as NaN (do not force 0)
    nap_line_match = re.search(r"^.*평균\s*낮잠\s*시간.*$", text, re.MULTILINE)
    if nap_line_match:
        nap_line = nap_line_match.group(0).replace('\xa0', ' ').strip()

        dur_p = r"(?:(\d+(?:\.\d+)?(?:[-~]\d+(?:\.\d+)?)?)\s*(?:시간|시))?(?:\s*(\d+(?:\.\d+)?(?:[-~]\d+(?:\.\d+)?)?)\s*분)?"

        for sfx, lbl in [('week', '평일'), ('free', '주말')]:
            m = re.search(rf"{lbl}\s*[:\s]\s*" + dur_p, nap_line)
            if m and (m.group(1) or m.group(2)):
                h, m_ = parse_time(None, m.group(1), m.group(2), 'duration')
                data[f'Nap_HH_{sfx}'] = h
                data[f'Nap_MM_{sfx}'] = m_
# 2. Extract Sleep Frequency (From your sample: "밤에 몇 번 깨십니까?")
    m_wake = re.search(r"밤에\s*몇\s*번\s*깨십니까\?.*?(\d+)\s*번", text)
    if m_wake:
        data['SQ_Wakefreq'] = int(m_wake.group(1))

    # # 3. Extract Insufficient Sleep Days ("한 달에 몇 일입ka?")
    # m_insuff = re.search(r"충분히\s*수면을\s*취하지\s*못하는.*?(\d+)\s*일", text)
    # if m_insuff:
    #     data['SQ_InsuffDays'] = int(m_insuff.group(1))

    m_ns = re.search(r"낮잠을 자고 나면 상쾌합니까\?\s*(예|아니오)", text)
    if m_ns: data['Nap_Satisfaction'] = 1 if m_ns.group(1) == '예' else 0

    # Narcolepsy (N_)
    n_map = {
        'N_Sleepattack': r"나도\s*모르게\s*잠에\s*빠져\s*든\s*적",
        'N_Cataplexy': r"몸에\s*실제로\s*힘이\s*빠져.*?넘어지거나",
        'N_Hallucination': r"환각이나\s*꿈\s*같은\s*이미지",
        'N_Paralysis': r"가위\s*눌림"
    }
    for k, p in n_map.items():
        # Match the pattern, then any characters (non-greedy) until "예" or "아니오"
        # re.DOTALL (re.S) allows the dot to match newlines.
        m = re.search(rf"{p}.*?(예|아니오)", text, re.S)
        
        if m:
            # Group(1) will capture "예" or "아니오"
            data[k] = 1 if "예" in m.group(1) else 0
        else:
            # Optional: handle missing data
            data[k] = None

    # --- 6. Scales (ESS, FSS, BQ, RLS, RBD, PHQ) ---    
    # SSS
    m_sss = re.search(r"Stanford\s*Sleepiness\s*Scale.*?(\d+)\.", text, re.S)
    if m_sss: data['SSS'] = int(m_sss.group(1))

    # ESS
    ess_start = text.find("The Epworth Sleepiness Scale")
    if ess_start != -1:
        ess_text = text[ess_start:]
        qs = ['book', 'tv', 'sitting', 'transport', 'rest', 'talk', 'meal', 'driving']
        keywords = ["앉아서 책", "텔레비전", "회의석상", "버스나", "오후 휴식", "누군가에게", "점심식사", "차를 운전"]
        for i, kw in enumerate(keywords):
            m = re.search(rf"(\d+)\s*\*\s*.*?{re.escape(kw)}", ess_text)
            if m:
                data[f'ESS_0{i+1}_{qs[i]}'] = int(m.group(1))

    # FSS (Fatigue Severity Scale)
    fss_start = text.find("피로 정도에 대한 설문")
    if fss_start != -1:
        fss_text = text[fss_start:]
        keywords = [
            "내가 피로해지면 나는 의욕이 낮아진다.",
            "운동을 하면 피로해진다.",
            "나는 쉽게 피곤해진다.",
            "피로로 인해 신체적 활동이 방해를 받는다.",
            "피로로 인해 빈번하게 문제가 발생한다.",
            "피로하면 신체적 활동을 지속할 수 없다.",
            "피로하면 의무와 책임을 다하지 못하게 된다.",
            "피로는 나에게 가장 지장을 초래하는 세 가지",
            "피로로 인해 나의 일, 가족, 혹은 사회생활이"]
        for i, kw in enumerate(keywords):
            m = re.search(rf"{re.escape(kw)}.*?\s+(\d+)", fss_text)
            if m:
                data[f'FSS_0{i+1}'] = int(m.group(1))

    # BQ (Berlin) - Corrected Map
    bq_sec = re.search(r"Berline\s*Questionnaire(.*?)(?:불면증|하지불안|$)", text, re.S)
    if bq_sec:
        bq_text = bq_sec.group(1)
        for i in range(1, 11):
            m = re.search(rf"{i}\..*?\n\s*(.*?)\s*(?:\n|$)", bq_text)
            if m:
                ans = m.group(1).strip()
                val = None
                
                # Question Specific logic
                if i == 1: # Weight: Inc(1), Dec(2), No(3)
                    if '증가' in ans: val = 1
                    elif '감소' in ans: val = 2
                    elif '변하지' in ans: val = 3
                elif i == 2: # Snore: Yes(1), No(2), DK(3)
                    if '예' in ans: val = 1
                    elif '아니오' in ans: val = 2
                    elif '모르' in ans: val = 3
                elif i == 3: # Loudness
                    if '매우' in ans: val = 4
                    elif '말하는 것보다' in ans: val = 3
                    elif '말하는 정도' in ans: val = 2
                    elif '숨쉬는' in ans: val = 1
                elif i in [4, 6, 7, 8]: # Freq
                    if '매일' in ans: val = 1 # almost daily
                    elif '3-4' in ans: val = 2
                    elif '1-2' in ans and '주' in ans: val = 3
                    elif '1-2' in ans and '달' in ans: val = 4
                    elif '전혀' in ans: val = 5
                elif i in [5, 9, 10]: # Yes/No
                    if '예' in ans: val = 1
                    elif '아니오' in ans: val = 2                
                if val is not None: data[f'BQ_{str(i).zfill(2)}'] = val

    # 1. Isolate the RLS section to prevent matching keywords elsewhere
    rls_match = re.search(r"하지불안증후군 / 주기성사지운동증후군(.*?)(?=하지불안증후군에 대한 설문|$)", text, re.S)
    if rls_match:
        rls_sec = rls_match.group(1)

        # Define the exact question anchors
        # We use \s+ to handle varying spaces/newlines within the text itself
        rls_patterns = {
            'RLS_01_urge': r"다리에\s*불편하거나\s*좋지\s*않은\s*감각으로\s*인해\s*다리를\s*자꾸\s*움직이고\s*싶습니까\?",
            'RLS_02_rest': r"다리를\s*움직이고\s*싶은\s*충동이나\s*좋지\s*않은\s*감각이\s*쉬거나.*?악화되나요\?",
            'RLS_03_move': r"다리를\s*움직이고\s*싶은\s*충동이나\s*좋지\s*않은\s*감각은\s*걷거나\s*움직임에\s*따라\s*완화되나요\?",
            'RLS_04_night': r"다리를\s*움직이고\s*싶은\s*충동이나\s*좋지\s*않은\s*감각이\s*낮보다\s*저녁이나\s*밤에\s*더\s*나빠지거나.*?발생하나요\?",
            'RLS_05_frequency': r"얼마나\s*자주\s*발생하나요\?",
            'RLS_06_family': r"가족\s*중에\s*이런\s*증세를\s*가진\s*분이\s*계신가요\?",
            'RLS_07_observed': r"수면\s*중\s*발목을\s*움직인다거나\s*다리를\s*주기적으로\s*움직인다는\s*이야기"
        }

        for key, question_regex in rls_patterns.items():
            # Logic: Find the question text, then skip any characters (.*?) 
            # until we hit a digit followed by a period (\d\.)
            # The re.S flag allows this search to cross newlines.
            m = re.search(rf"{question_regex}.*?(\d)\.", rls_sec, re.S)
            
            if m:
                data[key] = int(m.group(1))
            else:
                data[key] = None

    # 2. Extract IRLS items using exact phrases
    irls_match = re.search(r"하지불안증후군에 대한 설문 \(Restless Legs Syndrome Rating Scale\)(.*)", text, re.S)
    if irls_match:
        irls_sec = irls_match.group(1)
        
        irls_questions = [
            ('IRLS_01', r"불편이\s*어느\s*정도인가요\?"),
            ('IRLS_02', r"움직이고\(돌아다니고\)\s*싶은\s*정도는\?"), # Handle potential typo in '정도'
            ('IRLS_03', r"움직이게\(돌아다니게\)\s*되면.*?나아지나요\?"),
            ('IRLS_04', r"수면장애가\s*얼마나\s*심하나요\?"),
            ('IRLS_05', r"주간\s*피로감이나\s*졸림이\s*어느\s*정도인가요\?"),
            ('IRLS_06', r"전체적으로\s*볼\s*때.*?심한\s*정도는\?"),
            ('IRLS_07', r"얼마나\s*자주.*?발생하나요\?"),
            ('IRLS_08', r"평균\s*하루\s*어느\s*정도의\s*증상이\s*있나요\?"),
            ('IRLS_09', r"일상생활에\s*얼마나\s*지장을\s*주나요\?"),
            ('IRLS_10', r"기분장애.*?얼마나\s*심한가요\?")
        ]

        for key, q_regex in irls_questions:
            m = re.search(rf"{q_regex}.*?(\d)\.", irls_sec, re.S)
            if m:
                data[key] = int(m.group(1))

    # 1. Answer Mapping (Strict)
    def map_ans(val):
        if not val: return None
        v = val.strip()
        if v == "예": return 1
        if v == "아니오": return 0
        return 1 if "예" in v else 0

    rbd_match = re.search(r"수면중\s*이상행동(.*?)(?=수면의\s*질|주간\s*졸림도|$)", text, re.S)
    if rbd_match:
        rbd_text = rbd_match.group(1)

        # 1. Intro Questions (FreqDream, Observed)
        for key, phrase in [('RBD_FreqDream', '꿈을\s*많이\s*꾸십니까'), ('RBD_Observed', '고약한\s*잠버릇')]:
            m = re.search(rf"{phrase}.*?([예|아니오]+)\s*$", rbd_text, re.MULTILINE | re.S)
            if m: data[key] = map_ans(m.group(1))

        # 2. RBDSQ Standard Items (1-9)
        items = [
            (1, '01', "생생한"), (2, '02', "공격적인"), (3, '03', "일치한다"),
            (4, '04', "알고\s*있다"), (5, '05', "다칠\s*뻔하거나"), (7, '07', "스스로\s*깨기도"),
            (8, '08', "잘\s*기억한다"), (9, '09', "잘\s*깬다")
        ]
        for num, suffix, phrase in items:
            pattern = rf"^{num}\s+.*?{phrase}.*?\s+(예|아니오)\s*$"
            m = re.search(pattern, rbd_text, re.MULTILINE | re.S)
            if m: data[f'RBDSQ_{suffix}'] = map_ans(m.group(1))

        # 3. RBDSQ Sub-items (6.1-6.4)
        for sub in range(1, 5):
            pattern = rf"^6\.{sub}\s+.*?\s+(예|아니오)\s*$"
            m = re.search(pattern, rbd_text, re.MULTILINE | re.S)
            if m: data[f'RBDSQ_06_0{sub}'] = map_ans(m.group(1))

                # 4. RBDSQ Item 10 & Conditional Parsing for RBDSQ_10_Y
        # RBDSQ_10_Y is a comma-separated list of codes (1-8) for checked conditions:
        # 1 뇌졸중, 2 두부 외상, 3 파키슨증, 4 하지불안증후군, 5 기면발작, 6 우울증, 7 뇌전증, 8 중추신경계 염증
        m10 = re.search(r"^10\s+.*?신경계.*?\s+(예|아니오)\s*$", rbd_text, re.MULTILINE | re.S)
        if m10:
            ans_val = map_ans(m10.group(1))
            data['RBDSQ_10'] = ans_val

            if ans_val == 1:
                post_10_text = rbd_text[m10.end():]
                stop_marker = re.search(r"총점", post_10_text)
                conditions_block = post_10_text[:stop_marker.start()] if stop_marker else post_10_text

                cleaned = " ".join(conditions_block.split())

                cond_map = [
                    (1, r"뇌\s*졸중"),
                    (2, r"두부\s*외상"),
                    (3, r"파키?\s*슨|파킨\s*슨"),
                    (4, r"하지\s*불안\s*증후군"),
                    (5, r"기면\s*발작"),
                    (6, r"우울\s*증"),
                    (7, r"뇌전\s*증"),
                    (8, r"중추\s*신경계\s*염증")
                ]

                codes = []
                for code, pat in cond_map:
                    if re.search(pat, cleaned):
                        codes.append(code)

                if codes:
                    data['RBDSQ_10_Y'] = ",".join(str(c) for c in sorted(set(codes)))
                else:
                    data['RBDSQ_10_Y'] = None
# PHQ-9 (User Specified)
    phq_sec = re.search(r"Patient\s*Health\s*Questionnaire(.*?)(?:총점|$)", text, re.S)
    if phq_sec:
        for i in range(1, 10):
            m = re.search(rf"{i}\..*?(\d)\s*(?:\n|$)", phq_sec.group(1), re.S)
            if m: data[f'PHQ_{str(i).zfill(2)}'] = int(m.group(1))

    return data

# --- Execute and Save ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse NQ_text in a CSV and write parsed variables into columns.")
    parser.add_argument("-i", "--input", default=str(BASE_DIR / "test.csv"), help="Input CSV path (must contain column 'NQ_text').")
    parser.add_argument("-o", "--output", default=str(BASE_DIR / "test2.csv"), help="Output CSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if 'NQ_text' not in df.columns:
        raise ValueError("Input CSV must contain column 'NQ_text'.")

    parsed_rows = []
    for _, row in df.iterrows():
        parsed_rows.append(parse_nq_text(row['NQ_text']))

    parsed_df = pd.DataFrame(parsed_rows)

    # Update/add columns in the original dataframe
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    df.to_csv(args.output, index=False)
    print(f"Saved parsed CSV to: {args.output} (rows={len(df)}, parsed_cols={parsed_df.shape[1]})")
