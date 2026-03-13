import csv
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# MQ parsing (PSG Morning Questionnaire)
# -----------------------------------------------------------------------------
#
# Why this implementation:
# - In MQ_test, both *questions* and *answers* can start with "1.", "2.", ...
#   So splitting only by numbers is unreliable.
# - We instead locate segments by matching the *exact question sentences*
#   (unique Korean phrases), then parse each segment.
#
# This implementation is tuned to match the ground-truth labels in:
#   - patients18_answer - 시트1.csv
# when processing:
#   - patients18_base - 시트1.csv
# -----------------------------------------------------------------------------

def create_empty_result() -> Dict[str, Optional[object]]:
    """Create a dictionary with all required MQ output columns initialized."""
    return {
        "PSG_M_01_Hypnotics": None,
        "PSG_M_02_SubSL_HH": None,
        "PSG_M_02_SubSL_MM": None,
        "PSG_M_02_SubSL_min": None,
        "PSG_M_02_SubSL_Home": None,
        "PSG_M_03_SubSD_HH": None,
        "PSG_M_03_SubSD_MM": None,
        "PSG_M_03_SubSD_hr": None,
        "PSG_M_03_SubSD_Home": None,
        "PSG_M_04_WakeNo": None,
        "PSG_M_05_Alertness": None,
        "PSG_M_05_Complaint": None,
        "PSG_M_06_SQ_a": None,
        "PSG_M_06_SQ_b": None,
        "PSG_M_06_SQ_c": None,
        "PSG_M_06_SQ_d": None,
        "PSG_M_06_SQ_e": None,
        "PSG_M_07_Dream": None,
        "PSG_M_07_Dream_text": None,
        "PSG_M_08_Wake": None,
    }


def _make_phrase_regex(phrase: str) -> str:
    """Make a regex that matches the phrase with flexible whitespace."""
    tokens = phrase.strip().split()
    return r"\s+".join(map(re.escape, tokens))


# Order matters. We use these anchors to segment the MQ_test text.
_QUESTION_DEFS = [
    (1, "어제 밤 평소 복용하시던 수면제가 있다면 복용 여부를 알려주십시오."),
    (2, "어젯밤 불을 끈 후 잠이 드는데 까지 얼마나 걸렸습니까"),
    (3, "보통 집에서 잠이 드는데 걸리는 시간과 비교할 때 이 시간은"),
    (4, "어젯밤에 얼마나 오랫동안 잠을 잤다고 생각하십니까"),
    (5, "보통 집에서 잠자는 시간과 비교할 때 이 시간은"),
    (6, "어젯밤에 잠자는 동안 몇 번 깨었습니까"),
    (7, "현재 당신은 어떻다고 생각 되십니까"),
    (8, "오늘 아침 신체적으로 불편한 점이 있다면"),
    (9, "어젯밤 당신의 수면에 대한 평가를 내린다면 아래에 있는 다섯 가지 항목에서"),
    (10, "어젯밤 당신은 꿈을 기억하십니까"),
    (11, "오늘 아침 어떻게 잠에서 깨어났습니까"),
    (12, "어젯밤의 수면을 평소 집에서의 수면과 비교하면 어떻다고 생각하십니까"),
]

_QUESTION_PATTERNS = []
for num, phrase in _QUESTION_DEFS:
    phrase_re = _make_phrase_regex(phrase)
    # allow optional punctuation at the end ( ?, ., ) )
    pat = re.compile(rf"{num}\.\s*{phrase_re}\s*[\?\.\)]?", re.IGNORECASE)
    _QUESTION_PATTERNS.append((num, pat))


def _extract_segments_info(text: str) -> Dict[int, Dict[str, object]]:
    """
    Locate each question anchor and build segments.

    Returns:
        dict[qnum] = {
            "start": int,
            "anchor_end": int,
            "end": int,
            "seg": str,     # full segment from question start to next question start
            "ans": str,     # substring from anchor_end to end (where answer lives)
        }
    """
    matches = []
    for num, pat in _QUESTION_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append((m.start(), m.end(), num))
    matches.sort()

    info: Dict[int, Dict[str, object]] = {}
    for i, (start, anchor_end, num) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        info[num] = {
            "start": start,
            "anchor_end": anchor_end,
            "end": end,
            "seg": text[start:end],
            "ans": text[anchor_end:end],
        }
    return info


def _avg(a: float, b: float) -> float:
    return (a + b) / 2.0


def _parse_sleep_time(answer_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse time expressions from answer_text.

    Supports:
      - x 시간 y 분
      - y 분
      - x 시간
      - ranges like a-b 분, a-b 시간  (averaged)

    Returns:
      (HH, MM, total_minutes)
    """
    s = answer_text or ""

    # Explicit hours + minutes
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간\s*(\d+(?:\.\d+)?)\s*분", s)
    if m:
        hh = float(m.group(1))
        mm = float(m.group(2))
        return hh, mm, hh * 60.0 + mm

    # Range minutes: a-b 분
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~–]\s*(\d+(?:\.\d+)?)\s*분", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        mm = _avg(a, b)
        return 0.0, mm, mm

    # Single minutes
    m = re.search(r"(\d+(?:\.\d+)?)\s*분", s)
    if m:
        mm = float(m.group(1))
        return 0.0, mm, mm

    # Range hours: a-b 시간
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~–]\s*(\d+(?:\.\d+)?)\s*시간", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        avg_hr = _avg(a, b)
        hh = float(math.floor(avg_hr))
        mm = (avg_hr - hh) * 60.0
        return hh, mm, avg_hr * 60.0

    # Single hours
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간", s)
    if m:
        hh = float(m.group(1))
        return hh, 0.0, hh * 60.0

    return None, None, None


def _parse_wake_no(answer_text: str) -> Optional[float]:
    """Parse wake count (supports ranges like 2-3 번; use upper bound)."""
    s = answer_text or ""
    m = re.search(r"(\d+)\s*[-~–]\s*(\d+)\s*번", s)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        return float(max(a, b))
    m = re.search(r"(\d+)\s*번", s)
    if m:
        return float(int(m.group(1)))
    return None


def _parse_choice_1to5(answer_text: str) -> Optional[float]:
    """Parse a 1-5 choice from '3. 같다' style or from keywords."""
    s = answer_text or ""
    mapping = {"훨씬 길다": 1, "길다": 2, "같다": 3, "짧다": 4, "훨씬 짧다": 5}

    # Prefer longest keyword first to avoid '길다' matching inside '훨씬 길다'
    for k, v in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if k in s:
            return float(v)

    m = re.search(r"([1-5])\s*\.", s)
    if m:
        return float(int(m.group(1)))
    return None


def _parse_alertness_1to4(answer_text: str) -> Optional[float]:
    """Parse alertness scale 1-4."""
    s = answer_text or ""
    mapping = {
        "매우 피곤하다": 1,
        "깨어 있지만 정신이 맑지 못하다": 2,
        "편안하다": 3,
        "깨어 있으며 정신이 맑다": 4,
    }
    for k, v in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if k in s:
            return float(v)
    m = re.search(r"([1-4])\s*\.", s)
    if m:
        return float(int(m.group(1)))
    return None


def _parse_complaint(raw_after_anchor: str) -> Optional[str]:
    """
    Parse free-text complaint between Q8 anchor and next question.

    Ground-truth behavior (patients18_answer):
      - strip leading/trailing newlines
      - strip surrounding spaces normally
      - BUT if the content ends with '.' and originally had trailing spaces,
        keep exactly one trailing space (matches the label file).
    """
    if raw_after_anchor is None:
        return None

    s = raw_after_anchor.lstrip("\r\n")
    s_no_nl = s.rstrip("\r\n")

    # Special-case: keep a single trailing space after period if there was whitespace
    if re.search(r"\.\s+$", s_no_nl):
        base = s_no_nl.rstrip()
        if base.endswith("."):
            return base + " "

    s_clean = s_no_nl.strip()
    return s_clean if s_clean != "" else None


def _parse_dream_text(raw_after_detail_prompt: str) -> Optional[str]:
    """
    Parse free-text dream description after '자세히 설명하십시오.'.

    Ground-truth behavior (patients18_answer):
      - remove leading whitespace
      - remove trailing '\n' but keep '\r'
      - if original ended with '\r\n' and after removing '\n' the string ends with '.\r',
        add '\n' back (so it ends with '\r\n')
    """
    if raw_after_detail_prompt is None:
        return None

    s = raw_after_detail_prompt.lstrip()
    if s == "":
        return None

    had_crlf = s.endswith("\r\n")
    s_no_n = s.rstrip("\n")  # removes only LF, keeps CR
    if had_crlf and s_no_n.endswith(".\r"):
        s_no_n = s_no_n + "\n"

    return s_no_n if s_no_n != "" else None


def _parse_wake_method_1to5(answer_text: str) -> Optional[float]:
    """Parse wake method choice 1-5."""
    s = answer_text or ""
    mapping = {"소음으로": 1, "불안감으로": 2, "검사실에서 깨워": 3, "자발적으로": 4, "기타": 5}
    for k, v in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if k in s:
            return float(v)
    m = re.search(r"([1-5])\s*\.", s)
    if m:
        return float(int(m.group(1)))
    return None


def parse_mq_text(text) -> Dict[str, Optional[object]]:
    """
    Parse MQ_test column content and extract values into structured MQ columns.

    IMPORTANT: This parser intentionally anchors by the exact question sentences.
    """
    if pd.isna(text) or str(text) == "":
        return create_empty_result()

    text = str(text)
    result = create_empty_result()
    info = _extract_segments_info(text)

    # Q1: hypnotics (0=아니오, 1=예)
    if 1 in info:
        ans = info[1]["ans"]
        if "아니오" in ans:
            result["PSG_M_01_Hypnotics"] = 0
        elif "예" in ans:
            result["PSG_M_01_Hypnotics"] = 1

    # Q2: sleep latency (HH/MM/total minutes)
    if 2 in info:
        hh, mm, total = _parse_sleep_time(info[2]["ans"])
        result["PSG_M_02_SubSL_HH"] = hh
        result["PSG_M_02_SubSL_MM"] = mm
        result["PSG_M_02_SubSL_min"] = total

    # Q3: latency compared to home (1-5)
    if 3 in info:
        result["PSG_M_02_SubSL_Home"] = _parse_choice_1to5(info[3]["ans"])

    # Q4: sleep duration (HH/MM/total hours)
    if 4 in info:
        hh, mm, total_min = _parse_sleep_time(info[4]["ans"])
        result["PSG_M_03_SubSD_HH"] = hh
        result["PSG_M_03_SubSD_MM"] = mm
        if total_min is not None:
            # Match label file precision
            result["PSG_M_03_SubSD_hr"] = round(total_min / 60.0, 9)

    # Q5: duration compared to home (1-5)
    if 5 in info:
        result["PSG_M_03_SubSD_Home"] = _parse_choice_1to5(info[5]["ans"])

    # Q6: wake-ups count
    if 6 in info:
        result["PSG_M_04_WakeNo"] = _parse_wake_no(info[6]["ans"])

    # Q7: alertness (1-4)
    if 7 in info:
        result["PSG_M_05_Alertness"] = _parse_alertness_1to4(info[7]["ans"])

    # Q8: complaint (free)
    if 8 in info:
        result["PSG_M_05_Complaint"] = _parse_complaint(info[8]["ans"])

    # Q9: five sub-questions (㉠-㉤), each 1-7
    if 9 in info:
        ans = info[9]["ans"]
        sub_map = [
            ("㉠", "PSG_M_06_SQ_a"),
            ("㉡", "PSG_M_06_SQ_b"),
            ("㉢", "PSG_M_06_SQ_c"),
            ("㉣", "PSG_M_06_SQ_d"),
            ("㉤", "PSG_M_06_SQ_e"),
        ]
        for sym, key in sub_map:
            m = re.search(re.escape(sym) + r"\s*(\d+)", ans)
            if m:
                result[key] = float(int(m.group(1)))

    # Q10: dream (0/1) and dream text
    if 10 in info:
        ans = info[10]["ans"]
        if "아니오" in ans:
            result["PSG_M_07_Dream"] = 0
        elif "예" in ans:
            result["PSG_M_07_Dream"] = 1

        seg = info[10]["seg"]
        detail_idx = seg.find("자세히 설명하십시오.")
        if detail_idx != -1:
            raw = seg[detail_idx + len("자세히 설명하십시오.") :]
            result["PSG_M_07_Dream_text"] = _parse_dream_text(raw)

    # Q11: wake method (1-5)
    if 11 in info:
        result["PSG_M_08_Wake"] = _parse_wake_method_1to5(info[11]["ans"])

    return result


# Backward-compatibility: keep the previous name.
parse_mq_text_alternative = parse_mq_text


def _read_csv_with_fallback_encodings(path: str) -> pd.DataFrame:
    """Read CSV using common Korean encodings."""
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last try without specifying encoding
    return pd.read_csv(path)


def _write_csv_quote_newlines(df: pd.DataFrame, output_csv: str, encoding: str = "utf-8-sig") -> None:
    """
    Write CSV that safely handles fields containing '\r' as well as '\n'.

    NOTE:
    - pandas.DataFrame.to_csv does NOT quote fields that only contain '\r'
      (carriage return), which can corrupt the output CSV (extra rows).
    - Python's csv.writer correctly quotes any field containing '\r' or '\n'.
    """
    with open(output_csv, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        writer.writerow(df.columns.tolist())

        # Iterate efficiently without converting the whole frame to object
        for row in df.itertuples(index=False, name=None):
            out_row = []
            for v in row:
                # Keep blanks for NaN
                if pd.isna(v):
                    out_row.append("")
                else:
                    out_row.append(v)
            writer.writerow(out_row)


def process_csv(input_csv: str, output_csv: str = "mq_parsed_output.csv") -> str:
    """
    Load a CSV with an MQ_test column, parse MQ, and write updated CSV.
    Returns output path.
    """
    df = _read_csv_with_fallback_encodings(input_csv)

    if "MQ_test" not in df.columns:
        raise ValueError(
            f"Error: 'MQ_test' column not found in {input_csv}. Available columns: {list(df.columns)}"
        )

    parsed_rows = [parse_mq_text(mq_text) for mq_text in df["MQ_test"]]
    parsed_df = pd.DataFrame(parsed_rows)

    # Update original dataframe with parsed columns
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    _write_csv_quote_newlines(df, output_csv, encoding="utf-8-sig")
    return output_csv


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
    else:
        in_path = input("Enter CSV file path (or press Enter for 'test2.csv'): ").strip() or str(BASE_DIR / "test2.csv")

    out_path = sys.argv[2] if len(sys.argv) >= 3 else str(BASE_DIR / "test3.csv")

    out_file = process_csv(in_path, out_path)
    print("Processing complete!")
    print(f"Output saved to: {out_file}")
