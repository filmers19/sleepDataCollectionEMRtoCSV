from pathlib import Path
import re
from typing import Optional

import pandas as pd


def _looks_like_saf(text: str) -> bool:
    """
    .saf 파서가 실제로 사용할 수 있는 형태인지 간단히 확인.
    시작 시간은 '+' 접두사 형식으로 반복해서 등장하는 게 정상입니다.
    """
    plus_count = text.count("+")
    return plus_count >= 2 and bool(re.search(r"\+\d+(?:\.\d+)?", text))


def read_saf(filepath: str, fill_full_timeline: bool = False, sample_period: Optional[float] = None) -> pd.DataFrame:
    """
    MASS .saf 파일을 읽어서 [start_time, duration, label] DataFrame으로 반환
    - fill_full_timeline=True: + 타임스탬프를 전부 펼쳐서 라벨 구간을 채움
    """
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
        "iso-8859-1",
        "utf-16",
        "utf-16le",
        "utf-16be",
    ]

    raw = Path(filepath).read_bytes()
    decoded = None
    loaded_encoding = None

    for enc in encodings:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue

        if _looks_like_saf(text):
            decoded = text
            loaded_encoding = enc
            break

    if decoded is None:
        # 마지막 안전장치: 깨져도 파싱이 가능한 latin1 바이트 1:1 문자열로 강제 처리
        decoded = raw.decode("latin1")
        loaded_encoding = "latin1 (fallback)"

    print(f"Loaded with encoding: {loaded_encoding}")

    records = []
    all_starts = []

    def parse_float(token: str):
        token = token.strip("\x00\x14").strip()
        if not token:
            return None
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token):
            return None
        try:
            return float(token)
        except ValueError:
            return None

    # 예시:
    # +30310.3670235\x1530.0\x14Sleep stage ?\x14
    # +30312.0\x14\x14
    for seg in re.findall(r"\+\d+(?:\.\d+)?[^\+]*", decoded):
        seg = seg[1:]  # remove '+'
        start_text, has_annotation, rest = seg.partition("\x15")
        start = parse_float(start_text)
        if start is None:
            continue

        all_starts.append(start)

        duration = pd.NA
        label = ""

        if not has_annotation:
            continue

        # Annotation block: duration\x14(label)\x14...
        m = re.match(r"\x14*([+-]?\d+(?:\.\d+)?)(?:\x14(.*))?", rest, re.S)
        if not m:
            continue

        duration = parse_float(m.group(1))
        if m.group(2):
            label = re.sub(r"[\x00\x14\x15]", " ", m.group(2))
            label = re.sub(r"\s+", " ", label).strip()

        if duration is None and not label:
            continue

        records.append([start, duration, label])

    events = pd.DataFrame(records, columns=["start_time", "duration", "label"]).sort_values("start_time")

    if not fill_full_timeline:
        return events

    if not all_starts:
        return events

    all_starts_df = pd.DataFrame({"start_time": sorted(set(all_starts))})
    all_starts_df["start_time"] = all_starts_df["start_time"].astype(float)
    all_starts_df = all_starts_df.sort_values("start_time").reset_index(drop=True)

    deltas = all_starts_df["start_time"].diff().dropna()
    fallback_step = sample_period if sample_period is not None else 2.0
    base_step = float(deltas.mode().iloc[0]) if not deltas.empty else fallback_step
    base_step = fallback_step if base_step <= 0 else base_step

    all_starts_df["duration"] = all_starts_df["start_time"].shift(-1) - all_starts_df["start_time"]
    all_starts_df["duration"] = all_starts_df["duration"].fillna(base_step)

    ann = events[["start_time", "duration", "label"]].rename(columns={"duration": "ann_duration", "label": "ann_label"})

    all_starts_df = pd.merge_asof(
        all_starts_df,
        ann.sort_values("start_time"),
        on="start_time",
        direction="backward",
    )

    all_starts_df["duration"] = all_starts_df["ann_duration"].fillna(all_starts_df["duration"])
    all_starts_df["label"] = all_starts_df["ann_label"].ffill().fillna("")
    all_starts_df = all_starts_df.drop(columns=["ann_duration", "ann_label"])

    return all_starts_df


if __name__ == "__main__":
    df = read_saf("01-01-0001.saf")
    print(df.to_string(index=False))
    print("\nUnique labels:")
    print(df["label"].unique())
