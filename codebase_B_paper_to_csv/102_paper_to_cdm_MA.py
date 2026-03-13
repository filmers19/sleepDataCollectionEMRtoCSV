from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("sleep_cdm_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# -----------------------------
# Helpers
# -----------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_env() -> None:
    dotenv_path = REPO_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        load_dotenv(override=False)


def iter_patient_folders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def iter_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)


def image_to_data_url(image_path: Path, max_side: int = 2048) -> str:
    """
    Convert an image to a base64 data URL.
    Resizes (downscales) large images to avoid huge payloads while keeping OCR workable.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def safe_extract_json(text: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extraction:
    - Finds first {...} block
    - Attempts json.loads
    """
    text = text.strip()
    # If model returned pure JSON, great
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Otherwise, find a JSON object substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from model output. Output starts with: {text[:200]!r}")


def normalize_value(v: Any) -> Any:
    """
    Normalize trivial junk to None. Keep numbers, strings, lists as-is.
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"null", "none", "n/a", "na"}:
            return None
        return s
    return v


def merge_flat_json(dicts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Merge multiple flat JSON dicts.
    Strategy:
    - Prefer first non-null value
    - Track conflicts in a separate dict for audit
    """
    merged: Dict[str, Any] = {}
    conflicts: Dict[str, List[Any]] = {}

    for d in dicts:
        for k, v in d.items():
            v = normalize_value(v)
            if v is None:
                continue
            if k not in merged or merged[k] is None or merged[k] == "":
                merged[k] = v
            else:
                if merged[k] != v:
                    conflicts.setdefault(k, [])
                    # store unique conflict candidates
                    if merged[k] not in conflicts[k]:
                        conflicts[k].append(merged[k])
                    if v not in conflicts[k]:
                        conflicts[k].append(v)
    return merged, conflicts


# -----------------------------
# CDM "RAG" Retriever (local TF-IDF)
# -----------------------------
@dataclass
class CDMRow:
    key: str
    desc: str
    format_range: str
    options: Dict[str, str]  # code -> label


class CDMRetriever:
    """
    Local retrieval over CDM rows using TF-IDF.
    Works well for mixed Korean/English by using char_wb ngrams.
    """

    def __init__(self, cdm_csv_path: Path):
        self.cdm_df = pd.read_csv(cdm_csv_path)
        self.rows: List[CDMRow] = []
        self._texts: List[str] = []

        option_cols = [c for c in self.cdm_df.columns if re.fullmatch(r"\d+", str(c))]
        for _, r in self.cdm_df.iterrows():
            key = str(r.get("csv key", "")).strip()
            if not key or key.lower() == "nan":
                continue
            desc = str(r.get("설명", "")).strip()
            fr = str(r.get("Format/Range", "")).strip()

            opts: Dict[str, str] = {}
            for c in option_cols:
                val = r.get(c)
                if pd.isna(val):
                    continue
                label = str(val).strip()
                if label == "":
                    continue
                opts[str(c)] = label

            self.rows.append(CDMRow(key=key, desc=desc, format_range=fr, options=opts))

        # Build TF-IDF index text for each row
        for row in self.rows:
            opt_str = " | ".join([f"{code}:{label}" for code, label in sorted(row.options.items(), key=lambda x: int(x[0]))])
            txt = f"KEY={row.key}\nDESC={row.desc}\nFORMAT={row.format_range}\nOPTIONS={opt_str}"
            self._texts.append(txt)

        # char_wb handles Korean/English without tokenization pain
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self._texts)

    def search(self, query: str, k: int = 60) -> List[Tuple[CDMRow, float]]:
        qv = self.vectorizer.transform([query[:8000]])  # cap length for speed
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        out: List[Tuple[CDMRow, float]] = []
        for i in idxs:
            out.append((self.rows[int(i)], float(sims[int(i)])))
        return out


def format_candidate_rows(cands: List[Tuple[CDMRow, float]], max_chars: int = 12000) -> str:
    """
    Format candidates for the prompt, with a hard cap.
    """
    parts: List[str] = []
    total = 0
    for row, score in cands:
        opt_str = ", ".join([f"{code}={label}" for code, label in sorted(row.options.items(), key=lambda x: int(x[0]))])
        block = (
            f"- {row.key}\n"
            f"  desc: {row.desc}\n"
            f"  format/range: {row.format_range}\n"
            f"  options: {opt_str}\n"
            f"  (retrieval_score={score:.4f})\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


# -----------------------------
# Agent A: Image -> Text (Gemini)
# -----------------------------
def build_agent_a() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")
    model = os.getenv("GEMINI_MODEL", "gemini-3.0-vision")  # set to your real Gemini 3 vision model id
    # NOTE: ChatGoogleGenerativeAI reads GOOGLE_API_KEY from env by default
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        max_output_tokens=8192,
    )


AGENT_A_SYSTEM = """You are an OCR/transcription assistant for sleep clinic documents.
The input is a scanned photo of medical sleep questionnaire or sleep study paperwork (Korean/English mixed).
Task:
- Transcribe ALL visible printed + handwritten text as faithfully as possible.
- For checkboxes / radio buttons / circled options / check marks: explicitly state which option is selected.
  If options have numbers (e.g., ①②③, or (0)(1)(2)(3)), include the selected number and label.
- For tables: preserve row labels and indicate checked/selected cells.
- If you are unsure about a word, mark it with (unclear).
- The answer might have not been filled in the questionnaire. In that case, mark it with (not filled).
Output: plain text only (no JSON)."""

AGENT_A_USER_PREFIX = """Process each question at a time and extract all text and marked answers from this image."""


async def agent_a_image_to_text(llm_a: ChatGoogleGenerativeAI, image_path: Path) -> str:
    data_url = image_to_data_url(image_path)
    msg = [
        SystemMessage(content=AGENT_A_SYSTEM),
        HumanMessage(
            content=[
                {"type": "text", "text": AGENT_A_USER_PREFIX},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]
    resp = await llm_a.ainvoke(msg)
    return (resp.content or "").strip()


# -----------------------------
# Agent B: Text -> Flat JSON (Gemini) with CDM retrieval context
# -----------------------------
def build_agent_b() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or export it in your shell.")
    model = os.getenv("GEMINI_MODEL_B", os.getenv("GEMINI_MODEL", "gemini-3.0-vision"))
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        max_output_tokens=4096,
    )


AGENT_B_SYSTEM = """You are a clinical data extraction assistant.
You will be given:
1) OCR text from a sleep questionnaire / sleep study paperwork image (Korean/English).
2) A shortlist of candidate CDM fields (keys) with descriptions and allowed ranges/options.

Your job:
- Produce ONE flat JSON object (single-layer) mapping CDM keys -> extracted values.
- Use ONLY keys that appear in the candidate list.
- Do NOT invent values. If the OCR text does not contain a value, omit the key.
- When the candidate field has coded options (e.g., 0-3 or 1-5), output the numeric code, not the label.
- Normalize units:
  - Height: cm (number)
  - Weight: kg (number)
  - Times: if CDM has separate HH and MM keys, fill those numeric fields.
  - Dates: follow the CDM format/range (often YYYYMMDD).
- Output JSON only. No extra text."""

def build_agent_b_user(ocr_text: str, candidates_block: str) -> str:
    return f"""OCR TEXT:
\"\"\"{ocr_text[:12000]}\"\"\"

CANDIDATE CDM FIELDS (you may ONLY use these keys):
{candidates_block}

Return a single JSON object with extracted key-value pairs."""


async def agent_b_text_to_json(
    llm_b: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    ocr_text: str,
    top_k: int = 80,
    repair_llm: Optional[ChatGoogleGenerativeAI] = None,
) -> Dict[str, Any]:
    cands = retriever.search(ocr_text, k=top_k)
    candidates_block = format_candidate_rows(cands, max_chars=12000)

    msg = [
        SystemMessage(content=AGENT_B_SYSTEM),
        HumanMessage(content=build_agent_b_user(ocr_text, candidates_block)),
    ]

    resp = await llm_b.ainvoke(msg)
    raw = (resp.content or "").strip()

    try:
        obj = safe_extract_json(raw)
        return obj
    except Exception:
        # Repair once with the same model (or repair_llm if provided)
        fixer = repair_llm or llm_b
        fix_prompt = [
            SystemMessage(content="Fix the following into valid JSON object only. Do not add any explanation."),
            HumanMessage(content=raw),
        ]
        fixed = await fixer.ainvoke(fix_prompt)
        return safe_extract_json((fixed.content or "").strip())


# -----------------------------
# Pipeline Orchestration
# -----------------------------
async def gather_with_concurrency(n: int, coros: Iterable):
    sem = asyncio.Semaphore(n)

    async def _wrap(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_wrap(c) for c in coros])


def build_output_row(merged: Dict[str, Any], output_columns: List[str]) -> Dict[str, Any]:
    row = {c: None for c in output_columns}
    for k, v in merged.items():
        if k in row:
            row[k] = v
    return row


async def process_one_patient(
    patient_dir: Path,
    llm_a: ChatGoogleGenerativeAI,
    llm_b: ChatGoogleGenerativeAI,
    retriever: CDMRetriever,
    output_columns: List[str],
    a_concurrency: int,
    b_concurrency: int,
    save_intermediate: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    images = iter_images(patient_dir)
    if not images:
        logger.warning("No images found in %s", patient_dir)
        return {"patient": patient_dir.name, "row": None, "conflicts": {}}

    logger.info("Patient %s: %d images", patient_dir.name, len(images))

    # Agent A in parallel
    a_tasks = [agent_a_image_to_text(llm_a, img) for img in images]
    ocr_texts = await gather_with_concurrency(a_concurrency, a_tasks)

    if save_intermediate:
        (out_dir / "intermediate" / patient_dir.name).mkdir(parents=True, exist_ok=True)
        for img, txt in zip(images, ocr_texts):
            (out_dir / "intermediate" / patient_dir.name / f"{img.stem}.txt").write_text(txt, encoding="utf-8")

    # Agent B in parallel
    b_tasks = [agent_b_text_to_json(llm_b, retriever, txt) for txt in ocr_texts if txt.strip()]
    json_dicts = await gather_with_concurrency(b_concurrency, b_tasks)

    if save_intermediate:
        for i, jd in enumerate(json_dicts):
            (out_dir / "intermediate" / patient_dir.name / f"page_{i:03d}.json").write_text(
                json.dumps(jd, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    merged, conflicts = merge_flat_json(json_dicts)

    # Build final CSV row aligned to example.csv columns
    row = build_output_row(merged, output_columns)

    return {"patient": patient_dir.name, "row": row, "conflicts": conflicts}


async def run_pipeline(
    input_root: Path,
    cdm_csv: Path,
    example_csv: Path,
    output_dir: Path,
    a_concurrency: int,
    b_concurrency: int,
    save_intermediate: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Column order: match example.csv exactly
    example_df = pd.read_csv(example_csv)
    output_columns = list(example_df.columns)

    retriever = CDMRetriever(cdm_csv)

    llm_a = build_agent_a()
    llm_b = build_agent_b()

    patient_dirs = iter_patient_folders(input_root)
    logger.info("Found %d patient folders", len(patient_dirs))

    results: List[Dict[str, Any]] = []
    for pdir in patient_dirs:
        try:
            res = await process_one_patient(
                patient_dir=pdir,
                llm_a=llm_a,
                llm_b=llm_b,
                retriever=retriever,
                output_columns=output_columns,
                a_concurrency=a_concurrency,
                b_concurrency=b_concurrency,
                save_intermediate=save_intermediate,
                out_dir=output_dir,
            )
            results.append(res)

            # Write per-patient CSV
            if res["row"] is not None:
                df_one = pd.DataFrame([res["row"]], columns=output_columns)
                df_one.to_csv(output_dir / f"{pdir.name}.csv", index=False)

            # Write conflicts report
            if res["conflicts"]:
                (output_dir / "conflicts").mkdir(exist_ok=True)
                (output_dir / "conflicts" / f"{pdir.name}_conflicts.json").write_text(
                    json.dumps(res["conflicts"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        except Exception as e:
            logger.exception("Failed processing patient folder %s: %s", pdir, e)

    # Write combined CSV
    rows = [r["row"] for r in results if r.get("row") is not None]
    if rows:
        df_all = pd.DataFrame(rows, columns=output_columns)
        df_all.to_csv(output_dir / "all_patients.csv", index=False)
        logger.info("Wrote %d rows to %s", len(rows), output_dir / "all_patients.csv")
    else:
        logger.warning("No patient rows produced.")


def main():
    load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True, help="Root directory containing one folder per patient")
    ap.add_argument("--cdm_csv", type=str, required=True, help="Path to cdm.csv")
    ap.add_argument("--example_csv", type=str, required=True, help="Path to example.csv (column order template)")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for CSV files")
    ap.add_argument("--a_concurrency", type=int, default=4, help="Parallelism for Agent A (image->text)")
    ap.add_argument("--b_concurrency", type=int, default=8, help="Parallelism for Agent B (text->json)")
    ap.add_argument("--save_intermediate", action="store_true", help="Save OCR text and per-page JSON outputs")
    args = ap.parse_args()

    asyncio.run(
        run_pipeline(
            input_root=Path(args.input_root),
            cdm_csv=Path(args.cdm_csv),
            example_csv=Path(args.example_csv),
            output_dir=Path(args.output_dir),
            a_concurrency=args.a_concurrency,
            b_concurrency=args.b_concurrency,
            save_intermediate=args.save_intermediate,
        )
    )


if __name__ == "__main__":
    main()
