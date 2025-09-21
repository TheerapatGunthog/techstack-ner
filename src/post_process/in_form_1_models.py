from pathlib import Path
import os
import json
import re
import time
import contextlib
import numpy as np
import pandas as pd
import requests
from collections import defaultdict
from tqdm import tqdm

# ===== Runtime recommendations =====
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ===== Optional HF imports =====
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TokenClassificationPipeline,
        logging,
    )

    logging.set_verbosity_error()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForTokenClassification = None
    TokenClassificationPipeline = None

# =========================
# Configuration settings
# =========================
PROJECT_PATH = Path(os.getcwd())
MODEL_PATH_ROBERTA_BASE = PROJECT_PATH / "models/roberta-base/best_model/"
MODEL_PATH_ROBERTA_LARGE = PROJECT_PATH / "models/bootstrapping03/best_model/"
SINGLE_MODEL_PATH = MODEL_PATH_ROBERTA_BASE
SINGLE_CLASSES = {"PSML", "DB", "CP", "FAL", "HW", "TAS"}
DATA_PATH = PROJECT_PATH / "data/interim/preprocessed-data/scraping_data.csv"

OUTPUT_DIR = PROJECT_PATH / "data/post_processed"
OUTPUT_CSV_RAW = OUTPUT_DIR / "all_predictions.csv"
OUTPUT_CSV_DEDUP = OUTPUT_DIR / "all_predictions_dedup.csv"

CLASS_THRESHOLDS = {
    "PSML": 0.0,
    "DB": 0.0,
    "CP": 0.0,
    "FAL": 0.0,
    "HW": 0.0,
    "TAS": 0.93,
}

# Ollama for topic normalization only
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:latest"
SESSION = requests.Session()

# =========================
# Helper functions for Span
# =========================


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def pick_longest_highest(spans):
    spans = sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"]), -x["score"]))
    kept = []
    for s in spans:
        conflict = False
        for k in kept:
            if overlaps(k["start"], k["end"], s["start"], s["end"]) and k["label"] == s["label"]:
                k_len = k["end"] - k["start"]
                s_len = s["end"] - s["start"]
                if s_len > k_len or (s_len == k_len and s["score"] > k["score"]):
                    k.update(s)
                conflict = True
                break
        if not conflict:
            kept.append(s)
    return kept


SHORT_VALID_1CHAR = {"R", "C"}


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_" or c == "-"


def _clip_to_word_bounds(text: str, s: int, e: int):
    while s > 0 and _is_word_char(text[s - 1]):
        s -= 1
    n = len(text)
    while e < n and _is_word_char(text[e]):
        e += 1
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _clean_entity_span(text: str, start: int, end: int, label: str, txt: str):
    s, e = _clip_to_word_bounds(text, start, end)
    if e - s == 1 and text[s:e] not in SHORT_VALID_1CHAR:
        s, e = start, end
    return s, e, text[s:e]


def group_ner_entities(ner_results):
    grouped = []
    cur_tokens, cur_scores, cur_label = [], [], ""
    cur_start, cur_end = None, None
    for ent in ner_results:
        tag = ent.get("entity", "")
        if tag == "O" or tag == "":
            if cur_tokens:
                word = "".join(cur_tokens).replace("Ġ", " ").strip()
                grouped.append(
                    {
                        "word": word,
                        "label": cur_label,
                        "score": float(np.mean(cur_scores)),
                        "start": cur_start,
                        "end": cur_end,
                    }
                )
                cur_tokens, cur_scores, cur_label = [], [], ""
                cur_start, cur_end = None, None
            continue
        label = tag[2:] if "-" in tag else tag
        if tag.startswith("B-") or not cur_tokens or cur_label != label:
            if cur_tokens:
                word = "".join(cur_tokens).replace("Ġ", " ").strip()
                grouped.append(
                    {
                        "word": word,
                        "label": cur_label,
                        "score": float(np.mean(cur_scores)),
                        "start": cur_start,
                        "end": cur_end,
                    }
                )
            cur_tokens = [ent.get("word", "")]
            cur_scores = [float(ent.get("score", 0.0))]
            cur_label = label
            cur_start = int(ent.get("start", -1))
            cur_end = int(ent.get("end", -1))
        else:
            cur_tokens.append(ent.get("word", ""))
            cur_scores.append(float(ent.get("score", 0.0)))
            cur_end = int(ent.get("end", cur_end))
    if cur_tokens:
        word = "".join(cur_tokens).replace("Ġ", " ").strip()
        grouped.append(
            {
                "word": word,
                "label": cur_label,
                "score": float(np.mean(cur_scores)),
                "start": cur_start,
                "end": cur_end,
            }
        )
    return grouped


def _postprocess_entities(text: str, entities):
    out = []
    for e in entities:
        s, epos, clean_txt = _clean_entity_span(text, e["start"], e["end"], e["label"], e["word"])
        out.append(
            {
                "text": clean_txt,
                "label": e["label"],
                "score": float(e["score"]),
                "start": int(s),
                "end": int(epos),
            }
        )
    return out


def _merge_per_label_with_policy(entities):
    by_label = defaultdict(list)
    for e in entities:
        by_label[e["label"]].append(e)
    merged = []
    for spans in by_label.values():
        merged.extend(pick_longest_highest(spans))
    return sorted(merged, key=lambda x: x["start"])

# =========================
# Create a fast HF pipeline on 8GB RAM
# =========================


def _build_fast_pipeline(model_dir, device, dtype, max_length=256):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.model_max_length = max_length
    mod = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    mod.eval()
    if device and hasattr(device, "type") and device.type == "cuda":
        mod.to(device)
        if dtype == (torch.float16 if torch else None):
            mod.half()
    return TokenClassificationPipeline(
        model=mod,
        tokenizer=tok,
        device=(device.index if device and device.type == "cuda" else -1),
        aggregation_strategy="none",
    )


def _predict_batch(pipe, texts, allowed_labels, thresholds, batch_size=32):
    out_per_text = []
    if not texts:
        return out_per_text
    ctx = torch.inference_mode() if torch else contextlib.nullcontext()
    with ctx:
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"Inference {','.join(sorted(allowed_labels))}",
                      unit="batch"):
            chunk = texts[i : i + batch_size]
            toks_list = pipe(chunk)
            for text, toks in zip(chunk, toks_list):
                ents = group_ner_entities(toks)
                ents = _postprocess_entities(text, ents)
                kept = [
                    e for e in ents
                    if e["label"] in allowed_labels
                    and e["score"] >= float(thresholds.get(e["label"], 0.0))
                ]
                kept = _merge_per_label_with_policy(kept)
                out_per_text.append(
                    [{"entity": e["text"], "class": e["label"]} for e in kept]
                )
    return out_per_text

# =========================
# Ollama helper: Used for topic normalization only
# =========================


def _ollama_generate(model: str, prompt: str, max_retries: int = 2, timeout: int = 60):
    payload = {"model": model, "prompt": prompt, "stream": False}
    for attempt in range(max_retries):
        try:
            r = SESSION.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception:
            if attempt == max_retries - 1:
                return ""
            time.sleep(1.0 * (attempt + 1))


TOPIC_PROMPT_TMPL = (
    "You are a strict assistant for normalizing job titles.\n"
    "Return only the standardized job title, without company or department names.\n"
    'JSON format: {"title": "..."}\n'
    "Input data: {topic}\n"
    "JSON:"
)


def normalize_topic_with_llm(topic: str, model: str = LLM_MODEL):
    resp = _ollama_generate(model, TOPIC_PROMPT_TMPL.format(topic=topic))
    m = re.search(r"\{[\s\S]*\}", resp)
    title = topic.strip()
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and parsed.get("title"):
                title = str(parsed.get("title")).strip()
        except Exception:
            pass
    return title


# =========================
# Position-aware normalization helpers
# =========================
_POS_PAT_BRACKETS = re.compile(r"<\s*([^>]+?)\s*>")
_POS_PAT_PHRASE = re.compile(r"what can i earn as a[n]?\s*(.*)", re.IGNORECASE)


def extract_position_name(pos: str) -> str:
    if not isinstance(pos, str):
        return ""
    pos = pos.strip()
    if not pos:
        return ""
    m = _POS_PAT_BRACKETS.search(pos)
    if m:
        return m.group(1).strip()
    m = _POS_PAT_PHRASE.search(pos)
    if m:
        return m.group(1).strip()
    return pos


def normalize_with_position(topic: str, position: str, cache: dict) -> str:
    if position is None or (isinstance(position, float) and pd.isna(position)) or str(position).strip() == "":
        if topic not in cache:
            cache[topic] = normalize_topic_with_llm(topic)
        return cache[topic]
    name = extract_position_name(position)
    return name if name else (cache.get(topic) or normalize_topic_with_llm(topic))

# =========================
# Main process
# =========================


def run():
    # Load data
    ds = pd.read_csv(DATA_PATH)
    required_cols = {"Topic", "Sentence_Index", "Segmented_Qualification"}
    missing = required_cols - set(ds.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    has_position = "Position" in ds.columns

    # Create pipeline
    if torch is not None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.float16 if (torch and device.type == "cuda") else torch.float32
        try:
            pipe = _build_fast_pipeline(SINGLE_MODEL_PATH, device, dtype, max_length=256)
        except Exception:
            pipe = None
    else:
        pipe = None

    # Batch NER prediction -> Convert to row per entity
    rows = []
    texts = [str(x) for x in ds["Segmented_Qualification"].tolist()]

    if pipe is not None:
        single_preds = _predict_batch(pipe, texts, SINGLE_CLASSES, CLASS_THRESHOLDS, batch_size=32)
        for i, tpl in enumerate(tqdm(ds.itertuples(index=False),
                                     total=len(ds),
                                     desc="Collecting NER results",
                                     unit="row")):
            topic = tpl.Topic
            si = tpl.Sentence_Index
            pos = getattr(tpl, "Position", None) if has_position else None
            preds = single_preds[i] if i < len(single_preds) else []
            if preds:
                for p in preds:
                    rows.append(
                        {
                            "Topic": topic,
                            "Sentence_Index": si,
                            "Position": pos,
                            "Entity": p["entity"],
                            "Class": p["class"],
                        }
                    )
            else:
                rows.append(
                    {"Topic": topic, "Sentence_Index": si, "Position": pos, "Entity": "", "Class": ""}
                )
    else:
        pred_exists = "Predictions" in ds.columns
        for tpl in tqdm(ds.itertuples(index=False),
                        total=len(ds),
                        desc="Collecting DEV predictions",
                        unit="row"):
            topic = tpl.Topic
            si = tpl.Sentence_Index
            pos = getattr(tpl, "Position", None) if has_position else None
            merged = []
            if pred_exists and isinstance(getattr(tpl, "Predictions", None), str):
                try:
                    merged = json.loads(getattr(tpl, "Predictions"))
                except Exception:
                    merged = []
            if merged:
                for p in merged:
                    rows.append(
                        {
                            "Topic": topic,
                            "Sentence_Index": si,
                            "Position": pos,
                            "Entity": p.get("entity", ""),
                            "Class": p.get("class", ""),
                        }
                    )
            else:
                rows.append(
                    {"Topic": topic, "Sentence_Index": si, "Position": pos, "Entity": "", "Class": ""}
                )

    df = pd.DataFrame(rows)

    # ===== Aggregate data by Topic =====
    agg_raw = df.groupby(["Topic", "Sentence_Index"], as_index=False, sort=False).agg(
        {
            "Position": "first",
            "Entity": lambda s: ", ".join([x for x in s if isinstance(x, str) and x]),
            "Class": lambda s: ", ".join([x for x in s if isinstance(x, str) and x]),
        }
    )

    # Remove duplicate rows
    def _dedup_row_entities(entity_str: str, class_str: str):
        ents = [e.strip() for e in (entity_str or "").split(",") if e.strip()]
        clss = [c.strip() for c in (class_str or "").split(",") if c.strip()]
        kept_e, kept_c, seen = [], [], set()
        for e, c in zip(ents, clss):
            if e not in seen:
                kept_e.append(e)
                kept_c.append(c)
                seen.add(e)
        return ", ".join(kept_e), ", ".join(kept_c)

    tqdm.pandas(desc="Removing duplicates in rows")
    dedup_pairs = agg_raw[["Entity", "Class"]].progress_apply(
        lambda r: _dedup_row_entities(r["Entity"], r["Class"]),
        axis=1,
        result_type="expand",
    )
    agg_dedup = agg_raw.copy()
    agg_dedup[["Entity", "Class"]] = dedup_pairs

    # ===== Normalize Topic per row with Position-aware policy =====
    topic_norm_cache = {}
    agg_raw["Topic_Normalized"] = agg_raw.apply(
        lambda r: normalize_with_position(str(r["Topic"]), r.get("Position", None), topic_norm_cache),
        axis=1,
    )
    agg_dedup["Topic_Normalized"] = agg_dedup.apply(
        lambda r: normalize_with_position(str(r["Topic"]), r.get("Position", None), topic_norm_cache),
        axis=1,
    )

    # Select result columns
    out_raw = agg_raw[["Topic_Normalized", "Sentence_Index", "Entity", "Class"]].copy()
    out_dedup = agg_dedup[["Topic_Normalized", "Sentence_Index", "Entity", "Class"]].copy()

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_raw.to_csv(OUTPUT_CSV_RAW, index=False, encoding="utf-8-sig")
    out_dedup.to_csv(OUTPUT_CSV_DEDUP, index=False, encoding="utf-8-sig")
    print(f"CSV (raw)   -> {OUTPUT_CSV_RAW}")
    print(f"CSV (dedup) -> {OUTPUT_CSV_DEDUP}")


if __name__ == "__main__":
    run()
