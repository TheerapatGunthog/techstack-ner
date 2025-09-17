import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import uuid
from collections import defaultdict

# =========================
# Project paths & config
# =========================
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
MODEL_PATH = PROJECT_PATH / "models/roberta-base/best_model/"
DATA_PATH = PROJECT_PATH / "data/interim/preprocessed-data/kaggle_data.csv"

# Raw (debug/flexible) JSON
OUTPUT_JSON_RAW = (
    PROJECT_PATH
    / "data/interim/bootstrapping/003/labels_by_boostrapping_models_two.json"
)
# Label Studio import-ready JSON
OUTPUT_JSON_LS = (
    PROJECT_PATH
    / "data/interim/bootstrapping/003/labels_by_boostrapping_models_two_labelstudio.json"
)

# =========================
# Thresholds / knobs
# =========================
# per-class thresholds (ปรับได้ภายหลัง)
CLASS_TH = {
    "PSML": 0,
    "FAL": 0,
    "CP": 0,
    "DB": 0,
    "TAS": 0.7,
    "HW": 0,
}
DESIRED_LABELS = set(CLASS_TH.keys())

# Label Studio field names (must match your project's labeling config)
LS_FROM_NAME = "label"
LS_TO_NAME = "text"

# Post-process settings
STOP_CHUNKS = {"and", "on", "for"}
PUNCT_STRIP = ",.;:·•、。()[]{}<>“”\"'`"


# =========================
# Overlap helpers
# =========================
def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def pick_longest_highest(spans):
    """
    เมื่อทับซ้อน เลือกสแปนที่ยาวกว่า ถ้ายาวเท่ากันเลือกคะแนนสูงกว่า
    spans: list[dict{text,label,score,start,end}]
    """
    spans = sorted(
        spans, key=lambda x: (x["start"], -(x["end"] - x["start"]), -x["score"])
    )
    kept = []
    for s in spans:
        conflict = False
        for k in kept:
            if (
                overlaps(k["start"], k["end"], s["start"], s["end"])
                and k["label"] == s["label"]
            ):
                conflict = True
                # ถ้า s ยาวกว่า หรือเท่ายาวแต่คะแนนสูงกว่า ให้แทนที่
                if (s["end"] - s["start"], s["score"]) > (
                    k["end"] - k["start"],
                    k["score"],
                ):
                    k.update(s)
                break
        if not conflict:
            kept.append(s)
    return kept


# =========================
# Span cleaning: keep full words and full hyphenated compounds
# =========================
SHORT_VALID_1CHAR = {"R", "C"}  # allow known 1-char skills if needed


def _is_word_char(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def is_valid_full_span(text: str, start: int, end: int) -> bool:
    if start < 0 or end > len(text) or start >= end:
        return False
    span = text[start:end]
    core = span.strip()
    if not core:
        return False
    if len(core) == 1 and core not in SHORT_VALID_1CHAR:
        return False
    hyphens = set("-‐-‒–—―")
    if core[0] in hyphens or core[-1] in hyphens:
        return False
    if start > 0 and text[start - 1] in hyphens and core[0].isalnum():
        return False
    if end < len(text) and text[end] in hyphens and core[-1].isalnum():
        return False
    left_ok = (start == 0) or (not _is_word_char(text[start - 1]))
    right_ok = (end == len(text)) or (not _is_word_char(text[end]))
    return left_ok and right_ok


def clean_text_span(text: str, start: int, end: int):
    """ตัดวรรคตอนหัว-ท้าย และคืนสแปนใหม่"""
    s, e = start, end
    while s < e and text[s] in PUNCT_STRIP + " ":
        s += 1
    while e > s and text[e - 1] in PUNCT_STRIP + " ":
        e -= 1
    return s, e


# =========================
# Group function (BIO -> entities)
# =========================
def group_ner_entities(ner_results):
    grouped = []
    cur_tokens, cur_scores, cur_label = [], [], ""
    cur_start, cur_end = None, None
    for ent in ner_results:
        label = ent["entity"]  # e.g., B-PSML, I-PSML
        if label.startswith("B-"):
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
            cur_tokens = [ent["word"]]
            cur_scores = [ent["score"]]
            cur_label = label.split("-")[1]
            cur_start, cur_end = ent["start"], ent["end"]
        elif label.startswith("I-") and cur_label == label.split("-")[1]:
            cur_tokens.append(ent["word"])
            cur_scores.append(ent["score"])
            cur_end = ent["end"]
        else:
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


# =========================
# Label Studio helpers
# =========================
def make_ls_result(start: int, end: int, text: str, label: str, score=None):
    return {
        "id": str(uuid.uuid4()),
        "from_name": LS_FROM_NAME,
        "to_name": LS_TO_NAME,
        "type": "labels",
        "value": {
            "start": int(start),
            "end": int(end),
            "text": text,
            "labels": [label],
        },
        **({"score": float(score)} if score is not None else {}),
    }


def to_labelstudio_items(records):
    ls_items = []
    for rec in records:
        text = rec["original_text"]
        results = [
            make_ls_result(
                e["start"],
                e["end"],
                text[e["start"] : e["end"]],
                e["label"],
                e.get("score"),
            )
            for e in rec["ner_labels"]
        ]
        ls_items.append(
            {
                "id": str(uuid.uuid4()),
                "data": {"text": text},
                "predictions": [{"id": str(uuid.uuid4()), "result": results}],
            }
        )
    return ls_items


# =========================
# Main
# =========================
if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit(f"Error: model path not found: '{MODEL_PATH}'")

    # Load data
    df = pd.read_csv(DATA_PATH).fillna({"Segmented_Qualification": ""})
    # df = df.iloc[:1000]

    texts_to_process = [
        str(t).replace("\n", " ").replace("(", "").replace(")", "")
        for t in df["Segmented_Qualification"]
    ]

    # Load model
    tokenizer_main = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_main = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    pipe_main = pipeline(
        "ner",  # ใช้ BIO token-level, จะ group เอง
        model=model_main,
        tokenizer=tokenizer_main,
        device=0 if torch.cuda.is_available() else -1,
    )

    print("Running NER...")
    all_ner_results = pipe_main(texts_to_process, batch_size=8)
    print("NER complete. Processing...")

    raw_records = []
    cls_counts = defaultdict(int)
    kept_counts = defaultdict(int)

    for original_text, raw_results in tqdm(
        zip(texts_to_process, all_ner_results),
        total=len(texts_to_process),
        desc="Merging + cleaning + resolving overlaps",
    ):
        # 1) group BIO
        grouped = group_ner_entities(raw_results)

        # 2) per-class threshold filter
        filtered = []
        for ent in grouped:
            lab = ent["label"]
            if lab in DESIRED_LABELS:
                cls_counts[lab] += 1
                th = CLASS_TH.get(lab, 0.70)
                if ent["score"] >= th:
                    s, e = int(ent["start"]), int(ent["end"])
                    # trim punctuation
                    s2, e2 = clean_text_span(original_text, s, e)
                    if s2 >= e2:
                        continue
                    # drop stop-chunks
                    seg = original_text[s2:e2].strip()
                    if seg.lower() in STOP_CHUNKS:
                        continue
                    # keep only valid full spans
                    if not is_valid_full_span(original_text, s2, e2):
                        continue
                    filtered.append(
                        {
                            "text": original_text[s2:e2],
                            "label": lab,
                            "score": float(ent["score"]),
                            "start": s2,
                            "end": e2,
                        }
                    )

        # 3) resolve overlaps: longest span, per label
        filtered.sort(key=lambda x: (x["label"], x["start"], -x["end"], -x["score"]))
        final_labels = []
        for lab in DESIRED_LABELS:
            lab_spans = [x for x in filtered if x["label"] == lab]
            final_labels += pick_longest_highest(lab_spans)

        for x in final_labels:
            kept_counts[x["label"]] += 1

        if final_labels:
            raw_records.append(
                {"original_text": original_text, "ner_labels": final_labels}
            )

    # --------- stats ----------
    total_docs = len(texts_to_process)
    kept_docs = len(raw_records)
    print("\n=== Stats ===")
    print(
        f"Docs with >=1 entity: {kept_docs}/{total_docs} ({kept_docs/total_docs*100:.1f}%)"
    )
    for k in sorted(DESIRED_LABELS):
        print(f"{k}: detected={cls_counts[k]} kept={kept_counts[k]}")

    # Save RAW JSON
    OUTPUT_JSON_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_RAW, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved RAW to: {OUTPUT_JSON_RAW}")

    # Convert to Label Studio format & save
    ls_items = to_labelstudio_items(raw_records)
    with open(OUTPUT_JSON_LS, "w", encoding="utf-8") as f:
        json.dump(ls_items, f, ensure_ascii=False)
    print(f"Saved Label Studio JSON to: {OUTPUT_JSON_LS}")

    print("Done.")
