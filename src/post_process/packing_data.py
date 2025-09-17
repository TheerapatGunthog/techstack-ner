from pathlib import Path
import json
import pandas as pd

# ===== Input/Output =====
INPUT = Path(
    "/home/whilebell/Code/techstack-ner/data/post_processed/all_predictions_llm_filtered.csv"
)
OUT_DIR = INPUT.parent
OUT_JOBS = OUT_DIR / "jobs.json"
OUT_SKILLS = OUT_DIR / "skills.json"
OUT_LINKS = OUT_DIR / "job_skill_scores.json"
OUT_TRADING = OUT_DIR / "trending.json"


# ===== utils =====
def _safe_split_csv(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [t.strip() for t in str(x).split(",") if t and str(t).strip()]


def _explode_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, sent, e_str, c_str in df[
        ["Topic_Normalized", "Sentence_Index", "Entity", "Class"]
    ].itertuples(index=False, name=None):
        e_list = _safe_split_csv(e_str)
        c_list = _safe_split_csv(c_str)
        n = min(len(e_list), len(c_list))
        for e, c in zip(e_list[:n], c_list[:n]):
            rows.append((topic, sent, e, c))
    return pd.DataFrame(
        rows, columns=["Topic_Normalized", "Sentence_Index", "Entity", "Class"]
    )


# ===== 1) Jobs =====
def build_jobs_json(df: pd.DataFrame):
    jobs = (
        df["Topic_Normalized"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    # id เรียงจาก 1..N เสมอ + group_id สุ่ม 1–10
    items = [{"id": idx + 1, "name": name} for idx, name in enumerate(jobs)]
    return {"items": items}


# ===== 2) Skills =====
def build_skills_json(per_rows: pd.DataFrame):
    cc = (
        per_rows.groupby(["Entity", "Class"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["Entity", "cnt", "Class"], ascending=[True, False, True])
        .drop_duplicates(subset=["Entity"], keep="first")
        .sort_values("Entity")
    )
    items = []
    for idx, r in enumerate(cc.itertuples(index=False), start=1):
        group = r.Class
        if group == "HW":
            group = "ET"
        items.append({"id": idx, "name": r.Entity, "group": group})
    return {"items": items}


# ===== 3) Job–Skill scores =====
def build_job_skill_scores(per_rows: pd.DataFrame, job2id: dict, skill2id: dict):
    base = per_rows.drop_duplicates(["Topic_Normalized", "Sentence_Index", "Entity"])
    agg = base.groupby(["Topic_Normalized", "Entity"]).size().reset_index(name="score")
    agg = agg[agg["Entity"].isin(skill2id.keys())]

    items = [
        {
            "id": idx + 1,  # primary key autoincrement
            "job_id": job2id.get(job),
            "skill_id": skill2id.get(ent),
            "score": int(score),
        }
        for idx, (job, ent, score) in enumerate(
            agg[["Topic_Normalized", "Entity", "score"]].itertuples(
                index=False, name=None
            )
        )
    ]
    items.sort(key=lambda x: (x["job_id"], -x["score"], x["skill_id"]))
    return {"items": items}


# ===== 4) Trading (Trending level per position) =====
def build_trending_json(df: pd.DataFrame, job2id: dict):
    # นับจำนวน record ต่ออาชีพ
    counts = df["Topic_Normalized"].value_counts(normalize=True)  # เป็นสัดส่วน %
    items = []

    for job, frac in counts.items():
        if job not in job2id:
            continue
        # แบ่ง quantile 3 ระดับ
        if frac <= counts.quantile(1 / 3):
            trending = 1
        elif frac <= counts.quantile(2 / 3):
            trending = 2
        else:
            trending = 3

        items.append({"position_id": job2id[job], "trending_level": trending})

    return {"items": items}


# ===== validate =====
def validate_links(links, jobs, skills):
    job_ids = {j["id"] for j in jobs["items"]}
    skill_ids = {s["id"] for s in skills["items"]}
    invalid = [
        l
        for l in links["items"]
        if l["job_id"] not in job_ids or l["skill_id"] not in skill_ids
    ]
    print(f"Unmapped job-skill links: {len(invalid)}")
    if invalid:
        print("ตัวอย่าง:", invalid[:10])


# ===== main =====
if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    for col in ["Topic_Normalized", "Sentence_Index", "Entity", "Class"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    per = _explode_pairs(df)

    # jobs
    jobs_json = build_jobs_json(df)
    job2id = {x["name"]: x["id"] for x in jobs_json["items"]}
    OUT_JOBS.write_text(
        json.dumps(jobs_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # skills
    skills_json = build_skills_json(per)
    skill2id = {x["name"]: x["id"] for x in skills_json["items"]}
    OUT_SKILLS.write_text(
        json.dumps(skills_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # job-skill scores
    links_json = build_job_skill_scores(per, job2id, skill2id)
    OUT_LINKS.write_text(
        json.dumps(links_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # job-skill scores
    links_json = build_job_skill_scores(per, job2id, skill2id)
    validate_links(links_json, jobs_json, skills_json)  # ตรวจสอบ mapping
    OUT_LINKS.write_text(
        json.dumps(links_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # trading
    trending_json = build_trending_json(df, job2id)
    OUT_TRADING.write_text(
        json.dumps(trending_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
