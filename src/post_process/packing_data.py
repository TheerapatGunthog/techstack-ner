from pathlib import Path
import json
import pandas as pd
import os

# ===== Input/Output =====
INPUT = Path(os.getcwd() + "/data/post_processed/all_predictions_llm_filtered_grouped.csv")
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


def _explode_pairs_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    รับไฟล์ grouped: มีคอลัมน์ Topic_Normalized, Quantity, Entity, Class
    แตก Entity,Class ต่อแถวเป็นคู่ ๆ เพื่อใช้ทำสถิติและ scoring
    """
    rows = []
    for topic, e_str, c_str in df[["Topic_Normalized", "Entity", "Class"]].itertuples(
        index=False, name=None
    ):
        e_list = _safe_split_csv(e_str)
        c_list = _safe_split_csv(c_str)
        n = min(len(e_list), len(c_list))
        for e, c in zip(e_list[:n], c_list[:n]):
            rows.append((topic, e, c))
    return pd.DataFrame(rows, columns=["Topic_Normalized", "Entity", "Class"])


# ===== 1) Jobs =====
def build_jobs_json(df_grouped: pd.DataFrame):
    """
    ใช้คอลัมน์ Quantity จากไฟล์ grouped โดยตรง
    """
    dfj = (
        df_grouped[["Topic_Normalized", "Quantity"]]
        .dropna(subset=["Topic_Normalized"])
        .copy()
    )
    dfj["Topic_Normalized"] = dfj["Topic_Normalized"].astype(str).str.strip()
    dfj = dfj.loc[dfj["Topic_Normalized"] != ""].drop_duplicates(subset=["Topic_Normalized"])

    # สร้าง id ตามลำดับชื่อ
    dfj = dfj.sort_values("Topic_Normalized").reset_index(drop=True)
    items = [
        {"id": idx + 1, "name": row.Topic_Normalized, "quantity": int(row.Quantity)}
        for idx, row in dfj.itertuples(index=False, name="Row")
    ]
    return {"items": items}


# ===== 2) Skills =====
def build_skills_json(per_rows: pd.DataFrame):
    """
    คัดคลาสหลักของสกิลจากความถี่สูงสุดเหมือนเดิม
    และใส่ quantity = จำนวนครั้งที่สกิลนั้นปรากฏ (หลัง explode)
    """
    # นับจำนวน (Entity, Class)
    cc = (
        per_rows.groupby(["Entity", "Class"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["Entity", "cnt", "Class"], ascending=[True, False, True])
    )
    # เลือกคลาสหลักต่อสกิลตาม cnt มากสุด
    cc_top = cc.drop_duplicates(subset=["Entity"], keep="first").sort_values("Entity")

    # quantity ต่อสกิล = ผลรวมทุกคลาสของสกิลนั้น
    qty_all = (
        per_rows.groupby("Entity").size().rename("quantity").reset_index()
    )  # Entity, quantity
    merged = cc_top.merge(qty_all, on="Entity", how="left")

    items = []
    for idx, r in enumerate(merged.itertuples(index=False), start=1):
        group = r.Class
        if group == "HW":
            group = "ET"  # mapping เดิม
        items.append({"id": idx, "name": r.Entity, "group": group, "quantity": int(r.quantity)})
    return {"items": items}


# ===== 3) Job–Skill scores =====
def build_job_skill_scores(per_rows: pd.DataFrame, job2id: dict, skill2id: dict):
    """
    คะแนน = จำนวนครั้งที่สกิลนั้นปรากฏภายใต้ Topic_Normalized นั้น ๆ (นับซ้ำได้)
    """
    agg = (
        per_rows.groupby(["Topic_Normalized", "Entity"])
        .size()
        .reset_index(name="score")
    )
    agg = agg[agg["Entity"].isin(skill2id.keys())]

    items = [
        {
            "id": idx + 1,
            "job_id": job2id.get(job),
            "skill_id": skill2id.get(ent),
            "score": int(score),
        }
        for idx, (job, ent, score) in enumerate(
            agg[["Topic_Normalized", "Entity", "score"]].itertuples(index=False, name=None)
        )
        if job in job2id  # กันกรณี job ไม่อยู่ใน mapping
    ]
    items.sort(key=lambda x: (x["job_id"], -x["score"], x["skill_id"]))
    return {"items": items}


# ===== 4) Trending (Trending level per position) =====
def build_trending_json(df_grouped: pd.DataFrame, job2id: dict):
    """
    ใช้สัดส่วน Quantity ต่อผลรวมทั้งหมดเพื่อจัดระดับแนวโน้ม
    """
    if "Quantity" not in df_grouped.columns:
        # fallback: ให้ทุกตัวเท่ากัน
        df_grouped = df_grouped.copy()
        df_grouped["Quantity"] = 1

    total_qty = df_grouped["Quantity"].sum()
    if total_qty <= 0:
        return {"items": []}

    dfq = df_grouped[["Topic_Normalized", "Quantity"]].copy()
    dfq["frac"] = dfq["Quantity"] / total_qty

    q1 = dfq["frac"].quantile(1 / 3)
    q2 = dfq["frac"].quantile(2 / 3)

    items = []
    for job, frac in dfq[["Topic_Normalized", "frac"]].itertuples(index=False, name=None):
        if job not in job2id:
            continue
        trending = 1 if frac <= q1 else (2 if frac <= q2 else 3)
        items.append({"position_id": job2id[job], "trending_level": trending})
    return {"items": items}


# ===== validate =====
def validate_links(links, jobs, skills):
    job_ids = {j["id"] for j in jobs["items"]}
    skill_ids = {s["id"] for s in skills["items"]}
    invalid = [
        l for l in links["items"] if l["job_id"] not in job_ids or l["skill_id"] not in skill_ids
    ]
    print(f"Unmapped job-skill links: {len(invalid)}")
    if invalid:
        print("ตัวอย่าง:", invalid[:10])


# ===== main =====
if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    # columns: Topic_Normalized, Quantity, Entity, Class
    for col in ["Topic_Normalized", "Entity", "Class", "Quantity"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # explode เป็น per-entity-row (ไม่มี Sentence_Index ในไฟล์ grouped)
    per = _explode_pairs_grouped(df)

    # jobs (มี quantity)
    jobs_json = build_jobs_json(df)
    job2id = {x["name"]: x["id"] for x in jobs_json["items"]}
    OUT_JOBS.write_text(json.dumps(jobs_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # skills (มี quantity ของสกิล)
    skills_json = build_skills_json(per)
    skill2id = {x["name"]: x["id"] for x in skills_json["items"]}
    OUT_SKILLS.write_text(json.dumps(skills_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # job-skill scores
    links_json = build_job_skill_scores(per, job2id, skill2id)
    validate_links(links_json, jobs_json, skills_json)
    OUT_LINKS.write_text(json.dumps(links_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # trending จากสัดส่วน Quantity
    trending_json = build_trending_json(df, job2id)
    OUT_TRADING.write_text(json.dumps(trending_json, ensure_ascii=False, indent=2), encoding="utf-8")
