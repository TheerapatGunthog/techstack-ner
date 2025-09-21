# visualize_packing_data.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------- Config --------
DATA_DIR = Path(os.getcwd() + "/data/post_processed")
JOBS_JSON = DATA_DIR / "jobs.json"
SKILLS_JSON = DATA_DIR / "skills.json"  # fields: id, name, group, quantity
LINKS_JSON = DATA_DIR / "job_skill_scores.json"  # fields: id, job_id, skill_id, score

GROUPS = ["PSML", "TAS", "FAL", "DB", "CP", "ET"]
TOP_SKILLS = 20
TOP_JOBS = 20


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved: {path}")


# -------- Load --------
jobs = json.loads(JOBS_JSON.read_text(encoding="utf-8"))["items"]
skills = json.loads(SKILLS_JSON.read_text(encoding="utf-8"))["items"]
links = json.loads(LINKS_JSON.read_text(encoding="utf-8"))["items"]

jobs_df = pd.DataFrame(jobs)
skills_df = pd.DataFrame(skills)
links_df = pd.DataFrame(links)

# schema-safe rename
jobs_df = jobs_df.rename(columns={"id": "job_id", "name": "job"})
skills_df = skills_df.rename(columns={"id": "skill_id", "name": "skill"})
for col in ["job_id", "job"]:
    if col not in jobs_df.columns:
        raise KeyError(f"jobs.json missing column: {col}")
for col in ["skill_id", "skill", "group"]:
    if col not in skills_df.columns:
        raise KeyError(f"skills.json missing column: {col}")
for col in ["job_id", "skill_id", "score"]:
    if col not in links_df.columns:
        raise KeyError(f"links.json missing column: {col}")

# --- Merge
df = links_df.merge(jobs_df[["job_id", "job"]], on="job_id", how="left").merge(
    skills_df[["skill_id", "skill", "group"]], on="skill_id", how="left"
)

# -------- Base matrix (job x skill)
mat = df.pivot_table(
    index="job", columns="skill", values="score", aggfunc="sum", fill_value=0
)


# -------- Heatmap helper
def plot_heatmap(pv: pd.DataFrame, title: str, save_path: Path):
    if pv.empty:
        print(f"[skip] empty matrix for {title}")
        return
    fig_w = max(10, pv.shape[1] * 0.45)
    fig_h = max(6, pv.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pv.values, aspect="auto")
    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels(pv.columns, rotation=90)
    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels(pv.index)
    ax.set_xlabel("Skill")
    ax.set_ylabel("Job")
    ax.set_title(title)
    if pv.size <= 900:
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = int(pv.iat[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    _savefig(fig, save_path)
    plt.close(fig)


# -------- Per-group heatmaps (เหมือนเดิม)
for g in GROUPS:
    sel_skills = skills_df.loc[skills_df["group"] == g, "skill"].unique().tolist()
    if not sel_skills:
        print(f"[skip] group={g}: no skills")
        continue
    sub_mat = mat.loc[:, mat.columns.isin(sel_skills)]
    if sub_mat.shape[1] == 0:
        print(f"[skip] group={g}: matrix empty")
        continue

    skill_tot = sub_mat.sum(axis=0).sort_values(ascending=False)
    sel_cols = skill_tot.index[: min(TOP_SKILLS, len(skill_tot))]

    job_tot = sub_mat[sel_cols].sum(axis=1).sort_values(ascending=False)
    sel_rows = job_tot.index[: min(TOP_JOBS, len(job_tot))]

    pv = sub_mat.loc[sel_rows, sel_cols]
    title = (
        f"Job–Skill Scores | Group={g} | Jobs={len(sel_rows)} Skills={len(sel_cols)}"
    )
    out_png = (
        DATA_DIR / f"job_skill_heatmap_group_{g}_top{len(sel_rows)}x{len(sel_cols)}.png"
    )
    plot_heatmap(pv, title, out_png)

# -------- เพิ่มเติม: Top Jobs by total score
job_total_score = (
    df.groupby("job")["score"].sum().sort_values(ascending=False).head(TOP_JOBS)
)
fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(job_total_score))))
job_total_score[::-1].plot(kind="barh", ax=ax)
ax.set_xlabel("Total score")
ax.set_ylabel("Job")
ax.set_title("Top Jobs by total score")
_savefig(fig, DATA_DIR / f"bar_top_jobs_score_top{len(job_total_score)}.png")
plt.close(fig)

# -------- เพิ่มเติม: Top Skills by total score
skill_total_score = df.groupby(["skill", "group"])["score"].sum().reset_index()
skill_total_score = skill_total_score.sort_values("score", ascending=False).head(
    TOP_SKILLS
)
fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(skill_total_score))))
ax.barh(
    [
        f"{s} ({g})"
        for s, g in zip(skill_total_score["skill"], skill_total_score["group"])
    ],
    skill_total_score["score"],
)
ax.invert_yaxis()
ax.set_xlabel("Total score")
ax.set_ylabel("Skill (group)")
ax.set_title("Top Skills by total score")
_savefig(fig, DATA_DIR / f"bar_top_skills_score_top{len(skill_total_score)}.png")
plt.close(fig)

# -------- เพิ่มเติม: Skill group distribution (โดยจำนวนสกิลที่พบ)
group_counts = skills_df["group"].value_counts().reindex(GROUPS, fill_value=0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(group_counts.index, group_counts.values)
ax.set_ylabel("# skills")
ax.set_title("Skill group distribution")
_savefig(fig, DATA_DIR / "bar_group_distribution.png")
plt.close(fig)

# -------- เพิ่มเติม: Top Jobs by Quantity (ถ้า jobs.json มี quantity)
if "quantity" in jobs_df.columns:
    q = jobs_df[["job", "quantity"]].copy()
    q["quantity"] = pd.to_numeric(q["quantity"], errors="coerce").fillna(0).astype(int)
    q_top = q.sort_values("quantity", ascending=False).head(TOP_JOBS)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(q_top))))
    ax.barh(q_top["job"][::-1], q_top["quantity"][::-1])
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Job")
    ax.set_title("Top Jobs by Quantity")
    _savefig(fig, DATA_DIR / f"bar_top_jobs_quantity_top{len(q_top)}.png")
    plt.close(fig)
else:
    print("[note] jobs.json has no 'quantity' field; skip quantity chart.")
