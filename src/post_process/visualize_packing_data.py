# visualize_packing_data.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Config --------
DATA_DIR = Path("/home/whilebell/Code/techstack-ner/data/post_processed")
JOBS_JSON = DATA_DIR / "jobs.json"
SKILLS_JSON = DATA_DIR / "skills.json"  # ต้องมีฟิลด์: skill_id, name, group
LINKS_JSON = DATA_DIR / "job_skill_scores.json"

GROUPS = ["PSML", "TAS", "FAL", "DB", "CP", "HW"]
TOP_SKILLS = 20
TOP_JOBS = 20

# -------- Load --------
jobs = json.loads(JOBS_JSON.read_text(encoding="utf-8"))["items"]
skills = json.loads(SKILLS_JSON.read_text(encoding="utf-8"))["items"]
links = json.loads(LINKS_JSON.read_text(encoding="utf-8"))["items"]

jobs_df = pd.DataFrame(jobs)[["job_id", "name"]].rename(columns={"name": "job"})
skills_df = pd.DataFrame(skills)[["skill_id", "name", "group"]].rename(
    columns={"name": "skill"}
)
links_df = pd.DataFrame(links)

# --- Merge แบบ A ---
df = links_df.merge(jobs_df, on="job_id", how="left").merge(
    skills_df, on="skill_id", how="left"
)
# df columns: job_id, skill_id, score, job, skill, group


# -------- Helper: heatmap --------
def plot_heatmap(pv: pd.DataFrame, title: str, save_path: Path):
    fig_w = max(10, pv.shape[1] * 0.45)
    fig_h = max(6, pv.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pv.values, aspect="auto")  # no custom colors
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
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()


# -------- Build base matrix once --------
mat = df.pivot_table(
    index="job", columns="skill", values="score", aggfunc="sum", fill_value=0
)

# -------- Per group plots --------
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
