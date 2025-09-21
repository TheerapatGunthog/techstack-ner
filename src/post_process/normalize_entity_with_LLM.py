# normalize_and_filter_entities_llm.py
from pathlib import Path
import re
import requests
import pandas as pd
from tqdm import tqdm
import os

# ================= Config =================
INPUT = Path(
    os.getcwd() + "/data/post_processed/all_predictions_dedup.csv"
)
OUT_DIR = INPUT.parent
OUT_FILE = OUT_DIR / "all_predictions_llm_filtered.csv"
OUT_FILE_GROUPED = OUT_DIR / "all_predictions_llm_filtered_grouped.csv"

LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1:latest"
BATCH_SIZE = 20

# ============== Lexicon (จากเดิม) ==============
TOOLS_PATTERNS = {
    "Python": [r"\bpy(?:thon)?\b", r"\bphyton\b"],
    "C++": [r"\bc\+\+\b", r"\bc\s*plus\s*plus\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "Java": [r"\bjava\b"],
    "JavaScript": [r"\bjava\s*script\b", r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "Node.js": [r"\bnode\.?js\b", r"\bnode\s*js\b"],
    "React": [r"\breact(\.js)?\b"],
    "Angular": [r"\bangular(\.js)?\b"],
    "Vue.js": [r"\bvue(\.js)?\b"],
    "SQL": [r"\bsql\b"],
    "PostgreSQL": [r"\bpostgre\s*sql\b", r"\bpostgres\b"],
    "MySQL": [r"\bmysql\b"],
    "SQLite": [r"\bsqlite\b"],
    "SQL Server": [r"\bsql\s*server\b", r"\bms\s*sql\b", r"\bmssql\b"],
    "Oracle": [r"\boracle\b"],
    "MongoDB": [r"\bmonogo?db\b", r"\bmongo\s*db\b", r"\bmongodb\b"],
    "Redis": [r"\bredis\b"],
    "Elasticsearch": [r"\belastic\s*search\b", r"\belasticsearch\b"],
    "Kafka": [r"\bkafka\b"],
    "Spark": [r"\bspark\b"],
    "Hadoop": [r"\bhadoop\b"],
    "Airflow": [r"\bair\s*flow\b", r"\bairflow\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Jenkins": [r"\bjenkins\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b", r"\bbitbucket\b"],
    "Linux": [r"\blinux\b"],
    "Bash": [r"\bbash\b", r"\bshell\b"],
    "PowerShell": [r"\bpowershell\b"],
    "Excel": [r"\bms\s*excel\b", r"\bmicrosoft\s*excel\b", r"\bexcel\b"],
    "Word": [r"\bms\s*word\b", r"\bmicrosoft\s*word\b", r"\bword\b"],
    "PowerPoint": [r"\bpower\s*point\b", r"\bpowerpoint\b"],
    "Power BI": [r"\bpower\s*bi\b"],
    "Tableau": [r"\btableau\b"],
    "Looker": [r"\blooker\b"],
    "pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "SciPy": [r"\bscipy\b"],
    "scikit-learn": [r"\bscikit[- ]?learn\b", r"\bsklearn\b"],
    "XGBoost": [r"\bxgboost\b"],
    "LightGBM": [r"\blightgbm\b", r"\blgbm\b"],
    "CatBoost": [r"\bcatboost\b"],
    "TensorFlow": [r"\btensor\s*flow\b", r"\btensorflow\b"],
    "Keras": [r"\bkeras\b"],
    "PyTorch": [r"\bpytorch\b", r"\btorch\b"],
    "OpenCV": [r"\bopencv\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Flask": [r"\bflask\b"],
    "Django": [r"\bdjango\b"],
    ".NET": [r"\b\.net\b", r"\bdotnet\b"],
    "Spring": [r"\bspring\b"],
    "MATLAB": [r"\bmatlab\b"],
    "Simulink": [r"\bsimulink\b"],
    "AutoCAD": [r"\bautocad\b"],
    "SolidWorks": [r"\bsolidworks\b"],
    "Ansible": [r"\bansible\b"],
    "Terraform": [r"\bterraform\b"],
    "OpenShift": [r"\bopenshift\b"],
    "Databricks": [r"\bdatabricks\b"],
    "Snowflake": [r"\bsnowflake\b"],
    "BigQuery": [r"\bbig\s*query\b", r"\bbigquery\b"],
    "Redshift": [r"\bredshift\b"],
    "Hive": [r"\bhive\b"],
    "Trino": [r"\btrino\b", r"\bpresto\b"],
    "Grafana": [r"\bgrafana\b"],
    "Prometheus": [r"\bprometheus\b"],
    "Jira": [r"\bjira\b"],
    "Confluence": [r"\bconfluence\b"],
    "AWS": [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle\s+cloud\b"],
    "Azure": [r"\bazure\b"],
}
SOFT_PATTERNS = {
    "Communication": [
        r"\bcommunication(s)?\b",
        r"\bcommunicate\b",
        r"\bpresentation(s)?\b",
    ],
    "Teamwork": [
        r"\bteam\s*work\b",
        r"\bteam\s*player\b",
        r"\bcollaborat(e|ion|ive)\b",
    ],
    "Leadership": [r"\blead(ership)?\b", r"\bmentor(ing)?\b"],
    "Problem Solving": [r"\bproblem[- ]?solv(ing|er)\b"],
    "Critical Thinking": [r"\bcritical\s*thinking\b"],
    "Time Management": [r"\btime\s*management\b"],
    "Adaptability": [r"\badaptabilit(y|ies)\b", r"\badaptable\b", r"\badapt\b"],
    "Creativity": [r"\bcreativ(e|ity)\b", r"\binnovation\b"],
    "Attention to Detail": [r"\battention\s*to\s*detail\b", r"\bdetail[- ]?oriented\b"],
    "Stakeholder Management": [r"\bstakeholder\s*management\b"],
    "Project Management": [r"\bproject\s*management\b"],
    "Decision Making": [r"\bdecision\s*making\b"],
    "Negotiation": [r"\bnegotiation(s)?\b"],
    "Collaboration": [r"\bcollaboration\b"],
    "Presentation": [r"\bpresentation(s)?\b"],
}
EXACT_MAP = {
    "ms excel": "Excel",
    "microsoft excel": "Excel",
    "excel": "Excel",
    "ms word": "Word",
    "microsoft word": "Word",
    "word": "Word",
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "google sheets": "Google Sheets",
    "google sheet": "Google Sheets",
    "g sheets": "Google Sheets",
    "g sheet": "Google Sheets",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "js": "JavaScript",
}


def compile_dict(patterns: dict):
    return {
        canon: [re.compile(p, re.I) for p in plist] for canon, plist in patterns.items()
    }


TOOLS_RX = compile_dict(TOOLS_PATTERNS)
SOFT_RX = compile_dict(SOFT_PATTERNS)

# ============== Normalize helpers ==============


def base_clean(s: str) -> str:
    return (s or "").strip()


def low_key(s: str) -> str:
    return s.lower().replace("-", " ").replace("_", " ").strip()


def exact_canon(s: str):
    return EXACT_MAP.get(low_key(s))


def normalize_token_lexicon(ent: str):
    """คืนชื่อ canonical ถ้าเจอใน lexicon, ถ้าไม่เจอคืน None"""
    s = base_clean(ent)
    if not s:
        return None
    c = exact_canon(s)
    if c:
        return c
    for canon, plist in TOOLS_RX.items():
        if any(p.search(s) for p in plist):
            return canon
    for canon, plist in SOFT_RX.items():
        if any(p.search(s) for p in plist):
            return canon
    return None


def _safe_split_csv(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [t.strip() for t in str(x).split(",") if t and str(t).strip()]

# ============== LLM helpers ==============


def query_llm_batch(pairs):
    text = "\n".join(
        [
            f"{i+1}. Entity: '{e}' | Proposed class: '{c}'"
            for i, (e, c) in enumerate(pairs)
        ]
    )
    prompt = f"""
You are an assistant that validates whether an entity truly belongs to the proposed class.
Output strictly "yes" or "no" only. One answer per line, same order as input. No explanations.

Preprocess input entity: lowercase for matching; strip quotes, versions, brackets (e.g., "React 18", "PostgreSQL (v14)").

Classes:
- PSML = Programming and scripting language (Python, JavaScript, C++, R, Java, SQL, etc.)
- DB   = Database systems (MySQL, PostgreSQL, MongoDB, Oracle, BigQuery, DynamoDB, etc.)
- CP   = Cloud platform providers (AWS, Google Cloud, Microsoft Azure, Alibaba Cloud, etc.)
- FAL  = Frameworks/libraries/tools (React, Angular, Django, Flask, TensorFlow, PyTorch, Docker, Kubernetes, etc.)
- TAS  = Soft skills & techniques/methodologies (Communication, Teamwork, Leadership, Problem Solving, Agile, Scrum, etc.)
- HW   = Hardware devices/components (NVIDIA GPU, Raspberry Pi, Arduino, Intel CPU, etc.)

Disambiguation rules:
- Language vs library: languages → PSML; packages/frameworks → FAL.
- Cloud provider vs service: provider/platform → CP; database services (e.g., BigQuery, DynamoDB) → DB.
- Tools vs soft skills: software/tools → FAL; skills/methods → TAS.
- Hardware = physical device/component only.
If uncertain, answer "no".

### Examples
# PSML
Entity: 'Python' | Proposed class: 'PSML' -> yes
Entity: 'JavaScript' | Proposed class: 'DB' -> no
# DB
Entity: 'PostgreSQL' | Proposed class: 'DB' -> yes
Entity: 'BigQuery' | Proposed class: 'CP' -> no
# CP
Entity: 'AWS' | Proposed class: 'CP' -> yes
Entity: 'Azure' | Proposed class: 'PSML' -> no
# FAL
Entity: 'React' | Proposed class: 'FAL' -> yes
Entity: 'Docker' | Proposed class: 'CP' -> no
# TAS
Entity: 'Agile' | Proposed class: 'TAS' -> yes
Entity: 'Scrum' | Proposed class: 'FAL' -> no
# HW
Entity: 'Raspberry Pi' | Proposed class: 'HW' -> yes
Entity: 'NVIDIA GPU' | Proposed class: 'PSML' -> no

### Now validate the following
{text}
"""
    resp = requests.post(
        LLM_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
    )
    data = resp.json()
    if "response" not in data:
        raise RuntimeError(f"LLM error: {data}")
    raw = data["response"]
    lines = [l.strip().lower() for l in raw.splitlines() if l.strip()]
    result = []
    for i in range(len(pairs)):
        if i < len(lines) and lines[i].startswith("y"):
            result.append(True)
        else:
            result.append(False)
    return result


def filter_with_llm(df):
    # ดึง unique pairs ที่ไม่ match lexicon
    unique_pairs = (
        df[["Entity", "Class"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    pairs_to_check = []
    for ent, cls in unique_pairs:
        if not normalize_token_lexicon(ent):
            pairs_to_check.append((ent, cls))

    # query LLM แบบ batch พร้อม tqdm
    keep = {}
    for i in tqdm(range(0, len(pairs_to_check), BATCH_SIZE), desc="LLM validation"):
        batch = pairs_to_check[i : i + BATCH_SIZE]
        decisions = query_llm_batch(batch)
        for (ent, cls), ok in zip(batch, decisions):
            keep[(ent, cls)] = ok

    # apply filter row by row พร้อม tqdm
    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering rows"):
        ents = _safe_split_csv(row["Entity"])
        clss = _safe_split_csv(row["Class"])
        kept_e, kept_c = [], []
        for e, c in zip(ents, clss):
            canon = normalize_token_lexicon(e)
            if canon:  # lexicon ผ่านแน่นอน
                kept_e.append(canon)
                kept_c.append(c)
            else:
                if keep.get((e, c), False):
                    kept_e.append(e)
                    kept_c.append(c)
        if kept_e:
            out_rows.append(
                {
                    "Topic_Normalized": row["Topic_Normalized"],
                    "Sentence_Index": row["Sentence_Index"],
                    "Entity": ", ".join(kept_e),
                    "Class": ", ".join(kept_c),
                }
            )
    return pd.DataFrame(out_rows)

# ============== Final normalize & filter ==============


def _concat_nonempty(series: pd.Series) -> str:
    return ", ".join([s for s in series if isinstance(s, str) and s.strip()])


def _count_distinct_classes(cls_str: str) -> int:
    if not isinstance(cls_str, str) or not cls_str.strip():
        return 0
    return len({c.strip() for c in cls_str.split(",") if c.strip()})


def finalize_group_and_filter(df_filt: pd.DataFrame) -> pd.DataFrame:
    """
    1) รวมทุกแถวที่มี Topic_Normalized เดียวกันให้เหลือแถวเดียว
       โดยต่อ 'Entity' และ 'Class' เข้าด้วยกันแบบสะสม
    2) คัดทิ้งอาชีพที่มีจำนวน class ที่แตกต่างกัน < 3
    """
    grouped = (
        df_filt.groupby("Topic_Normalized", as_index=False)
        .agg({
            "Entity": _concat_nonempty,
            "Class": _concat_nonempty,
        })
    )
    mask = grouped["Class"].apply(_count_distinct_classes) >= 3
    return grouped.loc[mask].reset_index(drop=True)


# ============== Main ==============
if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    for col in ["Entity", "Class", "Topic_Normalized"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df_filt = filter_with_llm(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_filt.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[saved LLM filtered] {OUT_FILE}")
    print(f"Rows: {len(df_filt):,} | Unique entities: {df_filt['Entity'].nunique():,}")

    # ขั้นตอนสุดท้าย: รวมตาม Topic_Normalized แล้วคัดทิ้งอาชีพที่มี class < 3 แบบไม่ซ้ำ
    df_grouped = finalize_group_and_filter(df_filt)
    df_grouped.to_csv(OUT_FILE_GROUPED, index=False, encoding="utf-8-sig")
    print(f"[saved grouped+filtered] {OUT_FILE_GROUPED}")
    print(f"Topics kept: {len(df_grouped):,}")
