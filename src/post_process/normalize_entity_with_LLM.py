# normalize_and_filter_entities_llm.py
from pathlib import Path
import os
import re
import requests
import pandas as pd
from tqdm import tqdm

# ================= Config =================
INPUT = Path(os.getcwd() + "/data/post_processed/all_predictions_dedup.csv")
OUT_DIR = INPUT.parent
OUT_FILE = OUT_DIR / "all_predictions_llm_filtered.csv"
OUT_FILE_GROUPED = OUT_DIR / "all_predictions_llm_filtered_grouped.csv"

LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1:latest"
BATCH_SIZE = 20

# ============== Lexicon: canonicalize ชื่อ (ไม่ตัดสินคลาส) ==============
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
    "Communication": [r"\bcommunication(s)?\b", r"\bcommunicate\b", r"\bpresentation(s)?\b"],
    "Teamwork": [r"\bteam\s*work\b", r"\bteam\s*player\b", r"\bcollaborat(e|ion|ive)\b"],
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
    "ms excel": "Excel",
    "microsoft excel": "Excel",
    "excel": "Excel",
    "ms word": "Word",
    "microsoft word": "Word",
    "word": "Word",
}


def compile_dict(patterns: dict):
    return {k: [re.compile(p, re.I) for p in v] for k, v in patterns.items()}


TOOLS_RX = compile_dict(TOOLS_PATTERNS)
SOFT_RX = compile_dict(SOFT_PATTERNS)

# ============== Helpers ==============


def base_clean(s: str) -> str:
    return (s or "").strip()


def low_key(s: str) -> str:
    return s.lower().replace("-", " ").replace("_", " ").strip()


def exact_canon(s: str):
    return EXACT_MAP.get(low_key(s))


def canonical_name(s: str) -> str:
    """แปลงชื่อให้เป็น canonical ด้วย EXACT_MAP และ regex lexicon"""
    s0 = base_clean(s)
    if not s0:
        return ""
    c = exact_canon(s0)
    if c:
        return c
    for canon, plist in TOOLS_RX.items():
        if any(p.search(s0) for p in plist):
            return canon
    for canon, plist in SOFT_RX.items():
        if any(p.search(s0) for p in plist):
            return canon
    return s0


def _safe_split_csv(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [t.strip() for t in str(x).split(",") if t and str(t).strip()]


def _ekey(s: str) -> str:
    """คีย์ dedup ต่อเอนทิตี: ตัดวงเล็บ/อัญประกาศ เว้นวรรคซ้ำ เคสไม่แคร์"""
    s = (s or "").strip()
    s = re.sub(r'[\(\)\[\]\{\}"“”]', "", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()

# ============== LLM: reclass ต่อเอนทิตี (few-shot) ==============


def _llm_reclass_entities(unique_entities):
    """
    unique_entities: list[str] (canonical แล้ว)
    return: dict[key(entity)-> label in {PSML,DB,CP,FAL,TAS,HW} or 'NO']
    """
    entity_to_class = {}
    allowed = {"PSML", "DB", "CP", "FAL", "TAS", "HW"}
    for i in tqdm(range(0, len(unique_entities), BATCH_SIZE), desc="LLM reclass (unique entities)"):
        batch = unique_entities[i : i + BATCH_SIZE]
        text = "\n".join([f"{j+1}. Entity: '{e}'" for j, e in enumerate(batch)])
        prompt = f"""
Classify each entity into EXACTLY one of: PSML, DB, CP, FAL, TAS, HW.
If none fit, output: no
One token per line. No explanations. Output lines must equal input lines.

Disambiguation rules:
- Languages → PSML; frameworks/libraries/tools → FAL.
- Cloud providers/platforms → CP; database services (BigQuery, DynamoDB, Redshift, Firestore) → DB.
- Software/tools → FAL; skills/methodologies → TAS.
- Hardware are physical devices/components → HW.

Examples
Input:
1. Entity: 'Python'
2. Entity: 'BigQuery'
3. Entity: 'Docker'
4. Entity: 'Kubernetes'
5. Entity: 'Scrum'
6. Entity: 'Raspberry Pi'
7. Entity: 'SQL'
8. Entity: 'Firebase'
9. Entity: 'DynamoDB'
10. Entity: 'UnknownTool123'

Output:
PSML
DB
FAL
FAL
TAS
HW
PSML
CP
DB
no

Now classify the following
{text}
"""
        resp = requests.post(LLM_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False})
        if not resp.ok:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        lines = [l.strip().upper() for l in data.get("response", "").splitlines() if l.strip()]
        for e, lab in zip(batch, lines):
            lab = lab if lab in allowed else "NO"
            entity_to_class[_ekey(e)] = lab
    return entity_to_class

# ============== ขั้นตอน LLM + apply ทั้ง dataset ==============


def filter_with_llm_reclass_all_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    - canonicalize entity ชื่อด้วย lexicon
    - รวมเอนทิตีทั้งหมดแบบไม่ซ้ำ (ใช้ ekey)
    - ให้ LLM ชี้คลาสครั้งเดียวต่อเอนทิตี
    - ใช้คลาสจาก LLM แทนคลาสเดิมทุกกรณี; ถ้า 'NO' → ทิ้ง
    """
    # 1) เก็บเอนทิตี canonical แบบไม่ซ้ำ
    seen = set()
    ents_to_check = []
    key_to_display = {}
    for _, row in df.iterrows():
        for e in _safe_split_csv(row["Entity"]):
            if not e:
                continue
            display = canonical_name(e)
            k = _ekey(display)
            if k and k not in seen:
                seen.add(k)
                ents_to_check.append(display)
                key_to_display[k] = display

    # 2) LLM reclass ต่อเอนทิตี
    entity_to_class = _llm_reclass_entities(ents_to_check)

    # 3) apply กลับทั้งชุดข้อมูล
    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Apply LLM class per entity"):
        ents = _safe_split_csv(row["Entity"])
        kept_e, kept_c = [], []
        for e in ents:
            display = canonical_name(e)
            k = _ekey(display)
            lab = entity_to_class.get(k, "NO")
            if lab != "NO":
                kept_e.append(display)   # ชื่อ canonical
                kept_c.append(lab)       # คลาสจาก LLM
        if kept_e:
            out_rows.append({
                "Topic_Normalized": row["Topic_Normalized"],
                "Sentence_Index": row["Sentence_Index"],
                "Entity": ", ".join(kept_e),
                "Class": ", ".join(kept_c),
            })
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
    1) groupby Topic_Normalized แล้ว concat Entity/Class ต่อท้าย
    2) คัดทิ้งหัวข้อที่มีจำนวน class ต่างกัน < 3
    """
    grouped = (
        df_filt.groupby("Topic_Normalized", as_index=False)
        .agg({"Entity": _concat_nonempty, "Class": _concat_nonempty})
    )
    mask = grouped["Class"].apply(_count_distinct_classes) >= 3
    return grouped.loc[mask].reset_index(drop=True)


# ============== Main ==============
if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    for col in ["Entity", "Class", "Topic_Normalized"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df_filt = filter_with_llm_reclass_all_entities(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_filt.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[saved LLM filtered] {OUT_FILE}")
    print(f"Rows: {len(df_filt):,} | Unique entities: {df_filt['Entity'].nunique():,}")

    df_grouped = finalize_group_and_filter(df_filt)
    df_grouped.to_csv(OUT_FILE_GROUPED, index=False, encoding="utf-8-sig")
    print(f"[saved grouped+filtered] {OUT_FILE_GROUPED}")
    print(f"Topics kept: {len(df_grouped):,}")
