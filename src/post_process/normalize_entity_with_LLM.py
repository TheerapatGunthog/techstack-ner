# normalize_entity_with_LLM_single.py
from pathlib import Path
import os
import re
import time
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
TIMEOUT = 90
MAX_RETRIES = 2
RETRY_SLEEP = 1.0

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
    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "GitLab": [r"\bgitlab\b"],
    "Bitbucket": [r"\bbitbucket\b"],
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


def exact_canon(s: str):
    s0 = base_clean(s)
    if not s0:
        return ""
    key = s0.lower().replace("-", " ").replace("_", " ").strip()
    return EXACT_MAP.get(key, "")


def canonical_name(s: str) -> str:
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
    s = (s or "").strip()
    s = re.sub(r'[\(\)\[\]\{\}"“”]', "", s)
    s = re.sub(r"\b[vV]?\d+(\.\d+){0,3}\b", "", s)  # ตัดเวอร์ชัน
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


# ============== LLM: classify ทีละเอนทิตี ==============
_ALLOWED = {"PSML", "DB", "CP", "FAL", "TAS", "HW"}


def _prompt_one(entity: str) -> str:
    return f"""
You are a STRICT validator for ONE entity.
Return EXACTLY ONE token from: PSML DB CP FAL TAS HW NO
UPPERCASE only. No prose. No punctuation.

DEFINITIONS
- PSML: programming/scripting languages only (Python, Java, Kotlin, C/C++, C#, JavaScript, TypeScript, R, Go, Rust, SQL, Bash, PowerShell, MATLAB).
- DB: database engines/warehouses/services/key-value/document/search (PostgreSQL, MySQL, SQLite, Oracle, SQL Server, MongoDB, Redis, Cassandra, BigQuery, Snowflake, Redshift, DynamoDB, Hive, Trino, Presto, Elasticsearch).
- CP: cloud PROVIDER/PLATFORM names only (AWS, Google Cloud, Microsoft Azure, Alibaba Cloud, Firebase platform).
- FAL: software frameworks/libraries/tools/runtimes/OS/SDK/CI-CD/VCS/BI/office (React, Vue.js, Angular, Django, Flask, FastAPI, Spring, Android SDK, TensorFlow, PyTorch, scikit-learn, pandas, NumPy, Node.js, .NET, Docker, Kubernetes, Git, GitHub, GitLab, Jenkins, Jira, Confluence, Linux, Windows, macOS, Excel, Word, PowerPoint, Power BI, Tableau, Looker, OpenCV, Ansible, Terraform).
- TAS: soft skills, techniques, methodologies, patterns (Agile, Scrum, Kanban, Communication, Teamwork, Leadership, Problem Solving, Time Management, Stakeholder Management, Project Management, Negotiation, MVC, MVVM, MVP, Clean Architecture).
- HW: physical devices/components (Raspberry Pi, Arduino, NVIDIA GPU, Intel CPU, FPGA, microcontroller).

HARD RULES
- Managed DB services (BigQuery, Redshift, DynamoDB, Firestore) -> DB.
- CP only for provider names; non-database cloud services that are not tools -> NO.
- Adjectives/marketing terms/generic words (e.g., scalable, robust, enterprise) -> NO.
- Job titles, companies, locations, salaries, sentences, questions, responsibilities -> NO.
- Unknown brands or proper nouns that are NOT tools/techniques -> NO.
- If the token looks like two distinct tools glued together (e.g., 'KubernetesPython', 'FrameworkNode') and is NOT a known single tool name -> NO.
- If uncertain -> NO.

QUICK EXAMPLES
- 'Android SDK' -> FAL
- 'MVVM' -> TAS
- 'Java' -> PSML
- 'Kotlin' -> PSML
- 'Django' -> FAL
- 'Vue.js' -> FAL
- 'Docker' -> FAL
- 'scalable' -> NO
- 'Conicle' -> NO
- 'KubernetesPython' -> NO
- 'FrameworkNode' -> NO

Now classify exactly one token for:
Entity: '{entity}'
Output:
""".strip()


def _llm_classify_one(entity: str, session: requests.Session) -> str:
    """เรียก LLM สำหรับเอนทิตีเดียว พร้อม retry และพาร์สผลแบบเข้ม"""
    payload = {"model": LLM_MODEL, "prompt": _prompt_one(entity), "stream": False}
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.post(LLM_URL, json=payload, timeout=TIMEOUT)
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}")
            resp = r.json().get("response", "").strip()
            # ดึงโทเคนที่อนุญาตจากข้อความบรรทัดแรกที่พบ
            m = re.search(r"\b(PSML|DB|CP|FAL|TAS|HW|NO)\b", resp, flags=re.I)
            lab = m.group(1).upper() if m else "NO"
            return lab if lab in _ALLOWED or lab == "NO" else "NO"
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * (attempt + 1))
                continue
            return "NO"


def _llm_reclass_entities_one_by_one(unique_entities):
    entity_to_class = {}
    sess = requests.Session()
    for e in tqdm(unique_entities, desc="LLM reclass (1-by-1)"):
        lab = _llm_classify_one(e, sess)
        entity_to_class[_ekey(e)] = lab
    total = len(entity_to_class)
    no_cnt = sum(1 for v in entity_to_class.values() if v == "NO")
    print(f"[LLM map] entities: {total} | NO: {no_cnt} ({no_cnt/max(1,total):.2%})")
    return entity_to_class


# ============== ขั้นตอน LLM + apply ทั้ง dataset ==============
def filter_with_llm_reclass_all_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    - canonicalize entity
    - รวมเอนทิตีไม่ซ้ำ
    - เรียก LLM ทีละเอนทิตี
    - ใช้คลาสจาก LLM แทน; 'NO' → ทิ้ง
    """
    # 1) unique canonical entities
    seen, ents_to_check = set(), []
    for _, row in df.iterrows():
        for e in _safe_split_csv(row.get("Entity", "")):
            display = canonical_name(e)
            if not display:
                continue
            k = _ekey(display)
            if k not in seen:
                seen.add(k)
                ents_to_check.append(display)

    # 2) classify one-by-one
    entity_to_class = _llm_reclass_entities_one_by_one(ents_to_check)

    # 3) apply back
    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Apply LLM class per entity"):
        ents = _safe_split_csv(row.get("Entity", ""))
        kept_e, kept_c = [], []
        for e in ents:
            display = canonical_name(e)
            if not display:
                continue
            k = _ekey(display)
            lab = entity_to_class.get(k, "NO")
            if lab != "NO":
                kept_e.append(display)
                kept_c.append(lab)
        if kept_e:
            out_rows.append(
                {
                    "Topic_Normalized": row.get("Topic_Normalized", ""),
                    "Sentence_Index": row.get("Sentence_Index", ""),
                    "Entity": ", ".join(kept_e),
                    "Class": ", ".join(kept_c),
                }
            )
    return pd.DataFrame(
        out_rows, columns=["Topic_Normalized", "Sentence_Index", "Entity", "Class"]
    )


# ============== Final normalize & filter ==============
def _concat_nonempty(series: pd.Series) -> str:
    return ", ".join([s for s in series if isinstance(s, str) and s.strip()])


def _count_distinct_classes(cls_str: str) -> int:
    if not isinstance(cls_str, str) or not cls_str.strip():
        return 0
    return len({c.strip() for c in cls_str.split(",") if c.strip()})


def finalize_group_and_filter(df_filt: pd.DataFrame) -> pd.DataFrame:
    if df_filt.empty:
        return pd.DataFrame(columns=["Topic_Normalized", "Quantity", "Entity", "Class"])

    counts = df_filt.groupby("Topic_Normalized").size().rename("Quantity")

    agg = df_filt.groupby("Topic_Normalized").agg(
        Entity=("Entity", _concat_nonempty),
        Class=("Class", _concat_nonempty),
    )

    grouped = pd.concat([counts, agg], axis=1).reset_index()

    # กรอง: ต้องมีคลาสที่แตกต่างกันอย่างน้อย 2
    mask = grouped["Class"].apply(_count_distinct_classes) >= 2
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
    rows = len(df_filt)
    uniq_entities = df_filt["Entity"].nunique() if "Entity" in df_filt.columns else 0
    print(f"[saved LLM filtered] {OUT_FILE}")
    print(f"Rows: {rows:,} | Unique entities: {uniq_entities:,}")

    df_grouped = finalize_group_and_filter(df_filt)
    df_grouped.to_csv(OUT_FILE_GROUPED, index=False, encoding="utf-8-sig")
    print(f"[saved grouped+filtered] {OUT_FILE_GROUPED}")
    print(f"Topics kept: {len(df_grouped):,}")
