# normalize_and_filter_entities_uniquemean.py
from pathlib import Path
import re, math
import numpy as np
import pandas as pd

# ================= Config =================
INPUT = Path(
    "/home/whilebell/Code/techstack-ner/data/post_processed/all_predictions_dedup.csv"
)
OUT_DIR = INPUT.parent
OUT_NORM = OUT_DIR / "all_predictions_regex.csv"  # หลัง normalize

# กรองด้วย unique-mean
SCOPE = "global"  # "global" | "per_topic"
COUNT_MODE = "mentions"  # "rows" | "mentions"
ROUNDING = "ceil"  # "ceil" | "floor" | "none"
UNI_RATIO = 0.2  # threshold = unique-mean * UNI_RATIO
DROP_EMPTY_ROWS = True

# ============== Lexicon ==============
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

# ============== Helpers ==============
DASH_FIX = str.maketrans({"–": "-", "—": "-", "‐": "-", "“": '"', "”": '"', "’": "'"})
VER_PATTS = [
    re.compile(r"\b(v|ver\.?)\s*\d+(\.\d+)*\b", re.I),
    re.compile(r"\b\d+(\.\d+){1,}\b"),
]


def base_clean(s: str) -> str:
    s = (s or "").strip().translate(DASH_FIX)
    s = re.sub(r"[\t\r\n]+", " ", s)
    for p in VER_PATTS:
        s = p.sub("", s)
    s = re.sub(r"[ _/]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def low_key(s: str) -> str:
    s = s.lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s{2,}", " ", s).strip()


def compile_dict(patterns: dict):
    return {
        canon: [re.compile(p, re.I) for p in plist] for canon, plist in patterns.items()
    }


TOOLS_RX = compile_dict(TOOLS_PATTERNS)
SOFT_RX = compile_dict(SOFT_PATTERNS)


def exact_canon(s: str):
    return EXACT_MAP.get(low_key(s))


def normalize_token_keep_others(ent: str) -> str:
    s = base_clean(ent)
    if not s:
        return s
    c = exact_canon(s)
    if c:
        return c
    for canon, plist in TOOLS_RX.items():
        if any(p.search(s) for p in plist):
            return canon
    for canon, plist in SOFT_RX.items():
        if any(p.search(s) for p in plist):
            return canon
    return s  # อื่นคงเดิม


def _safe_split_csv(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [t.strip() for t in str(x).split(",") if t and str(t).strip()]


def normalize_row(entity_str: str, class_str: str):
    ents = _safe_split_csv(entity_str)
    clss = _safe_split_csv(class_str)
    n = min(len(ents), len(clss))
    ents, clss = ents[:n], clss[:n]
    kept_e, kept_c, seen = [], [], set()
    for e, c in zip(ents, clss):
        canon = normalize_token_keep_others(e)
        if not canon or canon in seen:
            continue
        kept_e.append(canon)
        kept_c.append(c)
        seen.add(canon)
    return ", ".join(kept_e), ", ".join(kept_c)


def explode_pairs(df: pd.DataFrame) -> pd.DataFrame:
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


def _round(x: float):
    if ROUNDING == "ceil":
        return math.ceil(x)
    if ROUNDING == "floor":
        return math.floor(x)
    return x


# ============== Unique-mean filter ==============
def filter_by_unique_mean(
    df: pd.DataFrame,
    scope: str = SCOPE,
    count_mode: str = COUNT_MODE,
    ratio: float = UNI_RATIO,
    rounding: str = ROUNDING,
    drop_empty_rows: bool = DROP_EMPTY_ROWS,
):
    per = explode_pairs(df)
    base = (
        per
        if count_mode == "mentions"
        else per.drop_duplicates(["Topic_Normalized", "Sentence_Index", "Entity"])
    )

    def _r(v):
        return (
            math.ceil(v)
            if rounding == "ceil"
            else (math.floor(v) if rounding == "floor" else v)
        )

    if scope == "global":
        freq = base["Entity"].value_counts()
        uniq_mean = float(np.mean(np.unique(freq.values))) if len(freq) else 0.0
        thr = _r(uniq_mean * ratio)
        allowed = set(freq[freq >= thr].index)

        def ok(topic, ent):
            return ent in allowed

        info = (
            f"GLOBAL unique-mean={uniq_mean:.3f} × {ratio} -> thr={thr} ({count_mode})"
        )
    else:
        g = (
            base.groupby(["Topic_Normalized", "Entity"])
            .size()
            .rename("cnt")
            .reset_index()
        )
        thr_by_topic = (
            g.groupby("Topic_Normalized")["cnt"]
            .apply(lambda s: _r(np.mean(np.unique(s.values)) * ratio))
            .to_dict()
        )
        allowed_by_topic = {
            t: set(sub.loc[sub["cnt"] >= thr_by_topic.get(t, 0), "Entity"])
            for t, sub in g.groupby("Topic_Normalized")
        }

        def ok(topic, ent):
            return ent in allowed_by_topic.get(topic, set())

        info = f"PER_TOPIC unique-mean × {ratio} ({count_mode})"

    out = []
    for topic, sent, e_str, c_str in df[
        ["Topic_Normalized", "Sentence_Index", "Entity", "Class"]
    ].itertuples(index=False, name=None):
        e_list = _safe_split_csv(e_str)
        c_list = _safe_split_csv(c_str)
        n = min(len(e_list), len(c_list))
        e_list, c_list = e_list[:n], c_list[:n]
        kept_e, kept_c = [], []
        for e, c in zip(e_list, c_list):
            if ok(topic, e):
                kept_e.append(e)
                kept_c.append(c)
        if kept_e or not drop_empty_rows:
            out.append(
                {
                    "Topic_Normalized": topic,
                    "Sentence_Index": sent,
                    "Entity": ", ".join(kept_e),
                    "Class": ", ".join(kept_c),
                }
            )
    return pd.DataFrame(out), info


# ============== Main ==============
if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    for col in ["Entity", "Class", "Topic_Normalized"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 1) Normalize (tools/soft skill เท่านั้น; อื่นคงเดิม) + dedup ต่อแถว
    pair = df[["Entity", "Class"]].apply(
        lambda r: pd.Series(
            normalize_row(r["Entity"], r["Class"]), index=["Entity", "Class"]
        ),
        axis=1,
    )
    df_norm = df.copy()
    df_norm[["Entity", "Class"]] = pair

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(OUT_NORM, index=False, encoding="utf-8-sig")
    print(f"[saved normalize] {OUT_NORM}")

    # สถิติหลัง normalize
    per = explode_pairs(df_norm)
    base = per.drop_duplicates(["Topic_Normalized", "Sentence_Index", "Entity"])
    freq = base["Entity"].value_counts()
    print("\nTop 5 after normalize (per sentence):")
    print(freq.head(5).to_string())
    print("\nBottom 5 after normalize (per sentence):")
    print(freq.sort_values(ascending=True).head(5).to_string())

    # 2) Filter by unique-mean × ratio
    df_filt, info = filter_by_unique_mean(
        df_norm,
        scope=SCOPE,
        count_mode=COUNT_MODE,
        ratio=UNI_RATIO,
        rounding=ROUNDING,
        drop_empty_rows=DROP_EMPTY_ROWS,
    )
    OUT_FILT = (
        OUT_DIR
        / f"all_predictions_regex_uniquemean_{SCOPE}_{COUNT_MODE}_r{int(UNI_RATIO*100)}.csv"
    )
    df_filt.to_csv(OUT_FILT, index=False, encoding="utf-8-sig")
    print(f"\n[filter] {info}")
    print(f"[saved filtered] {OUT_FILT}")

    # สถิติหลังกรอง
    per2 = explode_pairs(df_filt).drop_duplicates(
        ["Topic_Normalized", "Sentence_Index", "Entity"]
    )
    freq2 = per2["Entity"].value_counts()
    print("\nTop 5 after filter:")
    print(freq2.head(5).to_string())
    print("\nBottom 5 after filter:")
    print(freq2.sort_values(ascending=True).head(5).to_string())
    print(f"\nRows: {len(df_filt):,} | Unique entities: {freq2.size:,}")
