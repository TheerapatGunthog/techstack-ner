# normalize_entity_with_LLM_single.py
from pathlib import Path
import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= Config =================
INPUT = Path(os.getcwd() + "/data/post_processed/all_predictions_dedup.csv")
OUT_DIR = INPUT.parent
OUT_FILE = OUT_DIR / "all_predictions_llm_filtered.csv"
OUT_FILE_GROUPED = OUT_DIR / "all_predictions_llm_filtered_grouped.csv"

LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:8b"
TIMEOUT = 90
MAX_RETRIES = 2
RETRY_SLEEP = 1.0

# Speed knobs
BATCH_SIZE = 16            # รวมหลายเอนทิตีต่อคำขอ
MAX_WORKERS = 3            # ขนานแบบจำกัด
LLM_OPTIONS = {"temperature": 0, "num_predict": 4}
LLM_STOP = ["\n", "\r"]

# Expanded patterns for current Computer Engineering job postings.
# Regex are case-insensitive friendly. Aim: maximize root-term normalization coverage.

TOOLS_PATTERNS = {
    # ---------------- Languages ----------------
    "Python": [r"\bpy(?:thon)?\b", r"\bphyton\b"],
    "C++": [r"\bc\+\+\b", r"\bc\s*plus\s*plus\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "C": [r"\bc\b(?!\+\+|\s*sharp)"],
    "Java": [r"\bjava\b"],
    "JavaScript": [r"\bjava\s*script\b", r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btype\s*script\b", r"\btypescript\b", r"\bts\b"],
    "Go": [r"\bgo(lang)?\b"],
    "Rust": [r"\brust\b"],
    "R": [r"\br\b", r"\brstats\b"],
    "Scala": [r"\bscala\b"],
    "Perl": [r"\bperl\b"],
    "PHP": [r"\bphp\b"],
    "Swift": [r"\bswift\b"],
    "Objective-C": [r"\bobjective[\s-]?c\b", r"\bobj[\s-]?c\b"],
    "Ruby": [r"\bruby\b"],
    "Kotlin": [r"\bkotlin\b"],
    "Dart": [r"\bdart\b"],
    "HCL": [r"\bhcl\b"],
    "Groovy": [r"\bgroovy\b"],
    "Shell": [r"\bshell\b", r"\bsh\b"],
    "Bash": [r"\bbash\b"],
    "PowerShell": [r"\bpowershell\b"],
    "SQL": [r"\bsql\b"],
    "GraphQL": [r"\bgraph\s*q[l|1]\b", r"\bgraphql\b"],
    "HTML": [r"\bhtml?\b"],
    "CSS": [r"\bcss\b"],
    "Sass": [r"\bsass\b", r"\bscss\b"],
    "MATLAB": [r"\bmatlab\b"],
    "Fortran": [r"\bfortran\b"],

    # ---------------- Frameworks / Runtimes / Libs ----------------
    "Node.js": [r"\bnode\.?js\b", r"\bnode\s*js\b", r"\bnode\b"],
    "React": [r"\breact(\.js)?\b", r"\breactjs\b"],
    "Next.js": [r"\bnext\.?js\b"],
    "Angular": [r"\bangular(\.js)?\b", r"\bangularjs\b"],
    "Vue.js": [r"\bvue(\.js)?\b", r"\bvuejs\b"],
    "Nuxt.js": [r"\bnuxt\.?js\b"],
    "Svelte": [r"\bsvelte\b"],
    "SvelteKit": [r"\bsvelte\s*kit\b"],
    "Redux": [r"\bredux\b"],
    "Tailwind CSS": [r"\btailwind(\s*css)?\b"],
    "Bootstrap": [r"\bbootstrap\b"],
    "Material UI": [r"\bmaterial\s*ui\b", r"\bmui\b"],
    "Chakra UI": [r"\bchakra\s*ui\b"],
    "Three.js": [r"\bthree\.?js\b"],
    "D3.js": [r"\bd3(\.js)?\b"],
    "Webpack": [r"\bwebpack\b"],
    "Vite": [r"\bvite\b"],
    "Babel": [r"\bbabel\b"],
    "ESLint": [r"\beslint\b"],
    "Prettier": [r"\bprettier\b"],
    "Storybook": [r"\bstorybook\b"],

    "Django": [r"\bdjango\b"],
    "Flask": [r"\bflask\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Spring": [r"\bspring\b"],
    ".NET": [r"\b\.net\b", r"\bdotnet\b"],
    "ASP.NET": [r"\basp\.?net\b"],
    "RxJava": [r"\brx\s*java\b"],
    "gRPC": [r"\bgrpc\b"],
    "REST": [r"\brestful?\b", r"\brest\s*api\b"],
    "OpenAPI": [r"\bopenapi\b"],
    "Swagger": [r"\bswagger\b"],

    "TensorFlow": [r"\btensor\s*flow\b", r"\btensorflow\b", r"\btf\b"],
    "Keras": [r"\bkeras\b"],
    "PyTorch": [r"\bpytorch\b", r"\btorch\b"],
    "JAX": [r"\bjax\b"],
    "scikit-learn": [r"\bscikit[- ]?learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "SciPy": [r"\bscipy\b"],
    "Statsmodels": [r"\bstatsmodels?\b"],
    "XGBoost": [r"\bxgboost\b"],
    "LightGBM": [r"\blight\s*gbm\b", r"\blightgbm\b", r"\blgbm\b"],
    "CatBoost": [r"\bcatboost\b"],
    "OpenCV": [r"\bopen\s*cv\b", r"\bopencv\b"],
    "Dask": [r"\bdask\b"],
    "Ray": [r"\bray\b"],
    "MLflow": [r"\bmlflow\b"],
    "Weights & Biases": [r"\bweights?\s*&\s*biases\b", r"\bwandb\b"],
    "Optuna": [r"\boptuna\b"],
    "Hydra": [r"\bhydra\b"],
    "HuggingFace": [r"\bhugging\s*face\b"],
    "Transformers": [r"\btransformers?\b"],
    "LangChain": [r"\blang\s*chain\b"],
    "LlamaIndex": [r"\bllama\s*index\b"],
    "vLLM": [r"\bvllm\b"],
    "Triton Inference Server": [r"\btriton\s*inference\s*server\b", r"\bnvidia\s*triton\b"],

    # ---------------- Data / Streaming / Orchestration ----------------
    "Airflow": [r"\bair\s*flow\b", r"\bairflow\b"],
    "Prefect": [r"\bprefect\b"],
    "Luigi": [r"\bluigi\b"],
    "dbt": [r"\bdbt\b", r"\bdata\s*build\s*tool\b"],
    "Airbyte": [r"\bairbyte\b"],
    "Kafka": [r"\bkafka\b"],
    "Redpanda": [r"\bredpanda\b"],
    "RabbitMQ": [r"\brabbit\s*mq\b", r"\brabbitmq\b"],
    "ActiveMQ": [r"\bactive\s*mq\b", r"\bactivemq\b"],
    "NATS": [r"\bnats\b"],
    "MQTT": [r"\bmqtt\b"],
    "Spark": [r"\bspark\b"],
    "Hadoop": [r"\bhadoop\b"],
    "Flink": [r"\bflink\b"],
    "Beam": [r"\bapache\s*beam\b", r"\bbeam\b"],
    "Delta Lake": [r"\bdelta\s*lake\b"],
    "Apache Iceberg": [r"\biceberg\b"],
    "Apache Hudi": [r"\bhudi\b"],

    # ---------------- Databases / Storage / Search ----------------
    "PostgreSQL": [r"\bpostgre\s*sql\b", r"\bpostgres\b", r"\bpostgresql\b"],
    "MySQL": [r"\bmysql\b"],
    "SQLite": [r"\bsqlite\b"],
    "SQL Server": [r"\bsql\s*server\b", r"\bms\s*sql\b", r"\bmssql\b"],
    "Oracle": [r"\boracle\b"],
    "MariaDB": [r"\bmaria\s*db\b", r"\bmariadb\b"],
    "Cassandra": [r"\bcassandra\b"],
    "MongoDB": [r"\bmonogo?db\b", r"\bmongo\s*db\b", r"\bmongodb\b"],
    "DynamoDB": [r"\bdynamo\s*db\b", r"\bdynamodb\b"],
    "Firestore": [r"\bfire\s*store\b"],
    "CouchDB": [r"\bcouchdb\b"],
    "Neo4j": [r"\bneo\s*4j\b", r"\bneo4j\b"],
    "ArangoDB": [r"\barango\s*db\b"],
    "CosmosDB": [r"\bcosmos\s*db\b"],
    "CockroachDB": [r"\bcockroach\s*db\b"],
    "TiDB": [r"\btidb\b"],
    "ClickHouse": [r"\bclick\s*house\b", r"\bclickhouse\b"],
    "Vertica": [r"\bvertica\b"],
    "Teradata": [r"\bteradata\b"],
    "Db2": [r"\bdb2\b"],
    "SAP HANA": [r"\bsap\s*hana\b"],
    "Snowflake": [r"\bsnowflake\b"],
    "BigQuery": [r"\bbig\s*query\b", r"\bbigquery\b"],
    "Redshift": [r"\bredshift\b"],
    "Elasticsearch": [r"\belastic\s*search\b", r"\belasticsearch\b"],
    "OpenSearch": [r"\bopen\s*search\b", r"\bopensearch\b"],
    "Solr": [r"\bsolr\b"],
    "Memcached": [r"\bmemcached\b"],
    "Redis": [r"\bredis\b"],
    "Hazelcast": [r"\bhazelcast\b"],
    "Ignite": [r"\bapache\s*ignite\b", r"\bignite\b"],
    "MinIO": [r"\bminio\b"],
    "Parquet": [r"\bparquet\b"],
    "Avro": [r"\bavro\b"],
    "ORC": [r"\borc\b"],

    # ---------------- DevOps / Infra / SRE ----------------
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Helm": [r"\bhelm\b"],
    "Kustomize": [r"\bkustomize\b"],
    "OpenShift": [r"\bopen\s*shift\b", r"\bopenshift\b"],
    "Istio": [r"\bistio\b"],
    "Linkerd": [r"\blinkerd\b"],
    "Envoy": [r"\benvoy\b"],
    "Traefik": [r"\btraefik\b"],
    "Nginx": [r"\bnginx\b"],
    "Apache HTTPD": [r"\bapache\b", r"\bhttpd\b"],
    "HAProxy": [r"\bhaproxy\b"],
    "Consul": [r"\bconsul\b"],
    "Vault": [r"\bvault\b"],

    "Terraform": [r"\bterraform\b"],
    "CloudFormation": [r"\bcloud\s*formation\b", r"\bcloudformation\b"],
    "CDK": [r"\bcdk\b", r"\bcloud\s*development\s*kit\b"],
    "Pulumi": [r"\bpulumi\b"],
    "Ansible": [r"\bansible\b"],
    "Puppet": [r"\bpuppet\b"],
    "Chef": [r"\bchef\b"],
    "SaltStack": [r"\bsalt\s*stack\b", r"\bsaltstack\b"],

    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "GitLab": [r"\bgitlab\b"],
    "Bitbucket": [r"\bbitbucket\b"],
    "SVN": [r"\bsvn\b", r"\bsubversion\b"],
    "Mercurial": [r"\bmercurial\b", r"\bhg\b"],

    "Jenkins": [r"\bjenkins\b"],
    "GitHub Actions": [r"\bgithub\s*actions?\b"],
    "GitLab CI": [r"\bgitlab\s*ci\b"],
    "CircleCI": [r"\bcircle\s*ci\b"],
    "TravisCI": [r"\btravis\s*ci\b"],
    "TeamCity": [r"\bteam\s*city\b", r"\bteamcity\b"],
    "Bamboo": [r"\bbamboo\b"],
    "Azure DevOps": [r"\bazure\s*dev\s*ops\b"],
    "Bitrise": [r"\bbitrise\b"],

    "Nagios": [r"\bnagios\b"],
    "Zabbix": [r"\bzabbix\b"],
    "Prometheus": [r"\bprometheus\b"],
    "PromQL": [r"\bpromql\b"],
    "Grafana": [r"\bgrafana\b"],
    "Datadog": [r"\bdata\s*dog\b", r"\bdatadog\b"],
    "New Relic": [r"\bnew\s*relic\b"],
    "Splunk": [r"\bsplunk\b"],
    "Sentry": [r"\bsentry\b"],
    "OpenTelemetry": [r"\bopen\s*telemetry\b", r"\botel\b", r"\botel\b"],

    "SonarQube": [r"\bsonar\s*qube\b", r"\bsonarqube\b"],
    "Snyk": [r"\bsnyk\b"],
    "Trivy": [r"\btrivy\b"],
    "Anchore": [r"\banchore\b"],
    "Clair": [r"\bclair\b"],
    "Burp Suite": [r"\bburp\s*suite\b"],
    "OWASP ZAP": [r"\bowasp\s*zap\b", r"\bzap\s*proxy\b"],
    "Nessus": [r"\bnessus\b"],
    "Keycloak": [r"\bkey\s*cloak\b", r"\bkeycloak\b"],
    "Auth0": [r"\bauth0\b"],

    # ---------------- Build / Package / Testing ----------------
    "Maven": [r"\bmaven\b"],
    "Gradle": [r"\bgradle\b"],
    "Ant": [r"\bant\b"],
    "sbt": [r"\bsbt\b"],
    "Bazel": [r"\bbazel\b"],
    "Buck": [r"\bbuck\b"],
    "CMake": [r"\bcmake\b"],
    "Make": [r"\bmake\b", r"\bmakefile\b"],
    "Ninja": [r"\bninja\b"],

    "npm": [r"\bnpm\b"],
    "Yarn": [r"\byarn\b"],
    "pnpm": [r"\bpnpm\b"],
    "pip": [r"\bpip\b"],
    "pipenv": [r"\bpipenv\b"],
    "poetry": [r"\bpoetry\b"],
    "conda": [r"\bconda\b"],
    "virtualenv": [r"\bvirtual\s*env\b", r"\bvirtualenv\b"],
    "Composer": [r"\bcomposer\b"],
    "RubyGems": [r"\bgems?\b", r"\brubygems?\b"],
    "Cargo": [r"\bcargo\b"],
    "NuGet": [r"\bnuget\b"],
    "Go Modules": [r"\bgo\s*mod(ules)?\b"],

    "JUnit": [r"\bjunit\b"],
    "TestNG": [r"\btestng\b"],
    "NUnit": [r"\bnunit\b"],
    "xUnit": [r"\bxunit\b"],
    "pytest": [r"\bpy\s*test\b", r"\bpytest\b"],
    "unittest": [r"\bunit\s*test\b", r"\bunittest\b"],
    "nose": [r"\bnose\b"],
    "Jest": [r"\bjest\b"],
    "Mocha": [r"\bmocha\b"],
    "Chai": [r"\bchai\b"],
    "Jasmine": [r"\bjasmine\b"],
    "Cypress": [r"\bcypress\b"],
    "Playwright": [r"\bplaywright\b"],
    "Selenium": [r"\bselenium\b"],
    "Appium": [r"\bappium\b"],
    "Robot Framework": [r"\brobot\s*framework\b"],

    "Postman": [r"\bpostman\b"],
    "Newman": [r"\bnewman\b"],
    "JMeter": [r"\bjmeter\b"],
    "Gatling": [r"\bgatling\b"],
    "Locust": [r"\blocust\b"],
    "k6": [r"\bk6\b"],

    # ---------------- Mobile / Desktop ----------------
    "Android": [r"\bandroid\b"],
    "iOS": [r"\bios\b"],
    "Jetpack Compose": [r"\bjetpack\s*compose\b"],
    "SwiftUI": [r"\bswift\s*ui\b"],
    "UIKit": [r"\buikit\b"],
    "Xcode": [r"\bxcode\b"],
    "Fastlane": [r"\bfastlane\b"],
    "React Native": [r"\breact\s*native\b"],
    "Flutter": [r"\bflutter\b"],
    "CocoaPods": [r"\bcocoa\s*pods\b", r"\bcocoapods\b"],
    "Carthage": [r"\bcarthage\b"],

    # ---------------- BI / Analytics ----------------
    "Power BI": [r"\bpower\s*bi\b"],
    "Tableau": [r"\btableau\b"],
    "Looker": [r"\blooker\b"],
    "Superset": [r"\b(super\s*)?set\b", r"\bapache\s*superset\b"],
    "Metabase": [r"\bmetabase\b"],
    "MicroStrategy": [r"\bmicro\s*strategy\b", r"\bmicrostrategy\b"],
    "Qlik": [r"\bqlik\b", r"\bqlik\s*sense\b", r"\bqlik\s*view\b"],

    # ---------------- Productivity / OS ----------------
    "Linux": [r"\blinux\b"],
    "Windows": [r"\bwindows\b"],
    "macOS": [r"\bmac\s*os\b", r"\bmacos\b"],
    "Excel": [r"\bms\s*excel\b", r"\bmicrosoft\s*excel\b", r"\bexcel\b"],
    "Word": [r"\bms\s*word\b", r"\bmicrosoft\s*word\b", r"\bword\b"],
    "PowerPoint": [r"\bpower\s*point\b", r"\bpowerpoint\b"],
    "Google Sheets": [r"\bgoogle\s*sheets?\b", r"\bg\s*sheets?\b"],
    "Google Docs": [r"\bgoogle\s*docs?\b"],
    "Google Slides": [r"\bgoogle\s*slides?\b"],

    # ---------------- Auth / Identity ----------------
    "OAuth2": [r"\boauth\s*2(\.0)?\b", r"\boauth2\b"],
    "OIDC": [r"\boidc\b"],
    "SAML": [r"\bsaml\b"],

    # ---------------- Game / Engines ----------------
    "Unity": [r"\bunity\b"],
    "Unreal Engine": [r"\bunreal(\s*engine)?\b"],
}

SOFT_PATTERNS = {
    # Core soft skills
    "Communication": [r"\bcommunication(s)?\b", r"\bcommunicate\b", r"\bpresentation(s)?\b", r"\bwritten\s+and\s+verbal\b"],
    "Teamwork": [r"\bteam\s*work\b", r"\bteam\s*player\b", r"\bcollaborat(e|ion|ive)\b", r"\bcross[-\s]*functional\b"],
    "Leadership": [r"\blead(ership)?\b", r"\bmentor(ing|ship)?\b", r"\bpeople\s*management\b", r"\bteam\s*lead\b"],
    "Problem Solving": [r"\bproblem[- ]?solv(ing|er)\b", r"\btroubleshoot(ing)?\b"],
    "Critical Thinking": [r"\bcritical\s*thinking\b", r"\banalytical\s*thinking\b"],
    "Time Management": [r"\btime\s*management\b", r"\bdeadline[-\s]*driven\b", r"\bprioriti[sz]ation\b"],
    "Adaptability": [r"\badaptabilit(y|ies)\b", r"\badaptable\b", r"\badapt\b", r"\bflexib(le|ility)\b"],
    "Creativity": [r"\bcreativ(e|ity)\b", r"\binnovation\b"],
    "Attention to Detail": [r"\battention\s*to\s*detail\b", r"\bdetail[- ]?oriented\b", r"\bmeticulous\b"],
    "Stakeholder Management": [r"\bstakeholder\s*management\b", r"\bstakeholder\s*communication\b"],
    "Project Management": [r"\bproject\s*management\b", r"\bmanage\s*projects?\b"],
    "Decision Making": [r"\bdecision\s*making\b"],
    "Negotiation": [r"\bnegotiation(s)?\b"],
    "Collaboration": [r"\bcollaboration\b"],
    "Presentation": [r"\bpresentation(s)?\b"],
    "Ownership": [r"\bownership\b", r"\baccountab(le|ility)\b"],
    "Initiative": [r"\binitiative\b", r"\bproactiv(e|ity)\b", r"\bself[-\s]*starter\b"],
    "Learning Agility": [r"\bfast\s*learner\b", r"\bcontinuous\s*learning\b", r"\bself[-\s]*learning\b"],
    "Conflict Resolution": [r"\bconflict\s*resolution\b"],
    "Risk Management": [r"\brisk\s*management\b"],
    "Requirements Gathering": [r"\brequirements?\s*gather(ing)?\b"],
    "Documentation": [r"\bdocumentation\b", r"\btechnical\s*writing\b"],
    "Customer Focus": [r"\bcustomer\s*(focus|centric)\b"],
    "Strategic Thinking": [r"\bstrategic\s*thinking\b"],
    "Systems Thinking": [r"\bsystems?\s*thinking\b"],

    # Methods / practices
    "Agile": [r"\bagile\b"],
    "Scrum": [r"\bscrum\b"],
    "Kanban": [r"\bkanban\b"],
    "XP": [r"\bextreme\s*programming\b", r"\bxp\b"],
    "DevOps": [r"\bdev\s*ops\b", r"\bdevops\b"],
    "CI": [r"\bcontinuous\s*integration\b", r"\bci\b"],
    "CD": [r"\bcontinuous\s*delivery\b", r"\bcontinuous\s*deployment\b", r"\bcd\b"],
    "TDD": [r"\btdd\b", r"\btest[- ]?driven\s*development\b"],
    "BDD": [r"\bbdd\b", r"\bbehavior[- ]?driven\s*development\b"],
    "SOLID": [r"\bsolid\b"],
    "DRY": [r"\bdry\b"],
    "KISS": [r"\bkiss\b"],
    "YAGNI": [r"\byagni\b"],
    "DDD": [r"\bdomain[-\s]*driven\s*design\b", r"\bddd\b"],
    "Clean Architecture": [r"\bclean\s*architecture\b"],
    "MVC": [r"\bmvc\b"],
    "MVP": [r"\bmvp\b"],
    "MVVM": [r"\bmvvm\b"],
    "SDLC": [r"\bsdlc\b"],
    "GitFlow": [r"\bgit\s*flow\b"],
    "Trunk Based Development": [r"\btrunk[-\s]*based\s*development\b"],
    "OKR": [r"\bokr(s)?\b"],
    "KPI": [r"\bkpi(s)?\b"],
}

EXACT_MAP = {
    # Existing mappings
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
    "gdocs": "Google Docs",
    "google doc": "Google Docs",
    "gdoc": "Google Docs",
    "google slide": "Google Slides",
    "gslide": "Google Slides",
    "gslides": "Google Slides",
    "powerpoint": "PowerPoint",
    "ppt": "PowerPoint",
    "pptx": "PowerPoint",
    "word doc": "Word",
    "docx": "Word",
    "xlsx": "Excel",
    "csv": "Excel",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "ms sql": "SQL Server",
    "mssql": "SQL Server",
    "dotnet": ".NET",
    "obj-c": "Objective-C",

    # New convenient aliases
    "tf": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "ts": "TypeScript",
    "reactjs": "React",
    "vuejs": "Vue.js",
    "angularjs": "Angular",
    "nextjs": "Next.js",
    "nuxtjs": "Nuxt.js",
    "mui": "Material UI",
    "materialui": "Material UI",
    "tailwind": "Tailwind CSS",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "rest api": "REST",
    "swagger": "Swagger",
    "openapi": "OpenAPI",
    "gh actions": "GitHub Actions",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "azure devops": "Azure DevOps",
    "k8s": "Kubernetes",
    "kubectl": "Kubernetes",
    "helm": "Helm",
    "kustomize": "Kustomize",
    "elk": "Elasticsearch",
    "opensearch": "OpenSearch",
    "nginx": "Nginx",
    "httpd": "Apache HTTPD",
    "apache": "Apache HTTPD",
    "haproxy": "HAProxy",
    "dbt": "dbt",
    "mlops": "MLflow",
    "wandb": "Weights & Biases",
    "llama index": "LlamaIndex",
    "vllm": "vLLM",
    "triton": "Triton Inference Server",
    "sonarqube": "SonarQube",
    "zap": "OWASP ZAP",
    "burp": "Burp Suite",
    "keycloak": "Keycloak",
    "auth 0": "Auth0",
    "mac os": "macOS",
    "osx": "macOS",
    "g-suite": "Google Docs",
    "google workspace": "Google Docs",
    "jira": "Jira",
    "confluence": "Confluence",
    "ci/cd": "CI",
    "cicd": "CI",
}


def base_clean(s: str) -> str:
    return (s or "").strip()


def compile_dict(patterns: dict):
    return {k: [re.compile(p, re.I) for p in v] for k, v in patterns.items()}


# ใช้ dict เดิมที่ประกาศไว้นอกไฟล์นี้
TOOLS_RX = compile_dict(TOOLS_PATTERNS)
SOFT_RX = compile_dict(SOFT_PATTERNS)

# ============== Helpers ==============
_SLASH_EDGE_RX = re.compile(r"^\s*/\s*$")  # "/" ล้วนๆ
_SLASH_LEAD_RX = re.compile(r"^\s*/\s*")   # ขึ้นต้นด้วย "/"
_SLASH_TAIL_RX = re.compile(r"\s*/\s*$")   # ลงท้ายด้วย "/"


def _pre_normalize_entity(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    if _SLASH_EDGE_RX.match(s):
        return ""
    s = _SLASH_LEAD_RX.sub("", s)
    s = _SLASH_TAIL_RX.sub("", s)
    return s.strip()


def exact_canon(s: str):
    s0 = _pre_normalize_entity(base_clean(s))
    if not s0:
        return ""
    key = s0.lower().replace("-", " ").replace("_", " ").strip()
    return EXACT_MAP.get(key, "")


def canonical_name(s: str) -> str:
    s0 = _pre_normalize_entity(base_clean(s))
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
    s = _pre_normalize_entity((s or "").strip())
    s = re.sub(r'[\(\)\[\]\{\}"“”]', "", s)
    s = re.sub(r"\b[vV]?\d+(\.\d+){0,3}\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


# ============== LLM classify ==============
_ALLOWED = {"PSML", "DB", "CP", "FAL", "TAS", "HW"}


def _prompt_one(entity: str) -> str:
    return f"""
You are a STRICT validator for ONE entity.
Return EXACTLY ONE token from: PSML DB CP FAL TAS HW NO
UPPERCASE only. No prose. No punctuation.
DEFINITIONS
PSML: programming/scripting languages only (Python, Java, Kotlin, C/C++, C#, JavaScript, TypeScript, R, Go, Rust, SQL, Bash, PowerShell, MATLAB).
DB: database engines/warehouses/services/key-value/document/search (PostgreSQL, MySQL, SQLite, Oracle, SQL Server, MongoDB, Redis, Cassandra, BigQuery, Snowflake, Redshift, DynamoDB, Hive, Trino, Presto, Elasticsearch).
CP: cloud PROVIDER/PLATFORM names only (AWS, Google Cloud, Microsoft Azure, Alibaba Cloud, Firebase platform).
FAL: frameworks/libraries/tools/runtimes/OS/SDK/CI-CD/VCS/BI/office (React, Vue.js, Angular, Django, Flask, FastAPI, Spring, Android SDK, TensorFlow, PyTorch, scikit-learn, pandas, NumPy, Node.js, .NET, Docker, Kubernetes, Git, GitHub, GitLab, Jenkins, Jira, Confluence, Linux, Windows, macOS, Excel, Word, PowerPoint, Power BI, Tableau, Looker, OpenCV, Ansible, Terraform).
TAS: soft skills, techniques, methodologies, patterns (Agile, Scrum, Kanban, Communication, Teamwork, Leadership, Problem Solving, Time Management, Stakeholder Management, Project Management, Negotiation, MVC, MVVM, MVP, Clean Architecture).
HW: physical devices/components (Raspberry Pi, Arduino, NVIDIA GPU, Intel CPU, FPGA, microcontroller).
HARD RULES
Managed DB services (BigQuery, Redshift, DynamoDB, Firestore) → DB.
CP only for provider names; non-database cloud services that are not tools → NO.
Adjectives/marketing terms/generic words → NO.
Job titles, company names, locations, salaries, sentences, responsibilities → NO.
Unknown brands/proper nouns not in definitions → NO.
Glued-together fake names → NO.
If uncertain → NO.
Now classify exactly one token for:
Entity: {entity}
Output:
""".strip()


def _prompt_batch(entities: list[str]) -> str:
    items = "\n".join(f"{i+1}. {e}" for i, e in enumerate(entities))
    return (
        "Classify EACH line with EXACTLY ONE token from: PSML DB CP FAL TAS HW NO.\n"
        "Rules: provider-only=CP; managed-DB=DB; tools/libs/OS/CI/VCS/BI/office=FAL; "
        "soft-skills/methods=TAS; languages=PSML; hardware=HW; else NO.\n"
        "Return ONE LINE ONLY with N tokens separated by single spaces, in order.\n"
        f"{items}\n"
        "Output:\n"
    )


def _post_ollama(session: requests.Session, payload: dict) -> str:
    r = session.post(LLM_URL, json=payload, timeout=TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json().get("response", "").strip()


def _llm_classify_one(entity: str, session: requests.Session) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": _prompt_one(entity),
        "stream": False,                 # <-- แก้จากเดิมที่ใส่ list ผิด
        "options": LLM_OPTIONS,
        "stop": LLM_STOP,                # single entity เท่านั้นที่ใช้ stop ขึ้นบรรทัด
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.post(LLM_URL, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            resp = r.json().get("response", "").strip()
            m = re.search(r"\b(PSML|DB|CP|FAL|TAS|HW|NO)\b", resp, flags=re.I)
            lab = m.group(1).upper() if m else "NO"
            return lab if lab in _ALLOWED or lab == "NO" else "NO"
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * (attempt + 1))
                continue
            return "NO"


def _llm_classify_batch(entities: list[str]) -> list[str]:
    sess = requests.Session()
    payload = {
        "model": LLM_MODEL,
        "prompt": _prompt_batch(entities),
        "stream": False,
        "options": LLM_OPTIONS,
        # ไม่มี "stop" ที่เป็น newline ในโหมด batch
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = sess.post(LLM_URL, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            resp = r.json().get("response", "").strip()
            # คาดหวัง "FAL PSML DB ..." 1 บรรทัด
            toks = [t.strip().upper() for t in resp.split() if t.strip()]
            out = []
            for t in toks[:len(entities)]:
                out.append(t if t in _ALLOWED or t == "NO" else "NO")
            while len(out) < len(entities):
                out.append("NO")
            return out
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * (attempt + 1))
                continue
            return ["NO"] * len(entities)


def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _llm_reclass_entities_parallel_batched(unique_entities: list[str]) -> dict[str, str]:
    batches = list(_chunk(unique_entities, BATCH_SIZE))
    entity_to_class: dict[str, str] = {}

    def _work(batch):
        labels = _llm_classify_batch(batch)
        return [(e, lab) for e, lab in zip(batch, labels)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_work, b) for b in batches]
        for f in tqdm(as_completed(futures), total=len(futures), desc="LLM reclass (batched+parallel)"):
            for e, lab in f.result():
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
    - เรียก LLM แบบ batch + ขนานจำกัด
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

    # 2) classify batched + parallel
    entity_to_class = _llm_reclass_entities_parallel_batched(ents_to_check)

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
