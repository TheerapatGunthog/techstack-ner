import json
import requests
from pathlib import Path

BASE_URL = "https://3bf8cb510e1e.ngrok-free.app/"  # เปลี่ยนเป็น origin ของคุณถ้าไม่ใช่พอร์ตนี้

# แมปไฟล์ -> endpoint
TASKS = [
    # jobs.json -> positions/replace
    (
        "/home/whilebell/Code/techstack-ner/data/post_processed/jobs.json",
        "/api/positions/replace",
    ),
    # skills.json -> skills/replace
    (
        "/home/whilebell/Code/techstack-ner/data/post_processed/skills.json",
        "/api/skills/replace",
    ),
    # job_skill_scores.json -> job-skills/replace
    (
        "/home/whilebell/Code/techstack-ner/data/post_processed/job_skill_scores.json",
        "/api/job-skills/replace",
    ),
    # Trading.json -> tradings/replace
    (
        "/home/whilebell/Code/techstack-ner/data/post_processed/trending.json",
        "/api/position-details",
    ),
]


def post_items(file_path: Path, endpoint: str, dry_run: bool = False):
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # ไฟล์ทั้ง 3 เป็นรูปแบบ {"items": [...]} อยู่แล้ว
    if (
        "items" not in data
        or not isinstance(data["items"], list)
        or len(data["items"]) == 0
    ):
        raise ValueError(f"{file_path.name} ต้องมี key 'items' เป็น array และต้องไม่ว่าง")

    url = f"{BASE_URL}{endpoint}"
    payload = {"items": data["items"], "dryRun": dry_run}

    resp = requests.post(url, json=payload, timeout=60)
    try:
        resp_json = resp.json()
    except Exception:
        resp.raise_for_status()
        # ถ้าไม่มี body เป็น JSON แต่สถานะ OK ก็แสดง text
        resp_json = {"text": resp.text}

    return resp.status_code, resp_json


def main():
    # 1) Dry-run เช็คความถูกต้องก่อน (server จะ validate ให้)
    print("=== DRY RUN ===")
    for filename, endpoint in TASKS:
        path = Path(filename)
        code, out = post_items(path, endpoint, dry_run=True)
        print(f"{endpoint} [{filename}] -> {code}")
        print(out)

    # 2) ถ้า OK แล้ว ค่อยยิงจริง (ลบทั้งหมดแล้วแทนที่ใหม่ตามโค้ดฝั่ง server)
    print("\n=== REPLACE (WRITE) ===")
    for filename, endpoint in TASKS:
        path = Path(filename)
        code, out = post_items(path, endpoint, dry_run=False)
        print(f"{endpoint} [{filename}] -> {code}")
        print(out)


if __name__ == "__main__":
    main()
