import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH

with open(
    INTERIM_DATA_PATH / "change-a-path-here",  # <--- dont'forget to change the path
    "r",
    encoding="utf-8",
) as file:
    data = json.load(file)

split_index = len(data) // 2

part1 = data[:split_index]
part2 = data[split_index:]

with open(
    INTERIM_DATA_PATH / "./splited-data/dataset_part1.json", "w", encoding="utf-8"
) as file1:
    json.dump(part1, file1, ensure_ascii=False, indent=4)

with open(
    INTERIM_DATA_PATH / "./splited-data/dataset_part2.json", "w", encoding="utf-8"
) as file2:
    json.dump(part2, file2, ensure_ascii=False, indent=4)

print("Dataset has splited")
