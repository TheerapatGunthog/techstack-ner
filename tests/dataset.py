import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.raw import RAW_DATA_PATH


df_full = pd.read_csv(RAW_DATA_PATH / "merged.csv")
third_size = len(df_full) // 5
df = df_full.iloc[:third_size]

print(df.shape)
print(df_full.shape)
