"""预览所有数据集样本"""
import json
from pathlib import Path

import pandas as pd

SAMPLES = [
    ("chinese-adorable-high-eq", "train/data.parquet", "高情商AI女友对话"),
    ("chinese-multi-emotion", "train/data.parquet", "多情感对话"),
    ("roleplay-zh-sharegpt", "sharegpt_formatted_data-evol-gpt4/train/data.parquet", "GPT-4角色扮演"),
    ("chat-haruhi-54k", "train/data.parquet", "小说角色对话54K"),
    ("role-play-chinese", "train/data.parquet", "中文角色扮演Alpha"),
    ("chinese-roleplay-singleturn", "train/data.parquet", "角色扮演单轮Alpaca"),
    ("beyond-dialogue", "Role_playing_Dialogue_CN/data.parquet", "小说剧本角色对话CN"),
]

data_root = Path(r"D:\RoleMatrix\data\datasets")

for name, pq_rel, desc in SAMPLES:
    pq_path = data_root / name / pq_rel
    if not pq_path.exists():
        print(f"\n=== {name} ({desc}) === NOT FOUND: {pq_path}")
        continue

    df = pd.read_parquet(pq_path)
    print(f"\n{'='*70}")
    print(f"=== {name} ({desc}) === {len(df):,} rows ===")
    print(f"Columns: {list(df.columns)}")

    if name == "chinese-multi-emotion":
        print(f"\nEmotion distribution:")
        print(df["emotion"].value_counts().to_string())

    # Show first 2 samples
    for i in range(min(2, len(df))):
        row = df.iloc[i].to_dict()
        print(f"\n--- Sample {i+1} ---")
        for k, v in row.items():
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  [{k}]: {s}")
