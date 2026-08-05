"""验证已下载的数据集"""
import json
from pathlib import Path

import pandas as pd

data_root = Path(r"D:\RoleMatrix\data\datasets")
total_rows = 0
total_size = 0

print(f"{'Dataset':<38} {'Rows':>10} {'Size':>12}  Format")
print("-" * 76)

for d in sorted(data_root.iterdir()):
    if not d.is_dir():
        continue
    rows = 0
    size = 0
    for pq in d.rglob("*.parquet"):
        size += pq.stat().st_size
        try:
            rows += len(pd.read_parquet(pq))
        except Exception:
            pass

    meta_path = d / "meta.json"
    fmt = ""
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        fmt = meta.get("format", "")

    total_rows += rows
    total_size += size
    size_mb = size / 1024 / 1024
    print(f"{d.name:<38} {rows:>10,} {size_mb:>9.1f} MB  {fmt}")

print("-" * 76)
print(f"{'TOTAL':<38} {total_rows:>10,} {total_size/1024/1024:>9.1f} MB")
