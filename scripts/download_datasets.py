#!/usr/bin/env python3
"""
从 HuggingFace 下载中文 AI 伴侣 / 角色扮演数据集

数据集分级：
  Tier 1 — 核心 AI 伴侣数据集（高情商对话、角色扮演 ShareGPT）
  Tier 2 — 补充角色扮演 / 自然对话
  Tier 3 — 大规模可选数据集

用法：
  python scripts/download_datasets.py          # 下载 Tier 1
  python scripts/download_datasets.py --all    # 下载全部
  python scripts/download_datasets.py --tier 2 # 下载 Tier 1+2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------- 配置 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets"
CACHE_DIR = ROOT / ".tmp" / "hf_cache"

DATASETS = {
    # ==================== Tier 1: 核心 AI 伴侣 ====================
    "chinese-adorable-high-eq": {
        "name": "MemorialSummer/chinese-adorable-high-emotional-intelligence-chat",
        "tier": 1,
        "desc": "中文高情商可爱聊天 — AI 女友风格对话 (~170组)",
        "format": "json",
        "save_as": "parquet",
    },
    "roleplay-zh-sharegpt": {
        "name": "shibing624/roleplay-zh-sharegpt-gpt4-data",
        "tier": 1,
        "desc": "中文角色扮演 ShareGPT 格式 — GPT-4 生成多轮对话（4个子集）",
        "format": "sharegpt",
        "save_as": "parquet",
        "configs": [
            "sharegpt_formatted_data-evol-gpt4",
            "sharegpt_formatted_data-evol-gpt35",
            "sharegpt_formatted_data-evol-male-gpt35",
            "sharegpt_formatted_data-roleplay-chat-1k",
        ],
    },
    "chinese-multi-emotion": {
        "name": "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset",
        "tier": 1,
        "desc": "中文多情感对话数据集 — 带8类情感标签",
        "format": "json",
        "save_as": "parquet",
    },

    # ==================== Tier 2: 补充角色扮演 ====================
    "chat-haruhi-54k": {
        "name": "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
        "tier": 2,
        "desc": "春日野54K角色扮演对话 — 小说角色多轮对话",
        "format": "json",
        "save_as": "parquet",
    },
    "slim-lccc-zh": {
        "name": "lorinma/Slim-LCCC-zh",
        "tier": 2,
        "desc": "精选中文自然对话 — 从LCCC 1200万条中精选1万条 ShareGPT 格式",
        "format": "sharegpt",
        "save_as": "parquet",
    },
    "role-play-chinese": {
        "name": "Johnson8187/role-play-chinese",
        "tier": 2,
        "desc": "中文角色扮演 Alpha 格式 — AI生成多场景角色对话",
        "format": "alpha",
        "save_as": "parquet",
    },

    # ==================== Tier 3: 大规模 / 深度 ====================
    "coser": {
        "name": "Neph0s/CoSER",
        "tier": 3,
        "desc": "CoSER — 最大真实角色扮演数据集 (771部文学著作, ~30K对话)",
        "format": "json",
        "save_as": "parquet",
    },
    "chinese-roleplay-singleturn": {
        "name": "LooksJuicy/Chinese-Roleplay-SingleTurn",
        "tier": 3,
        "desc": "中文角色扮演单轮对话 — Alpaca 格式，适合 LoRA 微调",
        "format": "alpaca",
        "save_as": "parquet",
    },
    "beyond-dialogue": {
        "name": "yuyouyu/BeyondDialogue",
        "tier": 3,
        "desc": "BeyondDialogue — 中英小说剧本角色对话 (~3.5K sessions)",
        "format": "json",
        "save_as": "parquet",
    },
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(key: str, cfg: dict) -> bool:
    """下载单个数据集并保存为 Parquet。
    支持多 config 的数据集（如 shibing624/roleplay-zh-sharegpt-gpt4-data）。
    """
    from datasets import load_dataset

    out_path = DATA_DIR / key
    out_path.mkdir(parents=True, exist_ok=True)

    configs = cfg.get("configs", [None])

    print(f"\n{'='*60}")
    print(f"📥 下载: {key}")
    print(f"   {cfg['desc']}")
    print(f"   HuggingFace: {cfg['name']}")
    if len(configs) > 1:
        print(f"   子集数: {len(configs)}")
    print(f"{'='*60}")

    grand_total = 0
    all_success = True

    for config_name in configs:
        label = config_name or "(default)"
        try:
            if config_name:
                ds = load_dataset(cfg["name"], config_name, cache_dir=str(CACHE_DIR))
            else:
                ds = load_dataset(cfg["name"], cache_dir=str(CACHE_DIR))
        except Exception as e:
            # 如果是 gated dataset，给出明确提示
            err_msg = str(e)
            if "gated" in err_msg.lower() or "authenticated" in err_msg.lower():
                print(f"   🔒 {label}: 需要认证（gated dataset），请设置 HF_TOKEN 环境变量后重试")
                print(f"       访问 https://huggingface.co/{cfg['name']} 申请授权")
            else:
                print(f"   ❌ {label}: {e}")
            all_success = False
            continue

        # DatasetDict（多 split）
        if hasattr(ds, "keys"):
            sub_total = 0
            for split_name, split_ds in ds.items():
                # 有 config 时放子目录
                if config_name:
                    split_dir = out_path / config_name / split_name
                else:
                    split_dir = out_path / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                pq_path = split_dir / "data.parquet"
                split_ds.to_parquet(str(pq_path))
                rows = len(split_ds)
                sub_total += rows
                grand_total += rows
                print(f"   ✅ [{config_name}] {split_name}: {rows:,} 条 → {pq_path}")
            print(f"   📊 [{config_name}] 小计: {sub_total:,} 条")
        else:
            if config_name:
                cfg_dir = out_path / config_name
                cfg_dir.mkdir(parents=True, exist_ok=True)
                pq_path = cfg_dir / "data.parquet"
            else:
                pq_path = out_path / "data.parquet"
            ds.to_parquet(str(pq_path))
            rows = len(ds)
            grand_total += rows
            print(f"   ✅ [{config_name}]: {rows:,} 条 → {pq_path}")

    if len(configs) > 1:
        print(f"   📊 总计: {grand_total:,} 条")

    # 保存元信息
    meta = {
        "key": key,
        "hf_name": cfg["name"],
        "tier": cfg["tier"],
        "description": cfg["desc"],
        "format": cfg["format"],
        "configs": configs if len(configs) > 1 else None,
    }
    with open(out_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return all_success


def main() -> None:
    parser = argparse.ArgumentParser(description="下载中文 AI 伴侣数据集")
    parser.add_argument("--all", action="store_true", help="下载全部数据集 (Tier 1-3)")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="下载到第几层级 (默认: 1)")
    parser.add_argument("--dry-run", action="store_true", help="只列出，不下载")
    parser.add_argument("--only", type=str, nargs="+", metavar="KEY",
                        help="只下载指定数据集 (用 key 名，如 roleplay-zh-sharegpt)")
    args = parser.parse_args()

    ensure_dirs()

    # 筛选
    if args.only:
        selected = {k: v for k, v in DATASETS.items() if k in args.only}
        not_found = set(args.only) - set(selected.keys())
        if not_found:
            print(f"❌ 未找到: {not_found}")
            print(f"   可用 key: {', '.join(DATASETS.keys())}")
            sys.exit(1)
    else:
        max_tier = 3 if args.all else args.tier
        selected = {k: v for k, v in DATASETS.items() if v["tier"] <= max_tier}

    print(f"\n📋 计划下载 {len(selected)} 个数据集:\n")
    for key, cfg in sorted(selected.items(), key=lambda x: x[1]["tier"]):
        print(f"   [T{cfg['tier']}] {key:35s} — {cfg['desc']}")

    if args.dry_run:
        print(f"\n🔍 --dry-run 模式，不实际下载。")
        print(f"   目标目录: {DATA_DIR}")
        return

    # 确认
    print(f"\n目标目录: {DATA_DIR}")
    print(f"缓存目录: {CACHE_DIR}\n")

    # 下载
    success, fail = 0, 0
    for key in sorted(selected.keys(), key=lambda k: selected[k]["tier"]):
        ok = download_dataset(key, selected[key])
        if ok:
            success += 1
        else:
            fail += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"🎉 下载完成: {success} 成功 / {fail} 失败 / {len(selected)} 总计")
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
