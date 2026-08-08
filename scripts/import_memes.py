"""表情包一键导入：把本地目录里的表情包图片导入收藏库（自动打 tag）。

用法：
  1. 把表情包图片放进一个文件夹，例如 D:\\RoleMatrix\\表情包\\
     子目录名 = 标签（如 开心/撒娇/害羞），文件会被打上对应标签；
     直接放在根目录的文件按文件名打标签（去掉扩展名）。
  2. 运行：
     python scripts/import_memes.py D:\\RoleMatrix\\表情包
  3. 之后小R 的大脑决策 send_meme 就会按 tag 从这里挑图发给你。

支持 jpg/png/gif/webp；重复图片（hash 相同）自动跳过。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rolematrix.logger import get_logger
from rolematrix.tools.collection_store import init_collection_db
from rolematrix.tools.image_downloader import save_local_file

log = get_logger("tools.import_memes")

ALLOWED = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


async def import_dir(root: Path) -> int:
    await init_collection_db()
    total = 0
    for img in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED):
        # 标签：子目录名优先，否则文件名（去扩展名）
        if img.parent != root:
            tag = img.parent.name
        else:
            tag = img.stem
        tags = [tag] if tag and tag != img.stem else ([img.stem] if img.stem else [])
        try:
            result = await save_local_file(
                src_path=str(img),
                source="meme_import",
                tags=tags,
                description=f"表情包:{tag}",
            )
            if result:
                rel, _ = result
                print(f"  已导入 {img.name}  tag={tag}  -> {rel}", flush=True)
                total += 1
            else:
                print(f"  跳过（重复或失败）: {img.name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  导入失败 {img.name}: {e}", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="表情包一键导入收藏库")
    ap.add_argument("dir", help="表情包图片目录（子目录名=标签）")
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1
    n = asyncio.run(import_dir(root))
    print(f"\n完成：导入 {n} 张表情包。之后小R 发消息时可能按标签发给你。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
