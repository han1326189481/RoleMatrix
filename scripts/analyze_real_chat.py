"""分析 real-chat-extracted 数据集的真实构成。

摸清：技术对话占比、元标注污染占比、可用情感闲聊占比。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

SRC = Path(r"D:\RoleMatrix\data\datasets\real-chat-extracted\real_chat_sharegpt.jsonl")

# 技术对话特征词（出现即判定为技术）
TECH_PATTERNS = [
    r"\d+\.\d+\.\d+\.\d+",          # IP 地址
    r"192\.168",                      # 内网 IP
    r"localhost",
    r"https?://",
    r"upload|上传",
    r"截图",
    r"\.txt|\.php|\.js|\.py|\.exe|\.bat|\.cmd",
    r"const |var |function |class |import |def ",
    r"flag|ctf",
    r"按钮",
    r"网页",
    r"api/",
    r"mysql|sql|database",
    r"浏览器",
    r"模拟器",
    r"桌面",        # 桌面操作
    r"文件",
    r"鼠标",
    r"软件",
    r"安装",
    r"配置",
]

# 元标注污染（提取脚本加的，不是真实对话）
# 注意：中文括号需转义为字面量，否则 ）会被当成正则分组结束
META_PATTERN = r"（我之前说[：:]"

# 撒娇/傲娇/情感风格关键词（小R人设匹配）
SPOILED_PATTERNS = [
    r"哼", r"嘛", r"呀", r"呢", r"啦", r"哦", r"呜",
    r"嘻嘻", r"哈哈", r"嘿嘿", r"呵呵",
    r"不嘛", r"就要", r"讨厌", r"笨蛋",
    r"人家", r"俺",
    r"睡", r"梦", r"想",
    r"喜欢", r"开心", r"难过", r"生气",
    r"吃饭", r"无聊", r"累",
    r"陪我", r"不理",
    r"韩哥|哥哥|宝",
    r"～|~",
    r"！{2,}",       # 多个感叹号
    r"？{2,}",       # 多个问号
]


def classify(conv: list[dict]) -> str:
    """对单条对话分类：tech / meta / clean / spoiled"""
    user_msg = conv[0]["value"] if conv else ""
    bot_msg = conv[1]["value"] if len(conv) > 1 else ""
    full = user_msg + " " + bot_msg

    # 元标注污染
    if re.search(META_PATTERN, user_msg):
        return "meta"

    # 技术对话
    for pat in TECH_PATTERNS:
        if re.search(pat, full, re.I):
            return "tech"

    # 判断是否含撒娇/傲娇/情感词
    spoiled_hit = 0
    for pat in SPOILED_PATTERNS:
        if re.search(pat, full):
            spoiled_hit += 1

    if spoiled_hit >= 1:
        return "spoiled"
    return "clean"


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    buckets: dict[str, list[dict]] = {
        "tech": [], "meta": [], "spoiled": [], "clean": []
    }
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        conv = d.get("conversations", [])
        if len(conv) < 2:
            continue
        cat = classify(conv)
        buckets[cat].append(d)

    total = sum(len(v) for v in buckets.values())
    print(f"=== 数据构成分析 ===")
    print(f"总数: {total}")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)} ({len(v)*100//total}%)")

    # 各类样本展示
    for cat in ["tech", "meta", "spoiled", "clean"]:
        print(f"\n=== {cat} 样本5条 ===")
        for d in buckets[cat][:5]:
            conv = d["conversations"]
            u = conv[0]["value"][:60].replace("\n", " ")
            b = conv[1]["value"][:60].replace("\n", " ") if len(conv) > 1 else "(无)"
            print(f"  用户: {u}")
            print(f"  小R: {b}")
            print()

    # spoiled 类的回复长度分布
    if buckets["spoiled"]:
        lengths = [len(d["conversations"][1]["value"]) for d in buckets["spoiled"]
                   if len(d["conversations"]) > 1]
        print(f"=== spoiled 回复长度 ===")
        print(f"  平均: {sum(lengths)/len(lengths):.1f} 字")
        print(f"  最短: {min(lengths)} 字")
        print(f"  最长: {max(lengths)} 字")
        print(f"  分布: <=5字:{sum(1 for l in lengths if l<=5)}, "
              f"6-10字:{sum(1 for l in lengths if 6<=l<=10)}, "
              f"11-20字:{sum(1 for l in lengths if 11<=l<=20)}, "
              f">20字:{sum(1 for l in lengths if l>20)}")


if __name__ == "__main__":
    main()
