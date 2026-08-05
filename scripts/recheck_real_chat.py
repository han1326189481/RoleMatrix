"""严格复审 real-chat 清洗后的 88 条样本，找出漏网的课程/作业对话。

发现问题：之前 style_score >=2 通过的样本里，还有"上节课让做的哪部分呀"这种。
原因：风格分主要看撒娇词，但学校话题里也可能含"啦/呢"等语气词。

加强策略：
1. 加入学校场景黑名单（课程、作业、考试等命中即丢弃）
2. 复审 88 条，看还有多少漏网
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# 学校场景黑名单（命中即丢弃，不论风格分）
SCHOOL_BLACKLIST = [
    r"上节课|下节课|这节课|那节课",
    r"课件|课|考试|复习|学分|学号|老师|同学",
    r"作业|题目|答案|做出来|做完了|做完没",
    r"登录|登陆|账号|密码",
    r"选修|必修|绩点|成绩|挂科",
    r"实验|报告|论文|答辩",
]


def is_school_scene(text: str) -> bool:
    """判断是否为学校场景"""
    for pat in SCHOOL_BLACKLIST:
        if re.search(pat, text):
            return True
    return False


def main() -> None:
    src = Path(r"D:\RoleMatrix\data\datasets\cleaned\real_chat_clean_sharegpt.jsonl")
    lines = src.read_text(encoding="utf-8").splitlines()
    print(f"原清洗后: {len(lines)} 条\n")

    leaked = []
    clean = []

    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        conv = r["conversations"]
        user_msg = conv[0]["value"]
        bot_msg = conv[1]["value"] if len(conv) > 1 else ""

        if is_school_scene(user_msg) or is_school_scene(bot_msg):
            leaked.append(r)
        else:
            clean.append(r)

    print(f"=== 漏网学校场景: {len(leaked)} 条 ===")
    for r in leaked[:10]:
        conv = r["conversations"]
        u = conv[0]["value"][:60].replace("\n", " ")
        b = conv[1]["value"][:60].replace("\n", " ") if len(conv) > 1 else "(无)"
        print(f"  用户: {u}")
        print(f"  小R: {b}")
        print()

    print(f"\n=== 复审通过: {len(clean)} 条 ===")

    # 重新输出
    out_path = src
    with out_path.open("w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已覆写: {out_path} ({len(clean)} 条)")


if __name__ == "__main__":
    main()
