#!/usr/bin/env python3
"""
从真实微信聊天记录提取 AI 伴侣训练数据 — 优化版

改进：
  1. 基于文件名正确分类（11.txt→Angel, 22.txt→景行）
  2. 构建自然的完整多轮对话段
  3. 智能过滤：保留情感/日常，过滤纯技术指导
  4. 移除"我之前说"标记，使用真实对话历史
  5. 同时输出单轮和多轮版本
"""

import json
import re
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(r"D:\WeChatMsg")
OUT_DIR = Path(r"D:\RoleMatrix\data\datasets\real-chat-extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 需要过滤的系统消息
RE_SYSTEM = re.compile(
    r"^\[图片\]$|^\[动画表情\]$|^\[视频\]$|^\[语音\]$|^\[文件\]$|"
    r"^\[聊天记录\]$|^\[位置\]$|^\[转账\].*$|^\[.*?\]$"
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip()


def parse_file(filepath: Path, female_name: str) -> list[dict]:
    """
    解析聊天记录，返回消息列表。
    格式: {"role": "male"|"female", "text": "..."}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    messages = []
    # 匹配发言者行（支持 "Name:" "Name.:" "Name."）
    # 后跟消息内容（可能跨多行，直到下一个发言者或空行分隔）

    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r")

        # 检测发言者
        role = None
        speaker_base = None

        # 匹配 Owe
        m = re.match(r"^Owe[:.]*\s*(.*)", line)
        if m:
            role = "male"
            speaker_base = "Owe"
            content_start = m.group(1).strip()

        # 匹配 Angel
        if not role:
            m = re.match(r"^Angel\.?:?\s*(.*)", line)
            if m:
                role = "female"
                speaker_base = female_name
                content_start = m.group(1).strip()

        # 匹配 景行
        if not role:
            m = re.match(r"^景行[:.]?\s*(.*)", line)
            if m:
                role = "female"
                speaker_base = female_name
                content_start = m.group(1).strip()

        # 跳过其他人
        if not role:
            if re.match(r"^(张妮妮|BRO|BRO（)", line):
                i += 1
                while i < len(lines) and lines[i].strip():
                    i += 1
                continue
            i += 1
            continue

        # 收集消息内容
        parts = [content_start] if content_start else []
        i += 1
        while i < len(lines):
            next_line = lines[i].rstrip("\r")
            # 空行 = 消息结束
            if next_line.strip() == "":
                i += 1
                break
            # 新发言者 = 消息结束
            if re.match(r"^(Owe|Angel|景行|张妮妮|BRO)", next_line):
                break
            parts.append(next_line)
            i += 1

        text = clean("".join(parts))
        if text and not RE_SYSTEM.match(text):
            messages.append({"role": role, "text": text})

    return messages


def is_tech_dialogue(user_text: str, assistant_text: str) -> bool:
    """判断是否为纯技术指导对话（对AI伴侣训练无价值）"""
    tech_keywords = [
        "下载", "安装", "配置", "编译", "运行", "服务器", "端口",
        "命令", "ctrl", "shift", "控制台", "f12", "断点", "调试",
        "远程", "登录你的", "账号", "密码", "学号", "2023002",
        "提交", "压缩包", "文件夹放", "idea", "代码", "sql",
        "eureka", "注册中心", "网关", "熔断", "降级", "截屏",
        "192.168", "10.10.11", "phpstudy", "小皮", "todesk",
        "远程操控", "远程控制", "连接", "断了", "trae",
        "网页", "网站", "上传", "upload", "浏览器",
        "演示", "文档", "报告", "模板", "word",
        "截图给我", "图片加载", "批阅明细",
    ]
    score = sum(1 for kw in tech_keywords if kw in user_text.lower() or kw in assistant_text.lower())
    return score >= 2


def is_good_dialogue(user_text: str, assistant_text: str) -> bool:
    """判断是否为有价值的对话（情感/日常/闲聊）"""
    # 太短
    if len(assistant_text) < 2:
        return False

    # 纯英文短回复
    if assistant_text.lower() in {"yes", "no", "ok", "okk", "yeah", "then", "okok", "omk"}:
        return False

    # 纯确认（但不包括有语气的版本）
    if assistant_text in {"好", "行", "昂", "嗯", "对", "11", "1", "ok", "好的"}:
        return False

    # 纯技术
    if is_tech_dialogue(user_text, assistant_text):
        return False

    # 只包含emoji/特殊字符
    if re.match(r"^[\U0001F000-\U0001FFFF🎉👍❤️🥰😊😂😭💕✨🙏🐮🌹🍜]*$", assistant_text):
        return False

    return True


def build_conversation_segments(messages: list[dict], max_gap: int = 5) -> list[list[dict]]:
    """
    将消息按话题分割为对话段。
    max_gap: 同一段内允许的最大时间/消息间隔（简化：用消息数代替）
    """
    segments = []
    current_seg = []
    male_msg_count = 0
    female_msg_count = 0

    for msg in messages:
        current_seg.append(msg)
        if msg["role"] == "male":
            male_msg_count += 1
        else:
            female_msg_count += 1

        # 每积累到一定数量的男女对话，切分一段
        if male_msg_count >= 4 and female_msg_count >= 2:
            if len(current_seg) >= 4:
                segments.append(current_seg)
            current_seg = []
            male_msg_count = 0
            female_msg_count = 0

    if len(current_seg) >= 4:
        segments.append(current_seg)

    return segments


def extract_singleturn_pairs(messages: list[dict]) -> list[dict]:
    """
    提取单轮对话：用前1-3句男性发言作为context，
    女性的一句回复作为assistant。
    """
    pairs = []
    for i, msg in enumerate(messages):
        if msg["role"] != "female":
            continue

        assistant = msg["text"]
        # 收集前面的男性发言（回溯，最多3句）
        context_parts = []
        j = i - 1
        collected = 0
        while j >= 0 and collected < 3:
            prev = messages[j]
            if prev["role"] == "male":
                context_parts.insert(0, prev["text"])
                collected += 1
            elif prev["role"] == "female" and collected > 0:
                # 跳过女性自己的发言
                pass
            j -= 1

        if not context_parts:
            continue

        user_text = "\n".join(context_parts)

        if not is_good_dialogue(user_text, assistant):
            continue

        pairs.append({
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": assistant},
            ]
        })

    return pairs


def extract_multiturn_dialogues(messages: list[dict], min_turns: int = 3) -> list[dict]:
    """
    提取多轮对话：构建完整的多轮对话历史。
    选择以女性角色为主视角的对话片段。
    """
    segments = build_conversation_segments(messages)
    dialogues = []

    for seg in segments:
        # 过滤掉纯技术的段
        female_texts = [m["text"] for m in seg if m["role"] == "female"]
        male_texts = [m["text"] for m in seg if m["role"] == "male"]
        all_text = " ".join(female_texts + male_texts)
        tech_score = sum(1 for kw in [
            "下载", "安装", "配置", "代码", "运行", "服务器", "端口",
            "浏览器", "控制台", "eureka", "网关", "学号", "账号",
        ] if kw in all_text)
        if tech_score >= 3:
            continue

        # 女性至少发言 2 次
        if len(female_texts) < 2:
            continue

        # 构建 ShareGPT conversations
        conversations = []
        for m in seg:
            if m["role"] == "male":
                conversations.append({"from": "human", "value": m["text"]})
            else:
                # 过滤太短的
                if len(m["text"]) >= 2:
                    conversations.append({"from": "gpt", "value": m["text"]})

        if len(conversations) >= min_turns * 2:
            dialogues.append({"conversations": conversations})

    return dialogues


def style_analysis(messages: list[dict], name: str) -> dict:
    """分析女性角色的语言风格"""
    female_texts = [m["text"] for m in messages if m["role"] == "female"]
    all_text = " ".join(female_texts)

    features = {
        "总发言数": len(female_texts),
        "平均长度": round(sum(len(t) for t in female_texts) / max(len(female_texts), 1), 1),
        "最长发言": max(female_texts, key=len) if female_texts else "",
    }

    # 口头禅
    catchphrases = [
        "好嘟", "okk", "噗哈哈哈", "我去", "我滴妈", "妈呀", "yes",
        "韩哥", "美呀", "哈哈哈哈", "bro", "我去我去", "蛙趣",
        "嘿嘿", "好贴心", "好耶", "那咋啦", "好滴", "感动",
        "真的假的", "没事", "噗", "～", "过时间了", "俺",
    ]
    phrase_counts = {p: all_text.count(p) for p in catchphrases if all_text.count(p) > 0}
    features["口头禅"] = dict(sorted(phrase_counts.items(), key=lambda x: -x[1]))

    return features


def main():
    # ---- 解析两个文件 ----
    print("📖 解析聊天记录...\n")

    angel_msgs = parse_file(SRC_DIR / "11.txt", "Angel")
    jx_msgs = parse_file(SRC_DIR / "22.txt", "景行")

    print(f"  Angel 对话: {len(angel_msgs)} 条消息 (男: {sum(1 for m in angel_msgs if m['role']=='male')}, 女: {sum(1 for m in angel_msgs if m['role']=='female')})")
    print(f"  景行 对话:  {len(jx_msgs)} 条消息 (男: {sum(1 for m in jx_msgs if m['role']=='male')}, 女: {sum(1 for m in jx_msgs if m['role']=='female')})")

    # ---- 提取单轮问答 ----
    print("\n🔧 提取对话...")
    angel_single = extract_singleturn_pairs(angel_msgs)
    jx_single = extract_singleturn_pairs(jx_msgs)
    print(f"  Angel 单轮: {len(angel_single)} 条")
    print(f"  景行 单轮:  {len(jx_single)} 条")

    # ---- 提取多轮对话 ----
    angel_multi = extract_multiturn_dialogues(angel_msgs)
    jx_multi = extract_multiturn_dialogues(jx_msgs)
    print(f"  Angel 多轮: {len(angel_multi)} 段")
    print(f"  景行 多轮:  {len(jx_multi)} 段")

    # ---- 合并保存 ----
    # 单轮 (ShareGPT JSONL)
    all_single = angel_single + jx_single
    with open(OUT_DIR / "singleturn.jsonl", "w", encoding="utf-8") as f:
        for p in all_single:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 多轮 (ShareGPT JSONL)
    all_multi = angel_multi + jx_multi
    with open(OUT_DIR / "multiturn.jsonl", "w", encoding="utf-8") as f:
        for d in all_multi:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 单轮 + 多轮合并
    all_combined = all_single + all_multi
    with open(OUT_DIR / "combined.jsonl", "w", encoding="utf-8") as f:
        for item in all_combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 完整 JSON 方便查看
    with open(OUT_DIR / "singleturn.json", "w", encoding="utf-8") as f:
        json.dump(all_single, f, ensure_ascii=False, indent=2)

    # ---- 风格分析 ----
    angel_style = style_analysis(angel_msgs, "Angel")
    jx_style = style_analysis(jx_msgs, "景行")

    # ---- 元信息 ----
    meta = {
        "description": "从真实微信聊天记录提取的 AI 伴侣训练数据",
        "sources": [
            {
                "file": "11.txt",
                "name": "Angel",
                "style": "随性自然、小撒娇、用'韩哥'称呼、偶尔小傲娇、喜欢被夸照片",
                "catchphrases": list(angel_style["口头禅"].keys())[:10],
            },
            {
                "file": "22.txt",
                "name": "景行",
                "style": "温暖开朗、爱笑、善解人意、会鼓励人、暖心治愈、meme表情包爱好者",
                "catchphrases": list(jx_style["口头禅"].keys())[:10],
            },
        ],
        "format": "sharegpt",
        "statistics": {
            "angel": {
                "total_messages": len(angel_msgs),
                "female_messages": angel_style["总发言数"],
                "singleturn_pairs": len(angel_single),
                "multiturn_segments": len(angel_multi),
            },
            "jingxing": {
                "total_messages": len(jx_msgs),
                "female_messages": jx_style["总发言数"],
                "singleturn_pairs": len(jx_single),
                "multiturn_segments": len(jx_multi),
            },
            "total": {
                "singleturn": len(all_single),
                "multiturn": len(all_multi),
                "combined": len(all_combined),
            },
        },
        "style_analysis": {
            "angel": angel_style,
            "jingxing": jx_style,
        },
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 打印总结 ----
    print(f"\n{'='*60}")
    print(f"📊 提取完成!")
    print(f"{'='*60}")
    print(f"  单轮问答: {len(all_single):>5} 条  → singleturn.jsonl")
    print(f"  多轮对话: {len(all_multi):>5} 段  → multiturn.jsonl")
    print(f"  合并总计: {len(all_combined):>5} 条  → combined.jsonl")
    print(f"\n📁 输出目录: {OUT_DIR}")

    # Angel 风格
    print(f"\n{'='*60}")
    print(f"🔬 Angel 语言风格分析")
    print(f"{'='*60}")
    print(f"  发言数: {angel_style['总发言数']}, 平均长度: {angel_style['平均长度']}字")
    print(f"  口头禅: {', '.join(f'{k}({v})' for k,v in list(angel_style['口头禅'].items())[:12])}")

    # 景行 风格
    print(f"\n{'='*60}")
    print(f"🔬 景行 语言风格分析")
    print(f"{'='*60}")
    print(f"  发言数: {jx_style['总发言数']}, 平均长度: {jx_style['平均长度']}字")
    print(f"  口头禅: {', '.join(f'{k}({v})' for k,v in list(jx_style['口头禅'].items())[:12])}")

    # ---- 样本展示 ----
    print(f"\n{'='*60}")
    print(f"📝 Angel 单轮样本")
    print(f"{'='*60}")
    for i, p in enumerate(angel_single[:5]):
        conv = p["conversations"]
        print(f"\n  [{i+1}] 💬 {conv[0]['value'][:120]}")
        print(f"       💗 {conv[1]['value'][:150]}")

    print(f"\n{'='*60}")
    print(f"📝 景行 单轮样本")
    print(f"{'='*60}")
    for i, p in enumerate(jx_single[:5]):
        conv = p["conversations"]
        print(f"\n  [{i+1}] 💬 {conv[0]['value'][:120]}")
        print(f"       💗 {conv[1]['value'][:150]}")

    print(f"\n{'='*60}")
    print(f"📝 Angel 多轮样本")
    print(f"{'='*60}")
    for i, d in enumerate(angel_multi[:2]):
        print(f"\n  --- 对话段 {i+1} ({len(d['conversations'])}轮) ---")
        for turn in d["conversations"][:6]:
            role = "💬" if turn["from"] == "human" else "💗"
            print(f"  {role} {turn['value'][:130]}")

    print(f"\n{'='*60}")
    print(f"📝 景行 多轮样本")
    print(f"{'='*60}")
    for i, d in enumerate(jx_multi[:2]):
        print(f"\n  --- 对话段 {i+1} ({len(d['conversations'])}轮) ---")
        for turn in d["conversations"][:6]:
            role = "💬" if turn["from"] == "human" else "💗"
            print(f"  {role} {turn['value'][:130]}")


if __name__ == "__main__":
    main()
