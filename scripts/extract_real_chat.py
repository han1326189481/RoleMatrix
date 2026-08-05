#!/usr/bin/env python3
"""
从真实微信聊天记录中提取 AI 伴侣训练数据

源文件：D:\WeChatMsg\11.txt  (Owe↔Angel)
        D:\WeChatMsg\22.txt  (Owe↔景行)

提取策略：
  1. 识别 Angel / 景行 的发言
  2. 构建上下文（前1-3轮对话作为 user input，她们回复作为 assistant output）
  3. 过滤掉技术指导类/纯图片/无意义短回复
  4. 保留情感表达、日常闲聊、撒娇抱怨、关心安慰等自然对话
  5. 输出为 ShareGPT 格式 JSONL，可直接用于 SFT 微调
"""

import json
import re
from pathlib import Path

# ---------- 配置 ----------
SRC_DIR = Path(r"D:\WeChatMsg")
OUT_DIR = Path(r"D:\RoleMatrix\data\datasets\real-chat-extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 发言者映射
FEMALE_IDS = {"Angel.", "Angel:", "景行:", "景行"}
MALE_IDS = {"Owe:", "Owe"}

# 需要过滤的系统消息模式
SYSTEM_PATTERNS = [
    r"^\[图片\]$",
    r"^\[动画表情\]$",
    r"^\[视频\]$",
    r"^\[语音\]$",
    r"^\[文件\]$",
    r"^\[聊天记录\]$",
    r"^\[位置\]$",
    r"^\[.*?\]$",
    r"^\d{11}",
]

# 需要跳过的行（其他发言者）
SKIP_PATTERNS = [
    r"^张妮妮",
    r"^BRO",
    r"^BRO（",
]


def is_system_msg(text: str) -> bool:
    """判断是否是系统消息/纯媒体消息"""
    text = text.strip()
    if not text:
        return True
    for pat in SYSTEM_PATTERNS:
        if re.match(pat, text):
            return True
    return False


def is_skip_line(line: str) -> bool:
    """判断是否是需要跳过的行"""
    for pat in SKIP_PATTERNS:
        if re.match(pat, line):
            return True
    return False


def clean_text(text: str) -> str:
    """清洗文本：去掉多余空格、换行"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_chat_file(filepath: Path) -> list[dict]:
    """
    解析微信聊天记录文件
    格式：每行 "发言者:\n" 或 "发言者:\n消息内容\n"
    返回：[{"speaker": "Angel", "text": "...", "raw": "..."}, ...]
    """
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_speaker = None
    current_text_parts = []

    for line in lines:
        stripped = line.rstrip("\n").rstrip("\r")

        # 检测新发言者（支持 "Name:" "Name.:" "Name." 等格式）
        speaker = None
        content_after = ""

        # 尝试匹配女性发言者
        for fid_raw in FEMALE_IDS:
            fid = fid_raw.rstrip(":").rstrip(".")
            # 匹配 "Angel:" "Angel.:" "Angel." "Angel" 等
            for suffix in [":", ".:", ".", ""]:
                prefix = fid + suffix
                if stripped == prefix:
                    speaker = fid
                    break
                if stripped.startswith(prefix) and len(stripped) > len(prefix):
                    # 同一行有内容
                    speaker = fid
                    content_after = stripped[len(prefix):].strip()
                    break
            if speaker:
                break

        if not speaker:
            for mid_raw in MALE_IDS:
                mid = mid_raw.rstrip(":").rstrip(".")
                for suffix in [":", ".:", ".", ""]:
                    prefix = mid + suffix
                    if stripped == prefix:
                        speaker = mid
                        break
                    if stripped.startswith(prefix) and len(stripped) > len(prefix):
                        speaker = mid
                        content_after = stripped[len(prefix):].strip()
                        break
                if speaker:
                    break

        if speaker:
            # 保存上一条消息
            if current_speaker and current_text_parts:
                text = "".join(current_text_parts).strip()
                if text and not is_system_msg(text):
                    messages.append({
                        "speaker": current_speaker,
                        "text": clean_text(text),
                    })

            current_speaker = speaker
            current_text_parts = [content_after] if content_after else []
            continue

        # 跳过行
        if is_skip_line(stripped):
            continue

        # 普通消息行 / 空行
        if stripped == "":
            # 空行：如果已经有内容，可能是消息结束
            if current_speaker and current_text_parts:
                pass  # 不在此处保存，等待下一个发言者
            continue

        # 累积文本
        if current_speaker:
            current_text_parts.append(stripped)

    # 保存最后一条
    if current_speaker and current_text_parts:
        text = "".join(current_text_parts).strip()
        if text and not is_system_msg(text):
            messages.append({
                "speaker": current_speaker,
                "text": clean_text(text),
            })

    return messages


def build_conversations(messages: list[dict], female_name: str, context_window: int = 3) -> list[dict]:
    """
    构建问答对：以女性回复为中心，向前回溯 1-3 轮作为上下文

    返回 ShareGPT 格式：
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    pairs = []

    for i, msg in enumerate(messages):
        if msg["speaker"] != female_name:
            continue

        # 这是女性的一条回复
        assistant_text = msg["text"]

        # 跳过太短/无意义的回复
        if len(assistant_text) < 2:
            continue
        if assistant_text in {"ok", "okk", "yes", "好", "行", "昂", "嗯", "噢噢", "okok"}:
            # 太短的确认词，但保留有语气的版本
            if len(assistant_text) <= 3:
                continue

        # 跳过纯技术相关
        tech_keywords = [
            "idea", "文件夹", "压缩包", "提交", "代码", "配置",
            "安装", "下载了", "运行", "控制台", "f12", "浏览器",
            "eureka", "注册中心", "网关", "熔断", "降级",
        ]
        if any(kw in assistant_text.lower() for kw in tech_keywords):
            if len(assistant_text) < 30:  # 长回复可能包含技术+情感
                continue

        # 向前回溯上下文
        history = []
        j = i - 1
        turn_count = 0
        while j >= 0 and turn_count < context_window * 2:
            prev = messages[j]
            if prev["speaker"] == "Owe":
                history.insert(0, prev["text"])
                turn_count += 1
            elif prev["speaker"] == female_name and turn_count > 0:
                # 女性之前的发言，作为她的上下文
                history.insert(0, f"（我之前说：{prev['text']}）")
            j -= 1

        # 拼接上下文
        if history:
            context = "\n".join(history[-context_window:])
        else:
            continue  # 没有上下文的不收

        # 过滤纯技术指导的上下文
        if is_pure_tech_context(context):
            continue

        pairs.append({
            "conversations": [
                {"from": "human", "value": context},
                {"from": "gpt", "value": assistant_text},
            ]
        })

    return pairs


def is_pure_tech_context(text: str) -> bool:
    """判断上下文是否纯技术指导（这类对话对 AI 伴侣训练价值低）"""
    tech_markers = [
        "下载", "安装", "配置", "代码", "编译", "运行",
        "服务器", "端口", "命令", "ctrl", "shift",
        "浏览器", "控制台", "断点", "调试",
        "截图给我", "远程", "登录你的",
        "学号", "账号", "密码",
    ]
    score = sum(1 for m in tech_markers if m in text)
    return score >= 3  # 3个及以上技术标记 = 纯技术对话


def extract_dataset(filepath: Path, female_name: str, label: str) -> tuple[list[dict], dict]:
    """从单个文件提取数据集"""
    print(f"\n{'='*60}")
    print(f"📖 解析: {filepath.name} → 女性角色: {female_name}")
    print(f"{'='*60}")

    messages = parse_chat_file(filepath)
    print(f"   总消息数: {len(messages)}")

    # 统计发言
    from collections import Counter
    speaker_counts = Counter(m["speaker"] for m in messages)
    print(f"   发言分布: {dict(speaker_counts)}")

    # 女性发言数
    female_msgs = [m for m in messages if m["speaker"] == female_name]
    print(f"   女性发言数: {len(female_msgs)}")

    # 构建问答对
    pairs = build_conversations(messages, female_name, context_window=3)
    print(f"   提取问答对: {len(pairs)}")

    return pairs, {
        "source_file": str(filepath),
        "female_name": female_name,
        "label": label,
        "total_messages": len(messages),
        "female_messages": len(female_msgs),
        "extracted_pairs": len(pairs),
    }


def main():
    # ---- Angel 数据集 ----
    angel_pairs, angel_meta = extract_dataset(
        SRC_DIR / "11.txt",
        female_name="Angel",
        label="angel-casual"
    )

    # ---- 景行 数据集 ----
    jingxing_pairs, jingxing_meta = extract_dataset(
        SRC_DIR / "22.txt",
        female_name="景行",
        label="jingxing-warm"
    )

    # ---- 保存 ----
    all_pairs = angel_pairs + jingxing_pairs

    # JSONL 格式（ShareGPT，可直接用于 SFT）
    jsonl_path = OUT_DIR / "real_chat_sharegpt.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\n✅ JSONL 已保存: {jsonl_path} ({len(all_pairs):,} 条)")

    # JSON 数组格式（方便查看）
    json_path = OUT_DIR / "real_chat_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 已保存: {json_path}")

    # 保存元信息
    meta = {
        "description": "从真实微信聊天记录提取的 AI 伴侣训练数据",
        "sources": [
            {"file": "11.txt", "female": "Angel", "style": "随性自然、小傲娇、偶尔撒娇"},
            {"file": "22.txt", "female": "景行", "style": "温暖鼓励、爱笑、善解人意"},
        ],
        "format": "sharegpt",
        "statistics": {
            "angel_pairs": angel_meta,
            "jingxing_pairs": jingxing_meta,
            "total_pairs": len(all_pairs),
        },
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 打印统计 ----
    print(f"\n{'='*60}")
    print(f"📊 提取统计")
    print(f"{'='*60}")
    print(f"  Angel 问答对:  {angel_meta['extracted_pairs']:>5,} 条")
    print(f"  景行 问答对:   {jingxing_meta['extracted_pairs']:>5,} 条")
    print(f"  总计:          {len(all_pairs):>5,} 条")
    print(f"\n📁 输出目录: {OUT_DIR}")

    # ---- 风格分析 ----
    print(f"\n{'='*60}")
    print(f"🔬 两个女性角色风格差异")
    print(f"{'='*60}")

    # Angel 特征词
    angel_texts = [p["conversations"][1]["value"] for p in angel_pairs]
    angel_all = " ".join(angel_texts)
    angel_features = {
        "好嘟": angel_all.count("好嘟"),
        "okk": angel_all.count("okk"),
        "噗哈哈哈": angel_all.count("噗哈哈哈"),
        "我去": angel_all.count("我去"),
        "我滴妈": angel_all.count("我滴妈"),
        "yes": angel_all.count("yes"),
        "韩哥": angel_all.count("韩哥"),
        "妈呀": angel_all.count("妈呀"),
        "美呀": angel_all.count("美呀"),
    }
    print(f"\n  Angel 口头禅:")
    for k, v in sorted(angel_features.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"    {k}: {v}次")

    # 景行 特征词
    jx_texts = [p["conversations"][1]["value"] for p in jingxing_pairs]
    jx_all = " ".join(jx_texts)
    jx_features = {
        "哈哈哈哈": jx_all.count("哈哈哈哈"),
        "bro": jx_all.count("bro"),
        "我去我去": jx_all.count("我去我去"),
        "蛙趣": jx_all.count("蛙趣"),
        "嘿嘿": jx_all.count("嘿嘿"),
        "好贴心": jx_all.count("好贴心"),
        "～": jx_all.count("～"),
        "好耶": jx_all.count("好耶"),
        "那咋啦": jx_all.count("那咋啦"),
        "好滴": jx_all.count("好滴"),
        "感动": jx_all.count("感动"),
    }
    print(f"\n  景行 口头禅:")
    for k, v in sorted(jx_features.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"    {k}: {v}次")

    # 打印样本
    print(f"\n{'='*60}")
    print(f"📝 样本预览 (Angel)")
    print(f"{'='*60}")
    for i, pair in enumerate(angel_pairs[:3]):
        print(f"\n--- Angel Sample {i+1} ---")
        print(f"  User: {pair['conversations'][0]['value'][:150]}")
        print(f"  Angel: {pair['conversations'][1]['value'][:150]}")

    print(f"\n{'='*60}")
    print(f"📝 样本预览 (景行)")
    print(f"{'='*60}")
    for i, pair in enumerate(jingxing_pairs[:3]):
        print(f"\n--- 景行 Sample {i+1} ---")
        print(f"  User: {pair['conversations'][0]['value'][:150]}")
        print(f"  景行: {pair['conversations'][1]['value'][:150]}")


if __name__ == "__main__":
    main()
