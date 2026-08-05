"""从原始 11.txt / 22.txt 提取真实对话片段。

小R人设：计算机系女大学生，技术不太好，经常需要依靠用户（韩哥）帮忙。
- 学校生活、课程、作业、考试等：完全保留（增强真实感）
- 简单技术操作（桌面/浏览器/网页/截图/文件等）：保留（塑造技术不好的真实形象）
- 大段代码、纯专业架构术语：过滤（不适合对话训练）

输出 ShareGPT 格式：human=Owe(用户/韩哥), gpt=女生(小R原型)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC_DIR = Path(r"D:\RoleMatrix\data\datasets")
FILES = {
    "11.txt": {"male": "Owe", "female": "Angel.", "female_name": "Angel"},
    "22.txt": {"male": "Owe", "female": "景行", "female_name": "景行"},
}
DST = Path(r"D:\RoleMatrix\data\datasets\cleaned\real_chat_from_raw.jsonl")
DST.parent.mkdir(parents=True, exist_ok=True)

# 纯资源消息（图片/表情/语音/文件等，无文字）
PURE_MEDIA_RE = re.compile(
    r"^(\[图片\]|\[动画表情\]|\[视频\]|\[文件\]|\[语音\]|\[聊天记录\]|"
    r"\[位置\]|\[转账\].*|\[呆\]|\[失意\]|\[嘿嘿\]|\[好\]|\[呜呜呜\]|"
    r"\[给你花\]|\[花花\]|\[吃饭饭\]|\[谢谢\]|\[浇水\]|\[在听\]|"
    r"\[疑惑\]|\[哭\]|\[图片\])$"
)

# 硬核技术内容（只过滤这些：大段代码 + 纯专业架构/安全术语）
# 学校生活、简单技术操作词全部保留
TECH_HARD_ONLY = [
    # 多行编程代码特征
    r"const\s+\w+\s*=",
    r"var\s+\w+\s*=",
    r"function\s+\w+\s*\(",
    r"async\s+function",
    r"\.then\s*\(",
    r"console\.log\s*\(",
    r"fetch\s*\(",
    r"await\s+fetch",
    r"forEach\s*\(",
    r"String\.fromCharCode",
    r"substr\s*\(",
    r"ascii\s*\(",
    r"sleep\s*\(\s*\d+\s*\)",
    # 纯架构/安全专业长讨论
    r"eureka|htrix|hystrix|feign",
    r"服务注册中心|熔断|网关降级|网关升降级|声明式远程调用",
    r"sql注入|sqlmap|盲注|时间盲注",
    r"x-forwarded-for",
    # CTF 专业操作
    r"fl4g|flaaag",
    r"getflag|get_flag",
    r"\bctf\b",
    # 大段命令行
    r"mysql\s+-u",
    r"nmap\s+",
]


def is_media_only(text: str) -> bool:
    """是否为纯资源消息"""
    text = text.strip()
    if not text:
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    return all(PURE_MEDIA_RE.match(l) for l in lines)


def has_tech_hard(text: str) -> bool:
    """是否含硬核技术内容（大段代码/纯专业术语）"""
    for pat in TECH_HARD_ONLY:
        if re.search(pat, text, re.I):
            return True
    return False


def is_meaningless(text: str) -> bool:
    """是否为无意义短消息"""
    text = text.strip()
    if len(text) <= 1:
        return True
    if re.match(r"^(嗯|啊|哦|呢|啦|嘛|呀|的|了|是|不|好|行|对|ok|yes|no|昂|噢|嗷)+$", text, re.I):
        return True
    return False


def parse_raw(path: Path) -> list[tuple[str, str]]:
    """解析原始 txt，返回 [(说话人, 消息), ...]"""
    lines = path.read_text(encoding="utf-8").splitlines()
    messages: list[tuple[str, str]] = []
    current_speaker = None
    current_text: list[str] = []

    for line in lines:
        stripped = line.rstrip()
        m = re.match(r"^(.+?):\s*$", stripped)
        if m:
            if current_speaker and current_text:
                text = "\n".join(current_text).strip()
                if text:
                    messages.append((current_speaker, text))
            current_speaker = m.group(1)
            current_text = []
        elif stripped == "":
            if current_speaker and current_text:
                text = "\n".join(current_text).strip()
                if text:
                    messages.append((current_speaker, text))
                current_text = []
                current_speaker = None
        else:
            if current_speaker:
                current_text.append(stripped)

    if current_speaker and current_text:
        text = "\n".join(current_text).strip()
        if text:
            messages.append((current_speaker, text))

    return messages


def norm_speaker(speaker: str) -> str:
    """归一化说话人：male / female / other"""
    if "Owe" in speaker:
        return "male"
    if "Angel" in speaker or "景行" in speaker:
        return "female"
    return "other"


def merge_consecutive(msgs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """合并连续同说话人消息"""
    if not msgs:
        return []
    merged = [msgs[0]]
    for speaker, text in msgs[1:]:
        if speaker == merged[-1][0]:
            merged[-1] = (speaker, merged[-1][1] + " " + text)
        else:
            merged.append((speaker, text))
    return merged


def extract_dialogs(msgs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """切分对话窗口。

    切分规则：
    - 遇到硬核技术内容（大段代码）切断
    - 遇到 other 说话人切断（保持双人对话）
    - 学校生活、简单技术操作完全保留
    - 至少 2 轮（4 条消息）才保留
    """
    dialogs: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    for speaker, text in msgs:
        norm = norm_speaker(speaker)
        if norm == "other":
            if len(current) >= 4:
                dialogs.append(current)
            current = []
            continue

        # 硬核技术内容：切断并跳过
        if has_tech_hard(text):
            if len(current) >= 4:
                dialogs.append(current)
            current = []
            continue

        # 过滤纯媒体
        if is_media_only(text):
            continue

        # 过滤无意义短消息（不切断，只跳过）
        if is_meaningless(text):
            continue

        current.append((norm, text))

    if len(current) >= 4:
        dialogs.append(current)

    return dialogs


def is_dialog_clean(dialog: list[tuple[str, str]]) -> bool:
    """整窗校验：任一条消息含硬核技术，整个窗口作废"""
    for _, text in dialog:
        if has_tech_hard(text):
            return False
    # 至少有一条消息 >= 3 字
    return any(len(t) >= 3 for _, t in dialog)


def dialog_to_sharegpt(dialog: list[tuple[str, str]]) -> dict:
    """转 ShareGPT 格式"""
    merged = merge_consecutive(dialog)
    conversations = []
    for speaker, text in merged:
        role = "human" if speaker == "male" else "gpt"
        conversations.append({"from": role, "value": text.strip()})

    # 确保以 human 开头
    if conversations and conversations[0]["from"] != "human":
        conversations = conversations[1:]
    if len(conversations) < 2:
        return {}

    # 确保交替，合并同角色连续
    final = [conversations[0]]
    for msg in conversations[1:]:
        if msg["from"] != final[-1]["from"]:
            final.append(msg)
        else:
            final[-1]["value"] += " " + msg["value"]

    if len(final) < 2:
        return {}

    return {"conversations": final, "source": "real_chat_raw"}


def main() -> None:
    all_dialogs: list[dict] = []

    for fname, config in FILES.items():
        path = SRC_DIR / fname
        if not path.exists():
            print(f"[跳过] {fname} 不存在")
            continue

        print(f"\n=== 解析 {fname} (男={config['male']}, 女={config['female_name']}) ===")
        msgs = parse_raw(path)
        print(f"原始消息数: {len(msgs)}")

        dialogs = extract_dialogs(msgs)
        print(f"切分对话窗口: {len(dialogs)}")

        clean_dialogs = [d for d in dialogs if is_dialog_clean(d)]
        print(f"整窗校验通过: {len(clean_dialogs)}")

        for dialog in clean_dialogs:
            record = dialog_to_sharegpt(dialog)
            if record and len(record["conversations"]) >= 2:
                all_dialogs.append(record)

    # 去重
    seen = set()
    unique = []
    for d in all_dialogs:
        conv = d["conversations"]
        key = (conv[0]["value"][:40], conv[1]["value"][:40])
        if key not in seen:
            seen.add(key)
            unique.append(d)

    print(f"\n=== 最终结果 ===")
    print(f"总对话: {len(unique)} 条")

    lens = [len(d["conversations"]) for d in unique]
    if lens:
        print(f"轮数: 平均 {sum(lens)/len(lens):.1f}, 范围 {min(lens)}-{max(lens)}")

    with DST.open("w", encoding="utf-8") as f:
        for d in unique:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n已输出: {DST} ({len(unique)} 条)")

    print("\n=== 样本预览（前20条）===")
    for d in unique[:20]:
        print(f"\n--- ---")
        for msg in d["conversations"][:6]:
            v = msg["value"][:60].replace("\n", " ")
            print(f"  {msg['from']}: {v}")


if __name__ == "__main__":
    main()
