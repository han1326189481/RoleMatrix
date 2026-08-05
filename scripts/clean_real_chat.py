"""清洗 real-chat-extracted 数据集，输出适合小R人设的 ShareGPT 训练数据。

清洗流程：
1. 去除元标注污染（"（我之前说：xxx）"）
2. 合并被拆碎的连续消息
3. 过滤技术操作对话
4. 过滤只剩无意义碎片的对话
5. 按小R人设风格打分（撒娇/傲娇/口语化加分）
6. 输出清洗后的 ShareGPT 格式

小R人设：害羞内向、对男友敞开心扉、叽叽喳喳、小傲娇、爱撒娇
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(r"D:\RoleMatrix\data\datasets\real-chat-extracted\real_chat_sharegpt.jsonl")
DST_DIR = Path(r"D:\RoleMatrix\data\datasets\cleaned")
DST_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 第一步：去除元标注污染
# ============================================================
# 匹配 "（我之前说：xxx）" 或 "（我之前说：xxx）" 整段
META_RE = re.compile(r"（我之前说[：:].*?）", re.DOTALL)
# 连续多个空格合并
MULTI_SPACE_RE = re.compile(r"\s{2,}")


def strip_meta(text: str) -> str:
    """去掉元标注污染，返回干净文本"""
    text = META_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ============================================================
# 第二步：过滤技术对话
# ============================================================
# 强技术特征（命中任一即丢弃）
TECH_HARD_PATTERNS = [
    r"\d+\.\d+\.\d+\.\d+",             # IP
    r"192\.168",
    r"localhost",
    r"https?://",
    r"\bupload\b",
    r"const |var |function |class |import |def |return ",
    r"\bflag\b|\bctf\b",
    r"\.php|\.exe|\.bat|\.cmd|\.py\b",
    r"mysql|sql\b|database",
    r"localhost",
    r"\bapi/",
    r"模拟器",
    r"phpstudy",
]

# 弱技术特征（命中 2 个以上才丢弃，避免误杀日常词）
TECH_SOFT_PATTERNS = [
    r"上传", r"截图", r"按钮", r"网页", r"浏览器",
    r"桌面", r"鼠标", r"软件", r"安装", r"配置",
    r"文件", r"\.txt", r"\.js\b", r"\.gif",
    r"账号", r"登录", r"密码",
    r"老师", r"作业", r"题目", r"答案",
    r"提交", r"网址", r"界面",
    r"登陆",
    r"课件|上节课|下节课|这门课|这节课|那节课",  # 课程讨论
    r"考试|复习|学分|学号",                       # 学校场景
    r"做出来了吗|做完了|做完了没",                   # 作业进度
    r"flag|attack|defense",                        # CTF
]


def is_tech(text: str) -> bool:
    """判断是否为技术对话"""
    # 强特征命中即过滤
    for pat in TECH_HARD_PATTERNS:
        if re.search(pat, text, re.I):
            return True
    # 弱特征命中 2 个以上过滤
    hits = sum(1 for pat in TECH_SOFT_PATTERNS if re.search(pat, text, re.I))
    return hits >= 2


# ============================================================
# 第三步：判断无意义碎片
# ============================================================
# 单独的虚词/语气词不算有意义回复
MEANINGLESS_RE = re.compile(r"^(嗯|啊|哦|呢|啦|嘛|呀|的|了|是|不|好|行|对|ok|yes|no|嗯嗯|哦哦|哈哈|嘿嘿|呵呵|嘻嘻|噗)+$", re.I)


def is_meaningless(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    if len(text) <= 2:
        return True
    if MEANINGLESS_RE.match(text):
        return True
    return False


# ============================================================
# 第四步：小R人设风格打分
# ============================================================
# 加分项：撒娇/傲娇/情感/口语化
STYLE_POSITIVE = [
    (r"哼", 2),                          # 傲娇
    (r"嘛", 1), (r"呀", 1), (r"呢", 1), (r"啦", 1), (r"哦", 1), (r"呜", 2),
    (r"嘻嘻|哈哈|嘿嘿|呵呵|哈哈哈哈", 1),
    (r"不嘛|就要|讨厌|笨蛋|坏蛋", 3),      # 强撒娇
    (r"人家|俺", 2),
    (r"睡|梦|想", 1),                      # 情感词
    (r"喜欢|开心|难过|生气|无聊|累", 2),
    (r"陪我|不理我|想我", 3),              # 依赖感
    (r"韩哥|哥哥|宝", 2),
    (r"～|~", 1),                          # 软语气
    (r"！{2,}", 1),                        # 激动
    (r"？{1,}", 1),                        # 好奇
    (r"…|\.\.\.", 1),                     # 省略号（害羞/犹豫）
    (r"诶|欸|咦|呀|哇", 1),                # 感叹词
    (r"好嘟|好滴|好哒", 2),                # 软答应
    (r"没事|没事哒|没事的", 1),
    (r"我滴妈|妈呀|我去", 1),              # 口语
    (r"吃饭|吃东西|饿了", 1),              # 日常
]

# 减分项：冷淡/书面/技术腔
STYLE_NEGATIVE = [
    (r"好的，|是的。", -2),                # 书面
    (r"根据|由于|因此|所以,|然而", -2),     # 论述腔
    (r"请问|麻烦|抱歉", -1),               # 客服腔
    (r"\d{4,}", -2),                       # 长数字串
]


def style_score(text: str) -> int:
    """给文本打风格分，正值越接近小R人设"""
    score = 0
    for pat, w in STYLE_POSITIVE:
        if re.search(pat, text):
            score += w
    for pat, w in STYLE_NEGATIVE:
        if re.search(pat, text):
            score += w
    return score


# ============================================================
# 第五步：合并连续同角色消息
# ============================================================
def merge_consecutive(conv: list[dict]) -> list[dict]:
    """合并连续同角色的多条消息"""
    if not conv:
        return []
    merged = [conv[0]]
    for msg in conv[1:]:
        if msg["from"] == merged[-1]["from"]:
            merged[-1]["value"] += " " + msg["value"]
        else:
            merged.append(msg)
    return merged


# ============================================================
# 主流程
# ============================================================
def clean_message(text: str) -> str:
    """清洗单条消息：去元标注 + 去多余换行"""
    text = strip_meta(text)
    # 多个换行合并为单个
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def process_record(record: dict) -> dict | None:
    """处理单条记录，返回清洗后的 record 或 None（丢弃）"""
    conv = record.get("conversations", [])
    if len(conv) < 2:
        return None

    # 清洗每条消息
    cleaned_conv = []
    for msg in conv:
        new_value = clean_message(msg["value"])
        if new_value:
            cleaned_conv.append({"from": msg["from"], "value": new_value})

    if len(cleaned_conv) < 2:
        return None

    # 合并连续同角色
    cleaned_conv = merge_consecutive(cleaned_conv)

    if len(cleaned_conv) < 2:
        return None

    # 检查用户和小R消息是否都有实质内容
    user_msg = cleaned_conv[0]["value"]
    bot_msg = cleaned_conv[1]["value"]

    if is_meaningless(user_msg) or is_meaningless(bot_msg):
        return None

    # 过滤技术对话
    if is_tech(user_msg) or is_tech(bot_msg):
        return None

    # 风格打分
    score = style_score(user_msg) + style_score(bot_msg)

    return {
        "conversations": cleaned_conv,
        "style_score": score,
    }


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    kept: list[dict] = []
    dropped_tech = 0
    dropped_meaningless = 0
    dropped_meta_only = 0
    dropped_dup = 0

    # 用 (user_key, bot_key) 做去重
    seen: set[tuple[str, str]] = set()

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        result = process_record(record)
        if result is None:
            # 统计丢弃原因（粗略）
            conv = record.get("conversations", [])
            if len(conv) < 2:
                continue
            user_clean = strip_meta(conv[0]["value"])
            bot_clean = strip_meta(conv[1]["value"]) if len(conv) > 1 else ""
            full = user_clean + " " + bot_clean
            if is_tech(full):
                dropped_tech += 1
            elif is_meaningless(user_clean) or is_meaningless(bot_clean):
                dropped_meaningless += 1
            else:
                dropped_meta_only += 1
        else:
            # 去重：取用户消息前 40 字 + 小R消息前 40 字做 key
            conv = result["conversations"]
            key = (conv[0]["value"][:40], conv[1]["value"][:40])
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            kept.append(result)

    # 按风格分排序
    kept.sort(key=lambda x: x["style_score"], reverse=True)

    print(f"=== 清洗结果 ===")
    print(f"原始: {len(lines)} 条")
    print(f"保留: {len(kept)} 条 ({len(kept)*100//len(lines)}%)")
    print(f"丢弃 - 技术对话: {dropped_tech}")
    print(f"丢弃 - 无意义碎片: {dropped_meaningless}")
    print(f"丢弃 - 重复: {dropped_dup}")
    print(f"丢弃 - 其他: {dropped_meta_only}")

    # 分桶：高分（>=2）优质样本 / 中分（0-1）一般样本 / 低分（<0）边缘样本
    high = [r for r in kept if r["style_score"] >= 2]
    mid = [r for r in kept if 0 <= r["style_score"] < 2]
    low = [r for r in kept if r["style_score"] < 0]
    print(f"\n=== 风格分桶 ===")
    print(f"高分 (>=2, 强小R风格): {len(high)}")
    print(f"中分 (0-1, 一般口语): {len(mid)}")
    print(f"低分 (<0, 偏书面/冷淡): {len(low)}")

    # 输出样本预览
    for bucket_name, bucket in [("high", high), ("mid", mid), ("low", low)]:
        print(f"\n=== {bucket_name} 样本5条 ===")
        for r in bucket[:5]:
            conv = r["conversations"]
            u = conv[0]["value"][:60].replace("\n", " ")
            b = conv[1]["value"][:60].replace("\n", " ") if len(conv) > 1 else "(无)"
            print(f"  [score={r['style_score']}] 用户: {u}")
            print(f"  小R: {b}")
            print()

    # 输出清洗后的数据（ShareGPT 格式，去掉 style_score）
    # 只输出高分（>=2）样本作为精炼训练集
    # 中分样本太混杂，保留作为辅助
    out_path = DST_DIR / "real_chat_clean_sharegpt.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in high:
            f.write(json.dumps({"conversations": r["conversations"]}, ensure_ascii=False) + "\n")
    print(f"\n已输出精炼版: {out_path} ({len(high)} 条纯小R风格)")

    # 中分作为扩充候选
    out_path2 = DST_DIR / "real_chat_aux_sharegpt.jsonl"
    with out_path2.open("w", encoding="utf-8") as f:
        for r in mid:
            f.write(json.dumps({"conversations": r["conversations"]}, ensure_ascii=False) + "\n")
    print(f"已输出辅助版: {out_path2} ({len(mid)} 条辅助样本，需人工抽检)")


if __name__ == "__main__":
    main()
