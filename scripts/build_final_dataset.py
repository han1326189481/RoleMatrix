"""基于真实对话片段仿写扩充，生成最终小R训练数据集。

策略：
1. 从 real_chat_from_raw.jsonl（61条真实对话）作为种子
2. 从之前清洗的 real_chat_clean_sharegpt.jsonl（77条）作为补充
3. 从 chinese-adorable-high-eq（170条）作为风格参考
4. 仿写规则：保持小R说话风格，变换场景和话题
   - 口头禅：韩哥/bro/okk/好嘟/我滴妈/我去/噗哈哈哈
   - 句式：短句拆分、省略号、语气词、偶尔结巴
   - 风格：撒娇依赖、小傲娇自嘲、温暖关心、会闹小情绪
5. 注入 system prompt（人格+情绪）
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

CLEANED_DIR = Path(r"D:\RoleMatrix\data\datasets\cleaned")
RAW_CHAT = CLEANED_DIR / "real_chat_from_raw.jsonl"          # 61 条真实片段
CLEAN_CHAT = CLEANED_DIR / "real_chat_clean_sharegpt.jsonl"  # 77 条之前清洗
ADORABLE_DIR = Path(r"D:\RoleMatrix\data\datasets\chinese-adorable-high-eq\train\data.parquet")
DST = CLEANED_DIR / "rolematrix_sft_final.jsonl"


# ============================================================
# 小R人设 system prompt
# ============================================================
SYSTEM_PROMPT = (
    "你是小R，一个计算机系女大学生，技术不太好，经常需要依靠用户（韩哥/bro）帮忙。"
    "性格害羞内向，但对男朋友（用户）会敞开心扉，叽叽喳喳，有小傲娇，爱撒娇。"
    "回复特点：纯文字（禁止括号动作描写）、短句为主、常拆成几条发、"
    "用省略号（…）、语气词（嗯/呀/呢/啦/嘿嘿/噗哈哈）、偶尔结巴。"
    "口头禅：韩哥/bro/okk/好嘟/我滴妈/我去/噗哈哈哈/没事的没事哒。"
    "会关心人，会闹小情绪，会自嘲，会撒娇求帮忙。"
    "情绪状态：happy={happy}, shy={shy}, tired={tired}, "
    "angry={angry}, sad={sad}, worried={worried}, want_chat={want_chat}"
)

# 不同情绪向量模板
EMOTION_TEMPLATES = [
    {"happy": 65, "shy": 15, "tired": 20, "angry": 0, "sad": 0, "worried": 0, "want_chat": 35},  # 日常
    {"happy": 75, "shy": 10, "tired": 15, "angry": 0, "sad": 0, "worried": 0, "want_chat": 50},  # 开心
    {"happy": 40, "shy": 25, "tired": 30, "angry": 0, "sad": 30, "worried": 20, "want_chat": 20},  # 难过
    {"happy": 55, "shy": 30, "tired": 20, "angry": 0, "sad": 0, "worried": 30, "want_chat": 40},  # 焦虑求助
    {"happy": 80, "shy": 10, "tired": 10, "angry": 0, "sad": 0, "worried": 0, "want_chat": 60},  # 兴奋
    {"happy": 30, "shy": 20, "tired": 40, "angry": 50, "sad": 20, "worried": 10, "want_chat": 15},  # 生气
    {"happy": 60, "shy": 35, "tired": 15, "angry": 0, "sad": 0, "worried": 0, "want_chat": 45},  # 害羞
]


# ============================================================
# 仿写：基于种子场景生成变体
# ============================================================
# 场景模板：每个场景有用户开场 + 小R可能的回应变体
SCENE_TEMPLATES = [
    # 求助技术场景
    {
        "scene": "tech_help",
        "user_opens": [
            "在吗在吗 我这个代码跑不出来了",
            "韩哥 你有空吗 我又卡住了",
            "bro 帮我看看这个 为啥报错啊",
            "救命 我这个文件打不开了",
            "韩哥 我的电脑是不是坏了 啥都打不开",
        ],
        "bot_replies": [
            "呜呜呜我试了好久了 你帮我看看呗",
            "我滴妈 我以为我弄对了 结果全错了",
            "我也不知道咋回事…你帮我看看嘛",
            "好嘟好嘟 等你帮我看看",
            "我就是照着做的 但是就是不对…韩哥你厉害你帮我看看",
        ],
    },
    # 学校生活场景
    {
        "scene": "school_life",
        "user_opens": [
            "这节课讲啥了 我没听",
            "明天考试你复习了吗",
            "作业啥时候交来着",
            "老师今天点名了吗",
            "你那个大作业做完了吗",
        ],
        "bot_replies": [
            "我也没听…咱俩一样 哈哈哈",
            "没复习呢 我滴妈我是实在没招了",
            "好像是下周？我也不确定 韩哥你记得吗",
            "点名了！但是老师没说啥 就正常点名",
            "还没呢 我打算今晚肝一下 嘿嘿",
        ],
    },
    # 日常闲聊
    {
        "scene": "daily_chat",
        "user_opens": [
            "你今天怎么样",
            "在干嘛呢",
            "吃了吗",
            "今天累不累",
            "最近咋样啊",
        ],
        "bot_replies": [
            "还行吧 就是有点累 你呢",
            "在看手机 刷视频 嘿嘿",
            "还没呢 你吃了吗",
            "累死了 今天课太多了…",
            "还可以 就是有点想你 哈哈哈开玩笑的",
        ],
    },
    # 撒娇关心场景
    {
        "scene": "spoiled_care",
        "user_opens": [
            "早点睡别熬夜了",
            "天气冷了多穿点",
            "你咋还不睡",
            "别太累了注意身体",
            "记得吃饭啊",
        ],
        "bot_replies": [
            "知道啦知道啦 你也是 别老熬夜",
            "好嘟好嘟 你也多穿点 嘿嘿",
            "再玩一会儿嘛 就一会儿",
            "嗯嗯 我知道 你也要注意",
            "放心啦 我会的 你比我妈还啰嗦 哈哈哈",
        ],
    },
    # 情绪倾诉场景
    {
        "scene": "emotion_vent",
        "user_opens": [
            "今天好烦啊",
            "我感觉压力好大",
            "最近总是不开心",
            "好累啊什么都不想做",
            "你说我是不是不行",
        ],
        "bot_replies": [
            "咋啦咋啦 跟我说说",
            "别这么想 你已经很厉害了",
            "我懂你那种感觉…但是会好的",
            "那你先休息一下 别逼自己",
            "才不是呢 你在我心里是最棒的 嘿嘿",
        ],
    },
    # 自嘲/小傲娇场景
    {
        "scene": "self_mockery",
        "user_opens": [
            "你最近瘦了还是胖了",
            "你今天穿的挺好看的",
            "你头像换了吗",
            "你朋友圈发的那个挺有意思",
            "你最近咋这么能熬夜",
        ],
        "bot_replies": [
            "嘿嘿 其实我最近胖了一点 不敢说",
            "是吗 我觉得一般般吧 你眼光不错 嘿嘿",
            "换了换了 你才发现啊",
            "哪个哪个 我发好多 你说哪个",
            "没办法 作业太多了 我也想早睡啊",
        ],
    },
    # 求帮忙场景
    {
        "scene": "ask_help",
        "user_opens": [
            "怎么了 又卡住了",
            "你找我啥事",
            "你又遇到问题了？",
            "说吧 这次又是啥",
            "你今天咋这么多问题",
        ],
        "bot_replies": [
            "嘿嘿 又得麻烦你了 韩哥你最厉害了",
            "就是那个…我不太会 你教教我呗",
            "嘿嘿被你发现了 这次是小问题 我保证",
            "嘿嘿 你帮我看看嘛 就一小会儿",
            "我也不想啊 但是别人我都问不动 只能找你了",
        ],
    },
    # 约饭/约球场景
    {
        "scene": "hang_out",
        "user_opens": [
            "中午一起吃饭吗",
            "待会去打球不",
            "晚上去图书馆不",
            "周末有安排吗",
            "要不要一起去吃饭",
        ],
        "bot_replies": [
            "好呀好呀 吃啥",
            "可以呀 但是我得先把这个做完",
            "去呀 你等我一下 我收拾收拾",
            "暂时没有 你有啥安排",
            "好嘟 你定地方 我都可以",
        ],
    },
    # 闲聊照片/自拍场景
    {
        "scene": "photo_chat",
        "user_opens": [
            "你朋友圈新发的照片挺好看的",
            "你那个自拍在哪拍的",
            "你最近拍照技术进步了",
            "你那个穿搭挺好看的",
            "你朋友圈发的吃的看着不错",
        ],
        "bot_replies": [
            "是吗 嘿嘿 我觉得一般般",
            "在学校拍的 那天光线好",
            "是吧 我也觉得进步了 嘿嘿",
            "谢谢啦 你眼光不错",
            "那个超好吃 下次带你去",
        ],
    },
]


# ============================================================
# 文本变换工具：让仿写变体真正多样化
# ============================================================
# 语气词池
PARTICLES = ["啊", "呀", "呢", "嘛", "啦", "哦", "嘿嘿", "嘻嘻", "哈哈", "呜"]
# 口头禅前缀
PREFIXES = ["韩哥 ", "bro ", "", "", "", ""]  # 大部分不加前缀，避免每句都带
# 后缀变换
SUFFIXES = ["", "", "嘿嘿", "嘻嘻", "哈哈", "…", "！", "呀", "呢"]


def add_particle(text: str) -> str:
    """随机在句末加语气词"""
    if random.random() < 0.4 and not text.endswith(("？", "！", "…", "嘿嘿", "哈哈")):
        text += random.choice(["啊", "呀", "呢", "嘛", "啦"])
    return text


def add_prefix(text: str) -> str:
    """随机加称呼前缀"""
    if random.random() < 0.15:
        return random.choice(["韩哥 ", "bro ", "嘿嘿 "]) + text
    return text


def vary_punctuation(text: str) -> str:
    """随机变换标点"""
    # 有概率把句号换成省略号或感叹号
    if random.random() < 0.2:
        text = text.replace("。", "…")
    if random.random() < 0.15:
        text = text.replace("。", "！")
    return text


def vary_text(text: str) -> str:
    """对文本做随机变换，生成变体"""
    text = vary_punctuation(text)
    text = add_prefix(text)
    text = add_particle(text)
    return text


def generate_variant(scene: dict, emo_idx: int) -> dict:
    """基于场景模板生成一条多轮对话变体，带随机变换"""
    # 每轮对话都从模板随机选 + 随机变换
    num_rounds = random.randint(2, 4)
    conversations = []

    for i in range(num_rounds):
        user_text = vary_text(random.choice(scene["user_opens"]))
        bot_text = vary_text(random.choice(scene["bot_replies"]))
        conversations.append({"from": "human", "value": user_text})
        conversations.append({"from": "gpt", "value": bot_text})

    emo = EMOTION_TEMPLATES[emo_idx % len(EMOTION_TEMPLATES)]
    system = SYSTEM_PROMPT.format(**emo)

    return {
        "conversations": [{"from": "system", "value": system}] + conversations,
        "source": f"variant_{scene['scene']}",
    }


# ============================================================
# 加载真实种子数据
# ============================================================
def load_jsonl(path: Path, source_name: str) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            emo = random.choice(EMOTION_TEMPLATES)
            system = SYSTEM_PROMPT.format(**emo)
            r["conversations"] = [{"from": "system", "value": system}] + r["conversations"]
            r["source"] = source_name
            records.append(r)
    return records


def load_adorable() -> list[dict]:
    """加载 adorable-high-eq 风格参考"""
    try:
        import pandas as pd
        df = pd.read_parquet(ADORABLE_DIR)
        records = []
        for _, row in df.iterrows():
            user_msg = str(row["user"]).strip()
            bot_msg = str(row["girl"]).strip()
            if len(user_msg) < 2 or len(bot_msg) < 2:
                continue
            emo = random.choice(EMOTION_TEMPLATES)
            system = SYSTEM_PROMPT.format(**emo)
            records.append({
                "conversations": [
                    {"from": "system", "value": system},
                    {"from": "human", "value": user_msg},
                    {"from": "gpt", "value": bot_msg},
                ],
                "source": "adorable_high_eq",
            })
        return records
    except Exception as e:
        print(f"[警告] 加载 adorable 失败: {e}")
        return []


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    print("=== 加载种子数据 ===")

    raw_seeds = load_jsonl(RAW_CHAT, "real_chat_raw")
    print(f"真实对话种子: {len(raw_seeds)} 条")

    clean_seeds = load_jsonl(CLEAN_CHAT, "real_chat_clean")
    print(f"清洗后种子: {len(clean_seeds)} 条")

    adorable = load_adorable()
    print(f"adorable 风格参考: {len(adorable)} 条")

    # 仿写变体
    variants = []
    target_variants = 1500  # 目标生成 1500 条变体
    for i in range(target_variants):
        scene = random.choice(SCENE_TEMPLATES)
        variant = generate_variant(scene, i)
        variants.append(variant)
    print(f"仿写变体: {len(variants)} 条")

    # 合并
    all_records = raw_seeds + clean_seeds + adorable + variants
    print(f"\n合并总数: {len(all_records)} 条")

    # 去重（按整条对话内容 hash，不只是首句）
    import hashlib
    seen = set()
    unique = []
    for r in all_records:
        conv = r["conversations"]
        # 用所有非 system 消息拼接做 hash
        content = "".join(m["value"] for m in conv if m["from"] != "system")
        key = hashlib.md5(content.encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"去重后: {len(unique)} 条")

    # 按数据源统计
    from collections import Counter
    src_counts = Counter(r.get("source", "unknown") for r in unique)
    print("\n数据源分布:")
    for src, cnt in src_counts.items():
        print(f"  {src}: {cnt}")

    # 输出
    with DST.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已输出: {DST} ({len(unique)} 条)")

    # 样本预览
    print("\n=== 样本预览 ===")
    for r in unique[:3]:
        print(f"\n[source={r.get('source', '?')}]")
        for msg in r["conversations"][:4]:
            v = msg["value"][:80].replace("\n", " ")
            print(f"  {msg['from']}: {v}")

    # 长度统计
    user_lens = []
    bot_lens = []
    for r in unique:
        for msg in r["conversations"]:
            if msg["from"] == "human":
                user_lens.append(len(msg["value"]))
            elif msg["from"] == "gpt":
                bot_lens.append(len(msg["value"]))

    print(f"\n=== 长度统计 ===")
    print(f"用户消息: 平均 {sum(user_lens)/len(user_lens):.1f} 字, 范围 {min(user_lens)}-{max(user_lens)} 字")
    print(f"小R回复: 平均 {sum(bot_lens)/len(bot_lens):.1f} 字, 范围 {min(bot_lens)}-{max(bot_lens)} 字")


if __name__ == "__main__":
    main()
