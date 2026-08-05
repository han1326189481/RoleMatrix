"""深度预览 Angel 和 景行 数据集样本质量"""
import json
import random
from pathlib import Path

random.seed(42)

jsonl_path = Path(r"D:\RoleMatrix\data\datasets\real-chat-extracted\real_chat_sharegpt.jsonl")
with open(jsonl_path, "r", encoding="utf-8") as f:
    pairs = [json.loads(line) for line in f if line.strip()]

# 分离 Angel 和 景行
# Angel 的对话中 user text 通常有技术名词，景行的更生活化
# 用 assistant 回复特征判断
angel_pairs = []
jx_pairs = []
for p in pairs:
    assistant_text = p["conversations"][1]["value"]
    if any(kw in assistant_text for kw in ["好嘟", "okk", "韩哥", "噗哈哈哈", "我滴妈", "妈呀", "过时间了"]):
        angel_pairs.append(p)
    elif any(kw in assistant_text for kw in ["哈哈哈哈", "我去我去", "蛙趣", "好贴心", "好耶", "～", "感动"]):
        jx_pairs.append(p)
    else:
        # 用长度和内容猜
        if len(assistant_text) < 10:
            # 短回复，看上下文
            user_text = p["conversations"][0]["value"]
            if any(kw in user_text for kw in ["韩哥", "Angel", "学号", "提交"]):
                angel_pairs.append(p)
            else:
                jx_pairs.append(p)
        else:
            angel_pairs.append(p)

print(f"Angel 条数: {len(angel_pairs)}, 景行 条数: {len(jx_pairs)}")

# 过滤：只保留高质量对话
def quality_filter(pairs_list, min_assistant_len=4, max_tech_score=2):
    """质量过滤"""
    tech_words = [
        "下载", "安装", "配置", "编译", "运行", "服务器", "端口",
        "命令", "ctrl", "shift", "浏览器", "控制台", "断点", "调试",
        "截图给我", "远程", "登录", "学号", "账号", "密码",
        "提交", "压缩包", "文件夹", "idea", "代码",
        "eureka", "注册", "网关", "熔断", "降级",
    ]
    result = []
    for p in pairs_list:
        asst = p["conversations"][1]["value"]
        user = p["conversations"][0]["value"]

        # 太短
        if len(asst) < min_assistant_len:
            continue

        # 纯系统消息
        if asst in {"[图片]", "[动画表情]", "[文件]", "[语音]", "[视频]"}:
            continue

        # 技术分
        tech_score = sum(1 for w in tech_words if w in user + asst)
        if tech_score > max_tech_score:
            continue

        result.append(p)
    return result

angel_quality = quality_filter(angel_pairs)
jx_quality = quality_filter(jx_pairs)

print(f"过滤后 Angel: {len(angel_quality)}, 景行: {len(jx_quality)}")

# ---- 预览高质量样本 ----
print("\n" + "="*70)
print("🎯 Angel 高质量样本 (top 10)")
print("="*70)
for i, p in enumerate(angel_quality[:10]):
    user = p["conversations"][0]["value"]
    asst = p["conversations"][1]["value"]
    print(f"\n--- Angel #{i+1} ---")
    print(f"  💬 User: {user[:200]}")
    print(f"  💗 Angel: {asst[:250]}")

print("\n" + "="*70)
print("🎯 景行 高质量样本 (top 10)")
print("="*70)
for i, p in enumerate(jx_quality[:10]):
    user = p["conversations"][0]["value"]
    asst = p["conversations"][1]["value"]
    print(f"\n--- 景行 #{i+1} ---")
    print(f"  💬 User: {user[:200]}")
    print(f"  💗 景行: {asst[:250]}")

# ---- 随机抽样 ----
print("\n" + "="*70)
print("🎲 Angel 随机样本")
print("="*70)
for i, p in enumerate(random.sample(angel_quality, min(5, len(angel_quality)))):
    user = p["conversations"][0]["value"]
    asst = p["conversations"][1]["value"]
    print(f"\n--- Angel Random #{i+1} ---")
    print(f"  💬 User: {user[:200]}")
    print(f"  💗 Angel: {asst[:250]}")

print("\n" + "="*70)
print("🎲 景行 随机样本")
print("="*70)
for i, p in enumerate(random.sample(jx_quality, min(5, len(jx_quality)))):
    user = p["conversations"][0]["value"]
    asst = p["conversations"][1]["value"]
    print(f"\n--- 景行 Random #{i+1} ---")
    print(f"  💬 User: {user[:200]}")
    print(f"  💗 景行: {asst[:250]}")
