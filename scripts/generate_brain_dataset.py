"""用 DeepSeek API 给现有对话数据生成大脑决策标签

输入：data/datasets/cleaned/rolematrix_sft_final.jsonl (ShareGPT 格式)
输出：data/datasets/cleaned/rolematrix_brain_train.jsonl (instruction 格式)

每条输出格式：
{
  "messages": [
    {"role": "system", "content": "你是小R的大脑..."},
    {"role": "user", "content": "用户消息：xxx\n情绪状态：{}\n历史：[]"},
    {"role": "assistant", "content": "{\"emotion_delta\":{...},\"memory_recall\":null,\"reply_plan\":{...}}"}
  ]
}

DeepSeek 作为标注员：给定"用户消息 + 小R实际回复"，逆向推断大脑决策应该是什么。
"""
from __future__ import annotations

import os
import json
import time
import asyncio
import random
from pathlib import Path

# 加载 .env
_env_path = Path(r"D:\RoleMatrix\.env")
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import httpx

# ============================================================
# 配置
# ============================================================
INPUT_PATH = Path(r"D:\RoleMatrix\data\datasets\cleaned\rolematrix_sft_final.jsonl")
OUTPUT_PATH = Path(r"D:\RoleMatrix\data\datasets\cleaned\rolematrix_brain_train.jsonl")

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 60.0
MAX_RETRIES = 3
CONCURRENCY = 5  # 并发数（避免触发 503）

# 大脑的 system prompt（训练目标格式）
BRAIN_SYSTEM = """你是小R的大脑，负责内部决策（不直接发给用户）。
根据用户消息、情绪状态和历史，输出 JSON 决策：

```json
{
  "emotion_delta": {
    "happy": 整数(-5到+5),
    "tired": 整数(-5到+5),
    "shy": 整数(-5到+5),
    "want_chat": 整数(-5到+5)
  },
  "memory_recall": "回忆起的相关历史信息（无则填 null）",
  "reply_plan": {
    "tone": "温柔关心 | 调皮撒娇 | 坦诚求助 | 害羞小声 | 日常闲聊",
    "points": ["要点1", "要点2"],
    "length": "short | medium"
  }
}
```

你是计算机系女大学生小R，技术不太好，遇到技术问题会坦诚求助。
只输出 JSON，不要多余文字，不要 markdown 代码块标记。"""

# 标注员 prompt（让 DeepSeek 生成标签）
LABELER_SYSTEM = """你是数据标注员。根据给定的"用户消息"和"小R的实际回复"，
逆向推断小R的"大脑决策"应该是什么。

输出 JSON，格式：
{
  "emotion_delta": {
    "happy": 整数(-5到+5),
    "tired": 整数(-5到+5),
    "shy": 整数(-5到+5),
    "want_chat": 整数(-5到+5)
  },
  "memory_recall": "回忆起的相关历史信息（无则填 null）",
  "reply_plan": {
    "tone": "温柔关心 | 调皮撒娇 | 坦诚求助 | 害羞小声 | 日常闲聊",
    "points": ["要点1", "要点2"],
    "length": "short | medium"
  }
}

只输出 JSON，不要 markdown 代码块标记，不要解释。"""


# ============================================================
# DeepSeek API 调用
# ============================================================
async def label_one(client: httpx.AsyncClient, api_key: str,
                    user_msg: str, xiaor_reply: str,
                    semaphore: asyncio.Semaphore) -> dict | None:
    """让 DeepSeek 给一条对话生成大脑标签"""
    user_content = f"用户消息：{user_msg}\n\n小R的实际回复：{xiaor_reply}\n\n请生成大脑决策 JSON。"

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": LABELER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0.3,  # 低温度保证标签一致性
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    json=payload, headers=headers,
                )
                if resp.status_code == 503:
                    wait = 2 ** attempt + random.random()
                    print(f"    503, 等待 {wait:.1f}s 重试 ({attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # 去掉可能的 markdown
                content = content.replace("```json", "").replace("```", "").strip()
                # 解析 JSON
                s, e = content.find("{"), content.rfind("}")
                if s == -1 or e == -1:
                    return None
                return json.loads(content[s:e+1])
            except Exception as e:
                wait = 2 ** attempt + random.random()
                print(f"    错误: {e}, 等待 {wait:.1f}s 重试 ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
    return None


# ============================================================
# 主流程
# ============================================================
async def main():
    print("=" * 70)
    print("大脑训练数据生成（DeepSeek 标注）")
    print("=" * 70)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[错误] 未设置 DEEPSEEK_API_KEY")
        return

    # 1. 读入原始数据
    print("\n[1] 加载原始数据...")
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  共 {len(records)} 条")

    # 2. 过滤出有 user+assistant 的对话
    samples = []
    for r in records:
        convs = r.get("conversations", [])
        # 找第一个 user 和紧接着的 assistant
        for i, c in enumerate(convs):
            if c.get("from") == "human" and i+1 < len(convs) and convs[i+1].get("from") == "gpt":
                samples.append({
                    "user_msg": c["value"],
                    "xiaor_reply": convs[i+1]["value"],
                })
                break
    print(f"  有效样本: {len(samples)}（user+assistant 配对）")

    # 3. 已生成的跳过（断点续传）
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add(r.get("_user_msg_hash", ""))
        print(f"  已生成: {len(done)} 条（跳过）")

    # 4. 并发标注
    print(f"\n[2] 并发标注 (并发数={CONCURRENCY})...")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    output_file = open(OUTPUT_PATH, "a", encoding="utf-8")

    todo = [s for s in samples if hash(s["user_msg"]) not in done]
    print(f"  待生成: {len(todo)} 条")

    success = 0
    fail = 0
    t0 = time.time()

    async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
        # 分批处理，每批 50 条
        batch_size = 50
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start:batch_start+batch_size]
            tasks = [
                label_one(client, api_key, s["user_msg"], s["xiaor_reply"], semaphore)
                for s in batch
            ]
            results = await asyncio.gather(*tasks)

            for s, plan in zip(batch, results):
                if plan is None:
                    fail += 1
                    continue
                # 组装训练样本
                train_sample = {
                    "messages": [
                        {"role": "system", "content": BRAIN_SYSTEM},
                        {"role": "user", "content": f"用户消息：{s['user_msg']}\n情绪状态：{{}}\n历史：[]"},
                        {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                    ],
                    "_user_msg_hash": hash(s["user_msg"]),
                }
                output_file.write(json.dumps(train_sample, ensure_ascii=False) + "\n")
                success += 1

            output_file.flush()
            elapsed = time.time() - t0
            rate = (success + fail) / max(elapsed, 1)
            print(f"  进度: {batch_start+len(batch)}/{len(todo)} | "
                  f"成功 {success} 失败 {fail} | "
                  f"{rate:.1f} 条/秒 | 用时 {elapsed:.0f}s")

    output_file.close()

    print(f"\n[3] 完成！")
    print(f"  成功: {success} 条")
    print(f"  失败: {fail} 条")
    print(f"  输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
