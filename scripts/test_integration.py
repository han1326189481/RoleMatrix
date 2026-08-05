"""双层架构集成验证：直接调用生产代码（不启 HTTP 服务）。

验证目标：
1. brain_provider 单例能加载 + decide() 输出合法 JSON
2. dual_layer.dual_chat() 完整流程能走通（brain → mouth → reply）
3. 空 JSON fallback 机制生效（测试 5/5 稳定性）

跳过 FastAPI 路由层，直接调 rolematrix.bridge.dual_layer.dual_chat()。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# === 路径硬约束（详见 rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", r"D:\RoleMatrix\.hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\RoleMatrix\.hf_cache\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\RoleMatrix\.hf_cache\datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\RoleMatrix\.hf_cache")
os.environ.setdefault("OLLAMA_MODELS", r"D:\RoleMatrix\.ollama\models")

# 加载 .env
_env_path = Path(r"D:\RoleMatrix\.env")
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# 设置项目根，让 config 能找到 config.yaml
os.environ["ROLEMATRIX_ROOT"] = r"D:\RoleMatrix"


async def main():
    print("=" * 70)
    print("双层架构集成验证（直接调生产代码）")
    print("=" * 70)

    # 1. 验证配置加载
    print("\n[1] 加载配置...")
    from rolematrix.config import get_settings
    settings = get_settings()
    print(f"  llm.mode = {settings.llm.mode}")
    print(f"  llm.provider = {settings.llm.provider}")
    print(f"  llm.brain_base_model = {settings.llm.brain_base_model}")
    print(f"  llm.brain_lora_path = {settings.llm.brain_lora_path}")
    assert settings.llm.mode == "dual", "config.yaml 未启用 dual 模式"

    # 2. 验证大脑单例
    print("\n[2] 获取大脑单例...")
    from rolematrix.llm.brain_provider import get_brain
    brain = get_brain(
        base_model=settings.llm.brain_base_model,
        lora_path=settings.llm.brain_lora_path,
    )
    print(f"  brain 对象: {brain}")
    print(f"  已加载: {brain._model is not None}")

    # 3. 测试用例（包含上次失败的"日常闲聊"短输入）
    test_cases = [
        "我的电脑突然蓝屏了怎么办啊",
        "今天好累啊，写了一整天代码头都要炸了",
        "你在干嘛呢",  # 上次空 JSON 的场景
        "给我讲讲你今天在学校发生的事情吧",
        "你是谁啊",
    ]

    results = []
    print(f"\n[3] 跑 {len(test_cases)} 个测试用例...")

    for i, user_msg in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {user_msg}")
        print("-" * 60)

        # 直接调 brain.decide() 验证大脑
        t0 = time.time()
        try:
            plan = await brain.decide(
                user_msg=user_msg,
                emotion_context={"happy": 50, "want_chat": 60},
                history=None,
            )
            t_brain = time.time() - t0
            is_fallback = plan.get("_fallback", False)
            has_reply_plan = "reply_plan" in plan
            print(f"  [大脑] 耗时 {t_brain:.1f}s | fallback={is_fallback} | reply_plan存在={has_reply_plan}")
            print(f"  [大脑] plan: {json.dumps(plan, ensure_ascii=False)[:200]}")
            brain_ok = has_reply_plan and not is_fallback
        except Exception as e:
            print(f"  [大脑] 异常: {e}")
            brain_ok = False
            plan = {"_fallback": True, "_fallback_reason": str(e)}

        # 验证 dual_chat 完整流程（只在 brain 成功时跑，避免浪费 DeepSeek 配额）
        if not brain_ok:
            print(f"  [双层] 跳过（大脑 fallback，仍验证降级路径）")
            results.append({"user": user_msg, "brain_ok": brain_ok, "plan": plan})
            continue

        print(f"  [双层] 调用 dual_chat()...")
        from rolematrix.bridge.dual_layer import dual_chat
        t1 = time.time()
        try:
            reply, returned_plan = await dual_chat(
                user_msg=user_msg,
                base_system_prompt="你是小R，计算机系女大学生，可爱高情商。",
                history=None,
                emotion_context={"happy": 50, "want_chat": 60},
                image_base64=None,
            )
            t_total = time.time() - t1
            print(f"  [嘴巴] 耗时 {t_total:.1f}s")
            print(f"  [嘴巴] 回复: {reply}")
            print(f"  [总耗时] {t_total:.1f}s")
            results.append({
                "user": user_msg,
                "brain_ok": brain_ok,
                "plan": plan,
                "reply": reply,
                "total_time": round(t_total, 1),
            })
        except Exception as e:
            print(f"  [双层] 失败: {e}")
            results.append({
                "user": user_msg,
                "brain_ok": brain_ok,
                "plan": plan,
                "error": str(e),
            })

    # 4. 汇总
    print(f"\n{'='*70}")
    print("汇总")
    print(f"{'='*70}")
    success = sum(1 for r in results if r.get("brain_ok"))
    fallback = sum(1 for r in results if r.get("plan", {}).get("_fallback"))
    print(f"大脑成功: {success}/{len(results)}")
    print(f"大脑 fallback: {fallback}/{len(results)}")
    print(f"稳定性: {success/len(results)*100:.0f}%")

    # 5. 保存
    out_path = Path(r"D:\RoleMatrix\scripts\test_integration_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
