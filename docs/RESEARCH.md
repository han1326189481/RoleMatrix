# RoleMatrix 调研记录（2026-08-05）

> 本文档记录项目扩大过程中做过的外部调研结论与决策，供后续参考。
> 调研时间：2026-08-05。所有结论均有来源 URL 支撑，未经本机实测的项已标注。

## 一、微信通道调研：个人微信号做 AI 机器人

**结论：不推荐走个人微信。官方从未开放个人号自动化消息接口，唯一路径是灰色付费方案。**

- 《微信个人账号使用规范》（2026-04-29 更新）1.2.4/1.2.6 禁止第三方工具自动化接入微信，3.4 对高频收发可封号
  - 来源：https://weixin.qq.com/agreement/personal_account?lang=zh_CN
- 微信对话开放平台（openai.weixin.qq.com）多年无更新，且只服务公众号/小程序，不服务个人号
- wechaty web 协议已事实死亡（2017 年后注册账号无法登录 Web 微信）：
  - 来源：https://github.com/wechaty/puppet-wechat
- 唯一可行路径 = wechaty + **PadLocal/Paimon**（pad 协议，付费 token，7 天免费试用，月费官方不公示）：
  - https://wechaty.js.org/docs/puppet-services/ 、https://github.com/padlocal/wechaty-puppet-padlocal
- 合规替代：企业微信智能机器人（5.0.9 新增）；当前主渠道继续用已跑通的 Telegram

**决策：微信通道暂停，不投入。** 若未来要试，只试 PadLocal 7 天试用（风险自担）。

## 二、代码索引：jcodemunch-mcp 调研

**结论：项目对症且活跃，个人/非商用免费，但商用需付费许可（$79+）。**

- 仓库：https://github.com/jgravelle/jcodemunch-mcp （作者 jgravelle，非 org）
- 功能：tree-sitter 符号级代码索引（search_symbols / get_symbol_source / find_importers / get_blast_radius / get_class_hierarchy / find_dead_code / get_repo_map 等），支持 Python，PyPI 包 `jcodemunch-mcp`，Python 3.10+，OS 无关（Windows 可用）
- 集成：任何 MCP 客户端；`pip install jcodemunch-mcp` 后 `index <dir>` 建索引，`watch` 增量更新
- **许可：非 OSI 开源，个人/非商用免费，商用 $79/$349/$1999 一次性**
- 同类免费替代（如不接受许可）：lemon07r/Vera（MIT）、flupkede/codesearch（Apache-2.0）、Mathews-Tom/archex（Apache-2.0）

**决策：先试用（个人用途免费），验证索引效果后决定是否长期采用。**
**注意：本环境已有 Reasonix 内置 `.code-index/` + explore 子代理，先对比内置能力再引入。**

## 三、生图模型调研（RTX 4060 Laptop 8GB VRAM 硬约束）

**结论：8GB 显存上二次元角色图的最佳组合 = Anything-v5（SD1.5 轻量主力）+ Animagine XL 3.1（SDXL 画质档）。FLUX 原版跑不动。**

| 模型 | 权重 | 8GB 适配 | 速度预估(4060Laptop) | 风格 |
|---|---|---|---|---|
| Anything-v5（SD1.5） | ~2GB fp16 | ✅ 运行 3-4GB | 512px 20 步 ≈ 2-4s | 二次元 |
| Animagine XL 3.1（SDXL） | ~6.5GB 全家桶 | ⚠️ 紧，需 offload | 1024px 28 步 ≈ 10-20s | 二次元天花板 |
| FLUX.1-schnell 原版 | 23.8GB BF16 | ❌ 不可行 | — | 写实 |
| FLUX.1-schnell GGUF Q4 | 6.78GB | ⚠️ 勉强+慢 | 4 步 30-90s | 写实为主 |
| SD-Turbo / SDXL-Turbo | 小 | ✅ | 快 | ❌ 模型卡明示人物不佳 |

- 模型仓库：Anything-v5 已迁至 `genai-archive/anything-v5`（HF）；Animagine XL 3.1 = `cagliostrolab/animagine-xl-3.1`（支持 832×1216 竖构图，适合"自拍"比例）
- Counterfeit / Hassaku XL / WAI-NSFW-illustrious 主要在 CivitAI 分发，HF 无官方仓库（勿走镜像）
- 框架：diffusers 集成进现有 `.venv`（与 RoleMatrix Python 架构一致）或 ComfyUI 便携包（独立环境不污染 .venv）
- 权重下载走 hf-mirror（`HF_ENDPOINT` 已设），落 D 盘

**决策：生图方案 = diffusers + Anything-v5（主力）+ Animagine XL 3.1（画质档）。下一步做 POC 实测。**

## 四、独立收藏库 + 自主 web search —— 已实现（2026-08-05 上一轮落地）

- 独立 SQLite：`.xiaor_collection/collection.db`（与主记忆库完全隔离），图片存 `.xiaor_collection/images/YYYY-MM/`
- 收藏闭环：大脑 `save_to_collection`（URL/本地图入库）、`send_meme`（按 tag 查表情包 + 使用计数）、用户发图自动收藏，均已接通
- 自主 web search：大脑决策填 `web_search_query` 时自动搜索，限流（每 session 2 次/分）+ 1h query 缓存防滥用，搜索历史入库
- 可增强（非必须）：按 persona 拆列、表情包独立分类（source="meme"）

## 五、执行状态追踪

- [x] 2026-08-05 调研完成（微信 / jcodemunch-mcp / 生图模型）
- [x] 生图 POC：Anything-v5 + diffusers 实测出图成功
      → 20 步 512px 生成 **3.1s**，峰值显存 **2.61GB / 8GB**，图片 .tmp/poc_anything_v5.png
      → 首次需下载 ~2GB 权重（hf-mirror，约 15 分钟），之后走本地缓存
      → 结论：生图可行，后续集成 diffusers 到 rolematrix/tools/（小R 自动出图）
- [x] 小R 真人形象 LoRA 训练（2026-08-07）
      → 数据：28 张真人照片（超分 4x 到 1024px+，清洗剔除 3 张模糊/无脸图 → 25 张）
      → 基模：Realistic Vision v5.1（SD1.5，已固定到 models/base/RealisticVision_V5.1）
      → 训练：rank 32 / alpha 64 / 512px center-crop / 1500 steps (2 epoch) / 10.5 分钟
      → 产物：models/lora_xiaor_real_v1（触发词 xiaor，caption "photo of xiaor girl"）
      → 评估（minicpm-v 原图对比）：有 LoRA 相似度 90/100（1 epoch 仅 60），效果显著
      → 脚本：scripts/train_xiaor_real_lora.py / scripts/eval_xiaor_real_lora.py
      → 注意：训练图在 data/lora_xiaor_real/（隐私，不入库，.gitignore 已含 data/）；
        models/ 不入库（.gitignore），换机需重新训练或手动转移
      → 已知限制：眼镜特征未学全（训练图戴眼镜占比不均），可用 caption 标注强化
- [x] jcodemunch-mcp POC：索引 + 检索验证成功
      → rolematrix 包 322 符号索引约 0.45s；search_symbols 精确返回符号(file/line/signature)
      → 关键配置（本地已设置，换机需重配）：
        1. `tool_surface: "full"`（默认 counter 只暴露 6 个前门工具）— 已写入
           C:\Users\13261\.code-index\config.jsonc（serve 读取的全局配置）
           与项目 .jcodemunch.jsonc（双重保险）
        2. MCP serve 与 CLI 的索引存储不同（git vs local identity），
           首次需在 MCP 会话内调用 index_folder(path=...) 建索引
        3. `server_output: "raw"` 让结果输出明文 JSON（adaptive 会 MUNCH 压缩）
      → 结论：可用（个人/非商用免费），接入 Reasonix 时注册为 stdio MCP server
- [ ] 微信通道：暂缓（风险高，无官方接口）
- [ ] 生图集成进 rolematrix（diffusers 封装为 tools/image_gen.py + 小R 决策触发）
- [ ] jcodemunch-mcp 注册进 Reasonix MCP 配置（个人用途）

## 六、本地环境变更（不入库，仅本机）

- 新增 Python 依赖：`diffusers`（生图）、`jcodemunch-mcp`（代码索引）
- 修改 `C:\Users\13261\.code-index\config.jsonc`：tool_surface counter→full
- 修改 `D:\RoleMatrix\.code-index\config.jsonc`：tool_surface counter→full
- 新增 `D:\RoleMatrix\.jcodemunch.jsonc`（入库）：tool_surface=full + server_output=raw
- 生成图片 `.tmp/poc_anything_v5.png`（不入库）
