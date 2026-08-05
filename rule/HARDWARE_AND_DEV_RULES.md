# RoleMatrix 硬件配置与开发规则

> 本文档是 RoleMatrix 项目的硬件约束与开发准则，所有模型选择、训练参数、
> 运行时配置必须先对照本文档，不得擅自下载超出硬件能力的模型。

---

## 一、硬件配置（不可逾越的硬约束）

| 项目 | 配置 | 备注 |
|------|------|------|
| GPU | NVIDIA RTX 4060 Laptop | 8GB VRAM |
| 内存 | 16GB | 系统总内存，Ollama + 训练 + IDE 共享 |
| 磁盘 | D 盘剩余 ~114GB（C 盘仅剩 12.93GB） | **C 盘爆红，所有缓存必须走 D 盘** |
| OS | Windows 11 | PowerShell 环境 |
| Python | 3.11.9 (.venv) | **必须用 `D:\RoleMatrix\.venv\Scripts\python.exe`**，默认 `python` 是 3.9 无 torch |
| CUDA | 12.4 | torch 2.6.0+cu124 |
| 设备类型 | 普通笔记本 | 非工作站，无外接显卡 |

### 硬约束规则
1. **路径硬约束**：所有模型、缓存、下载文件必须存放在 `D:\RoleMatrix\` 下，**禁止写入 C 盘任何位置**（详见第二节"路径硬约束"）
2. **推理**：单模型权重 ≤ 6GB（Q4 量化），7B Q4(4.7GB) 完全可用
3. **训练**：QLoRA 峰值显存 ≤ 8GB，用 4bit 预量化权重避免下载 15GB 原权重
4. **内存**：下载/加载阶段 ≤ 16GB，**禁止下载 fp16/bf16 原权重 7B+ 模型**（15GB 超内存）
5. **同时运行进程**：训练时关掉 Ollama 同模型，避免显存冲突
6. **训练方案**：用 unsloth 4bit 预量化权重（如 `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`，~5GB）
7. **Python 解释器**：禁止用系统默认 `python`（3.9.0 无 torch），必须用 `.venv\Scripts\python.exe`

---

## 二、路径硬约束（红线，不可逾越）

> **背景**：C 盘仅剩 12.93GB，曾因未设置环境变量导致 HF 缓存默认写入 `~/.cache/huggingface` 撑爆 C 盘，且代理上 HF Hub 限流导致下载卡死（17 分钟 0 增量）。
> **本节将所有下载路径写死到 D 盘，并通过 Windows 用户级环境变量持久化，确保任何新进程都继承 D 盘路径。**

### 2.1 环境变量清单（已通过 `[Environment]::SetEnvironmentVariable(..., "User")` 持久化）

| 变量名 | 值 | 用途 |
|--------|-----|------|
| `HF_HOME` | `D:\RoleMatrix\.hf_cache` | HuggingFace 根缓存 |
| `HUGGINGFACE_HUB_CACHE` | `D:\RoleMatrix\.hf_cache\hub` | HF Hub 模型权重缓存 |
| `HF_DATASETS_CACHE` | `D:\RoleMatrix\.hf_cache\datasets` | HF 数据集缓存 |
| `TRANSFORMERS_CACHE` | `D:\RoleMatrix\.hf_cache` | Transformers 缓存 |
| `OLLAMA_MODELS` | `D:\RoleMatrix\.ollama\models` | Ollama 本地模型存储 |
| `PIP_CACHE_DIR` | `D:\RoleMatrix\.pip_cache` | pip 下载缓存 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像（避免直连 HF Hub 限流卡死） |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片 |

### 2.2 目录结构（全部位于 D 盘）

```
D:\RoleMatrix\
├── .hf_cache\                          # HuggingFace 所有缓存
│   ├── hub\                            # 模型权重（unsloth/Qwen2.5-7B-Instruct-bnb-4bit 等）
│   └── datasets\                       # 数据集
├── .ollama\models\                     # Ollama 模型（blobs + manifests）
├── .pip_cache\                         # pip 下载缓存
├── models\                             # 训练产出的 LoRA adapter（lora_xiaor_v1 等）
├── data\datasets\                      # 原始/清洗后训练数据
└── scripts\                            # 脚本和训练日志 train_log_*.log
```

### 2.3 Python 脚本必须在文件顶部 `import torch` 之前包含以下代码

```python
import os
# === 路径硬约束：所有缓存写死到 D 盘，禁止 C 盘 ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\RoleMatrix\.hf_cache\hub"
os.environ["HF_DATASETS_CACHE"] = r"D:\RoleMatrix\.hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\RoleMatrix\.hf_cache"
os.environ["OLLAMA_MODELS"] = r"D:\RoleMatrix\.ollama\models"
os.environ["PIP_CACHE_DIR"] = r"D:\RoleMatrix\.pip_cache"
os.environ["HF_ENDPOINT"] = r"https://hf-mirror.com"  # 国内镜像
```

### 2.4 下载模型的正确姿势（避免卡死）

**禁止**直接 `from_pretrained()` 触发隐式下载（无进度条、卡死难发现）。
**必须**先用 `hf download` 单独下载（有进度条、可断点续传）：

```powershell
# 注意：huggingface_hub 1.26+ 已废弃 huggingface-cli，改用 hf
$env:HF_ENDPOINT = "https://hf-mirror.com"
& "D:\RoleMatrix\.venv\Scripts\hf.exe" download "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
```

下载完成后（exit code 0），再启动训练脚本，此时 `from_pretrained()` 会直接命中本地缓存，不再联网。

### 2.5 验证命令（每次新会话开始前先跑一遍）

```powershell
# 1. 验证环境变量都已继承 D 盘
echo "HF_HOME: $env:HF_HOME"
echo "OLLAMA_MODELS: $env:OLLAMA_MODELS"
echo "HF_ENDPOINT: $env:HF_ENDPOINT"

# 2. 验证 C 盘无残留
if (Test-Path "$env:USERPROFILE\.cache\huggingface") { "警告: C盘有HF残留" } else { "OK: C盘干净" }
if (Test-Path "$env:USERPROFILE\.ollama") { "警告: C盘有Ollama残留" } else { "OK: C盘干净" }

# 3. 验证 Ollama 能识别 D 盘模型
ollama list
```

### 2.6 禁止事项

1. **禁止**使用默认 C 盘路径：`~/.cache/huggingface`、`~/AppData/Local/pip/Cache`、`~/.ollama`
2. **禁止**下载时不设置环境变量就启动（脚本顶部必须有 2.3 节的代码）
3. **禁止**使用已废弃的 `HF_HUB_ENABLE_HF_TRANSFER`（已改为 `HF_XET_HIGH_PERFORMANCE`，但默认已够用，无需设置）
4. **禁止**直连 HF Hub 不走镜像（`HF_ENDPOINT` 必须设为 `https://hf-mirror.com`）
5. **禁止**用系统默认 `python`（3.9.0 无 torch），必须用 `D:\RoleMatrix\.venv\Scripts\python.exe`

---

## 三、当前已部署模型清单

### Ollama 本地模型（位于 `D:\RoleMatrix\.ollama\models`，推理用）
| 模型 | 大小 | 用途 |
|------|------|------|
| qwen2.5:7b (Q4) | 4.7GB | 备用推理（单层 single 模式降级用） |
| qwen2.5:1.5b | 986MB | 轻量任务/结构化决策候选 |
| phi4-mini:3.8b | 2.5GB | 备用推理 |
| minicpm-v:latest | 5.5GB | 视觉模型（图片输入） |

### HuggingFace 缓存（位于 `D:\RoleMatrix\.hf_cache\hub`）
| 路径 | 状态 |
|------|------|
| `models--unsloth--Qwen2.5-7B-Instruct-bnb-4bit` | 训练基模（4bit 预量化，~5GB），**已下载到本地** |
| `models--Qwen--Qwen2.5-7B-Instruct` | **已删除**（bf16 原权 15GB 超内存） |

### 训练基模本地路径（手动下载，避免代理卡死）
- 路径：`D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit\`
- 关键文件：`model.safetensors`（5.29 GB，单文件权重，非分片）+ config.json + tokenizer.json
- 训练脚本 `BASE_MODEL` 已指向此本地路径，不再联网

### LoRA Adapter 清单（训练产物，位于 `D:\RoleMatrix\models\`）

| Adapter 路径 | 用途 | 训练数据 | 格式 | 使用位置 |
|---|---|---|---|---|
| `models/lora_xiaor_v1/` | **说话风格 LoRA**：让模型输出小R风格的自然语言（短句、口语化、技术小白人设） | `rolematrix_sft_final.jsonl`（1765 条 ShareGPT 对话，含真实聊天+仿写） | ShareGPT 多轮对话 | 测试脚本 `test_lora_compare.py` A 模式；暂未集成进生产 |
| `models/lora_brain_v1/` | **大脑决策 LoRA**：让模型输出结构化 JSON 决策（emotion_delta + memory_recall + reply_plan） | `rolematrix_brain_train.jsonl`（1761 条 instruction，DeepSeek 标注） | instruction (system+user+assistant JSON) | **生产双层架构大脑**：`rolematrix/llm/brain_provider.py` |

**两个 adapter 的区别**：
- `lora_xiaor_v1` 学的是"小R 怎么说话"——直接输出短句自然语言
- `lora_brain_v1` 学的是"小R 怎么想"——输出 JSON 决策给 DeepSeek 嘴巴参考

**不能混用**：`lora_xiaor_v1` 不会输出 JSON；`lora_brain_v1` 直出文字会很奇怪（只会输出 JSON 格式）。

**训练参数对比**：

| 参数 | lora_xiaor_v1 | lora_brain_v1 |
|---|---|---|
| epochs | 2 | 3（结构化输出更难） |
| max_seq_length | 512 | 768（JSON 输出更长） |
| 最终 loss | 0.247 | 0.135 |
| token_accuracy | 96.33% | 97.88% |
| 训练耗时 | 2 小时 | 4.6 小时 |
| adapter 大小 | 78.9 MB | 78.9 MB |

---

## 四、核心开发思路：大脑 + 嘴巴 双层架构

### 架构图
```
用户消息
  ↓
┌─────────────────────────────────────────┐
│  本地 LoRA 大脑 (lora_brain_v1)          │
│  基模：Qwen2.5-7B-Instruct-bnb-4bit     │
│  职责：                                 │
│  1. 根据用户消息+情绪+历史输出 JSON 决策 │
│  2. JSON 含 emotion_delta / reply_plan │
│  3. 失败时返回 fallback 默认策略          │
└─────────────────────────────────────────┘
  ↓ 把决策 JSON 拼到 system prompt
┌─────────────────────────────────────────┐
│  DeepSeek API = 嘴巴                    │
│  职责：                                 │
│  1. 拿到大脑给出的要点 + 风格            │
│  2. 生成自然流畅的最终回复               │
│  3. 拆分成多条短消息（模拟打字发送）      │
└─────────────────────────────────────────┘
  ↓
用户收到回复
```

### 模式开关（config.yaml `llm.mode`）

| mode | 流程 | 用途 |
|---|---|---|
| `single` | 直接走 `llm.provider`（Ollama 或 DeepSeek） | 默认/降级/调试 |
| `dual` | brain LoRA 决策 → mouth 生成 | **生产模式**（默认开启） |

### 生产代码集成位置

| 文件 | 职责 |
|---|---|
| [rolematrix/llm/brain_provider.py](file:///d:/RoleMatrix/rolematrix/llm/brain_provider.py) | 大脑单例封装：4bit base + brain LoRA + 空 JSON fallback |
| [rolematrix/bridge/dual_layer.py](file:///d:/RoleMatrix/rolematrix/bridge/dual_layer.py) | 双层编排器：brain.decide() → build_mouth_system_prompt() → mouth.chat() |
| [rolematrix/bridge/chat.py](file:///d:/RoleMatrix/rolematrix/bridge/chat.py) | chat 端点：根据 `mode` 分发到单层或双层，含三级降级 |
| [rolematrix/config.py](file:///d:/RoleMatrix/rolematrix/config.py) | LLMConfig 加 mode/brain_base_model/brain_lora_path 字段 |

### 降级策略（chat.py 三级降级）

1. **dual 模式大脑失败** → 自动降级到单层 `llm.provider`，brain_plan 走 fallback
2. **DeepSeek 嘴巴失败** → 降级到本地 Ollama
3. **Ollama 也失败** → 抛 HTTP 500

### 空 JSON fallback 机制（brain_provider.py）

大脑可能输出空 `{}` 或非法 JSON（测试中 5/5 有 1 次失败）。`_parse_brain_json()` 检测到以下情况返回 fallback：
- 输出为空
- JSON 解析失败
- 解析成功但缺 `reply_plan` 字段
- 空 `{}` 对象

fallback 策略：`tone=日常闲聊, points=["简短自然回应"], length=short`，带 `_fallback=True` 标记，调用方可识别。

### 设计原则
1. **真人对话模拟**：大脑先"想一下"（回忆+判断情绪+组织语言），DeepSeek 再"开口说"
2. **不追求回复速度**：真实聊天情景，可以慢一点
3. **隐私优先**：用户画像和历史记忆不出本机
4. **成本可控**：大脑做结构化决策（token 少），DeepSeek 只负责最终文字生成
5. **稳定优先**：大脑失败有 fallback，嘴巴失败有降级，绝不裸抛

---

## 五、训练策略（8GB VRAM 训练 7B QLoRA）

### 关键结论
7B Q4 当大脑推理完全可行（4.7GB，已部署）。训练 7B 也可行，但要用 **4bit 预量化权重**避免下载 15GB 原权重撑爆 16GB 内存。

### 训练方案：transformers + bnb 4bit（放弃 unsloth，详见错误记录 #3）
- 基模：`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`（~5GB，已量化好）
- 框架：transformers 5.5.0 + peft 0.20.0 + trl 1.9.2 + bnb 0.50.0
- 显存占用：~6-7GB（8GB VRAM 有余量）
- 内存占用：~6GB（16GB 内存宽裕）
- 脚本：`scripts/train_xiaor_lora.py`（错误记录见脚本顶部注释，共 6 条）

### 训练数据
- `data/datasets/cleaned/rolematrix_sft_final.jsonl`（1765 条）
- ShareGPT 格式，含 system(人格+情绪) + human/gpt 多轮

### 训练参数（通用模板，按方案调整）
```
LoRA rank: 16, alpha: 32
learning_rate: 2e-4
batch_size: 1, grad_accum: 4
max_seq_length: 1024
epochs: 3
bf16: True
gradient_checkpointing: True
optim: paged_adamw_8bit
```

---

## 六、禁止事项（红线）

1. **禁止**写入 C 盘任何缓存位置（详见第二节"路径硬约束"）
2. **禁止**下载 fp16/bf16 原权重 7B+ 模型（15GB 超 16GB 内存）——必须用 4bit 预量化版本
3. **禁止**训练时同时跑 Ollama 同模型（显存冲突）
4. **禁止**直连 HF Hub 不走镜像（必须设 `HF_ENDPOINT=https://hf-mirror.com`）
5. **禁止**用系统默认 `python`（3.9.0），必须用 `.venv\Scripts\python.exe`
6. **禁止**擅自决定模型规模——必须对照本规则文档

---

## 七、待办

- [x] 删除未下完的 HF 缓存（C 盘残留已清理）
- [x] 用 transformers + bnb 重写训练脚本（放弃 unsloth，详见错误记录 #3）
- [x] 路径硬约束写入 rule 文档，环境变量持久化到用户级
- [x] Ollama 服务重启并验证 `ollama list` 识别 D 盘模型
- [x] 下载 `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`（走 hf-mirror，避免卡死）
- [x] 启动 7B QLoRA 训练（说话风格 lora_xiaor_v1，2 epochs，loss 0.247）
- [x] 训练大脑决策 LoRA（lora_brain_v1，3 epochs，loss 0.135，token_acc 97.88%）
- [x] 双层架构端到端测试验证（5/5 通过 4 个，1 个 fallback 降级）
- [x] 双层架构集成进生产代码（brain_provider.py + dual_layer.py + chat.py）
- [x] 空 JSON fallback 机制实现（`_parse_brain_json` + DEFAULT_PLAN）
- [ ] 端到端集成测试验证（启动 FastAPI 服务，HTTP 调用 /chat 接口）
- [ ] 接入 OpenClaw 桥接插件，真实场景压测
