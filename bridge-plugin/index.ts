/**
 * RoleMatrix Bridge Plugin
 *
 * 薄桥接层：注册 OpenClaw 的对话 hook，把 event+ctx 打包成 HookRequest
 * POST 到本地 RoleMatrix Python 服务，把返回的 Result 作为 hook 结果回填。
 *
 * Python 侧负责：人格注入、情绪状态机、记忆、关系成长。
 * 本插件只做转发，不包含业务逻辑。
 *
 * before_agent_reply 特殊处理：
 * - Python 侧返回的 reply.text 可能包含 "\n\n" 分段标记
 * - 第一段作为主 reply 返回给 OpenClaw（走标准 reply 流）
 * - 后续段通过 api.runtime.channel.outbound.loadAdapter 异步发送
 * - 若 loadAdapter 不可用，降级为单条消息（用 "\n" 合并）
 */
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// ---- 与 Python 侧 rolematrix/bridge/contracts.py 对齐的请求体 ----
interface HookContext {
  agent_id?: string;
  session_key?: string;
  session_id?: string;
  channel?: string;
  chat_id?: string;
  sender_id?: string;
  sender_is_owner?: boolean;
  trigger?: string;
  model_id?: string;
  model_provider_id?: string;
}

interface HookRequest {
  hook: string;
  event: Record<string, unknown>;
  ctx: HookContext;
}

interface BridgeConfig {
  endpoint?: string;
  timeoutMs?: number;
}

// 段落分隔符：与 Python 侧 dual_layer.py 的 build_mouth_system_prompt 约定一致
// Python 侧指示 LLM 用 "\n\n" 分段，每段是一条独立消息
const SEGMENT_SEPARATOR = "\n\n";

// 图片占位符标记：OpenClaw 在 cleanedBody 中注入此标记表示有图片附件
// 见 _ref/openclaw/extensions/telegram/src/bot-message-context.body.ts formatMediaPlaceholderText
const MEDIA_PLACEHOLDER_PATTERN = /\[media attached:[^\]]*\]/i;

// 从 OpenClaw 的 ctx 提取桥接所需字段
function extractCtx(ctx: any): HookContext {
  return {
    agent_id: ctx?.agentId,
    session_key: ctx?.sessionKey,
    session_id: ctx?.sessionId,
    channel: ctx?.channel,
    chat_id: ctx?.chatId,
    sender_id: ctx?.senderId,
    sender_is_owner: ctx?.senderIsOwner,
    trigger: ctx?.trigger,
    model_id: ctx?.modelId,
    model_provider_id: ctx?.modelProviderId,
  };
}

// POST 到 Python 服务，返回 hook Result
async function forward(
  cfg: BridgeConfig,
  req: HookRequest,
  log?: { warn: (msg: string) => void; error: (msg: string) => void },
): Promise<Record<string, unknown>> {
  const endpoint = cfg.endpoint ?? "http://127.0.0.1:8765";
  const timeoutMs = cfg.timeoutMs ?? 2000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${endpoint}/hook`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
      signal: controller.signal,
    });
    if (!res.ok) {
      log?.warn(`RoleMatrix bridge HTTP ${res.status} for hook ${req.hook}`);
      return {};
    }
    return (await res.json()) as Record<string, unknown>;
  } catch (err) {
    // Python 服务不可达/超时时静默放行，不阻断 OpenClaw 主流程，但记录便于排查
    const reason = (err as Error)?.name === "AbortError" ? "超时" : "不可达";
    log?.warn(`RoleMatrix bridge ${reason}（hook=${req.hook}）: ${(err as Error)?.message ?? err}`);
    return {};
  } finally {
    clearTimeout(timer);
  }
}

// 拆分回复文本为多段。
// 兼容多种 LLM 输出形式：
// 1. 真换行符 \n\n（理想情况，按两次回车键产生）
// 2. 字面字符串 "\\n\\n"（4字符：反斜杠n反斜杠n，LLM 字面复制提示词）
// 3. 单换行 \n（兜底，至少能分段）
function splitReplySegments(text: string): string[] {
  if (!text) return [];
  let normalized = text;
  // 字面字符串 \n\n → 真换行符
  if (normalized.includes("\\n\\n")) {
    normalized = normalized.replace(/\\n\\n/g, "\n\n");
  }
  // 字面字符串 \n → 真换行符（仅当不存在真换行时才替换，避免误伤）
  if (!normalized.includes("\n") && normalized.includes("\\n")) {
    normalized = normalized.replace(/\\n/g, "\n");
  }
  return normalized
    .split(/\n\s*\n/) // 按空行（可能含空白）分隔
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

// 异步发送后续段（fire-and-forget，失败仅日志）
async function sendExtraSegments(
  apiRef: any,
  log: { warn: (m: string) => void; error: (m: string) => void; info?: (m: string) => void } | undefined,
  segments: string[],
  ctx: HookContext,
): Promise<void> {
  if (segments.length === 0) return;
  const channel = ctx.channel ?? "openclaw-weixin";
  const to = ctx.sender_id ?? "";
  if (!to) {
    log?.warn("sendExtraSegments: 缺少 sender_id，跳过后续段发送");
    return;
  }

  // 1. 尝试加载 channel outbound adapter
  let adapter: any = undefined;
  try {
    if (apiRef?.runtime?.channel?.outbound?.loadAdapter) {
      adapter = await apiRef.runtime.channel.outbound.loadAdapter(channel);
    }
  } catch (e) {
    log?.warn(`loadAdapter(${channel}) 异常: ${(e as Error)?.message ?? e}`);
  }

  if (!adapter || typeof adapter.sendText !== "function") {
    log?.warn(
      `channel ${channel} outbound adapter 不可用或无 sendText，后续段将被丢弃（段数=${segments.length}）`,
    );
    return;
  }

  // 2. 依次发送后续段（每段间隔 300ms，模拟真人打字）
  const cfg = apiRef?.config;
  for (let i = 0; i < segments.length; i++) {
    try {
      await new Promise((r) => setTimeout(r, 300));
      await adapter.sendText({ cfg, to, text: segments[i] });
      log?.info?.(`已发送后续段 ${i + 2}/${segments.length + 1}: "${segments[i].slice(0, 20)}..."`);
    } catch (e) {
      log?.error(`发送后续段 ${i + 2} 失败: ${(e as Error)?.message ?? e}`);
    }
  }
}

// 需要桥接的 hook 列表（与 Python 侧 SUPPORTED_HOOKS 对齐）
const BRIDGED_HOOKS = [
  "before_prompt_build",
  "agent_turn_prepare",
  "heartbeat_prompt_contribution",
  "llm_output",
  "before_agent_reply",
] as const;

// 本地视觉模型配置（图片消息时 override 到此模型）
// Ollama 中已下载 minicpm-v:latest（5.5GB）
const VISION_MODEL = "minicpm-v:latest";
const VISION_PROVIDER = "ollama";

export default definePluginEntry({
  id: "rolematrix-bridge",
  name: "RoleMatrix Bridge",
  description: "将 OpenClaw 对话 hook 转发给本地 RoleMatrix 人格运行时",

  register(api) {
    const cfg = (api.pluginConfig ?? {}) as BridgeConfig;
    const log = api.logger as
      | { warn: (m: string) => void; error: (m: string) => void; info?: (m: string) => void }
      | undefined;
    // 保存 api 引用，用于 before_agent_reply 的后续段发送
    const apiRef = api;
    // hook 等待预算：小R 双层架构（大脑决策 + 嘴巴生成）首条消息冷启动可能需 120s+，留 180s
    // 后续消息走单进程单例，会快很多（10-30s）
    const hookTimeout = (cfg.timeoutMs ?? 180000) + 500;

    // ===== before_model_resolve hook：检测图片时 override 到本地视觉模型 =====
    // 时序：此 hook 在图片加载前触发，返回的 override 会作用于后续 LLM 调用
    // 详见 _ref/openclaw/src/plugins/hook-before-agent-start.types.ts
    if (typeof api.on === "function") {
      api.on(
        "before_model_resolve" as any,
        async (event: any, _ctx: any) => {
          const attachments = event?.attachments;
          const hasImage = Array.isArray(attachments) &&
            attachments.some((a: any) => a?.kind === "image");
          if (hasImage) {
            log?.info?.(
              `before_model_resolve: 检测到图片附件，override 到 ${VISION_PROVIDER}/${VISION_MODEL}`,
            );
            return {
              modelOverride: VISION_MODEL,
              providerOverride: VISION_PROVIDER,
            };
          }
          return {};
        },
        { timeoutMs: 5000 },
      );
    }

    for (const hookName of BRIDGED_HOOKS) {
      const options: Record<string, unknown> = { timeoutMs: hookTimeout };
      // before_agent_reply 只响应 user 触发，避免吞掉 cron/heartbeat
      if (hookName === "before_agent_reply") {
        options.eligibleTriggers = ["user"];
      }

      api.on(
        hookName as any,
        async (event: any, ctx: any) => {
          const req: HookRequest = {
            hook: hookName,
            event: event ?? {},
            ctx: extractCtx(ctx),
          };

          // before_agent_reply 图片消息特殊处理：
          // 检测到 cleanedBody 含 [media attached: ...] 时，不接管，让默认 agent 处理图片
          // 因为 before_agent_reply hook 的 event 没有 image 字段，接管会让图片完全丢失
          if (hookName === "before_agent_reply") {
            const cleanedBody = (event as any)?.cleanedBody ?? "";
            if (cleanedBody && MEDIA_PLACEHOLDER_PATTERN.test(cleanedBody)) {
              log?.info?.(
                `before_agent_reply: 检测到图片消息（含 [media attached:] 占位符），不接管，交由默认 agent 处理`,
              );
              return {}; // 不接管，走 OpenClaw 默认 agent + before_model_resolve override
            }
          }

          const result = await forward(cfg, req, log);

          // before_agent_reply 特殊处理：拆分多段回复
          if (
            hookName === "before_agent_reply" &&
            result &&
            result.handled === true &&
            result.reply &&
            typeof (result.reply as any).text === "string"
          ) {
            const replyObj = result.reply as { text: string };
            const segments = splitReplySegments(replyObj.text);

            log?.info?.(
              `before_agent_reply reply: 段数=${segments.length} 原文repr=${JSON.stringify(replyObj.text.slice(0, 200))}`,
            );

            if (segments.length > 1) {
              const firstSegment = segments[0];
              const extraSegments = segments.slice(1);

              // 同步检查 loadAdapter 是否可用（不 await，只检查存在性）
              const canLoadAdapter =
                typeof apiRef?.runtime?.channel?.outbound?.loadAdapter === "function";

              if (canLoadAdapter) {
                // loadAdapter 可用：第一段作为主 reply，后续段异步发送
                Promise.resolve()
                  .then(() => sendExtraSegments(apiRef, log, extraSegments, req.ctx))
                  .catch((e) =>
                    log?.error(`sendExtraSegments 异常: ${(e as Error)?.message ?? e}`),
                  );

                log?.info?.(
                  `before_agent_reply 拆分: 主段="${firstSegment.slice(0, 20)}..." + ${extraSegments.length} 后续段`,
                );

                return {
                  handled: true,
                  reply: { text: firstSegment },
                  reason: result.reason ?? "rolematrix-bridge: 小R 双层架构接管（分段）",
                };
              } else {
                // loadAdapter 不可用：降级合并为单条消息（用 \n 分隔）
                const merged = segments.join("\n");
                log?.warn?.(
                  `loadAdapter 不可用，降级合并 ${segments.length} 段为单条消息`,
                );
                return {
                  handled: true,
                  reply: { text: merged },
                  reason: result.reason ?? "rolematrix-bridge: 小R 双层架构接管（合并降级）",
                };
              }
            }
            // 单段：原样返回
            return result;
          }

          return result;
        },
        options,
      );
    }
  },
});
