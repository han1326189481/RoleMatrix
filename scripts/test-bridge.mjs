/**
 * 端到端桥接闭环验证脚本（不依赖 openclaw 运行时）。
 *
 * 复用 bridge-plugin/index.ts 的 forward 逻辑，模拟 OpenClaw 触发各 hook
 * 时桥接插件的行为，验证 TS->HTTP->Python->Result 完整链路。
 *
 * 运行前提：RoleMatrix Python server 已在 http://127.0.0.1:8765 运行。
 * 运行：node scripts/test-bridge.mjs
 */
const ENDPOINT = "http://127.0.0.1:8765";
const TIMEOUT_MS = 2000;

async function forward(hook, event, ctx) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${ENDPOINT}/hook`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ hook, event, ctx }),
      signal: controller.signal,
    });
    if (!res.ok) {
      console.error(`  HTTP ${res.status}`);
      return null;
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function get(path) {
  const res = await fetch(`${ENDPOINT}${path}`);
  return res.json();
}

let pass = 0;
let fail = 0;

function check(name, cond, detail) {
  if (cond) {
    console.log(`  ✅ ${name}`);
    pass++;
  } else {
    console.log(`  ❌ ${name} ${detail ?? ""}`);
    fail++;
  }
}

console.log("=== RoleMatrix 桥接闭环验证 ===\n");

// 1) 健康检查
console.log("[1] 健康检查");
const health = await get("/health");
check("server 健康", health.status === "ok");
check("版本号存在", typeof health.version === "string");

// 2) 人格列表
console.log("\n[2] 人格加载");
const personas = await get("/personas");
check("default 人格存在", personas.personas?.includes("default"));

// 3) before_prompt_build：核心注入点
console.log("\n[3] before_prompt_build（核心注入点）");
const ctx1 = {
  agent_id: "test-agent",
  session_key: "e2e-session",
  channel: "wechat",
  sender_id: "tester",
  trigger: "user",
};
const r1 = await forward("before_prompt_build", { prompt: "你好呀" }, ctx1);
check(
  "返回 prependSystemContext",
  typeof r1?.prependSystemContext === "string" && r1.prependSystemContext.length > 0,
);
check("人格块含'当前人格'", r1?.prependSystemContext?.includes("当前人格"));
check(
  "返回 appendContext",
  typeof r1?.appendContext === "string" && r1.appendContext.includes("情绪"),
);

// 4) 情绪随事件变化（收到消息 happy+5, want_chat-20）
console.log("\n[4] 情绪状态机");
const emo1 = await get("/emotion/e2e-session");
check("happy 在 50-60 区间（基准50+5）", emo1.happy >= 50 && emo1.happy <= 60);
check("want_chat 低于基准60（-20）", emo1.want_chat < 60);

// 5) llm_output：观察类，回复后 want_chat 再降
console.log("\n[5] llm_output（观察类 hook）");
const r2 = await forward(
  "llm_output",
  { assistantTexts: ["嗯…你好呀"], model: "qwen2.5:7b" },
  ctx1,
);
check("返回空 result（观察类）", JSON.stringify(r2) === "{}");
const emo2 = await get("/emotion/e2e-session");
check("want_chat 进一步下降", emo2.want_chat < emo1.want_chat);

// 6) heartbeat：want_chat 较低，暂不贡献主动聊天
console.log("\n[6] heartbeat_prompt_contribution");
const r3 = await forward(
  "heartbeat_prompt_contribution",
  { heartbeatName: "daily" },
  { agent_id: "test-agent", session_key: "e2e-session", trigger: "heartbeat" },
);
// want_chat 经过前面两次下降，可能 <50 返回空，也可能仍 >=50 返回 context
check("返回结构合法", r3 !== null);
if (Object.keys(r3).length === 0) {
  console.log("    （want_chat 不足，未贡献主动聊天 —— 符合预期）");
} else {
  check("包含 appendContext", typeof r3.appendContext === "string");
  console.log(`    （want_chat 足够，贡献了主动聊天信号）`);
}

// 7) 人格绑定切换
console.log("\n[7] 人格绑定");
await fetch(`${ENDPOINT}/personas/assign`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ agent_id: "test-agent", persona: "default" }),
});
const r4 = await forward("before_prompt_build", { prompt: "test" }, {
  agent_id: "test-agent",
  session_key: "bind-test",
});
check("绑定后仍正确注入人格", r4?.prependSystemContext?.includes("当前人格"));

// 8) Python 不可达时的容错（指向错误端口）
console.log("\n[8] 容错：Python 不可达时放行");
const badRes = await fetch("http://127.0.0.1:59999/hook", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: "{}",
}).catch(() => null);
check("不可达返回 null（插件应静默放行）", badRes === null);

console.log(`\n=== 结果：${pass} 通过，${fail} 失败 ===`);
process.exit(fail > 0 ? 1 : 0);
