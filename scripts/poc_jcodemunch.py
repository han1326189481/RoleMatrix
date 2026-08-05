"""jcodemunch-mcp POC v2：验证 tier 分层机制下如何开放检索工具。

流程：连接 serve → 打印 set_tool_tier/menu schema → 升级 tier → 重新列工具
→ 调用 search_symbols 查真实符号。
"""
from __future__ import annotations

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _call(session: ClientSession, name: str, args: dict):
    res = await session.call_tool(name, args)
    return "".join(c.text for c in res.content if hasattr(c, "text"))


async def main() -> None:
    server = StdioServerParameters(
        command=r"D:\RoleMatrix\.venv\Scripts\python.exe",
        args=["-m", "jcodemunch_mcp", "serve"],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"初始工具数: {len(names)} -> {names}", flush=True)

            # 看 tier 工具的 schema
            for tn in ("set_tool_tier", "menu"):
                t = next((x for x in tools.tools if x.name == tn), None)
                if t:
                    print(f"\n{tn} schema: {t.inputSchema}", flush=True)

            # 尝试升级 tier（先试 full，失败再试 core/standard）
            for tier in ("full", "standard", "core"):
                try:
                    r = await _call(session, "set_tool_tier", {"tier": tier})
                    print(f"\nset_tool_tier('{tier}') => {r[:200]}", flush=True)
                    tools2 = await session.list_tools()
                    names2 = sorted(t.name for t in tools2.tools)
                    print(f"升级后工具数: {len(names2)}", flush=True)
                    if "search_symbols" in names2:
                        break
                except Exception as e:  # noqa: BLE001
                    print(f"set_tool_tier('{tier}') 失败: {e}", flush=True)

            # MCP server 与 CLI 索引存储不同，需在会话内 index_folder 建索引
            repo_id = None
            try:
                idx = await _call(
                    session, "index_folder",
                    {"path": r"D:\RoleMatrix\rolematrix"},
                )
                print(f"index_folder => {idx[:300]}", flush=True)
                import json as _json

                d = _json.loads(idx)
                repo_id = (
                    d.get("repo") or d.get("repo_id") or d.get("id")
                    or d.get("display_name")
                )
            except Exception as e:  # noqa: BLE001
                print(f"index_folder 失败: {e}", flush=True)
            print(f"解析到 repo_id = {repo_id}", flush=True)

            for q in ("EmotionEngine", "MemoryManager", "dual_chat"):
                for repo in ([repo_id] if repo_id else []) + ["rolematrix"]:
                    if not repo:
                        continue
                    try:
                        r = await _call(
                            session, "search_symbols",
                            {"repo": repo, "query": q},
                        )
                        if '"error"' not in r:
                            print(f"search_symbols(repo={repo}, '{q}') => {r[:400]}", flush=True)
                            break
                    except Exception as e:  # noqa: BLE001
                        print(f"search_symbols(repo={repo}, '{q}') 失败: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
