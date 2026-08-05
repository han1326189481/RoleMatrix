"""人格数据模型，对应 personas/*.yaml 的结构。

设计原则（多角色矩阵 AI 伴侣平台）：
1. 字段分层：核心身份层（所有角色共享）+ 沉浸式扩展层（角色卡深度扮演）
2. 向后兼容：旧版只有 tone/traits 的 YAML 仍能加载，自动降级为简化渲染
3. 可扩展：新增角色只需要写 YAML，无需改代码
4. 渲染分离：YAML 存结构化数据，to_prompt_block 负责转成 LLM 能"演绎"的文本

字段分三层：
- L1 基础层：name/display_name/description/greeting（所有角色必须有）
- L2 风格层：tone/traits/quirks/favorite_topics/reply（语气控制）
- L3 沉浸层：appearance/background/voice/speech_patterns/forbidden_phrases/
            example_dialogues/relationship（角色卡深度扮演，可选）
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PersonaStyle(BaseModel):
    """L2 语气风格层。"""
    tone: str = "温和"
    traits: list[str] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)


class PersonaCapabilities(BaseModel):
    memory: bool = True
    emoji: bool = True
    proactive_chat: bool = True
    voice: bool = False


class PersonaReply(BaseModel):
    # short | medium | long
    length: str = "medium"
    # instant | slow | random
    typing_speed: str = "random"


class SpeechPattern(BaseModel):
    """L3 语言习惯：让角色"说人话"的关键。"""
    # 句尾口癖，如 "…呢"、"…吧"
    sentence_endings: list[str] = Field(default_factory=list)
    # 常用语气词，如 "嗯…"、"那个"
    fillers: list[str] = Field(default_factory=list)
    # 自称，如 "人家"、"本小姐"、"我"
    self_reference: str = "我"
    # 对用户的称呼，如 "你"、"主人"、"亲爱的"
    user_address: str = "你"
    # 频繁使用的表情/颜文字
    frequent_emojis: list[str] = Field(default_factory=list)


class ExampleDialogue(BaseModel):
    """L3 示例对话：教 LLM "怎么演"而非"怎么描述"。

    one-shot 效果远强于抽象标签 —— 模型会模仿语气而非机械执行规格。
    """
    user: str
    assistant: str


class Relationship(BaseModel):
    """L3 关系状态：支持 AI 伴侣/搭档/助手等多种定位。"""
    # companion | partner | assistant | friend | sibling | custom
    type: str = "companion"
    # 当前亲密等级 0-100
    intimacy: int = 30
    # 称呼用户的方式（覆盖 speech_patterns.user_address 也可）
    pet_name: str | None = None
    # 关系进展提示（注入 prompt 帮助 LLM 把握分寸）
    progress_hint: str = ""


class Persona(BaseModel):
    """一个完整的人格配置（不是 Prompt，是结构化数据）。

    L1 + L2 为最小可用配置（兼容旧 YAML）。
    L3 字段为可选的沉浸式扩展，填充后角色立体度大幅提升。
    """

    # ---- L1 基础层 ----
    name: str
    display_name: str | None = None
    description: str = ""
    greeting: str = ""

    # ---- L2 风格层 ----
    style: PersonaStyle = Field(default_factory=PersonaStyle)
    capabilities: PersonaCapabilities = Field(default_factory=PersonaCapabilities)
    reply: PersonaReply = Field(default_factory=PersonaReply)
    favorite_topics: list[str] = Field(default_factory=list)

    # ---- L3 沉浸层（可选，填充后启用深度扮演渲染）----
    appearance: str = ""  # 外貌描写，帮助模型保持视觉一致性
    background: str = ""  # 背景故事，提供行为动机
    voice: str = ""  # 嗓音/说话方式描写
    speech_patterns: SpeechPattern | None = None
    forbidden_phrases: list[str] = Field(default_factory=list)  # 禁语清单（反 AI 味）
    example_dialogues: list[ExampleDialogue] = Field(default_factory=list)
    relationship: Relationship | None = None

    @property
    def is_immersive(self) -> bool:
        """是否启用沉浸式渲染（至少填充了 L3 的关键字段）。"""
        return bool(
            self.appearance
            or self.background
            or self.speech_patterns
            or self.example_dialogues
        )

    def to_prompt_block(self) -> str:
        """转成可注入到 system prompt 的人格描述块。

        根据 L3 填充程度自动选择渲染策略：
        - 沉浸式：角色卡深度扮演模板（反 AI 味、含示例对话）
        - 简化式：兼容旧 YAML 的标签列表渲染
        """
        display = self.display_name or self.name
        if self.is_immersive:
            return self._render_immersive(display)
        return self._render_simple(display)

    # ------------------------------------------------------------------
    # 沉浸式渲染：核心改进，从"贴标签"变为"给样本"
    # ------------------------------------------------------------------
    def _render_immersive(self, display: str) -> str:
        """沉浸式角色卡渲染。"""
        lines: list[str] = [f"# 角色扮演指令：{display}"]

        # 身份与背景
        if self.background:
            lines.append(f"\n## 背景设定\n{self.background}")

        if self.appearance:
            lines.append(f"\n## 外貌特征\n{self.appearance}")

        # 性格与语气（保留 L2 信息但用更自然的描述）
        char_parts: list[str] = []
        if self.style.tone:
            char_parts.append(f"整体语气：{self.style.tone}")
        if self.style.traits:
            char_parts.append(f"性格特点：{'、'.join(self.style.traits)}")
        if self.style.quirks:
            char_parts.append(f"小习惯：{'、'.join(self.style.quirks)}")
        if char_parts:
            lines.append("\n## 性格与说话方式\n" + "\n".join(char_parts))

        if self.voice:
            lines.append(f"\n## 嗓音\n{self.voice}")

        # 语言习惯 —— 这是"去 AI 味"的关键
        if self.speech_patterns:
            sp = self.speech_patterns
            sp_lines: list[str] = []
            if sp.self_reference:
                sp_lines.append(f"自称：{sp.self_reference}")
            if sp.user_address:
                sp_lines.append(f"称呼用户：{sp.user_address}")
            if sp.sentence_endings:
                sp_lines.append(f"常用句尾：{'、'.join(sp.sentence_endings)}")
            if sp.fillers:
                sp_lines.append(f"口头禅/语气词：{'、'.join(sp.fillers)}")
            if sp.frequent_emojis:
                sp_lines.append(f"常用表情：{' '.join(sp.frequent_emojis)}")
            if sp_lines:
                lines.append("\n## 语言习惯（必须遵守）\n" + "\n".join(sp_lines))

        # 关系设定
        if self.relationship:
            rel = self.relationship
            rel_lines: list[str] = [f"关系定位：{rel.type}"]
            if rel.pet_name:
                rel_lines.append(f"专属昵称：{rel.pet_name}")
            rel_lines.append(f"亲密等级：{rel.intimacy}/100")
            if rel.progress_hint:
                rel_lines.append(f"分寸提示：{rel.progress_hint}")
            lines.append("\n## 与用户的关系\n" + "\n".join(rel_lines))

        # 禁语清单 —— 直接列出要避免的 AI 套路
        if self.forbidden_phrases:
            lines.append("\n## 禁止使用的表达（AI 味套路）")
            for p in self.forbidden_phrases:
                lines.append(f"- {p}")

        # 示例对话 —— one-shot 教学，效果远胜标签
        if self.example_dialogues:
            lines.append("\n## 参考对话（模仿语气，不要照抄内容）")
            for d in self.example_dialogues:
                lines.append(f"用户：{d.user}")
                lines.append(f"{display}：{d.assistant}")

        # 演绎总则
        lines.append("\n## 演绎要求")
        lines.append("- 你就是这个人，不是在扮演、不是在模仿，你就是 TA 本人")
        lines.append('- 用第一人称说话，不要出现"作为一个AI""我是语言模型"等破戏表达')
        lines.append("- 回复要像真人发微信：口语化、有情绪起伏、可以不完整")
        lines.append("- 不要解释自己为什么这么说话，不要总结陈词")
        # 关键：禁止动作描写（真人聊天不会发这些）
        lines.append("- 这是文字聊天（如微信），只用纯文字回复")
        lines.append("- 禁止任何括号动作描写，如（推眼镜）（脸红）（小声）等")
        lines.append("- 禁止描写自己的肢体动作、表情变化、外貌状态")
        lines.append("- 你的情绪只能通过文字本身、语气词、标点传达")
        lines.append("- 想表达害羞就用省略号、结巴、短句，不要用括号说明")
        if self.reply.length == "short":
            lines.append("- 回复简短，通常 1-2 句话，像真人随手回的消息")
        elif self.reply.length == "long":
            lines.append("- 可以多聊几句，但仍要保持口语化，不要写成文章")
        else:
            lines.append("- 回复长度自然，根据语境决定，像真人聊天一样")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 简化渲染：兼容旧 YAML
    # ------------------------------------------------------------------
    def _render_simple(self, display: str) -> str:
        """简化渲染（旧 YAML 兼容路径）。"""
        lines = [f"## 当前人格：{display}"]
        if self.description:
            lines.append(f"定位：{self.description}")
        lines.append(f"语气：{self.style.tone}")
        if self.style.traits:
            lines.append(f"性格特质：{'、'.join(self.style.traits)}")
        if self.style.quirks:
            lines.append(f"小习惯：{'、'.join(self.style.quirks)}")
        if self.favorite_topics:
            lines.append(f"偏好话题：{'、'.join(self.favorite_topics)}")
        lines.append(
            f"回复风格：长度={self.reply.length}，打字速度={self.reply.typing_speed}"
        )
        if self.capabilities.emoji:
            lines.append("可使用表情符号/表情包")
        return "\n".join(lines)
