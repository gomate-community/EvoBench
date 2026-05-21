"""doc_to_qa 的 LLM 增强版 Skill（与 doc_to_qa 兄弟级，符合 skill 子目录约定）。

设计：
- main 上 ``DocumentToQASkill`` 是纯模板实现，本子类只在父类基础上**额外**注入 LLM 分支；
- LLM 不可用时直接 ``super().generate(...)`` 走父类模板兜底，行为与 main 一致；
- bootstrap 通过 ``SkillRegistry._factories["doc_to_qa"] = DocumentToQALLMSkill`` 把默认工厂替换为本类，
  从而在不改 main 的前提下让默认行为等价于 wsy_skill_dev 上的 LLM 模式。
"""

from __future__ import annotations

from benchmark.agents.skills.doc_to_qa_llm.skill import DocumentToQALLMSkill

__all__ = ["DocumentToQALLMSkill"]
