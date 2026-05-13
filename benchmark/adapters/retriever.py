from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from benchmark.schemas import SourceDocument


class RetrieverAdapter(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[SourceDocument]: ...


class MockRetriever(RetrieverAdapter):
    async def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        now = datetime.utcnow()
        topic = query or "AI"
        docs = [
            SourceDocument(
                source_id="src_demo_001",
                title=f"{topic} 资讯：某公司发布新一代 AI 芯片",
                url="https://example.com/demo-ai-chip",
                source_type="news",
                publisher="Example News",
                published_at=now,
                content=(
                    "某公司于今日发布新一代 AI 芯片，宣称推理性能提升 30%。"
                    "该公司表示，新芯片面向数据中心推理场景，并计划在下季度开始向部分客户供货。"
                    "报道未披露独立第三方基准测试结果。"
                ),
                trust_level=3,
            ),
            SourceDocument(
                source_id="src_demo_002",
                title=f"{topic} 公司技术博客：新芯片架构说明",
                url="https://example.com/blog-ai-chip",
                source_type="company_blog",
                publisher="Example Corp",
                published_at=now - timedelta(hours=2),
                content=(
                    "Example Corp 在技术博客中称，新一代 AI 芯片采用改进的内存带宽设计。"
                    "公司给出的内部测试显示，部分推理任务性能提升约 30%。"
                    "博客强调，实际性能会随模型结构和部署环境变化。"
                ),
                trust_level=4,
            ),
            SourceDocument(
                source_id="src_demo_003",
                title=f"分析：{topic} 芯片性能声明仍需第三方验证",
                url="https://example.com/analysis-ai-chip",
                source_type="analysis",
                publisher="Example Research",
                published_at=now - timedelta(days=1),
                content=(
                    "行业分析师认为，厂商关于 AI 芯片性能提升 30% 的说法需要第三方测试验证。"
                    "目前公开材料主要来自公司新闻稿和技术博客，尚未看到独立实验室报告。"
                    "若性能声明成立，该产品可能影响云端推理成本结构。"
                ),
                trust_level=4,
            ),
        ]
        return docs[:limit]
