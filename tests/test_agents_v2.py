import asyncio

from benchmark.adapters.retriever import MockRetriever
from benchmark.agents.claim_agent import ClaimAgent
from benchmark.agents.question_agent import QuestionAgent
from benchmark.agents.source_agent import SourceAgent
from benchmark.agents.verifier_agent import VerifierAgent
from benchmark.schemas import SampleType


def test_agents_generate_multiple_sample_types():
    async def run():
        docs = await SourceAgent(MockRetriever()).collect("AI", limit=3)
        claims = await ClaimAgent().extract_claims(docs)
        items = await QuestionAgent().generate(
            claims[:1],
            docs=docs,
            sample_types=[
                SampleType.fact_verification,
                SampleType.source_attribution,
                SampleType.evidence_selection,
                SampleType.abstention,
                SampleType.data_value_judgement,
            ],
        )
        assert len(items) >= 5
        assert {item.sample_type for item in items} >= {
            SampleType.fact_verification,
            SampleType.source_attribution,
            SampleType.evidence_selection,
            SampleType.abstention,
            SampleType.data_value_judgement,
        }
        verified = VerifierAgent().verify_batch(items)
        assert any(item.status == "verified" for item in verified)
        assert all(item.annotation_guideline.label_schema for item in verified)

    asyncio.run(run())
