from __future__ import annotations

from fastapi import FastAPI

from benchmark.agents.skills import SkillRegistry
from benchmark.core.config import settings
from benchmark.pipelines.evaluation import EvaluationPipeline
from benchmark.pipelines.generation import GenerationPipeline
from benchmark.pipelines.sample_generation import SampleGenerationPipeline
from benchmark.schemas import ErrorSample, GenerationRequest, SampleType, SkillGenerationRequest, TaskType
from benchmark.storage.db import init_db
from benchmark.storage.repository import BenchmarkRepository

app = FastAPI(title="EvoBench", version="0.3.0")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": "0.3.0",
        "llm": {
            "enabled": settings.llm_enabled,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "api_base": settings.llm_api_base,
        },
        "storage": {
            "corpus_jsonl_path": settings.corpus_jsonl_path,
            "samples_jsonl_path": settings.samples_jsonl_path,
            "items_jsonl_path": settings.items_jsonl_path,
        },
    }


@app.get("/skills")
def list_skills(task_type: TaskType | None = None) -> dict:
    registry = SkillRegistry()
    return {"skills": [definition.model_dump() for definition in registry.list_definitions(task_type)]}


@app.post("/generate")
async def generate(topic: str = "AI", sample_types: str | None = None) -> dict:
    parsed = [SampleType(sample_type.strip()) for sample_type in sample_types.split(",")] if sample_types else None
    count = await GenerationPipeline().run(topic=topic, limit=5, sample_types=parsed)
    return {"saved": count}


@app.post("/generate/batch")
async def generate_batch(request: GenerationRequest) -> dict:
    count = await GenerationPipeline().run_request(request)
    return {"saved": count, "request": request.model_dump()}


@app.post("/samples/generate")
async def generate_samples(request: SkillGenerationRequest) -> dict:
    result = await SampleGenerationPipeline().run_request(request)
    return {
        "metrics": result.metrics,
        "samples": [sample.model_dump(mode="json") for sample in result.samples],
        "rejected": [sample.model_dump(mode="json") for sample in result.rejected],
    }


@app.post("/samples/from-errors")
async def generate_from_errors(error_samples: list[ErrorSample], limit: int = 20) -> dict:
    request = SkillGenerationRequest(
        task_type=TaskType.error_to_training_set,
        skill_ids=["error_to_training_samples"],
        limit=limit,
        error_samples=error_samples,
    )
    result = await SampleGenerationPipeline().run_request(request)
    return {"metrics": result.metrics, "samples": [sample.model_dump(mode="json") for sample in result.samples]}


@app.get("/corpus")
def list_corpus(limit: int = 200, topic: str | None = None, source_type: str | None = None) -> dict:
    docs = BenchmarkRepository().list_documents(limit=limit, topic=topic, source_type=source_type)
    return {"documents": [doc.model_dump(mode="json") for doc in docs]}


@app.get("/items")
def list_items(status: str | None = None, sample_type: str | None = None) -> dict:
    items = BenchmarkRepository().list_items(status=status, limit=200)
    if sample_type:
        items = [item for item in items if item.sample_type == sample_type or item.skill_id == sample_type]
    return {"items": [item.model_dump(mode="json") for item in items]}


@app.get("/samples")
def list_samples(status: str | None = None, task_type: TaskType | None = None, skill_id: str | None = None) -> dict:
    samples = BenchmarkRepository().list_samples(status=status, task_type=task_type, skill_id=skill_id, limit=200)
    return {"samples": [sample.model_dump(mode="json") for sample in samples]}


@app.post("/evaluate/{model_id}")
async def evaluate(model_id: str) -> dict:
    return await EvaluationPipeline().evaluate_mock_model(model_id)
