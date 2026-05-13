from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich import print

from benchmark.agents.skills import SkillRegistry
from benchmark.pipelines.evaluation import EvaluationPipeline
from benchmark.pipelines.generation import GenerationPipeline
from benchmark.pipelines.sample_generation import SampleGenerationPipeline
from benchmark.postprocessors.experience_to_qa import ExperienceToQAPostprocessor
from benchmark.schemas import ErrorSample, SampleType, SkillGenerationRequest, TaskType
from benchmark.storage.db import init_db as _init_storage
from benchmark.storage.repository import BenchmarkRepository

app = typer.Typer(help="EvoBench CLI")


def _coerce_cli_value(raw: str):
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_skill_config(raw_items: list[str] | None) -> dict:
    if not raw_items:
        return {}
    parsed: dict = {}
    for item in raw_items:
        if "=" not in item:
            raise typer.BadParameter("--skill-config must use key=value form.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter("--skill-config key cannot be empty.")
        parsed[key] = _coerce_cli_value(value)
    return parsed


def _paper_template_path() -> Path:
    return Path(__file__).resolve().parent / "agents" / "skills" / "templates" / "paper_corpus_template.jsonl"


@app.command("init-db")
@app.command("init-storage")
def init_storage() -> None:
    _init_storage()
    print("[green]JSONL storage initialized.[/green]")


@app.command("list-skills")
def list_skills(task_type: str | None = None) -> None:
    parsed = TaskType(task_type) if task_type else None
    for skill in SkillRegistry().list_definitions(parsed):
        print(f"[cyan]{skill.skill_id}[/cyan] | {skill.task_type.value} | {skill.description}")


@app.command("generate-samples")
def generate_samples(
    topic: str | None = typer.Option(None, help="Topic filter for retriever or corpus documents."),
    corpus_jsonl: Path | None = typer.Option(None, help="Read source documents from a corpus.jsonl file."),
    samples_jsonl: Path | None = typer.Option(None, help="Write generated UnifiedSample records to this JSONL path."),
    task_type: str | None = typer.Option(None, help="document_to_x/document_to_xy/document_to_xty/error_to_training_set"),
    skill_ids: str = typer.Option("doc_to_qa", help="Comma-separated skill ids."),
    skill_config: list[str] | None = typer.Option(
        None,
        help="Repeatable skill config in key=value form, e.g. --skill-config pairs_per_doc=3.",
    ),
    limit: int = typer.Option(10, min=1),
    domain: str = "technology",
    language: str = "zh-CN",
) -> None:
    request = SkillGenerationRequest(
        topic=topic,
        corpus_jsonl_path=str(corpus_jsonl) if corpus_jsonl else None,
        samples_jsonl_path=str(samples_jsonl) if samples_jsonl else None,
        task_type=TaskType(task_type) if task_type else None,
        skill_ids=[skill_id.strip() for skill_id in skill_ids.split(",") if skill_id.strip()],
        skill_config=_parse_skill_config(skill_config),
        limit=limit,
        domain=domain,
        language=language,
    )
    result = asyncio.run(SampleGenerationPipeline().run_request(request))
    print(
        f"[green]Generated {result.metrics['generated']} samples; "
        f"verified {result.metrics['verified']}; rejected {result.metrics['rejected']}.[/green]"
    )
    for sample in result.samples[:5]:
        print(f"- {sample.sample_id} | {sample.task_type.value} | {sample.skill_id} | x={str(sample.x)[:80]}")


@app.command("generate-from-errors")
def generate_from_errors(
    errors_json: Path = typer.Option(..., help="JSON file containing a list of ErrorSample records."),
    limit: int = typer.Option(20, min=1),
) -> None:
    raw = json.loads(errors_json.read_text(encoding="utf-8"))
    errors = [ErrorSample.model_validate(item) for item in raw]
    request = SkillGenerationRequest(
        task_type=TaskType.error_to_training_set,
        skill_ids=["error_to_training_samples"],
        limit=limit,
        error_samples=errors,
    )
    result = asyncio.run(SampleGenerationPipeline().run_request(request))
    print(f"[green]Generated {result.metrics['verified']} training samples from {len(errors)} errors.[/green]")


@app.command("list-samples")
def list_samples(
    status: str | None = None,
    task_type: str | None = None,
    skill_id: str | None = None,
    limit: int = 20,
) -> None:
    parsed = TaskType(task_type) if task_type else None
    samples = BenchmarkRepository().list_samples(status=status, task_type=parsed, skill_id=skill_id, limit=limit)
    for sample in samples:
        print(f"[cyan]{sample.sample_id}[/cyan] | {sample.status} | {sample.task_type.value} | {sample.skill_id} | x={str(sample.x)[:80]}")


@app.command("list-corpus")
def list_corpus(limit: int = 20, topic: str | None = None, source_type: str | None = None) -> None:
    docs = BenchmarkRepository().list_documents(limit=limit, topic=topic, source_type=source_type)
    for doc in docs:
        print(f"[cyan]{doc.source_id}[/cyan] | {doc.source_type} | {doc.title[:80]}")


@app.command("init-paper-corpus-template")
def init_paper_corpus_template(
    output: Path = typer.Option(Path("data/paper_corpus_template.jsonl"), help="Destination for the paper corpus template."),
    overwrite: bool = typer.Option(False, help="Overwrite the destination if it already exists."),
) -> None:
    template_path = _paper_template_path()
    if output.exists() and not overwrite:
        raise typer.BadParameter(f"{output} already exists. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[green]Wrote paper corpus template to {output}[/green]")


@app.command("postprocess-experience-to-qa")
def postprocess_experience_to_qa(
    source_skill_id: str = typer.Option("paper_to_experience", help="Source skill id to transform."),
    status: str | None = typer.Option("verified", help="Only transform samples with this status. Use empty value to include all."),
    samples_jsonl: Path | None = typer.Option(None, help="Read and write UnifiedSample records from this JSONL path."),
    limit: int = typer.Option(100, min=1),
) -> None:
    repo = BenchmarkRepository(samples_path=samples_jsonl) if samples_jsonl else BenchmarkRepository()
    source_samples = repo.list_samples(status=status or None, skill_id=source_skill_id, limit=limit)
    qa_samples = ExperienceToQAPostprocessor().transform(source_samples, limit=limit, source_skill_id=source_skill_id)
    for sample in qa_samples:
        repo.upsert_sample(sample)
    print(f"[green]Generated {len(qa_samples)} experience_to_qa samples from {len(source_samples)} source samples.[/green]")


@app.command("ingest-sample")
def ingest_sample(
    topic: str = "AI",
    limit: int = 5,
    sample_types: str = typer.Option(
        "fact_verification,source_attribution,evidence_selection,temporal_awareness,abstention,data_value_judgement,conflict_resolution,causal_attribution",
        help="Comma-separated sample types.",
    ),
) -> None:
    parsed = [SampleType(sample_type.strip()) for sample_type in sample_types.split(",") if sample_type.strip()]
    count = asyncio.run(GenerationPipeline().run(topic=topic, limit=limit, sample_types=parsed))
    print(f"[green]Generated and saved {count} verified benchmark items.[/green]")


@app.command("run-eval")
def run_eval(model_id: str = "demo_model") -> None:
    result = asyncio.run(EvaluationPipeline().evaluate_mock_model(model_id))
    print(result)


if __name__ == "__main__":
    app()
