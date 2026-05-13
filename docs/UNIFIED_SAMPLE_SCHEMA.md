# Unified Sample Schema

## Goal

The project uses one shared `UnifiedSample` format to support:

1. `d -> x`: document to a single target such as a question or answer
2. `d -> (x, y)`: document to paired outputs such as question-answer or future-problem plus experience-card
3. `d -> (x, T, y)`: document to question, explicit solution steps, and answer
4. `error -> training samples`: error-driven augmentation that yields corrected, contrastive, or boundary variants

## Core Idea

- A sample is represented as `input -> output artifacts`, not as a fixed QA-only object.
- `x`, `y`, and `T` are artifact keys whose meanings depend on the generating skill.
- Provenance such as documents, evidence, claims, and error samples stays attached to the same record.
- Each skill now exposes its own output schema so downstream code can validate the exact sample shape it emits.

## Main Objects

### `UnifiedSample`

- `sample_id`: unique sample id
- `task_type`: task family such as `document_to_xy`
- `skill_id`: generating skill
- `input`: documents, artifacts, claims, or raw payload
- `output`: generated artifacts plus the skill-specific `target_schema`
- `source_refs`: source-document references
- `evidence`: evidence spans supporting the output
- `annotation_guideline`: review and labeling guidance
- `quality_signals`: automatic quality indicators
- `parent_sample_ids`: lineage for transformed or augmented samples

### `SampleArtifact`

- `role`: semantic role such as `question`, `answer`, `reasoning_trace`, or `label`
- `key`: business key such as `x`, `y`, `T`, or `experience_type`
- `value`: JSON-serializable payload
- `evidence_ids`: ids of supporting evidence spans
- `metadata`: extensible per-artifact metadata

### `SkillDefinition`

- `skill_id`: skill id
- `task_type`: top-level task family
- `input_requirements`: expected inputs
- `output_schema`: skill-level output contract
- `quality_rules`: quality gates
- `config`: skill-specific configuration

## Skill-Specific Output Schemas

Built-in skills expose explicit output schemas:

- `doc_to_question`: `DocumentQuestionSampleSchema`
- `doc_to_answer`: `DocumentAnswerSampleSchema`
- `doc_to_qa`: `DocumentQASampleSchema`
- `doc_to_qa_steps`: `DocumentQAStepsSampleSchema`
- `paper_to_experience`: `PaperExperienceSampleSchema`, whose `y` is an `ExperienceCard`
- `error_to_training_samples`: variant-specific schemas for corrected, contrastive, and boundary outputs

At runtime, these appear in `sample.output.target_schema` as a structured schema bundle with:

- `schema_name`
- `artifacts`
- `description`
- `json_schema` for direct validation

## Adding a New Skill

1. Define or reuse a `TaskType`.
2. Add a Pydantic sample schema for that skill's output.
3. Register the skill with its `output_schema`.
4. Implement `SkillBase.generate()` to emit `UnifiedSample`.
5. Attach evidence and annotation guidance.
6. Verify and store the resulting samples.
