# paper_to_experience

Extract transferable research experience from an academic paper so the result can support future related problems.

## Input

- `documents`

## Output

- `x`: a future problem scenario
- `y`: a structured `ExperienceCard`

## Experience Types

- `fact`: a fact supported by experiments, data, observations, or theorem-like evidence
- `strategy`: an effective method, workflow, procedure, or heuristic
- `mechanism`: a causal, explanatory, or theory-backed relationship
- `boundary`: an applicability limit or scope condition
- `failure`: a failed attempt, negative result, non-significant result, or harmful side effect

## Notes

- Every experience card should be grounded in paper evidence.
- `y` contains `experience_type`, `experience_title`, `statement_nature`, `experience_statement`, `applicability`, `supporting_evidence`, `paper_location`, `is_verifiable`, `verification_method`, `possible_counterexample`, `confidence`, `benchmark_transformable`, `actionable_advice`, and `caveats`.
- This skill is designed for "distill experience from papers, then transfer it to future tasks" style sample construction.
