---
name: Proposal-Codebase Alignment Check
description: "Review cv_project.pdf against the current codebase and recommend concrete code and test changes to align implementation with the proposal. Use when checking scope drift, missing features, or evaluation gaps."
argument-hint: "Optional focus: specific module, claim, or section to prioritize"
agent: agent
---
Task: verify whether the codebase follows the project proposal and recommend specific improvements.

Primary spec document:
- [Proposal PDF](../../cv_project.pdf)

Codebase scope:
- [hurricane_debris package](../../hurricane_debris/)
- [README](../../README.md)
- [requirements](../../requirements.txt)

How to work:
1. Read the proposal PDF first and extract the key implementation commitments:
- datasets and preprocessing pipeline
- model architecture and training strategy
- evaluation metrics and baselines
- experiments, ablations, and reporting claims
- non-functional constraints (runtime, reproducibility, robustness)

2. Inspect the repository and map each commitment to evidence in code:
- mark as `Implemented`, `Partially Implemented`, `Missing`, or `Unclear`
- cite concrete file references for each decision

3. Produce a gap analysis that is actionable for engineering.

Output format:
1. `Alignment Summary` (3-6 bullets)
2. `Findings by Severity`
- `Critical`: likely to invalidate proposal claims
- `Major`: important divergence from proposal
- `Minor`: quality or completeness issues
For each finding include:
- proposal claim (short quote/paraphrase)
- current code evidence (file refs)
- risk if unchanged
- recommended change
3. `Prioritized Change Plan`
- top 5 code changes with expected impact and effort (`S`, `M`, `L`)
4. `Test and Validation Additions`
- specific unit/integration/evaluation tests to add
5. `Open Questions`
- assumptions that require user confirmation

Quality bar:
- Do not make claims without file evidence.
- Prefer concrete edits over general advice.
- If something cannot be verified from available files, label it `Unclear`.
- Use the user argument (if provided) to prioritize a section/module, while still reporting critical mismatches globally.
