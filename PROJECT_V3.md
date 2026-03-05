# Project V3: Proposal-Codebase Alignment and Revised Scope

Date: March 5, 2026
Project: Hurricane Debris Detection (Florence-2 + SAM2)

## Alignment Table

| Proposal Commitment | Status | Evidence in Codebase | Gap / Notes | Proposed Change |
|---|---|---|---|---|
| RescueNet, DesignSafe-CI, MSNet dataset support | Implemented | `hurricane_debris/data/rescuenet.py`, `hurricane_debris/data/designsafe.py`, `hurricane_debris/data/msnet.py`, dataset switch in `main.py`, smoke tests in `hurricane_debris/tests/test_datasets.py` | Dataset loader smoke tests added for all three loaders | Keep as-is |
| Preprocessing + augmentation pipeline | Implemented | `hurricane_debris/data/transforms.py`, dataset usage, global seeding in `main.py` | Global deterministic setup added; augmentation internals still rely on library RNG behavior | Keep as-is; consider explicit transform-level seed hooks if strict reproducibility is required |
| Florence-2 open-vocabulary detection fine-tuning | Partially Implemented | `hurricane_debris/models/florence2.py`, LoRA config in `hurricane_debris/config.py`, collate integration test in `hurricane_debris/tests/test_florence2.py` | Core training path exists; no full tiny end-to-end train-step integration test yet | Add one minimal train-step smoke test with mocked processor/model |
| SAM2 partial fine-tuning (freeze image encoder, train prompt+mask decoders) | Implemented | `hurricane_debris/models/sam2_trainer.py`, config flags, module-trainability assertion test in `hurricane_debris/tests/test_sam2.py` | Proposal strategy now verified by test | Keep as-is |
| Cascaded inference Florence-2 -> SAM2 | Implemented | `hurricane_debris/models/cascade.py`, CLI path in `main.py --infer`, output tests in `hurricane_debris/tests/test_cascade.py` | Pipeline and output formats are in place | Add optional mocked full pipeline integration test later for stronger E2E confidence |
| Priority-based output/taxonomy filtering | Implemented | Priority map and deterministic sorting in `hurricane_debris/models/cascade.py`, test in `hurricane_debris/tests/test_cascade.py` | Deterministic ordering is now validated | Keep as-is |
| Structured JSON reporting output | Implemented | `InferenceResult.to_json()` in `hurricane_debris/models/cascade.py`, CLI output in `main.py`, schema-like test in `hurricane_debris/tests/test_cascade.py` | Structured payload shape validated by test | Keep as-is |
| GIS-compatible GeoJSON output | Partially Implemented | `InferenceResult.to_geojson()` in `hurricane_debris/models/cascade.py`, CLI `--output-geojson` in `main.py`, Gradio output in `app.py` | Image-pixel GeoJSON implemented; CRS/georeferenced reprojection still unavailable | Keep partial scope; maintain final-paper clarification |
| mIoU, F1, AP evaluation metrics implementation | Implemented | Evaluator in `hurricane_debris/evaluation/metrics.py`, evaluate loop wired in `main.py` with per-dataset metric persistence | Execution path now wired to inference loop | Keep as-is |
| Cross-dataset validation (train on RescueNet, test on DesignSafe/MSNet) | Implemented | `--cross-dataset` mode in `main.py`, summary output `cross_dataset_summary.json`, integration test in `hurricane_debris/tests/test_main_integration.py` | Automated CLI benchmark path added | Keep as-is |
| AP@[0.5:0.95] correctness | Implemented | Threshold-specific rematching in `hurricane_debris/evaluation/metrics.py`, regression test in `hurricane_debris/tests/test_evaluator.py` | AP now recomputed per IoU threshold from raw detection samples | Keep as-is |
| Baselines / ablation experiments and reporting | Partially Implemented | Experiment matrix `scripts/experiment_matrix.json`, runner and aggregator `scripts/run_experiments.py` | Baseline matrix and summary generation exist; no paper-style curated baseline table yet | Add one curated report template/table for manuscript-ready presentation |
| Reproducibility controls (global seed, deterministic setup) | Implemented | `set_seed()` in `main.py`, `--seed` CLI arg, run artifact logging via `run_config.json` | Runtime seed + config artifact persistence implemented | Keep as-is |
| Deployment framework: Gradio web interface | Implemented | `app.py`, dependency in `requirements.txt`, usage docs in `README.md` | Local web deployment path now available | Keep as-is |
| Test coverage for end-to-end claims | Partially Implemented | Added tests in `hurricane_debris/tests/test_main_integration.py`, `hurricane_debris/tests/test_cascade.py`, `hurricane_debris/tests/test_evaluator.py`, `hurricane_debris/tests/test_datasets.py`, `hurricane_debris/tests/test_sam2.py`, `hurricane_debris/tests/test_florence2.py` | Coverage improved significantly; full heavy-model E2E runs are still not in CI-friendly tests | Add optional nightly/integration suite with real checkpoints and sample assets |

## GeoJSON Scope Clarification for Final Paper

Use this wording in your final report:

"GeoJSON export is partially implemented at image-coordinate level. Full GIS-georeferenced GeoJSON was not completed because the current datasets do not provide the geospatial metadata required for pixel-to-CRS reprojection (for example camera pose/orthorectification/GCP transforms)."

## Project V3 Priorities (Execution Order)

1. Implement end-to-end evaluation loop in `main.py` and persist per-dataset metrics.
2. Fix AP threshold logic in `hurricane_debris/evaluation/metrics.py`.
3. Add cross-dataset benchmark mode and output summary table.
4. Add pixel-coordinate GeoJSON export path from cascade output.
5. Add reproducibility setup (`set_seed`) and integration tests.
