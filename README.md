# CV-Project

Hurricane debris detection project with training, evaluation, and inference pipelines.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Download Datasets

Before training, download the required datasets.  A built-in pipeline handles
download, extraction, and directory-layout validation:

```bash
# Download RescueNet (primary training/evaluation dataset)
python main.py --download --dataset rescuenet --dataset-dir ./datasets

# Download MSNet (cross-dataset evaluation)
python main.py --download --dataset msnet --dataset-dir ./datasets

# Download DesignSafe-CI (cross-dataset evaluation — requires free account)
python main.py --download --dataset designsafe --dataset-dir ./datasets

# Download all datasets in one go
python main.py --download --dataset all --dataset-dir ./datasets

# Force re-download even if the data already exists
python main.py --download --dataset rescuenet --dataset-dir ./datasets --force-download

# Keep the downloaded archive after extraction
python main.py --download --dataset rescuenet --dataset-dir ./datasets --keep-archive
```

> **Note:** Some datasets (MSNet, DesignSafe-CI) require registration or manual
> access-request steps.  When automatic download is unavailable the pipeline
> prints step-by-step instructions for obtaining the data manually.

## Run

Examples:

```bash
python main.py --full-pipeline
python main.py --train-florence --dataset rescuenet
python main.py --train-sam2 --dataset rescuenet
python main.py --evaluate --dataset msnet
python main.py --evaluate --dataset rescuenet --cross-dataset --metrics-dir ./outputs/metrics
python main.py --infer --image path/to/image.jpg
python main.py --infer --image path/to/image.jpg --output-json out.json --output-geojson out.geojson
```

## Baseline and Ablation Matrix

Run the experiment matrix and aggregate metrics:

```bash
python scripts/run_experiments.py --matrix scripts/experiment_matrix.json --output-dir ./outputs/experiments
```

This generates:

- `outputs/experiments/experiment_summary.json`
- `outputs/experiments/experiment_summary.csv`

## Gradio Deployment Demo

Launch the local web app:

```bash
python app.py --florence-dir ./models/florence2_debris --sam2-checkpoint ./models/sam2_debris/best_model.pth
```

Then open `http://127.0.0.1:7860` in your browser.
