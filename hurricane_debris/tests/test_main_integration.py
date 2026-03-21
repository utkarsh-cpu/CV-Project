"""Integration-style tests for evaluation wiring in main.py."""

import argparse
import json
from pathlib import Path

import numpy as np

import main
from hurricane_debris.config import ExperimentConfig


class DummyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "image_path": f"/tmp/fake_{idx}.png",
            "pixel_values": np.zeros((3, 64, 64), dtype=float),
            "target": {
                "bboxes": np.array([[10.0, 10.0, 20.0, 20.0]], dtype=float),
                "category_ids": np.array([3], dtype=int),
                "labels": ["damaged or collapsed building with debris"],
                "masks": np.zeros((0, 64, 64), dtype=float),
                "semantic_mask": np.zeros((64, 64), dtype=np.int64),
            },
            "image_id": idx,
        }


class DummyPredictor:
    def predict(self, sample):
        # Perfect predictions in xyxy format.
        return {
            "bboxes": np.array([[10.0, 10.0, 30.0, 30.0]], dtype=float),
            "scores": np.array([0.99], dtype=float),
            "labels": np.array([3], dtype=int),
            "semantic_mask": np.zeros((64, 64), dtype=np.int64),
        }


def test_evaluate_writes_metrics_files(monkeypatch, tmp_path):
    args = argparse.Namespace(
        dataset_dir=str(tmp_path),
        dataset="rescuenet",
        florence_dir="./models/florence2_debris",
        sam2_dir="./models/sam2_debris",
        sam2_checkpoint="./checkpoints/sam2_hiera_large.pt",
        image=None,
        output_json=None,
        output_geojson=None,
        metrics_dir=str(tmp_path / "metrics"),
        cross_dataset=True,
        strict_eval_model=False,
        epochs_florence=1,
        epochs_sam2=1,
        lr_florence=1e-5,
        lr_sam2=1e-5,
        batch_size=1,
        image_size=64,
        device="cpu",
        log_file=str(tmp_path / "test.log"),
        seed=123,
        full_pipeline=False,
        train_florence=False,
        train_sam2=False,
        evaluate=True,
        infer=False,
    )

    config = ExperimentConfig()

    monkeypatch.setattr(main, "load_dataset", lambda _a, _c, _s: DummyDataset())
    monkeypatch.setattr(main, "_build_predictor", lambda _a, _c: DummyPredictor())

    main.evaluate(args, config)

    metrics_dir = Path(args.metrics_dir)
    assert (metrics_dir / "metrics_rescuenet.json").exists()
    assert (metrics_dir / "metrics_designsafe.json").exists()
    assert (metrics_dir / "metrics_msnet.json").exists()
    assert (metrics_dir / "cross_dataset_summary.json").exists()

    payload = json.loads((metrics_dir / "metrics_rescuenet.json").read_text())
    assert "f1" in payload
    assert "ap50" in payload


def test_resolve_dataset_dir_uses_sibling_dataset_folder(tmp_path):
    datasets_root = tmp_path / "datasets"
    rescuenet_root = datasets_root / "rescuenet"
    msnet_root = datasets_root / "msnet"
    rescuenet_root.mkdir(parents=True)
    msnet_root.mkdir(parents=True)

    resolved = main._resolve_dataset_dir(str(rescuenet_root), "msnet")

    assert resolved == str(msnet_root)
