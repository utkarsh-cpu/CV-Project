"""Lazy exports for model trainers and inference pipeline."""

__all__ = ["Florence2Trainer", "SAM2Trainer", "CascadedInference"]


def __getattr__(name):
    if name == "Florence2Trainer":
        from hurricane_debris.models.florence2 import Florence2Trainer

        return Florence2Trainer
    if name == "SAM2Trainer":
        from hurricane_debris.models.sam2_trainer import SAM2Trainer

        return SAM2Trainer
    if name == "CascadedInference":
        from hurricane_debris.models.cascade import CascadedInference

        return CascadedInference
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
