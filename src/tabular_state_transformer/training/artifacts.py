from __future__ import annotations

import pickle
import json
from pathlib import Path

from tabular_state_transformer.training.trainer import TrainingResult


def save_training_artifacts(result: TrainingResult, save_directory: str | Path) -> Path:
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    result.model.save_pretrained(save_path)
    with (save_path / "preprocessor.pkl").open("wb") as fh:
        pickle.dump(result.preprocessor, fh)
    metadata = {
        "processed_feature_names": result.processed_feature_names,
        "best_epoch": result.best_epoch,
        "best_val_metric": result.best_val_metric,
        "final_val_metric": result.final_val_metric,
        "effective_training_status": result.effective_training_status,
        "class_labels": result.class_labels.tolist() if result.class_labels is not None else None,
    }
    (save_path / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    return save_path
