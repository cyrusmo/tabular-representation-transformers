from __future__ import annotations

def load_openml_dataset(dataset_id: int):
    try:
        import openml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install openml to use OpenML benchmark integration") from exc
    dataset = openml.datasets.get_dataset(dataset_id)
    return dataset.get_data(target=dataset.default_target_attribute)

CURATED_OPENML_IDS = [31, 37, 44, 1464, 1480]
TABPFN_SMALL_DATA_IDS = [31, 37, 44]
