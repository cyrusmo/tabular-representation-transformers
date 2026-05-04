from __future__ import annotations

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # pragma: no cover
    PyTorchModelHubMixin = object  # type: ignore[misc,assignment]


class TabularStateHubMixin(PyTorchModelHubMixin):
    """Optional mixin placeholder for future direct Hub integration."""
