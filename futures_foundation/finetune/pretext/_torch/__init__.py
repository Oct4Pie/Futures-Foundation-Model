"""Torch layer for the SSL pretext trainers — window helpers, frozen-embedding/ONNX primitives,
the shared BaseTrainer, and one module per pretext (mask / forecast / contrastive). Imported
lazily (never by the torch-free orchestrator or the task registry)."""
