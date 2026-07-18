from pathlib import Path

from setuptools import setup, find_packages


def _data_tree(root: str):
    """Preserve a tracked evidence tree under the wheel installation prefix."""
    base = Path(root)
    if not base.is_dir():
        return []
    grouped = {}
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        grouped.setdefault(str(path.parent), []).append(str(path))
    return sorted(grouped.items())

setup(
    name="futures-foundation-model",
    version="2.0.0",
    description="Futures-market foundation layer on pretrained Chronos-Bolt: frozen embeddings + strategy-pluggable training/eval pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/johnamcruz/Futures-Foundation-Model",
    packages=find_packages(),
    data_files=[
        ("config/foundation_models", [
            "config/foundation_models/native_contracts.json",
            "config/foundation_models/native_contract_evidence.json",
            "config/foundation_models/trusted_approvers.json",
            "config/foundation_models/historical_native_contract_snapshot.json",
        ]),
        *_data_tree("output/native_parity_evidence_v2_canonical"),
    ],
    python_requires=">=3.9",
    # Core install is torch-free (the parent process must never load torch —
    # see futures_foundation/foundation.py). Torch/Chronos run only inside
    # the embed subprocess; install them via the [foundation] extra.
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "cryptography>=42",
    ],
    extras_require={
        "foundation": ["torch>=2.0", "chronos-forecasting>=2.3.1,<3"],
        # External zero-shot benchmark only. The official Kronos Git checkout is deliberately
        # supplied separately and commit-pinned by scripts/benchmark_kronos.py.
        "kronos": ["torch>=2.0", "huggingface_hub>=0.33", "einops>=0.8",
                    "safetensors>=0.4"],
        # Frozen external benchmark. The MOMENT checkout and model revision are supplied and
        # commit-pinned by scripts/benchmark_moment.py rather than vendored into this project.
        "moment": ["torch>=2.0", "transformers>=4.54.1", "huggingface_hub>=0.33",
                   "safetensors>=0.4"],
        "heads": ["xgboost>=2.0", "joblib>=1.3"],
        "regime": ["hmmlearn>=0.3"],   # futures_foundation.regime market-state HMM
        "data": ["pyarrow>=16"],       # sealed parquet corpus preparation
        "onnx": ["onnxmltools", "skl2onnx"],
        "dev": ["pytest>=7.0", "black", "ruff", "hmmlearn>=0.3"],
    },
    entry_points={
        "console_scripts": [
            "ffm-native-parity-evidence=futures_foundation.finetune.native_evidence_cli:main",
            "ffm-native-parity-matrix=futures_foundation.finetune.native_parity_matrix_cli:main",
        ],
    },
)
