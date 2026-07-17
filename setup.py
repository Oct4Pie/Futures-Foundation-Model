from setuptools import setup, find_packages

setup(
    name="futures-foundation-model",
    version="2.0.0",
    description="Futures-market foundation layer on pretrained Chronos-Bolt: frozen embeddings + strategy-pluggable training/eval pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/johnamcruz/Futures-Foundation-Model",
    packages=find_packages(),
    python_requires=">=3.9",
    # Core install is torch-free (the parent process must never load torch —
    # see futures_foundation/foundation.py). Torch/Chronos run only inside
    # the embed subprocess; install them via the [foundation] extra.
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
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
)
