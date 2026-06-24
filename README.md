# Stellar Classification: Ensembled Pipeline

A multi-class classification pipeline (Galaxies, Stars, Quasars) built to handle specific hardware bottlenecks and dependency failures in tabular data processing. It utilizes a Level-1 blend of a Gradient Boosting framework (XGBoost) and a Multi-Layer Perceptron (PyTorch), optimized via SciPy Nelder-Mead.

## Engineering Constraints & Solutions

* **Vectorized Feature Synthesis (Dependency Bypass):** Standard automated feature engineering tools (Featuretools/Woodwork) failed due to weak-reference memory leaks during EntitySet creation. Replaced with a custom, zero-dependency Python script using vectorized NumPy combinations to generate 200+ non-linear features natively.
* **VRAM Optimization (DataLoader Overhead):** PyTorch's standard `DataLoader` iteration speed starved the GPU during tabular training due to CPython interpreter lag. Resolved by engineering a direct host-to-device memory transfer protocol—pushing the entire fold directly into GPU VRAM and using native C++ tensor slicing for batch generation. 
* **State Isolation:** Orchestration script executes training in isolated subprocesses to guarantee system RAM and CUDA contexts are cleanly flushed between model folds.
* **Orthogonal Blending:** Used SciPy's Nelder-Mead optimization on Out-Of-Fold (OOF) probabilities to calculate the precise fractional weights needed to blend the PyTorch MLP's non-linear boundaries with XGBoost's rigid tree structures.

## Architecture

* `train_xgb.py`: 5-Fold Stratified CV on engineered features. Serializes model weights and extracts OOF probabilities.
* `train_mlp.py`: Baseline training on raw features with strict within-fold standard scaling to prevent data leakage. Executes VRAM-optimized batching.
* `optimize_blend.py`: Ingests OOF probability arrays and optimizes voting weights against the Balanced Accuracy Score (BAS).
* `run_pipeline.py`: Master orchestrator managing subprocess execution and memory state.

## Metrics & Validation

* **Validation Strategy:** 5-Fold Stratified Cross-Validation
* **Evaluation Metric:** Balanced Accuracy Score (BAS)
* **XGBoost Standalone OOF:** 0.9649
* **PyTorch Standalone OOF:** 0.9532
* **Optimized Ensemble OOF:** 0.9650
* **Test Set / Public Leaderboard:** 0.9657

## Setup & Execution

**1. Environment Setup**
```bash
git clone https://github.com/danielweser/predicting-stellar-class.git
cd predicting-stellar-class
pip install -r requirements.txt
```

**2. Download and Generate the Dataset**

Downloads the competition data, re-maps string columns, engineers new features for XGBoost.

Note: downloads dataset from Kagglehub, which requires API token.

```bash
python src/generate_dataset.py
```

**3. Run Training**
```bash
python src/run_pipeline.py
```

## Dataset
The dataset is taken from the Kaggle competition [Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6/overview).
