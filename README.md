# Crypto Anomaly Forecasting

This project studies next-hour price anomaly classification across 17 cryptocurrencies. It compares models that treat each cryptocurrency independently with graph neural networks that pass information between assets.

## Task and models

At each hour, a model uses information available up to that time to classify the return from the current close to the next hourly close:

- `up`: at least +1%;
- `down`: at most -1%;
- `normal`: otherwise.

The independent models are a majority baseline, a temporal MLP, a temporal GAT, and a GAT with Natural Visibility Graph edges. The Panel GAT processes the 17 assets in one graph, using either temporal edges, DTW-clustered cross-asset edges, or all-to-all cross-asset edges.

The independent models compare two input sets. **OHLC** contains the hourly Open, High, Low, and Close price features. **OHLCV** adds Volume. This comparison tests whether volume contributes useful information beyond price movements. The Panel GAT experiments use OHLC.

## Evaluation and results

The 2021 development period uses nested walk-forward evaluation: each month is an outer test period, and inner folds select model configurations. The selected configurations are frozen for the confirmatory evaluation on 2022. Macro-F1 is the primary metric. Statistical comparisons use paired monthly results, a circular block bootstrap, and Holm correction for multiple comparisons.

The table reports mean Macro-F1 by model and input set. These are descriptive averages; statistical significance is assessed separately.

| Model | Input | 2021 | 2022 |
| --- | --- | ---: | ---: |
| Majority baseline | — | 0.258 | 0.298 |
| MLP temporal | OHLC | **0.379** | **0.383** |
| MLP temporal | OHLCV | 0.376 | 0.378 |
| GAT temporal | OHLC | 0.368 | 0.373 |
| GAT temporal | OHLCV | 0.367 | 0.372 |
| GAT with NVG | OHLC | 0.353 | 0.363 |
| GAT with NVG | OHLCV | 0.349 | 0.359 |
| Panel GAT temporal | OHLC | 0.362 | 0.368 |
| Panel GAT clustered | OHLC | 0.338 | 0.364 |
| Panel GAT all-to-all | OHLC | 0.317 | 0.348 |

The MLP with OHLC has the highest mean Macro-F1 in both periods. The visibility edges do not improve the temporal GAT in these experiments. The clustered Panel GAT scores higher than the all-to-all version, but its advantage over the temporal Panel GAT is not statistically significant in 2022.

## Project layout

```text
crypto_gnn/
├── data/          data description and fold-level preparation
├── graph/         temporal, visibility, and panel graph construction
├── clustering/    DTW distances and k-medoids partitions
├── training/      model training and experiment protocols
├── evaluation/    summaries and statistical comparisons
└── models.py      MLP, GAT, and Panel GAT architectures
```

## Environment

The completed reference runs used Python 3.12.14. Install the pinned dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Workflow

Data preparation is part of the training workflow: targets and causal features are prepared when the data are loaded, while standardization is fitted separately within each training fold. There is no separate command that creates one globally standardized dataset.

Run the commands below from the project root and in order. The data report and graph visualizations are optional diagnostics. Independent models are trained and evaluated before the DTW clustering and relational experiments.

```bash
# Optional data and graph diagnostics
python -m crypto_gnn.data.understanding
python -m crypto_gnn.graph.temporal
python -m crypto_gnn.graph.plotting

# Train and evaluate the independent models
python -m crypto_gnn.training.independent
python -m crypto_gnn.evaluation.independent

# Prepare DTW clusters, then train and evaluate the relational models
python -m crypto_gnn.clustering.prepare
python -m crypto_gnn.clustering.analysis
python -m crypto_gnn.training.relational
python -m crypto_gnn.evaluation.relational

# Freeze the selected configurations and run the 2022 confirmation
python -m crypto_gnn.training.protocol
python -m crypto_gnn.clustering.prepare_confirmatory
python -m crypto_gnn.training.confirmatory
python -m crypto_gnn.evaluation.confirmatory
```

Training and evaluation outputs are written under `artifacts/`. They include the experiment configuration, progress information, seed-level results, model summaries, planned contrasts, and bootstrap sensitivity results.
