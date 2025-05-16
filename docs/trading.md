# TradingEnv

## Create a conda environment

```bash
conda create -n env-name python=3.12
```

## Necessary requirements

```bash
pip install -r requirements.txt
pip install -r gym4real/envs/trading/requirements.txt
```

## Reproducibility

In order to reproduce the results, open the notebook in `examples/trading/benchmarks.ipynb` and run the whole notebook.
It is possible to retrain the algorithm from scratch, enabling the related flags.
The trained model are saved in the directory `examples/trading/trained_models`.

