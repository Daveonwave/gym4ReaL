# MicrogridEnv

## Create a conda environment

```bash
conda create -n env-name python=3.12
```

## Necessary requirements

```bash
pip install -r requirements.txt
pip install -r  gym4real/envs/microgrid/requirements.txt
```

## Reproducibility

For a tutorial for training your own RL algorithm refer to `examples/microgrid/training-tutorial.ipynb`.
To obtain the trained models presented in the paper launch this command from the main directory.

```bash
python gym4real/algorithms/microgrid/ppo.py
```

To reproduce the results, open the notebook in `examples/microgrid/benchmarks.ipynb` and run the whole notebook.
