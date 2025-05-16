# ElevatorEnv

## Create a conda environment

```bash
conda create -n env-name python=3.12
```

## Necessary requirements

```bash
pip install -r requirements.txt
```

## Reproducibility

For a tutorial for training your own RL algorithm refer to `examples/elevator/training-tutorial.ipynb`.
To obtain the trained models presented in the paper launch this command from the main directory.

```bash
python gym4real/algorithms/elevator/[sarsa|qlearning].py
```

To reproduce the results, open the notebook in `examples/elevator/benchmarks.ipynb` and run the whole notebook.
