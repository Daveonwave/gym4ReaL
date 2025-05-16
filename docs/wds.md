# WDSEnv

Water Distribution System environment

## Create a conda environment

```bash
conda create -n env-name python=3.12
```

To use WDS env on MacOS with Apple silicon you need to create an env compatible with Intel x64 cpu.
Please run the command

```bash
conda create --platform osx-64 --name env-name python=3.12
```

## Necessary requirements

```bash
pip install -r requirements.txt
pip install -r  gym4real/envs/wds/requirements.txt
```

## Reproducibility

A tutorial from training your own RL algorithm refer to `examples/wds/training-tutorial.ipynb`.
To obtain the trained models presented in the paper launch this command from the main directory.

```bash
python gym4real/algorithms/wds/dqn.py
```

To reproduce the results, open the notebook in `examples/wds/benchmarks.ipynb` and run the whole notebook.
