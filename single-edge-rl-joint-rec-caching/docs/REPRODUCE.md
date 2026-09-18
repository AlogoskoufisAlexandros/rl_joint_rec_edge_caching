# Reproducing the results

All commands run from `single-edge-rl-joint-rec-caching/` (`cd` into it first).

## 1. Install

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install jupyter        # optional
```

Python 3.11. `requirements.txt` pins the versions the results were produced with
(`torch==2.5.1+cu121`; the extra index is where that wheel lives). Without a CUDA GPU the
code runs on CPU, several times slower.

## 2. Data

Put MovieLens 25M's `ratings.csv` in `single-edge-rl-joint-rec-caching/` (see DATA.md). On the first run
the notebook derives the simulator inputs, `u_file_movielens.npy` (5 000 x 5 000 item
similarities, about 100 MB) and `popularity_file_movielens.npy`, and saves them next to
it. This reads the 680 MB file once and takes a few minutes.

The `u` matrix used for the results has mean 0.08565857. The driver prints a note if yours
differs (a different MovieLens release would do that); in that case expect numbers close
to, but not identical with, the tables.

## 3. Smoke test

```bash
python run_seed_sweep.py --smoke
```

Runs WC, NC and NC + PBRS for seed 0 with 250 training and 60 evaluation episodes, into
`results/smoke/`. It checks the install, the data and the pipeline. The numbers it prints
are meaningless; the agents need the full episode budget.

## 4. Headline table: WC and NC, 10 seeds

```bash
python run_seed_sweep.py --launch --jobs WC:0-9,NC:0-9 --parallel 2
python run_seed_sweep.py --collect
```

One run is 20 000 training and 10 000 evaluation episodes; a run takes hours, and the
full table is a multi-day job on a single GPU (how long depends on the machine). Each
process needs about 2 GB of RAM; the model is small and the GPU is mostly idle, so more
processes share it well if you have the RAM.

Output goes to `results/new_sweep/`: one CSV and one training-curve file per run.
Launching again skips finished runs. `--collect` writes `seed_sweep_results.csv` and
`seed_sweep_aggregate.json` and prints the per-seed table next to the shipped values.

Expected: WC about 0.80, NC about 0.50, WC minus NC about +0.31 in every seed. Runs are
seeded end to end (`set_seed`, deterministic cuDNN). WC seed 7 re-run through this driver
reproduced the shipped value, 0.7968983.

## 5. Control: NC + PBRS, 10 seeds

```bash
python run_seed_sweep.py --launch --jobs NCPBRS:0-9 --parallel 2
```

Expected about 0.48; per-seed values in RESULTS.md.

## 6. Using the notebook directly

Open `single_edge_rl.ipynb` and run it top to bottom. The config cell already selects
MovieLens and the settings above; the cells after it train and evaluate one agent at a time
for a single seed (set `max_iter` there; the driver uses 20 000). The notebook is for
reading and experimenting; the driver runs the same code across seeds in parallel.

## What is held fixed

- Metric: total hits / total steps over the evaluation episodes, hit counted against the
  cache before the action.
- Within a seed every agent is trained and evaluated on the same environment and the same
  evaluation episodes: `_run_rl_agent` seeds and rebuilds the environment before training
  and again before evaluation.
- Evaluation uses the best training checkpoint: the highest 1 000-episode rolling mean of
  the per-episode hit rate in the second half of training.
- Every run's CSV row and `seed_sweep_meta.json` record the notebook's md5, the software
  versions and the protocol.
