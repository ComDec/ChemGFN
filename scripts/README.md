# Scripts

Helper scripts for reproducing the experiments. No model weights ship with this repository, so
every evaluation script expects you to train the corresponding experiments first.

| Script | Purpose |
| --- | --- |
| `generate_var_expr24_buffer.py` | Regenerate the Expr24 dataset buffer of exact solutions. |
| `run_amp_all.sh` | Train the four AMP experiments (TB, SubTB, RapTB, RapTB+SubM), one per GPU. |
| `run_eval_all.sh` | Evaluate every trained SMILES experiment (`L_max` 10 and 15). |
| `run_eval_expr24_all.sh` | Evaluate every trained Expr24 experiment. |

## `generate_var_expr24_buffer.py`

Enumerates every parenthesis-free digit/operator expression that evaluates exactly to the target
value (24 by default), tokenizes them, and saves a padded 2-D int64 tensor that
`BufferDataModule(buffer_sample_path=...)` loads as the dataset buffer.

```bash
# Reproduce data/24_points/buffer_24_len1to9_non_zero.pt (57904 x 9)
python scripts/generate_var_expr24_buffer.py --lengths 3 5 7 9
```

The default `--lengths` also enumerates length 11, which produces a much larger buffer than any
of the files checked into `data/24_points/`. Pass `--output` to write somewhere else.

## `run_amp_all.sh`

```bash
GPUS="0 1 2 3" bash scripts/run_amp_all.sh
```

Launches the four AMP experiments in the background with the trainer overrides the reported AMP
numbers were produced with. Logs land in `${LOG_DIR:-logs}`.

## `run_eval_all.sh` and `run_eval_expr24_all.sh`

Both scripts require `CKPT_ROOT` to point at a directory holding one subdirectory per trained
run, each containing the checkpoint named by `CKPT_NAME` (default `last.ckpt`):

```bash
CKPT_ROOT=/path/to/checkpoints GPUS="0 1 2 3" bash scripts/run_eval_all.sh
```

In both scripts the subdirectory is the experiment's `exp_name`, which is the config path with
`/` replaced by `_` (e.g. `smiles/raptb_subm` becomes `smiles_raptb_subm`, and
`expr24/rp_tb` becomes `expr24_rp_tb`). Runs with no checkpoint at the expected path are
skipped with a message rather than failing the batch.

The evaluation protocol is fixed in the scripts — 100 test batches for SMILES, 200 for Expr24,
and 3 independent sampling repeats in both cases. Change it only if you are not comparing
against the reported numbers.
