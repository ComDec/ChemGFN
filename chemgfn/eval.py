from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

# Adds the project root to PYTHONPATH and exports PROJECT_ROOT, which "configs/paths/default.yaml"
# resolves paths against. See https://github.com/ashleve/rootutils.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from chemgfn.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sample from a trained checkpoint on the test split and report the task metrics.

    ``test_repeats`` independent passes are run, each with its own trainer so that per-repeat
    sample dumps land in separate directories; the model tags itself with a repeat suffix so
    ``on_test_epoch_end`` can name its output files accordingly.

    Args:
        cfg: Hydra configuration composed from ``configs/eval.yaml``.

    Returns:
        The callback metrics of the final repeat and a dict of every instantiated object.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    repeats = int(cfg.get("test_repeats", 1))
    log.info(f"Starting testing for {repeats} repeat(s)!")

    rep_trainer = trainer
    for rep in range(repeats):
        rep_trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            default_root_dir=f"{cfg.trainer.default_root_dir}/repeat_{rep}"
            if repeats > 1
            else cfg.trainer.default_root_dir,
        )
        object_dict["trainer"] = rep_trainer

        model.test_repeat_suffix = f"_run{rep}" if repeats > 1 else ""

        log.info(f"Testing repeat {rep + 1}/{repeats}")
        rep_trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    return dict(rep_trainer.callback_metrics), object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for evaluation.

    Args:
        cfg: Hydra configuration composed from ``configs/eval.yaml``.
    """
    with open_dict(cfg):
        cfg.exp_name = f"{cfg.exp_name}_eval"
        wandb_project = cfg.get("wandb_project")
        if wandb_project is not None and cfg.get("logger") is not None:
            if "wandb" in cfg.logger:
                cfg.logger.wandb.project = wandb_project

    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
