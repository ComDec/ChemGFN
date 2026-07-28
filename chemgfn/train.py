import os
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Adds the project root to PYTHONPATH and exports PROJECT_ROOT, which "configs/paths/default.yaml"
# resolves paths against. See https://github.com/ashleve/rootutils.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from chemgfn.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Instantiate every component from the config and fit the model.

    Wrapped in ``@task_wrapper`` so that failures are logged and the run directory is closed out
    cleanly, which matters for multiruns.

    Args:
        cfg: Hydra configuration composed from ``configs/train.yaml``.

    Returns:
        The trainer's callback metrics and a dict of every instantiated object.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        datamodule.setup("fit")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    return dict(trainer.callback_metrics), object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for training.

    Args:
        cfg: Hydra configuration composed from ``configs/train.yaml``.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
