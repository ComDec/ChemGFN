"""Entry-point utilities shared by ``chemgfn/train.py`` and ``chemgfn/eval.py``."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any, Callable

from omegaconf import DictConfig

from chemgfn.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply the optional conveniences declared under ``cfg.extras``.

    Honours ``ignore_warnings`` (silence Python warnings), ``enforce_tags``
    (prompt for run tags) and ``print_config`` (print the config tree with Rich).

    Args:
        cfg: The composed config tree.
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Wrap a task so failures are logged and the wandb run is always closed.

    The wrapped function still re-raises, so a failing run is not silently
    swallowed; the ``finally`` block reports the output directory and calls
    ``wandb.finish()`` when wandb is installed and active.

    Args:
        task_func: Task taking ``cfg`` and returning ``(metric_dict, object_dict)``.

    Returns:
        The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        except Exception as ex:
            log.exception("")
            raise ex

        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Read a metric logged by the ``LightningModule``.

    Args:
        metric_dict: Metrics returned by the task.
        metric_name: Name of the metric to read; ``None`` skips the lookup.

    Returns:
        The metric value, or ``None`` when ``metric_name`` is ``None``.

    Raises:
        Exception: If ``metric_name`` was not logged.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure the metric name logged in the LightningModule is correct."
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
