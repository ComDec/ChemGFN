"""Data modules for the released tasks.

:class:`CommonGenDataModule` is resolved lazily because it pulls in ``datasets``, which ships
only with the optional ``chemgfn[commongen]`` extra; importing this package must stay free of
that dependency for the SMILES, Expr24 and AMP tasks.
"""

from typing import Any

from .gfn_datamodule import BufferDataModule

__all__ = [
    "BufferDataModule",
    "CommonGenDataModule",
]


def __getattr__(name: str) -> Any:
    if name == "CommonGenDataModule":
        from .common_gen_datamodule import CommonGenDataModule

        return CommonGenDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
