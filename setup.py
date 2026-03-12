#!/usr/bin/env python

import os
import subprocess

from setuptools import find_packages, setup


def get_version() -> str:
    base = "0.1.0"
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return f"{base}+{commit}"
    except Exception:
        return base


setup(
    name="chemgfn",
    version=get_version(),
    description="GFlowNet with LLMs for chemistry and arithmetic generation",
    author="",
    author_email="",
    url="https://github.com/ComDec/ChemGFN",
    install_requires=[
        "torch",
        "torchvision",
        "lightning",
        "torchmetrics",
        "hydra-core",
        "hydra-colorlog",
        "rootutils",
        "transformers",
        "transformers_cfg==0.2.6",
        "peft",
        "rdkit",
        "partialsmiles",
        "sentence-transformers",
        "editdistance",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "chemgfn-merge-logs=chemgfn.cli.merge_csv_groups:main",
            "chemgfn-extract-step=chemgfn.cli.extract_csv_step:main",
            "chemgfn-unmerge-logs=chemgfn.cli.unmerge_logs:main",
        ],
    },
)
