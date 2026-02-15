#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="chemgfn",
    version="0.0.1",
    description="Gflownet for chemistry",
    author="",
    author_email="",
    url="https://github.com/user/project",
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
