#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="chemgfn",
    version="0.0.1",
    description="Gflownet for chemistry",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "chemgfn-merge-logs=chemgfn.cli.merge_csv_groups:main",
        ],
    },
)
