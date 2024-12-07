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
    # use this to customize global commands available in the terminal after installing the package
)
