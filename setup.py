#!/usr/bin/env python

from setuptools import setup


setup(
    name="dpbench",
    version="0.0.1",
    url="https://https://github.com/IntelPython/dpbench",
    author="Intel Corp.",
    author_email="diptorup.deb@intel.com",
    description="dpBench",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: Linux",
    ],
    packages=["dpbench", "dpbench.infrastructure"],
    python_requires=">=3.8",
)
