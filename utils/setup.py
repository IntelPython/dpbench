from setuptools import setup, find_packages


PACKAGENAME = "dpbench_utils"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    description="Utils package for dpbench applications.",
    long_description="Utils package for dpbench applications containing common data generation scripts and python scripts.",
    # install_requires=["numpy", "scikit-learn"],
    packages=find_packages(),
)
